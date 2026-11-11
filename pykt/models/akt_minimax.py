"""AKT trained with MinimaxFairAUC (Yang et al., AAAI 2023, Algorithm 1)."""

import math

import torch

from .akt import AKT


N_GROUPS = 3
TRAIN_SAMPLE_LIMIT = 64


def group_pair_risks(scores, labels, groups, *, training, exact=False):
    """Clipped logistic loss per directed pair; NaN marks absent pairs."""
    limit = TRAIN_SAMPLE_LIMIT if training else 256

    def stratum(group, label):
        values = scores[(groups == group) & (labels == label)]
        if exact or values.numel() <= limit:
            return values
        if training:
            return values[torch.randperm(values.numel(), device=values.device)[:limit]]
        # Fixed score quantiles avoid enumerating all validation pairs.
        ranks = (torch.arange(limit, device=values.device) + 0.5) * (values.numel() / limit)
        return values.sort().values[ranks.long()]

    positive = [stratum(group, 1) for group in range(N_GROUPS)]
    negative = [stratum(group, 0) for group in range(N_GROUPS)]
    risks = []
    for pos in positive:
        for neg in negative:
            if not pos.numel() or not neg.numel():
                risks.append(scores.new_full((), float("nan")))
                continue
            block_size = max(1, (1 << 20) // neg.numel())
            total = scores.new_zeros(())
            for block in pos.split(block_size):
                margins = block[:, None] - neg[None, :]
                probabilities = margins.sigmoid().clamp(1e-6, 1 - 1e-6)
                total = total - probabilities.log().sum()
            risks.append(total / (pos.numel() * neg.numel()))
    return torch.stack(risks).reshape(N_GROUPS, N_GROUPS)


def sampled_group_pair_risks(scores, labels, groups, mask):
    """Sample all six strata without dynamic-size tensor indexing."""
    groups = groups[:, None].expand_as(mask)
    stratum = (2 * groups + labels.long()).flatten()
    strata = torch.arange(2 * N_GROUPS, device=scores.device)[:, None]
    eligible = (strata == stratum) & mask.flatten()
    # Unique priorities give each eligible stratum a uniform sample without
    # replacement. Negative priorities mark unused slots in small strata.
    priorities = torch.randperm(scores.numel(), device=scores.device)
    priorities = torch.where(eligible, priorities, -1)
    selected, indices = priorities.topk(min(TRAIN_SAMPLE_LIMIT, scores.numel()), dim=1, sorted=False)
    sampled = scores.flatten()[indices].reshape(N_GROUPS, 2, -1)
    present = (selected >= 0).reshape_as(sampled)
    sampled = sampled.masked_fill(~present, 0.0)

    positive, negative = sampled[:, 1], sampled[:, 0]
    margins = positive[:, None, :, None] - negative[None, :, None, :]
    valid_pairs = present[:, 1][:, None, :, None] & present[:, 0][None, :, None, :]
    counts = valid_pairs.sum(dim=(-2, -1))
    losses = -margins.sigmoid().clamp(1e-6, 1 - 1e-6).log()
    mean_loss = losses.masked_fill(~valid_pairs, 0.0).sum(dim=(-2, -1)) / counts.clamp_min(1)
    return mean_loss.masked_fill(counts == 0, float("nan"))


class AKT_MINIMAX(AKT):
    """Keep AKT inference and fit its scorer by minimax bipartite ranking."""

    def __init__(self, *args, minimax_temperature=7.0, minimax_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(minimax_temperature) or minimax_temperature <= 0:
            raise ValueError("minimax_temperature must be finite and positive.")
        if not math.isfinite(minimax_weight) or minimax_weight < 0:
            raise ValueError("minimax_weight must be finite and nonnegative.")
        self.minimax_weight = minimax_weight
        self.model_name = "akt_minimax"
        self.minimax_temperature = minimax_temperature
        self.register_buffer("log_pair_weights", torch.full((N_GROUPS, N_GROUPS), -math.log(N_GROUPS**2)))

    def forward_logits(self, concepts, responses, questions):
        """Use AKT's layers and expose its pre-sigmoid scores for ranking."""
        q, qa = self.base_emb(concepts, responses)
        difficulty, reg = None, 0.0
        if self.n_pid > 0:
            difficulty = self.difficult_param(questions)
            q_diff = self.q_embed_diff(concepts)
            qa_diff = self.qa_embed_diff(responses)
            q = q + difficulty * q_diff
            qa = qa + difficulty * (qa_diff if self.separate_qa else qa_diff + q_diff)
            reg = difficulty.square().sum() * self.l2
        hidden = self.model(q, qa, difficulty, reuse_attention_geometry=self.model_name == "akt_minimax")
        return self.out(torch.cat((hidden, q), dim=-1)).squeeze(-1), reg

    def sequence_scores(self, batch):
        device = self.log_pair_weights.device

        def sequence(key):
            return torch.cat((batch[key][:, :1], batch["shft_" + key]), dim=1).to(device=device, dtype=torch.long)

        scores, reg = self.forward_logits(sequence("cseqs"), sequence("rseqs"), sequence("qseqs"))
        return scores[:, 1:], reg

    @staticmethod
    def grouped_scores(scores, batch):
        device = scores.device
        mask = batch["minimax_mask"].to(device)
        labels = batch["shft_rseqs"].to(device)[mask]
        groups = batch["minimax_group"].to(device)[:, None].expand_as(mask)[mask]
        return scores[mask], labels, groups

    @torch.no_grad()
    def initialize_pair_weights(self, group_label_counts):
        counts = torch.as_tensor(group_label_counts, device=self.log_pair_weights.device, dtype=self.log_pair_weights.dtype)
        if counts.shape != (N_GROUPS, 2) or not torch.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("Expected nonempty negative and positive strata in all three groups.")
        prior = counts[:, 1].log()[:, None] + counts[:, 0].log()[None, :]
        self.log_pair_weights.copy_(prior - prior.flatten().logsumexp(0))

    def minimax_loss(self, risks):
        observed = risks.isfinite()
        weights = self.log_pair_weights.exp() * observed
        risks = torch.where(observed, risks, 0.0)
        return (risks * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)

    @torch.no_grad()
    def update_pair_weights(self, risks, learning_rate):
        # Mirror ascent with the reference implementation's temperature * SGD LR.
        # Only compare pairs observed in this batch. Missing pairs receive
        # the observed mean risk so their weights do not drift by default.
        observed = risks.isfinite()
        mean_risk = torch.where(observed, risks, 0.0).sum() / observed.sum().clamp_min(1)
        centered = torch.where(observed, risks - mean_risk, 0.0)
        updated = self.log_pair_weights + self.minimax_temperature * learning_rate * centered
        self.log_pair_weights.copy_(updated - updated.flatten().logsumexp(0))


class AKT_MINIMAXFAIRAUC(AKT_MINIMAX):
    """Paper Algorithm 1 with AKT sequence logits as the scorer."""
    def __init__(self, *args, minimax_temperature=10.0, **kwargs):
        super().__init__(*args, minimax_temperature=minimax_temperature, **kwargs)
        self.model_name = "akt_minimaxfairauc"
