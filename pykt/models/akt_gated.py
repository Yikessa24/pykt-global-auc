"""AKT with a gated residual prediction head and sampled question ranking."""

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from .akt import AKT
from .akt_minimax import N_GROUPS, sampled_group_pair_risks


class AKT_GATED(AKT):
    def __init__(self, *args, minimax_weight=0.1, max_auc_drop=0.003, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(minimax_weight) or minimax_weight < 0:
            raise ValueError("minimax_weight must be finite and nonnegative.")
        if not math.isfinite(max_auc_drop) or not 0 <= max_auc_drop <= 1:
            raise ValueError("max_auc_drop must be between zero and one.")
        self.model_name = "akt_gated"
        self.minimax_weight = minimax_weight
        self.max_auc_drop = max_auc_drop
        feature_size = self.out[0].in_features
        self.residual_gate = nn.Linear(feature_size, 2)
        self.residual_experts = nn.Sequential(
            nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, 2),
        )
        nn.init.zeros_(self.residual_experts[-1].weight)
        nn.init.zeros_(self.residual_experts[-1].bias)

    def score_logits(self, features):
        weights = self.residual_gate(features).softmax(dim=-1)
        corrections = self.residual_experts(features)
        return super().score_logits(features) + (weights * corrections).sum(dim=-1)

    def load_akt_checkpoint(self, state):
        initial_state = self.state_dict()
        if state.keys() != {key for key in initial_state if not key.startswith("residual_")}:
            raise ValueError("Expected a plain AKT checkpoint with matching architecture.")
        self.load_state_dict({**initial_state, **state})

    def sequence_scores(self, batch):
        device = self.out[0].weight.device

        def sequence(key):
            return torch.cat((batch[key][:, :1], batch["shft_" + key]), dim=1).to(device, dtype=torch.long)

        logits, reg, _ = self._forward_logits(
            sequence("cseqs"), sequence("rseqs"), sequence("qseqs"), reuse_attention_geometry=True,
        )
        return logits[:, 1:], reg

    def training_loss(self, batch):
        logits, reg = self.sequence_scores(batch)
        valid = batch["smasks"].to(logits.device)
        labels = batch["shft_rseqs"].to(logits)
        bce = binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = bce.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1) + reg
        if self.minimax_weight == 0:
            return loss

        risks = sampled_group_pair_risks(
            logits, labels, batch["minimax_group"].to(logits.device), batch["minimax_mask"].to(logits.device),
        )
        # Smooth approximation to the worst observed cell. Give an empty
        # batch a finite zero loss and zero gradient from the ranking term.
        temperature = 7.0
        observed = risks.isfinite()
        scaled = (temperature * torch.where(observed, risks, -torch.inf)).flatten()
        count = observed.sum()
        fallback = scaled.new_zeros(1).masked_fill((count > 0).reshape(1), -torch.inf)
        ranking_loss = (torch.cat((scaled, fallback)).logsumexp(0) - count.clamp_min(1).log()) / temperature
        return loss + self.minimax_weight * ranking_loss

    @torch.no_grad()
    def validation_scores(self, loader):
        """Exact AUCs on the first predicted concept of each validation question."""
        self.eval()
        rows = []
        for batch in loader:
            logits, _ = self.sequence_scores(batch)
            mask = batch["question_mask"].to(logits.device)
            groups = batch["minimax_group"].to(logits.device)[:, None].expand_as(mask)
            groups = groups.masked_fill(~batch["minimax_mask"].to(logits.device), -1)
            rows.append(torch.stack((logits[mask], batch["shft_rseqs"].to(logits)[mask], groups[mask])))
        logits, labels, groups = torch.cat(rows, dim=1).cpu().numpy()
        auc = roc_auc_score(labels, logits)
        acc = float(((logits >= 0) == (labels == 1)).mean())
        cells = []
        for positive in range(N_GROUPS):
            for negative in range(N_GROUPS):
                mask = ((groups == positive) & (labels == 1)) | ((groups == negative) & (labels == 0))
                cells.append(roc_auc_score(labels[mask], logits[mask]))
        matrix = np.asarray(cells).reshape(N_GROUPS, N_GROUPS)
        return auc, acc, float(matrix.min()), float(np.abs(matrix - matrix.T).max())
