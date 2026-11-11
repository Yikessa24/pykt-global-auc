"""AKT adaptation of the noisy-group AUC constraints of Wu et al. (2025)."""

import math

import torch

from .akt_minimax import AKT_MINIMAX, N_GROUPS


def worst_group_error(errors, support, radius):
    """Minimize error over the probability simplex in the paper's TV ball.

    The adversary maximizes overall error minus the group error. Transfer up
    to radius mass from the highest errors in the nominal support to a
    global minimum. This solves the linear inner problem exactly.
    """
    ranked = errors[support].sort(descending=True).values
    nominal = 1 / ranked.numel()
    moved = (radius - torch.arange(ranked.numel(), device=errors.device,
                                   dtype=errors.dtype) * nominal).clamp(0, nominal)
    return ranked.mean() - (moved * (ranked - errors.min())).sum()


def _project_dual(values, limit):
    """Project to nonnegative multipliers with sum at most limit."""
    positive = values.clamp_min(0)
    ordered = positive.sort(descending=True).values
    ranks = torch.arange(1, ordered.numel() + 1, device=values.device, dtype=values.dtype)
    threshold = ((ordered.cumsum(0) - limit) / ranks).max().clamp_min(0)
    return (positive - threshold).clamp_min(0)


class AKT_NOISYAUC(AKT_MINIMAX):
    def __init__(self, *args, noisy_gamma, noisy_pair_cap=32, noisy_sam_radius=0.0005,
                 noisy_dual_lr=0.01, noisy_dual_limit=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        radii = torch.as_tensor(noisy_gamma, dtype=torch.float32)
        if radii.numel() == 1:
            radii = radii.repeat(N_GROUPS ** 2)
        if radii.numel() != N_GROUPS ** 2 or not torch.isfinite(radii).all() or not ((0 <= radii) & (radii <= 1)).all():
            raise ValueError("Supply one TV radius or nine directed radii in [0, 1].")
        if (noisy_pair_cap < 1 or not math.isfinite(noisy_sam_radius) or noisy_sam_radius < 0
                or not math.isfinite(noisy_dual_lr) or noisy_dual_lr <= 0
                or not math.isfinite(noisy_dual_limit) or noisy_dual_limit <= 0):
            raise ValueError("Invalid noisy AUC sampler or optimizer parameters.")
        self.model_name = "akt_noisyauc"
        self.noisy_pair_cap = noisy_pair_cap
        self.noisy_sam_radius = noisy_sam_radius
        self.noisy_dual_lr = noisy_dual_lr
        self.noisy_dual_limit = noisy_dual_limit
        self.register_buffer("noisy_radii", radii.reshape(N_GROUPS, N_GROUPS))
        self.register_buffer("noisy_dual", torch.zeros(N_GROUPS, N_GROUPS))

    def sample_positions(self, labels, groups, *, training):
        samples = []
        for label in (1, 0):
            strata = []
            for group in range(N_GROUPS):
                positions = torch.where((labels == label) & (groups == group))[0]
                if positions.numel() > self.noisy_pair_cap:
                    if training:
                        order = torch.randperm(positions.numel(), device=positions.device)
                    else:
                        order = torch.arange(self.noisy_pair_cap, device=positions.device)
                        order = ((order + 0.5) * positions.numel() / self.noisy_pair_cap).long()
                    positions = positions[order[:self.noisy_pair_cap]]
                strata.append(positions)
            samples.append(torch.cat(strata))
        return samples

    def objective(self, scores, groups, positions):
        pos, neg = positions
        if not pos.numel() or not neg.numel():
            zero = scores.sum() * 0
            return zero, torch.zeros_like(self.noisy_dual), zero
        margins = scores[pos, None] - scores[neg][None, :]
        errors = -margins.sigmoid().clamp(1e-6, 1 - 1e-6).log()
        overall = errors.mean()
        constraints = []
        for i in range(N_GROUPS):
            for j in range(N_GROUPS):
                mask = (groups[pos] == i)[:, None] & (groups[neg] == j)[None, :]
                if mask.any():
                    local = worst_group_error(errors, mask, self.noisy_radii[i, j])
                    constraints.append(overall - local)
                else:
                    constraints.append(overall * 0)
        violation = torch.stack(constraints).reshape_as(self.noisy_dual)
        return overall + (self.noisy_dual * violation).sum(), violation, overall

    def _batch_objective(self, batch, positions=None):
        scores, reg = self.sequence_scores(batch)
        scores, labels, groups = self.grouped_scores(scores, batch)
        if positions is None:
            positions = self.sample_positions(labels, groups, training=True)
        objective, violation, _ = self.objective(scores, groups, positions)
        return objective + reg, violation, positions

    def train_noisy_batch(self, batch, optimizer):
        """SAM descent for the scorer and projected ascent for the dual."""
        optimizer.zero_grad()
        loss, violation, positions = self._batch_objective(batch)
        loss.backward()
        if self.noisy_sam_radius:
            active = [p for p in self.parameters() if p.grad is not None]
            if active:
                with torch.no_grad():
                    norm = torch.stack([p.grad.norm() for p in active]).norm()
                    scale = self.noisy_sam_radius / norm.clamp_min(1e-12)
                    originals = [p.detach().clone() for p in active]
                    for p in active:
                        p.add_(p.grad * scale)
                optimizer.zero_grad()
                try:
                    loss, violation, _ = self._batch_objective(batch, positions)
                    loss.backward()
                finally:
                    with torch.no_grad():
                        for p, original in zip(active, originals):
                            p.copy_(original)
        optimizer.step()
        with torch.no_grad():
            updated = (self.noisy_dual + self.noisy_dual_lr * violation).flatten()
            self.noisy_dual.copy_(_project_dual(updated, self.noisy_dual_limit).reshape_as(self.noisy_dual))
        return loss.detach()

    def validation_loss(self, scores, labels, groups):
        positions = self.sample_positions(labels, groups, training=False)
        _, constraints, overall = self.objective(scores, groups, positions)
        # Feasible iterates precede infeasible ones. If a run never satisfies
        # the surrogate constraints, retain the least-violating iterate.
        violation = constraints.max().clamp_min(0).item()
        return violation, overall.item()
