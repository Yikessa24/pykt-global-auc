"""AKT with the online PG-DRO update of Kim and Lu (2025)."""

import math

import torch
from torch.nn.functional import binary_cross_entropy

from .akt import AKT


class AKT_PGDRO(AKT):
    def __init__(self, *args, pgdro_group_lr=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(pgdro_group_lr) or pgdro_group_lr <= 0:
            raise ValueError("PG-DRO group step size must be finite and positive.")
        self.model_name = "akt_pgdro"
        self.pgdro_group_lr = pgdro_group_lr
        self.register_buffer("pgdro_kc_weights", torch.zeros(self.n_question))
        self.register_buffer("pgdro_log_group_weights", torch.full((2,), -math.log(2)))

    def pgdro_loss(self, predictions, labels, concepts, mask, group):
        """Use one weighted KC loss for the ascent and the subsequent descent."""
        labels = labels.to(predictions.dtype)
        losses = binary_cross_entropy(predictions, labels, reduction="none").masked_fill(~mask, 0)
        indices = concepts.reshape(-1)
        sums = self.pgdro_kc_weights.new_zeros(self.n_question).scatter_add_(0, indices, losses.reshape(-1))
        counts = torch.zeros_like(sums).scatter_add_(0, indices, mask.reshape(-1).to(sums.dtype))
        risk = torch.dot(self.pgdro_kc_weights, sums / counts.clamp_min(1))

        with torch.no_grad():
            self.pgdro_log_group_weights[group].add_(self.pgdro_group_lr * risk.detach())
            self.pgdro_log_group_weights.sub_(self.pgdro_log_group_weights.logsumexp(0))
            group_weight = self.pgdro_log_group_weights[group].exp()
        return group_weight * risk
