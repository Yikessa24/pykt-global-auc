"""AKT with the xAUC constraint of Vogel, Bellet and Clémençon (2021)."""

import math

import torch
from sklearn.metrics import roc_auc_score

from .akt_minimax import AKT_MINIMAX, N_GROUPS


def _ranking_error(positive, negative, num_pairs):
    # Incomplete U-statistic from the reference implementation. The factor of
    # two is the difference between its +1 and -1 label encodings.
    p = torch.randint(positive.numel(), (num_pairs,), device=positive.device)
    n = torch.randint(negative.numel(), (num_pairs,), device=negative.device)
    return torch.sigmoid(2 * (negative[n] - positive[p])).mean()


class AKT_INTERFAIRAUC(AKT_MINIMAX):
    def __init__(self, *args, interfair_weight=1.0, interfair_num_pairs=1000, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(interfair_weight) or interfair_weight < 0 or interfair_num_pairs < 1:
            raise ValueError("Use a nonnegative InterFairAUC weight and a positive pair sample size.")
        self.interfair_weight = interfair_weight
        self.interfair_num_pairs = interfair_num_pairs
        self.model_name = "akt_interfairauc"

    def ranking_loss(self, scores, labels, grouped_scores, grouped_labels, groups):
        """Global ranking error plus the sum of absolute directional gaps."""
        positives, negatives = scores[labels == 1], scores[labels == 0]
        loss = scores.sum() * 0
        if positives.numel() and negatives.numel():
            loss = _ranking_error(positives, negatives, self.interfair_num_pairs)
        positive = [grouped_scores[(groups == i) & (grouped_labels == 1)] for i in range(N_GROUPS)]
        negative = [grouped_scores[(groups == i) & (grouped_labels == 0)] for i in range(N_GROUPS)]
        for i in range(N_GROUPS):
            for j in range(i + 1, N_GROUPS):
                if min(positive[i].numel(), positive[j].numel(), negative[i].numel(), negative[j].numel()):
                    forward = _ranking_error(positive[i], negative[j], self.interfair_num_pairs)
                    reverse = _ranking_error(positive[j], negative[i], self.interfair_num_pairs)
                    loss = loss + self.interfair_weight * (forward - reverse).abs()
        return loss

    def validation_loss(self, scores, labels, groups, auc):
        scores, labels, groups = (tensor.cpu().numpy() for tensor in (scores, labels, groups))

        def cross_auc(i, j):
            mask = ((groups == i) & (labels == 1)) | ((groups == j) & (labels == 0))
            return roc_auc_score(labels[mask], scores[mask])

        gap = sum(abs(cross_auc(i, j) - cross_auc(j, i))
                  for i in range(N_GROUPS) for j in range(i + 1, N_GROUPS))
        return 1 - auc + self.interfair_weight * gap
