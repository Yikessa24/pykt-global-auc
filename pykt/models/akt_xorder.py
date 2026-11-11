"""xOrder dynamic programming for score exports from a fitted AKT.

Based on Cui et al. (TPAMI 2023), Algorithm 1 and Section IV-B, and the
MIT-licensed author reference at cuis15/xorder (utils.py, revision 6dde5a1).
"""

import numpy as np


def insertion_positions(reference_labels, target_labels, penalty):
    """Choose each target's position in the fixed reference ordering."""
    a = np.asarray(reference_labels, dtype=np.int64)
    b = np.asarray(target_labels, dtype=np.int64)
    m, n = len(a), len(b)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("xOrder penalty must be finite and nonnegative.")
    if set(a) != {0, 1} or set(b) != {0, 1}:
        raise ValueError("xOrder requires both labels in each calibration group.")

    remaining_a = np.r_[np.cumsum((1 - a)[::-1])[::-1], 0]
    remaining_b = np.r_[np.cumsum((1 - b)[::-1])[::-1], 0]
    scale_a = penalty / (a.sum() * remaining_b[0])
    scale_b = penalty / (b.sum() * remaining_a[0])
    # Each antidiagonal depends only on the preceding one. Keep only its
    # utility and signed gap; the boolean lattice records the backtrace.
    chose_a = np.empty((m + 1, n + 1), dtype=bool)
    utility = np.zeros(m + 1)
    disparity = np.zeros(m + 1)
    a = np.r_[0, a]
    b = np.r_[0, b]
    for diagonal in range(1, m + n + 1):
        i = np.arange(max(0, diagonal - n), min(m, diagonal) + 1)
        j = diagonal - i
        from_a = np.maximum(i - 1, 0)
        gain_a, gain_b = a[i] * remaining_b[j], b[j] * remaining_a[i]
        utility_a, utility_b = utility[from_a] + gain_a, utility[i] + gain_b
        gap_a = disparity[from_a] + scale_a * gain_a
        gap_b = disparity[i] - scale_b * gain_b
        choose = (j == 0) | ((i > 0) & (utility_a - abs(gap_a) > utility_b - abs(gap_b)))
        chose_a[i, j] = choose
        utility[i] = np.where(choose, utility_a, utility_b)
        disparity[i] = np.where(choose, gap_a, gap_b)

    positions = np.empty(n, dtype=int)
    i, j = m, n
    while j:
        if chose_a[i, j]:
            i -= 1
        else:
            positions[j - 1] = i
            j -= 1
    return positions


def _adjust_calibration(reference_scores, positions):
    """Space inserted scores evenly between adjacent reference scores."""
    edges = np.r_[1., reference_scores, 0.]
    _, starts, counts = np.unique(positions, return_index=True, return_counts=True)
    offsets = np.arange(len(positions)) - np.repeat(starts, counts) + 1
    share = offsets / (np.repeat(counts, counts) + 1)
    return edges[positions] + share * (edges[positions + 1] - edges[positions])


def _adjust_evaluation(original, adjusted, scores):
    """Rank-uniform interpolation as used by the public implementation."""
    order = np.argsort(-scores, kind="stable")
    bins = np.searchsorted(-original, -scores[order], side="right")
    edges = np.r_[1., adjusted, 0.]
    _, starts, counts = np.unique(bins, return_index=True, return_counts=True)
    share = (np.arange(len(bins)) - np.repeat(starts, counts)) / np.repeat(counts, counts)
    result = scores.copy()
    result[order] = edges[bins] + share * (edges[bins + 1] - edges[bins])
    return result


def apply_xorder(calibration, evaluation, *, group_order, penalty, calibration_limit=None):
    """Iteratively merge a new group into the current ordering (Section IV-B)."""
    if len(group_order) < 2 or len(set(group_order)) != len(group_order):
        raise ValueError("xOrder needs at least two distinct groups in merge order.")
    if calibration_limit is not None and calibration_limit < 2:
        raise ValueError("xOrder calibration limit must be at least two.")
    fit = calibration.copy()
    result = evaluation.copy()
    reference_groups = [group_order[0]]
    for group in group_order[1:]:
        a = fit.loc[fit.cluster.isin(reference_groups)].sort_values("pred", ascending=False, kind="stable")
        b = fit.loc[fit.cluster == group].sort_values("pred", ascending=False, kind="stable")
        if calibration_limit is not None:
            if len(a) > calibration_limit:
                a = a.iloc[np.linspace(0, len(a) - 1, calibration_limit, dtype=int)]
            if len(b) > calibration_limit:
                b = b.iloc[np.linspace(0, len(b) - 1, calibration_limit, dtype=int)]
        positions = insertion_positions(a.label.to_numpy(), b.label.to_numpy(), penalty)
        original = b.pred.to_numpy()
        adjusted = _adjust_calibration(a.pred.to_numpy(), positions)
        selected = result.cluster == group
        result.loc[selected, "pred"] = _adjust_evaluation(original, adjusted, result.loc[selected, "pred"].to_numpy())
        fit.loc[fit.cluster == group, "pred"] = _adjust_evaluation(
            original, adjusted, fit.loc[fit.cluster == group, "pred"].to_numpy(),
        )
        fit.loc[b.index, "pred"] = adjusted
        reference_groups.append(group)
    return result
