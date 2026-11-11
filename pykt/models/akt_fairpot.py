"""FairPOT global-AUC score transport for predictions from a fitted AKT."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _full_transport(reference_scores, target_scores):
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    target = np.sort(np.asarray(target_scores, dtype=float))
    if not reference.size or not target.size or not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("FairPOT needs finite calibration scores for both groups.")
    # For squared distance in one dimension the optimal coupling is monotone.
    # The integral of the reference quantile function over each target atom
    # equals its transported mass times its barycentric destination.
    n_ref, n_target = len(reference), len(target)
    scaled = np.arange(n_target + 1, dtype=np.int64) * n_ref
    index, remainder = np.divmod(scaled, n_target)
    prefix = np.r_[0., np.cumsum(reference)]
    integral = prefix[index] + remainder / n_target * reference[np.minimum(index, n_ref - 1)]
    destination = np.clip(np.diff(integral) * n_target / n_ref, reference[0], reference[-1])
    return target, destination


def _partial_knots(target, destination, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("FairPOT fraction must be between 0 and 1.")
    adjusted = target.copy()
    size = int(np.ceil(fraction * len(target)))
    if size:
        # Use the upper score tail as defined in the text of Algorithm 1.
        top = target >= target[-size]
        adjusted[top] = destination[top]
    # A score-only interpolation needs a single value per tied input score.
    knots, first, count = np.unique(target, return_index=True, return_counts=True)
    return knots, np.add.reduceat(adjusted, first) / count


def fairpot_knots(reference_scores, target_scores, fraction):
    """Return the original and transported score knots (Liu et al., Eq. 1–3)."""
    return _partial_knots(*_full_transport(reference_scores, target_scores), fraction)


def _calibration_transport(calibration, reference_group, target_group):
    if reference_group == target_group:
        raise ValueError("FairPOT reference and target must be different groups.")
    reference = calibration.loc[calibration.cluster == reference_group, "pred"]
    target = calibration.loc[calibration.cluster == target_group, "pred"]
    return _full_transport(reference, target)


def apply_fairpot(calibration, evaluation, *, reference_group, target_group, fraction):
    """Fit on calibration predictions and adjust the designated group only."""
    transport = _calibration_transport(calibration, reference_group, target_group)
    knots, mapped = _partial_knots(*transport, fraction)
    result = evaluation.copy()
    selected = result.cluster == target_group
    if fraction:
        result.loc[selected, "pred"] = np.interp(result.loc[selected, "pred"], knots, mapped)
    return result


def select_fairpot_fraction(calibration, *, reference_group, target_group, max_auc_drop):
    """Evaluate Algorithm 1's fraction grid on validation and select within an AUC budget."""
    if not 0 <= max_auc_drop <= 1:
        raise ValueError("FairPOT validation AUC drop must be between 0 and 1.")
    transport = _calibration_transport(calibration, reference_group, target_group)
    labels = calibration.label.to_numpy()
    scores = calibration.pred.to_numpy()
    reference = calibration.cluster.eq(reference_group).to_numpy()
    target = calibration.cluster.eq(target_group).to_numpy()
    reference_positive_target_negative = (reference & (labels == 1)) | (target & (labels == 0))
    target_positive_reference_negative = (target & (labels == 1)) | (reference & (labels == 0))
    if any(set(labels[pair]) != {0, 1}
           for pair in (reference_positive_target_negative, target_positive_reference_negative)):
        raise ValueError("FairPOT fraction selection requires both validation labels in each of the two groups.")

    results = []
    for step in range(11):
        fraction = step / 10
        adjusted = scores.copy()
        if step:
            knots, mapped = _partial_knots(*transport, fraction)
            adjusted[target] = np.interp(scores[target], knots, mapped)
        reference_xauc = roc_auc_score(labels[reference_positive_target_negative],
                                       adjusted[reference_positive_target_negative])
        target_xauc = roc_auc_score(labels[target_positive_reference_negative],
                                    adjusted[target_positive_reference_negative])
        results.append({
            "transport_fraction": fraction,
            "global_auc": roc_auc_score(labels, adjusted),
            "xauc_reference_positive": reference_xauc,
            "xauc_target_positive": target_xauc,
            "xauc_gap": abs(reference_xauc - target_xauc),
        })

    sweep = pd.DataFrame(results)
    auc = sweep.global_auc.to_numpy()
    gap = sweep.xauc_gap.to_numpy()
    better_auc = auc[:, None] > auc
    smaller_gap = gap[:, None] < gap
    dominates = (auc[:, None] >= auc) & (gap[:, None] <= gap) & (better_auc | smaller_gap)
    sweep["pareto_optimal"] = ~dominates.any(axis=0)
    sweep["auc_drop"] = auc[0] - auc
    sweep["within_auc_budget"] = sweep["auc_drop"] <= max_auc_drop + 1e-12
    best = sweep.loc[sweep["within_auc_budget"]].sort_values(
        ["xauc_gap", "global_auc", "transport_fraction"], ascending=[True, False, True],
    ).index[0]
    sweep["selected"] = sweep.index == best
    return float(sweep.loc[best, "transport_fraction"]), sweep
