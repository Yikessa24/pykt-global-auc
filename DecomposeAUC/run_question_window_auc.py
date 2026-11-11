"""Decompose the full pyKT question-window late-mean AUC by student group."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .plot_auc_matrix import compute_auc_matrix_by_cluster, plot_auc_matrix
from .plot_prediction_scores_by_cluster import plot_boxplot
from .run_posthoc_auc_irt_tertiles import GROUPS


UNASSIGNED = "unassigned"


def read_question_targets(test_csv):
    """Recover question IDs from the original test.csv order and is_repeat."""
    source = pd.read_csv(test_csv, dtype=str, usecols=["uid", "responses", "is_repeat"])
    if source["uid"].isna().any() or not source["uid"].is_unique:
        raise ValueError("test.csv must contain exactly one history per student.")

    targets = []
    next_qidx = 0
    for orirow, row in enumerate(source.itertuples(index=False)):
        labels = row.responses.split(",")
        repeats = row.is_repeat.split(",")
        if len(labels) != len(repeats) or repeats[0] != "0":
            raise ValueError(f"Invalid question sequence in test.csv at orirow={orirow}.")
        question_labels = [int(label) for label, repeat in zip(labels, repeats) if repeat == "0"]
        # The first question in each history is context only.
        targets.extend(
            (next_qidx + index, orirow, row.uid, label)
            for index, label in enumerate(question_labels[1:], start=1)
        )
        next_qidx += len(question_labels)
    return pd.DataFrame(targets, columns=["qidx", "orirow", "student_id", "label"])


def read_grouped_predictions(pred_txt, test_csv, cluster_csv):
    predictions = pd.read_csv(
        pred_txt, sep="\t", usecols=["qidx", "orirow", "late_mean", "late_trues"],
        dtype={"qidx": "Int64", "orirow": "Int64", "late_mean": float, "late_trues": float},
    ).rename(columns={"late_mean": "pred"})
    if (predictions[["qidx", "orirow"]].isna().any().any()
            or not predictions["qidx"].is_unique
            or not predictions["pred"].between(0, 1).all()
            or not predictions["late_trues"].isin([0, 1]).all()):
        raise ValueError("The prediction export has invalid or duplicate question rows.")

    targets = read_question_targets(test_csv)
    joined = targets.merge(predictions, on=["qidx", "orirow"], validate="one_to_one")
    if len(joined) != len(targets) or len(joined) != len(predictions):
        raise ValueError(
            f"Expected all {len(targets)} question targets from test.csv; "
            f"matched {len(joined)} of {len(predictions)} exported predictions."
        )
    if not joined["label"].eq(joined.pop("late_trues")).all():
        raise ValueError("The exported labels do not match test.csv.")

    clusters = pd.read_csv(cluster_csv, dtype=str, usecols=["student_id", "cluster"])
    if (clusters["student_id"].isna().any() or not clusters["student_id"].is_unique
            or not clusters["cluster"].isin(GROUPS).all()
            or not clusters["student_id"].isin(targets["student_id"]).all()):
        raise ValueError("Expected unique test-student IDs and low/mid/high IRT labels.")
    joined["cluster"] = joined["student_id"].map(clusters.set_index("student_id")["cluster"]).fillna(UNASSIGNED)
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-txt", type=Path, required=True, help="qid_test_question_window_predictions.txt")
    parser.add_argument("--test-csv", type=Path, required=True, help="Original test.csv from the same preprocessing run")
    parser.add_argument("--cluster-csv", type=Path, required=True, help="student_clusters_irt_tertiles.csv")
    parser.add_argument("--expected-auc", type=float, required=True, help="windowauclate_mean from wandb_predict")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites a run")
    parser.add_argument("--postprocessor", choices=["akt_fairpot", "akt_xorder"])
    parser.add_argument("--calibration-pred-txt", type=Path)
    parser.add_argument("--calibration-test-csv", type=Path)
    parser.add_argument("--calibration-cluster-csv", type=Path)
    parser.add_argument("--reference-group", choices=GROUPS, default="high")
    parser.add_argument("--target-group", choices=GROUPS, default="low")
    fairpot = parser.add_mutually_exclusive_group()
    fairpot.add_argument("--transport-fraction", type=float)
    fairpot.add_argument("--fairpot-max-auc-drop", type=float,
                         help="Select FairPOT fraction on validation within this global AUC drop")
    parser.add_argument("--xorder-penalty", type=float)
    parser.add_argument("--xorder-groups", nargs="+", choices=GROUPS, default=["high", "low", "mid"])
    parser.add_argument("--xorder-calibration-limit", type=int,
                        help="Optional deterministic rank sample per side of each quadratic DP.")
    args = parser.parse_args()
    if args.fairpot_max_auc_drop is not None and args.postprocessor != "akt_fairpot":
        parser.error("--fairpot-max-auc-drop requires --postprocessor akt_fairpot.")
    if args.postprocessor == "akt_fairpot" and args.transport_fraction is None and args.fairpot_max_auc_drop is None:
        parser.error("akt_fairpot needs --transport-fraction or --fairpot-max-auc-drop.")

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    rows = read_grouped_predictions(args.pred_txt, args.test_csv, args.cluster_csv)
    baseline_auc = roc_auc_score(rows["label"], rows["pred"])
    # wandb_predict displays the final metric rounded to six decimals.
    if not np.isclose(baseline_auc, args.expected_auc, rtol=0, atol=5e-7 + 1e-12):
        raise ValueError(f"Recomputed AUC {baseline_auc:.9f} differs from {args.expected_auc}.")

    if args.postprocessor:
        inputs = (args.calibration_pred_txt, args.calibration_test_csv, args.calibration_cluster_csv)
        if not all(inputs):
            parser.error("Score adjustment needs the three calibration files.")
        calibration = read_grouped_predictions(*inputs)
        if not set(calibration.student_id).isdisjoint(rows.student_id):
            raise ValueError("Calibration and evaluation must use disjoint students from the same ID space.")

    fairpot_sweep = None
    if args.postprocessor == "akt_fairpot":
        from pykt.models.akt_fairpot import apply_fairpot, select_fairpot_fraction

        transport_fraction = args.transport_fraction
        if args.fairpot_max_auc_drop is not None:
            transport_fraction, fairpot_sweep = select_fairpot_fraction(
                calibration, reference_group=args.reference_group, target_group=args.target_group,
                max_auc_drop=args.fairpot_max_auc_drop,
            )
        rows = apply_fairpot(calibration, rows, reference_group=args.reference_group,
                            target_group=args.target_group, fraction=transport_fraction)
    elif args.postprocessor == "akt_xorder":
        from pykt.models.akt_xorder import apply_xorder

        if args.xorder_penalty is None:
            parser.error("akt_xorder needs --xorder-penalty.")
        rows = apply_xorder(calibration, rows, group_order=args.xorder_groups,
                           penalty=args.xorder_penalty, calibration_limit=args.xorder_calibration_limit)

    matrix, weights, counts, summary = compute_auc_matrix_by_cluster(
        rows, cluster_order=GROUPS + [UNASSIGNED],
    )
    if summary["n_samples"] != len(rows) or summary["abs_gap"] > 1e-12:
        raise ValueError("The matrix did not reconstruct AUC on the full set of predictions.")

    counts["n_students"] = counts["cluster"].map(rows.groupby("cluster")["student_id"].nunique())
    summary.update(
        metric="windowauclate_mean",
        reported_auc=args.expected_auc,
        prediction_source=str(args.pred_txt.resolve()),
        test_source=str(args.test_csv.resolve()),
        cluster_source=str(args.cluster_csv.resolve()),
        n_students=int(rows["student_id"].nunique()),
    )

    if args.postprocessor:
        summary.update(
            postprocessor=args.postprocessor,
            baseline_auc=baseline_auc,
            calibration_source=str(args.calibration_pred_txt.resolve()),
        )
    if args.postprocessor == "akt_fairpot":
        summary.update(
            reference_group=args.reference_group,
            target_group=args.target_group,
            transport_fraction=transport_fraction,
        )
        if fairpot_sweep is not None:
            summary.update(
                fairpot_max_auc_drop=args.fairpot_max_auc_drop,
                fairpot_validation_metrics=fairpot_sweep.loc[fairpot_sweep["selected"]].iloc[0].to_dict(),
            )

    if args.postprocessor == "akt_xorder":
        summary.update(
            xorder_penalty=args.xorder_penalty,
            xorder_groups=args.xorder_groups,
            xorder_calibration_limit=args.xorder_calibration_limit,
        )

    args.output_dir.mkdir(parents=True)
    if fairpot_sweep is not None:
        fairpot_sweep.to_csv(args.output_dir / "fairpot_validation_sweep.csv", index=False)
    matrix.to_csv(args.output_dir / "auc_matrix.csv")
    weights.to_csv(args.output_dir / "auc_weights.csv")
    counts.to_csv(args.output_dir / "auc_counts.csv", index=False)
    fig, _ = plot_auc_matrix(
        matrix, title="Question-window late-mean AUC", output_fig=str(args.output_dir / "auc_matrix.png"),
    )
    fig.clear()
    plot_boxplot(
        rows.rename(columns={"cluster": "student_cluster", "pred": "y_pred", "label": "true_label"}),
        dataset_name=args.test_csv.parent.name,
        output_path=str(args.output_dir / "prediction_score_boxplot.png"),
        cluster_order=list(matrix.index),
        hue_order=[1, 0],
        title="Question-window late-mean prediction scores",
    )
    (args.output_dir / "auc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
