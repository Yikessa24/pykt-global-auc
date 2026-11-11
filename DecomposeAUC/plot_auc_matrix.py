"""
Compute and plot cluster-level AUC decomposition matrix.

This module is fully decoupled from pykt training code. It accepts model
predictions and student cluster labels, then computes AUC_ij for all
(pos_cluster=i, neg_cluster=j) pairs and plots a heatmap.

The decomposition follows Theorem 1 style aggregation:
Global AUC = sum_{i,j} w_ij * AUC_ij,
where w_ij = P(C+=i, C-=j) = (N_pos_i * N_neg_j)/(N_pos * N_neg).

Inputs:
- Prediction DataFrame with columns: student_id, pred, label
- Cluster DataFrame with columns: student_id, cluster or cluster_label
    (the script prefers cluster_label when it is available)

Outputs:
- AUC matrix DataFrame (rows: pos cluster i, cols: neg cluster j)
- Optional matrix/weight CSV exports
- Heatmap figure

Dependencies:
- pandas, numpy, matplotlib, seaborn, sklearn

Example:
python plot_auc_matrix.py \
  --pred-csv ./predictions.csv \
  --cluster-csv ./assist2015_student_clusters.csv \
  --output-fig ./auc_matrix.png \
  --output-matrix-csv ./auc_matrix.csv

Or directly from pykt eval txt:
python plot_auc_matrix.py \
    --pred-txt ../examples/saved_model/.../qid_test_ratio0.9_False_predictions.txt \
    --cluster-csv ./assist2015_student_clusters.csv \
    --output-fig ./auc_matrix.png

python DecomposeAUC/plot_auc_matrix.py \
 --pred-csv DecomposeAUC/assist2015_akt_pykt_pred_long.csv \
 --cluster-csv DecomposeAUC/auc_figures/assist2015_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/student_clusters_irtwofit.csv \
 --output-fig DecomposeAUC/auc_figures/assist2015_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/auc_matrix.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

try:
    import seaborn as sns
except Exception:
    sns = None


DEFAULT_OUTPUT_DIR = "DecomposeAUC/auc_figures"
_DATASET_NAME_PATTERN = re.compile(
    r"(assist\d+|assistments\d+|algebra2005|bridge2algebra2006|junyi|ednet|statics2011|nips_task34|poj)",
    re.IGNORECASE,
)


def _validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_cluster_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _infer_dataset_name_from_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        match = _DATASET_NAME_PATTERN.search(str(value))
        if match:
            return match.group(1)
    return "dataset"


def _clean_artifact_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-_")
    return cleaned or "run"


def _strip_prediction_suffix(stem: str) -> str:
    suffixes = [
        r"(_pykt_pred_long)$",
        r"(_pred_long)$",
        r"(_test_window_predictions)$",
        r"(_window_predictions)$",
        r"(_test_predictions)$",
        r"(_predictions)$",
        r"(_pred)$",
    ]

    out = stem
    changed = True
    while changed:
        changed = False
        for pattern in suffixes:
            new_out = re.sub(pattern, "", out, flags=re.IGNORECASE)
            if new_out != out:
                out = new_out
                changed = True
    return out.rstrip("_-") or "run"


def _infer_model_tag(pred_csv: str, pred_txt: str) -> str:
    src = pred_txt if pred_txt else pred_csv
    stem = os.path.splitext(os.path.basename(src))[0]
    stem = _strip_prediction_suffix(stem)
    return _clean_artifact_name(stem)


def _infer_cluster_method_name(
    cluster_csv: str,
    cluster_df: pd.DataFrame,
    cluster_col: str,
) -> str:
    if cluster_csv:
        stem = os.path.splitext(os.path.basename(cluster_csv))[0]
        stem = re.sub(r"^(student_)?clusters?[_-]?", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"^(cluster|clusters)[_-]?", "", stem, flags=re.IGNORECASE)
        if stem and stem.lower() not in {"student", "cluster", "clusters"}:
            return _clean_artifact_name(stem)

    if cluster_col not in cluster_df.columns:
        return "cluster"

    labels = {
        _normalize_cluster_value(value)
        for value in pd.unique(cluster_df[cluster_col])
        if _normalize_cluster_value(value)
    }

    if not labels:
        return "cluster"

    if labels <= {"bad", "mid", "good"}:
        if any(column in cluster_df.columns for column in ["cluster_label", "cluster_legacy", "theta_report", "threshold_low_theta"]):
            return "localband_irt"
        return "cluster3"

    if {"mid_chaotic", "mid_stable"}.issubset(labels) or labels <= {"bad", "mid_chaotic", "mid_stable", "good"}:
        return "irtwofit"

    return "cluster"


def _build_artifact_prefix(model_tag: str, cluster_method: str) -> str:
    model_tag = _clean_artifact_name(model_tag)
    cluster_method = _clean_artifact_name(cluster_method)
    if not cluster_method or cluster_method == "cluster":
        return model_tag
    return f"{model_tag}_{cluster_method}"


def _resolve_cluster_column(df: pd.DataFrame, preferred_cluster_col: str = "cluster") -> str:
    if "cluster_label" in df.columns and df["cluster_label"].notna().any():
        return "cluster_label"
    if preferred_cluster_col in df.columns:
        return preferred_cluster_col
    if "cluster" in df.columns:
        return "cluster"
    if "cluster_label" in df.columns:
        return "cluster_label"
    raise ValueError("Could not find a cluster column. Expected 'cluster_label' or 'cluster'.")


def _resolve_cluster_order(
    df: pd.DataFrame,
    cluster_col: str,
    cluster_order: Optional[List[str]] = None,
) -> List[str]:
    if cluster_col not in df.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")

    work = df[[cluster_col] + (["theta"] if "theta" in df.columns else [])].copy()
    work[cluster_col] = work[cluster_col].map(_normalize_cluster_value)
    work = work[work[cluster_col] != ""]

    observed_order = [value for value in pd.unique(work[cluster_col]) if value != ""]
    if not observed_order:
        raise ValueError(f"No valid cluster labels found in column: {cluster_col}")

    if cluster_order is not None:
        explicit_order = [str(item).strip() for item in cluster_order if str(item).strip()]
        if explicit_order:
            explicit_order = [item for item in explicit_order if item in observed_order]
            if explicit_order and set(explicit_order) == set(observed_order):
                return explicit_order

    if "theta" in work.columns and work["theta"].notna().any():
        theta_order = (
            work.dropna(subset=["theta"])
            .groupby(cluster_col, sort=False)["theta"]
            .mean()
            .sort_values(kind="mergesort")
            .index.astype(str)
            .tolist()
        )
        theta_order = [item for item in theta_order if item in observed_order]
        remaining = [item for item in observed_order if item not in theta_order]
        return theta_order + remaining

    return observed_order


def _compute_plot_style(cluster_order: Sequence[str]) -> Dict[str, float]:
    n_clusters = max(1, len(cluster_order))
    max_label_len = max((len(str(label)) for label in cluster_order), default=1)

    cell_size = 1.05 if n_clusters <= 5 else 0.95 if n_clusters <= 8 else 0.85
    width = max(6.8, cell_size * n_clusters + 2.2 + 0.35 * max_label_len)
    height = max(5.8, cell_size * n_clusters + 2.2)

    annot_fontsize = max(6, min(12, int(round(13 - 0.5 * max(0, n_clusters - 4)))))
    tick_fontsize = max(8, min(12, int(round(12 - 0.25 * max(0, n_clusters - 4)))))
    title_fontsize = max(12, min(18, int(round(20 - 0.5 * max(0, n_clusters - 4)))))
    x_rotation = 45 if max_label_len <= 8 else 60

    return {
        "fig_width": float(width),
        "fig_height": float(height),
        "annot_fontsize": float(annot_fontsize),
        "tick_fontsize": float(tick_fontsize),
        "title_fontsize": float(title_fontsize),
        "x_rotation": float(x_rotation),
    }


def _load_pred_from_pykt_txt(
    txt_path: str,
    student_col: str,
    pred_col: str,
    label_col: str,
    uid_index: int = 1,
    label_index: int = -3,
    pred_index: int = -2,
    delimiter: str = "\t",
) -> pd.DataFrame:
    """
    Parse pykt split-predict txt lines into a DataFrame with
    [student_col, pred_col, label_col].

    Default index mapping follows pykt evaluate_model split output:
    ... \t uid \t ... \t true \t pred \t pred_label
    """
    rows: List[Tuple[str, float, float]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split(delimiter)
            try:
                uid = str(parts[uid_index]).strip()
                label = float(parts[label_index])
                pred = float(parts[pred_index])
            except Exception:
                continue

            rows.append((uid, pred, label))

    if not rows:
        raise ValueError(f"No valid prediction rows parsed from txt: {txt_path}")

    return pd.DataFrame(rows, columns=[student_col, pred_col, label_col])


def _auc_from_pos_neg(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return np.nan

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y_true, y_score))


def compute_auc_matrix_by_cluster(
    df: pd.DataFrame,
    pred_col: str = "pred",
    label_col: str = "label",
    cluster_col: str = "cluster",
    cluster_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    resolved_cluster_col = _resolve_cluster_column(df, cluster_col)

    work_cols = [pred_col, label_col, resolved_cluster_col]
    if "theta" in df.columns:
        work_cols.append("theta")

    _validate_columns(df, work_cols)

    work = df[work_cols].copy()
    work = work.dropna(subset=[pred_col, label_col, resolved_cluster_col])
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[pred_col, label_col])
    work[label_col] = (work[label_col] >= 0.5).astype(int)
    work[resolved_cluster_col] = work[resolved_cluster_col].map(_normalize_cluster_value)
    work = work[work[resolved_cluster_col] != ""]

    resolved_order = _resolve_cluster_order(work, resolved_cluster_col, cluster_order)

    total_pos = int((work[label_col] == 1).sum())
    total_neg = int((work[label_col] == 0).sum())
    if total_pos == 0 or total_neg == 0:
        raise ValueError("Need both positive and negative labels to compute AUC.")

    pos_counts = {
        c: int(((work[label_col] == 1) & (work[resolved_cluster_col] == c)).sum()) for c in resolved_order
    }
    neg_counts = {
        c: int(((work[label_col] == 0) & (work[resolved_cluster_col] == c)).sum()) for c in resolved_order
    }
    total_counts = {
        c: int((work[resolved_cluster_col] == c).sum()) for c in resolved_order
    }

    auc_mat = np.full((len(resolved_order), len(resolved_order)), np.nan, dtype=np.float64)
    w_mat = np.zeros_like(auc_mat)

    for i, pos_cluster in enumerate(resolved_order):
        pos_scores = work.loc[
            (work[label_col] == 1) & (work[resolved_cluster_col] == pos_cluster), pred_col
        ].to_numpy(dtype=float)

        for j, neg_cluster in enumerate(resolved_order):
            neg_scores = work.loc[
                (work[label_col] == 0) & (work[resolved_cluster_col] == neg_cluster), pred_col
            ].to_numpy(dtype=float)

            auc_mat[i, j] = _auc_from_pos_neg(pos_scores, neg_scores)
            w_mat[i, j] = (len(pos_scores) * len(neg_scores)) / float(total_pos * total_neg)

    auc_df = pd.DataFrame(auc_mat, index=resolved_order, columns=resolved_order)
    weight_df = pd.DataFrame(w_mat, index=resolved_order, columns=resolved_order)

    global_auc = float(roc_auc_score(work[label_col].to_numpy(), work[pred_col].to_numpy()))
    reconstructed_auc = float(np.nansum(auc_mat * w_mat))

    counts_df = pd.DataFrame(
        {
            "cluster": resolved_order,
            "n_total": [total_counts[c] for c in resolved_order],
            "n_pos": [pos_counts[c] for c in resolved_order],
            "n_neg": [neg_counts[c] for c in resolved_order],
        }
    )

    summary = {
        "global_auc": global_auc,
        "reconstructed_auc": reconstructed_auc,
        "abs_gap": float(abs(global_auc - reconstructed_auc)),
        "n_samples": int(work.shape[0]),
        "n_pos": total_pos,
        "n_neg": total_neg,
        "n_clusters": int(len(resolved_order)),
        "cluster_labels": resolved_order,
        "cluster_col_used": resolved_cluster_col,
    }

    return auc_df, weight_df, counts_df, summary


def plot_auc_matrix(
    auc_df: pd.DataFrame,
    cluster_order: Optional[List[str]] = None,
    title: str = "AUC matrix AUC_ij (pos=i, neg=j)",
    xlabel: str = "neg cluster (j)",
    ylabel: str = "pos cluster (i)",
    cmap: str = "coolwarm",
    vmin: float = 0.3,
    vmax: float = 1.0,
    dpi: int = 150,
    output_fig: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if cluster_order is None:
        cluster_order = list(auc_df.index)

    cluster_order = [str(item) for item in cluster_order]
    plot_df = auc_df.reindex(index=cluster_order, columns=cluster_order)
    style = _compute_plot_style(cluster_order)

    fig, ax = plt.subplots(figsize=(style["fig_width"], style["fig_height"]))

    if sns is not None:
        sns.set_theme(style="white")
        sns.heatmap(
            plot_df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": style["annot_fontsize"]},
            cbar_kws={"label": "value", "shrink": 0.92},
            linewidths=0.0,
            square=True,
        )
    else:
        arr = plot_df.to_numpy(dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                txt = "nan" if np.isnan(val) else f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=style["annot_fontsize"])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("value")
        ax.set_xticks(np.arange(len(cluster_order)))
        ax.set_yticks(np.arange(len(cluster_order)))
        ax.set_xticklabels(cluster_order)
        ax.set_yticklabels(cluster_order)

    ax.set_title(title, fontsize=style["title_fontsize"], pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", rotation=style["x_rotation"], labelsize=style["tick_fontsize"])
    ax.tick_params(axis="y", rotation=0, labelsize=style["tick_fontsize"])

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")

    fig.tight_layout()

    if output_fig:
        fig.savefig(output_fig, dpi=dpi, bbox_inches="tight")

    return fig, ax


def _parse_cluster_order(order_str: str) -> List[str]:
    vals = [s.strip() for s in order_str.split(",") if s.strip()]
    if not vals:
        raise ValueError("cluster_order cannot be empty")
    return vals


def _resolve_output_path(path_arg: str, output_dir: str, default_name: str) -> str:
    """Resolve output path; bare filenames are placed under output_dir."""
    if path_arg:
        if os.path.isabs(path_arg) or os.path.dirname(path_arg):
            parent = os.path.dirname(path_arg)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return path_arg
        return os.path.join(output_dir, path_arg)
    return os.path.join(output_dir, default_name)


def _infer_run_tag(pred_csv: str, pred_txt: str) -> str:
    return _infer_model_tag(pred_csv, pred_txt)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute and plot cluster-level AUC matrix.")
    parser.add_argument("--pred-csv", type=str, default="")
    parser.add_argument("--pred-txt", type=str, default="")
    parser.add_argument("--cluster-csv", type=str, default="")

    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--output-fig", type=str, default="")
    parser.add_argument("--output-matrix-csv", type=str, default="")
    parser.add_argument("--output-weights-csv", type=str, default="")
    parser.add_argument("--output-counts-csv", type=str, default="")
    parser.add_argument("--output-summary-json", type=str, default="")

    parser.add_argument("--student-col", type=str, default="student_id")
    parser.add_argument("--pred-col", type=str, default="pred")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--cluster-col", type=str, default="cluster")
    parser.add_argument("--pred-txt-uid-index", type=int, default=1)
    parser.add_argument("--pred-txt-label-index", type=int, default=-3)
    parser.add_argument("--pred-txt-score-index", type=int, default=-2)
    parser.add_argument("--pred-txt-delimiter", type=str, default="\t")
    parser.add_argument(
        "--cluster-order",
        type=str,
        default="",
        help="Optional comma-separated order; leave empty to auto-detect the actual cluster labels.",
    )

    parser.add_argument("--vmin", type=float, default=0.3)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.pred_csv and not args.pred_txt:
        raise ValueError("At least one of --pred-csv or --pred-txt must be provided.")

    if args.pred_csv:
        pred_df = pd.read_csv(args.pred_csv)
    else:
        pred_df = _load_pred_from_pykt_txt(
            txt_path=args.pred_txt,
            student_col=args.student_col,
            pred_col=args.pred_col,
            label_col=args.label_col,
            uid_index=args.pred_txt_uid_index,
            label_index=args.pred_txt_label_index,
            pred_index=args.pred_txt_score_index,
            delimiter=args.pred_txt_delimiter,
        )

    pred_df[args.student_col] = pred_df[args.student_col].astype(str)

    cluster_source_df = pred_df
    if args.cluster_csv:
        cluster_df = pd.read_csv(args.cluster_csv)
        _validate_columns(cluster_df, [args.student_col])
        cluster_df[args.student_col] = cluster_df[args.student_col].astype(str)
        cluster_source_df = cluster_df

        cluster_columns_to_merge = [args.student_col]
        for candidate in [args.cluster_col, "cluster_label", "cluster_legacy", "theta"]:
            if candidate in cluster_df.columns and candidate not in pred_df.columns and candidate not in cluster_columns_to_merge:
                cluster_columns_to_merge.append(candidate)

        if len(cluster_columns_to_merge) > 1:
            pred_df = pred_df.merge(
                cluster_df[cluster_columns_to_merge],
                on=args.student_col,
                how="left",
            )

    _validate_columns(pred_df, [args.pred_col, args.label_col, args.student_col])

    if not any(column in pred_df.columns for column in ["cluster_label", args.cluster_col, "cluster"]):
        raise ValueError(
            "Could not find cluster information. Provide --cluster-csv, or pass a DataFrame that already contains 'cluster_label' or 'cluster'."
        )

    resolved_cluster_col = _resolve_cluster_column(cluster_source_df, args.cluster_col)
    cluster_method = _infer_cluster_method_name(args.cluster_csv, cluster_source_df, resolved_cluster_col)
    model_tag = _infer_run_tag(args.pred_csv, args.pred_txt)
    artifact_prefix = _build_artifact_prefix(model_tag, cluster_method)

    cluster_order_arg = [item.strip() for item in args.cluster_order.split(",") if item.strip()]
    if cluster_order_arg:
        cluster_order = cluster_order_arg
    elif args.cluster_csv:
        cluster_order = _resolve_cluster_order(cluster_source_df, resolved_cluster_col, None)
    else:
        cluster_order = None

    fig_path = _resolve_output_path(args.output_fig, args.output_dir, f"{artifact_prefix}_auc_matrix.png")
    output_base_dir = os.path.dirname(fig_path) or args.output_dir

    matrix_path = _resolve_output_path(args.output_matrix_csv, output_base_dir, f"{artifact_prefix}_auc_matrix.csv")
    weights_path = _resolve_output_path(args.output_weights_csv, output_base_dir, f"{artifact_prefix}_auc_weights.csv")
    counts_path = _resolve_output_path(args.output_counts_csv, output_base_dir, f"{artifact_prefix}_cluster_counts.csv")
    summary_path = _resolve_output_path(args.output_summary_json, output_base_dir, f"{artifact_prefix}_auc_summary.json")

    dataset_name = _infer_dataset_name_from_text(args.cluster_csv, args.pred_csv, args.pred_txt, args.tag, fig_path)

    auc_df, weight_df, counts_df, summary = compute_auc_matrix_by_cluster(
        pred_df,
        pred_col=args.pred_col,
        label_col=args.label_col,
        cluster_col=resolved_cluster_col,
        cluster_order=cluster_order,
    )

    n_clusters = int(auc_df.shape[0])
    title = f"Decomposed AUC Matrix ({n_clusters} clusters - {dataset_name.upper()})"

    plot_auc_matrix(
        auc_df,
        cluster_order=list(auc_df.index.astype(str)),
        title=title,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
        output_fig=fig_path,
    )

    print("AUC decomposition summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    auc_df.to_csv(matrix_path)
    print(f"Saved AUC matrix to: {matrix_path}")

    weight_df.to_csv(weights_path)
    print(f"Saved AUC weights to: {weights_path}")

    counts_df.to_csv(counts_path, index=False)
    print(f"Saved cluster counts to: {counts_path}")

    summary_payload = {
        **summary,
        "dataset_name": dataset_name,
        "n_clusters": n_clusters,
        "cluster_labels": [str(item) for item in auc_df.index.tolist()],
        "title": title,
        "artifact_prefix": artifact_prefix,
        "model_tag": model_tag,
        "cluster_method": cluster_method,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=True, indent=2)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()
