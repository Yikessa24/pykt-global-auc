"""
Prediction score boxplot by student cluster.

功能说明：
    1. 读取已经整理好的明细数据，或者从 pykt 的 pred_txt + cluster_csv + test_sequences.csv
   还原到交互级别数据。
    2. 对每个 (student_cluster, true_label) 组合做 IQR (1.5 * IQR) 异常值诊断统计。
    3. 打印每个分组的异常值统计、过滤前后样本数量、过滤后的分数统计。
    4. 为保证与 AUC 分解矩阵的数据口径一致，箱型图使用原始预测分数绘制，不把 IQR 过滤后的子集直接拿去画图。
    5. 按参考图风格绘制箱型图。
    6. 将图片保存到与 AUC 分解矩阵脚本一致的 run 目录中：DecomposeAUC/auc_figures/<run_tag>/prediction_score_boxplot.png。

推荐用法：
cd /root/autodl-tmp/pykt-toolkit-main
python DecomposeAUC/plot_prediction_scores_by_cluster.py \
  --save-dir examples/saved_model/assist2017_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0 \
  --pred-txt examples/saved_model/assist2017_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/qid_test_predictions.txt

python DecomposeAUC/plot_prediction_scores_by_cluster.py \
  --save-dir examples/saved_model/assist2009_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0 \
  --pred-txt examples/saved_model/assist2009_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/qid_test_predictions.txt

python DecomposeAUC/plot_prediction_scores_by_cluster.py \
  --save-dir examples/saved_model/assist2017_akt_dro_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_2.0_0.7_0.5_1_4_0.2_0.0_4096_0.2_0.4_0.7_0.35_0.001_0.05_0_0 \
  --pred-txt examples/saved_model/assist2017_akt_dro_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_2.0_0.7_0.5_1_4_0.2_0.0_4096_0.2_0.4_0.7_0.35_0.001_0.05_0_0/qid_test_predictions.txt

如果你已经有整理好的 DataFrame/CSV，也可以直接传：
conda run -n pykt python DecomposeAUC/plot_prediction_scores_by_cluster.py \
  --input-csv path/to/your_dataframe.csv

CSV 需要至少包含以下列之一组映射：
student_cluster / true_label / y_pred
或者通过参数显式指定列名。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "DecomposeAUC" / "auc_figures"
DEFAULT_CLUSTER_ORDER = ["bad", "mid_chaotic", "mid_stable", "good"]
DEFAULT_LABEL_ORDER = [1, 0]
POINT_JITTER_SEED = 3407

LABEL_DISPLAY_MAP = {
    1: "Positive (1)",
    0: "Negative (0)",
}

LABEL_COLOR_MAP = {
    1: "#2ecc71",
    0: "#e74c3c",
}


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def _safe_read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _infer_dataset_name(save_dir: str, dataset_name: str = "") -> str:
    if dataset_name:
        return dataset_name

    if save_dir:
        cfg_path = Path(save_dir) / "config.json"
        if cfg_path.exists():
            try:
                config = _safe_read_json(cfg_path)
                inferred = str(config.get("params", {}).get("dataset_name", "")).strip()
                if inferred:
                    return inferred
            except Exception:
                pass

        return Path(save_dir).name

    return "dataset"


def _infer_run_tag(save_dir: str, dataset_name: str = "", tag: str = "") -> str:
    if tag:
        return tag.strip()

    if save_dir:
        return Path(save_dir.rstrip("/")).name

    if dataset_name:
        return dataset_name.strip()

    return "run"


def _infer_sequence_csv(save_dir: str, dataset_name: str) -> str:
    if not save_dir:
        return ""

    cfg_path = Path(save_dir) / "config.json"
    if not cfg_path.exists():
        return ""

    try:
        config = _safe_read_json(cfg_path)
        data_config = config.get("data_config", {})
        dpath = data_config.get("dpath", "")
        test_file = data_config.get("test_file", "test_sequences.csv")
        if dpath:
            return _normalize_path(str(Path(save_dir).parents[1] / dpath / test_file))
    except Exception:
        return ""

    return ""


def _infer_cluster_csv(output_root: str, run_tag: str) -> str:
    if not output_root or not run_tag:
        return ""

    candidate_names = [
        "student_clusters_smooth.csv",
        "student_clusters_irt.csv",
        "student_clusters_teststyle_student.csv",
        "student_clusters.csv",
    ]

    run_dir = Path(output_root) / run_tag
    for name in candidate_names:
        candidate = run_dir / name
        if candidate.exists():
            return str(candidate)
    return ""


def _read_uid_sequence(sequence_csv: str, uid_col: str) -> List[str]:
    df = pd.read_csv(sequence_csv, usecols=[uid_col])
    return df[uid_col].astype(str).tolist()


def _parse_pykt_sequence_prediction_txt(
    pred_txt: str,
    sequence_uids: Sequence[str],
) -> pd.DataFrame:
    """把 pykt 的 sequence 级预测 txt 展开成交互级明细。"""
    rows: List[Tuple[str, float, float]] = []

    with open(pred_txt, "r", encoding="utf-8") as fin:
        pred_lines = [raw.strip() for raw in fin if raw.strip()]

    if len(pred_lines) != len(sequence_uids):
        raise ValueError(
            "预测 txt 行数与 sequence CSV 行数不一致："
            f"{len(pred_lines)} != {len(sequence_uids)}"
        )

    for seq_idx, raw in enumerate(pred_lines):
        try:
            parsed = ast.literal_eval(raw.replace("nan", "None"))
        except Exception as exc:
            raise ValueError(f"无法解析预测文件第 {seq_idx + 1} 行：{raw[:120]}") from exc

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 5:
            raise ValueError(f"预测文件第 {seq_idx + 1} 行格式不正确：{raw[:120]}")

        true_labels = parsed[3]
        pred_scores = parsed[4]
        if not isinstance(true_labels, (list, tuple)) or not isinstance(pred_scores, (list, tuple)):
            raise ValueError(f"预测文件第 {seq_idx + 1} 行的标签/分数不是列表：{raw[:120]}")

        uid = sequence_uids[seq_idx]
        for y_true, y_pred in zip(true_labels, pred_scores):
            if y_true is None or y_pred is None:
                continue
            rows.append((uid, float(y_true), float(y_pred)))

    return pd.DataFrame(rows, columns=["student_id", "true_label", "y_pred"])


def _parse_tab_prediction_txt(pred_txt: str) -> pd.DataFrame:
    """兼容旧的 tab 分隔预测文本。"""
    rows: List[Tuple[str, float, float]] = []
    with open(pred_txt, "r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue

            parts = line.split("\t")
            try:
                uid = str(parts[1]).strip()
                y_true = float(parts[-3])
                y_pred = float(parts[-2])
            except Exception:
                continue

            rows.append((uid, y_true, y_pred))

    return pd.DataFrame(rows, columns=["student_id", "true_label", "y_pred"])



def _apply_cluster_merge(pred_df, cluster_csv, cluster_student_col, cluster_label_col):
    cluster_df = pd.read_csv(cluster_csv)
    if cluster_student_col not in cluster_df.columns:
        raise ValueError(
            f"cluster_csv 中找不到学生 ID 列 `{cluster_student_col}`，实际列名：{list(cluster_df.columns)}"
        )
    if cluster_label_col not in cluster_df.columns:
        raise ValueError(
            f"cluster_csv 中找不到簇标签列 `{cluster_label_col}`，实际列名：{list(cluster_df.columns)}"
        )

    cluster_df = cluster_df[[cluster_student_col, cluster_label_col]].copy()
    cluster_df[cluster_student_col] = cluster_df[cluster_student_col].astype(str)

    pred_df["student_id"] = pred_df["student_id"].astype(str)
    merged = pred_df.merge(
        cluster_df,
        left_on="student_id",
        right_on=cluster_student_col,
        how="left",
    )

    matched_rows = int(merged[cluster_label_col].notna().sum())
    if matched_rows == 0:
        raise ValueError(
            "prediction_csv 中的学生与 cluster_csv 没有任何交集，无法绘图。"
            f"pred_students={pred_df['student_id'].nunique()}, cluster_students={cluster_df[cluster_student_col].nunique()}."
        )

    if merged[cluster_label_col].isna().any():
        missing = int(merged[cluster_label_col].isna().sum())
        print(f"[WARN] 有 {missing} 条预测记录没有匹配到 cluster，已保留但在绘图前剔除。")

    merged = merged.rename(
        columns={
            cluster_label_col: "student_cluster",
            "true_label": "true_label",
            "y_pred": "y_pred",
        }
    )
    merged = merged[["student_cluster", "true_label", "y_pred"]].copy()
    merged["student_cluster"] = merged["student_cluster"].astype(str)
    merged["true_label"] = pd.to_numeric(merged["true_label"], errors="coerce")
    merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
    merged = merged.dropna(subset=["student_cluster", "true_label", "y_pred"])
    merged["true_label"] = merged["true_label"].astype(int)

    return merged

def load_prediction_dataframe_from_csv(
    pred_csv: str,
    cluster_csv: str,
    cluster_student_col: str,
    cluster_label_col: str,
) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv)
    if "pred" in pred_df.columns:
        pred_df = pred_df.rename(columns={"pred": "y_pred"})
    if "label" in pred_df.columns:
        pred_df = pred_df.rename(columns={"label": "true_label"})
    if "student_id" not in pred_df.columns:
        if "uid" in pred_df.columns:
            pred_df = pred_df.rename(columns={"uid": "student_id"})
    return _apply_cluster_merge(pred_df, cluster_csv, cluster_student_col, cluster_label_col)

def load_prediction_dataframe_from_files(
    pred_txt: str,
    cluster_csv: str,
    sequence_csv: str,
    sequence_uid_col: str,
    cluster_student_col: str,
    cluster_label_col: str,
) -> pd.DataFrame:
    """从 pykt 预测结果还原出 student_cluster / true_label / y_pred 三列明细。"""
    if not os.path.exists(pred_txt):
        raise FileNotFoundError(f"pred_txt 不存在：{pred_txt}")
    if not os.path.exists(cluster_csv):
        raise FileNotFoundError(f"cluster_csv 不存在：{cluster_csv}")
    if not os.path.exists(sequence_csv):
        raise FileNotFoundError(f"sequence_csv 不存在：{sequence_csv}")

    sequence_uids = _read_uid_sequence(sequence_csv, sequence_uid_col)

    with open(pred_txt, "r", encoding="utf-8") as fin:
        first_non_empty = next((line.strip() for line in fin if line.strip()), "")

    if first_non_empty.startswith("[[") or first_non_empty.startswith("["):
        pred_df = _parse_pykt_sequence_prediction_txt(pred_txt, sequence_uids)
    else:
        pred_df = _parse_tab_prediction_txt(pred_txt)

    return _apply_cluster_merge(pred_df, cluster_csv, cluster_student_col, cluster_label_col)


def load_input_dataframe(
    input_csv: str,
    cluster_col: str,
    label_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """直接读取用户已经整理好的 DataFrame/CSV。"""
    df = pd.read_csv(input_csv)

    rename_map = {}
    if cluster_col in df.columns and cluster_col != "student_cluster":
        rename_map[cluster_col] = "student_cluster"
    if label_col in df.columns and label_col != "true_label":
        rename_map[label_col] = "true_label"
    if pred_col in df.columns and pred_col != "y_pred":
        rename_map[pred_col] = "y_pred"

    df = df.rename(columns=rename_map)

    required = ["student_cluster", "true_label", "y_pred"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"输入 CSV 缺少必要列：{missing}。当前列：{list(df.columns)}"
        )

    df = df[required].copy()
    df["student_cluster"] = df["student_cluster"].astype(str)
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=required)
    df["true_label"] = df["true_label"].astype(int)
    return df


def apply_iqr_filter(
    df: pd.DataFrame,
    cluster_col: str = "student_cluster",
    label_col: str = "true_label",
    pred_col: str = "y_pred",
    cluster_order: Optional[Sequence[str]] = None,
    label_order: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """按每个簇 + 标签组合做 IQR 异常值过滤，并返回详细统计。"""
    filtered_parts: List[pd.DataFrame] = []
    stats: List[Dict] = []

    if cluster_order is None:
        cluster_order = list(df[cluster_col].dropna().astype(str).unique())
    if label_order is None:
        label_order = sorted(df[label_col].dropna().astype(int).unique().tolist(), reverse=True)

    grouped = df.groupby([cluster_col, label_col], sort=False)
    for cluster_name in cluster_order:
        for label_value in label_order:
            if (cluster_name, label_value) not in grouped.groups:
                continue

            group = grouped.get_group((cluster_name, label_value))
            scores = group[pred_col].astype(float)
            q1 = scores.quantile(0.25)
            q3 = scores.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask = (scores < lower) | (scores > upper)
            inlier_mask = ~outlier_mask

            removed = group.loc[outlier_mask, pred_col].astype(float)
            kept = group.loc[inlier_mask].copy()

            filtered_parts.append(kept)

            stats.append(
                {
                    "cluster": cluster_name,
                    "label": int(label_value),
                    "original_count": int(len(group)),
                    "removed_count": int(outlier_mask.sum()),
                    "remaining_count": int(len(kept)),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "removed_min": float(removed.min()) if len(removed) else None,
                    "removed_max": float(removed.max()) if len(removed) else None,
                    "removed_examples": [round(float(v), 6) for v in removed.head(8).tolist()],
                    "filtered_min": float(kept[pred_col].min()) if len(kept) else None,
                    "filtered_max": float(kept[pred_col].max()) if len(kept) else None,
                    "filtered_mean": float(kept[pred_col].mean()) if len(kept) else None,
                    "filtered_median": float(kept[pred_col].median()) if len(kept) else None,
                    "filtered_q1": float(kept[pred_col].quantile(0.25)) if len(kept) else None,
                    "filtered_q3": float(kept[pred_col].quantile(0.75)) if len(kept) else None,
                }
            )

    filtered_df = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else df.copy()
    return filtered_df, stats


def _format_cluster_name(cluster: str) -> str:
    return str(cluster)


def _format_label_name(label: int) -> str:
    return LABEL_DISPLAY_MAP.get(int(label), str(label))


def _compute_hue_offsets(n_hues: int, width: float = 0.7) -> List[float]:
    """计算 dodge 后每个 hue 的横向中心位置。"""
    if n_hues <= 1:
        return [0.0]

    step = width / n_hues
    start = -width / 2 + step / 2
    return [start + i * step for i in range(n_hues)]


def _overlay_raw_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    cluster_order: Sequence[str],
    hue_order: Sequence[int],
    cluster_col: str = "student_cluster",
    label_col: str = "true_label",
    pred_col: str = "y_pred",
    width: float = 0.7,
) -> None:
    """叠加原始分数散点，专门用于把真实重叠关系可视化出来。

    说明：
    这一层不是均值线，也不是额外统计，只是把每个样本的原始预测分数
    用极淡的散点画出来。这样即使箱体的四分位区间看起来不重叠，
    也能直接看到真实数据在 good 簇内部是有交叠的。
    """
    rng = np.random.default_rng(POINT_JITTER_SEED)
    offsets = _compute_hue_offsets(len(hue_order), width=width)
    jitter_half_width = width / max(10, len(hue_order) * 10)

    for cluster_idx, cluster_name in enumerate(cluster_order):
        for hue_idx, hue_value in enumerate(hue_order):
            subset = df[(df[cluster_col] == cluster_name) & (df[label_col] == hue_value)]
            if subset.empty:
                continue

            x_center = cluster_idx + offsets[hue_idx]
            jitter = rng.uniform(-jitter_half_width, jitter_half_width, size=len(subset))
            xs = x_center + jitter
            ys = subset[pred_col].to_numpy(dtype=float)

            ax.scatter(
                xs,
                ys,
                s=7,
                color=LABEL_COLOR_MAP[int(hue_value)],
                alpha=0.08,
                linewidths=0,
                edgecolors="none",
                zorder=1,
            )


def print_filter_statistics(stats: List[Dict]) -> None:
    """打印每个簇+标签的异常值统计和过滤后的分数概况。"""
    print("\n" + "=" * 96)
    print("IQR 异常值过滤统计")
    print("=" * 96)

    for item in stats:
        cluster_name = _format_cluster_name(item["cluster"])
        label_name = _format_label_name(item["label"])
        print(f"\n[{cluster_name}] x [{label_name}]")
        print(f"  原始样本数量: {item['original_count']}")
        print(f"  剔除异常值数量: {item['removed_count']}")
        print(
            f"  IQR 边界: lower={item['lower_bound']:.6f}, upper={item['upper_bound']:.6f}"
        )
        if item["removed_count"] > 0:
            print(
                f"  被剔除值范围: min={item['removed_min']:.6f}, max={item['removed_max']:.6f}"
            )
            print(f"  被剔除值示例: {item['removed_examples']}")
        else:
            print("  被剔除值范围: 无")
            print("  被剔除值示例: []")
        print(f"  剔除后剩余数量: {item['remaining_count']}")
        print(
            "  过滤后分数统计: "
            f"min={item['filtered_min']:.6f}, max={item['filtered_max']:.6f}, "
            f"mean={item['filtered_mean']:.6f}, median={item['filtered_median']:.6f}, "
            f"Q1={item['filtered_q1']:.6f}, Q3={item['filtered_q3']:.6f}"
        )

    print("\n" + "=" * 96)


def plot_boxplot(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    cluster_order: Sequence[str],
    hue_order: Sequence[int],
    title: str = "",
) -> None:
    """绘制和参考图尽量一致风格的箱型图。"""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    palette = [LABEL_COLOR_MAP[int(label)] for label in hue_order]

    # 先画极淡的原始点层，帮助读者直接看到 good 簇内正负样本的真实交叠。
    # 这样不会改变箱体统计，只是把“分布重叠”更直观地表现出来。
    _overlay_raw_points(ax, df, cluster_order, hue_order)

    sns.boxplot(
        data=df,
        x="student_cluster",
        y="y_pred",
        hue="true_label",
        order=list(cluster_order),
        hue_order=list(hue_order),
        palette=palette,
        width=0.7,
        dodge=True,
        linewidth=1.1,
        fliersize=3.0,
        whis=1.5,
        ax=ax,
    )

    # 让箱体颜色更饱满，同时保留边框清晰度。
    for patch in ax.patches:
        patch.set_alpha(0.9)

    ax.set_title(title or dataset_name.upper(), fontsize=20, pad=14)
    ax.set_xlabel("Student Cluster", fontsize=14)
    ax.set_ylabel("Predicted Probability (y_pred)", fontsize=14)
    ax.set_ylim(-0.03, 1.05)
    ax.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--", alpha=0.7, zorder=0)
    ax.set_xticks(np.arange(len(cluster_order)))
    ax.set_xticklabels([_format_cluster_name(c) for c in cluster_order], fontsize=11)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    handles, _ = ax.get_legend_handles_labels()
    labels = [_format_label_name(label) for label in hue_order]
    legend = ax.legend(
        handles,
        labels,
        title="True Label (Ground Truth)",
        loc="center left",
        bbox_to_anchor=(0.01, 0.5),
        frameon=True,
        borderaxespad=0.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.8")

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot prediction score distributions by student cluster after IQR filtering."
    )
    parser.add_argument("--input-csv", type=str, default="", help="已整理好的明细 CSV，需包含簇、标签、预测分数列。")
    parser.add_argument("--pred-csv", type=str, default="", help="由 trans.py 展开的预测长表 CSV。")
    parser.add_argument("--pred-txt", type=str, default="", help="pykt 预测 txt 文件。")
    parser.add_argument("--cluster-csv", type=str, default="", help="学生簇 CSV。")
    parser.add_argument("--sequence-csv", type=str, default="", help="测试序列 CSV，用于将 pred_txt 展开到交互级。")
    parser.add_argument("--save-dir", type=str, default="", help="训练保存目录，用于推断数据集名称和默认路径。")
    parser.add_argument("--dataset-name", type=str, default="", help="数据集名称，若不传则尝试从 save_dir 推断。")
    parser.add_argument("--tag", type=str, default="", help="run 目录名；不传时默认使用 save_dir 的目录名。")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录，默认 DecomposeAUC/auc_figures。")
    parser.add_argument("--output-name", type=str, default="prediction_score_boxplot.png", help="输出图片文件名。")
    parser.add_argument("--title", type=str, default="", help="图标题；不传则使用数据集名称。")

    parser.add_argument("--student-cluster-col", type=str, default="student_cluster", help="输入 CSV 中的簇列名。")
    parser.add_argument("--true-label-col", type=str, default="true_label", help="输入 CSV 中的真实标签列名。")
    parser.add_argument("--pred-col", type=str, default="y_pred", help="输入 CSV 中的预测分数列名。")
    parser.add_argument("--cluster-student-id-col", type=str, default="student_id", help="cluster_csv 中的学生 ID 列名。")
    parser.add_argument("--cluster-label-col", type=str, default="cluster", help="cluster_csv 中的簇标签列名。")
    parser.add_argument("--sequence-uid-col", type=str, default="uid", help="sequence_csv 中的 uid 列名。")
    parser.add_argument(
        "--cluster-order",
        type=str,
        default=",".join(DEFAULT_CLUSTER_ORDER),
        help="簇顺序，逗号分隔，默认 bad,mid_stable,mid_chaotic,good",
    )
    parser.add_argument(
        "--label-order",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LABEL_ORDER),
        help="标签顺序，逗号分隔，默认 1,0",
    )
    return parser


def _resolve_inputs(args: argparse.Namespace, dataset_name: str, output_root: str, run_tag: str) -> pd.DataFrame:
    if args.input_csv:
        return load_input_dataframe(
            input_csv=_normalize_path(args.input_csv),
            cluster_col=args.student_cluster_col,
            label_col=args.true_label_col,
            pred_col=args.pred_col,
        )

    if not args.pred_txt and not args.pred_csv:
        raise ValueError("请至少提供 --input-csv 或 --pred-txt 或 --pred-csv。")

    save_dir = _normalize_path(args.save_dir)
    cluster_csv = _normalize_path(args.cluster_csv) if args.cluster_csv else _infer_cluster_csv(output_root, run_tag)
    if not cluster_csv:
        raise ValueError("无法推断 cluster_csv，请显式提供 --cluster-csv。")

    if args.pred_csv:
        return load_prediction_dataframe_from_csv(
            pred_csv=_normalize_path(args.pred_csv),
            cluster_csv=cluster_csv,
            cluster_student_col=args.cluster_student_id_col,
            cluster_label_col=args.cluster_label_col,
        )

    pred_txt = _normalize_path(args.pred_txt)
    sequence_csv = _normalize_path(args.sequence_csv) if args.sequence_csv else _infer_sequence_csv(save_dir, dataset_name)
    if not sequence_csv:
        raise ValueError("无法推断 sequence_csv，请显式提供 --sequence-csv。")

    return load_prediction_dataframe_from_files(
        pred_txt=pred_txt,
        cluster_csv=cluster_csv,
        sequence_csv=sequence_csv,
        sequence_uid_col=args.sequence_uid_col,
        cluster_student_col=args.cluster_student_id_col,
        cluster_label_col=args.cluster_label_col,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()

    dataset_name = _infer_dataset_name(args.save_dir, args.dataset_name)
    run_tag = _infer_run_tag(args.save_dir, args.dataset_name, args.tag)
    output_root = _normalize_path(args.output_root)
    output_dir = os.path.join(output_root, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    df = _resolve_inputs(args, dataset_name, output_root, run_tag)

    cluster_order = [s.strip() for s in args.cluster_order.split(",") if s.strip()]
    label_order = [int(x.strip()) for x in args.label_order.split(",") if x.strip()]

    # 按 IQR 规则提前过滤异常值，确保箱型图展示的是“清洗后”的结果。
    # 自查结论：上一版把 IQR 过滤后的子集直接用于画图，
    # 会让图中的分布口径和 AUC 分解矩阵（基于完整测试集）不一致。
    # 这里保留 IQR 统计用于诊断，但绘图时明确使用原始预测分数，
    # 这样 boxplot 的分位数、箱体和矩阵都基于同一份数据。
    filtered_df, stats = apply_iqr_filter(
        df,
        cluster_col="student_cluster",
        label_col="true_label",
        pred_col="y_pred",
        cluster_order=cluster_order,
        label_order=label_order,
    )
    print_filter_statistics(stats)

    output_path = os.path.join(output_dir, args.output_name)

    plot_title = args.title.strip() if args.title.strip() else dataset_name.upper()
    plot_boxplot(
        df=df,
        dataset_name=dataset_name,
        output_path=output_path,
        cluster_order=cluster_order,
        hue_order=label_order,
        title=plot_title,
    )

    print("\n绘图完成，已保存图片")
    print(f"保存路径: {output_path}")
    print(f"数据集名称: {dataset_name}")
    print(f"原始总样本数: {len(df)}")
    print(f"IQR 诊断后样本数（仅统计，不用于绘图）: {len(filtered_df)}")
    print("关键统计信息已在上方完整打印。")


if __name__ == "__main__":
    main()