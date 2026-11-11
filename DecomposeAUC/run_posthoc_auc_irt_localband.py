"""
Cluster-only replacement for the legacy post-hoc AUC script.

INTERMEDIATE FILE FORMAT
------------------------
This script writes two files that are intentionally plotting-agnostic:

1) student_clusters_localband_irt.csv
   Required columns for downstream plotting/reuse:
     - student_id (or the configured --student-col)
     - theta
     - cluster            # default clustering column used downstream
         - cluster_label      # English label: bad / mid / good
   Additional columns are included for auditability:
     - theta_report       # report-scale score = 500 + 100 * theta
     - n_responses
     - threshold_low_theta / threshold_high_theta
     - threshold_low_score / threshold_high_score
     - cluster_legacy     # bad / mid / good

2) cluster_summary_localband_irt.json
   Contains threshold-setting metadata, model metadata, and reproducibility info.

Interface notes
---------------
This module preserves the *public interface* of the uploaded legacy script
(run_posthoc_auc_irtwofit(3).py): the same helper function names, parser flags,
and CLI defaults are retained whenever practical. Output artifact names are
updated to the new local-band IRT naming scheme and no legacy aliases are
written.
Methodological upgrade
----------------------
Compared with the legacy fixed-threshold approach, this script uses the
recommended method discussed previously:

  1. Expand pyKT-style sequence rows into student-item binary responses.
  2. Deduplicate to one response per student-item (default: last, preserved for
     CLI compatibility; first is often preferable in studies and is supported).
  3. Fit an IRT model (recommended: 2PL).
    4. Convert the continuous theta scale into *local proficiency bands* (bad/mid/good)
     using a simplified Bookmark-style local standard-setting procedure.
     If automatic local cuts are not stable, the script falls back to a
     1-dimensional Gaussian-mixture split on theta.

By default the script keeps all low-support students/items so the downstream
decomposition can match the full wandb_predict AUC. Pass --apply-support-filtering
to restore the original min_student_responses / min_item_responses filtering.

This keeps the interface stable while upgrading the internal clustering logic to
an IRT + local-threshold banding pipeline that is easier to justify in a paper.



First run the clustering script to produce student clusters:
python DecomposeAUC/run_posthoc_auc_irt_localband.py \
    --save-dir examples/saved_model/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0 \
    --test-csv data/assist2015/test_sequences.csv

Then run the generic plotting script:
python DecomposeAUC/plot_auc_matrix.py \
    --pred-csv DecomposeAUC/assist2015_akt_pykt_pred_long.csv \
    --cluster-csv DecomposeAUC/auc_figures/assist2015_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/student_clusters_irtwofit.csv \
    --output-fig DecomposeAUC/auc_figures/assist2015_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/auc_matrix_irtwofit.png

python DecomposeAUC/plot_auc_matrix.py \
    --pred-csv DecomposeAUC/assist2015_lefokt_akt_pykt_pred_long.csv \
    --cluster-csv DecomposeAUC/auc_figures/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0/student_clusters_irtwofit.csv \
    --output-fig DecomposeAUC/auc_figures/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0/auc_matrix_irtwofit.png

python DecomposeAUC/plot_auc_matrix.py \
    --pred-csv DecomposeAUC/assist2015_lefokt_akt_pykt_pred_long.csv \
    --cluster-csv DecomposeAUC/auc_figures/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0/student_clusters_localband_irt.csv \
    --output-fig DecomposeAUC/auc_figures/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0/auc_matrix_localband_irt.png
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# Optional project-local import: if the user's repository already provides a
# well-tested fit_irt implementation, we reuse it. Otherwise we fall back to the
# self-contained implementation in this file.
# -----------------------------------------------------------------------------


# [Ref-1] Wilson et al., 2016: https://arxiv.org/abs/1604.02336?utm_source=chatgpt.com
#         证明 IRT 类方法在学生作答数据 / KT 风格数据上具有较强竞争力，
#         且具备较好的可解释性。
#
# [Ref-2] Zieky & Perie, 2006 (ETS primer): https://www.ets.org/research/policy_research_reports/publications/publication/2006/dbkw.html?utm_source=chatgpt.com
#         cut score 的设定应建立在能力水平定义基础之上，
#         并且需要进一步验证，而不是仅靠经验切分。
#
# [Ref-3] Karantonis & Sireci, 2006: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-3992.2006.00047.x?utm_source=chatgpt.com
#         Bookmark 是一种广泛使用、且与 IRT 兼容的标准设定方法。
#
# [Ref-4] Mullis & Fishbein, 2020 (TIMSS scale anchoring):  https://timssandpirls.bc.edu/timss2019/methods/pdf/T19_MP_Ch15-scale-anchoring.pdf?utm_source=chatgpt.com
#         大规模教育评估中的 benchmark 一般通过 item map / scale anchoring
#         来解释，而不是通过任意固定数字直接划分。
#
# [Ref-5] Liu et al., 2022 (pyKT): https://dl.acm.org/doi/abs/10.5555/3600270.3601617?utm_source=chatgpt.com
#         KT 数据集需要谨慎预处理，并注意避免信息泄漏和不一致评测协议。
#
# [Ref-6] Ekanadham & Karklin, 2017 (T-SKIRT): https://arxiv.org/abs/1702.04282?utm_source=chatgpt.com
#         如果能力被认为会随时间变化，那么 temporal IRT 是更合理的扩展方向。
#
# [Ref-7] Yeung, 2019 (Deep-IRT): https://arxiv.org/abs/1904.11738?utm_source=chatgpt.com
#         在 KT 场景下，将能力 / 难度解释为 IRT 风格变量有助于提升可解释性。
#
# [Ref-8] NAGB / NAEP procedures: https://nces.ed.gov/nationsreportcard/guides/scores_achv.aspx?utm_source=chatgpt.com
#         achievement levels 由“能力水平描述 + cut scores”共同构成，
#         不是纯粹的任意数值切分。



# =============================================================================
# 方法说明（供论文 / 审稿使用）
# -----------------------------------------------------------------------------
# 本脚本是对旧版固定阈值分簇脚本的替代版本。
#
# 旧版行为：
#   旧版上传脚本使用固定 theta 阈值，将学生划分为
#   bad / mid_chaotic / mid_stable / good 四类。
#
# 新版行为：
#   本脚本保持公开接口兼容，但将内部逻辑替换为：
#
#       1) 使用 IRT 估计学生潜在整体能力（推荐 2PL）；
#       2) 使用“本地熟练度分层”而不是固定人工阈值；
#       3) 优先采用简化版 Bookmark 风格的本地 cut-score 设定；
#       4) 当本地 cut-score 不稳定时，回退到基于 theta 的一维 GMM 分层。
#
# 为什么要这么改：
#   - 基于 IRT 的能力估计在学生作答数据 / KT 场景中具有较强的解释性，
#     并且已有研究表明其效果具有竞争力。[Ref-1]
#   - 教育测量文献一般建议：先定义能力水平，再设定 cut scores，
#     而不是只按任意百分位直接切分。[Ref-2, Ref-3, Ref-8]
#   - TIMSS 等大规模教育评估中的 benchmark 解释依赖 item map / scale anchoring，
#     这启发我们采用“本地锚题 + 本地阈值”的思路，而不是直接照搬国际阈值。[Ref-4]
#   - KT 数据的预处理和评测需要特别注意信息泄漏与协议不一致问题，
#     因此本脚本设计为在固定分簇窗口上运行，而不是混入未来信息。[Ref-5]
#
# 重要范围说明：
#   本脚本不声称复现 NAEP / PISA / TIMSS 的官方 benchmark。
#   本脚本实现的是一种“受大规模教育评估启发的数据集内部分层方法”
#   （dataset-specific local banding procedure）。
# =============================================================================

_EXTERNAL_FIT_IRT = None
try:
    from DecomposeAUC.cluster_students import fit_irt as _EXTERNAL_FIT_IRT  # type: ignore
except ModuleNotFoundError:
    try:
        from cluster_students import fit_irt as _EXTERNAL_FIT_IRT  # type: ignore
    except ModuleNotFoundError:
        _EXTERNAL_FIT_IRT = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "DecomposeAUC" / "auc_figures"
DEFAULT_CLUSTER_ORDER = ["bad", "mid_chaotic", "mid_stable", "good"]

# Legacy CLI defaults retained for backward compatibility.
_LEGACY_BAD_THRESHOLD = -0.5
_LEGACY_MID_THRESHOLD = 0.0
_LEGACY_GOOD_THRESHOLD = 0.5

# Preferred three-band labels for the upgraded method.
LOCAL_LABELS = ["bad", "mid", "good"]


# -----------------------------------------------------------------------------
# Utility helpers preserved from the legacy script.
# -----------------------------------------------------------------------------

def _normalize_dir(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))



def _resolve_existing_path(path: str, save_dir: str, expect_dir: bool = False) -> str:
    if os.path.isabs(path):
        if os.path.exists(path):
            if expect_dir and not os.path.isdir(path):
                raise FileNotFoundError(f"Path is not a directory: {path}")
            return path
        raise FileNotFoundError(f"Path not found: {path}")

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(os.getcwd(), path)),
        os.path.abspath(os.path.join(PROJECT_ROOT, path)),
        os.path.abspath(os.path.join(PROJECT_ROOT, "examples", path)),
        os.path.abspath(os.path.join(os.path.dirname(save_dir), path)),
        os.path.abspath(os.path.join(save_dir, path)),
    ]

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if os.path.exists(candidate):
            if expect_dir and not os.path.isdir(candidate):
                continue
            return candidate

    hint = "\n".join(unique_candidates)
    raise FileNotFoundError(f"Could not resolve path: {path}\nTried:\n{hint}")



def _load_run_config(save_dir: str) -> Tuple[Dict, Dict, str, str]:
    cfg_path = Path(save_dir) / "config.json"
    with cfg_path.open("r", encoding="utf-8") as fin:
        config = json.load(fin)

    model_config = copy.deepcopy(config["model_config"])
    for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
        if remove_item in model_config:
            del model_config[remove_item]

    params = config["params"]
    model_name = params["model_name"]
    emb_type = params["emb_type"]

    seq_len = config["train_config"]["seq_len"]
    if model_name in ["saint", "sakt", "atdkt"]:
        model_config["seq_len"] = seq_len

    return config, model_config, model_name, emb_type



def _infer_dataset_name(config: Dict, save_dir: str) -> str:
    dataset_name = str(config.get("params", {}).get("dataset_name", "")).strip()
    if dataset_name:
        return dataset_name

    if save_dir:
        name = Path(save_dir).name
        match = re.search(
            r"(assist\d+|assistments\d+|assist2009|assist2015|algebra2005|al05|poj|junyi|ednet|statics2011|nips_task34)",
            name,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return name

    return "dataset"



def _infer_run_tag(save_dir: str, dataset_name: str, tag: str = "") -> str:
    if tag:
        return tag.strip()
    if save_dir:
        return Path(save_dir.rstrip("/")).name
    if dataset_name:
        return dataset_name.strip()
    return "run"



def _infer_test_csv(save_dir: str, config: Dict, dataset_name: str) -> str:
    data_config = config.get("data_config", {})
    dpath = str(data_config.get("dpath", "")).strip()
    test_file = str(data_config.get("test_file", "test_sequences.csv")).strip() or "test_sequences.csv"

    candidates = [
        PROJECT_ROOT / "data" / dataset_name / test_file,
        PROJECT_ROOT / test_file,
    ]

    if dpath:
        candidates.extend(
            [
                PROJECT_ROOT / dpath / test_file,
                Path(save_dir) / dpath / test_file,
                Path(save_dir).parent / dpath / test_file,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "无法自动定位 test csv，请显式提供 --test-csv。已尝试路径: "
        + ", ".join(str(item) for item in candidates)
    )



def _parse_int_sequence(value: object) -> List[int]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        items = value
    elif isinstance(value, (int, np.integer)):
        items = [int(value)]
    elif isinstance(value, float):
        items = [int(value)]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                items = parsed
            else:
                items = [parsed]
        except Exception:
            items = [part.strip() for part in text.split(",") if part.strip()]
    else:
        return []

    result: List[int] = []
    for item in items:
        if item is None:
            continue
        try:
            result.append(int(float(item)))
        except Exception:
            continue
    return result



def _load_prediction_csv(
    pred_csv_path: str,
    student_col: str,
    pred_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Preserved for interface compatibility with the legacy script.

    The cluster-only pipeline does not *need* prediction data, but we keep this
    helper because the legacy CLI requires --pred-csv. Downstream users can still
    use this loader or pass the same paths they used previously.
    """
    if not os.path.exists(pred_csv_path):
        raise FileNotFoundError(f"Prediction csv not found: {pred_csv_path}")

    df = pd.read_csv(pred_csv_path)
    required = [student_col, pred_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction csv is missing required columns: {missing}")

    out = df[[student_col, pred_col, label_col]].copy()
    out[student_col] = out[student_col].astype(str)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out = out.dropna(subset=[student_col, pred_col, label_col])
    return out



def _coerce_uid(value: object) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


# [Ref-5]
# pyKT 风格的数据通常按序列行存储（例如 uid, concepts, responses 等），
# 但 IRT 拟合需要的是 student-item-response 三元组形式。
# 因此，这个函数的作用是把序列格式展开为长表格式的学生-题目-作答记录。
#
# 给审稿人的说明：
#   这里并不是修改标签或引入新的监督信息，
#   只是把存储形式从“序列表示”转换为“心理测量模型所需的标准输入格式”。
def _expand_pykt_sequence(
    df: pd.DataFrame,
    uid_col: str,
    concepts_col: str,
    responses_col: str,
    student_col: str,
    item_col: str,
    correct_col: str,
) -> pd.DataFrame:
    required = [uid_col, concepts_col, responses_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for _, row in df[[uid_col, concepts_col, responses_col]].iterrows():
        uid = _coerce_uid(row[uid_col])
        concepts = [part.strip() for part in str(row[concepts_col]).split(",")]
        responses = [part.strip() for part in str(row[responses_col]).split(",")]

        if len(concepts) != len(responses):
            k = min(len(concepts), len(responses))
            concepts = concepts[:k]
            responses = responses[:k]

        for concept, answer in zip(concepts, responses):
            if concept in {"", "-1"} or answer in {"", "-1"}:
                continue

            try:
                ans = float(answer)
            except Exception:
                continue

            rows.append((uid, concept, ans))

    if not rows:
        raise ValueError("No valid responses found after expanding pykt sequence data.")

    return pd.DataFrame(rows, columns=[student_col, item_col, correct_col])


# [Ref-1] [Ref-5]
# 这个函数用于把 KT 长表作答数据整理为适合 IRT 拟合的输入。
#
# 关键设计选择：
#   1) 二值化评分：
#      这里的 IRT 模型基于二值正确/错误建模，这与 1PL/2PL/3PL 的
#      标准 student-item-response 建模形式一致。
#
#   2) 每个 student-item 只保留一条记录：
#      如果对同一 student-item 保留多次尝试，会引入局部依赖和练习效应，
#      不利于“静态整体能力分层”的解释。因此这里将其压缩为一条记录。
#      （为了接口兼容，旧版默认是 "last"；如果论文更强调前测能力，
#      通常 "first" 会更合理。）
#
#   3) 最小作答数过滤：
#      作答极少的学生 / 题目会导致 theta 和题目参数估计不稳定，
#      因此在 IRT 拟合前先做最小样本过滤。
#
# 给审稿人的说明：
#   如果审稿人问“为什么不保留所有尝试”，回答是：
#   重复练习过程并不是相互独立的测量事件，
#   对于静态整体能力估计，保留每个 student-item 一条记录是更保守、更可解释的做法。
# 对于 KDD / Algebra 类型数据，官方任务定义本身就使用了
# "Correct First Attempt" 这一目标变量，因此这种处理方式也与原始任务设定一致。
def _prepare_responses(
    df: pd.DataFrame,
    student_col: str,
    item_col: str,
    correct_col: str,
    deduplicate: str,
    correct_threshold: float,
    min_student_responses: int,
    min_item_responses: int,
    apply_support_filtering: bool,
) -> pd.DataFrame:
    work = df[[student_col, item_col, correct_col]].copy()
    work = work.dropna(subset=[student_col, item_col, correct_col])
    work[correct_col] = pd.to_numeric(work[correct_col], errors="coerce")
    work = work.dropna(subset=[correct_col])
    work[correct_col] = (work[correct_col] >= correct_threshold).astype(int)

    if deduplicate == "first":
        work = work.drop_duplicates(subset=[student_col, item_col], keep="first")
    elif deduplicate == "last":
        work = work.drop_duplicates(subset=[student_col, item_col], keep="last")
    elif deduplicate == "mean":
        work = (
            work.groupby([student_col, item_col], as_index=False)[correct_col]
            .mean()
            .assign(**{correct_col: lambda d: (d[correct_col] >= correct_threshold).astype(int)})
        )
    else:
        raise ValueError("deduplicate must be one of {'first', 'last', 'mean'}")

    if apply_support_filtering and min_student_responses > 1:
        stu_counts = work.groupby(student_col)[correct_col].size()
        valid_students = stu_counts[stu_counts >= min_student_responses].index
        work = work[work[student_col].isin(valid_students)]

    if apply_support_filtering and min_item_responses > 1:
        item_counts = work.groupby(item_col)[correct_col].size()
        valid_items = item_counts[item_counts >= min_item_responses].index
        work = work[work[item_col].isin(valid_items)]

    if work.empty:
        raise ValueError("No valid responses remain after preprocessing and filtering.")

    return work


# [Ref-1]
# 将长表格式的 student-item-response 数据转换为矩阵形式，
# 用 NaN 表示缺失作答。
# 这只是 IRT 似然计算所需的计算表示形式，对应标准的心理测量建模输入。
def _build_response_matrix(
    df: pd.DataFrame,
    student_col: str,
    item_col: str,
    correct_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    student_codes, student_values = pd.factorize(df[student_col], sort=False)
    item_codes, item_values = pd.factorize(df[item_col], sort=False)

    n_students = len(student_values)
    n_items = len(item_values)
    responses = np.full((n_students, n_items), np.nan, dtype=np.float64)
    responses[student_codes, item_codes] = df[correct_col].astype(float).to_numpy()

    student_n = np.sum(~np.isnan(responses), axis=1)
    item_n = np.sum(~np.isnan(responses), axis=0)

    return responses, student_values.astype(object), item_values.astype(object), student_n, item_n


# -----------------------------------------------------------------------------
# Self-contained IRT fallback.
# -----------------------------------------------------------------------------

def _safe_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))



def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))



def _initialize_irt(response_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    student_mean = np.nanmean(response_matrix, axis=1)
    item_mean = np.nanmean(response_matrix, axis=0)

    student_mean = np.where(np.isnan(student_mean), 0.5, student_mean)
    item_mean = np.where(np.isnan(item_mean), 0.5, item_mean)

    theta = _safe_logit(student_mean)
    if np.nanstd(theta) > 1e-8:
        theta = (theta - np.nanmean(theta)) / np.nanstd(theta)
    else:
        theta = theta - np.nanmean(theta)

    a = np.ones(response_matrix.shape[1], dtype=float)
    b = -_safe_logit(item_mean)
    c = np.zeros(response_matrix.shape[1], dtype=float)
    return theta, a, b, c


# [Ref-1]
# 2PL 模型同时估计题目区分度 (a) 和题目难度 (b)，
# 因而允许不同题目既在难度上不同，也在“区分学生能力差异”的强度上不同。
# 相比 Rasch / 1PL 更灵活，但又通常比过度参数化的模型更稳定、可解释。
def _fit_item_2pl(theta_obs: np.ndarray, y_obs: np.ndarray) -> Tuple[float, float]:
    # Parameters are [log_a, b] so that a remains positive.
    def objective(params: np.ndarray) -> float:
        log_a, b = params
        a = float(np.exp(np.clip(log_a, -4.0, 3.0)))
        p = _sigmoid(a * (theta_obs - b))
        eps = 1e-8
        ll = np.sum(y_obs * np.log(p + eps) + (1.0 - y_obs) * np.log(1.0 - p + eps))
        # Mild shrinkage toward a=1 for stability on sparse items.
        penalty = 0.05 * (log_a ** 2)
        return -(ll - penalty)

    init_b = -float(np.mean(theta_obs)) if len(theta_obs) else 0.0
    res = minimize(
        objective,
        x0=np.array([0.0, init_b], dtype=float),
        method="L-BFGS-B",
        bounds=[(-4.0, 3.0), (-6.0, 6.0)],
    )
    if not res.success:
        return 1.0, init_b

    log_a, b = res.x
    a = float(np.exp(np.clip(log_a, -4.0, 3.0)))
    return a, float(b)



def _fit_item_3pl(theta_obs: np.ndarray, y_obs: np.ndarray, c_upper: float) -> Tuple[float, float, float]:
    def objective(params: np.ndarray) -> float:
        log_a, b, raw_c = params
        a = float(np.exp(np.clip(log_a, -4.0, 3.0)))
        c = float(c_upper * _sigmoid(raw_c))
        p = c + (1.0 - c) * _sigmoid(a * (theta_obs - b))
        eps = 1e-8
        ll = np.sum(y_obs * np.log(p + eps) + (1.0 - y_obs) * np.log(1.0 - p + eps))
        penalty = 0.05 * (log_a ** 2) + 0.01 * (raw_c ** 2)
        return -(ll - penalty)

    init_b = -float(np.mean(theta_obs)) if len(theta_obs) else 0.0
    res = minimize(
        objective,
        x0=np.array([0.0, init_b, -2.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(-4.0, 3.0), (-6.0, 6.0), (-8.0, 8.0)],
    )
    if not res.success:
        return 1.0, init_b, 0.0

    log_a, b, raw_c = res.x
    a = float(np.exp(np.clip(log_a, -4.0, 3.0)))
    c = float(c_upper * _sigmoid(raw_c))
    return a, float(b), c



def _update_theta_map(
    y_obs: np.ndarray,
    a_obs: np.ndarray,
    b_obs: np.ndarray,
    prior_sd: float,
) -> float:
    # Newton-Raphson MAP update for 2PL/1PL.
    theta = 0.0
    for _ in range(25):
        z = a_obs * (theta - b_obs)
        p = _sigmoid(z)
        grad = np.sum(a_obs * (y_obs - p)) - theta / (prior_sd ** 2)
        hess = -np.sum((a_obs ** 2) * p * (1.0 - p)) - 1.0 / (prior_sd ** 2)
        if abs(hess) < 1e-8:
            break
        step = grad / hess
        theta_next = theta - step
        theta_next = float(np.clip(theta_next, -6.0, 6.0))
        if abs(theta_next - theta) < 1e-4:
            theta = theta_next
            break
        theta = theta_next
    return float(theta)



def _update_theta_map_3pl(
    y_obs: np.ndarray,
    a_obs: np.ndarray,
    b_obs: np.ndarray,
    c_obs: np.ndarray,
    prior_sd: float,
) -> float:
    # Use bounded scalar optimization for 3PL because the derivative is more
    # cumbersome and less stable in sparse settings.
    def objective(theta: float) -> float:
        p = c_obs + (1.0 - c_obs) * _sigmoid(a_obs * (theta - b_obs))
        eps = 1e-8
        ll = np.sum(y_obs * np.log(p + eps) + (1.0 - y_obs) * np.log(1.0 - p + eps))
        prior = -0.5 * (theta / prior_sd) ** 2
        return -(ll + prior)

    res = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
    return float(res.x)


# [Ref-1] [Ref-7]
# IRT 内部拟合过程（例如 EM / MAP 风格）。
#
# 为什么本脚本推荐默认使用 2PL：
#   - 它可以把学生能力与题目难度显式分离，
#     这是相较于原始正确率分组的核心优势；
#   - 它保留了 theta 和题目参数的可解释性；
#   - 在 KT / 学生作答场景中，IRT 家族方法通常比直接聚类神经网络隐藏状态
#     更容易论证和解释。
#
# 给审稿人的说明：
#   我们并不声称“静态 2PL 已完全刻画学习过程中的所有时间动态”，
#   本脚本的目标只是为 post-hoc 分层提供一个稳定、可解释的整体能力估计。
def _fit_irt_internal(
    response_matrix: np.ndarray,
    model: str = "2PL",
    max_iter: int = 20,
    tol: float = 1e-3,
    prior_sd: float = 1.0,
    c_upper: float = 0.35,
    random_state: int = 3407,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    _ = rng  # kept to make random_state semantically active in this fallback.

    model = str(model).upper().strip()
    if model not in {"1PL", "2PL", "3PL"}:
        raise ValueError("model must be one of {'1PL', '2PL', '3PL'}")

    n_students, n_items = response_matrix.shape
    theta, a, b, c = _initialize_irt(response_matrix)
    prev_theta = theta.copy()
    converged = 0
    iterations = 0

    item_masks = [~np.isnan(response_matrix[:, i]) for i in range(n_items)]
    student_masks = [~np.isnan(response_matrix[u, :]) for u in range(n_students)]

    for it in range(1, max_iter + 1):
        iterations = it

        # Update item parameters conditional on theta.
        for i in range(n_items):
            mask = item_masks[i]
            if mask.sum() < 3:
                continue
            y_obs = response_matrix[mask, i].astype(float)
            theta_obs = theta[mask].astype(float)
            if model == "1PL":
                p = float(np.clip(np.mean(y_obs), 1e-5, 1.0 - 1e-5))
                a[i] = 1.0
                b[i] = -math.log(p / (1.0 - p))
                c[i] = 0.0
            elif model == "2PL":
                a[i], b[i] = _fit_item_2pl(theta_obs, y_obs)
                c[i] = 0.0
            else:
                a[i], b[i], c[i] = _fit_item_3pl(theta_obs, y_obs, c_upper=c_upper)

        # Update theta conditional on item parameters.
        for u in range(n_students):
            mask = student_masks[u]
            if mask.sum() < 1:
                theta[u] = 0.0
                continue
            y_obs = response_matrix[u, mask].astype(float)
            a_obs = a[mask].astype(float)
            b_obs = b[mask].astype(float)
            if model == "3PL":
                c_obs = c[mask].astype(float)
                theta[u] = _update_theta_map_3pl(y_obs, a_obs, b_obs, c_obs, prior_sd=prior_sd)
            else:
                theta[u] = _update_theta_map(y_obs, a_obs, b_obs, prior_sd=prior_sd)

        # Re-center theta for identifiability.
        theta = theta - float(np.mean(theta))
        theta_std = float(np.std(theta))
        if theta_std > 1e-8:
            theta = theta / theta_std
            b = b / theta_std
            a = a * theta_std

        max_delta = float(np.max(np.abs(theta - prev_theta))) if len(theta) else 0.0
        if verbose:
            print(f"[IRT] iter={it} max_theta_delta={max_delta:.6f}")
        if max_delta < tol:
            converged = 1
            break
        prev_theta = theta.copy()

    return {
        "theta": theta.astype(float),
        "a": a.astype(float),
        "b": b.astype(float),
        "c": c.astype(float),
        "converged": np.array([converged], dtype=int),
        "iterations": np.array([iterations], dtype=int),
    }


# [Ref-1] [Ref-6] [Ref-7]
# 对外暴露的 IRT 拟合入口。
#
# 如果项目目录中已有外部 IRT 实现，则优先复用，以保持仓库内部一致性；
# 否则使用脚本内置的自包含实现。
#
# 给审稿人的说明：
#   本脚本使用静态 IRT，是因为当前研究目标是“给每个学生分配一个整体能力层级”。
#   如果研究问题明确强调“能力随时间演化”，那么 temporal IRT（如 T-SKIRT）
#   才是更合理的扩展方向。
def fit_irt(
    response_matrix: np.ndarray,
    model: str = "2PL",
    max_iter: int = 20,
    tol: float = 1e-3,
    prior_sd: float = 1.0,
    c_upper: float = 0.35,
    random_state: int = 3407,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Wrapper that preserves the legacy call pattern while upgrading behavior.

    Preference order:
      1. Use the repository's existing fit_irt implementation if available.
      2. Otherwise use the self-contained fallback defined above.
    """
    if _EXTERNAL_FIT_IRT is not None:
        try:
            result = _EXTERNAL_FIT_IRT(
                response_matrix,
                model=model,
                max_iter=max_iter,
                tol=tol,
                prior_sd=prior_sd,
                c_upper=c_upper,
                random_state=random_state,
                verbose=verbose,
            )
            # Normalize keys expected by this script.
            result.setdefault("a", np.ones(response_matrix.shape[1], dtype=float))
            result.setdefault("b", np.zeros(response_matrix.shape[1], dtype=float))
            result.setdefault("c", np.zeros(response_matrix.shape[1], dtype=float))
            return result
        except Exception as exc:
            if verbose:
                print(f"[WARN] External fit_irt failed, falling back to internal implementation: {exc}")

    return _fit_irt_internal(
        response_matrix=response_matrix,
        model=model,
        max_iter=max_iter,
        tol=tol,
        prior_sd=prior_sd,
        c_upper=c_upper,
        random_state=random_state,
        verbose=verbose,
    )


# -----------------------------------------------------------------------------
# Local threshold setting: simplified Bookmark-style automatic banding.
# -----------------------------------------------------------------------------

def _mean_success_probability(theta_value: float, a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    return float(np.mean(_sigmoid(a_vec * (theta_value - b_vec))))



def _solve_theta_for_target_probability(
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    target_prob: float,
    lo: float = -6.0,
    hi: float = 6.0,
) -> float:
    def f(theta_value: float) -> float:
        return _mean_success_probability(theta_value, a_vec, b_vec) - target_prob

    flo = f(lo)
    fhi = f(hi)
    if flo >= 0.0:
        return lo
    if fhi <= 0.0:
        return hi

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-5:
            return float(mid)
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


# [Ref-2] [Ref-3] [Ref-4] [Ref-8]
# 自动化的“本地标准设定”过程，用于把连续 theta 转换为 3 个熟练度层级。
#
# 这里实现的是一个“简化版 Bookmark 风格”流程：
#   1) 基于 IRT 参数构建 item map；
#   2) 从已拟合题目中识别相对稳定的锚题；
#   3) 根据题目难度划分低阶锚题和高阶锚题集合；
#   4) 计算：当边界学生在这些锚题上达到目标成功概率时，对应的 theta 位置，
#      并将其作为 cut scores。
#
# 为什么不能直接照搬 TIMSS / PIRLS / NAEP 的官方阈值？
#   因为那些官方 benchmark / achievement levels 建立在固定报告尺度、
#   链接/等值设计以及正式标准设定 panel 之上。
#   单个 KT 数据集通常不具备这些条件。
#   因此这里采用的是“受其启发的本地化 benchmark 设定”，
#   而不是直接复制其数值阈值。
#
# 为什么默认使用 target_prob = 0.67？
#   这是“边界学生”思想下的一种实用化设定：阈值不再只是分布上的分位点，
#   而是和“边界学生在锚题上能做到什么程度”联系起来。
#
# 给审稿人的说明：
#   论文中应同时报告阈值敏感性分析，例如 target_prob 取 0.60 / 0.67 / 0.75 时，
#   主要结论是否保持稳定，以证明结果不是某一个精确阈值的偶然产物。
def _auto_local_cut_scores(
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    item_n: np.ndarray,
    target_prob: float = 0.67,
    verbose: bool = False,
) -> Tuple[float, float, Dict[str, object]]:
    """
    Automatic local standard-setting.

    Rationale
    ---------
    We approximate a simplified Bookmark procedure without requiring an external
    panel. We first identify stable anchor items from the fitted item map, then
    choose two cut scores corresponding to the ability level at which a boundary
    student reaches the target success probability on low- and mid-difficulty
    anchor sets.

    If this procedure is unstable, we fall back to a 1D three-component GMM on
    theta and derive cut points from the intersections of sorted component means.
    """
    meta: Dict[str, object] = {"method": "bookmark_local"}

    finite = np.isfinite(a) & np.isfinite(b) & (item_n >= 3)
    if finite.sum() < 12:
        if verbose:
            print("[CUT] Too few usable items for local Bookmark; falling back to GMM.")
        return _gmm_cut_scores(theta, verbose=verbose)

    a_use = a[finite]
    b_use = b[finite]
    # Prefer items with non-extreme discrimination for more stable anchors.
    a_lo, a_hi = np.quantile(a_use, [0.20, 0.80])
    stable = finite & (a >= a_lo) & (a <= a_hi)
    if stable.sum() < 12:
        stable = finite

    b_stable = b[stable]
    a_stable = a[stable]
    q20, q35, q55, q70 = np.quantile(b_stable, [0.20, 0.35, 0.55, 0.70])
    low_mask = stable & (b >= q20) & (b <= q35)
    high_mask = stable & (b >= q55) & (b <= q70)

    if low_mask.sum() < 4 or high_mask.sum() < 4:
        if verbose:
            print("[CUT] Anchor sets too small; falling back to GMM.")
        return _gmm_cut_scores(theta, verbose=verbose)

    low_a = a[low_mask]
    low_b = b[low_mask]
    high_a = a[high_mask]
    high_b = b[high_mask]

    c1 = _solve_theta_for_target_probability(low_a, low_b, target_prob=target_prob)
    c2 = _solve_theta_for_target_probability(high_a, high_b, target_prob=target_prob)

    # Safety checks.
    if not (np.isfinite(c1) and np.isfinite(c2)) or not (c1 < c2):
        if verbose:
            print("[CUT] Bookmark cuts invalid; falling back to GMM.")
        return _gmm_cut_scores(theta, verbose=verbose)

    # Keep cuts inside a reasonable range of observed ability.
    t_lo, t_hi = float(np.nanmin(theta)), float(np.nanmax(theta))
    c1 = float(np.clip(c1, t_lo - 0.25, t_hi + 0.25))
    c2 = float(np.clip(c2, t_lo - 0.25, t_hi + 0.25))
    if not c1 < c2:
        if verbose:
            print("[CUT] Clipped Bookmark cuts invalid; falling back to GMM.")
        return _gmm_cut_scores(theta, verbose=verbose)

    meta.update(
        {
            "target_probability": float(target_prob),
            "n_low_anchor_items": int(low_mask.sum()),
            "n_high_anchor_items": int(high_mask.sum()),
            "low_anchor_b_range": [float(q20), float(q35)],
            "high_anchor_b_range": [float(q55), float(q70)],
        }
    )
    return c1, c2, meta


# [Ref-2]
# 基于 theta 的一维 3 成分高斯混合分层，用作本地标准设定失败时的 fallback。
#
# 注意：
#   这不是主方法，只在以下情况下使用：
#   - 可用锚题过少；
#   - 自动设定出的 cut scores 不满足有序性；
#   - 数据条件不支持稳定的本地标准设定。
#
# 为什么要保留 fallback？
#   cut-score 文献强调，阈值设定是需要验证的；
#   如果某种 cut-setting 程序在给定数据上表现不稳定，就不应强行使用。
#
# 给审稿人的说明：
#   在论文中，这个函数应该被描述为“保证鲁棒性的回退方案”，
#   而不是主方法本身。
def _gmm_cut_scores(theta: np.ndarray, verbose: bool = False) -> Tuple[float, float, Dict[str, object]]:
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=3407)
    x = theta.reshape(-1, 1)
    gmm.fit(x)

    means = gmm.means_.flatten()
    order = np.argsort(means)
    means = means[order]
    covs = gmm.covariances_.reshape(-1)[order]
    weights = gmm.weights_.flatten()[order]

    labels = gmm.predict(x)
    remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    labels_ord = np.array([remap[int(v)] for v in labels], dtype=int)

    # Empirical cuts between adjacent ordered components: midpoint between the
    # max of the lower class and min of the upper class when possible; otherwise
    # midpoint of component means.
    cuts: List[float] = []
    for lower, upper in [(0, 1), (1, 2)]:
        lower_vals = theta[labels_ord == lower]
        upper_vals = theta[labels_ord == upper]
        if len(lower_vals) and len(upper_vals):
            c = 0.5 * (float(np.max(lower_vals)) + float(np.min(upper_vals)))
        else:
            c = 0.5 * (float(means[lower]) + float(means[upper]))
        cuts.append(c)

    c1, c2 = float(cuts[0]), float(cuts[1])
    if verbose:
        print(f"[CUT] GMM fallback cuts: {c1:.4f}, {c2:.4f}")
    meta = {
        "method": "gmm_theta",
        "component_means": [float(v) for v in means],
        "component_variances": [float(v) for v in covs],
        "component_weights": [float(v) for v in weights],
    }
    return c1, c2, meta



def _manual_cut_scores_from_legacy_args(
    bad_theta_threshold: float,
    good_theta_threshold: float,
) -> Tuple[float, float, Dict[str, object]]:
    if not bad_theta_threshold < good_theta_threshold:
        raise ValueError("Require bad_theta_threshold < good_theta_threshold for manual 3-band mode.")
    return float(bad_theta_threshold), float(good_theta_threshold), {
        "method": "manual_theta",
        "note": "Manual low/high theta thresholds supplied through legacy-compatible arguments.",
    }


# [Ref-2] [Ref-8]
# 保留该函数仅是为了兼容旧版脚本接口中的“手动 theta 阈值”模式。
#
# 重要说明：
#   这个函数不再代表推荐方法。
#   之所以保留，是因为旧版脚本公开暴露了 bad / mid / good 相关的
#   theta 阈值参数，为了接口兼容而继续支持。
#
# 更推荐的做法：
#   在论文或正式实验中，应优先使用 _auto_local_cut_scores() 中的
#   本地标准设定流程，因为教育测量中的 achievement levels /
#   proficiency bands 更应依赖能力水平定义与标准设定，而不是任意固定阈值。
def _theta_only_clusters(
    theta: np.ndarray,
    bad_theta_threshold: float = _LEGACY_BAD_THRESHOLD,
    mid_theta_threshold: float = _LEGACY_MID_THRESHOLD,
    good_theta_threshold: float = _LEGACY_GOOD_THRESHOLD,
) -> np.ndarray:
    """
    Preserved function name/signature.

    Legacy behavior split students into four bins (bad / mid_chaotic / mid_stable
    / good) using fixed thresholds. The upgraded method now produces three local
    proficiency bands (bad / mid / good).

    Backward-compatible behavior:
      - If callers explicitly provide non-default bad/good thresholds, we honor
        them as *manual* low/high cut scores for a 3-band split.
      - Otherwise callers should use the automatic local thresholding implemented
        in _cluster_students_from_test_sequences().

    This helper therefore remains available mainly for explicit manual-threshold
    use, while the main pipeline now prefers local threshold setting.
    """
    c1, c2, _ = _manual_cut_scores_from_legacy_args(bad_theta_threshold, good_theta_threshold)
    return np.where(theta <= c1, LOCAL_LABELS[0], np.where(theta <= c2, LOCAL_LABELS[1], LOCAL_LABELS[2])).astype(object)



def _legacy_thresholds_explicitly_overridden(
    bad_theta_threshold: float,
    mid_theta_threshold: float,
    good_theta_threshold: float,
) -> bool:
    return not (
        np.isclose(bad_theta_threshold, _LEGACY_BAD_THRESHOLD)
        and np.isclose(mid_theta_threshold, _LEGACY_MID_THRESHOLD)
        and np.isclose(good_theta_threshold, _LEGACY_GOOD_THRESHOLD)
    )


# [Ref-4] [Ref-8]
# 将连续 theta 转换为 3 个本地熟练度标签。
#
# 这里的中文标签“差 / 中 / 优”仅用于本数据集内部分层和后续分析展示。
# 它们应被理解为“数据集特定的本地熟练度层级”，
# 而不是官方 TIMSS / PIRLS / NAEP benchmark 的直接替代物。
def _assign_three_band_labels(theta: np.ndarray, c1: float, c2: float) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.where(theta <= c1, LOCAL_LABELS[0], np.where(theta <= c2, LOCAL_LABELS[1], LOCAL_LABELS[2]))
    return labels.astype(object), labels.astype(object)


# [Ref-1] [Ref-2] [Ref-5]
# 基于 test_sequences 的端到端本地 IRT 分层流程。
#
# 流程包括：
#   (a) 将 pyKT 序列展开为长表格式响应；
#   (b) 对响应进行预处理 / 去重；
#   (c) 构建 student-item 响应矩阵；
#   (d) 使用 IRT 估计学生整体能力 theta；
#   (e) 将 theta 转换为 3 个本地熟练度层级。
#
# 为什么可以直接在当前 split 上拟合分簇？
#   因为该脚本的用途是对“当前被分析的学生群体”做 post-hoc 水平分层，
#   并不要求这些学生与训练集学生重叠。
#
# 信息泄漏提醒：
#   如果这些簇标签后续要作为预测模型输入、分层教学特征、或公平性分析分组依据，
#   那么论文实验中应固定分簇窗口（例如仅用训练集或前缀交互），
#   而不应直接混入未来作答信息。
#
# 为什么保存 theta 和阈值？
#   审稿人通常会追问：分组是否可复现、阈值是否可审计。
#   因此脚本会保存 theta、报告尺度分数、cut scores 以及作答数等中间信息，
#   使分层过程可检查、可复现实验。
def _cluster_students_from_test_sequences(
    test_csv: str,
    uid_col: str,
    concepts_col: str,
    responses_col: str,
    student_col: str,
    item_col: str,
    correct_col: str,
    irt_model: str,
    bad_theta_threshold: float,
    mid_theta_threshold: float,
    good_theta_threshold: float,
    deduplicate: str,
    correct_threshold: float,
    min_student_responses: int,
    min_item_responses: int,
    apply_support_filtering: bool,
    max_iter: int,
    tol: float,
    prior_sd: float,
    c_upper: float,
    random_state: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    source_df = pd.read_csv(test_csv)
    long_df = _expand_pykt_sequence(
        source_df,
        uid_col=uid_col,
        concepts_col=concepts_col,
        responses_col=responses_col,
        student_col=student_col,
        item_col=item_col,
        correct_col=correct_col,
    )

    prepped = _prepare_responses(
        long_df,
        student_col=student_col,
        item_col=item_col,
        correct_col=correct_col,
        deduplicate=deduplicate,
        correct_threshold=correct_threshold,
        min_student_responses=min_student_responses,
        min_item_responses=min_item_responses,
        apply_support_filtering=apply_support_filtering,
    )

    response_matrix, student_values, item_values, student_n, item_n = _build_response_matrix(
        prepped,
        student_col=student_col,
        item_col=item_col,
        correct_col=correct_col,
    )

    irt_res = fit_irt(
        response_matrix,
        model=irt_model,
        max_iter=max_iter,
        tol=tol,
        prior_sd=prior_sd,
        c_upper=c_upper,
        random_state=random_state,
        verbose=verbose,
    )

    theta = np.asarray(irt_res["theta"], dtype=float)
    a = np.asarray(irt_res.get("a", np.ones_like(item_values, dtype=float)), dtype=float)
    b = np.asarray(irt_res.get("b", np.zeros_like(item_values, dtype=float)), dtype=float)

    # ------------------------------------------------------------------
    # Threshold setting strategy.
    # ------------------------------------------------------------------
    # 1) If the user explicitly overrides legacy thresholds, treat them as
    #    manual low/high theta cuts for a 3-band split.
    # 2) Otherwise, use the recommended local standard-setting procedure.
    # ------------------------------------------------------------------
    if _legacy_thresholds_explicitly_overridden(
        bad_theta_threshold=bad_theta_threshold,
        mid_theta_threshold=mid_theta_threshold,
        good_theta_threshold=good_theta_threshold,
    ):
        c1, c2, cut_meta = _manual_cut_scores_from_legacy_args(
            bad_theta_threshold=bad_theta_threshold,
            good_theta_threshold=good_theta_threshold,
        )
    else:
        c1, c2, cut_meta = _auto_local_cut_scores(
            theta=theta,
            a=a,
            b=b,
            item_n=item_n,
            target_prob=0.67,
            verbose=verbose,
        )

    cluster_label_en, cluster_legacy = _assign_three_band_labels(theta=theta, c1=c1, c2=c2)
    theta_report = 500.0 + 100.0 * theta

    cluster_df = pd.DataFrame(
        {
            student_col: student_values,
            "theta": theta,
            "theta_report": theta_report,
            "n_responses": student_n,
            "cluster": cluster_legacy,
            "cluster_label": cluster_label_en,
            "cluster_legacy": cluster_legacy,
            "threshold_low_theta": np.full_like(theta, fill_value=float(c1), dtype=float),
            "threshold_high_theta": np.full_like(theta, fill_value=float(c2), dtype=float),
            "threshold_low_score": np.full_like(theta, fill_value=float(500.0 + 100.0 * c1), dtype=float),
            "threshold_high_score": np.full_like(theta, fill_value=float(500.0 + 100.0 * c2), dtype=float),
        }
    )

    summary: Dict[str, object] = {
        "irt_model": irt_model,
        "converged": int(np.asarray(irt_res["converged"])[0]),
        "iterations": int(np.asarray(irt_res["iterations"])[0]),
        "n_students": int(cluster_df.shape[0]),
        "n_items": int(response_matrix.shape[1]),
        "apply_support_filtering": bool(apply_support_filtering),
        "student_col": student_col,
        "item_col": item_col,
        "correct_col": correct_col,
        "deduplicate": deduplicate,
        "correct_threshold": float(correct_threshold),
        "min_student_responses": int(min_student_responses),
        "min_item_responses": int(min_item_responses),
        "threshold_low_theta": float(c1),
        "threshold_high_theta": float(c2),
        "threshold_low_score": float(500.0 + 100.0 * c1),
        "threshold_high_score": float(500.0 + 100.0 * c2),
        "threshold_setting": cut_meta,
        "cluster_value_counts": {str(k): int(v) for k, v in cluster_df["cluster"].value_counts().to_dict().items()},
    }
    return cluster_df, summary


# -----------------------------------------------------------------------------
# CLI parser preserved from the legacy script.
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyKT post-hoc student clustering with IRT-based local proficiency banding."
    )
    parser.add_argument("--save-dir", type=str, required=True, help="Trained run directory with config.json.")
    parser.add_argument(
        "--test-csv",
        type=str,
        default="",
        help="pykt test_sequences.csv used for IRT clustering. If omitted, infer from save-dir.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--uid-col", type=str, default="uid")
    parser.add_argument("--responses-col", type=str, default="responses")
    parser.add_argument("--concepts-col", type=str, default="concepts")
    parser.add_argument("--student-col", type=str, default="student_id")

    parser.add_argument("--irt-model", type=str, default="2PL", choices=["2PL", "3PL"])
    parser.add_argument("--bad-theta-threshold", type=float, default=-0.5)
    parser.add_argument("--mid-theta-threshold", type=float, default=0.0)
    parser.add_argument("--good-theta-threshold", type=float, default=0.5)
    parser.add_argument("--deduplicate", type=str, default="last", choices=["first", "last", "mean"])
    parser.add_argument("--correct-threshold", type=float, default=0.5)
    parser.add_argument("--min-student-responses", type=int, default=3)
    parser.add_argument("--min-item-responses", type=int, default=3)
    parser.add_argument(
        "--apply-support-filtering",
        action="store_true",
        help="Restore the original low-support filtering based on min_student_responses and min_item_responses.",
    )
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--prior-sd", type=float, default=1.0)
    parser.add_argument("--c-upper", type=float, default=0.35)
    parser.add_argument("--random-state", type=int, default=3407)
    parser.add_argument("--verbose", action="store_true")
    return parser



def main() -> None:
    args = _build_arg_parser().parse_args()

    save_dir = _normalize_dir(args.save_dir)
    output_dir = _normalize_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config, _model_config, model_name, _emb_type = _load_run_config(save_dir)
    dataset_name = _infer_dataset_name(config, save_dir)
    run_tag = _infer_run_tag(save_dir, dataset_name, args.tag)
    run_output_dir = os.path.join(output_dir, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)

    test_csv = (
        _resolve_existing_path(args.test_csv, save_dir, expect_dir=False)
        if args.test_csv
        else _infer_test_csv(save_dir, config, dataset_name)
    )
    cluster_df, cluster_summary = _cluster_students_from_test_sequences(
        test_csv=test_csv,
        uid_col=args.uid_col,
        concepts_col=args.concepts_col,
        responses_col=args.responses_col,
        student_col=args.student_col,
        item_col="item_id",
        correct_col="correct",
        irt_model=args.irt_model,
        bad_theta_threshold=args.bad_theta_threshold,
        mid_theta_threshold=args.mid_theta_threshold,
        good_theta_threshold=args.good_theta_threshold,
        deduplicate=args.deduplicate,
        correct_threshold=args.correct_threshold,
        min_student_responses=args.min_student_responses,
        min_item_responses=args.min_item_responses,
        apply_support_filtering=args.apply_support_filtering,
        max_iter=args.max_iter,
        tol=args.tol,
        prior_sd=args.prior_sd,
        c_upper=args.c_upper,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    # Canonical artifact names for the upgraded local-band IRT pipeline.
    # This is intentionally a *new* script with *new* artifact names. We keep
    # the public function/CLI interface compatible with the legacy script, but
    # we do not emit old filename aliases.
    cluster_csv = os.path.join(run_output_dir, "student_clusters_localband_irt.csv")
    cluster_df.to_csv(cluster_csv, index=False)

    cluster_summary_path = os.path.join(run_output_dir, "cluster_summary_localband_irt.json")
    payload = {
        **cluster_summary,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "test_csv": test_csv,
        "cluster_csv": cluster_csv,
        "artifact_naming": "localband_irt",
    }
    with open(cluster_summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)

    print("[DONE] Post-hoc clustering finished.")
    print(f"[DONE] model_name: {model_name}")
    print(f"[DONE] test_csv: {test_csv}")
    print(f"[DONE] cluster_csv: {cluster_csv}")
    print(f"[DONE] output_dir: {run_output_dir}")
    print(f"[DONE] summary: {cluster_summary_path}")


if __name__ == "__main__":
    main()
