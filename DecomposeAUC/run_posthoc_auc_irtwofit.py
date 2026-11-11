"""
Usage:

First run the clustering script to produce student clusters:
python DecomposeAUC/run_posthoc_auc_irtwofit.py \
    --save-dir examples/saved_model/assist2015_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0 \
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

This script runs post-hoc student clustering with IRT only.
Clustering is fit on the test split itself, so it does not require any overlap
between training and test students, and it does not use person-fit statistics.

By default the script keeps all low-support students/items so the downstream
decomposition can match the full wandb_predict AUC. Pass --apply-support-filtering
to restore the original min_student_responses / min_item_responses filtering.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from DecomposeAUC.cluster_students import fit_irt
except ModuleNotFoundError:
    from cluster_students import fit_irt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "DecomposeAUC" / "auc_figures"
DEFAULT_CLUSTER_ORDER = ["bad", "mid_chaotic", "mid_stable", "good"]


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
        match = re.search(r"(assist\d+|junyi|ednet|statics2011|nips_task34)", name, re.IGNORECASE)
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


def _theta_only_clusters(
    theta: np.ndarray,
    bad_theta_threshold: float = -0.5,
    mid_theta_threshold: float = 0.0,
    good_theta_threshold: float = 0.5,
) -> np.ndarray:
    if not (bad_theta_threshold < mid_theta_threshold < good_theta_threshold):
        raise ValueError("Require bad_theta_threshold < mid_theta_threshold < good_theta_threshold")

    clusters = np.empty(theta.shape[0], dtype=object)

    bad_mask = theta <= bad_theta_threshold
    good_mask = theta > good_theta_threshold
    mid_mask = ~(bad_mask | good_mask)

    clusters[bad_mask] = "bad"
    clusters[good_mask] = "good"

    lower_mid_mask = mid_mask & (theta <= mid_theta_threshold)
    upper_mid_mask = mid_mask & (theta > mid_theta_threshold)

    clusters[lower_mid_mask] = "mid_chaotic"
    clusters[upper_mid_mask] = "mid_stable"

    return clusters


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

    response_matrix, student_values, _item_values, student_n, _item_n = _build_response_matrix(
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

    theta = irt_res["theta"]
    clusters = _theta_only_clusters(
        theta=theta,
        bad_theta_threshold=bad_theta_threshold,
        mid_theta_threshold=mid_theta_threshold,
        good_theta_threshold=good_theta_threshold,
    )

    cluster_df = pd.DataFrame(
        {
            student_col: student_values,
            "theta": theta,
            "n_responses": student_n,
            "cluster": clusters,
        }
    )

    summary = {
        "bad_theta_threshold": float(bad_theta_threshold),
        "mid_theta_threshold": float(mid_theta_threshold),
        "good_theta_threshold": float(good_theta_threshold),
        "irt_model": irt_model,
        "converged": int(irt_res["converged"][0]),
        "iterations": int(irt_res["iterations"][0]),
        "n_students": int(cluster_df.shape[0]),
        "n_items": int(response_matrix.shape[1]),
        "apply_support_filtering": bool(apply_support_filtering),
    }
    return cluster_df, summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyKT post-hoc student clustering with test-set IRT and no person-fit."
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

    test_csv = _resolve_existing_path(args.test_csv, save_dir, expect_dir=False) if args.test_csv else _infer_test_csv(save_dir, config, dataset_name)
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

    cluster_csv = os.path.join(run_output_dir, "student_clusters_irtwofit.csv")
    cluster_df.to_csv(cluster_csv, index=False)

    cluster_summary_path = os.path.join(run_output_dir, "cluster_summary_irtwofit.json")
    with open(cluster_summary_path, "w", encoding="utf-8") as fout:
        payload = {
            **cluster_summary,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "test_csv": test_csv,
            "cluster_csv": cluster_csv,
        }
        json.dump(payload, fout, ensure_ascii=False, indent=2)

    print("[DONE] Post-hoc clustering finished.")
    print(f"[DONE] model_name: {model_name}")
    print(f"[DONE] test_csv: {test_csv}")
    print(f"[DONE] cluster_csv: {cluster_csv}")
    print(f"[DONE] output_dir: {run_output_dir}")
    print(f"[DONE] summary: {cluster_summary_path}")


if __name__ == "__main__":
    main()