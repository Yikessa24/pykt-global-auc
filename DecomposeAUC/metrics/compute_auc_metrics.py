"""
Compute AUC metrics for DecomposeAUC outputs.

This script is intentionally decoupled from training code. It reads existing
prediction files and, optionally, an AUC decomposition matrix CSV, then writes
a human-readable TXT report.

What it computes:
- The mean value of the AUC decomposition matrix (all valid cells)
- Per-student AUC values and their average
- Recomputed student-average AUC under multiple minimum-length thresholds

Important behavior:
- Input files are never modified.
- Students with all-1 or all-0 labels are handled by appending one opposite
  label with the last observed score duplicated, so AUC can still be computed.
- Threshold filtering is done on in-memory copies only.

Typical usage:
python DecomposeAUC/metrics/compute_auc_metrics.py \
  --dataset assist2009 --model akt_dro --run-id 3407 \
  --save-dir examples/saved_model/assist2009_akt_dro_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_2.0_0.7_0.5_1_4_0.2_0.0_4096_0.2_0.4_0.7_0.35_0.001_0.05_0_0\
  --pred-txt examples/saved_model/assist2009_akt_dro_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_2.0_0.7_0.5_1_4_0.2_0.0_4096_0.2_0.4_0.7_0.35_0.001_0.05_0_0/qid_test_predictions.txt \
  --matrix-csv DecomposeAUC/auc_figures/assist2009_akt_dro_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_2.0_0.7_0.5_1_4_0.2_0.0_4096_0.2_0.4_0.7_0.35_0.001_0.05_0_0/auc_matrix.csv



python DecomposeAUC/metrics/compute_auc_metrics.py \
  --dataset assist2009 --model akt --run-id 3407 \
  --save-dir examples/saved_model/assist2009_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0\
  --pred-txt examples/saved_model/assist2009_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/qid_test_predictions.txt \
  --matrix-csv DecomposeAUC/auc_figures/assist2009_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/auc_matrix.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_THRESHOLDS = [3, 5, 7, 10]


def _as_path(value: str) -> Optional[Path]:
    if not value:
        return None
    return Path(os.path.expanduser(value)).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_token(value: str) -> str:
    token = str(value).strip()
    if not token:
        return "unknown"
    out = []
    for ch in token:
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(out)


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_int(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return str(int(value))


def _read_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _load_save_dir_context(save_dir: Optional[Path]) -> Dict[str, str]:
    if save_dir is None:
        return {}

    config_path = save_dir / "config.json"
    config = _read_json_if_exists(config_path)
    if not config:
        return {}

    params = config.get("params", {}) if isinstance(config, dict) else {}
    context = {
        "dataset": str(params.get("dataset_name", params.get("dataset", ""))),
        "model": str(params.get("model_name", params.get("model", ""))),
        "emb_type": str(params.get("emb_type", "")),
        "fold": str(params.get("fold", "")),
    }
    return {k: v for k, v in context.items() if v}


def _resolve_existing_path(
    explicit_path: Optional[Path],
    candidates: Sequence[Path],
    label: str,
) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"{label} not found: {explicit_path}")
        return explicit_path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n{tried}")


def _pick_best_path(paths: Sequence[Path], dataset: str, model: str, run_id: str) -> Optional[Path]:
    if not paths:
        return None

    def score(path: Path) -> Tuple[int, int]:
        text = str(path)
        total = 0
        if dataset and dataset in text:
            total += 3
        if model and model in text:
            total += 3
        if run_id and run_id in text:
            total += 2
        if text.endswith("auc_matrix.csv"):
            total += 1
        return total, -len(text)

    return sorted(paths, key=score, reverse=True)[0]


def _resolve_pred_txt(
    pred_txt: Optional[Path],
    save_dir: Optional[Path],
    context: Dict[str, str],
) -> Path:
    candidates: List[Path] = []
    if save_dir is not None:
        emb_type = context.get("emb_type", "")
        model = context.get("model", "")
        if emb_type:
            candidates.append(save_dir / f"{emb_type}_test_predictions.txt")
            candidates.append(save_dir / f"{emb_type}_test_window_predictions.txt")
        if model:
            candidates.append(save_dir / f"{model}_test_predictions.txt")
        candidates.append(save_dir / "qid_test_predictions.txt")
        candidates.append(save_dir / "test_predictions.txt")

    return _resolve_existing_path(pred_txt, candidates, "prediction txt")


def _resolve_matrix_csv(
    matrix_csv: Optional[Path],
    save_dir: Optional[Path],
    dataset: str,
    model: str,
    run_id: str,
) -> Path:
    candidates: List[Path] = []

    if save_dir is not None:
        candidates.append(save_dir / "auc_matrix.csv")
        if dataset and model:
            candidates.append(save_dir / f"{dataset}_{model}_auc_matrix.csv")
        if model:
            candidates.append(save_dir / f"{model}_auc_matrix.csv")

    figures_dir = PROJECT_ROOT / "DecomposeAUC" / "auc_figures"
    if figures_dir.exists():
        if dataset and model:
            candidates.append(figures_dir / f"{dataset}_{model}_auc_matrix.csv")
        if model:
            candidates.append(figures_dir / f"{model}_auc_matrix.csv")
        if dataset and model and run_id:
            candidates.append(figures_dir / f"{dataset}_{model}_run{run_id}_auc_matrix.csv")

        discovered = [path.resolve() for path in figures_dir.rglob("auc_matrix.csv") if path.is_file()]
        best = _pick_best_path(discovered, dataset, model, run_id)
        if best is not None:
            candidates.insert(0, best)

    if save_dir is not None and save_dir.exists():
        discovered = [path.resolve() for path in save_dir.rglob("auc_matrix.csv") if path.is_file()]
        best = _pick_best_path(discovered, dataset, model, run_id)
        if best is not None:
            candidates.insert(0, best)

    return _resolve_existing_path(matrix_csv, candidates, "matrix csv")


def _coerce_float_list(value: object) -> List[float]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        result: List[float] = []
        for item in value:
            if item is None:
                continue
            try:
                result.append(float(item))
            except Exception:
                continue
        return result

    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text.replace("nan", "None"))
        except Exception:
            parsed = [part.strip() for part in text.split(",") if part.strip()]
        return _coerce_float_list(parsed)

    return []


def _looks_like_raw_sequence_line(line: str) -> bool:
    try:
        parsed = ast.literal_eval(line.replace("nan", "None"))
    except Exception:
        return False

    if not isinstance(parsed, (list, tuple)) or len(parsed) < 5:
        return False

    labels = parsed[3]
    scores = parsed[4]
    return isinstance(labels, (list, tuple)) and isinstance(scores, (list, tuple))


def _parse_raw_sequence_file(pred_txt: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(pred_txt, "r", encoding="utf-8") as fin:
        for seq_idx, raw in enumerate(fin):
            line = raw.strip()
            if not line:
                continue

            try:
                parsed = ast.literal_eval(line.replace("nan", "None"))
            except Exception:
                continue

            if not isinstance(parsed, (list, tuple)) or len(parsed) < 5:
                continue

            labels = _coerce_float_list(parsed[3])
            scores = _coerce_float_list(parsed[4])
            if not labels or len(labels) != len(scores):
                continue

            records.append(
                {
                    "student_id": f"seq_{seq_idx:05d}",
                    "labels": labels,
                    "scores": scores,
                    "source_format": "raw_sequence",
                }
            )

    return records


def _parse_tabular_prediction_file(pred_txt: Path) -> List[Dict[str, object]]:
    grouped: "OrderedDict[str, Dict[str, object]]" = OrderedDict()

    with open(pred_txt, "r", encoding="utf-8") as fin:
        for row_idx, raw in enumerate(fin):
            line = raw.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split("\t")]
            if len(parts) < 4:
                continue

            uid = parts[1] if len(parts) > 1 else parts[0]

            label_value: Optional[float] = None
            score_value: Optional[float] = None
            candidate_pairs = [(-3, -2), (-2, -1), (2, 3)]
            for label_idx, score_idx in candidate_pairs:
                try:
                    label_value = float(parts[label_idx])
                    score_value = float(parts[score_idx])
                    break
                except Exception:
                    continue

            if label_value is None or score_value is None:
                continue

            if uid not in grouped:
                grouped[uid] = {
                    "student_id": uid,
                    "labels": [],
                    "scores": [],
                    "source_format": "tabular",
                }

            grouped[uid]["labels"].append(label_value)  # type: ignore[index]
            grouped[uid]["scores"].append(score_value)  # type: ignore[index]

    return list(grouped.values())


def load_prediction_sequences(pred_txt: Path) -> List[Dict[str, object]]:
    with open(pred_txt, "r", encoding="utf-8") as fin:
        first_line = ""
        for raw in fin:
            first_line = raw.strip()
            if first_line:
                break

    if not first_line:
        raise ValueError(f"Prediction txt is empty: {pred_txt}")

    if _looks_like_raw_sequence_line(first_line):
        records = _parse_raw_sequence_file(pred_txt)
    else:
        records = _parse_tabular_prediction_file(pred_txt)

    if not records:
        raise ValueError(f"No valid prediction sequences could be parsed from: {pred_txt}")

    return records


def _append_synthetic_opposite_label(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    labels = labels.astype(int, copy=True)
    scores = scores.astype(float, copy=True)

    unique_labels = np.unique(labels)
    synthetic_added = False

    if unique_labels.size == 1:
        synthetic_added = True
        synthetic_label = 0 if int(unique_labels[0]) == 1 else 1
        synthetic_score = float(scores[-1]) if scores.size else 0.5
        labels = np.append(labels, synthetic_label)
        scores = np.append(scores, synthetic_score)

    return labels, scores, synthetic_added


def compute_student_auc_records(sequences: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for seq in sequences:
        student_id = str(seq["student_id"])
        labels = np.asarray(seq["labels"], dtype=float)
        scores = np.asarray(seq["scores"], dtype=float)

        if labels.size == 0 or scores.size == 0:
            continue

        labels = (labels >= 0.5).astype(int)
        original_len = int(labels.size)
        original_pos = int(labels.sum())
        original_neg = int(original_len - original_pos)

        labels_fixed, scores_fixed, synthetic_added = _append_synthetic_opposite_label(labels, scores)

        try:
            auc_value = float(roc_auc_score(labels_fixed, scores_fixed))
        except Exception:
            auc_value = float("nan")

        rows.append(
            {
                "student_id": student_id,
                "length": original_len,
                "pos_count": original_pos,
                "neg_count": original_neg,
                "synthetic_fix": synthetic_added,
                "auc": auc_value,
                "source_format": seq.get("source_format", "unknown"),
            }
        )

    if not rows:
        raise ValueError("No valid student-level AUC rows were produced.")

    return pd.DataFrame(rows)


def compute_matrix_mean(matrix_csv: Path) -> Dict[str, object]:
    df = pd.read_csv(matrix_csv, index_col=0)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    valid_values = values[finite_mask]

    if valid_values.size == 0:
        raise ValueError(f"No numeric values found in matrix csv: {matrix_csv}")

    return {
        "rows": int(numeric.shape[0]),
        "cols": int(numeric.shape[1]),
        "valid_cells": int(valid_values.size),
        "mean": float(np.mean(valid_values)),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
    }


def summarize_student_auc(df: pd.DataFrame) -> Dict[str, object]:
    valid = df[~df["auc"].isna()].copy()
    if valid.empty:
        return {
            "n_students": int(df.shape[0]),
            "valid_students": 0,
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "median_auc": float("nan"),
            "min_auc": float("nan"),
            "max_auc": float("nan"),
            "synthetic_fixed_students": int(df["synthetic_fix"].sum()),
        }

    return {
        "n_students": int(df.shape[0]),
        "valid_students": int(valid.shape[0]),
        "mean_auc": float(valid["auc"].mean()),
        "std_auc": float(valid["auc"].std(ddof=0)),
        "median_auc": float(valid["auc"].median()),
        "min_auc": float(valid["auc"].min()),
        "max_auc": float(valid["auc"].max()),
        "synthetic_fixed_students": int(df["synthetic_fix"].sum()),
    }


def recompute_for_thresholds(df: pd.DataFrame, thresholds: Sequence[int]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for threshold in thresholds:
        filtered = df[df["length"] >= int(threshold)].copy()
        summary = summarize_student_auc(filtered) if not filtered.empty else {
            "n_students": 0,
            "valid_students": 0,
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "median_auc": float("nan"),
            "min_auc": float("nan"),
            "max_auc": float("nan"),
            "synthetic_fixed_students": 0,
        }
        summary.update(
            {
                "threshold": int(threshold),
                "removed_students": int(df.shape[0] - filtered.shape[0]),
                "kept_students": int(filtered.shape[0]),
            }
        )
        results.append(summary)
    return results


def _format_section_title(title: str) -> str:
    return "=" * 80 + f"\n{title}\n" + "=" * 80


def _format_key_value_lines(items: Sequence[Tuple[str, object]]) -> List[str]:
    lines = []
    for key, value in items:
        if isinstance(value, float):
            rendered = _format_float(value)
        elif isinstance(value, int):
            rendered = _format_int(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return lines


def _render_student_table(df: pd.DataFrame) -> List[str]:
    columns = ["student_id", "length", "pos_count", "neg_count", "synthetic_fix", "auc", "source_format"]
    table = df[columns].copy()
    table = table.sort_values(by=["length", "student_id"], ascending=[True, True])

    lines = ["student_id\tlength\tpos_count\tneg_count\tsynthetic_fix\tauc\tsource_format"]
    for _, row in table.iterrows():
        lines.append(
            "\t".join(
                [
                    str(row["student_id"]),
                    str(int(row["length"])),
                    str(int(row["pos_count"])),
                    str(int(row["neg_count"])),
                    "yes" if bool(row["synthetic_fix"]) else "no",
                    _format_float(row["auc"], digits=6),
                    str(row["source_format"]),
                ]
            )
        )
    return lines


def build_report(
    dataset: str,
    model: str,
    run_id: str,
    save_dir: Optional[Path],
    pred_txt: Path,
    matrix_csv: Path,
    student_df: pd.DataFrame,
    matrix_summary: Dict[str, object],
    threshold_summaries: List[Dict[str, object]],
    thresholds: Sequence[int],
) -> str:
    summary = summarize_student_auc(student_df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append(_format_section_title("AUC Metrics Report"))
    lines.extend(
        _format_key_value_lines(
            [
                ("dataset", dataset),
                ("model", model),
                ("run_id", run_id),
                ("generated_at", now),
                ("save_dir", str(save_dir) if save_dir else ""),
                ("prediction_txt", str(pred_txt)),
                ("matrix_csv", str(matrix_csv)),
            ]
        )
    )

    lines.append("")
    lines.append(_format_section_title("Method Notes"))
    lines.extend(
        [
            "1. Input files are not modified.",
            "2. If a student sequence is all-1 or all-0, one opposite label is appended with the last score duplicated.",
            "3. Threshold filtering is applied in memory to copies of the parsed student table.",
            "4. Student AUC is the unweighted average across students after parsing and optional filtering.",
        ]
    )

    lines.append("")
    lines.append(_format_section_title("AUC Matrix Summary"))
    lines.extend(
        _format_key_value_lines(
            [
                ("rows", matrix_summary["rows"]),
                ("cols", matrix_summary["cols"]),
                ("valid_cells", matrix_summary["valid_cells"]),
                ("mean_matrix_value", matrix_summary["mean"]),
                ("min_matrix_value", matrix_summary["min"]),
                ("max_matrix_value", matrix_summary["max"]),
            ]
        )
    )

    lines.append("")
    lines.append(_format_section_title("Student-Level AUC Summary"))
    lines.extend(
        _format_key_value_lines(
            [
                ("n_students_parsed", summary["n_students"]),
                ("valid_students", summary["valid_students"]),
                ("synthetic_fixed_students", summary["synthetic_fixed_students"]),
                ("mean_student_auc", summary["mean_auc"]),
                ("std_student_auc", summary["std_auc"]),
                ("median_student_auc", summary["median_auc"]),
                ("min_student_auc", summary["min_auc"]),
                ("max_student_auc", summary["max_auc"]),
            ]
        )
    )

    lines.append("")
    lines.append(_format_section_title("Per-Student AUC Table"))
    lines.extend(_render_student_table(student_df))

    lines.append("")
    lines.append(_format_section_title("Threshold Recalculation Summary"))
    lines.append("thresholds: " + ", ".join(str(int(x)) for x in thresholds))
    for item in threshold_summaries:
        lines.append("")
        lines.extend(
            _format_key_value_lines(
                [
                    ("threshold", item["threshold"]),
                    ("kept_students", item["kept_students"]),
                    ("removed_students", item["removed_students"]),
                    ("valid_students", item["valid_students"]),
                    ("mean_student_auc", item["mean_auc"]),
                    ("std_student_auc", item["std_auc"]),
                    ("median_student_auc", item["median_auc"]),
                    ("min_student_auc", item["min_auc"]),
                    ("max_student_auc", item["max_auc"]),
                    ("synthetic_fixed_students", item["synthetic_fixed_students"]),
                ]
            )
        )

    lines.append("")
    lines.append(_format_section_title("End of Report"))
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute AUC metrics from DecomposeAUC outputs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. assist2009")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. akt_dro")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier used for naming and reporting")
    parser.add_argument("--save-dir", type=str, default="", help="Optional model save directory")
    parser.add_argument("--pred-txt", type=str, default="", help="Prediction txt path")
    parser.add_argument("--matrix-csv", type=str, default="", help="AUC matrix CSV path")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="*",
        default=DEFAULT_THRESHOLDS,
        help="Minimum sequence lengths to keep, e.g. --thresholds 3 5 7 10",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="TXT output directory")
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional explicit output file name; if omitted, dataset/model/run-id is used",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset = str(args.dataset).strip()
    model = str(args.model).strip()
    run_id = str(args.run_id).strip()

    save_dir = _as_path(args.save_dir)
    pred_txt_arg = _as_path(args.pred_txt)
    matrix_csv_arg = _as_path(args.matrix_csv)
    output_dir = _as_path(args.output_dir) or DEFAULT_OUTPUT_DIR
    _ensure_dir(output_dir)

    context = _load_save_dir_context(save_dir)

    pred_txt = _resolve_pred_txt(pred_txt_arg, save_dir, context)
    matrix_csv = _resolve_matrix_csv(matrix_csv_arg, save_dir, dataset, model, run_id)

    sequences = load_prediction_sequences(pred_txt)
    student_df = compute_student_auc_records(sequences)
    matrix_summary = compute_matrix_mean(matrix_csv)
    threshold_summaries = recompute_for_thresholds(student_df, args.thresholds)

    if args.output_name.strip():
        output_name = args.output_name.strip()
    else:
        run_part = f"_run{_sanitize_token(run_id)}" if run_id else ""
        output_name = f"{_sanitize_token(dataset)}_{_sanitize_token(model)}{run_part}_auc_metrics.txt"

    output_path = output_dir / output_name
    report = build_report(
        dataset=dataset,
        model=model,
        run_id=run_id,
        save_dir=save_dir,
        pred_txt=pred_txt,
        matrix_csv=matrix_csv,
        student_df=student_df,
        matrix_summary=matrix_summary,
        threshold_summaries=threshold_summaries,
        thresholds=args.thresholds,
    )

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(report)

    print(f"Saved metrics report to: {output_path}")
    print(f"Prediction txt used: {pred_txt}")
    print(f"Matrix csv used: {matrix_csv}")
    print(f"Students parsed: {student_df.shape[0]}")


if __name__ == "__main__":
    main()