import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd



# python DecomposeAUC/trans.py --pred-txt examples/saved_model/assist2017_akt_qid_saved_model_3407_0_0.2_256_512_8_4_0.0001_0_0/qid_test_predictions.txt
# python DecomposeAUC/trans.py --pred-txt examples/saved_model/assist2015_lefokt_akt_qid_saved_model_3407_0_0.2_256_512_4_1_0.0001_16_100_0.01_50_32_0_0/qid_test_predictions.txt



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _default_path(name: str) -> Path:
    return SCRIPT_DIR / name


def _load_run_config(pred_txt: Path) -> dict:
    config_path = pred_txt.parent / "config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _infer_dataset_name(pred_txt: Path, config: dict) -> str:
    params = config.get("params", {}) if isinstance(config, dict) else {}
    dataset_name = params.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        return dataset_name.strip()

    for part in pred_txt.parts:
        match = re.search(r"(assist\d+|junyi|ednet|statics2011|nips_task34)", part, re.IGNORECASE)
        if match:
            return match.group(1)

    raise ValueError(f"无法从 {pred_txt} 推断数据集名称")


def _infer_seq_csv(pred_txt: Path, config: dict) -> Path:
    dataset_name = _infer_dataset_name(pred_txt, config)
    data_config = config.get("data_config", {}) if isinstance(config, dict) else {}
    test_file = data_config.get("test_file", "test_sequences.csv")
    if not isinstance(test_file, str) or not test_file.strip():
        test_file = "test_sequences.csv"

    candidates = [
        PROJECT_ROOT / "data" / dataset_name / test_file,
        PROJECT_ROOT / test_file,
        pred_txt.parent.parent / "data" / dataset_name / test_file,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "无法自动定位测试序列文件，已尝试: " + ", ".join(str(item) for item in candidates)
    )


def _infer_out_csv(pred_txt: Path, config: dict) -> Path:
    dataset_name = _infer_dataset_name(pred_txt, config)
    params = config.get("params", {}) if isinstance(config, dict) else {}
    model_name = params.get("model_name", "akt")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = "akt"
    return SCRIPT_DIR / f"{dataset_name}_{model_name}_pykt_pred_long.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 pykt 的 sequence 级预测 txt 展开为交互级 CSV。",
    )
    parser.add_argument(
        "--pred-txt",
        default=str(_default_path("qid_test_predictions.txt")),
        help="pykt 生成的预测文本文件路径",
    )
    parser.add_argument(
        "--seq-csv",
        default="",
        help="包含 uid 列的测试序列 CSV 路径；不传时自动根据 pred-txt 反推",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="输出的长表 CSV 路径；不传时会按数据集自动命名",
    )
    return parser.parse_args()


def _load_prediction_line(raw: str, line_no: int):
    try:
        parsed = ast.literal_eval(raw.replace("nan", "None"))
    except Exception as exc:
        raise ValueError(f"无法解析预测文件第 {line_no} 行: {raw[:120]}") from exc

    if not isinstance(parsed, (list, tuple)) or len(parsed) < 5:
        raise ValueError(f"预测文件第 {line_no} 行格式不正确: {raw[:120]}")

    labels = parsed[3]
    scores = parsed[4]
    if not isinstance(labels, (list, tuple)) or not isinstance(scores, (list, tuple)):
        raise ValueError(f"预测文件第 {line_no} 行的标签或分数不是列表: {raw[:120]}")
    if len(labels) != len(scores):
        raise ValueError(f"预测文件第 {line_no} 行标签和分数长度不一致: {len(labels)} != {len(scores)}")

    return labels, scores


def main() -> None:
    args = _parse_args()

    pred_txt = Path(args.pred_txt)
    config = _load_run_config(pred_txt)

    if args.seq_csv:
        seq_csv = Path(args.seq_csv)
    else:
        seq_csv = _infer_seq_csv(pred_txt, config)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = _infer_out_csv(pred_txt, config)

    if not pred_txt.exists():
        raise FileNotFoundError(f"prediction 文件不存在: {pred_txt}")
    if not seq_csv.exists():
        raise FileNotFoundError(f"sequence 文件不存在: {seq_csv}")

    seq_df = pd.read_csv(seq_csv)
    if "uid" not in seq_df.columns:
        raise ValueError(f"{seq_csv} 中缺少 uid 列")

    uids = seq_df["uid"].astype(str).tolist()

    with open(pred_txt, "r", encoding="utf-8") as fin:
        pred_lines = [line.strip() for line in fin if line.strip()]

    if len(pred_lines) != len(uids):
        raise ValueError(f"prediction 行数和 test_sequences 行数不一致: {len(pred_lines)} != {len(uids)}")

    rows = []
    for line_no, (uid, line) in enumerate(zip(uids, pred_lines), start=1):
        labels, scores = _load_prediction_line(line, line_no)

        for t, (y_true, y_score) in enumerate(zip(labels, scores)):
            if y_true is None or y_score is None:
                continue

            rows.append(
                {
                    "student_id": uid,
                    "interaction_index": t,
                    "pred": float(y_score),
                    "label": float(y_true),
                }
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved to {out_csv}, n={len(df)}")


if __name__ == "__main__":
    main()