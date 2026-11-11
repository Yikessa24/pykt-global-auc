"""Export held-out AKT question-window scores for score postprocessing."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from torch.utils.data import DataLoader

from pykt.datasets.data_loader import KTDataset
from pykt.models import evaluate_question
from pykt.preprocess.split_datasets import generate_question_sequences


HISTORY_COLUMNS = ("fold", "uid", "questions", "concepts", "responses", "is_repeat")


def _validation_batches(histories, input_type, maxlen, batch_size):
    # Bound window generation and dataset tensors to a few students at a time.
    chunk_size = 8
    question_start = 0
    for start in range(0, len(histories), chunk_size):
        chunk = histories.iloc[start:start + chunk_size]
        _, windows = generate_question_sequences(
            chunk, HISTORY_COLUMNS, maxlen=maxlen, start_row=start, start_question=question_start,
        )
        question_start += sum(repeats.split(",").count("0") for repeats in chunk["is_repeat"])
        if not windows.empty:
            yield from DataLoader(KTDataset.from_sequences(windows, input_type, qtest=True), batch_size=batch_size)


def export_validation(model, data_config, fold, output_dir, batch_size, fusion_type):
    """Use the checkpoint's validation fold and pyKT's test-time fusion rules."""
    if model.model_name != "akt" or "test_question_window_file" not in data_config:
        raise ValueError("Calibration export requires a standard AKT checkpoint on a question-window dataset.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Calibration output directory already exists: {output_dir}")

    source = Path(data_config["dpath"]) / data_config["train_valid_original_file"]
    histories = pd.read_csv(source, dtype=str, keep_default_na=False, usecols=HISTORY_COLUMNS)
    histories = histories.loc[histories["fold"] == str(fold)]
    if histories.empty or histories["uid"].eq("").any() or not histories["uid"].is_unique:
        raise ValueError("Calibration requires one original history per student in the checkpoint's validation fold.")

    batches = _validation_batches(histories, data_config["input_type"], data_config["maxlen"], batch_size)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="akt-calibration-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "output"
        staging.mkdir()
        histories.to_csv(staging / "calibration.csv", index=False)
        predictions = staging / f"{model.emb_type}_validation_question_window_predictions.txt"
        aucs, _ = evaluate_question(model, batches, "akt", fusion_type, str(predictions))
        staging.rename(output_dir)
    print(f"AKT calibration fold {fold} ({len(histories)} students), windowauclate_mean = {aucs['late_mean']:.6f}")
    print(f"Calibration files: {output_dir}")
