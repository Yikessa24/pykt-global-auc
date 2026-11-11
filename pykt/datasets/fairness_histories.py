"""Training-fold student histories for PG-DRO and BeShift."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StudentHistory:
    concepts: np.ndarray
    responses: np.ndarray
    question_starts: np.ndarray

    @property
    def accuracy(self):
        return self.responses[self.question_starts].mean()


def training_histories(config, fold):
    path = Path(config["dpath"]) / config["train_valid_original_file"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    folds = set(map(str, config["folds"])) - {str(fold)}
    rows = rows.loc[rows["fold"].isin(folds)].reset_index(drop=True)
    if rows.empty or rows["uid"].eq("").any() or not rows["uid"].is_unique:
        raise ValueError("Expected one original history per training-fold student.")

    histories = []
    for row in rows.itertuples(index=False):
        concepts = np.fromstring(row.concepts, dtype=int, sep=",")
        responses = np.fromstring(row.responses, dtype=int, sep=",")
        repeat = np.fromstring(row.is_repeat, dtype=int, sep=",") if hasattr(row, "is_repeat") else np.zeros_like(responses)
        if (not len(responses) or len(concepts) != len(responses) or len(repeat) != len(responses)
                or not np.isin(responses, [0, 1]).all() or not np.isin(repeat, [0, 1]).all()
                or (concepts < 0).any() or (concepts >= config["num_c"]).any() or repeat[0] != 0
                or np.any((repeat[1:] == 1) & (responses[1:] != responses[:-1]))):
            raise ValueError(f"Invalid original training history for student {row.uid}.")
        histories.append(StudentHistory(concepts, responses, np.flatnonzero(repeat == 0)))
    return rows, histories


def concept_statistics(histories, num_concepts):
    count = np.zeros(num_concepts, dtype=np.int64)
    correct = np.zeros(num_concepts, dtype=np.float64)
    for history in histories:
        count += np.bincount(history.concepts, minlength=num_concepts)
        correct += np.bincount(history.concepts, weights=history.responses, minlength=num_concepts)
    rate = np.divide(correct, count, out=np.zeros_like(correct), where=count > 0)
    return count, rate
