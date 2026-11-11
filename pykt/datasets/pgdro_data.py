"""Balanced student groups and knowledge-component weights for PG-DRO."""

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler

from .data_loader import KTDataset
from .fairness_histories import concept_statistics, training_histories


def pgdro_concept_weights(low, high, num_concepts):
    low_count, low_rate = concept_statistics(low, num_concepts)
    high_count, high_rate = concept_statistics(high, num_concepts)
    count = low_count + high_count
    common = (low_count > 0) & (high_count > 0)
    if not common.any():
        raise ValueError("PG-DRO needs knowledge components observed in both training groups.")
    # Equation 5 is undefined for a group with no observations of a skill.
    raw = np.where(common, np.abs(high_rate - low_rate) * np.log1p(count / count.sum()), 0.0)
    if not raw.any():
        raw = common.astype(float)
    return raw / raw.sum()


class PGDROSequences(Dataset):
    def __init__(self, dataset, groups):
        self.dataset = dataset
        self.groups = groups

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        row["pgdro_group"] = int(self.groups[index])
        return row


class UniformGroupBatches(Sampler):
    """Algorithm 1 samples one group uniformly before drawing a mini-batch."""

    def __init__(self, groups, batch_size, seed):
        if batch_size < 1:
            raise ValueError("PG-DRO batch size must be positive.")
        self.pools = [np.flatnonzero(groups == group) for group in (0, 1)]
        if any(not len(pool) for pool in self.pools):
            raise ValueError("Both PG-DRO groups must have eligible training sequences.")
        self.batch_size = batch_size
        self.steps = ceil(len(groups) / batch_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            pool = self.pools[self.rng.integers(2)]
            yield self.rng.choice(pool, min(self.batch_size, len(pool)), replace=False).tolist()


def pgdro_loaders(config, fold, batch_size, seed):
    rows, histories = training_histories(config, fold)
    rng = np.random.default_rng(seed)
    accuracies = np.array([history.accuracy for history in histories])
    # Random tie-breaking keeps the rank-based cohorts disjoint even when
    # several students have the same observed grade.
    ranked = np.lexsort((rng.random(len(rows)), accuracies))
    num_high = int(0.2 * len(ranked))
    if not num_high:
        raise ValueError("PG-DRO needs at least five training students for its top-20% group.")
    low = rng.choice(ranked[:len(ranked) // 2], num_high, replace=False)
    high = ranked[-num_high:]
    group_by_uid = {rows.iloc[index]["uid"]: group
                    for group, indices in enumerate((low, high)) for index in indices}
    weights = pgdro_concept_weights(
        [histories[index] for index in low], [histories[index] for index in high], config["num_c"],
    )

    path = Path(config["dpath"]) / config["train_valid_file"]
    sequences = pd.read_csv(path, dtype=str, keep_default_na=False)
    sequences = sequences.loc[sequences["fold"].isin(set(rows["fold"])) & sequences["uid"].isin(group_by_uid)]
    groups = sequences["uid"].map(group_by_uid).to_numpy(dtype=int)
    batches = UniformGroupBatches(groups, batch_size, seed)
    dataset = PGDROSequences(KTDataset.from_sequences(sequences, config["input_type"]), groups)
    train = DataLoader(dataset, batch_sampler=batches)
    train.pgdro_kc_weights = weights
    valid = KTDataset(str(path), config["input_type"], {fold})
    print(f"PG-DRO training students per group: {num_high}; nonzero KC weights: {np.count_nonzero(weights)}")
    return train, DataLoader(valid, batch_size=batch_size)
