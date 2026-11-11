"""Prefix-based IRT group labels and stratified interaction batches for AKT."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate


GROUP_INDEX = {"low": 0, "mid": 1, "high": 2}


class MinimaxSequences(Dataset):
    """Attach a fixed student group and mask away predictions inside its prefix."""

    def __init__(self, dataset, path, folds, histories, group_ids, prefix_length, *, question_starts=False):
        self.dataset = dataset
        columns = ["uid", "fold", "responses"]
        if question_starts:
            columns.append("is_repeat")
        rows = pd.read_csv(path, dtype=str, usecols=columns)
        rows = rows.loc[rows["fold"].isin(folds)]
        if len(rows) != len(dataset):
            raise ValueError("AKT minimax sequence rows do not match the PyKT dataset cache.")
        responses = dataset.dori["rseqs"].cpu().numpy()
        self.labels = responses[:, 1:]
        self.mask = (dataset.dori["masks"] & dataset.dori["smasks"]).cpu().clone()
        self.question_mask = self.mask.clone() if question_starts else None
        self.groups = torch.full((len(dataset),), -1, dtype=torch.long)
        offsets = {}
        for idx, row in enumerate(rows.itertuples(index=False)):
            uid = row.uid
            if uid not in histories:
                raise ValueError(f"AKT minimax could not find original history for student {uid}.")
            answers = np.fromstring(row.responses, sep=",", dtype=int)
            if not np.array_equal(responses[idx], answers):
                raise ValueError(f"AKT minimax found stale or misordered sequence data for student {uid}.")
            if question_starts:
                repeats = np.fromstring(row.is_repeat, sep=",", dtype=int)
                if repeats.shape != answers.shape or not np.isin(repeats[answers != -1], [0, 1]).all():
                    raise ValueError(f"Invalid is_repeat values for student {uid}.")
                # Later concepts of a question can already see its response
                # in a standard AKT training sequence.
                self.question_mask[idx] &= torch.from_numpy(repeats[1:] == 0)
                self.mask[idx] &= self.question_mask[idx]
            start = offsets.get(uid, 0)
            actual = answers[answers != -1]
            expected = [label for _, label, _ in histories[uid][start:start + len(actual)]]
            if not np.array_equal(actual, expected):
                raise ValueError(f"AKT minimax found misaligned history chunks for student {uid}.")
            offsets[uid] = start + len(actual)
            group = group_ids.get(uid)
            if group is None:
                self.mask[idx].fill_(False)
            else:
                self.groups[idx] = group
                # Target offset zero in the shifted array is interaction 1.
                self.mask[idx, :max(0, prefix_length - start - 1)] = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        row["minimax_mask"] = self.mask[idx]
        row["minimax_group"] = self.groups[idx]
        if self.question_mask is not None:
            row["question_mask"] = self.question_mask[idx]
        return row

    def batch(self, indices):
        """Gather a sequence batch with the same masking as KTDataset."""
        data = self.dataset.dori
        rows = torch.as_tensor(indices, dtype=torch.long, device=self.mask.device)
        data_device = data["masks"].device
        data_rows = rows.to(data_device, non_blocking=True)
        mask = data["masks"].index_select(0, data_rows)
        batch = {
            "masks": mask,
            "smasks": data["smasks"].index_select(0, data_rows),
            "minimax_mask": self.mask.index_select(0, rows).to(data_device, non_blocking=True),
            "minimax_group": self.groups.index_select(0, rows).to(data_device, non_blocking=True),
        }
        if self.question_mask is not None:
            batch["question_mask"] = self.question_mask.index_select(0, rows).to(data_device, non_blocking=True)
        for key in ("cseqs", "qseqs", "rseqs"):
            values = data[key]
            if values.numel():
                values = values.index_select(0, data_rows)
                batch[key] = values[:, :-1] * mask
                batch["shft_" + key] = values[:, 1:] * mask
            else:
                batch[key] = batch["shft_" + key] = values.expand(len(rows), 0)
        return batch

    def group_label_counts(self):
        eligible = self.mask.numpy()
        groups = np.broadcast_to(self.groups.numpy()[:, None], eligible.shape)
        strata = 2 * groups[eligible] + self.labels[eligible].astype(int)
        return np.bincount(strata, minlength=2 * len(GROUP_INDEX)).reshape(-1, 2)


class MinimaxStratifiedTargets:
    """Appendix B sampler over the six group/label strata."""
    def __init__(self, dataset, batch_size, counts):
        self.dataset = dataset
        self.group_label_counts = counts
        self.pools = [
            np.argwhere(dataset.mask.numpy() & (dataset.groups.numpy()[:, None] == group)
                        & (dataset.labels == label))
            for group in range(len(GROUP_INDEX)) for label in (0, 1)
        ]
        flat = counts.ravel()
        self.take = np.maximum(1, batch_size * flat // flat.sum())
        self.steps = int((flat // self.take).min())
        if not self.steps:
            raise ValueError("The batch size is too large for the smallest stratum.")

    def __len__(self):
        return self.steps

    def __iter__(self):
        orders = [np.random.permutation(len(pool)) for pool in self.pools]
        for step in range(self.steps):
            selected = {}
            for pool, order, size in zip(self.pools, orders, self.take):
                for row, position in pool[order[step * size:(step + 1) * size]]:
                    selected.setdefault(int(row), []).append(int(position))
            batch = default_collate([self.dataset[row] for row in selected])
            batch["minimax_mask"].zero_()
            for i, positions in enumerate(selected.values()):
                batch["minimax_mask"][i, positions] = True
            yield batch


def minimax_loaders(config, fold, train, valid, batch_size, prefix_length=15, item_col="concepts",
                    *, full_validation=False, stratified_targets=False, batched_sequences=False,
                    question_starts=False):
    # Reuse the same 2PL/EAP calibration and score cuts as the AUC audit.
    from DecomposeAUC.run_posthoc_auc_irt_tertiles import (
        assign_groups, calibrate, histories, read_sequences, score,
    )

    if prefix_length < 3 or item_col not in config["input_type"]:
        raise ValueError("Use a prefix of at least three interactions and an item column available in the dataset.")
    root = Path(config["dpath"])
    original = root / config["train_valid_original_file"]
    sequence_path = root / config["train_valid_file"]
    students, student_folds, _ = read_sequences(original, item_col)
    train_folds = set(map(str, config["folds"])) - {str(fold)}
    valid_folds = {str(fold)}
    prefix = histories(students, prefix_length, "first", require_future=True)
    fit_rows = {uid: row for uid, row in prefix.items() if student_folds[uid] in train_folds}
    ref_rows = {uid: row for uid, row in prefix.items() if student_folds[uid] in valid_folds}
    index, _, logits, grid, prior, _ = calibrate(fit_rows, min_item_responses=30)
    ref = score(ref_rows, index, logits, grid, prior, min_items=3)
    if len(ref) < 3:
        raise ValueError("Too few scored students in the AKT minimax reference fold.")
    cuts = np.quantile(ref["theta"], [1 / 3, 2 / 3])
    if cuts[0] >= cuts[1]:
        raise ValueError("AKT minimax reference tertile cuts coincide.")
    scored = pd.concat([score(fit_rows, index, logits, grid, prior, min_items=3), ref], ignore_index=True)
    assign_groups(scored, cuts)
    group_ids = dict(zip(scored["student_id"], scored["cluster"].map(GROUP_INDEX)))
    train = MinimaxSequences(train, sequence_path, train_folds, students, group_ids, prefix_length,
                             question_starts=question_starts)
    valid = MinimaxSequences(valid, sequence_path, valid_folds, students, group_ids, prefix_length,
                             question_starts=question_starts)
    train_counts = train.group_label_counts()
    if (train_counts == 0).any() or (valid.group_label_counts() == 0).any():
        raise ValueError("AKT minimax needs positives and negatives in all three groups in both splits.")
    if stratified_targets:
        train_loader = MinimaxStratifiedTargets(train, batch_size, train_counts)
    else:
        if batched_sequences:
            train_loader = DataLoader(range(len(train)), batch_size=batch_size, shuffle=True, collate_fn=train.batch)
        else:
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        train_loader.group_label_counts = train_counts
    valid_rows = range(len(valid)) if full_validation else torch.where(valid.mask.any(dim=1))[0].tolist()
    if batched_sequences:
        valid_loader = DataLoader(valid_rows, batch_size=batch_size, collate_fn=valid.batch)
    else:
        valid_loader = DataLoader(valid if full_validation else Subset(valid, valid_rows), batch_size=batch_size)
    print(f"AKT minimax fixed IRT cuts: {cuts.tolist()}, scored students: {len(scored)}, "
          f"batches per epoch: {len(train_loader)}")
    return train_loader, valid_loader
