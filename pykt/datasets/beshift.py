"""BeShift augmentation of the original histories in a pyKT training fold."""

import math

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from ..preprocess.split_datasets import generate_sequences
from .data_loader import KTDataset
from .fairness_histories import concept_statistics, training_histories


def accuracy_bins(accuracies, width, num_bins):
    mean = float(np.mean(accuracies))
    if not 0 < width <= 1 or mean <= 0:
        raise ValueError("BeShift requires a positive training accuracy and a bin width in (0, 1].")
    if num_bins is None:
        # The paper uses the low-performing training cohort's mean. In the
        # fixed pyKT fold use the lower half by student accuracy.
        low = np.sort(accuracies)[:max(1, len(accuracies) // 2)]
        num_bins = max(1, math.floor(np.mean(low) / width + 1e-12))
    if num_bins < 1 or (num_bins - 1) * width >= mean:
        raise ValueError("BeShift accuracy bins must fit below the training mean.")
    edges = np.maximum(mean - np.arange(num_bins + 1) * width, 0.0)
    return list(zip(edges[1:], edges[:-1]))


def allocate_bins(num_sequences, num_bins, decay):
    if num_sequences < 1 or not math.isfinite(decay) or decay <= 0:
        raise ValueError("BeShift requires a positive sequence count and bin decay.")
    weights = np.exp(-decay * np.arange(num_bins))
    expected = num_sequences * weights / weights.sum()
    counts = expected.astype(int)
    remaining = num_sequences - counts.sum()
    largest = np.argsort(counts - expected, kind="stable")[:remaining]
    counts[largest] += 1
    return counts


def flip_probabilities(history, skill_rates, behavior, severity):
    """Equations 3–6, with one flip for all KC copies of a question."""
    shape = 1.0 + 4.0 * severity  # gamma or kappa, in [1, 5]
    if behavior == "fatigue":
        count = len(history.question_starts)
        return np.expm1(shape * np.arange(1, count + 1) / count) / np.expm1(shape)
    rates = skill_rates[history.concepts]
    if behavior == "forgetting":
        probabilities = np.expm1(shape * (1.0 - rates)) / np.expm1(shape)
    else:
        epsilon = 0.05 + 0.30 * severity
        probabilities = 1.0 - np.maximum(0.0, rates - epsilon)
    starts = history.question_starts
    return np.add.reduceat(probabilities, starts) / np.diff(np.r_[starts, len(rates)])


def _conditioned_bernoulli(probabilities, low, high, rng):
    """Draw independent Bernoulli variables conditioned on a sum interval."""
    # Rejection is fast for ordinary bins. Dynamic programming samples from
    # exactly the same conditional law when the interval is very unlikely.
    for _ in range(16):
        sample = rng.random(len(probabilities)) < probabilities
        if low <= sample.sum() <= high:
            return sample

    complement = len(probabilities) - low < high
    if complement:
        probabilities = 1.0 - probabilities
        low, high = len(probabilities) - high, len(probabilities) - low

    log_yes = np.log(probabilities)
    log_no = np.log1p(-probabilities)
    suffix = np.full((len(probabilities) + 1, high + 1), -np.inf)
    suffix[-1, 0] = 0.0
    for i in range(len(probabilities) - 1, -1, -1):
        suffix[i] = log_no[i] + suffix[i + 1]
        suffix[i, 1:] = np.logaddexp(suffix[i, 1:], log_yes[i] + suffix[i + 1, :-1])
    totals = suffix[0, low:high + 1]
    weights = np.exp(totals - totals.max())
    remaining = low + rng.choice(len(weights), p=weights / weights.sum())
    sample = np.zeros(len(probabilities), dtype=bool)
    for i in range(len(probabilities)):
        if not remaining:
            break
        if remaining == len(probabilities) - i:
            sample[i:] = True
            break
        log_probability = log_yes[i] + suffix[i + 1, remaining - 1] - suffix[i, remaining]
        if rng.random() < np.exp(log_probability):
            sample[i] = True
            remaining -= 1
    return ~sample if complement else sample


def calibrated_responses(history, probabilities, interval, rng):
    """Return a flipped response array, or None if the bin is unreachable."""
    starts = history.question_starts
    correct = history.responses[starts] == 1
    forced_correct = correct & (probabilities == 0)
    uncertain = correct & (probabilities > 0) & (probabilities < 1)
    left, right = interval
    fixed = int(forced_correct.sum())
    low = max(0, math.ceil(left * len(starts) - 1e-12) - fixed)
    high = min(int(uncertain.sum()), math.ceil(right * len(starts) - 1e-12) - 1 - fixed)
    if high < low:
        return None

    result = forced_correct.copy()
    if uncertain.any():
        result[uncertain] = _conditioned_bernoulli(1.0 - probabilities[uncertain], low, high, rng)
    return np.repeat(result.astype(int), np.diff(np.r_[starts, len(history.responses)]))


def generate_beshift(rows, histories, num_concepts, *, seed, fraction=0.3, num_sequences=None,
                     bin_width=0.1, num_bins=None, decay=1.0):
    if not 0 < fraction <= 1:
        raise ValueError("BeShift source fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(rows), max(1, round(len(rows) * fraction)), replace=False)
    if num_sequences is None:
        num_sequences = len(pool)
    _, skill_rates = concept_statistics(histories, num_concepts)
    bins = accuracy_bins([history.accuracy for history in histories], bin_width, num_bins)
    counts = allocate_bins(num_sequences, len(bins), decay)
    behaviors = ("forgetting", "fatigue", "knowledge_gap")
    used_sources = np.zeros(len(pool), dtype=bool)
    output = []
    for index, (interval, count) in enumerate(zip(bins, counts)):
        severity = index / max(1, len(bins) - 1)
        for _ in range(count):
            pairs = rng.permutation(len(pool) * len(behaviors))
            used = used_sources[pairs // len(behaviors)]
            for pair in np.concatenate((pairs[~used], pairs[used])):
                source_index, behavior = divmod(pair, len(behaviors))
                source = pool[source_index]
                history = histories[source]
                probabilities = flip_probabilities(history, skill_rates, behaviors[behavior], severity)
                responses = calibrated_responses(history, probabilities, interval, rng)
                if responses is not None:
                    break
            else:
                raise ValueError(f"BeShift bin [{interval[0]:.4f}, {interval[1]:.4f}) cannot be reached by the sampled students.")
            used_sources[source_index] = True
            row = rows.iloc[source].to_dict()
            row["uid"] = f"beshift_{len(output)}"
            row["responses"] = ",".join(map(str, responses))
            output.append(row)
    return pd.DataFrame(output, columns=rows.columns), list(zip(bins, counts.tolist()))


def beshift_loaders(config, fold, train, valid, batch_size, **augmentation):
    rows, histories = training_histories(config, fold)
    synthetic, allocation = generate_beshift(rows, histories, config["num_c"], **augmentation)
    keys = [key for key in ("fold", "uid", "responses", "concepts", "questions", "timestamps", "usetimes", "is_repeat")
            if key in synthetic]
    sequences = generate_sequences(synthetic, keys, min_seq_len=config["min_seq_len"], maxlen=config["maxlen"])
    augmented = KTDataset.from_sequences(sequences, config["input_type"])
    print(f"BeShift synthesized {len(synthetic)} student histories; accuracy bin allocation: {allocation}")
    train = DataLoader(ConcatDataset([train, augmented]), batch_size=batch_size, shuffle=True)
    return train, DataLoader(valid, batch_size=batch_size)
