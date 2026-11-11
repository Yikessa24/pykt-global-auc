"""Score historical prefixes on a fixed 2PL scale, then apply reference tertiles."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logsumexp

if __package__:
    from .plot_auc_matrix import compute_auc_matrix_by_cluster, plot_auc_matrix
else:
    from plot_auc_matrix import compute_auc_matrix_by_cluster, plot_auc_matrix


GROUPS = ["low", "mid", "high"]


def read_sequences(path, item_col, *, test=False):
    columns = ["uid", "responses", item_col]
    columns += ["cidxs", "selectmasks", "concepts"] if test else ["fold"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=set(columns))
    students, folds, chunks = {}, {}, []
    seen_indices = set()
    for row in frame.to_dict("records"):
        uid = row["uid"]
        labels = [int(y) for y in row["responses"].split(",")]
        items = row[item_col].split(",")
        if not uid or len(items) != len(labels):
            raise ValueError(f"{path}: malformed sequence for student {uid}")
        positions = [i for i, y in enumerate(labels) if y != -1]
        if positions != list(range(len(positions))) or any(
            labels[i] not in (0, 1) or items[i] in ("", "-1") for i in positions
        ):
            raise ValueError(f"{path}: invalid response, item, or padding for student {uid}")
        history = students.setdefault(uid, [])
        if test:
            indices = [int(x) for x in row["cidxs"].split(",")]
            masks = [int(x) for x in row["selectmasks"].split(",")]
            concepts = [int(x) for x in row["concepts"].split(",")]
            if not len(indices) == len(masks) == len(concepts) == len(labels):
                raise ValueError(f"{path}: misaligned test columns for student {uid}")
            for i in positions:
                idx = indices[i]
                if idx < 0 or idx in seen_indices or (history and idx <= history[-1][2]):
                    raise ValueError(f"{path}: use chronological, non-overlapping test_sequences.csv")
                seen_indices.add(idx)
                history.append((items[i], labels[i], idx))
            selected = [i for i in positions[1:] if masks[i] != -1]
            offset = len(history) - len(positions)
            chunks.append((
                uid, concepts[:max(0, len(positions) - 1)], labels[:max(0, len(positions) - 1)],
                [concepts[i] for i in selected], [labels[i] for i in selected],
                [(offset + i, indices[i]) for i in selected],
            ))
        else:
            if uid in folds:
                raise ValueError(f"{path}: use train_valid.csv with one full sequence per student")
            folds[uid] = row["fold"]
            history.extend((items[i], labels[i], i) for i in positions)
    return students, folds, chunks


def histories(students, prefix_length, deduplicate, *, require_future):
    result = {}
    for uid, sequence in students.items():
        if len(sequence) < prefix_length + int(require_future):
            continue
        responses = {}
        for item, label, _ in sequence[:prefix_length]:
            if deduplicate == "last" or item not in responses:
                responses[item] = label
        result[uid] = responses
    return result


def response_matrices(rows, item_index):
    uids = list(rows)
    rr, cc, yy = [], [], []
    for i, responses in enumerate(rows.values()):
        for item, label in responses.items():
            if item in item_index:
                rr.append(i)
                cc.append(item_index[item])
                yy.append(label)
    yy = np.asarray(yy, dtype=float)
    shape = (len(uids), len(item_index))
    right = csr_matrix((yy, (rr, cc)), shape=shape)
    wrong = csr_matrix((1 - yy, (rr, cc)), shape=shape)
    n = np.asarray((right + wrong).sum(axis=1)).ravel().astype(int)
    return uids, right, wrong, n


def posterior_log_joint(right, wrong, logits, log_prior):
    return (
        right @ -np.logaddexp(0, -logits)
        + wrong @ -np.logaddexp(0, logits)
        + log_prior
    )


def calibrate(rows, min_item_responses):
    counts = {}
    for responses in rows.values():
        for item, label in responses.items():
            tally = counts.setdefault(item, [0, 0])
            tally[label] += 1
    items = sorted(i for i, n in counts.items() if sum(n) >= min_item_responses and min(n) > 0)
    if not items:
        raise ValueError("No calibration items meet the response count and outcome requirements.")
    item_index = {item: i for i, item in enumerate(items)}
    _, right, wrong, n = response_matrices(rows, item_index)
    right, wrong = right[n > 0], wrong[n > 0]
    observed = right + wrong

    # Integrate a standard normal ability prior. Weak penalties stabilize poorly
    # identified items (normal priors on log discrimination and difficulty).
    nodes, weights = hermgauss(41)
    grid = np.sqrt(2) * nodes
    log_prior = np.log(weights / np.sqrt(np.pi))
    initial_b = [np.log(counts[item][0] / counts[item][1]) for item in items]

    def objective(params):
        log_a, b = np.split(params, 2)
        a = np.exp(log_a)
        logits = a[:, None] * (grid - b[:, None])
        log_joint = posterior_log_joint(right, wrong, logits, log_prior)
        marginal = logsumexp(log_joint, axis=1, keepdims=True)
        posterior = np.exp(log_joint - marginal)
        residual = right.T @ posterior - expit(logits) * (observed.T @ posterior)
        penalty = 0.5 * np.sum(log_a**2 + (b / 3)**2)
        grad_a = -(residual * logits).sum(axis=1) + log_a
        grad_b = a * residual.sum(axis=1) + b / 9
        return -marginal.sum() + penalty, np.concatenate([grad_a, grad_b])

    result = minimize(
        objective, np.r_[np.zeros(len(items)), np.clip(initial_b, -6, 6)], jac=True,
        method="L-BFGS-B", bounds=[(np.log(0.1), np.log(5))] * len(items) + [(-6, 6)] * len(items),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"2PL calibration failed: {result.message}")
    log_a, b = np.split(result.x, 2)
    a = np.exp(log_a)
    parameters = pd.DataFrame({
        "item_id": items, "a": a, "b": b, "n_responses": [sum(counts[i]) for i in items],
    })
    return item_index, parameters, a[:, None] * (grid - b[:, None]), grid, log_prior, result.nit


def score(rows, item_index, logits, grid, log_prior, min_items):
    uids, right, wrong, n = response_matrices(rows, item_index)
    keep = n >= min_items
    log_joint = posterior_log_joint(right[keep], wrong[keep], logits, log_prior)
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    theta = posterior @ grid
    return pd.DataFrame({
        "student_id": np.asarray(uids, dtype=str)[keep],
        "theta": theta,
        "theta_se": np.sqrt(np.maximum(0, posterior @ (grid**2) - theta**2)),
        "n_responses": n[keep],
    })


def assign_groups(students, cuts):
    band = np.searchsorted(cuts, students["theta"], side="left")
    students["cluster"] = np.asarray(GROUPS)[band]


def evaluate_predictions(path, chunks, students, prefix_length, output):
    rows = []
    with path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != len(chunks):
        raise ValueError(f"{path}: {len(lines)} prediction lines for {len(chunks)} test chunks")
    for line_no, (line, chunk) in enumerate(zip(lines, chunks), 1):
        uid, context_items, context_labels, target_items, expected_labels, positions = chunk
        # The long CSV alone cannot identify repeated sequence chunks. Verify
        # both context and targets against the original pyKT prediction file.
        try:
            saved = ast.literal_eval(line.replace("nan", "None"))
            labels, scores = saved[3:5]
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid pyKT prediction row") from exc
        if (list(saved[:3]) != [context_items, context_labels, target_items]
                or labels != expected_labels or not isinstance(scores, (list, tuple))
                or len(scores) != len(positions)):
            raise ValueError(f"{path}:{line_no}: prediction context/targets do not match the test chunk")
        for (position, index), concept, label, pred in zip(positions, target_items, labels, scores):
            if position >= prefix_length:
                if pred is None or not np.isfinite(pred) or not 0 <= pred <= 1:
                    raise ValueError(f"{path}:{line_no}: invalid probability")
                rows.append((uid, index, concept, label, pred))
    predictions = pd.DataFrame(rows, columns=["student_id", "interaction_id", "concept", "label", "pred"])
    joined = predictions.merge(students[["student_id", "cluster"]], validate="many_to_one")
    matrix, weights, counts, summary = compute_auc_matrix_by_cluster(joined, cluster_order=GROUPS)
    summary.update({
        "prediction_source": str(path.resolve()),
        "verified_prediction_chunks": len(chunks),
        "n_suffix_predictions_before_scoring_filter": len(predictions),
        "n_students": joined["student_id"].nunique(),
    })
    groups = joined.assign(error=joined.pred - joined.label, squared_error=(joined.pred - joined.label)**2)
    metrics = groups.groupby("cluster").agg(
        n_students=("student_id", "nunique"), n_predictions=("pred", "size"),
        observed_correct_rate=("label", "mean"), mean_prediction=("pred", "mean"),
        calibration_bias=("error", "mean"), brier_score=("squared_error", "mean"),
    ).reindex(GROUPS)
    # Give each shared concept equal weight in every band. Require support in
    # all three bands so rare concepts do not dominate this descriptive check.
    by_concept = groups.groupby(["concept", "cluster"]).agg(
        n=("label", "size"), correct_rate=("label", "mean"), bias=("error", "mean"),
    )
    common = by_concept["n"].unstack().reindex(columns=GROUPS).ge(20).all(axis=1)
    shared = by_concept.loc[by_concept.index.get_level_values("concept").isin(common[common].index)]
    metrics["common_concept_correct_rate"] = shared.groupby("cluster")["correct_rate"].mean()
    metrics["common_concept_calibration_bias"] = shared.groupby("cluster")["bias"].mean()
    summary["n_common_concepts"] = int(common.sum())
    summary["min_suffix_responses_per_group_for_common_concept"] = 20
    output.mkdir()
    matrix.to_csv(output / "auc_matrix.csv")
    weights.to_csv(output / "auc_weights.csv")
    counts.to_csv(output / "auc_counts.csv", index=False)
    metrics.to_csv(output / "group_metrics.csv")
    (output / "auc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, _ = plot_auc_matrix(matrix, output_fig=str(output / "auc_matrix.png"), title="History IRT tertiles")
    fig.clear()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True, help="Unchunked pyKT train_valid.csv")
    parser.add_argument("--test-csv", type=Path, required=True, help="Non-overlapping pyKT test_sequences.csv")
    parser.add_argument("--item-col", choices=["questions", "concepts"], required=True)
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--reference-fold", default="0")
    parser.add_argument("--deduplicate", choices=["first", "last"], default="first")
    parser.add_argument("--min-items", type=int, default=3)
    parser.add_argument("--min-item-responses", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites a run")
    parser.add_argument("--prediction", nargs=2, action="append", metavar=("NAME", "TXT"), default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if min(args.prefix_length, args.min_items, args.min_item_responses) < 1:
        raise ValueError("Prefix length and response requirements must be positive.")
    names = [name for name, _ in args.prediction]
    if len(set(names)) != len(names) or any(not name.isascii() or not name.replace("_", "").isalnum() for name in names):
        raise ValueError("Use distinct alphanumeric prediction names (underscores allowed).")
    train, folds, _ = read_sequences(args.train_csv, args.item_col)
    test, _, chunks = read_sequences(args.test_csv, args.item_col, test=True)
    if train.keys() & test.keys():
        raise ValueError("Training/reference and test student IDs overlap.")

    calibration_ids = {uid for uid in train if folds[uid] != args.reference_fold}
    reference_ids = train.keys() - calibration_ids
    train_rows = histories(train, args.prefix_length, args.deduplicate, require_future=True)
    test_rows = histories(test, args.prefix_length, args.deduplicate, require_future=True)
    calibration = {uid: row for uid, row in train_rows.items() if uid in calibration_ids}
    reference = {uid: row for uid, row in train_rows.items() if uid in reference_ids}
    item_index, parameters, logits, grid, log_prior, iterations = calibrate(calibration, args.min_item_responses)
    ref_scores = score(reference, item_index, logits, grid, log_prior, args.min_items)
    test_scores = score(test_rows, item_index, logits, grid, log_prior, args.min_items)
    if len(ref_scores) < 3 or test_scores.empty:
        raise ValueError("Too few students with enough scored prefix items.")
    cuts = np.quantile(ref_scores["theta"], [1/3, 2/3])
    if cuts[0] >= cuts[1]:
        raise ValueError("Reference tertile cuts coincide; the data do not support three bands.")
    assign_groups(ref_scores, cuts)
    assign_groups(test_scores, cuts)

    args.output_dir.mkdir(parents=True)
    test_scores.to_csv(args.output_dir / "student_clusters_irt_tertiles.csv", index=False)
    ref_scores.to_csv(args.output_dir / "reference_scores.csv", index=False)
    parameters.to_csv(args.output_dir / "item_parameters.csv", index=False)
    summary = {
        "method": "2PL EAP on history prefixes; fixed reference-sample tertiles",
        "train_csv": str(args.train_csv.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "reference_fold": args.reference_fold,
        "prefix_length": args.prefix_length,
        "deduplicate": args.deduplicate,
        "item_col": args.item_col,
        "min_items": args.min_items,
        "min_item_responses": args.min_item_responses,
        "n_calibration_students": len(calibration),
        "n_items": len(item_index),
        "calibration_converged": True,
        "calibration_iterations": iterations,
        "cuts": cuts.tolist(),
    }
    for name, scores, total, prefix_eligible in (
        ("reference", ref_scores, len(reference_ids), len(reference)),
        ("test", test_scores, len(test), len(test_rows)),
    ):
        summary[name] = {
            "n_students_total": total, "n_with_prefix_and_suffix": prefix_eligible,
            "n_scored": len(scores), "median_theta_se": float(scores.theta_se.median()),
            "median_scored_items": float(scores.n_responses.median()),
            "group_counts": scores["cluster"].value_counts().reindex(GROUPS, fill_value=0).to_dict(),
        }
    summary["predictions"] = {
        name: evaluate_predictions(Path(path), chunks, test_scores, args.prefix_length, args.output_dir / name)
        for name, path in args.prediction
    }
    (args.output_dir / "cluster_summary_irt_tertiles.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
