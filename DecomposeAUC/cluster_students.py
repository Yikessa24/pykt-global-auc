"""
Post-hoc student clustering with IRT and person-fit statistics.

This module is fully decoupled from pykt training code and can be run after any
KT model finishes training. It only needs real student-item binary responses.

Inputs:
- DataFrame with at least: student_id, item_id, correct

Outputs:
- Student-level DataFrame/CSV with columns:
  student_id, theta, outfit, infit, n_responses, cluster
  where cluster in {"bad", "mid_stable", "mid_chaotic", "good"}
- Optional item parameter CSV for auditing (a, b, c, n_responses)

Dependencies:
- pandas, numpy, scipy

Example:
python cluster_students.py \
  --input-csv ../data/assist2015/train_valid.csv \
    --input-format pykt_sequence \
    --uid-col uid --concepts-col concepts --responses-col responses \
  --output-csv ./assist2015_student_clusters.csv \
    --student-col student_id --item-col item_id --correct-col correct \
  --irt-model 2PL --fit-stat outfit --fit-threshold 1.2
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


EPS = 1e-9


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(np.clip(x, -35.0, 35.0))


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _expand_pykt_sequence(
    df: pd.DataFrame,
    uid_col: str,
    concepts_col: str,
    responses_col: str,
    student_col: str,
    item_col: str,
    correct_col: str,
) -> pd.DataFrame:
    """
    Expand pykt sequence-style rows into long-format responses.

    Input row example:
        uid=16894,
        concepts="19,19,19,...",
        responses="0,1,0,..."
    """
    _validate_columns(df, [uid_col, concepts_col, responses_col])

    rows = []
    for _, row in df[[uid_col, concepts_col, responses_col]].iterrows():
        uid = str(row[uid_col])
        concepts = str(row[concepts_col]).split(",")
        responses = str(row[responses_col]).split(",")

        if len(concepts) != len(responses):
            k = min(len(concepts), len(responses))
            concepts = concepts[:k]
            responses = responses[:k]

        for concept, answer in zip(concepts, responses):
            concept = concept.strip()
            answer = answer.strip()

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

    if min_student_responses > 1:
        stu_counts = work.groupby(student_col)[correct_col].size()
        valid_students = stu_counts[stu_counts >= min_student_responses].index
        work = work[work[student_col].isin(valid_students)]

    if min_item_responses > 1:
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


def _initialize_theta(responses: np.ndarray) -> np.ndarray:
    raw = np.nanmean(responses, axis=1)
    raw = np.where(np.isnan(raw), 0.5, raw)
    theta = (raw - np.mean(raw)) / (np.std(raw) + EPS)
    return np.clip(theta, -2.0, 2.0)


def _fit_item_2pl(theta: np.ndarray, x: np.ndarray, a0: float, b0: float) -> Tuple[float, float]:
    def objective(params: np.ndarray) -> float:
        log_a, b = params
        a = np.exp(log_a)
        p = _sigmoid(a * (theta - b))
        nll = -np.sum(x * np.log(p + EPS) + (1.0 - x) * np.log(1.0 - p + EPS))
        reg = 1e-3 * (log_a ** 2 + b ** 2)
        return float(nll + reg)

    init = np.array([np.log(max(a0, 0.1)), float(b0)], dtype=np.float64)
    res = minimize(objective, init, method="L-BFGS-B")
    if not res.success:
        return float(max(a0, 0.1)), float(b0)

    log_a, b = res.x
    a = float(np.exp(log_a))
    return float(np.clip(a, 0.1, 10.0)), float(np.clip(b, -6.0, 6.0))


def _fit_item_3pl(
    theta: np.ndarray,
    x: np.ndarray,
    a0: float,
    b0: float,
    c0: float,
    c_upper: float,
) -> Tuple[float, float, float]:
    def objective(params: np.ndarray) -> float:
        log_a, b, logit_c = params
        a = np.exp(log_a)
        c = c_upper * expit(logit_c)
        p = c + (1.0 - c) * _sigmoid(a * (theta - b))
        nll = -np.sum(x * np.log(p + EPS) + (1.0 - x) * np.log(1.0 - p + EPS))
        reg = 1e-3 * (log_a ** 2 + b ** 2 + logit_c ** 2)
        return float(nll + reg)

    c0 = float(np.clip(c0, 1e-4, c_upper - 1e-4))
    init = np.array([np.log(max(a0, 0.1)), float(b0), np.log(c0 / (c_upper - c0))], dtype=np.float64)
    bounds = [(-2.5, 2.5), (-6.0, 6.0), (-8.0, 8.0)]
    res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return float(max(a0, 0.1)), float(b0), float(c0)

    log_a, b, logit_c = res.x
    a = float(np.exp(log_a))
    c = float(c_upper * expit(logit_c))
    return float(np.clip(a, 0.1, 10.0)), float(np.clip(b, -6.0, 6.0)), float(np.clip(c, 1e-4, c_upper - 1e-4))


def _update_theta_2pl(
    responses: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    prior_sd: float,
    newton_steps: int = 8,
) -> np.ndarray:
    updated = theta.copy()
    prior_var = prior_sd ** 2

    for s in range(responses.shape[0]):
        mask = ~np.isnan(responses[s])
        if not np.any(mask):
            continue

        x = responses[s, mask]
        ai = a[mask]
        bi = b[mask]
        t = float(updated[s])

        for _ in range(newton_steps):
            p = _sigmoid(ai * (t - bi))
            grad = np.sum(ai * (x - p)) - t / prior_var
            hess = -np.sum((ai ** 2) * p * (1.0 - p)) - 1.0 / prior_var
            step = grad / (hess - EPS)
            t_new = np.clip(t - step, -6.0, 6.0)
            if abs(t_new - t) < 1e-4:
                t = t_new
                break
            t = t_new

        updated[s] = t

    return updated


def _update_theta_3pl(
    responses: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    prior_sd: float,
) -> np.ndarray:
    updated = theta.copy()
    prior_var = prior_sd ** 2

    for s in range(responses.shape[0]):
        mask = ~np.isnan(responses[s])
        if not np.any(mask):
            continue

        x = responses[s, mask]
        ai = a[mask]
        bi = b[mask]
        ci = c[mask]

        def objective(t: float) -> float:
            p = ci + (1.0 - ci) * _sigmoid(ai * (t - bi))
            nll = -np.sum(x * np.log(p + EPS) + (1.0 - x) * np.log(1.0 - p + EPS))
            prior = 0.5 * (t ** 2) / prior_var
            return float(nll + prior)

        res = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
        updated[s] = float(res.x if res.success else updated[s])

    return updated


def fit_irt(
    responses: np.ndarray,
    model: str = "2PL",
    max_iter: int = 20,
    tol: float = 1e-3,
    prior_sd: float = 1.0,
    c_upper: float = 0.35,
    random_state: int = 3407,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    if model not in {"2PL", "3PL"}:
        raise ValueError("model must be '2PL' or '3PL'")

    rng = np.random.default_rng(random_state)
    n_students, n_items = responses.shape

    theta = _initialize_theta(responses)
    a = np.ones(n_items, dtype=np.float64)

    item_mean = np.nanmean(responses, axis=0)
    item_mean = np.where(np.isnan(item_mean), 0.5, np.clip(item_mean, 1e-3, 1 - 1e-3))
    b = -np.log(item_mean / (1.0 - item_mean))
    b = np.clip(b, -3.0, 3.0)

    if model == "3PL":
        c = np.full(n_items, 0.1, dtype=np.float64)
    else:
        c = np.zeros(n_items, dtype=np.float64)

    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        theta_prev = theta.copy()

        for j in range(n_items):
            mask = ~np.isnan(responses[:, j])
            if np.sum(mask) < 3:
                continue

            x = responses[mask, j]
            th = theta[mask]
            if model == "2PL":
                a[j], b[j] = _fit_item_2pl(th, x, a[j], b[j])
            else:
                a[j], b[j], c[j] = _fit_item_3pl(th, x, a[j], b[j], c[j], c_upper)

        if model == "2PL":
            theta = _update_theta_2pl(responses, theta, a, b, prior_sd=prior_sd)
        else:
            theta = _update_theta_3pl(responses, theta, a, b, c, prior_sd=prior_sd)

        mu = float(np.mean(theta))
        sigma = float(np.std(theta) + EPS)
        theta = (theta - mu) / sigma
        a = a * sigma
        b = (b - mu) / sigma

        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, -6.0, 6.0)

        max_change = float(np.max(np.abs(theta - theta_prev)))
        if verbose:
            print(f"[IRT] iter={it + 1}, max_theta_change={max_change:.6f}")

        if max_change < tol:
            converged = True
            break

        if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            noise = rng.normal(0.0, 0.01, size=theta.shape)
            theta = np.nan_to_num(theta, nan=0.0) + noise
            a = np.nan_to_num(a, nan=1.0)
            b = np.nan_to_num(b, nan=0.0)

    return {
        "theta": theta,
        "a": a,
        "b": b,
        "c": c,
        "converged": np.array([int(converged)], dtype=int),
        "iterations": np.array([iterations], dtype=int),
    }


def compute_person_fit(
    responses: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if c is None:
        c = np.zeros_like(a)

    outfit = np.full(responses.shape[0], np.nan, dtype=np.float64)
    infit = np.full(responses.shape[0], np.nan, dtype=np.float64)
    n_resp = np.sum(~np.isnan(responses), axis=1).astype(int)

    for s in range(responses.shape[0]):
        mask = ~np.isnan(responses[s])
        if not np.any(mask):
            continue

        x = responses[s, mask]
        ai = a[mask]
        bi = b[mask]
        ci = c[mask]

        p = ci + (1.0 - ci) * _sigmoid(ai * (theta[s] - bi))
        v = np.clip(p * (1.0 - p), EPS, None)
        residual2 = (x - p) ** 2

        outfit[s] = float(np.mean(residual2 / v))
        infit[s] = float(np.sum(residual2) / np.sum(v))

    return outfit, infit, n_resp


def assign_clusters(
    theta: np.ndarray,
    outfit: np.ndarray,
    infit: np.ndarray,
    bad_theta_threshold: float = -0.5,
    good_theta_threshold: float = 0.5,
    fit_stat: str = "outfit",
    fit_threshold: float = 1.2,
    fit_quantile: Optional[float] = None,
) -> np.ndarray:
    if fit_stat not in {"outfit", "infit"}:
        raise ValueError("fit_stat must be 'outfit' or 'infit'")

    fit_values = outfit if fit_stat == "outfit" else infit
    clusters = np.empty(theta.shape[0], dtype=object)

    bad_mask = theta <= bad_theta_threshold
    good_mask = theta > good_theta_threshold
    mid_mask = ~(bad_mask | good_mask)

    clusters[bad_mask] = "bad"
    clusters[good_mask] = "good"

    mid_fit = fit_values[mid_mask]
    if fit_quantile is not None:
        valid_mid_fit = mid_fit[np.isfinite(mid_fit)]
        if valid_mid_fit.size > 0:
            fit_threshold = float(np.quantile(valid_mid_fit, fit_quantile))

    for idx in np.where(mid_mask)[0]:
        fv = fit_values[idx]
        if not np.isfinite(fv):
            clusters[idx] = "mid_stable"
        elif fv > fit_threshold:
            clusters[idx] = "mid_chaotic"
        else:
            clusters[idx] = "mid_stable"

    return clusters


def cluster_students_with_irt(
    responses_df: pd.DataFrame,
    student_col: str = "student_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    input_format: str = "long",
    uid_col: str = "uid",
    concepts_col: str = "concepts",
    responses_col: str = "responses",
    irt_model: str = "2PL",
    fit_stat: str = "outfit",
    bad_theta_threshold: float = -0.5,
    good_theta_threshold: float = 0.5,
    fit_threshold: float = 1.2,
    fit_quantile: Optional[float] = None,
    deduplicate: str = "last",
    correct_threshold: float = 0.5,
    min_student_responses: int = 3,
    min_item_responses: int = 3,
    max_iter: int = 20,
    tol: float = 1e-3,
    prior_sd: float = 1.0,
    c_upper: float = 0.35,
    random_state: int = 3407,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if input_format == "pykt_sequence":
        work_df = _expand_pykt_sequence(
            responses_df,
            uid_col=uid_col,
            concepts_col=concepts_col,
            responses_col=responses_col,
            student_col=student_col,
            item_col=item_col,
            correct_col=correct_col,
        )
    elif input_format == "long":
        work_df = responses_df
    else:
        raise ValueError("input_format must be 'long' or 'pykt_sequence'")

    _validate_columns(work_df, [student_col, item_col, correct_col])

    prepped = _prepare_responses(
        work_df,
        student_col=student_col,
        item_col=item_col,
        correct_col=correct_col,
        deduplicate=deduplicate,
        correct_threshold=correct_threshold,
        min_student_responses=min_student_responses,
        min_item_responses=min_item_responses,
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

    theta = irt_res["theta"]
    a = irt_res["a"]
    b = irt_res["b"]
    c = irt_res["c"]

    outfit, infit, n_resp = compute_person_fit(response_matrix, theta, a, b, c)

    clusters = assign_clusters(
        theta=theta,
        outfit=outfit,
        infit=infit,
        bad_theta_threshold=bad_theta_threshold,
        good_theta_threshold=good_theta_threshold,
        fit_stat=fit_stat,
        fit_threshold=fit_threshold,
        fit_quantile=fit_quantile,
    )

    student_cluster_df = pd.DataFrame(
        {
            student_col: student_values,
            "theta": theta,
            "outfit": outfit,
            "infit": infit,
            "n_responses": n_resp,
            "cluster": clusters,
        }
    )

    item_params_df = pd.DataFrame(
        {
            item_col: item_values,
            "a": a,
            "b": b,
            "c": c,
            "n_responses": item_n,
        }
    )

    return student_cluster_df, item_params_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster students with IRT theta + person-fit.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--export-item-params", type=str, default="")

    parser.add_argument("--input-format", type=str, default="long", choices=["long", "pykt_sequence"])

    parser.add_argument("--student-col", type=str, default="student_id")
    parser.add_argument("--item-col", type=str, default="item_id")
    parser.add_argument("--correct-col", type=str, default="correct")
    parser.add_argument("--uid-col", type=str, default="uid")
    parser.add_argument("--concepts-col", type=str, default="concepts")
    parser.add_argument("--responses-col", type=str, default="responses")

    parser.add_argument("--irt-model", type=str, default="2PL", choices=["2PL", "3PL"])
    parser.add_argument("--fit-stat", type=str, default="outfit", choices=["outfit", "infit"])

    parser.add_argument("--bad-theta-threshold", type=float, default=-0.5)
    parser.add_argument("--good-theta-threshold", type=float, default=0.5)
    parser.add_argument("--fit-threshold", type=float, default=1.2)
    parser.add_argument("--fit-quantile", type=float, default=None)

    parser.add_argument("--deduplicate", type=str, default="last", choices=["first", "last", "mean"])
    parser.add_argument("--correct-threshold", type=float, default=0.5)

    parser.add_argument("--min-student-responses", type=int, default=3)
    parser.add_argument("--min-item-responses", type=int, default=3)

    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--prior-sd", type=float, default=1.0)
    parser.add_argument("--c-upper", type=float, default=0.35)
    parser.add_argument("--random-state", type=int, default=3407)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    student_cluster_df, item_params_df = cluster_students_with_irt(
        responses_df=df,
        student_col=args.student_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        input_format=args.input_format,
        uid_col=args.uid_col,
        concepts_col=args.concepts_col,
        responses_col=args.responses_col,
        irt_model=args.irt_model,
        fit_stat=args.fit_stat,
        bad_theta_threshold=args.bad_theta_threshold,
        good_theta_threshold=args.good_theta_threshold,
        fit_threshold=args.fit_threshold,
        fit_quantile=args.fit_quantile,
        deduplicate=args.deduplicate,
        correct_threshold=args.correct_threshold,
        min_student_responses=args.min_student_responses,
        min_item_responses=args.min_item_responses,
        max_iter=args.max_iter,
        tol=args.tol,
        prior_sd=args.prior_sd,
        c_upper=args.c_upper,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    student_cluster_df.to_csv(args.output_csv, index=False)
    print(f"Saved student clusters to: {args.output_csv}")

    if args.export_item_params:
        item_params_df.to_csv(args.export_item_params, index=False)
        print(f"Saved item parameters to: {args.export_item_params}")


if __name__ == "__main__":
    main()
