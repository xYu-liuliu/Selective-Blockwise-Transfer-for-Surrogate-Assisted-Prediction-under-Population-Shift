#!/usr/bin/env python3
"""Recompute SUPTRANS from the saved per-repeat representations.

WHY
---
`fit_suptrans` in the main pipeline stops after steps 1-2 of Li, Cai & Duan
(2023): a Lasso GLM on the source labelled rows, then a sparse correction on
the target labelled rows around that fixed source offset. The final estimator
is therefore anchored at the source fit; the source and target labelled rows
are never used *together* in one joint fit. The pipeline's fit has neither
step 3 of their procedure, the joint penalised refit over SL and TL with the
estimated difference offset out of the source rows, nor the hard thresholding
of delta.

This script reads the representations each repeat already wrote to disk and
recomputes SUPTRANS following equations (1)-(3) of Li, Cai & Duan (2023). The
three-step estimator is the SUPTRANS the paper reports, the "three-step
transferring procedure" that Cai, Li & Liu (2025, Sec. 4) use as a benchmark.
It reads the image block X alone: neither the surrogates nor the unlabelled
rows enter.

    Step 1  w_hat   = argmin_b  (1/n_S) sum_{i in SL} l(y_i, a + x_i'b) + lam_w ||b||_1
    Step 2  d_hat   = argmin_d  (1/n_T) sum_{i in TL} l(y_i, (a_w + d_0) + x_i'(w_hat + d))
                                 + lam_d ||d||_1
            d_check = H_k(d_hat),  k = floor( sqrt(n_T / log p) )
    Step 3  beta    = argmin_b  (1/N) [ sum_{i in TL} l(y_i, b_0 + x_i'b)
                                      + sum_{i in SL} l(y_i, b_0 + x_i'b - (d_0 + x_i'd_check)) ]
                                 + lam_b ||b||_1,     N = n_S + n_T

Sign convention: step 2 fits the target loss at (w_hat + d), so d_hat estimates
beta_target - beta_source. Source rows in step 3 are therefore evaluated at
b - d_check, which is equation (3) of Li et al. as written.

Tuning, as in the paper: step 1 uses the pipeline's C grid and five-fold
stratified CV; step 2 the correction grid lambda_max * {1, .5, .2, .1, .05,
.02, .01}; step 3 the same lambda_max-relative grid, chosen by five-fold CV
whose strata combine population and outcome, with the pooled loss rather than
the target-only loss as the criterion.

Optionally the aggregation step of their Section 2.2 is applied: a binary eta
in {e_1, e_2} choosing between the three-step estimator and a target-only
Lasso, selected by cross-fitted target log loss on TL. The paper reports the
three-step estimator as SUPTRANS; the aggregated one is stored as a separate
endpoint (suptrans_agg) and not reported.

WHAT IT REUSES
--------------
The script imports the main pipeline module by path and reuses its solvers,
its Lasso tuning grids and its `evaluate_predictions`, so the recomputed
numbers are directly comparable with `raw_results.csv`. Nothing is refit that
does not have to be: the image block of every branch (after its encoder and
PCA) comes from the saved `.npy` files. SUPTRANS reads X alone, so the
surrogate block is not loaded.

INPUTS  (all already on disk from the main run; <result_dir> and <prep_dir>
default to the pipeline's RESULT_DIR and PREP_DIR, see project_paths.py)
    <result_dir>/repeat_features/rep{r}_nT{NT}/X_SL_{simclr,biovil,no_ssl}.npy
    <result_dir>/repeat_features/rep{r}_nT{NT}/X_TARGET_{simclr,biovil,no_ssl}.npy
    <result_dir>/selected_indices/repeat{r}_nT{NT}_patient_split.npz
    <prep_dir>/SL.npz         (only the Y array is read)
    <prep_dir>/TARGET.npz     (only the Y array is read)
    <prep_dir>/row_audit.csv  (patient ids for the clustered bootstrap)

OUTPUTS  (<out_dir> defaults to <result_dir>/suptrans_recompute)
    <out_dir>/suptrans_three_step_results.csv     metric rows, raw_results schema
    <out_dir>/suptrans_three_step_fit_info.csv    per-fit diagnostics
    <out_dir>/suptrans_three_step_summary.csv     mean AUC over repeats
    <out_dir>/suptrans_old_vs_new_paired.csv      three-step against the stored fit
    <result_dir>/raw_results_with_suptrans3.csv   optional merge, with --merge-raw

USAGE
    # the pipeline next to this script, with its RESULT_DIR and PREP_DIR
    python -u recompute_suptrans_three_step.py

    # a quick smoke test on one repeat and one branch
    python -u recompute_suptrans_three_step.py --repeats 0 --branches ssl

    # skip the extra aggregation endpoint (roughly halves the runtime)
    python -u recompute_suptrans_three_step.py --no-aggregation
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold


# The image-block file suffix each branch was saved under.
BRANCH_FEATURE_SUFFIX = {"ssl": "simclr", "biovil": "biovil", "no_ssl": "no_ssl"}

NEW_ENDPOINT = "suptrans"                 # three-step, Li et al. (2023) (1)-(3)
AGG_ENDPOINT = "suptrans_agg"             # + aggregation step of Section 2.2
OLD_ENDPOINT = "suptrans_2step_recomputed"  # the pipeline's two-step fit, rebuilt
TGT_ENDPOINT = "suptrans_target_only_lasso"  # the aggregation's other candidate


# ============================================================
# Pipeline import
# ============================================================

def import_pipeline(path):
    """Import the main pipeline module so every solver and grid is shared."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline script not found: {path}\n"
            "Pass the correct path with --pipeline."
        )
    spec = importlib.util.spec_from_file_location("strifle_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strifle_pipeline"] = module
    spec.loader.exec_module(module)          # module level only prints and mkdirs
    required = [
        "fit_l1_logistic_cv", "fit_l1_bias_correction_cv",
        "_fit_offset_logistic_l1", "_weighted_log_loss_from_logits",
        "evaluate_predictions", "seed_everything",
        "STRIFLE_TUNING_FOLDS", "STRIFLE_CORRECTION_LAMBDA_RATIOS",
        "STRIFLE_CORRECTION_MAX_ITER", "STRIFLE_CORRECTION_TOL",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{path} does not expose: {missing}")
    return module


# ============================================================
# Loading the artefacts of a finished run
# ============================================================

def load_labels_only(npz_path, split_name):
    """Read the Y array out of a prepared NPZ without touching the images."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "Y" not in data.files:
            raise KeyError(f"{npz_path} has no Y array (split {split_name}).")
        return data["Y"].astype(np.float32)


def load_target_patient_ids(prep_dir, n_target):
    audit = pd.read_csv(os.path.join(prep_dir, "row_audit.csv"))
    if "patient_id_parsed" not in audit.columns:
        raise KeyError("row_audit.csv must contain patient_id_parsed.")
    block = audit[audit["split_group"] == "TARGET"].reset_index(drop=True)
    if len(block) != n_target:
        raise ValueError(
            f"row_audit TARGET rows ({len(block)}) != TARGET.npz rows ({n_target})."
        )
    return block["patient_id_parsed"].astype(str).to_numpy()


def discover_repeats(feature_dir, nT):
    reps = []
    if not os.path.isdir(feature_dir):
        return reps
    for name in os.listdir(feature_dir):
        if name.startswith("rep") and name.endswith(f"_nT{nT}"):
            try:
                reps.append(int(name[3:].split("_nT")[0]))
            except ValueError:
                continue
    return sorted(reps)


def load_repeat(result_dir, feature_dir, nT, rep, branch):
    """Return (X_SL, X_TARGET) for one repeat and branch, plus the split."""
    suffix = BRANCH_FEATURE_SUFFIX[branch]
    run_key = f"rep{rep}_nT{nT}"
    fdir = os.path.join(feature_dir, run_key)
    x_sl_path = os.path.join(fdir, f"X_SL_{suffix}.npy")
    x_tg_path = os.path.join(fdir, f"X_TARGET_{suffix}.npy")
    for path in (x_sl_path, x_tg_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    X_SL = np.load(x_sl_path).astype(np.float64)
    X_TARGET = np.load(x_tg_path).astype(np.float64)
    return X_SL, X_TARGET


def load_split(result_dir, nT, rep):
    path = os.path.join(result_dir, "selected_indices",
                        f"repeat{rep}_nT{nT}_patient_split.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        out = {
            "tl_idx": data["tl_idx"].astype(np.int64),
            "tu_idx": data["tu_idx"].astype(np.int64),
            "valid_idx": data["valid_idx"].astype(np.int64),
            "tl_labels": data["tl_labels"].astype(np.int64),
            "valid_labels": data["valid_labels"].astype(np.int64),
            "seed": int(data["seed"]) if "seed" in data.files else None,
        }
    return out


# ============================================================
# The three-step estimator
# ============================================================

def hard_threshold(vector, k):
    """H_k: keep the k largest entries in magnitude, zero the rest."""
    vector = np.asarray(vector, dtype=np.float64)
    k = int(max(0, k))
    if k >= int(np.count_nonzero(vector)):
        return vector.copy()
    out = np.zeros_like(vector)
    if k > 0:
        keep = np.argsort(np.abs(vector))[::-1][:k]
        out[keep] = vector[keep]
    return out


def _intercept_only_offset_fit(offset, y):
    """Unpenalised intercept for a logistic model with a fixed offset."""
    offset = np.asarray(offset, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a = 0.0
    for _ in range(200):
        p = expit(offset + a)
        grad = float(np.mean(p - y))
        hess = max(float(np.mean(p * (1.0 - p))), 1e-10)
        step = np.clip(grad / hess, -5.0, 5.0)
        a -= step
        if abs(step) < 1e-12:
            break
    return float(a)


def pooled_offset_lasso_cv(P, Z, y, offset, is_target, seed, label,
                           cv_target_only=False, verbose=True):
    """Step 3: L1 logistic on the pooled SL+TL rows with a per-row offset.

    The loss is the unweighted mean over the pooled rows, which is the
    (1/N) sum of equation (3). The penalty is chosen from lambda_max times
    STRIFLE_CORRECTION_LAMBDA_RATIOS, lambda_max being the smallest penalty
    that zeroes every slope at the refitted intercept, by stratified CV whose
    strata combine the outcome and the domain, so every fold keeps target
    rows. The validation criterion is the pooled loss (cv_target_only=False),
    as in the paper; True evaluates it on the target rows only, which matches
    the estimand but not the fitting loss.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    offset = np.asarray(offset, dtype=np.float64).ravel()
    is_target = np.asarray(is_target, dtype=bool).ravel()
    n = len(y)

    b0 = _intercept_only_offset_fit(offset, y)
    lambda_max = max(
        float(np.max(np.abs(Z.T @ (expit(offset + b0) - y))) / n), 1e-6
    )
    lambdas = lambda_max * np.asarray(P.STRIFLE_CORRECTION_LAMBDA_RATIOS,
                                      dtype=np.float64)

    strata = is_target.astype(int) * 2 + y
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < 2:                       # fall back to plain y strata
        strata = y
        _, counts = np.unique(strata, return_counts=True)
    n_splits = int(min(int(P.STRIFLE_TUNING_FOLDS), int(counts.min())))
    n_splits = max(n_splits, 2)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    fold_losses = np.full((n_splits, len(lambdas)), np.nan, dtype=np.float64)
    for fold_id, (tr, va) in enumerate(cv.split(Z, strata)):
        weight = is_target[va].astype(np.float64) if cv_target_only else None
        if weight is not None and weight.sum() <= 0:
            continue
        for lam_id, lam in enumerate(lambdas):
            a, coef, _, _ = P._fit_offset_logistic_l1(
                Z[tr], y[tr], offset[tr], lam,
                P.STRIFLE_CORRECTION_MAX_ITER, P.STRIFLE_CORRECTION_TOL,
            )
            fold_losses[fold_id, lam_id] = P._weighted_log_loss_from_logits(
                y[va], offset[va] + a + Z[va] @ coef, weight
            )

    mean_losses = np.nanmean(fold_losses, axis=0)
    best_id = int(np.nanargmin(mean_losses))
    best_lambda = float(lambdas[best_id])
    intercept, beta, objective, iterations = P._fit_offset_logistic_l1(
        Z, y, offset, best_lambda,
        P.STRIFLE_CORRECTION_MAX_ITER, P.STRIFLE_CORRECTION_TOL,
    )
    if verbose:
        print(f"    [{label}] pooled L1 CV folds={n_splits}, "
              f"lambda={best_lambda:.4g}, CV-logloss={mean_losses[best_id]:.6f}, "
              f"nonzero={int(np.count_nonzero(beta))}/{Z.shape[1]}, "
              f"iterations={iterations}")
    return {
        "intercept": float(intercept),
        "beta": beta,
        "lambda": best_lambda,
        "cv_loss": float(mean_losses[best_id]),
        "nonzero": int(np.count_nonzero(beta)),
        "objective": float(objective),
        "n_pooled": int(n),
    }


def fit_step1_source(P, X_SL, Y_SL, seed, label, verbose=True):
    """Step 1: Lasso GLM on the source labelled rows alone (equation 1)."""
    model, best_C, _ = P.fit_l1_logistic_cv(
        np.asarray(X_SL, dtype=np.float64), np.asarray(Y_SL, dtype=int).ravel(),
        int(seed), label=f"{label}-step1-w",
    )
    return {
        "intercept": float(model.intercept_[0]),
        "w": model.coef_[0].astype(np.float64),
        "C": float(best_C),
        "nonzero": int(np.count_nonzero(model.coef_)),
    }


def fit_suptrans_three_step(P, X_SL, Y_SL, X_TL, Y_TL, seed, label,
                            step1=None, hard_thresh=True,
                            cv_target_only=False, verbose=True):
    """Equations (1)-(3) of Li, Cai & Duan (2023) with a single source study."""
    X_SL = np.asarray(X_SL, dtype=np.float64)
    X_TL = np.asarray(X_TL, dtype=np.float64)
    Y_SL = np.asarray(Y_SL, dtype=int).ravel()
    Y_TL = np.asarray(Y_TL, dtype=int).ravel()
    n_T, p = X_TL.shape

    # ---- Step 1: source only ------------------------------------------
    if step1 is None:
        step1 = fit_step1_source(P, X_SL, Y_SL, seed + 1, label, verbose=verbose)
    a_w, w = step1["intercept"], step1["w"]

    # ---- Step 2: sparse correction on the target labelled rows ---------
    empty_s = np.zeros((n_T, 0), dtype=np.float64)
    correction = P.fit_l1_bias_correction_cv(
        a_w, w, np.zeros(0, dtype=np.float64),
        X_TL, empty_s, Y_TL, int(seed) + 2, label=f"{label}-step2-delta",
    )
    d0 = float(correction["delta_intercept"])
    delta = np.asarray(correction["delta_beta"], dtype=np.float64)

    k_keep = int(np.floor(np.sqrt(n_T / np.log(max(p, 2)))))
    k_keep = max(k_keep, 1)
    delta_checked = hard_threshold(delta, k_keep) if hard_thresh else delta.copy()
    if verbose:
        print(f"    [{label}-step2] ||delta||_0={int(np.count_nonzero(delta))} "
              f"-> {int(np.count_nonzero(delta_checked))} after H_k with "
              f"k=floor(sqrt(nT/log p))={k_keep}"
              f"{'' if hard_thresh else '  (thresholding disabled)'}")

    # ---- Step 3: joint penalised refit on SL and TL together -----------
    # Target rows: logit = b0 + x'b.  Source rows: logit = b0 + x'b - (d0 + x'd).
    Z = np.vstack([X_TL, X_SL])
    y_pooled = np.concatenate([Y_TL, Y_SL])
    is_target = np.concatenate([np.ones(n_T, bool), np.zeros(len(Y_SL), bool)])
    offset = np.concatenate([
        np.zeros(n_T, dtype=np.float64),
        -(d0 + X_SL @ delta_checked),
    ])
    step3 = pooled_offset_lasso_cv(
        P, Z, y_pooled, offset, is_target, int(seed) + 3,
        label=f"{label}-step3-beta", cv_target_only=cv_target_only,
        verbose=verbose,
    )

    info = {
        "step1_C": step1["C"],
        "step1_nonzero": step1["nonzero"],
        "step2_lambda": float(correction["lambda"]),
        "step2_nonzero": int(correction["nonzero"]),
        "step2_l1_norm": float(correction["delta_l1_norm"]),
        "step2_k_keep": int(k_keep),
        "step2_nonzero_after_threshold": int(np.count_nonzero(delta_checked)),
        "step2_l1_after_threshold": float(np.abs(delta_checked).sum()),
        "step3_lambda": step3["lambda"],
        "step3_nonzero": step3["nonzero"],
        "step3_cv_loss": step3["cv_loss"],
        "n_source_labeled": int(len(Y_SL)),
        "n_target_labeled": int(n_T),
        "p_image": int(p),
    }
    return {
        # the three-step estimator (equations 1-3)
        "intercept": step3["intercept"],
        "beta": step3["beta"],
        # the pipeline's two-step fit (steps 1-2), for a like-for-like check
        "two_step_intercept": a_w + d0,
        "two_step_beta": w + delta,
        "delta_checked": delta_checked,
        "step1": step1,
        "info": info,
    }


def fit_target_only_lasso(P, X_TL, Y_TL, seed, label, verbose=True):
    model, best_C, _ = P.fit_l1_logistic_cv(
        np.asarray(X_TL, dtype=np.float64), np.asarray(Y_TL, dtype=int).ravel(),
        int(seed), label=f"{label}-target-only",
    )
    return float(model.intercept_[0]), model.coef_[0].astype(np.float64), float(best_C)


def cross_fitted_aggregation(P, X_SL, Y_SL, X_TL, Y_TL, seed, label,
                             step1=None, hard_thresh=True, cv_target_only=False,
                             epsilon0=0.0):
    """Section 2.2 aggregation: binary eta over {target-only, three-step}.

    Both candidates are refit on one half of TL and scored on the other half,
    in both directions, as the pipeline cross-fits the binary rule of Cai et
    al. (2025, eq. 9). Step 1 touches no target row, so the source fit is
    shared across folds rather than recomputed.
    """
    X_TL = np.asarray(X_TL, dtype=np.float64)
    Y_TL = np.asarray(Y_TL, dtype=int).ravel()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(seed))

    loss_transfer, loss_target = [], []
    for fold_id, (tr, va) in enumerate(cv.split(X_TL, Y_TL)):
        fit = fit_suptrans_three_step(
            P, X_SL, Y_SL, X_TL[tr], Y_TL[tr], int(seed) + 10 * (fold_id + 1),
            label=f"{label}-agg{fold_id}", step1=step1, hard_thresh=hard_thresh,
            cv_target_only=cv_target_only, verbose=False,
        )
        a_t, b_t, _ = fit_target_only_lasso(
            P, X_TL[tr], Y_TL[tr], int(seed) + 100 * (fold_id + 1),
            label=f"{label}-agg{fold_id}", verbose=False,
        )
        loss_transfer.append(P._weighted_log_loss_from_logits(
            Y_TL[va], fit["intercept"] + X_TL[va] @ fit["beta"]))
        loss_target.append(P._weighted_log_loss_from_logits(
            Y_TL[va], a_t + X_TL[va] @ b_t))

    mean_transfer = float(np.mean(loss_transfer))
    mean_target = float(np.mean(loss_target))
    eta = int(mean_transfer <= mean_target - float(epsilon0))
    print(f"    [{label}-agg] cross-fitted target logloss: "
          f"three-step={mean_transfer:.6f}  target-only={mean_target:.6f}  "
          f"-> eta={eta} ({'three-step' if eta else 'target-only'})")
    return {
        "eta": eta,
        "loss_transfer": mean_transfer,
        "loss_target": mean_target,
        "fold_loss_transfer": [float(v) for v in loss_transfer],
        "fold_loss_target": [float(v) for v in loss_target],
    }


# ============================================================
# Driver
# ============================================================

def build_row(P, rep, seed, branch, endpoint, eval_name, y, p, groups,
              nT, p_image, n_tl_positive):
    metrics = P.evaluate_predictions(y, p, groups)
    row = {
        "repeat": int(rep), "seed": int(seed), "nT": int(nT),
        "branch": branch, "model": f"{branch}_{endpoint}",
        "endpoint": endpoint, "evaluation_split": eval_name,
        "p_image": int(p_image), "n_tl_labeled": int(nT),
        "n_tl_positive": int(n_tl_positive),
    }
    row.update(metrics)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Recompute SUPTRANS (Li et al. 2023 three-step, SL + TL) "
                    "from saved per-repeat representations.")
    parser.add_argument("--pipeline", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_patient_split_nT200_valid200_history_bow.py"),
        help="Path to the main pipeline script; its solvers and metrics are reused.")
    parser.add_argument("--result-dir", default=None,
                        help="Defaults to RESULT_DIR of the pipeline module.")
    parser.add_argument("--prep-dir", default=None,
                        help="Defaults to PREP_DIR of the pipeline module.")
    parser.add_argument("--feature-dir", default=None,
                        help="Defaults to <result-dir>/repeat_features.")
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to <result-dir>/suptrans_recompute.")
    parser.add_argument("--nT", type=int, default=None)
    parser.add_argument("--repeats", type=int, nargs="*", default=None,
                        help="Repeat indices; default is every repeat on disk.")
    parser.add_argument("--branches", nargs="*", default=None,
                        help=f"Subset of {list(BRANCH_FEATURE_SUFFIX)}.")
    parser.add_argument("--no-aggregation", dest="aggregation",
                        action="store_false",
                        help="Skip the Section 2.2 aggregation endpoint.")
    parser.add_argument("--no-hard-threshold", dest="hard_threshold",
                        action="store_false",
                        help="Skip H_k on delta; the paper applies it.")
    parser.add_argument("--cv-target-only", action="store_true",
                        help="Select the step-3 penalty on target rows only.")
    parser.add_argument("--merge-raw", action="store_true",
                        help="Write raw_results_with_suptrans3.csv: the "
                             "pipeline's two-step suptrans rows renamed "
                             "suptrans_2step_old, the three-step rows added "
                             "as suptrans.")
    parser.set_defaults(aggregation=True, hard_threshold=True)
    args = parser.parse_args()

    P = import_pipeline(args.pipeline)
    result_dir = args.result_dir or P.RESULT_DIR
    prep_dir = args.prep_dir or P.PREP_DIR
    feature_dir = args.feature_dir or os.path.join(result_dir, "repeat_features")
    out_dir = args.out_dir or os.path.join(result_dir, "suptrans_recompute")
    nT = int(args.nT if args.nT is not None else P.NT)
    branches = list(args.branches or P.IMAGE_BRANCHES)
    unknown = [b for b in branches if b not in BRANCH_FEATURE_SUFFIX]
    if unknown:
        raise ValueError(f"Unknown branches {unknown}; "
                         f"expected a subset of {list(BRANCH_FEATURE_SUFFIX)}.")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 88)
    print("SUPTRANS recomputation: Li, Cai & Duan (2023) equations (1)-(3)")
    print("=" * 88)
    print(f"pipeline    : {args.pipeline}")
    print(f"result dir  : {result_dir}")
    print(f"prep dir    : {prep_dir}")
    print(f"features    : {feature_dir}")
    print(f"out dir     : {out_dir}")
    print(f"nT={nT}  branches={branches}  aggregation={args.aggregation}  "
          f"hard_threshold={args.hard_threshold}  "
          f"cv_target_only={args.cv_target_only}")

    Y_SL = load_labels_only(os.path.join(prep_dir, "SL.npz"), "SL")
    Y_TARGET = load_labels_only(os.path.join(prep_dir, "TARGET.npz"), "TARGET")
    patient_ids_target = load_target_patient_ids(prep_dir, len(Y_TARGET))
    print(f"\nSL labels: n={len(Y_SL)}, prevalence={float(np.mean(Y_SL)):.4f}")
    print(f"TARGET labels: n={len(Y_TARGET)}, "
          f"prevalence={float(np.mean(Y_TARGET)):.4f}")

    repeats = args.repeats if args.repeats else discover_repeats(feature_dir, nT)
    if not repeats:
        raise SystemExit(
            f"No saved repeats found under {feature_dir}. "
            "The main run must have been executed with SAVE_FEATURES=True."
        )
    print(f"Repeats to process: {repeats}")

    rows, fit_infos = [], []
    started = time.time()
    for rep in repeats:
        split = load_split(result_dir, nT, rep)
        run_seed = split["seed"] if split["seed"] is not None \
            else P.SEED + rep * 1000 + nT
        P.seed_everything(run_seed)

        tl_idx, tu_idx, valid_idx = split["tl_idx"], split["tu_idx"], split["valid_idx"]
        Y_TL = Y_TARGET[tl_idx].astype(int)
        if not np.array_equal(Y_TL, split["tl_labels"].astype(int)):
            raise AssertionError(
                f"repeat {rep}: TL labels in the split file do not match "
                "TARGET.npz; result-dir and prep-dir are out of sync."
            )
        Y_TU, Y_VALID = Y_TARGET[tu_idx].astype(int), Y_TARGET[valid_idx].astype(int)

        print("\n" + "#" * 88)
        print(f"Repeat {rep}  seed={run_seed}  TL={len(tl_idx)} "
              f"(positives={int(Y_TL.sum())})  VALID={len(valid_idx)}  "
              f"TU={len(tu_idx)}")
        print("#" * 88)

        for branch in branches:
            try:
                X_SL, X_TARGET = load_repeat(result_dir, feature_dir, nT, rep, branch)
            except FileNotFoundError as exc:
                print(f"  [skip] repeat {rep} branch {branch}: missing {exc}")
                continue
            if len(X_SL) != len(Y_SL) or len(X_TARGET) != len(Y_TARGET):
                raise AssertionError(
                    f"repeat {rep} branch {branch}: feature rows "
                    f"({len(X_SL)}, {len(X_TARGET)}) do not match label rows "
                    f"({len(Y_SL)}, {len(Y_TARGET)})."
                )
            X_TL, X_TU, X_VALID = X_TARGET[tl_idx], X_TARGET[tu_idx], X_TARGET[valid_idx]
            p_image = int(X_TL.shape[1])
            print(f"\n--- repeat {rep}, branch {branch}, p_image={p_image} ---")

            # The base seed matches fit_suptrans's `seed + 7`, so steps 1 and 2
            # reproduce the numbers already in raw_results.csv exactly.
            base_seed = int(run_seed) + 7
            label = f"{branch}-suptrans3"
            fit = fit_suptrans_three_step(
                P, X_SL, Y_SL, X_TL, Y_TL, base_seed, label,
                hard_thresh=args.hard_threshold,
                cv_target_only=args.cv_target_only,
            )

            estimators = {
                NEW_ENDPOINT: (fit["intercept"], fit["beta"]),
                OLD_ENDPOINT: (fit["two_step_intercept"], fit["two_step_beta"]),
            }

            agg = None
            if args.aggregation:
                a_t, b_t, C_t = fit_target_only_lasso(
                    P, X_TL, Y_TL, base_seed + 4, label)
                agg = cross_fitted_aggregation(
                    P, X_SL, Y_SL, X_TL, Y_TL, base_seed + 5, label,
                    step1=fit["step1"], hard_thresh=args.hard_threshold,
                    cv_target_only=args.cv_target_only,
                    epsilon0=getattr(P, "STRIFLE_GATE_EPSILON", 0.0),
                )
                estimators[TGT_ENDPOINT] = (a_t, b_t)
                estimators[AGG_ENDPOINT] = (
                    estimators[NEW_ENDPOINT] if agg["eta"] == 1 else (a_t, b_t)
                )

            eval_sets = {
                "valid": (X_VALID, Y_VALID, patient_ids_target[valid_idx]),
                "tu_transductive": (X_TU, Y_TU, patient_ids_target[tu_idx]),
            }
            for endpoint, (intercept, beta) in estimators.items():
                for eval_name, (X_eval, y_eval, groups) in eval_sets.items():
                    prob = expit(intercept + X_eval @ np.asarray(beta, np.float64))
                    rows.append(build_row(
                        P, rep, run_seed, branch, endpoint, eval_name,
                        y_eval, prob, groups, nT, p_image, int(Y_TL.sum()),
                    ))

            info = {"repeat": int(rep), "seed": int(run_seed), "branch": branch}
            info.update(fit["info"])
            if agg is not None:
                info.update({
                    "agg_eta": agg["eta"],
                    "agg_loss_three_step": agg["loss_transfer"],
                    "agg_loss_target_only": agg["loss_target"],
                    "target_only_C": C_t,
                })
            fit_infos.append(info)

            for eval_name in eval_sets:
                line = "  ".join(
                    f"{ep}={[r for r in rows if r['repeat'] == rep and r['branch'] == branch and r['endpoint'] == ep and r['evaluation_split'] == eval_name][0]['auc']:.4f}"
                    for ep in estimators
                )
                print(f"  [{branch}] {eval_name} AUC: {line}")

            # ---- persist after every branch, so a crash never loses work ----
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "suptrans_three_step_results.csv"),
                index=False)
            pd.DataFrame(fit_infos).to_csv(
                os.path.join(out_dir, "suptrans_three_step_fit_info.csv"),
                index=False)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("Nothing was computed.")

    print("\n" + "=" * 88)
    print("SUMMARY: mean AUC over repeats")
    print("=" * 88)
    summary = (frame.groupby(["evaluation_split", "branch", "endpoint"])
               .agg(n_repeats=("auc", "size"), auc_mean=("auc", "mean"),
                    auc_sd=("auc", "std"), pr_auc_mean=("pr_auc", "mean"),
                    brier_mean=("brier", "mean"))
               .reset_index()
               .sort_values(["evaluation_split", "branch", "endpoint"]))
    summary.to_csv(os.path.join(out_dir, "suptrans_three_step_summary.csv"),
                   index=False)
    print(summary.to_string(index=False))

    # ---- comparison with the suptrans column already on disk --------------
    raw_path = os.path.join(result_dir, "raw_results.csv")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        old = raw[raw["endpoint"] == "suptrans"][
            ["repeat", "branch", "evaluation_split", "auc", "pr_auc"]
        ].rename(columns={"auc": "auc_old_2step", "pr_auc": "pr_auc_old_2step"})
        new = frame[frame["endpoint"] == NEW_ENDPOINT][
            ["repeat", "branch", "evaluation_split", "auc", "pr_auc"]
        ].rename(columns={"auc": "auc_new_3step", "pr_auc": "pr_auc_new_3step"})
        paired = old.merge(new, on=["repeat", "branch", "evaluation_split"],
                           how="inner")
        if not paired.empty:
            paired["delta_auc"] = paired["auc_new_3step"] - paired["auc_old_2step"]
            paired.to_csv(
                os.path.join(out_dir, "suptrans_old_vs_new_paired.csv"), index=False)
            print("\nPaired change, three-step minus the pipeline's two-step fit:")
            print(paired.groupby(["evaluation_split", "branch"])
                  .agg(n=("delta_auc", "size"),
                       delta_auc_mean=("delta_auc", "mean"),
                       delta_auc_sd=("delta_auc", "std"),
                       wins=("delta_auc", lambda v: int((v > 0).sum())))
                  .reset_index().to_string(index=False))

        if args.merge_raw:
            merged = raw.copy()
            merged.loc[merged["endpoint"] == "suptrans", "endpoint"] = "suptrans_2step_old"
            merged.loc[merged["model"].astype(str).str.endswith("_suptrans"), "model"] = \
                merged.loc[merged["model"].astype(str).str.endswith("_suptrans"),
                           "model"].astype(str) + "_2step_old"
            add = frame[frame["endpoint"].isin([NEW_ENDPOINT, AGG_ENDPOINT])].copy()
            merged = pd.concat([merged, add], ignore_index=True, sort=False)
            merged_path = os.path.join(result_dir, "raw_results_with_suptrans3.csv")
            merged.to_csv(merged_path, index=False)
            print(f"\nMerged table written to {merged_path} "
                  "(the original raw_results.csv is untouched).")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min. "
          f"Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
