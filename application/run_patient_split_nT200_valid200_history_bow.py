"""Application pipeline of "Selective Blockwise Transfer for Surrogate-Assisted
Prediction under Population Shift".

Pneumothorax in CheXpert-Plus, White source and Asian target, at one
labelled-target budget NT per run. The paper runs two arms, NT = 200 and
NT = 400, with 20 repeats each. This file is the NT = 200 arm; the NT = 400 arm
is the same file with NT = 400 below, which then writes to its own result
directory (the Supervised fine-tuning schedule follows NT by itself). Each
repeat redraws TL, VALID and TU with seed 42 + 1000 r + NT, builds the three
image representations and runs the construction on each of them.

Estimation
----------
Density ratio. A ridge logistic domain classifier on Z = (X, S) over the
pooled SU and TU rows, applied to SL. The design is standardised on those rows
first, because the hard threshold kappa = 10 sqrt(log d / (N_S + N_T)) of
Cai et al. (2025) is defined for unit-variance covariates and on any other
scale would prune by column variance; the surviving slopes are mapped back to
the original scale. The log ratio is centred so that E_S[w] = 1, then clipped
to [0.01, 10] and renormalised. Centring comes first because clipping is not
shift-equivariant, and it also keeps both the classifier intercept and the
log(N_S / N_T) offset away from w.

Imputation model. A target-only L1 logistic fit on TL; a pooled fit on SL,
weighted by w, and TL; and a sparse offset correction delta on TL, which gives
the meta fit theta_tilde + delta. Penalties act on slopes only.

Coalitions and the two rules. theta^(c) takes each block from the meta fit
when the block is in c and from the target-only fit otherwise, with the
intercept refitted. The four coalitions are scored by two-fold cross-fitted
target log loss L(c), and v(c) = L(none) - L(c). The binary rule of Cai et al.
(2025, eq. 9) is rho = 1{L(XS) <= L(none) - eps0}. The coalition rule picks
c_hat, the coalition with the smallest L(c) among none and every c with
L(c) <= L(none) - eps0.

Attribution. The two-block Shapley values phi_X and phi_S of v sum to v(XS)
exactly. They read the four losses only, never eps0.

Outcome model. An image-only L1 logistic fit on the TL labels and on soft
labels for TU from theta^(c), fitted once per coalition, so every endpoint is
a complete estimator rather than a re-scoring of a shared fit.

The dead zone eps0
------------------
The method uses eps0 = log(d) / NT, with d = p + q of each branch: in the
paper .0243 for BioViL (d = 130) and .0270 for SimCLR and Supervised (d = 221)
at NT = 200, and .0122 and .0135 at NT = 400. This script fits at
STRIFLE_GATE_EPSILON = 0 and stores the four coalition losses of every repeat;
replay_gate_epsilon.py then applies log(d) / NT and rebuilds Gate-Binary and
Gate-Block from them. The losses, the four outcome models and phi do not depend
on eps0, so the replay gives what a fit at log(d) / NT gives.

Names in the code
-----------------
The identifiers predate the paper's notation and are kept, since the output
columns carry them:

    code                                  paper
    beta, gamma (imputation blocks)       X and S, coefficients theta_X, theta_S
    coalitions none, beta, gamma, both    none, X, S, XS
    phi_beta, phi_gamma                   phi_X, phi_S
    rho, endpoint paper_gate              binary rule rho_hat, Gate-Binary
    selected_endpoint_shapley,            coalition rule c_hat, Gate-Block
      endpoint shapley_gate
    to, beta_only, gamma_only,            Target-Only, Borrow-X, Borrow-S,
      corrected_meta                        Corrected-Meta
    branches ssl, biovil, no_ssl          SimCLR, BioViL, Supervised (feature
                                            files name the first "simclr")
    lr_to, lr_dr, target_ceiling          SUP, CS, Ceiling
    cnn_direct / dr_finetune              CNN
    suptrans                              steps 1 and 2 of SUPTRANS only; the
                                            reported SUPTRANS is recomputed by
                                            recompute_suptrans_three_step.py

The "shapley" names on the coalition rule are historical: c_hat is chosen from
the four losses and is not a Shapley quantity.

Evaluation
----------
VALID enters no fit and is the primary evaluation set; TU is reported
separately as transductive. Every endpoint is scored with AUC, PR-AUC, Brier
score and expected calibration error over ten equal-width bins, together with
F1 and the calibration slope and intercept. AUC and PR-AUC also carry a
500-replicate patient-clustered bootstrap interval. Contrasts are paired within
a repeat against Target-Only.

Artefacts
---------
Density-ratio weights, the coalition losses of both folds, the reduced image
features and the SimCLR backbones are written to disk, so the threshold
replay, the SUPTRANS recomputation and the block-strength diagnostic never
refit a representation.
"""

import os
import gc
import copy
import json
import time
import random
import hashlib
import warnings
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.special import expit, logsumexp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, brier_score_loss,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import DenseNet121_Weights

import project_paths


# ============================================================
# Config
# ============================================================

# torchvision caches the DenseNet-121 ImageNet weights under TORCH_HOME when it
# is set, and under its own default cache otherwise.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Locations come from project_paths.py (or the STRIFLE_* environment variables).
# RESULT_DIR is set below, once NT and N_VALID are defined.
PREP_DIR = project_paths.PREP_DIR
BIOVIL_IMAGE_BASE = project_paths.CHEXPERT_DIR   # BioViL reads the released JPEGs
BIOVIL_CACHE_DIR = os.path.join(PREP_DIR, "biovil_projected_global_128")
BIOVIL_PACKAGE_NAME = "hi-ml-multimodal"
BIOVIL_PACKAGE_VERSION = "0.2.2"
BIOVIL_MODEL_TYPE = "BIOVIL"
BIOVIL_WEIGHTS_ID = (
    "microsoft/BiomedVLP-CXR-BERT-specialized@v1.1/"
    "biovil_image_resnet50_proj_size_128.pt"
)
BIOVIL_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"
BIOVIL_CACHE_SCHEMA_VERSION = 3

SEED = 42
REPEATS = 20                  # 20 per arm, as in the paper; with 21 TL cases at
                              # NT = 200 the between-split variance dominates
                              # any one run
NT = 200                      # 400 for the paper's second arm
N_VALID = 200
MAX_IMAGES_PER_PATIENT = 6    # applied to TL *and* VALID, so the labelled and
                              # evaluation samples come from the same patient
                              # population

# Named after NT and N_VALID, so another arm writes to its own directory instead
# of resuming from this one's finished repeats.
RESULT_DIR = project_paths.run_dir(NT, N_VALID)
STANDARDIZER_DIR = os.path.join(RESULT_DIR, "standardizers")
WEIGHT_DIR = os.path.join(RESULT_DIR, "density_ratio")
FEATURE_DIR = os.path.join(RESULT_DIR, "repeat_features")
BACKBONE_DIR = os.path.join(RESULT_DIR, "backbones")
for _d in (RESULT_DIR, STANDARDIZER_DIR, WEIGHT_DIR, FEATURE_DIR, BACKBONE_DIR):
    os.makedirs(_d, exist_ok=True)

MIN_TL_POSITIVES = 4
MIN_TL_NEGATIVES = 4
MIN_VALID_POSITIVES = 4
MIN_VALID_NEGATIVES = 4

OUTCOME_LR_C = 0.1

STRIFLE_C_GRID = np.logspace(-3, 1, 9)
STRIFLE_TUNING_FOLDS = 5
STRIFLE_L1_SOLVER = "saga"
STRIFLE_L1_MAX_ITER = 10000
STRIFLE_L1_TOL = 1e-4

STRIFLE_DENSITY_RATIO_C = 1.0
STRIFLE_DENSITY_THRESHOLD_MULTIPLIER = 10.0
DENSITY_CLIP_MIN = 0.01
DENSITY_CLIP_MAX = 10.0

STRIFLE_CORRECTION_LAMBDA_RATIOS = np.asarray(
    [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], dtype=np.float64
)
STRIFLE_CORRECTION_MAX_ITER = 5000
STRIFLE_CORRECTION_TOL = 1e-7

# The method's dead zone is eps0 = log(d) / NT for each branch. The fit runs at
# 0 and stores the four coalition losses, which do not depend on eps0;
# replay_gate_epsilon.py then applies log(d) / NT to them and rebuilds the two
# gate endpoints, which is what a fit at that value would give.
STRIFLE_GATE_EPSILON = 0.0

# The pooled fit's objective fixes its fitting weights (source rows carry w,
# target rows weight one) but not how the cross-validation loss that selects
# its penalty is weighted. Under the fitting weights the 1200 reweighted source
# rows carry about 86% of that loss at NT = 200 (75% at NT = 400), so the
# penalty minimises a source-weighted risk while the estimand is target risk;
# restricting the loss to target rows aligns the two at the cost of a much
# noisier criterion at small NT. The paper's results use the fitting weights
# (False). True restricts the validation loss to the target rows.
STRIFLE_META_CV_TARGET_ONLY = False

IMAGE_PCA_DIM = 128
FEAT_DIM = 1024
BIOVIL_FEAT_DIM = 128
PCA_WHITEN = True             # unit variance, which the threshold kappa assumes

# Whitening divides each component by the square root of its eigenvalue, so a
# direction carrying essentially no variance is rescaled to unit variance and
# becomes pure numerical noise wearing the same clothes as real signal. BioViL
# is the case that forces this: its projection head leaves the 128 embedding
# dimensions strongly linearly dependent, so a 128 -> 128 "PCA" is badly
# conditioned and whitening would amplify its trailing components by orders of
# magnitude relative to the first. Components below this fraction of the
# leading eigenvalue are therefore dropped, which bounds the condition number
# at the reciprocal of the ratio. Branches keep whatever dimension survives
# (in the paper's runs BioViL keeps 37, SimCLR and Supervised all 128), so the
# image block can differ in width between branches; d = p + q is recomputed
# per branch wherever it is used.
PCA_MIN_EIGENVALUE_RATIO = 1e-3

SIMCLR_EPOCHS = 50
SIMCLR_BATCH_SIZE = 128
SIMCLR_TEMPERATURE = 0.07
SIMCLR_LR = 3e-4
SIMCLR_WEIGHT_DECAY = 1e-4

EXTRACT_BATCH_SIZE = 64
SAVE_BACKBONE = True
SAVE_FEATURES = True
SAVE_DENSITY_WEIGHTS = True

# Every repeat writes its results before the next one starts, so a crash never
# loses finished work; without RESUME the run would still redo it, and one
# repeat costs an hour of SimCLR. Set False to force a clean rerun.
RESUME = True

# Supervised branch (code name no_ssl): fine-tuning limits.
NO_SSL_MAX_EPOCHS = 50
NO_SSL_PATIENCE = 5

BOOTSTRAP_REPLICATES = 500    # patient-clustered, for AUC and PR-AUC
CALIBRATION_BINS = 10

IMAGE_BRANCHES = ("ssl", "biovil", "no_ssl")   # SimCLR, BioViL, Supervised

# External comparators, fitted on the same features, splits and evaluation
# rows. SUP, CS and SUPTRANS are benchmarks of Cai et al. (2025, Sec. 4); the
# Ceiling is fitted on every TU label and is a reference, not a competitor.
# None of them shares the construction's stages, so a contrast against them
# says whether the construction helps, never which block carries the gain.
# The suptrans fit here stops after steps 1 and 2 of Li, Cai and Duan (2023);
# the reported SUPTRANS is the three-step estimator recomputed by
# recompute_suptrans_three_step.py. Second entries are descriptions only.
BASELINE_VARIANTS = (
    ("lr_to", "SUP: ridge logistic on TL"),
    ("lr_dr", "CS: ridge logistic on SL and TL, image-only density ratio"),
    ("suptrans", "SUPTRANS steps 1 and 2 only; the reported SUPTRANS is the "
                 "three-step recompute"),
    ("target_ceiling", "Ceiling: ridge logistic on every TU label"),
)
# The six endpoints of the construction, as (code name, name in the paper).
STRIFLE_VARIANTS = (
    ("to", "Target-Only"),
    ("corrected_meta", "Corrected-Meta"),
    ("paper_gate", "Gate-Binary, the binary rule of Cai et al. (2025, eq. 9)"),
    ("beta_only", "Borrow-X"),
    ("gamma_only", "Borrow-S"),
    ("shapley_gate", "Gate-Block, the coalition rule"),
)


# ============================================================
# Reproducibility / memory
# ============================================================

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_memory(*objs):
    for obj in objs:
        try:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
        except Exception:
            pass
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def installed_version(name: str) -> str:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return "unknown"


# ============================================================
# Metrics: discrimination, calibration, clustered intervals
# ============================================================

def calibration_metrics(y, p, n_bins: int = CALIBRATION_BINS) -> dict:
    """Brier score, logistic calibration slope/intercept, ECE and MCE.

    ECE and MCE use n_bins equal-width probability bins (ten, as in the
    paper). The calibration slope regresses the outcome on the predicted
    logit; a slope below one is the classic signature of overfitted
    predictions, and the intercept at a fixed slope of one measures
    calibration-in-the-large.
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    out = {"brier": float(brier_score_loss(y, p))}

    logit = np.log(p / (1.0 - p))
    if len(np.unique(y)) == 2 and np.std(logit) > 1e-8:
        model = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=2000, fit_intercept=True
        ).fit(logit.reshape(-1, 1), y.astype(int))
        out["calibration_slope"] = float(model.coef_[0, 0])
        out["calibration_intercept"] = float(model.intercept_[0])
        # Calibration-in-the-large: intercept with the slope pinned at one.
        a = 0.0
        for _ in range(200):
            q = expit(logit + a)
            grad = float(np.mean(q - y))
            hess = max(float(np.mean(q * (1 - q))), 1e-10)
            step = np.clip(grad / hess, -5.0, 5.0)
            a -= step
            if abs(step) < 1e-12:
                break
        out["calibration_in_the_large"] = float(a)
    else:
        out["calibration_slope"] = np.nan
        out["calibration_intercept"] = np.nan
        out["calibration_in_the_large"] = np.nan

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        sel = bin_id == b
        if not sel.any():
            continue
        gap = abs(float(y[sel].mean()) - float(p[sel].mean()))
        ece += gap * sel.mean()
        mce = max(mce, gap)
    out["ece"] = float(ece)
    out["mce"] = float(mce)
    return out


def discrimination_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
                "prevalence": float(y.mean())}
    prevalence = float(y.mean())
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        # PR-AUC lives above the prevalence baseline, not above 0.5
        "pr_auc_lift": float(average_precision_score(y, p) / max(prevalence, 1e-12)),
        "f1": float(f1_score(y, (p > 0.5).astype(int), zero_division=0)),
        "prevalence": prevalence,
    }


def clustered_bootstrap(y, p, groups, statistics=("auc", "pr_auc"),
                        n_boot: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> dict:
    """Percentile interval from a patient-clustered bootstrap.

    Patients are resampled, not rows. VALID holds one image per patient, so
    there this is the ordinary bootstrap; TU holds several, and a row
    bootstrap would treat correlated rows as independent and understate the
    interval. evaluate_predictions requests AUC and PR-AUC only; "brier" is
    available but not requested.
    """
    y = np.asarray(y, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)

    draws = {name: [] for name in statistics}
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[g] for g in picked])
        if len(np.unique(y[rows])) < 2:
            continue
        if "auc" in draws:
            draws["auc"].append(roc_auc_score(y[rows], p[rows]))
        if "pr_auc" in draws:
            draws["pr_auc"].append(average_precision_score(y[rows], p[rows]))
        if "brier" in draws:
            draws["brier"].append(brier_score_loss(y[rows], p[rows]))

    out = {}
    for name, values in draws.items():
        if len(values) < 20:
            out[f"{name}_ci_low"] = np.nan
            out[f"{name}_ci_high"] = np.nan
            out[f"{name}_boot_sd"] = np.nan
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        out[f"{name}_ci_low"] = float(low)
        out[f"{name}_ci_high"] = float(high)
        out[f"{name}_boot_sd"] = float(np.std(values, ddof=1))
    return out


def evaluate_predictions(y, p, groups=None, with_bootstrap: bool = True) -> dict:
    result = discrimination_metrics(y, p)
    result.update(calibration_metrics(y, p))
    result["n_rows"] = int(len(y))
    if groups is not None:
        result["n_patients"] = int(len(np.unique(groups)))
        if with_bootstrap:
            result.update(clustered_bootstrap(y, p, groups))
    return result


# ============================================================
# Density ratios
# ============================================================

def _standardised_domain_fit(X_src, X_tgt, seed, C, hard_threshold: bool):
    """Ridge domain classifier on a standardised pooled design matrix.

    Returns coefficients and an intercept expressed on the ORIGINAL scale, so
    the caller can apply them to unstandardised rows.
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    domain_X = np.vstack([X_src, X_tgt])
    domain_y = np.hstack([
        np.zeros(len(X_src), dtype=int), np.ones(len(X_tgt), dtype=int),
    ])

    scaler = StandardScaler().fit(domain_X)
    domain_Z = scaler.transform(domain_X)
    model = LogisticRegression(
        C=float(C), solver="lbfgs", max_iter=5000, class_weight=None,
        random_state=int(seed),
    ).fit(domain_Z, domain_y)

    coef_standardised = model.coef_[0].astype(np.float64, copy=True)
    n_retained = int(len(coef_standardised))
    cutoff = np.nan
    if hard_threshold:
        # The cutoff kappa = 10 sqrt(log d / (N_S + N_T)) of Cai et al. (2025)
        # is defined for unit-variance covariates, which is why the fit above
        # is done in standardised coordinates.
        cutoff = STRIFLE_DENSITY_THRESHOLD_MULTIPLIER * np.sqrt(
            np.log(max(2, domain_X.shape[1])) / len(domain_X)
        )
        coef_standardised[np.abs(coef_standardised) < cutoff] = 0.0
        n_retained = int(np.count_nonzero(coef_standardised))

    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    coef = coef_standardised / scale
    # De-standardise the intercept for the thresholded coefficients: the fitted
    # value is kept and only the mean-shift term is recomputed. This is not a
    # refit at fixed slopes. It is retained so that log_ratio below is the
    # density ratio rather than the classifier logit; the centring in
    # _weights_from_log_ratio absorbs it, so it does not reach w_hat.
    intercept = float(model.intercept_[0]) - float(scaler.mean_ @ coef)

    domain_score = model.decision_function(domain_Z)
    domain_auc = (
        float(roc_auc_score(domain_y, domain_score))
        if len(np.unique(domain_y)) == 2 else np.nan
    )
    diagnostics = {
        "n_features": int(domain_X.shape[1]),
        "n_domain_rows": int(len(domain_X)),
        "cutoff": float(cutoff) if np.isfinite(cutoff) else None,
        "n_retained": n_retained,
        "domain_auc": domain_auc,
        "column_sd_min": float(scaler.scale_.min()),
        "column_sd_max": float(scaler.scale_.max()),
    }
    return coef, intercept, diagnostics


def _weights_from_log_ratio(log_ratio, tag: str, diagnostics: dict):
    """Centre to E_S[w] = 1, then clip, then renormalise.

    Clipping before centring lets a systematic offset in the log ratio be
    absorbed asymmetrically at the bounds, and dividing by the mean afterwards
    cannot undo that.
    """
    log_ratio = np.asarray(log_ratio, dtype=np.float64).ravel()
    centred = log_ratio - (logsumexp(log_ratio) - np.log(len(log_ratio)))

    low, high = np.log(DENSITY_CLIP_MIN), np.log(DENSITY_CLIP_MAX)
    n_low = int((centred < low).sum())
    n_high = int((centred > high).sum())
    weights = np.exp(np.clip(centred, low, high))
    weights = (weights / weights.mean()).astype(np.float32)

    quantiles = np.percentile(weights, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    diagnostics.update({
        "n_clipped_low": n_low,
        "n_clipped_high": n_high,
        "frac_clipped": float((n_low + n_high) / len(weights)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_sd": float(weights.std()),
        "weight_quantiles": [float(v) for v in quantiles],
        # Effective sample size after weighting; a collapse here means the
        # density ratio is doing almost nothing, or almost everything.
        "effective_sample_size": float(
            weights.sum() ** 2 / max(float((weights ** 2).sum()), 1e-12)
        ),
        "n_source_rows": int(len(weights)),
    })
    print(
        f"  [DR {tag}] d={diagnostics['n_features']}, "
        f"retained={diagnostics['n_retained']}/{diagnostics['n_features']}, "
        f"domain AUC={diagnostics['domain_auc']:.4f}, "
        f"w in ({weights.min():.3f}, {weights.max():.3f}), "
        f"clipped={diagnostics['frac_clipped']:.2%}, "
        f"ESS={diagnostics['effective_sample_size']:.0f}/{len(weights)}"
    )
    if np.isfinite(diagnostics["domain_auc"]) and diagnostics["domain_auc"] > 0.95:
        print(
            "    [WARN] apparent domain AUC above the prespecified 0.95: the two "
            "populations are nearly separable, so w(z)=p_T/p_S is barely "
            "identifiable and the bounded sup w_0/w_hat that the classification "
            "route assumes is doubtful; read this fit's transfer results with "
            "that qualification."
        )
    return weights


def compute_feat_density_ratio_image_only(X_SU_feat, X_target_unlab_feat,
                                          X_SL_feat, seed, tag="image"):
    """Image-only density ratio used by CS: the same fit on X alone, with no
    hard threshold."""
    coef, intercept, diag = _standardised_domain_fit(
        X_SU_feat, X_target_unlab_feat, seed, C=1.0, hard_threshold=False
    )
    log_ratio = (
        intercept
        + np.asarray(X_SL_feat, dtype=np.float64) @ coef
        + np.log(len(X_SU_feat) / len(X_target_unlab_feat))
    )
    weights = _weights_from_log_ratio(log_ratio, tag, diag)
    return weights, diag


def compute_feat_density_ratio_xs(X_SU_feat, S_SU, X_target_unlab_feat,
                                  S_target_unlab, X_SL_feat, S_SL, seed,
                                  tag="paper X+S"):
    """Density ratio of the construction: ridge domain classifier on
    Z = (X, S) over SU and TU, hard-thresholded at kappa, applied to SL."""
    X_src = np.hstack([X_SU_feat, S_SU])
    X_tgt = np.hstack([X_target_unlab_feat, S_target_unlab])
    X_apply = np.hstack([X_SL_feat, S_SL]).astype(np.float64, copy=False)

    coef, intercept, diag = _standardised_domain_fit(
        X_src, X_tgt, seed, C=STRIFLE_DENSITY_RATIO_C, hard_threshold=True
    )
    log_ratio = (
        intercept + X_apply @ coef + np.log(len(X_src) / len(X_tgt))
    )
    weights = _weights_from_log_ratio(log_ratio, tag, diag)
    return weights, diag


def compute_pixel_density_ratio_image_only(X_SU_raw, X_target_unlab_raw,
                                           X_SL_raw, seed, svd_dim=24):
    """Pixel-space density ratio used to weight SL in the Supervised branch.

    Raw images scaled to [0, 1] are mean-centred and projected onto svd_dim
    truncated-SVD components, fitted on at most 20,000 images drawn
    proportionally from SU and TU; the domain classifier then runs on the
    projections without the hard threshold. The mean and the projections are
    computed in chunks to bound peak memory.
    """
    print(f"  [pixel DR] source=SU({len(X_SU_raw)}), "
          f"target_unlab={len(X_target_unlab_raw)}, svd_dim={svd_dim}")
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)

    def to_float(block):
        return np.asarray(block, dtype=np.float32) / 255.0

    # Mean over both domains without materialising the stacked matrix.
    n_total = len(X_SU_raw) + len(X_target_unlab_raw)
    mean = np.zeros(X_SU_raw.shape[1], dtype=np.float64)
    for block in (X_SU_raw, X_target_unlab_raw):
        for start in range(0, len(block), 2000):
            mean += to_float(block[start:start + 2000]).sum(axis=0)
    mean /= n_total
    mean = mean.astype(np.float32)

    sample_rows = min(20000, n_total)
    rng = np.random.default_rng(seed)
    take_su = int(round(sample_rows * len(X_SU_raw) / n_total))
    take_tu = sample_rows - take_su
    fit_block = np.vstack([
        to_float(X_SU_raw[rng.choice(len(X_SU_raw), take_su, replace=False)]) - mean,
        to_float(X_target_unlab_raw[
            rng.choice(len(X_target_unlab_raw), take_tu, replace=False)]) - mean,
    ])
    svd.fit(fit_block)
    del fit_block
    gc.collect()

    def project(block):
        out = np.empty((len(block), svd_dim), dtype=np.float32)
        for start in range(0, len(block), 2000):
            chunk = to_float(block[start:start + 2000]) - mean
            out[start:start + len(chunk)] = chunk @ svd.components_.T
        return out

    weights, diag = compute_feat_density_ratio_image_only(
        project(X_SU_raw), project(X_target_unlab_raw), project(X_SL_raw),
        seed=seed, tag="pixel",
    )
    return weights, diag


# ============================================================
# Representation transforms
# ============================================================

def fit_transform_shared_scaler_pca(
    reference_arrays, transform_arrays, n_components, tag, file_stem,
    random_state=SEED, whiten=PCA_WHITEN,
    fit_policy="SL + SU + repeat-specific TL + TU; labels unused",
):
    """One label-free StandardScaler and PCA per modality.

    The reference pool must exclude validation rows and every sibling image of
    a selected TL or validation patient. No outcome label enters either fit.
    """
    if len(reference_arrays) == 0 or len(transform_arrays) == 0:
        raise ValueError(f"[{tag}] scaler/PCA requires non-empty array lists.")

    n_features = int(np.asarray(reference_arrays[0]).shape[1])
    reference_rows = int(sum(len(a) for a in reference_arrays))
    max_components = min(reference_rows, n_features)
    if not 1 <= int(n_components) <= max_components:
        raise ValueError(
            f"[{tag}] n_components={n_components} exceeds {max_components}."
        )

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    for array in reference_arrays:
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] != n_features:
            raise ValueError(f"[{tag}] inconsistent reference shape {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"[{tag}] reference features contain NaN or infinity.")
        if len(array) > 0:
            scaler.partial_fit(array)

    reference_standardized = np.vstack([
        scaler.transform(np.asarray(a)).astype(np.float32, copy=False)
        for a in reference_arrays if len(a) > 0
    ])
    pca_solver = "full" if int(n_components) == n_features else "randomized"
    pca = PCA(n_components=int(n_components), whiten=whiten,
              svd_solver=pca_solver, random_state=random_state)
    pca.fit(reference_standardized)
    del reference_standardized
    gc.collect()

    # Drop trailing components whose eigenvalue is negligible relative to the
    # leading one. With whiten=True each component is already divided by its
    # own sqrt(eigenvalue), so truncating output columns is exact.
    lam = pca.explained_variance_
    eigen_ratio = lam / max(float(lam[0]), 1e-300)
    n_keep = int(np.searchsorted(-eigen_ratio, -float(PCA_MIN_EIGENVALUE_RATIO),
                                 side="right"))
    n_keep = int(np.clip(n_keep, 1, len(lam)))
    n_dropped = int(len(lam) - n_keep)

    transformed = []
    for array in transform_arrays:
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] != n_features:
            raise ValueError(f"[{tag}] inconsistent transform shape {array.shape}.")
        reduced = pca.transform(scaler.transform(array))[:, :n_keep]
        reduced = reduced.astype(np.float32, copy=False)
        if not np.isfinite(reduced).all():
            raise ValueError(f"[{tag}] PCA features contain NaN or infinity.")
        transformed.append(reduced)

    np.savez_compressed(
        os.path.join(STANDARDIZER_DIR, f"{file_stem}.npz"),
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        var=scaler.var_.astype(np.float64),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float64),
        pca_explained_variance=pca.explained_variance_.astype(np.float64),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        pca_n_components=np.asarray(pca.n_components_, dtype=np.int64),
        pca_n_components_kept=np.asarray(n_keep, dtype=np.int64),
        pca_min_eigenvalue_ratio=np.asarray(PCA_MIN_EIGENVALUE_RATIO),
        pca_whiten=np.asarray(whiten),
        pca_solver=np.asarray(pca_solver),
        fit_policy=np.asarray(fit_policy),
        tag=np.asarray(tag),
    )

    condition_full = float(lam[0] / max(lam[-1], 1e-12))
    condition_kept = float(lam[0] / max(lam[n_keep - 1], 1e-12))
    kept_evr = float(pca.explained_variance_ratio_[:n_keep].sum())
    print(
        f"  [Scaler+PCA {tag}] rows={reference_rows}, "
        f"{n_features}->{n_keep}, whiten={whiten}, "
        f"explained={kept_evr:.4f}, condition={condition_kept:.4g}"
    )
    if n_dropped:
        print(f"    dropped {n_dropped} components below "
              f"{PCA_MIN_EIGENVALUE_RATIO:g} of the leading eigenvalue "
              f"(condition before truncation {condition_full:.4g}); whitening "
              "them would have turned numerical noise into unit-variance "
              "features")
    if condition_kept > 1e3:
        print("    [WARN] condition number still above 1e3; raise "
              "PCA_MIN_EIGENVALUE_RATIO.")
    return tuple(transformed)


def reduce_image_representation(X_SL, X_SU, X_TARGET, target_fit_idx, tag, file_stem):
    return fit_transform_shared_scaler_pca(
        reference_arrays=[X_SL, X_SU, X_TARGET[target_fit_idx]],
        transform_arrays=[X_SL, X_SU, X_TARGET],
        n_components=IMAGE_PCA_DIM, tag=tag, file_stem=file_stem,
        fit_policy=("SL + SU + repeat-specific TL + TU; validation and "
                    "selected-patient siblings excluded"),
    )


def standardize_block(reference_arrays, transform_arrays, tag, file_stem):
    """Standardise every column of a block; no rotation, no truncation.

    Used for the bag-of-words text block and for the full metadata block, both
    standardised per repeat on SL, SU, TL and TU with VALID excluded. Nothing
    rotates or reduces the surrogate block, so every coefficient of theta_S
    reads back to its term or field; unit variance is what the Lasso
    penalties and the density-ratio threshold kappa assume. Constant columns
    keep a scale of one so they stay at zero rather than exploding.
    """
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    for array in reference_arrays:
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"[{tag}] reference block contains NaN or infinity.")
        scaler.partial_fit(array)

    raw_sd = np.sqrt(np.maximum(scaler.var_, 0.0))
    near_constant = int((raw_sd < 1e-8).sum())
    transformed = [
        scaler.transform(np.asarray(a, dtype=np.float64)).astype(np.float32)
        for a in transform_arrays
    ]
    np.savez_compressed(
        os.path.join(STANDARDIZER_DIR, f"{file_stem}.npz"),
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        var=scaler.var_.astype(np.float64),
        tag=np.asarray(tag),
        fit_policy=np.asarray(
            "SL + SU + repeat-specific TL + TU; validation and "
            "selected-patient siblings excluded"
        ),
    )
    stacked = np.vstack(transformed)
    print(
        f"  [Scaler {tag}] d={scaler.n_features_in_}, "
        f"raw sd in ({raw_sd.min():.4g}, {raw_sd.max():.4g}), "
        f"near-constant={near_constant}, "
        f"max|z|={np.abs(stacked).max():.1f}"
    )
    return tuple(transformed)


# ============================================================
# Dataset / DenseNet helpers
# ============================================================

class RawImageDataset(Dataset):
    def __init__(self, X_flat, Y=None, weights=None, augment=False):
        self.X = X_flat
        self.Y = Y.astype(np.float32) if Y is not None else np.zeros(len(X_flat), np.float32)
        self.weights = (
            weights.astype(np.float32) if weights is not None
            else np.ones(len(X_flat), np.float32)
        )
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, tensor):
        angle = (torch.rand(1).item() - 0.5) * 10.0
        tensor = TF.rotate(tensor, angle)
        factor = 0.9 + torch.rand(1).item() * 0.2
        tensor = torch.clamp(tensor * factor, -3.0, 3.0)
        tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.01, -3.0, 3.0)
        return tensor

    def __getitem__(self, idx):
        img = self.X[idx].reshape(224, 224).astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_3ch = (img_3ch - mean[:, None, None]) / std[:, None, None]
        tensor = torch.tensor(img_3ch, dtype=torch.float32)
        if self.augment:
            tensor = self._augment(tensor)
        return (tensor,
                torch.tensor(self.Y[idx], dtype=torch.float32),
                torch.tensor(self.weights[idx], dtype=torch.float32))


class SimCLRTwoViewDataset(RawImageDataset):
    def __init__(self, X_flat):
        super().__init__(X_flat, augment=True)

    def __getitem__(self, idx):
        img = self.X[idx].reshape(224, 224).astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_3ch = (img_3ch - mean[:, None, None]) / std[:, None, None]
        base = torch.tensor(img_3ch, dtype=torch.float32)
        return self._augment(base.clone()), self._augment(base.clone())


def build_densenet121(mode="extractor"):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    if mode == "finetune":
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif mode == "extractor":
        model.classifier = nn.Identity()
    else:
        raise ValueError("mode must be 'finetune' or 'extractor'")
    return model.to(DEVICE)


def _amp_context(enabled: bool):
    return torch.autocast(device_type=DEVICE.type, enabled=enabled)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(DEVICE.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def extract_features(X_flat, extractor, batch_size=EXTRACT_BATCH_SIZE, tag=""):
    extractor.eval()
    loader = DataLoader(RawImageDataset(X_flat, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    feats, done = [], 0
    for X_b, _, _ in loader:
        feats.append(extractor(X_b.to(DEVICE)).cpu().numpy())
        done += len(X_b)
        if tag and (done % 5000 == 0 or done == len(X_flat)):
            print(f"  [{tag}] features {done}/{len(X_flat)}")
    return np.vstack(feats).astype(np.float32)


@torch.no_grad()
def predict_proba_densenet(X_flat, model, batch_size=EXTRACT_BATCH_SIZE, tag=""):
    model.eval()
    loader = DataLoader(RawImageDataset(X_flat, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for X_b, _, _ in loader:
        probs.append(torch.sigmoid(model(X_b.to(DEVICE)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs)


# ============================================================
# SimCLR
# ============================================================

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B, device=z.device),
                        torch.arange(0, B, device=z.device)])
    return F.cross_entropy(sim, labels)


def train_simclr(X_SSL_mem, seed, n_epochs=SIMCLR_EPOCHS, batch_size=SIMCLR_BATCH_SIZE):
    print(f"\n[SSL SimCLR] N={len(X_SSL_mem)}, epochs={n_epochs}, batch={batch_size}")
    loader = DataLoader(
        SimCLRTwoViewDataset(X_SSL_mem), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    backbone = build_densenet121(mode="extractor")
    proj_head = SimCLRProjectionHead(in_dim=FEAT_DIM).to(DEVICE)
    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": SIMCLR_LR * 0.1},
        {"params": proj_head.parameters(), "lr": SIMCLR_LR},
    ], weight_decay=SIMCLR_WEIGHT_DECAY)

    use_amp = DEVICE.type == "cuda"
    scaler = _grad_scaler(use_amp)
    best_loss, best_state, history = np.inf, None, []

    for epoch in range(1, n_epochs + 1):
        backbone.train(); proj_head.train()
        total, nb = 0.0, 0
        for x1, x2 in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(use_amp):
                loss = nt_xent_loss(proj_head(backbone(x1)), proj_head(backbone(x2)),
                                    SIMCLR_TEMPERATURE)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item(); nb += 1
        avg = total / max(nb, 1)
        history.append(float(avg))
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [SSL] epoch {epoch:3d} loss={avg:.4f}"
                  + (" *" if avg < best_loss else ""))
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}

    backbone.load_state_dict(best_state)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    free_memory(proj_head, optimizer, loader)
    print(f"  [SSL] done best_loss={best_loss:.4f}")
    return backbone, {"best_loss": float(best_loss), "loss_history": history}


# ============================================================
# STRIFLE estimating stages
# ============================================================

def _weighted_log_loss_from_logits(y, logits, sample_weight=None):
    y = np.asarray(y, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    losses = np.logaddexp(0.0, logits) - y * logits
    if sample_weight is None:
        return float(losses.mean())
    sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
    if sample_weight.shape != losses.shape or sample_weight.sum() <= 0:
        raise ValueError("Invalid sample weights for logistic loss.")
    return float(np.average(losses, weights=sample_weight))


def _refit_intercept(linear_predictor, y):
    """Unpenalised intercept given fixed slopes.

    A coalition splices slope blocks from two fits calibrated to different
    populations, the meta fit to a reweighted mixture of source and target and
    the target-only fit to the target, so an inherited intercept would carry a
    prevalence difference that no block put there. Refitting it on the rows
    the coalition is formed on keeps the comparison about the slopes.
    """
    linear_predictor = np.asarray(linear_predictor, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a = 0.0
    for _ in range(200):
        p = expit(linear_predictor + a)
        grad = float(np.mean(p - y))
        hess = max(float(np.mean(p * (1.0 - p))), 1e-10)
        step = np.clip(grad / hess, -5.0, 5.0)
        a -= step
        if abs(step) < 1e-12:
            break
    return float(a)


def _fit_l1_logistic_fixed_c(X, y, C, seed, sample_weight=None, warm_start=False):
    model = LogisticRegression(
        C=float(C), penalty="l1", solver=STRIFLE_L1_SOLVER,
        max_iter=STRIFLE_L1_MAX_ITER, tol=STRIFLE_L1_TOL, fit_intercept=True,
        class_weight=None, warm_start=bool(warm_start), random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y, sample_weight=sample_weight)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print(f"    [warning] L1 logistic reached max_iter at C={float(C):.4g}")
    return model


def fit_l1_logistic_cv(X, y, seed, sample_weight=None, label="theta",
                       cv_weight=None):
    """Lasso tuning by stratified cross-validated log loss over STRIFLE_C_GRID.

    By default the validation loss carries the fitting weights, which is what
    the paper's results use. ``cv_weight`` evaluates it under other weights
    instead: for the pooled fit, target-only validation weights align the
    criterion with target risk (see STRIFLE_META_CV_TARGET_ONLY).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).ravel()
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
    if cv_weight is None:
        cv_weight = sample_weight
    else:
        cv_weight = np.asarray(cv_weight, dtype=np.float64).ravel()

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or int(counts.min()) < 2:
        raise ValueError(f"{label}: L1 logistic CV needs at least two rows per class.")
    n_splits = min(int(STRIFLE_TUNING_FOLDS), int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    Cs = np.asarray(STRIFLE_C_GRID, dtype=np.float64)
    fold_losses = np.full((n_splits, len(Cs)), np.nan, dtype=np.float64)

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if cv_weight[val_idx].sum() <= 0:
            continue
        model = None
        for c_idx, C in enumerate(Cs):
            if model is None:
                model = _fit_l1_logistic_fixed_c(
                    X[train_idx], y[train_idx], C, seed + fold_id,
                    sample_weight=sample_weight[train_idx], warm_start=True,
                )
            else:
                model.set_params(C=float(C))
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X[train_idx], y[train_idx],
                              sample_weight=sample_weight[train_idx])
            fold_losses[fold_id, c_idx] = _weighted_log_loss_from_logits(
                y[val_idx], model.decision_function(X[val_idx]), cv_weight[val_idx]
            )

    mean_losses = np.nanmean(fold_losses, axis=0)
    best_idx = int(np.nanargmin(mean_losses))
    best_C = float(Cs[best_idx])
    fitted = _fit_l1_logistic_fixed_c(
        X, y, best_C, seed + 100003, sample_weight=sample_weight, warm_start=False
    )
    print(
        f"    [{label}] L1 CV folds={n_splits}, C={best_C:.4g}, "
        f"CV-logloss={mean_losses[best_idx]:.6f}, "
        f"nonzero={int(np.count_nonzero(fitted.coef_))}/{X.shape[1]}"
    )
    return fitted, best_C, mean_losses


_OFFSET_NONCONVERGENCE = {"count": 0}


def _fit_offset_logistic_l1(Z, y, offset, lam, max_iter, tol):
    """FISTA for a logistic offset correction with an L1 slope penalty."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    offset = np.asarray(offset, dtype=np.float64).ravel()
    n, d = Z.shape
    A = np.column_stack([np.ones(n, dtype=np.float64), Z])
    spectral = float(np.linalg.norm(A, ord=2))
    step = 1.0 / max(0.25 * spectral * spectral / n, 1e-8)

    current = np.zeros(d + 1, dtype=np.float64)
    accelerated = current.copy()
    momentum, converged, iteration = 1.0, False, 0

    for iteration in range(1, int(max_iter) + 1):
        gradient = (A.T @ (expit(offset + A @ accelerated) - y)) / n
        proposal = accelerated - step * gradient
        proposal[1:] = np.sign(proposal[1:]) * np.maximum(
            np.abs(proposal[1:]) - step * float(lam), 0.0
        )
        scale = 1.0 + np.max(np.abs(current))
        if np.max(np.abs(proposal - current)) <= float(tol) * scale:
            current, converged = proposal, True
            break
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = proposal + ((momentum - 1.0) / next_momentum) * (proposal - current)
        current, momentum = proposal, next_momentum

    objective = _weighted_log_loss_from_logits(y, offset + A @ current) \
        + float(lam) * np.abs(current[1:]).sum()
    if not np.isfinite(current).all() or not np.isfinite(objective):
        raise RuntimeError("L1 offset correction produced a non-finite solution.")
    if not converged:
        _OFFSET_NONCONVERGENCE["count"] += 1
    return float(current[0]), current[1:], float(objective), iteration


def fit_l1_bias_correction_cv(base_intercept, base_beta, base_gamma,
                              X, S, y, seed, label="delta"):
    """The correction delta of FitPair (Algorithm 1 of the paper).

    A sparse offset fit on the target rows around the fixed linear predictor
    of theta_tilde, intercept unpenalised. The penalty is chosen by stratified
    cross-validated target log loss over lambda_max times
    STRIFLE_CORRECTION_LAMBDA_RATIOS, lambda_max being the smallest penalty
    that zeroes every slope at the refitted intercept.
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64).reshape(len(X), -1)
    y = np.asarray(y, dtype=int).ravel()
    # S may legitimately be zero-width: SUPTRANS corrects on X alone.
    Z = np.hstack([X, S]) if S.shape[1] else X
    coefficients = (np.hstack([base_beta, base_gamma]) if S.shape[1]
                    else np.asarray(base_beta, dtype=np.float64))
    offset = float(base_intercept) + Z @ coefficients

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or int(counts.min()) < 2:
        raise ValueError(f"{label}: correction CV needs at least two rows per class.")

    delta0 = 0.0
    for _ in range(200):
        p = expit(offset + delta0)
        step = np.clip(float(np.mean(p - y)) / max(float(np.mean(p * (1 - p))), 1e-8),
                       -5.0, 5.0)
        delta0 -= step
        if abs(step) < 1e-10:
            break
    lambda_max = max(float(np.max(np.abs(Z.T @ (expit(offset + delta0) - y))) / len(y)),
                     1e-6)
    lambdas = lambda_max * STRIFLE_CORRECTION_LAMBDA_RATIOS

    n_splits = min(int(STRIFLE_TUNING_FOLDS), int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    fold_losses = np.zeros((n_splits, len(lambdas)), dtype=np.float64)
    _OFFSET_NONCONVERGENCE["count"] = 0
    for fold_id, (train_idx, val_idx) in enumerate(cv.split(Z, y)):
        for lam_idx, lam in enumerate(lambdas):
            d0, dcoef, _, _ = _fit_offset_logistic_l1(
                Z[train_idx], y[train_idx], offset[train_idx], lam,
                STRIFLE_CORRECTION_MAX_ITER, STRIFLE_CORRECTION_TOL,
            )
            fold_losses[fold_id, lam_idx] = _weighted_log_loss_from_logits(
                y[val_idx], offset[val_idx] + d0 + Z[val_idx] @ dcoef
            )

    mean_losses = fold_losses.mean(axis=0)
    best_idx = int(np.argmin(mean_losses))
    best_lambda = float(lambdas[best_idx])
    d0, dcoef, objective, iterations = _fit_offset_logistic_l1(
        Z, y, offset, best_lambda, STRIFLE_CORRECTION_MAX_ITER, STRIFLE_CORRECTION_TOL
    )
    split = X.shape[1]
    if _OFFSET_NONCONVERGENCE["count"]:
        print(f"    [{label}] {_OFFSET_NONCONVERGENCE['count']} of "
              f"{n_splits * len(lambdas) + 1} FISTA solves hit max_iter "
              f"(small lambda ends of the path)")
    print(
        f"    [{label}] L1 CV folds={n_splits}, lambda={best_lambda:.4g}, "
        f"CV-logloss={mean_losses[best_idx]:.6f}, "
        f"nonzero={int(np.count_nonzero(dcoef))}, "
        f"||delta||_1={float(np.abs(dcoef).sum()):.4f}, iterations={iterations}"
    )
    return {
        "delta_intercept": float(d0),
        "delta_beta": dcoef[:split],
        "delta_gamma": dcoef[split:],
        "delta_l1_norm": float(np.abs(dcoef).sum()),
        "delta_beta_l1": float(np.abs(dcoef[:split]).sum()),
        "delta_gamma_l1": float(np.abs(dcoef[split:]).sum()),
        "lambda": best_lambda,
        "cv_losses": mean_losses,
        "objective": float(objective),
        "nonzero": int(np.count_nonzero(dcoef)),
    }


# ============================================================
# Coalitions, the two rules and the Shapley attribution
# ============================================================

# none, X, S and XS in the paper; beta is the image block, gamma the surrogate.
COALITIONS = ("none", "beta", "gamma", "both")


def _coalition_slopes(to_beta, to_gamma, meta_beta, meta_gamma, coalition):
    """theta^(c): each slope block from the meta fit if it is in c, from the
    target-only fit otherwise. The caller refits the intercept."""
    beta = meta_beta if coalition in ("beta", "both") else to_beta
    gamma = meta_gamma if coalition in ("gamma", "both") else to_gamma
    return beta, gamma


def _fit_target_only_and_meta(X_tr, S_tr, y_tr, XS_SL, Y_SL, w_hat, feat_dim,
                              seed, label):
    """FitPair of Algorithm 1 on one set of TL rows: the target-only fit, the
    pooled fit theta_tilde and its correction delta, which give the meta fit."""
    XS_tr = np.hstack([X_tr, S_tr])
    to_model, _, _ = fit_l1_logistic_cv(
        XS_tr, y_tr, seed + 1, label=f"{label}-theta-T"
    )
    to = (float(to_model.intercept_[0]),
          to_model.coef_[0, :feat_dim], to_model.coef_[0, feat_dim:])

    fit_weights = np.hstack([w_hat, np.ones(len(y_tr), dtype=np.float64)])
    cv_weights = (
        np.hstack([np.zeros(len(Y_SL)), np.ones(len(y_tr))])
        if STRIFLE_META_CV_TARGET_ONLY else None
    )
    meta_model, _, _ = fit_l1_logistic_cv(
        np.vstack([XS_SL, XS_tr]), np.hstack([Y_SL, y_tr]), seed + 2,
        sample_weight=fit_weights, cv_weight=cv_weights,
        label=f"{label}-theta-tilde",
    )
    tilde = (float(meta_model.intercept_[0]),
             meta_model.coef_[0, :feat_dim], meta_model.coef_[0, feat_dim:])
    correction = fit_l1_bias_correction_cv(
        tilde[0], tilde[1], tilde[2], X_tr, S_tr, y_tr, seed + 3, label=f"{label}-delta"
    )
    meta = (tilde[0] + correction["delta_intercept"],
            tilde[1] + correction["delta_beta"],
            tilde[2] + correction["delta_gamma"])
    return to, tilde, meta, correction, to_model, meta_model


def cross_fitted_shapley_gate(X_TL, Y_TL, S_TL, XS_SL, w_hat, Y_SL, feat_dim,
                              seed, epsilon0=STRIFLE_GATE_EPSILON, label="gate"):
    """Cross-fitted coalition losses, the two rules and the Shapley values.

    TL is split into two stratified folds. On each, FitPair is refitted, every
    coalition is assembled from that one pair with its intercept refitted on
    the fold, and its log loss is scored on the other fold; L(c) averages the
    two. Coalitions ``none`` and ``both`` reproduce the target-only and meta
    fits up to that intercept, so comparing them recovers the binary rule of
    Cai et al. (2025, eq. 9). With v(c) = L(none) - L(c):

        rho     = 1{L(both) <= L(none) - epsilon0}                  binary rule
        c_hat   = argmin L(c) over none and every c with
                  L(c) <= L(none) - epsilon0                      coalition rule
        phi_beta + phi_gamma = v(both) exactly           (phi_X and phi_S)

    The method's epsilon0 is log(d) / NT; see STRIFLE_GATE_EPSILON for why
    the run uses 0 and the replay applies it.
    """
    X_TL = np.asarray(X_TL)
    Y_TL = np.asarray(Y_TL, dtype=int)
    S_TL = np.asarray(S_TL)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(seed))
    fold_losses = {c: [] for c in COALITIONS}
    fold_records = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_TL, Y_TL)):
        X_tr, S_tr, y_tr = X_TL[train_idx], S_TL[train_idx], Y_TL[train_idx]
        to, _, meta, _, _, _ = _fit_target_only_and_meta(
            X_tr, S_tr, y_tr, XS_SL, Y_SL, w_hat, feat_dim,
            seed + 100 * fold_id, f"{label}-fold{fold_id + 1}",
        )
        record = {"fold": fold_id}
        for coalition in COALITIONS:
            beta, gamma = _coalition_slopes(to[1], to[2], meta[1], meta[2], coalition)
            linear_tr = X_tr @ beta + S_tr @ gamma
            intercept = _refit_intercept(linear_tr, y_tr)
            logits = intercept + X_TL[val_idx] @ beta + S_TL[val_idx] @ gamma
            loss = _weighted_log_loss_from_logits(Y_TL[val_idx], logits)
            fold_losses[coalition].append(loss)
            record[f"loss_{coalition}"] = float(loss)
        fold_records.append(record)

    mean_loss = {c: float(np.mean(fold_losses[c])) for c in COALITIONS}
    # Value = reduction in cross-fitted target loss relative to no transfer.
    value = {c: float(mean_loss["none"] - mean_loss[c]) for c in COALITIONS}
    phi_beta = 0.5 * ((value["beta"] - value["none"]) + (value["both"] - value["gamma"]))
    phi_gamma = 0.5 * ((value["gamma"] - value["none"]) + (value["both"] - value["beta"]))
    efficiency_gap = abs((phi_beta + phi_gamma) - value["both"])

    rho = int(mean_loss["both"] <= mean_loss["none"] - float(epsilon0))
    eligible = {c: mean_loss[c] for c in COALITIONS
                if c == "none" or mean_loss[c] <= mean_loss["none"] - float(epsilon0)}
    shapley_choice = min(eligible, key=eligible.get)

    print(
        f"    [gate] losses none={mean_loss['none']:.6f} "
        f"beta={mean_loss['beta']:.6f} gamma={mean_loss['gamma']:.6f} "
        f"both={mean_loss['both']:.6f}"
    )
    print(
        f"    [shapley] phi_beta={phi_beta:+.6f}, phi_gamma={phi_gamma:+.6f}, "
        f"total={value['both']:+.6f}, efficiency gap={efficiency_gap:.2e}, "
        f"rho={rho}, coalition={shapley_choice}"
    )
    if efficiency_gap > 1e-9:
        raise AssertionError("Shapley efficiency violated; check the value function.")

    total = abs(phi_beta) + abs(phi_gamma)
    return {
        "rho": rho,
        "loss_to": mean_loss["none"],
        "loss_meta": mean_loss["both"],
        "loss_beta_only": mean_loss["beta"],
        "loss_gamma_only": mean_loss["gamma"],
        "transfer_margin": float(mean_loss["none"] - mean_loss["both"]),
        "epsilon0": float(epsilon0),
        "value": value,
        "phi_beta": float(phi_beta),
        "phi_gamma": float(phi_gamma),
        "phi_total": float(value["both"]),
        "phi_beta_share": float(abs(phi_beta) / total) if total > 0 else np.nan,
        "phi_gamma_share": float(abs(phi_gamma) / total) if total > 0 else np.nan,
        "efficiency_gap": float(efficiency_gap),
        "shapley_coalition": shapley_choice,
        "fold_records": fold_records,
        "fold_to_losses": [float(v) for v in fold_losses["none"]],
        "fold_meta_losses": [float(v) for v in fold_losses["both"]],
        "fold_beta_losses": [float(v) for v in fold_losses["beta"]],
        "fold_gamma_losses": [float(v) for v in fold_losses["gamma"]],
        "fold_loss_sd_to": float(np.std(fold_losses["none"], ddof=1)),
        "fold_loss_sd_meta": float(np.std(fold_losses["both"], ddof=1)),
    }


def fit_beta(theta_intercept, theta_x, theta_s, X_target_unlab, S_target_unlab,
             X_TL, Y_TL, seed, label="beta"):
    """The outcome model: image-only L1 logistic on TL and TU.

    TL keeps its observed labels and TU gets the soft labels
    m = g(eta(z; theta)) of the given coalition; no binarisation is applied.
    The penalty is chosen by cross-validation over the rows the model is
    fitted on (stratified folds on TL, plain folds on TU), so on TU the
    validation target is the soft label. The folds are drawn from ``seed``,
    and run_strifle passes a different seed for each coalition.
    """
    m = expit(theta_intercept
              + np.hstack([X_target_unlab, S_target_unlab])
              @ np.hstack([theta_x, theta_s]))
    X_soft = np.vstack([X_TL, X_target_unlab]).astype(np.float64, copy=False)
    y_soft = np.hstack([Y_TL, m]).astype(np.float64, copy=False)
    n_tl = len(Y_TL)

    classes, counts = np.unique(Y_TL, return_counts=True)
    if len(classes) != 2 or int(counts.min()) < 2:
        raise ValueError(f"{label}: final beta CV needs at least two TL rows per class.")
    n_splits = min(int(STRIFLE_TUNING_FOLDS), int(counts.min()),
                   max(2, len(X_target_unlab)))

    fold_id = np.full(len(X_soft), -1, dtype=int)
    for k, (_, val_idx) in enumerate(
        StratifiedKFold(n_splits=n_splits, shuffle=True,
                        random_state=int(seed)).split(X_TL, Y_TL)
    ):
        fold_id[val_idx] = k
    if len(X_target_unlab) >= n_splits:
        for k, (_, val_idx) in enumerate(
            KFold(n_splits=n_splits, shuffle=True,
                  random_state=int(seed) + 1).split(X_target_unlab)
        ):
            fold_id[n_tl + val_idx] = k
    else:
        fold_id[n_tl:] = np.arange(len(X_target_unlab)) % n_splits
    if np.any(fold_id < 0):
        raise RuntimeError(f"{label}: failed to assign every row to a CV fold.")

    Cs = np.asarray(STRIFLE_C_GRID, dtype=np.float64)
    fold_losses = np.zeros((n_splits, len(Cs)), dtype=np.float64)
    for k in range(n_splits):
        train_mask = fold_id != k
        X_train, y_train_soft = X_soft[train_mask], y_soft[train_mask]
        # A soft label y in [0,1] is fit by duplicating the row with labels 1
        # and 0 carrying weights y and 1-y; the total weight per original row
        # is one, so the penalty scale matches an ordinary fit.
        X_doubled = np.vstack([X_train, X_train])
        y_doubled = np.hstack([np.ones(len(X_train), int), np.zeros(len(X_train), int)])
        train_weights = np.hstack([y_train_soft, 1.0 - y_train_soft])
        model = None
        for c_idx, C in enumerate(Cs):
            if model is None:
                model = _fit_l1_logistic_fixed_c(
                    X_doubled, y_doubled, C, seed + k,
                    sample_weight=train_weights, warm_start=True,
                )
            else:
                model.set_params(C=float(C))
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_doubled, y_doubled, sample_weight=train_weights)
            fold_losses[k, c_idx] = _weighted_log_loss_from_logits(
                y_soft[~train_mask], model.decision_function(X_soft[~train_mask])
            )

    mean_losses = fold_losses.mean(axis=0)
    best_C = float(Cs[int(np.argmin(mean_losses))])
    X_full = np.vstack([X_soft, X_soft])
    y_full = np.hstack([np.ones(len(X_soft), int), np.zeros(len(X_soft), int)])
    fitted = _fit_l1_logistic_fixed_c(
        X_full, y_full, best_C, seed + 200003,
        sample_weight=np.hstack([y_soft, 1.0 - y_soft]),
    )
    print(f"    [{label}] C={best_C:.4g}, "
          f"nonzero={int(np.count_nonzero(fitted.coef_))}/{X_soft.shape[1]}")
    return (float(fitted.intercept_[0]), fitted.coef_[0], best_C,
            int(np.count_nonzero(fitted.coef_)))


# ============================================================
# STRIFLE driver: six endpoints from one set of estimating stages
# ============================================================

def run_strifle(X_TL_feat, Y_TL, S_TL, XS_SL, w_hat, Y_SL,
                X_target_unlab_feat, S_target_unlab,
                eval_sets, feat_dim, seed, tag=""):
    """Fit every endpoint and score each on every evaluation set.

    FitPair on all of TL gives the target-only and meta fits that the four
    coalitions share; each coalition gets its intercept refitted on TL and one
    outcome model. The two gate endpoints reuse the outcome model of the
    coalition their rule selects at STRIFLE_GATE_EPSILON.

    ``eval_sets`` maps a name to (X, S, Y, patient_ids). The outcome model is
    image-only, so S is carried only for bookkeeping.
    """
    XS_TL = np.hstack([X_TL_feat, S_TL])
    nT = len(Y_TL)

    to_model, _, _ = fit_l1_logistic_cv(XS_TL, Y_TL, seed + 1, label=f"{tag}-theta-T")
    to = (float(to_model.intercept_[0]),
          to_model.coef_[0, :feat_dim], to_model.coef_[0, feat_dim:])

    fit_weights = np.hstack([w_hat, np.ones(nT, dtype=np.float64)])
    cv_weights = (
        np.hstack([np.zeros(len(Y_SL)), np.ones(nT)])
        if STRIFLE_META_CV_TARGET_ONLY else None
    )
    meta_model, meta_C, _ = fit_l1_logistic_cv(
        np.vstack([XS_SL, XS_TL]), np.hstack([Y_SL, Y_TL]), seed + 2,
        sample_weight=fit_weights, cv_weight=cv_weights,
        label=f"{tag}-theta-tilde",
    )
    tilde = (float(meta_model.intercept_[0]),
             meta_model.coef_[0, :feat_dim], meta_model.coef_[0, feat_dim:])
    correction = fit_l1_bias_correction_cv(
        tilde[0], tilde[1], tilde[2], X_TL_feat, S_TL, Y_TL, seed + 3,
        label=f"{tag}-delta",
    )
    meta = (tilde[0] + correction["delta_intercept"],
            tilde[1] + correction["delta_beta"],
            tilde[2] + correction["delta_gamma"])

    gate = cross_fitted_shapley_gate(
        X_TL_feat, Y_TL, S_TL, XS_SL, w_hat, Y_SL, feat_dim, seed + 4,
        epsilon0=STRIFLE_GATE_EPSILON, label=f"{tag}-gate",
    )

    # Full-data theta for each coalition, with the intercept refit on TL.
    coalition_theta, coalition_beta = {}, {}
    for coalition in COALITIONS:
        beta, gamma = _coalition_slopes(to[1], to[2], meta[1], meta[2], coalition)
        intercept = _refit_intercept(X_TL_feat @ beta + S_TL @ gamma, Y_TL)
        coalition_theta[coalition] = (intercept, beta, gamma)
        coalition_beta[coalition] = fit_beta(
            intercept, beta, gamma, X_target_unlab_feat, S_target_unlab,
            X_TL_feat, Y_TL, seed + 5 + COALITIONS.index(coalition),
            label=f"{tag}-beta-{coalition}",
        )

    endpoint_to_coalition = {
        "to": "none",
        "corrected_meta": "both",
        "beta_only": "beta",
        "gamma_only": "gamma",
        "paper_gate": "both" if gate["rho"] == 1 else "none",
        "shapley_gate": gate["shapley_coalition"],
    }

    results = {
        "rho": int(gate["rho"]),
        "gate": gate,
        "selected_endpoint_paper": endpoint_to_coalition["paper_gate"],
        "selected_endpoint_shapley": endpoint_to_coalition["shapley_gate"],
        "theta_to_nonzero": int(np.count_nonzero(to_model.coef_)),
        "theta_to_l1": float(np.abs(to_model.coef_).sum()),
        "theta_tilde_C": float(meta_C),
        "theta_tilde_nonzero": int(np.count_nonzero(meta_model.coef_)),
        "theta_meta_nonzero": int(np.count_nonzero(np.hstack([meta[1], meta[2]]))),
        "theta_meta_l1": float(np.abs(np.hstack([meta[1], meta[2]])).sum()),
        "delta_lambda": float(correction["lambda"]),
        "delta_nonzero": int(correction["nonzero"]),
        "delta_l1_norm": float(correction["delta_l1_norm"]),
        "delta_beta_l1": float(correction["delta_beta_l1"]),
        "delta_gamma_l1": float(correction["delta_gamma_l1"]),
        "endpoint_to_coalition": endpoint_to_coalition,
        "metrics": {},
        "predictions": {},
    }

    for endpoint, coalition in endpoint_to_coalition.items():
        intercept_f, beta_f, best_C, nonzero = coalition_beta[coalition]
        results[f"beta_{endpoint}_C"] = float(best_C)
        results[f"beta_{endpoint}_nonzero"] = int(nonzero)
        results[f"beta_{endpoint}_intercept"] = float(intercept_f)
        for eval_name, (X_eval, _S_eval, Y_eval, groups) in eval_sets.items():
            p = expit(intercept_f + X_eval @ beta_f)
            results["metrics"][f"{endpoint}|{eval_name}"] = evaluate_predictions(
                Y_eval, p, groups
            )
            results["predictions"][f"{endpoint}|{eval_name}"] = p.astype(np.float32)

    for eval_name in eval_sets:
        auc_to = results["metrics"][f"to|{eval_name}"]["auc"]
        line = "  ".join(
            f"{ep}={results['metrics'][f'{ep}|{eval_name}']['auc']:.4f}"
            for ep, _ in STRIFLE_VARIANTS
        )
        print(f"  [{tag}] {eval_name} AUC: {line}")
        pr = "  ".join(
            f"{ep}={results['metrics'][f'{ep}|{eval_name}']['pr_auc']:.4f}"
            for ep, _ in STRIFLE_VARIANTS
        )
        print(f"  [{tag}] {eval_name} PR-AUC: {pr}   (baseline="
              f"{results['metrics'][f'to|{eval_name}']['prevalence']:.4f})")
        del auc_to
    return results


# ============================================================
# External comparators on the image block
# ============================================================

def fit_suptrans(X_SL_feat, Y_SL, X_TL_feat, Y_TL, seed, label="suptrans"):
    """Steps 1 and 2 of SUPTRANS (Li, Cai and Duan 2023), on X alone.

    Step 1 fits an L1 logistic model on the source labelled rows alone; step 2
    fits a sparse correction eta on the target labelled rows around that fixed
    offset, and their sum is returned. Neither the surrogates nor the
    unlabelled rows enter. This is not the SUPTRANS the paper reports: the
    paper uses the complete three-step procedure (hard-thresholded correction,
    then a joint penalised refit on SL and TL), which
    recompute_suptrans_three_step.py computes from the saved features and
    whose steps 1 and 2 reproduce this fit exactly. The endpoint is kept here
    under its original name so that comparison stays possible.
    """
    X_SL_feat = np.asarray(X_SL_feat, dtype=np.float64)
    X_TL_feat = np.asarray(X_TL_feat, dtype=np.float64)
    Y_SL = np.asarray(Y_SL, dtype=int).ravel()
    Y_TL = np.asarray(Y_TL, dtype=int).ravel()

    source_model, source_C, _ = fit_l1_logistic_cv(
        X_SL_feat, Y_SL, seed + 1, label=f"{label}-beta-source"
    )
    beta_source = source_model.coef_[0]
    intercept_source = float(source_model.intercept_[0])

    # Reuse the offset-Lasso solver by passing an empty surrogate block, so the
    # correction is the same estimator as delta but on X alone.
    empty_s = np.zeros((len(X_TL_feat), 0), dtype=np.float64)
    correction = fit_l1_bias_correction_cv(
        intercept_source, beta_source, np.zeros(0, dtype=np.float64),
        X_TL_feat, empty_s, Y_TL, seed + 2, label=f"{label}-eta",
    )
    intercept = intercept_source + correction["delta_intercept"]
    beta = beta_source + correction["delta_beta"]
    info = {
        "source_C": float(source_C),
        "source_nonzero": int(np.count_nonzero(beta_source)),
        "eta_lambda": float(correction["lambda"]),
        "eta_nonzero": int(correction["nonzero"]),
        "eta_l1_norm": float(correction["delta_l1_norm"]),
        "final_nonzero": int(np.count_nonzero(beta)),
    }
    print(f"    [{label}] source nonzero={info['source_nonzero']}, "
          f"eta nonzero={info['eta_nonzero']}, "
          f"||eta||_1={info['eta_l1_norm']:.4f}")
    return intercept, beta, info


def run_image_lr_baselines(X_TL_feat, Y_TL, X_SL_feat, Y_SL, w_x_hat,
                           eval_sets, seed, tag="",
                           X_ceiling=None, Y_ceiling=None):
    """SUP, CS, the two-step suptrans fit and the Ceiling, all on X alone.

    SUP is a ridge logistic fit (C = OUTCOME_LR_C) on TL; CS the same on SL and
    TL with the image-only density ratio on the source rows; the Ceiling the
    same on every TU label, scored on VALID only.
    """
    lr_to = LogisticRegression(C=OUTCOME_LR_C, solver="lbfgs", max_iter=2000,
                               random_state=seed).fit(X_TL_feat, Y_TL)
    lr_dr = LogisticRegression(C=OUTCOME_LR_C, solver="lbfgs", max_iter=2000,
                               random_state=seed)
    lr_dr.fit(np.vstack([X_SL_feat, X_TL_feat]), np.hstack([Y_SL, Y_TL]),
              sample_weight=np.hstack([w_x_hat, np.ones(len(Y_TL), np.float32)]))

    out = {"metrics": {}, "predictions": {}, "suptrans_info": None}
    # Prefixed so these never collide with the STRIFLE endpoint named "to",
    # which would make the paired lookup ambiguous.
    for endpoint, model in (("lr_to", lr_to), ("lr_dr", lr_dr)):
        for eval_name, (X_eval, _S, Y_eval, groups) in eval_sets.items():
            p = model.predict_proba(X_eval)[:, 1]
            out["metrics"][f"{endpoint}|{eval_name}"] = evaluate_predictions(
                Y_eval, p, groups
            )
            out["predictions"][f"{endpoint}|{eval_name}"] = p.astype(np.float32)

    intercept_st, beta_st, out["suptrans_info"] = fit_suptrans(
        X_SL_feat, Y_SL, X_TL_feat, Y_TL, seed + 7, label=f"{tag}-suptrans"
    )
    for eval_name, (X_eval, _S, Y_eval, groups) in eval_sets.items():
        p = expit(intercept_st + np.asarray(X_eval, dtype=np.float64) @ beta_st)
        out["metrics"][f"suptrans|{eval_name}"] = evaluate_predictions(
            Y_eval, p, groups
        )
        out["predictions"][f"suptrans|{eval_name}"] = p.astype(np.float32)

    # Ceiling: the same ridge logistic model fitted on every TU label, outside
    # the label budget every other model respects. It says how far NT target
    # labels get rather than competing with the estimators, so it is reported
    # as a level on VALID and kept out of the paired contrasts. The
    # representation has read TU, so even this level is transductive.
    if X_ceiling is not None and len(np.unique(Y_ceiling)) == 2:
        ceiling = LogisticRegression(C=OUTCOME_LR_C, solver="lbfgs",
                                     max_iter=2000, random_state=seed)
        ceiling.fit(X_ceiling, Y_ceiling)
        for eval_name, (X_eval, _S, Y_eval, groups) in eval_sets.items():
            if eval_name == "tu_transductive":
                continue      # in-sample for this model, so not scored
            p = ceiling.predict_proba(X_eval)[:, 1]
            out["metrics"][f"target_ceiling|{eval_name}"] = evaluate_predictions(
                Y_eval, p, groups
            )
            out["predictions"][f"target_ceiling|{eval_name}"] = p.astype(np.float32)

    for eval_name in eval_sets:
        line = (f"  [{tag}] {eval_name} baselines AUC: "
                f"SUP={out['metrics'][f'lr_to|{eval_name}']['auc']:.4f}  "
                f"CS={out['metrics'][f'lr_dr|{eval_name}']['auc']:.4f}  "
                f"suptrans(2 steps)="
                f"{out['metrics'][f'suptrans|{eval_name}']['auc']:.4f}")
        key = f"target_ceiling|{eval_name}"
        if key in out["metrics"]:
            line += f"  ceiling={out['metrics'][key]['auc']:.4f}"
        print(line)
    return out


# ============================================================
# Supervised branch (code name no_ssl): fine-tuned DenseNet-121,
# which also gives the CNN endpoint
# ============================================================

def split_tl_train_val(Y_TL, nT, seed, val_frac=0.2):
    """Stratified 80:20 split of TL. The first part is pooled with SL for
    training; the second is used only for early stopping."""
    Y_TL = np.asarray(Y_TL)
    pos_idx = np.where(Y_TL == 1)[0]
    neg_idx = np.where(Y_TL == 0)[0]
    rng = np.random.default_rng(seed + int(nT))
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    nv_pos = max(1, int(len(pos_idx) * val_frac)) if len(pos_idx) else 0
    nv_neg = max(1, int(len(neg_idx) * val_frac)) if len(neg_idx) else 0
    val_idx = np.hstack([pos_idx[:nv_pos], neg_idx[:nv_neg]])
    train_idx = np.hstack([pos_idx[nv_pos:], neg_idx[nv_neg:]])
    rng.shuffle(val_idx); rng.shuffle(train_idx)
    return train_idx, val_idx


def freeze_densenet_by_nT(model, nT):
    """Freeze the early layers by NT. At NT = 200 the initial convolution, the
    initial normalisation, dense blocks 1-3 and transitions 1-3 are frozen; at
    NT = 400 the same through block 2. The paper uses only these two arms; the
    last case (NT > 500, nothing frozen) is never reached."""
    if nT <= 200:
        prefixes = ("features.conv0", "features.norm0", "features.denseblock1",
                    "features.transition1", "features.denseblock2",
                    "features.transition2", "features.denseblock3",
                    "features.transition3")
        desc = "freeze blocks 1-3"
    elif nT <= 500:
        prefixes = ("features.conv0", "features.norm0", "features.denseblock1",
                    "features.transition1", "features.denseblock2",
                    "features.transition2")
        desc = "freeze blocks 1-2"
    else:
        prefixes, desc = (), "finetune all"
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [freeze] {desc}; trainable={sum(p.numel() for p in trainable):,}")
    return trainable


def joint_finetune_cnn_dr(X_SL_raw, Y_SL, w_pixel, X_TL_raw, Y_TL, nT, seed,
                          max_epochs=NO_SSL_MAX_EPOCHS, patience=NO_SSL_PATIENCE):
    """Fine-tune the ImageNet DenseNet-121, with a single-logit classifier, on
    SL and the training part of TL.

    Source rows carry the pixel-space density ratio and target rows unit
    weight, through a weighted binary cross-entropy with no class weights.
    AdamW with weight decay 1e-4 and no scheduler; batch size and learning
    rate depend on NT (8 and 5e-5 at 200, 16 and 1e-4 at 400). Training stops
    after max_epochs (50), or after `patience` (5) epochs without an
    improvement of at least 1e-5 in the TL validation loss, and the best state
    is kept. Returns the fitted network, which is the CNN endpoint, and a copy
    with the classifier replaced by the identity, which gives the
    1024-dimensional Supervised features.
    """
    batch_size, lr =(8, 5e-5) if nT <= 200 else ((16, 1e-4) if nT <= 500 else (32, 2e-4))
    train_idx, val_idx = split_tl_train_val(Y_TL, nT=nT, seed=seed)
    X_TL_train, Y_TL_train = X_TL_raw[train_idx], Y_TL[train_idx]
    X_TL_val, Y_TL_val = X_TL_raw[val_idx], Y_TL[val_idx]
    print(f"  [noSSL finetune] nT={nT}, TL_train={len(train_idx)}, "
          f"TL_val={len(val_idx)}, batch={batch_size}, lr={lr:.0e}")

    X_all = np.vstack([X_SL_raw, X_TL_train]).astype(np.float32)
    Y_all = np.hstack([Y_SL, Y_TL_train]).astype(np.float32)
    W_all = np.hstack([w_pixel, np.ones(len(train_idx), np.float32)]).astype(np.float32)

    # Uniform sampling: the density ratio enters exactly once, in the weighted
    # loss. Sampling by W_all as well would optimise a squared-weight objective.
    loader = DataLoader(
        RawImageDataset(X_all, Y_all, weights=W_all, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed + int(nT)),
    )
    loader_val = DataLoader(RawImageDataset(X_TL_val, Y_TL_val, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_densenet121(mode="finetune")
    trainable = freeze_densenet_by_nT(model, nT)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    use_amp = DEVICE.type == "cuda"
    scaler = _grad_scaler(use_amp)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train(); train_loss = 0.0
        for X_b, Y_b, W_b in loader:
            X_b = X_b.to(DEVICE)
            Y_b = Y_b.to(DEVICE).unsqueeze(1)
            W_b = W_b.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(use_amp):
                loss = (criterion(model(X_b), Y_b) * W_b).mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * len(Y_b)
        train_loss /= len(Y_all)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b, _ in loader_val:
                logits = model(X_b.to(DEVICE))
                val_loss += criterion(logits, Y_b.to(DEVICE).unsqueeze(1)).mean().item() * len(Y_b)
        val_loss /= max(len(Y_TL_val), 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stop @ epoch {epoch}, best_val={best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    predictor = model.eval()
    extractor = copy.deepcopy(model)
    extractor.classifier = nn.Identity()
    extractor.eval()
    free_memory(X_all, Y_all, W_all, loader, loader_val, optimizer)
    return predictor, extractor


# ============================================================
# Patient split
# ============================================================

def patient_split_tl_valid_tu(patient_ids, target_labels, n_tl, n_valid,
                              max_images_per_patient, positive_ratio, seed,
                              min_tl_positive_count=4, min_tl_negative_count=4,
                              min_valid_positive_count=4, min_valid_negative_count=4,
                              max_sampling_attempts=2000):
    """Patient-disjoint TL, VALID and TU for one repeat.

    From target patients with at most max_images_per_patient images, n_tl TL
    patients and n_valid VALID patients are drawn, one image each, the two
    sets disjoint and each stratified to the cohort event rate by fixed
    count (positive_ratio, subject to the minimum class counts). The cap
    applies to both, since patients with many films are inpatients and so at
    elevated pneumothorax risk; were it applied to TL alone, the labelled
    sample would come from a different patient population than VALID. Every
    image of every unselected patient forms TU, which has no cap. The other
    images of selected patients are excluded from every fit
    (excluded_sibling_idx), and target_fit_idx is TL together with TU.
    """
    patient_ids = np.asarray(patient_ids).astype(str)
    labels = np.asarray(target_labels).astype(np.int8, copy=False)
    if len(labels) != len(patient_ids):
        raise ValueError("target_labels and patient_ids must have equal length.")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("target_labels must be binary 0/1.")

    def class_counts(n_rows, min_pos, min_neg):
        n_pos = max(int(min_pos), int(round(n_rows * positive_ratio)))
        n_pos = min(n_pos, int(n_rows) - int(min_neg))
        if not 0 < n_pos < n_rows:
            raise ValueError(f"Invalid stratified class count for n={n_rows}.")
        return n_pos, int(n_rows) - n_pos

    tl_pos, tl_neg = class_counts(n_tl, min_tl_positive_count, min_tl_negative_count)
    valid_pos, valid_neg = class_counts(n_valid, min_valid_positive_count,
                                        min_valid_negative_count)

    table = pd.DataFrame({"row_idx": np.arange(len(patient_ids)), "patient_id": patient_ids})
    grouped = table.groupby("patient_id", sort=False)
    patient_rows = {str(p): np.asarray(r, dtype=int) for p, r in grouped.indices.items()}
    patient_counts = grouped.size()
    all_patients = patient_counts.index.to_numpy(dtype=str)
    eligible = patient_counts[patient_counts <= max_images_per_patient].index.to_numpy(str)

    def capable(patients, label):
        return np.asarray(
            [str(p) for p in patients if np.any(labels[patient_rows[str(p)]] == label)],
            dtype=str,
        )

    tl_pos_pool = capable(eligible, 1)
    tl_neg_pool = capable(eligible, 0)
    if len(tl_pos_pool) < tl_pos + valid_pos or len(tl_neg_pool) < tl_neg + valid_neg:
        raise ValueError(
            "TL and validation class composition is infeasible among patients "
            f"with at most {max_images_per_patient} images: need "
            f"{tl_pos + valid_pos} positive-capable and {tl_neg + valid_neg} "
            f"negative-capable; found {len(tl_pos_pool)} and {len(tl_neg_pool)}."
        )

    rng = np.random.default_rng(seed)
    for attempt in range(1, max_sampling_attempts + 1):
        tl_positive = rng.choice(tl_pos_pool, size=tl_pos, replace=False)
        tl_pos_set = set(tl_positive.tolist())
        neg_available = np.asarray([p for p in tl_neg_pool if p not in tl_pos_set], str)
        if len(neg_available) < tl_neg:
            continue
        tl_negative = rng.choice(neg_available, size=tl_neg, replace=False)
        tl_patients = np.concatenate([tl_positive, tl_negative])
        tl_set = set(tl_patients.tolist())

        remaining = np.asarray([p for p in eligible if p not in tl_set], dtype=str)
        valid_pos_pool = capable(remaining, 1)
        valid_neg_pool = capable(remaining, 0)
        if len(valid_pos_pool) < valid_pos:
            continue
        valid_positive = rng.choice(valid_pos_pool, size=valid_pos, replace=False)
        valid_pos_set = set(valid_positive.tolist())
        neg_available = np.asarray(
            [p for p in valid_neg_pool if p not in valid_pos_set], dtype=str
        )
        if len(neg_available) < valid_neg:
            continue
        valid_negative = rng.choice(neg_available, size=valid_neg, replace=False)
        valid_patients = np.concatenate([valid_positive, valid_negative])
        break
    else:
        raise RuntimeError(
            f"Could not build disjoint TL and validation patients in "
            f"{max_sampling_attempts} attempts."
        )

    def draw_rows(selected_positive, selected_negative):
        rows = []
        for p in selected_positive:
            candidates = patient_rows[str(p)]
            rows.append(int(rng.choice(candidates[labels[candidates] == 1])))
        for p in selected_negative:
            candidates = patient_rows[str(p)]
            rows.append(int(rng.choice(candidates[labels[candidates] == 0])))
        return np.sort(np.asarray(rows, dtype=int))

    tl_idx = draw_rows(tl_positive, tl_negative)
    valid_idx = draw_rows(valid_positive, valid_negative)
    selected_patients = set(tl_patients.tolist()) | set(valid_patients.tolist())
    selected_patient_mask = np.fromiter(
        (p in selected_patients for p in patient_ids), dtype=bool, count=len(patient_ids)
    )
    selected_image_mask = np.zeros(len(patient_ids), dtype=bool)
    selected_image_mask[tl_idx] = True
    selected_image_mask[valid_idx] = True
    tu_idx = np.flatnonzero(~selected_patient_mask)
    excluded_sibling_idx = np.flatnonzero(selected_patient_mask & ~selected_image_mask)
    target_fit_idx = np.sort(np.concatenate([tl_idx, tu_idx]))

    tl_patient_set = set(patient_ids[tl_idx].tolist())
    valid_patient_set = set(patient_ids[valid_idx].tolist())
    tu_patient_set = set(patient_ids[tu_idx].tolist())
    assert len(tl_idx) == n_tl and len(tl_patient_set) == n_tl
    assert len(valid_idx) == n_valid and len(valid_patient_set) == n_valid
    assert int(labels[tl_idx].sum()) == tl_pos
    assert int(labels[valid_idx].sum()) == valid_pos
    assert tl_patient_set.isdisjoint(valid_patient_set)
    assert tl_patient_set.isdisjoint(tu_patient_set)
    assert valid_patient_set.isdisjoint(tu_patient_set)
    assert not np.intersect1d(target_fit_idx, valid_idx).size
    assert not np.intersect1d(target_fit_idx, excluded_sibling_idx).size

    return {
        "tl_idx": tl_idx, "valid_idx": valid_idx, "tu_idx": tu_idx,
        "target_fit_idx": target_fit_idx, "excluded_sibling_idx": excluded_sibling_idx,
        "tl_patients": tl_patients, "valid_patients": valid_patients,
        "eligible_patient_count": int(len(eligible)),
        "tl_positive_count": int(tl_pos), "tl_negative_count": int(tl_neg),
        "tl_positive_ratio": float(tl_pos / n_tl),
        "valid_positive_count": int(valid_pos), "valid_negative_count": int(valid_neg),
        "valid_positive_ratio": float(valid_pos / n_valid),
        "sampling_attempts": int(attempt),
    }


# ============================================================
# Loading and BioViL
# ============================================================

def load_prepared_split(split_name, text_dim, metadata_dim):
    path = os.path.join(PREP_DIR, f"{split_name}.npz")
    with np.load(path, allow_pickle=False) as data:
        required = {"X_image_uint8", "S_text_bow", "S_meta", "prepared_row_index"}
        if split_name in {"SL", "TARGET"}:
            required.add("Y")
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing NPZ keys: {missing}")
        images = data["X_image_uint8"]
        text = data["S_text_bow"].astype(np.float32, copy=False)
        metadata = data["S_meta"].astype(np.float32, copy=False)
        prepared_row_index = data["prepared_row_index"].astype(np.int64)
        labels = data["Y"].astype(np.float32) if "Y" in data.files else None

    if images.ndim != 3 or images.shape[1:] != (224, 224):
        raise ValueError(f"{split_name} images must be (N,224,224); got {images.shape}.")
    if images.dtype != np.uint8:
        raise ValueError(f"{split_name} images must be uint8; got {images.dtype}.")
    images = images.reshape(len(images), -1)
    if text.shape != (len(images), text_dim):
        raise ValueError(f"{split_name} text must be (N,{text_dim}); got {text.shape}.")
    if metadata.shape != (len(images), metadata_dim):
        raise ValueError(f"{split_name} metadata shape mismatch: {metadata.shape}.")
    if len(prepared_row_index) != len(images):
        raise ValueError(f"{split_name} prepared_row_index length mismatch.")
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"{split_name} Y length mismatch.")
    if not np.isfinite(text).all() or not np.isfinite(metadata).all():
        raise ValueError(f"{split_name} text/metadata contains NaN or infinity.")
    return {"X": images, "text": text, "meta": metadata, "Y": labels,
            "prepared_row_index": prepared_row_index}


def resolve_biovil_image_path(path_to_image):
    p = str(path_to_image)
    normalized = p.replace("\\", "/")
    candidates = [p, os.path.join(BIOVIL_IMAGE_BASE, p),
                  os.path.join(BIOVIL_IMAGE_BASE, p.lstrip("/\\"))]
    parts = Path(p).parts
    candidates.extend(os.path.join(BIOVIL_IMAGE_BASE, *parts[i:])
                      for i in range(len(parts)))
    for candidate in candidates:
        if os.path.exists(candidate) and \
                str(candidate).replace("\\", "/").endswith(normalized):
            return Path(candidate)
    raise FileNotFoundError(f"Cannot resolve BioViL input image: {p}")


def ordered_path_sha256(split_audit):
    digest = hashlib.sha256()
    for raw_path in split_audit["path_to_image"].astype(str):
        encoded = raw_path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def expected_biovil_manifest(split_audit, split_name):
    return {
        "cache_schema_version": BIOVIL_CACHE_SCHEMA_VERSION,
        "split_name": split_name,
        "row_count": int(len(split_audit)),
        "feature_dim": BIOVIL_FEAT_DIM,
        "dtype": "float32",
        "ordered_path_sha256": ordered_path_sha256(split_audit),
        "package_name": BIOVIL_PACKAGE_NAME,
        "package_version": BIOVIL_PACKAGE_VERSION,
        "model_type": BIOVIL_MODEL_TYPE,
        "model_weights": BIOVIL_WEIGHTS_ID,
        "model_weights_md5": BIOVIL_WEIGHTS_MD5,
        "embedding": "l2_normalized_projected_global_embedding",
    }


def load_or_extract_biovil_features(audit):
    """Load the cached embeddings, extracting only what is missing.

    Each split has one array and one manifest in BIOVIL_CACHE_DIR. A cached
    split is used only if its manifest matches: the same rows in the same
    order (a hash over the split's path_to_image list), the same package
    version, model and checkpoint. Any other split is extracted again.

    Extraction reads each image of the downsampled CheXpert release under
    BIOVIL_IMAGE_BASE as released, through the encoder's published
    preprocessing, rather than the stored 224x224 array, and keeps the
    L2-normalised projected global embedding. Images are read one at a time,
    so the first run over the ~100k-image cohort takes hours; later runs, and
    the scripts that import this pipeline, read the cache.
    """
    os.makedirs(BIOVIL_CACHE_DIR, exist_ok=True)
    engine, features = None, {}
    for split_name in ("SL", "SU", "TARGET"):
        split_audit = audit[audit["split_group"] == split_name].reset_index(drop=True)
        cache_path = os.path.join(BIOVIL_CACHE_DIR, f"X_{split_name}_biovil128.npy")
        manifest_path = os.path.join(
            BIOVIL_CACHE_DIR, f"X_{split_name}_biovil128.meta.json"
        )
        expected = expected_biovil_manifest(split_audit, split_name)

        cached = None
        if os.path.exists(cache_path) and os.path.exists(manifest_path):
            try:
                candidate = np.load(cache_path, mmap_mode="r")
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                mismatched = [k for k, v in expected.items() if manifest.get(k) != v]
                if candidate.shape == (len(split_audit), BIOVIL_FEAT_DIM) \
                        and candidate.dtype == np.dtype("float32") and not mismatched:
                    cached = candidate
                else:
                    print(f"  [BioViL cache ignored] {split_name}: {mismatched or 'shape/dtype'}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"  [BioViL cache ignored] {split_name}: {exc}")
        if cached is not None:
            print(f"  [BioViL cache] {split_name}: {cached.shape}")
            features[split_name] = np.asarray(cached)
            continue

        if engine is None:
            installed = installed_version(BIOVIL_PACKAGE_NAME)
            if installed != BIOVIL_PACKAGE_VERSION:
                raise RuntimeError(
                    f"{BIOVIL_PACKAGE_NAME}=={BIOVIL_PACKAGE_VERSION} required; "
                    f"found {installed}."
                )
            from health_multimodal.image import get_image_inference
            from health_multimodal.image.utils import ImageModelType
            engine = get_image_inference(ImageModelType[BIOVIL_MODEL_TYPE])

        # Progress is reported about twenty times per split. A fixed interval
        # goes silent on any split smaller than the interval, and extraction is
        # slow: each image of the downsampled release is read as released, one
        # at a time, rather than taken from the stored 224x224 arrays.
        block = np.empty((len(split_audit), BIOVIL_FEAT_DIM), dtype=np.float32)
        step = max(1, min(500, len(split_audit) // 20))
        started = time.time()
        for i, raw_path in enumerate(split_audit["path_to_image"].astype(str)):
            if i % step == 0 or i + 1 == len(split_audit):
                done = i + 1
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (len(split_audit) - done) / rate if rate > 0 else float("nan")
                print(f"  [BioViL {split_name}] {done}/{len(split_audit)}"
                      f"  {rate:.1f} img/s  eta {remaining / 60:.1f} min")
            with torch.inference_mode():
                embedding = engine.get_projected_global_embedding(
                    resolve_biovil_image_path(raw_path)
                )
            block[i] = embedding.detach().cpu().numpy().astype(np.float32).reshape(-1)
        np.save(cache_path, block)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(expected, f, indent=2, sort_keys=True)
        features[split_name] = block
        print(f"  [BioViL cache saved] {split_name}: {block.shape}")
    free_memory(engine)
    return features["SL"], features["SU"], features["TARGET"]


# ============================================================
# Result accumulation
# ============================================================

def append_metric_rows(rows, repeat, seed, branch, model, metrics_by_eval,
                       split_info, q_text, q_meta):
    """One row per (model, evaluation split) with every metric flattened."""
    for key, metrics in metrics_by_eval.items():
        endpoint, eval_name = key.split("|", 1)
        if endpoint != model:
            continue
        row = {
            "repeat": int(repeat), "seed": int(seed), "nT": NT,
            "branch": branch, "model": f"{branch}_{model}",
            "endpoint": model, "evaluation_split": eval_name,
            "q_text": int(q_text), "q_metadata": int(q_meta),
            "s_dim": int(q_text + q_meta),
            # The CNN endpoint (branch cnn_direct) reads pixels and has no
            # image or surrogate block, so p and d = p + q are undefined there
            # rather than zero.
            "p_image": (int(split_info["p_image"])
                        if split_info.get("p_image") is not None else np.nan),
            "d_total": (int(split_info["p_image"] + q_text + q_meta)
                        if split_info.get("p_image") is not None else np.nan),
            "n_tl_labeled": int(split_info["n_tl_labeled"]),
            "n_tl_patients": int(split_info["n_tl_patients"]),
            "n_tl_positive": int(split_info["n_tl_positive"]),
            "n_target_unlabeled_total": int(split_info["n_target_unlabeled_total"]),
            "n_excluded_selected_patient_sibling_images":
                int(split_info["n_excluded_selected_patient_siblings"]),
        }
        row.update({k: v for k, v in metrics.items()})
        rows.append(row)


def paired_differences(rows_frame, reference_endpoint="to"):
    """Within-repeat, within-branch differences against Target-Only.

    VALID carries 21 cases, so the level of any one held-out figure is
    governed largely by which rows fell into it. Every endpoint of a branch is
    scored on the same split within a repeat, so their difference is
    attributable to the endpoint while the level of either is not. The CNN
    endpoint has no Target-Only row in its branch and drops out here.
    """
    if rows_frame.empty:
        return pd.DataFrame()
    keys = ["repeat", "branch", "evaluation_split"]
    metrics = ["auc", "pr_auc", "f1", "brier", "ece", "calibration_slope"]
    # The Ceiling reads every TU label, so it is a reference reported as a
    # level and kept out of the paired contrasts.
    rows_frame = rows_frame[rows_frame["endpoint"] != "target_ceiling"]
    reference = rows_frame[rows_frame["endpoint"] == reference_endpoint]
    if reference.empty:
        return pd.DataFrame()
    duplicated = reference.duplicated(subset=keys).any()
    if duplicated:
        raise AssertionError(
            f"Endpoint {reference_endpoint!r} appears more than once per "
            f"{keys}; endpoint names must be unique within a branch."
        )
    reference = reference.set_index(keys)[metrics].sort_index()
    lookup = reference.to_dict(orient="index")
    out = []
    for _, row in rows_frame.iterrows():
        key = tuple(row[k] for k in keys)
        base = lookup.get(key)
        if base is None:
            continue
        record = {k: row[k] for k in keys}
        record["endpoint"] = row["endpoint"]
        record["model"] = row["model"]
        for metric in metrics:
            value, reference_value = row.get(metric), base.get(metric)
            record[f"delta_{metric}"] = (
                float(value - reference_value)
                if pd.notna(value) and pd.notna(reference_value) else np.nan
            )
        out.append(record)
    return pd.DataFrame(out)


def summarize_paired(paired_frame):
    """Mean paired difference over repeats with a 95% t interval, the number
    of positive differences and a Wilcoxon signed-rank p-value."""
    if paired_frame.empty:
        return pd.DataFrame()
    from scipy import stats

    records = []
    grouped = paired_frame.groupby(["branch", "endpoint", "evaluation_split"])
    for (branch, endpoint, eval_name), group in grouped:
        record = {"branch": branch, "endpoint": endpoint,
                  "evaluation_split": eval_name, "n_repeats": int(len(group))}
        for metric in ("auc", "pr_auc", "brier", "ece"):
            values = group[f"delta_{metric}"].dropna().to_numpy()
            if len(values) < 2:
                record[f"delta_{metric}_mean"] = np.nan
                continue
            mean = float(values.mean())
            sem = float(stats.sem(values)) if values.std(ddof=1) > 0 else 0.0
            half = float(stats.t.ppf(0.975, len(values) - 1) * sem) if sem > 0 else 0.0
            record[f"delta_{metric}_mean"] = mean
            record[f"delta_{metric}_ci_low"] = mean - half
            record[f"delta_{metric}_ci_high"] = mean + half
            record[f"delta_{metric}_n_positive"] = int((values > 0).sum())
            if len(values) >= 3 and values.std(ddof=1) > 0:
                record[f"delta_{metric}_wilcoxon_p"] = float(
                    stats.wilcoxon(values, zero_method="zsplit").pvalue
                )
        records.append(record)
    return pd.DataFrame(records)


def summarize_raw(rows_frame):
    if rows_frame.empty:
        return pd.DataFrame()
    aggregations = {}
    for metric in ("auc", "pr_auc", "f1", "brier", "ece",
                   "calibration_slope", "calibration_in_the_large"):
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
    aggregations["n_runs"] = ("auc", "count")
    return (
        rows_frame.groupby(["branch", "endpoint", "evaluation_split"], as_index=False)
        .agg(**aggregations)
        .sort_values(["evaluation_split", "branch", "endpoint"])
    )


# ============================================================
# One image branch inside one repeat
# ============================================================

def run_branch(branch, X_SL_feat, X_SU_feat, X_TARGET_feat, S_SL, S_SU, S_TARGET,
               Y_SL, Y_TARGET, split, patient_ids, run_seed, repeat,
               split_info, rows, gate_rows, weight_records, q_text, q_meta):
    tl_idx = split["tl_idx"]
    tu_idx = split["tu_idx"]
    valid_idx = split["valid_idx"]

    X_TL, X_TU, X_VALID = (X_TARGET_feat[tl_idx], X_TARGET_feat[tu_idx],
                           X_TARGET_feat[valid_idx])
    S_TL, S_TU, S_VALID = S_TARGET[tl_idx], S_TARGET[tu_idx], S_TARGET[valid_idx]
    Y_TL, Y_TU, Y_VALID = Y_TARGET[tl_idx], Y_TARGET[tu_idx], Y_TARGET[valid_idx]

    # VALID enters no fit and is the primary evaluation set. TU is scored as
    # well, but it enters the representation, the standardisation, the
    # density ratio and the imputation, so its results are transductive and
    # reported separately.
    eval_sets = {
        "valid": (X_VALID, S_VALID, Y_VALID, patient_ids[valid_idx]),
        "tu_transductive": (X_TU, S_TU, Y_TU, patient_ids[tu_idx]),
    }

    split_info = dict(split_info)
    split_info["p_image"] = int(X_TARGET_feat.shape[1])
    print(f"\n--- branch: {branch} ---  image block p={split_info['p_image']}")
    w_image, diag_image = compute_feat_density_ratio_image_only(
        X_SU_feat, X_TU, X_SL_feat, seed=run_seed, tag=f"{branch} image-only"
    )
    baselines = run_image_lr_baselines(
        X_TL, Y_TL, X_SL_feat, Y_SL, w_image, eval_sets, run_seed,
        tag=f"{branch}", X_ceiling=X_TU, Y_ceiling=Y_TU,
    )
    for endpoint, _rule in BASELINE_VARIANTS:
        if any(k.startswith(f"{endpoint}|") for k in baselines["metrics"]):
            append_metric_rows(rows, repeat, run_seed, branch, endpoint,
                               baselines["metrics"], split_info, q_text, q_meta)

    XS_SL = np.hstack([X_SL_feat, S_SL]).astype(np.float32, copy=False)
    w_xs, diag_xs = compute_feat_density_ratio_xs(
        X_SU_feat, S_SU, X_TU, S_TU, X_SL_feat, S_SL, seed=run_seed,
        tag=f"{branch} paper X+S",
    )
    strifle = run_strifle(
        X_TL, Y_TL, S_TL, XS_SL, w_xs, Y_SL, X_TU, S_TU,
        eval_sets, feat_dim=int(X_TL.shape[1]), seed=run_seed,
        tag=f"{branch} STRIFLE",
    )
    for endpoint, _rule in STRIFLE_VARIANTS:
        append_metric_rows(rows, repeat, run_seed, branch, endpoint,
                           strifle["metrics"], split_info, q_text, q_meta)

    gate = strifle["gate"]
    gate_rows.append({
        "repeat": repeat, "seed": run_seed, "branch": branch, "nT": NT,
        "q_text": q_text, "q_metadata": q_meta,
        "rho": strifle["rho"],
        "selected_endpoint_paper": strifle["selected_endpoint_paper"],
        "selected_endpoint_shapley": strifle["selected_endpoint_shapley"],
        "loss_none": gate["loss_to"], "loss_beta": gate["loss_beta_only"],
        "loss_gamma": gate["loss_gamma_only"], "loss_both": gate["loss_meta"],
        "value_beta": gate["value"]["beta"], "value_gamma": gate["value"]["gamma"],
        "value_both": gate["value"]["both"],
        "phi_beta": gate["phi_beta"], "phi_gamma": gate["phi_gamma"],
        "phi_total": gate["phi_total"],
        "phi_beta_share": gate["phi_beta_share"],
        "phi_gamma_share": gate["phi_gamma_share"],
        "efficiency_gap": gate["efficiency_gap"],
        "transfer_margin": gate["transfer_margin"],
        "fold_to_losses": json.dumps(gate["fold_to_losses"]),
        "fold_meta_losses": json.dumps(gate["fold_meta_losses"]),
        "fold_beta_losses": json.dumps(gate["fold_beta_losses"]),
        "fold_gamma_losses": json.dumps(gate["fold_gamma_losses"]),
        "fold_loss_sd_to": gate["fold_loss_sd_to"],
        "fold_loss_sd_meta": gate["fold_loss_sd_meta"],
        "theta_to_nonzero": strifle["theta_to_nonzero"],
        "theta_to_l1": strifle["theta_to_l1"],
        "theta_tilde_C": strifle["theta_tilde_C"],
        "theta_tilde_nonzero": strifle["theta_tilde_nonzero"],
        "theta_meta_nonzero": strifle["theta_meta_nonzero"],
        "theta_meta_l1": strifle["theta_meta_l1"],
        "delta_lambda": strifle["delta_lambda"],
        "delta_nonzero": strifle["delta_nonzero"],
        "delta_l1_norm": strifle["delta_l1_norm"],
        "delta_beta_l1": strifle["delta_beta_l1"],
        "delta_gamma_l1": strifle["delta_gamma_l1"],
        "p_image": int(X_TL.shape[1]),
        "d_total": int(X_TL.shape[1] + q_text + q_meta),
        "l1_budget": float(np.sqrt(NT / np.log(X_TL.shape[1] + q_text + q_meta))),
        "suptrans_source_nonzero": baselines["suptrans_info"]["source_nonzero"],
        "suptrans_eta_nonzero": baselines["suptrans_info"]["eta_nonzero"],
        "suptrans_eta_l1_norm": baselines["suptrans_info"]["eta_l1_norm"],
        "dr_domain_auc": diag_xs["domain_auc"],
        "dr_retained": diag_xs["n_retained"],
        "dr_frac_clipped": diag_xs["frac_clipped"],
        "dr_effective_sample_size": diag_xs["effective_sample_size"],
    })

    weight_records.append({
        "repeat": repeat, "branch": branch, "kind": "image_only", **diag_image,
    })
    weight_records.append({
        "repeat": repeat, "branch": branch, "kind": "paper_x_plus_s", **diag_xs,
    })
    if SAVE_DENSITY_WEIGHTS:
        np.savez_compressed(
            os.path.join(WEIGHT_DIR, f"repeat{repeat}_{branch}_weights.npz"),
            w_image_only=w_image, w_paper_xs=w_xs,
            diagnostics_image=json.dumps(diag_image),
            diagnostics_xs=json.dumps(diag_xs),
        )
    return strifle, baselines, w_image


# ============================================================
# Main
# ============================================================

def load_completed_repeats(rows, gate_rows, weight_records, all_results):
    """Restore finished repeats so a restart resumes instead of redoing them.

    Each repeat writes every artefact only after all of its branches finish, so
    a repeat present in raw_results.csv is complete. Splits are derived from
    SEED + rep * 1000 + NT before the loop starts, so a resumed run reuses the
    identical patient assignments and the results remain one coherent
    experiment rather than a mixture of two.
    """
    if not RESUME:
        print("RESUME is off: starting from repeat 0 and overwriting results.")
        return set()

    raw_path = os.path.join(RESULT_DIR, "raw_results.csv")
    if not os.path.exists(raw_path):
        return set()

    try:
        prior = pd.read_csv(raw_path)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"[resume] could not read {raw_path} ({exc}); starting fresh")
        return set()
    if prior.empty or "repeat" not in prior.columns:
        return set()

    completed = {int(v) for v in prior["repeat"].dropna().unique()}
    prior = prior.drop_duplicates(
        subset=[c for c in ("repeat", "branch", "endpoint", "evaluation_split")
                if c in prior.columns]
    )
    rows.extend(prior.to_dict("records"))
    for filename, container in (
        ("gate_and_shapley_audit.csv", gate_rows),
        ("density_ratio_diagnostics.csv", weight_records),
    ):
        path = os.path.join(RESULT_DIR, filename)
        if os.path.exists(path):
            try:
                container.extend(pd.read_csv(path).to_dict("records"))
            except (OSError, pd.errors.ParserError) as exc:
                print(f"[resume] skipping {filename}: {exc}")

    results_path = os.path.join(RESULT_DIR, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, encoding="utf-8") as handle:
                all_results.update(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[resume] skipping results.json: {exc}")

    remaining = sorted(set(range(REPEATS)) - completed)
    print(f"\n[resume] {len(completed)} repeats already complete: "
          f"{sorted(completed)}")
    print(f"[resume] {len(remaining)} to run: {remaining}")
    if not remaining:
        print("[resume] nothing left to do; the final report will be rebuilt "
              "from the stored rows.")
    return completed


def main():
    seed_everything(SEED)
    print("=" * 88)
    print("Pneumothorax STRIFLE: History bag-of-words surrogate")
    print("Whitened image PCA, standardised surrogates, thresholded density "
          "ratio, coalition rule and Shapley block attribution")
    print("=" * 88)
    print(f"Prepared data : {PREP_DIR}")
    print(f"Results       : {RESULT_DIR}")

    required = ["SL.npz", "SU.npz", "TARGET.npz", "surrogate_vocabulary.csv",
                "metadata_feature_names.csv", "row_audit.csv", "prepare_config.json"]
    missing = [n for n in required if not os.path.exists(os.path.join(PREP_DIR, n))]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared files in {PREP_DIR}: {missing}. Run "
            "prepare_pneumothorax_canonical_blank0_source1200_history_bow_npz.py "
            "first, or set STRIFLE_PREP_DIR."
        )

    with open(os.path.join(PREP_DIR, "prepare_config.json"), encoding="utf-8") as h:
        prepare_config = json.load(h)
    if prepare_config.get("target_label") != "Pneumothorax":
        raise ValueError("This pipeline requires target_label='Pneumothorax'.")
    if prepare_config.get("text", {}).get("representation") != "bag_of_words":
        raise ValueError(
            "This pipeline requires the bag-of-words preparation output; found "
            f"{prepare_config.get('text', {}).get('representation')!r}."
        )
    if prepare_config.get("canonical_blank_included_as_zero") is not True:
        raise ValueError("This pipeline requires the blank-as-zero preparation.")

    vocabulary = pd.read_csv(os.path.join(PREP_DIR, "surrogate_vocabulary.csv"))
    metadata_spec = pd.read_csv(os.path.join(PREP_DIR, "metadata_feature_names.csv"))
    metadata_names = metadata_spec["feature"].astype(str).tolist()
    q_text = int(len(vocabulary))
    q_meta = int(len(metadata_names))
    q_total = q_text + q_meta
    # p here is the PCA cap of 128, the realised p of SimCLR and Supervised.
    # BioViL keeps fewer components (37 in the paper), so its d and log(d)/NT
    # are smaller; the gate rows record the realised p per branch, and the
    # replay computes eps0 from those.
    d_total = IMAGE_PCA_DIM + q_total
    print(f"Surrogate: q_text={q_text}, q_metadata={q_meta}, q={q_total}")
    print(f"d = p + q = {IMAGE_PCA_DIM} + {q_total} = {d_total};  "
          f"log(d)/nT = {np.log(d_total) / NT:.4f};  "
          f"L1 budget sqrt(nT/log d) = {np.sqrt(NT / np.log(d_total)):.2f}")

    print("\nLoading prepared NPZ files ...")
    sl = load_prepared_split("SL", q_text, q_meta)
    su = load_prepared_split("SU", q_text, q_meta)
    target = load_prepared_split("TARGET", q_text, q_meta)

    X_SL_raw, X_SU_raw, X_TARGET_raw = sl["X"], su["X"], target["X"]
    Y_SL, Y_TARGET = sl["Y"], target["Y"]
    T_SL_raw, T_SU_raw, T_TARGET_raw = sl["text"], su["text"], target["text"]
    M_SL_raw, M_SU_raw, M_TARGET_raw = sl["meta"], su["meta"], target["meta"]

    audit = pd.read_csv(os.path.join(PREP_DIR, "row_audit.csv"))
    ordered = []
    for name, prepared in (("SL", sl), ("SU", su), ("TARGET", target)):
        block = audit[audit["split_group"] == name].reset_index(drop=True)
        if len(block) != len(prepared["text"]):
            raise ValueError(f"{name} NPZ/audit row-count mismatch.")
        if not np.array_equal(block["prepared_row_index"].to_numpy(np.int64),
                              prepared["prepared_row_index"]):
            raise ValueError(f"{name} NPZ/audit row order mismatch.")
        ordered.append(block)
    audit = pd.concat(ordered, ignore_index=True)

    if "patient_id_parsed" not in audit.columns:
        raise KeyError("row_audit.csv must contain patient_id_parsed.")
    patient_ids_all = audit["patient_id_parsed"].astype(str).to_numpy()
    target_mask = audit["split_group"].to_numpy() == "TARGET"
    patient_ids_target = patient_ids_all[target_mask]
    patient_ids_source = patient_ids_all[~target_mask]
    if set(patient_ids_target).intersection(set(patient_ids_source)):
        print("  [WARN] a patient appears in both the source and target cohorts.")

    observed_prevalence = float(np.mean(Y_TARGET))
    print(f"Prepared shapes: SL={X_SL_raw.shape}, SU={X_SU_raw.shape}, "
          f"TARGET={X_TARGET_raw.shape}")
    print(f"Target prevalence={observed_prevalence:.6f}; "
          f"target patients={len(np.unique(patient_ids_target)):,}")
    print(f"nT={NT} -> {max(MIN_TL_POSITIVES, int(round(NT * observed_prevalence)))} "
          f"positives;  nVALID={N_VALID} -> "
          f"{max(MIN_VALID_POSITIVES, int(round(N_VALID * observed_prevalence)))} "
          "positives")

    print("\nPreflight of every repeat split ...")
    repeat_splits = []
    for rep in range(REPEATS):
        run_seed = SEED + rep * 1000 + NT
        split = patient_split_tl_valid_tu(
            patient_ids=patient_ids_target, target_labels=Y_TARGET,
            n_tl=NT, n_valid=N_VALID,
            max_images_per_patient=MAX_IMAGES_PER_PATIENT,
            positive_ratio=observed_prevalence, seed=run_seed,
            min_tl_positive_count=MIN_TL_POSITIVES,
            min_tl_negative_count=MIN_TL_NEGATIVES,
            min_valid_positive_count=MIN_VALID_POSITIVES,
            min_valid_negative_count=MIN_VALID_NEGATIVES,
        )
        repeat_splits.append(split)
    print(f"  {REPEATS} splits built; TL={NT}, VALID={N_VALID}, "
          f"TU={len(repeat_splits[0]['tu_idx'])}, "
          f"excluded siblings={len(repeat_splits[0]['excluded_sibling_idx'])}")

    print("\n" + "=" * 80)
    print("Frozen BioViL-128 features")
    print("=" * 80)
    X_SL_biovil, X_SU_biovil, X_TARGET_biovil = load_or_extract_biovil_features(audit)

    config = {
        "target_label": "Pneumothorax",
        "PREP_DIR": PREP_DIR, "RESULT_DIR": RESULT_DIR,
        "SEED": SEED, "REPEATS": REPEATS, "nT": NT, "nVALID": N_VALID,
        "max_images_per_patient": MAX_IMAGES_PER_PATIENT,
        "max_images_per_patient_applies_to": ["TL", "VALID"],
        "observed_target_prevalence": observed_prevalence,
        "dimensions": {"p_image": IMAGE_PCA_DIM, "q_text": q_text,
                       "q_metadata": q_meta, "q_total": q_total, "d_total": d_total,
                       "log_d_over_nT": float(np.log(d_total) / NT),
                       "l1_budget": float(np.sqrt(NT / np.log(d_total)))},
        "text_transform": ("prepared bag-of-words used directly; standardised "
                           "per repeat on SL+SU+TL+TU; no PCA or SVD"),
        "metadata_transform": ("every column standardised per repeat on "
                               "SL+SU+TL+TU; constant columns dropped in prepare"),
        "image_transform": {"simclr": f"DenseNet {FEAT_DIM} -> scaler -> PCA "
                                      f"{IMAGE_PCA_DIM}",
                            "biovil": f"official {BIOVIL_FEAT_DIM} -> scaler -> "
                                      f"PCA {IMAGE_PCA_DIM}",
                            "no_ssl": f"repeat-specific DenseNet {FEAT_DIM} -> "
                                      f"scaler -> PCA {IMAGE_PCA_DIM}",
                            "pca_whiten": PCA_WHITEN},
        "strifle_density_ratio": (
            "ridge domain classifier on a STANDARDISED pooled design matrix; "
            "slopes hard-thresholded at 10*sqrt(log d/(NS+NT)) in standardised "
            "intercept de-standardised for the thresholded coefficients (inert w.r.t. w_hat under the centring below); log ratio "
            "centred to E_S[w]=1 before clipping to [0.01, 10]"
        ),
        "strifle_penalty": "L1 for theta_T, theta_tilde, delta and the final beta",
        "strifle_C_grid": [float(v) for v in STRIFLE_C_GRID],
        "strifle_tuning_folds": STRIFLE_TUNING_FOLDS,
        "strifle_meta_cv_target_only": STRIFLE_META_CV_TARGET_ONLY,
        "strifle_meta_cv_weighting": (
            "penalty selected on target rows only, so the 1200 reweighted "
            "source rows do not dominate the validation loss"
            if STRIFLE_META_CV_TARGET_ONLY else
            "penalty selected under the fitting weights, as in the original "
            "script; the paper does not specify the CV loss weighting"
        ),
        "deviations_from_paper": {
            "weight_clipping": (
                f"density-ratio weights clipped to "
                f"[{DENSITY_CLIP_MIN}, {DENSITY_CLIP_MAX}] and renormalised. "
                "The paper does not clip. Clipping enforces in finite samples the bounded sup_z w0(z)/w_hat(z) that Assumption 1 requires; "
                "it bounds w_hat away from zero, which bounds the ratio given the assumption's bound on w0. Disclosed in the methods section."
            ),
            "density_ratio_estimator": (
                "logistic domain classifier rather than the tilting loss "
                "L^w_N(zeta); permitted by Remark 1 of the paper"
            ),
            "cv_folds": (
                "five folds, matching the paper's simulation setting; the "
                "paper's real-data analysis uses ten"
            ),
            "shapley_extension": (
                "four-coalition block attribution added alongside, not "
                "replacing, the paper's binary transferability gate"
            ),
        },
        "strifle_gate": ("paper equation (9): rho=1 when the corrected-meta "
                         "cross-fitted loss beats target-only by epsilon0"),
        "table2_comparators": {
            "SUP": "lr_to",
            "SAS": "to",
            "CS": "lr_dr",
            "SUPTrans": ("suptrans; three-step supervised transfer on beta "
                         "following Li et al. (2021); uses neither the "
                         "surrogates nor the unlabelled data"),
            "Meta": "corrected_meta",
            "STRIFLE": "paper_gate",
        },
        "target_supervised_ceiling": (
            "target_ceiling; fit on every TU outcome label, roughly 80 times "
            "the label budget of every other model. Reported on the held-out "
            "validation split only, as a reference for how far 200 target "
            "labels reach; excluded from the paired comparison. The "
            "representation was fit on TL and TU, so this ceiling is "
            "transductive."
        ),
        "strifle_gate_epsilon0": STRIFLE_GATE_EPSILON,
        "shapley": {
            "players": ["beta (image coefficients)", "gamma (surrogate coefficients)"],
            "coalitions": list(COALITIONS),
            "value_function": "reduction in two-fold cross-fitted target log loss "
                              "relative to the target-only estimator",
            "intercept": "refit unpenalised on the training fold for every coalition",
            "efficiency": "phi_beta + phi_gamma = v(both) exactly",
        },
        "evaluation": {
            "valid": "held out from every fit; primary endpoint",
            "tu_transductive": ("used for SimCLR, PCA, scaler, density ratio and "
                                "soft-label imputation, so its metrics are "
                                "transductive and reported as secondary"),
            "metrics": ["auc", "pr_auc", "pr_auc_lift", "f1", "brier",
                        "calibration_slope", "calibration_intercept",
                        "calibration_in_the_large", "ece", "mce"],
            "intervals": "patient-clustered bootstrap, 500 replicates",
            "paired": "within-repeat difference against the target-only endpoint",
        },
        "artefacts": {"save_features": SAVE_FEATURES, "save_backbone": SAVE_BACKBONE,
                      "save_density_weights": SAVE_DENSITY_WEIGHTS,
                      "resume": RESUME},
        "versions": {"torch": torch.__version__,
                     "sklearn": installed_version("scikit-learn"),
                     "numpy": np.__version__},
    }
    with open(os.path.join(RESULT_DIR, "config.json"), "w", encoding="utf-8") as h:
        json.dump(config, h, indent=2)

    rows, gate_rows, weight_records, all_results = [], [], [], {}
    completed_repeats = load_completed_repeats(
        rows, gate_rows, weight_records, all_results
    )

    for rep in range(REPEATS):
        if rep in completed_repeats:
            print(f"\nRepeat {rep + 1}/{REPEATS} already on disk, skipping")
            continue
        run_seed = SEED + rep * 1000 + NT
        seed_everything(run_seed)
        print("\n" + "#" * 90)
        print(f"Repeat {rep + 1}/{REPEATS}, nT={NT}, seed={run_seed}")
        print("#" * 90)

        split = repeat_splits[rep]
        tl_idx, tu_idx = split["tl_idx"], split["tu_idx"]
        valid_idx, target_fit_idx = split["valid_idx"], split["target_fit_idx"]
        Y_TL, Y_TU, Y_VALID = Y_TARGET[tl_idx], Y_TARGET[tu_idx], Y_TARGET[valid_idx]

        split_info = {
            "n_tl_labeled": len(tl_idx),
            "n_tl_patients": len(np.unique(patient_ids_target[tl_idx])),
            "n_tl_positive": int(Y_TL.sum()),
            "n_target_unlabeled_total": len(tu_idx),
            "n_excluded_selected_patient_siblings": len(split["excluded_sibling_idx"]),
            "n_valid_images": len(valid_idx),
            "n_valid_positive": int(Y_VALID.sum()),
            "n_tu_images": len(tu_idx),
            "n_tu_positive": int(Y_TU.sum()),
        }
        print(f"TL={len(tl_idx)} images / {split_info['n_tl_patients']} patients, "
              f"positives={int(Y_TL.sum())}")
        print(f"VALID={len(valid_idx)} images, positives={int(Y_VALID.sum())}; "
              f"TU={len(tu_idx)} images, positives={int(Y_TU.sum())}; "
              f"excluded siblings={len(split['excluded_sibling_idx'])}")

        split_dir = os.path.join(RESULT_DIR, "selected_indices")
        os.makedirs(split_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(split_dir, f"repeat{rep}_nT{NT}_patient_split.npz"),
            tl_idx=tl_idx, tu_idx=tu_idx, valid_idx=valid_idx,
            target_fit_idx=target_fit_idx,
            excluded_sibling_idx=split["excluded_sibling_idx"],
            tl_patients=split["tl_patients"].astype(str),
            valid_patients=split["valid_patients"].astype(str),
            tl_labels=Y_TL.astype(np.int8), valid_labels=Y_VALID.astype(np.int8),
            seed=run_seed,
        )

        run_key = f"rep{rep}_nT{NT}"
        all_results[run_key] = {"split_info": split_info, "seed": run_seed}

        # ---- repeat-specific surrogate standardisation -------------------
        T_SL, T_SU, T_TARGET = standardize_block(
            reference_arrays=[T_SL_raw, T_SU_raw, T_TARGET_raw[target_fit_idx]],
            transform_arrays=[T_SL_raw, T_SU_raw, T_TARGET_raw],
            tag=f"BoW text {run_key}", file_stem=f"text_bow_scaler_{run_key}",
        )
        M_SL, M_SU, M_TARGET = standardize_block(
            reference_arrays=[M_SL_raw, M_SU_raw, M_TARGET_raw[target_fit_idx]],
            transform_arrays=[M_SL_raw, M_SU_raw, M_TARGET_raw],
            tag=f"metadata {run_key}", file_stem=f"metadata_scaler_{run_key}",
        )
        S_SL = np.hstack([T_SL, M_SL]).astype(np.float32)
        S_SU = np.hstack([T_SU, M_SU]).astype(np.float32)
        S_TARGET = np.hstack([T_TARGET, M_TARGET]).astype(np.float32)
        if S_SL.shape[1] != q_total:
            raise AssertionError(f"Surrogate dimension {S_SL.shape[1]} != {q_total}.")

        # ---- SimCLR ------------------------------------------------------
        backbone, ssl_info = train_simclr(
            X_TARGET_raw[target_fit_idx], seed=run_seed,
            n_epochs=SIMCLR_EPOCHS, batch_size=SIMCLR_BATCH_SIZE,
        )
        all_results[run_key]["simclr"] = ssl_info
        if SAVE_BACKBONE:
            torch.save(backbone.state_dict(),
                       os.path.join(BACKBONE_DIR, f"simclr_backbone_{run_key}.pt"))
        X_SL_ssl_raw = extract_features(X_SL_raw, backbone, tag=f"SSL SL {run_key}")
        X_SU_ssl_raw = extract_features(X_SU_raw, backbone, tag=f"SSL SU {run_key}")
        X_TG_ssl_raw = extract_features(X_TARGET_raw, backbone, tag=f"SSL TARGET {run_key}")
        free_memory(backbone)
        X_SL_ssl, X_SU_ssl, X_TG_ssl = reduce_image_representation(
            X_SL_ssl_raw, X_SU_ssl_raw, X_TG_ssl_raw, target_fit_idx,
            tag=f"SimCLR {run_key}", file_stem=f"simclr_image_pca_{run_key}",
        )
        del X_SL_ssl_raw, X_SU_ssl_raw, X_TG_ssl_raw
        gc.collect()

        # ---- BioViL ------------------------------------------------------
        X_SL_bio, X_SU_bio, X_TG_bio = reduce_image_representation(
            X_SL_biovil, X_SU_biovil, X_TARGET_biovil, target_fit_idx,
            tag=f"BioViL {run_key}", file_stem=f"biovil_image_pca_{run_key}",
        )

        # ---- Supervised: pixel-space density ratio, fine-tuned DenseNet ---
        # The fitted network is scored directly as the CNN endpoint (branch
        # cnn_direct); its feature extractor gives the Supervised branch.
        X_TL_raw = np.asarray(X_TARGET_raw[tl_idx])
        w_pixel, diag_pixel = compute_pixel_density_ratio_image_only(
            X_SU_raw, X_TARGET_raw[tu_idx], X_SL_raw, seed=run_seed
        )
        weight_records.append({"repeat": rep, "branch": "no_ssl_pixel",
                               "kind": "pixel", **diag_pixel})
        predictor, extractor = joint_finetune_cnn_dr(
            X_SL_raw, Y_SL, w_pixel, X_TL_raw, Y_TL, nT=NT, seed=run_seed
        )
        for eval_name, idx in (("valid", valid_idx), ("tu_transductive", tu_idx)):
            p = predict_proba_densenet(np.asarray(X_TARGET_raw[idx]), predictor,
                                       tag=f"noSSL {eval_name} {run_key}")
            metrics = evaluate_predictions(Y_TARGET[idx], p, patient_ids_target[idx])
            append_metric_rows(rows, rep, run_seed, "cnn_direct", "dr_finetune",
                               {f"dr_finetune|{eval_name}": metrics},
                               split_info, q_text, q_meta)
        free_memory(predictor)

        X_SL_ns_raw = extract_features(X_SL_raw, extractor, tag=f"noSSL SL {run_key}")
        X_SU_ns_raw = extract_features(X_SU_raw, extractor, tag=f"noSSL SU {run_key}")
        X_TG_ns_raw = extract_features(X_TARGET_raw, extractor, tag=f"noSSL TG {run_key}")
        free_memory(extractor)
        X_SL_ns, X_SU_ns, X_TG_ns = reduce_image_representation(
            X_SL_ns_raw, X_SU_ns_raw, X_TG_ns_raw, target_fit_idx,
            tag=f"noSSL CNN {run_key}", file_stem=f"no_ssl_image_pca_{run_key}",
        )
        del X_SL_ns_raw, X_SU_ns_raw, X_TG_ns_raw
        gc.collect()

        if SAVE_FEATURES:
            feature_dir = os.path.join(FEATURE_DIR, run_key)
            os.makedirs(feature_dir, exist_ok=True)
            for name, array in {
                "S_SL": S_SL, "S_SU": S_SU, "S_TARGET": S_TARGET,
                "X_SL_simclr": X_SL_ssl, "X_SU_simclr": X_SU_ssl,
                "X_TARGET_simclr": X_TG_ssl,
                "X_SL_biovil": X_SL_bio, "X_SU_biovil": X_SU_bio,
                "X_TARGET_biovil": X_TG_bio,
                "X_SL_no_ssl": X_SL_ns, "X_SU_no_ssl": X_SU_ns,
                "X_TARGET_no_ssl": X_TG_ns,
            }.items():
                np.save(os.path.join(feature_dir, f"{name}.npy"),
                        np.asarray(array, dtype=np.float32))

        # ---- the three linear branches -----------------------------------
        branch_inputs = {
            "ssl": (X_SL_ssl, X_SU_ssl, X_TG_ssl),
            "biovil": (X_SL_bio, X_SU_bio, X_TG_bio),
            "no_ssl": (X_SL_ns, X_SU_ns, X_TG_ns),
        }
        for branch in IMAGE_BRANCHES:
            X_sl_b, X_su_b, X_tg_b = branch_inputs[branch]
            strifle, baselines, _ = run_branch(
                branch, X_sl_b, X_su_b, X_tg_b, S_SL, S_SU, S_TARGET,
                Y_SL, Y_TARGET, split, patient_ids_target, run_seed, rep,
                split_info, rows, gate_rows, weight_records, q_text, q_meta,
            )
            all_results[run_key][branch] = {
                "metrics": strifle["metrics"],
                "baseline_metrics": baselines["metrics"],
                "phi_beta": strifle["gate"]["phi_beta"],
                "phi_gamma": strifle["gate"]["phi_gamma"],
                "rho": strifle["rho"],
                "selected_endpoint_shapley": strifle["selected_endpoint_shapley"],
            }
            free_memory(strifle["predictions"], baselines["predictions"])

        # ---- incremental persistence -------------------------------------
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(RESULT_DIR, "raw_results.csv"), index=False)
        summarize_raw(frame).to_csv(
            os.path.join(RESULT_DIR, "summary_raw.csv"), index=False
        )
        paired = paired_differences(frame)
        paired.to_csv(os.path.join(RESULT_DIR, "paired_differences.csv"), index=False)
        summarize_paired(paired).to_csv(
            os.path.join(RESULT_DIR, "summary_paired.csv"), index=False
        )
        pd.DataFrame(gate_rows).to_csv(
            os.path.join(RESULT_DIR, "gate_and_shapley_audit.csv"), index=False
        )
        pd.DataFrame(weight_records).to_csv(
            os.path.join(RESULT_DIR, "density_ratio_diagnostics.csv"), index=False
        )
        with open(os.path.join(RESULT_DIR, "results.json"), "w", encoding="utf-8") as h:
            json.dump(all_results, h, indent=2, default=float)

        valid_view = frame[frame["evaluation_split"] == "valid"]
        print("\nVALID summary so far (primary endpoint):")
        print(summarize_raw(valid_view)[
            ["branch", "endpoint", "auc_mean", "auc_sd", "pr_auc_mean", "brier_mean"]
        ].to_string(index=False))

        free_memory(X_SL_ssl, X_SU_ssl, X_TG_ssl, X_SL_bio, X_SU_bio, X_TG_bio,
                    X_SL_ns, X_SU_ns, X_TG_ns, S_SL, S_SU, S_TARGET,
                    T_SL, T_SU, T_TARGET, M_SL, M_SU, M_TARGET,
                    X_TL_raw, w_pixel)

    # ---- final report ----------------------------------------------------
    frame = pd.DataFrame(rows)
    gate_frame = pd.DataFrame(gate_rows)
    if frame.empty:
        print("\nNo results to report.")
        return
    paired = paired_differences(frame)
    paired_summary = summarize_paired(paired)

    shapley_summary = (
        gate_frame.groupby("branch")
        .agg(phi_beta_mean=("phi_beta", "mean"), phi_beta_sd=("phi_beta", "std"),
             phi_gamma_mean=("phi_gamma", "mean"), phi_gamma_sd=("phi_gamma", "std"),
             phi_total_mean=("phi_total", "mean"),
             rho_rate=("rho", "mean"),
             gate_fold_sd_meta=("fold_loss_sd_meta", "mean"),
             dr_domain_auc=("dr_domain_auc", "mean"),
             dr_frac_clipped=("dr_frac_clipped", "mean"),
             delta_l1_mean=("delta_l1_norm", "mean"),
             theta_meta_l1_mean=("theta_meta_l1", "mean"),
             l1_budget=("l1_budget", "first"))
        .reset_index()
    )
    shapley_summary.to_csv(
        os.path.join(RESULT_DIR, "shapley_summary.csv"), index=False
    )
    coalition_counts = (
        gate_frame.groupby(["branch", "selected_endpoint_shapley"]).size()
        .rename("n_repeats").reset_index()
    )
    coalition_counts.to_csv(
        os.path.join(RESULT_DIR, "shapley_coalition_counts.csv"), index=False
    )

    print("\n" + "=" * 88)
    print("FINAL REPORT")
    print("=" * 88)
    print("\nShapley block attribution (mean over repeats):")
    print(shapley_summary.to_string(index=False))
    print("\nSelected coalition frequency:")
    print(coalition_counts.to_string(index=False))
    print("\nPaired differences against target-only, VALID split:")
    view = paired_summary[paired_summary["evaluation_split"] == "valid"]
    columns = [c for c in ["branch", "endpoint", "n_repeats", "delta_auc_mean",
                           "delta_auc_ci_low", "delta_auc_ci_high",
                           "delta_pr_auc_mean", "delta_auc_wilcoxon_p"]
               if c in view.columns]
    print(view[columns].to_string(index=False))
    print(f"\nAll artefacts in: {RESULT_DIR}")
    print("VALID is the primary evaluation set. The tu_transductive rows are "
          "reported separately because TU enters SimCLR, the PCA, the density "
          "ratio and the soft-label imputation, so its metrics are transductive.")
    print(f"paper_gate and shapley_gate above use eps0 = {STRIFLE_GATE_EPSILON}. "
          "The reported Gate-Binary and Gate-Block apply eps0 = log(d)/nT "
          "through replay_gate_epsilon.py, and the reported SUPTRANS is the "
          "three-step fit of recompute_suptrans_three_step.py.")


if __name__ == "__main__":
    main()
