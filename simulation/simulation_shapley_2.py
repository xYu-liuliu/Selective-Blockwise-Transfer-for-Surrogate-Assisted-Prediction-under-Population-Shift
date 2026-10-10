"""Simulation study of "Selective Blockwise Transfer for Surrogate-Assisted
Prediction under Population Shift" (Section 3 and the Simulation Supplement).

Every run in the paper's table of simulation runs is a run of this one file,
and the runs differ only in their command-line options: see "The paper's
runs" below, and the exact commands at the end of --help. The paired re-solve
of the correction (fista_paired.py, among the numerical checks) imports this
file and swaps in its own solver.

Generating process
------------------
Following Cai et al. (2025, Sec. 4), with block structure added to the shift.
A latent Gaussian matrix with AR(1) correlation nu = 0.1 is passed through
t -> floor(log(1 + e^t)) to give the image block; the first three surrogate
columns are built from overlapping windows of the first ten image columns, so
the two blocks are coupled, and the rest are transformed latent draws. After
standardisation Z has p = 128 image and q = 79 + 14 = 93 surrogate
coordinates, d = 221. Domain membership is drawn from a logistic model in Z
rather than assigned. The reference cell has n_T = 200, n_S = 1200,
N_T = N_S = 10^4 and 200 held-out labelled target rows; the master seed is 42.

The target parameter theta_bar (theta_T in the code) has seven non-zero slopes
in each block, each signal_base * U(0.7, 1.3) with a random sign
(signal_base = 0.85), times the block multipliers of the strength rung. Its
intercept is set by bisection so that the target event rate is 0.105. Source
outcomes are drawn from theta_bar + Delta, where Delta = theta_bar^S -
theta_bar carries no intercept component, so the source rate is left free. The
working model is correctly specified in every run the paper reports.

Scenarios
---------
    code   paper      support of Delta
    S0     D_empty    none: the covariate-shift assumption holds
    S1     D_X        seven coordinates of the image block
    S2     D_S        seven coordinates of the surrogate block
    S3     D_XS       four coordinates of each block, swept in magnitude
    S4     D_dense    all 118 free image and 90 free surrogate coordinates,
                      with weights decaying as 1 / sqrt(rank)

The ten image and three surrogate columns that carry the coupling are never
perturbed, so a shift on one block cannot reach the other through the
covariates.

Ground truth and its calibration
--------------------------------
The method responds to delta_0 = theta_bar - theta_star, which is computed
rather than approximated: theta_bar and theta_star are unpenalised logistic
projections against soft labels on 5000 unlabelled target rows, theta_star of
the mixture a g(theta_bar^S) + (1 - a) g(theta_bar), a = n_S / (n_S + n_T).
Magnitudes are multiples of the radius sqrt(n_T / log d), the rate at which
Assumption S4(i) of Cai et al. bounds ||delta_0||_1 (6.09 at the reference
cell; the code calls it the budget). The linearisation delta_0 = -a Delta does
not deliver the requested norm, so the scale of Delta is solved for until the
realised ||delta_0||_1 equals it. Each replicate is labelled by the block
carrying more than 65% of its own realised ||delta_0||_1. In the n_T sweep the
norm is anchored at the reference radius (--delta0-anchor-nt 200), so
||delta_0||_1 = 6.09 at every n_T.

Estimation and scoring
----------------------
Algorithm 1 is reimplemented rather than imported from the application
pipeline, so that a change in one is not silently inherited by the other.
Every constant is the pipeline's; the one deliberate departure is the solver
of the L1 fits (liblinear with intercept_scaling = 1e5). The correction's FISTA
step is also computed once on the full design rather than per CV subproblem
as in the pipeline, which the paired re-solve shows leaves rho and c_hat
unchanged. The dead zone is the paper's schedule epsilon0 = log(d) / n_T,
recomputed per cell. Each replicate
records the four cross-fitted coalition losses, phi_X and phi_S (phi_beta and
phi_gamma), both rules (rho, and selected for c_hat), and both layers for
every coalition: the imputation model theta^(c) (columns *_imp_*) and the
deployed image-only beta(theta^(c)) (columns *_pred_*). The "gate" columns of
the negative-transfer summary follow the coalition rule, i.e. Gate-Block. It
also records T_RSC = (log d / n_T) ||theta_hat(meta) - theta_bar||_1^2 over
the slopes (eps0_required), the unit-weight proxy for the RSC tolerance term
that the paper evaluates in its assessment of epsilon0, and T_RSC / epsilon0
(eps0_conservatism).

The paper's runs
----------------
Seven batches, fifty replicates per scenario-magnitude point (--reps 50),
correctly specified nuisance models throughout. Each writes to its own folder
results/<batch>, under the batch's name on the original server:

    batch          options                                            reps
    negtransfer    S0, S3 at 0; .5 .75 1 1.25 1.5 2 3                  400
    dense          S4 at .5 1 1.5 2                                    200
    dense_holdout  --delta-holdout; S3, S4 at .5 1 1.5 2               400
    strength       S1, S2 at .5 1 1.5 2; rungs beta_strong_2x,        1600
                   balanced, gamma_strong, gamma_strong_2x
    strength_fine  S1, S2 at .5 1 1.5 2; rungs gamma_soft, gamma_med   800
    grid           S1, S2 at 1; --sweep n_T=100,200,300,400,500,800,   600
                   --delta0-anchor-nt 200
    oracle         --oracle-imputation; S0, S3 at 0; .5 ... 2          350

oracle is paired with negtransfer on the same seeds. The exact commands are
listed at the end of --help.

Each run writes simulation_parametric_raw.csv, one row per replicate, and the
summary tables printed by report() (simulation_*.csv, simulation_summary.json)
to its --out-dir; the paper's tables are read from these files.

Not used in the paper: the misspecified nuisance models (--dgp), the other
strength levels, the plasmode arm, epsilon0 overrides and the operating-point
diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

_NEW_SKLEARN = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)

# Default output folder: the original default, simulation_output, now under
# <repo>/results (or under STRIFLE_RESULTS_DIR when that is set, as for the
# application scripts). The paper's batches each pass their own --out-dir,
# results/<batch> (see --help).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_OVERRIDE = os.environ.get("STRIFLE_RESULTS_DIR", "").strip()
RESULTS_DIR = (os.path.abspath(os.path.expanduser(_RESULTS_OVERRIDE))
               if _RESULTS_OVERRIDE else os.path.join(REPO_ROOT, "results"))
DEFAULT_OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")


# ======================================================================
# Configuration, mirroring the application so the simulation describes
# the same estimator.
# ======================================================================

@dataclass
class Config:
    p: int = 128                 # image block width
    q_text: int = 79             # surrogate term block
    q_meta: int = 14             # surrogate metadata block
    n_T: int = 200               # labelled target
    n_S: int = 1200              # labelled source
    N_T: int = 10_000            # unlabelled target
    N_S: int = 10_000            # unlabelled source
    n_valid: int = 200
    # The application's target prevalence. Its patient split fixes the class
    # composition (21 positives in the 200 VALID rows, and 21 in TL at
    # n_T = 200), whereas the outcome here is drawn, so the labelled positive
    # count is Binomial(n_T, 0.105): mean 21, sd 4.3 at n_T = 200. That is a
    # layer of design noise the application does not have; n_tl_positive and
    # n_val_positive are recorded per replicate so it stays visible (the
    # paper's supplementary table of positive counts).
    prevalence: float = 0.105
    nu: float = 0.1              # AR(1) correlation in the latent draw
    seed: int = 42

    C_grid: tuple = field(default_factory=lambda: tuple(np.logspace(-3, 1, 9)))
    cv_folds: int = 5
    domain_C: float = 1.0
    clip: tuple = (0.01, 10.0)

    # Dead zone. None means the paper's schedule log(d) / n_T, recomputed per
    # cell (see eps0 below). A number overrides it; none of the paper's runs
    # does, since its threshold sensitivity replays the stored coalition
    # losses instead of refitting.
    epsilon_0: float | None = None

    # Reference n_T for the Delta scaling. None keeps the cell's own radius
    # sqrt(n_T / log d) (the code's "budget"), so magnitude reads as a
    # multiple of that cell's radius. The paper's n_T sweep sets it to 200:
    # the radius grows as sqrt(n_T), so a fixed magnitude would make
    # ||delta_0||_1 grow with the sample and vary difficulty and sample size
    # together. Anchoring holds ||delta_0||_1 at one absolute value (6.09)
    # across the sweep, which is the comparison the application poses:
    # delta_0 is whatever it is, and the question is whether more labelled
    # target data resolves it.
    delta0_anchor_n_T: int | None = None

    # Nuisance-model specification. The imputation model is misspecified by
    # adding quadratic and interaction terms to the outcome linear predictor,
    # and the density ratio by making domain membership a non-linear function
    # of Z, both following Cai et al. (2025, Sec. 4). Delta is injected on the
    # linear part in every case, so its support, and hence the scenario label,
    # survives misspecification. The paper reports only the correctly
    # specified case.
    misspecify_m: bool = False
    misspecify_w: bool = False

    # Overall signal level: the scale of every non-zero coefficient in
    # theta_T before the block multipliers below are applied. The paper's
    # runs use 0.85.
    #
    # The choice is forced by a scale mismatch. Magnitudes are multiples of
    # the radius sqrt(n_T / log d) = 6.09, an absolute quantity that does not
    # scale with theta, so what "one radius" means depends on how large theta
    # is. At 0.85, ||theta_bar||_1 is about 11.9 at the balanced rung and a
    # one-radius defect is about half of it (||delta_0||_1 / ||theta_bar||_1 =
    # 0.507 under D_XS in the paper's calibration table), which is what a
    # partial failure of transferability should look like. A small signal
    # level, closer to the discrimination the application's blocks afford,
    # makes a one-radius shift several times theta itself: "the image block
    # does not transfer" then degenerates into "the source parameter is
    # unrelated noise", and the link can no longer express the requested
    # ||delta_0||_1 at the larger magnitudes. The simulation therefore does
    # not match the application on absolute discrimination, as the paper
    # states.
    signal_base: float = 0.85

    # Relative signal in the two blocks: the rung of the strength ladder (see
    # BLOCK_STRENGTH_LEVELS). The reference cell is the balanced rung.
    block_strength: str = "balanced"

    # Whether to solve for the scale of Delta so the realised ||delta_0||_1
    # equals magnitude * anchor_budget, as in every run of the paper. False
    # keeps the linearisation delta_0 = -a Delta, whose error is not a
    # constant: it peaks near one radius, falls as the logit saturates, and
    # is close to exact only under D_dense. Kept as a sensitivity arm only.
    calibrate_delta0: bool = True

    # Ablation, the paper's oracle imputation. The unlabelled target rows get
    # soft labels from the true theta_bar instead of the coalition's own
    # fitted theta, with the rest of the deployed stage unchanged. Paired with
    # the reference run, it leaves the imputation layer bit-identical and
    # isolates the channel through which a coalition reaches the deployed
    # beta: the imputed labels (the paper's section on what phi measures).
    # Not a runnable estimator: it reads the truth.
    oracle_imputation: bool = False

    # Ablation, the paper's held-out correction. Fit the correction on target
    # rows held out of the meta fit instead of on I_{T,L}. Algorithm 1 uses the
    # same target labels for both steps, so theta_tilde has already absorbed
    # them and the residuals the correction is read from are systematically
    # small. Splits n_T in half, so the two arms are not sample-size
    # comparable; the comparison of interest is the fraction of replicates at
    # delta_hat = 0, not the estimation error.
    delta_holdout: bool = False

    # Whether to fit Algorithm 1's final image-only stage (the deployed layer)
    # and score it, as every run in the paper's table of runs does. False
    # scores the imputation layer alone and is roughly two and a half times
    # cheaper; the paper's paired re-solve of the correction uses it on 160 of
    # its 172 pairs, where only the imputation layer is compared.
    fit_prediction_layer: bool = True

    # Whether the covariates carry the parametric arm's block coupling, where
    # the first ten image columns feed the first three surrogates. make_delta
    # excludes those columns so Delta's support matches its scenario label.
    # Real features have no such construction, so the exclusion there would
    # merely drop the leading image components, the highest-variance ones,
    # which is where a PCA basis puts the signal. load_plasmode clears it.
    block_coupling: bool = True

    # Mirrors STRIFLE_META_CV_TARGET_ONLY in the pipeline. False is the
    # paper's setting: theta-tilde's penalty is selected under the fitting
    # weights, so the 1200 reweighted source rows dominate a validation loss
    # whose estimand is target risk. True selects on target rows only.
    meta_cv_target_only: bool = False

    @property
    def q(self) -> int:
        return self.q_text + self.q_meta

    @property
    def d(self) -> int:
        return self.p + self.q

    @property
    def budget(self) -> float:
        return float(np.sqrt(self.n_T / np.log(self.d)))

    @property
    def anchor_budget(self) -> float:
        """The radius Delta is scaled against, which is the cell's own unless
        an anchor is set. Kept separate from ``budget`` (the cell's radius
        sqrt(n_T / log d)) so every diagnostic that divides by it still
        reports the cell's own scale: the anchor changes what is generated,
        never how it is measured."""
        if self.delta0_anchor_n_T is None:
            return self.budget
        return float(np.sqrt(self.delta0_anchor_n_T / np.log(self.d)))

    @property
    def eps0(self) -> float:
        """The dead zone: the paper's schedule epsilon0 = log(d) / n_T.

        For a logistic link, Cai et al. (2025, Supplement S.4) give no
        computable cutoff: zero is allowed for the exact-sparse linear model
        with identity link, and the sufficient condition in the proof of
        their Theorem S2 involves the unknown theta_bar, the error of the meta
        fit and restricted-strong-convexity constants. The paper prespecifies
        log(d) / n_T, the explicit prefactor of the RSC tolerance term with
        the multiplier fixed at one, and does not claim that it satisfies the
        condition. It equals 1 / radius^2 and is recomputed per cell, since
        the grid sweeps n_T. Because theta_bar is known here, each replicate
        also records the RSC tolerance term at its realised norm,
        T_RSC = (log d / n_T) ||theta_hat(meta) - theta_bar||_1^2
        (eps0_required), and T_RSC / epsilon0 (eps0_conservatism).
        """
        if self.epsilon_0 is not None:
            return float(self.epsilon_0)
        return float(np.log(self.d) / self.n_T)

    @property
    def dgp_label(self) -> str:
        return ("m_mis" if self.misspecify_m else "m_cor") + "/" + \
               ("w_mis" if self.misspecify_w else "w_cor")


# ======================================================================
# Estimation components. Deliberately self-contained: the simulation must
# exercise the same logic as the pipeline without importing it, so that a
# change in one is not silently inherited by the other.
# ======================================================================

# The pipeline's correction path, copied rather than re-derived: seven ratios
# of lambda_max, at most 5000 FISTA iterations, and a stopping rule relative to
# the iterate's scale (the paper's Supplement, "The correction").
CORRECTION_LAMBDA_RATIOS = np.asarray(
    [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], dtype=np.float64
)
CORRECTION_MAX_ITER = 5000
CORRECTION_TOL = 1e-7

# Counted rather than warned about, as the pipeline does: at the small-lambda
# end some solves legitimately hit the cap, and the number is a diagnostic
# worth reporting, not an error worth aborting on.
_FISTA_NONCONVERGENCE = {"count": 0, "total": 0}


# The one deliberate departure from the pipeline (the paper's Supplement,
# "The l1 fits"). liblinear implements the intercept as a synthetic constant
# column, and that column is regularised along with everything else, whereas
# saga, which the pipeline uses, leaves the intercept unpenalised. At the
# sparse end of the C grid the difference is not cosmetic: with
# intercept_scaling at its default of 1, liblinear returns an intercept of
# exactly 0 at C <= 0.01, a model predicting 0.5 for every row at a
# prevalence of 0.105. Raising the scaling makes the penalty on that column
# negligible; at 1e5 the recovered intercept matches the exact unpenalised
# value to four decimals and matches saga wherever saga converges (saga
# itself fails to converge at the sparse end even at 1e5 iterations).
INTERCEPT_SCALING = 1e5


def _lasso(C: float, seed: int) -> LogisticRegression:
    kw = dict(C=float(C), solver="liblinear", max_iter=5000,
              intercept_scaling=INTERCEPT_SCALING, random_state=int(seed))
    return LogisticRegression(l1_ratio=1.0, **kw) if _NEW_SKLEARN else \
        LogisticRegression(penalty="l1", **kw)


def _weighted_log_loss_from_logits(y, logits, sample_weight=None):
    """The pipeline's loss, on logits rather than probabilities.

    log_loss on predict_proba clips nothing since scikit-learn 1.5, and an L1
    fit at the loose end of the C grid saturates readily, so a probability of
    exactly 0 or 1 on a misclassified validation row returns inf and poisons
    the fold mean. logaddexp is exact at any logit.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    losses = np.logaddexp(0.0, logits) - y * logits
    if sample_weight is None:
        return float(losses.mean())
    sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
    if sample_weight.sum() <= 0:
        return float("nan")
    return float(np.average(losses, weights=sample_weight))


def fit_l1_cv(X, y, cfg: Config, seed: int, sample_weight=None, cv_weight=None):
    """L1 logistic with CV over a fixed penalty grid.

    Five-fold stratified CV log loss over cfg.C_grid; the fold count drops to
    the smaller class count when that is below five, and a fit with fewer
    than two rows in a class falls back to C = 1. The validation loss carries
    the fitting weights, matching the pipeline at
    STRIFLE_META_CV_TARGET_ONLY = False. For theta-tilde the pool is 1200
    reweighted source rows against 200 target rows, so an unweighted
    validation loss would select the penalty under a different measure from
    the one being fitted, which is neither the pipeline's rule nor its
    target-only variant. cv_weight overrides the validation weighting and is
    how that variant is reproduced.
    """
    y = np.asarray(y, int)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    cv_weight = (sample_weight if cv_weight is None
                 else np.asarray(cv_weight, dtype=np.float64))

    counts = np.bincount(y, minlength=2)
    k = min(cfg.cv_folds, int(counts.min()))
    if k < 2:                                     # degenerate fold structure
        m = _lasso(1.0, seed)
        m.fit(X, y, sample_weight=sample_weight)
        return m, 1.0

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    fold_losses = np.full((k, len(cfg.C_grid)), np.nan, dtype=np.float64)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        sw = None if sample_weight is None else sample_weight[tr]
        vw = None if cv_weight is None else cv_weight[te]
        if vw is not None and vw.sum() <= 0:      # a target-only fold that
            continue                              # drew no target rows
        for j, C in enumerate(cfg.C_grid):
            m = _lasso(C, seed)
            m.fit(X[tr], y[tr], sample_weight=sw)
            fold_losses[i, j] = _weighted_log_loss_from_logits(
                y[te], m.decision_function(X[te]), vw)

    mean_losses = np.nanmean(fold_losses, axis=0)
    if not np.isfinite(mean_losses).any():
        raise RuntimeError("every CV fold was skipped; check cv_weight")
    best = float(cfg.C_grid[int(np.nanargmin(mean_losses))])
    m = _lasso(best, seed)
    m.fit(X, y, sample_weight=sample_weight)
    return m, best


def fit_density_ratio(Z_SU, Z_TU, Z_SL, cfg: Config, seed: int):
    """Classification form of the density ratio, as in the pipeline: ridge
    logistic domain classifier on the standardised pooled unlabelled rows, the
    hard threshold kappa on the standardised scale, then centring before
    clipping. The intercept is refitted at the surviving coefficients, where
    the pipeline keeps the fitted one; the centring removes any constant from
    the log ratio, so the weights are the same either way."""
    X = np.vstack([Z_SU, Z_TU])
    y = np.r_[np.zeros(len(Z_SU), int), np.ones(len(Z_TU), int)]
    sc = StandardScaler().fit(X)
    Xz = sc.transform(X)

    clf = LogisticRegression(C=cfg.domain_C, max_iter=5000, solver="lbfgs",
                             random_state=int(seed)).fit(Xz, y)
    coef = clf.coef_[0].copy()
    # The reported domain AUC is that of the untruncated fit, matching the
    # quantity the application reports under the same name (dr_domain_auc):
    # the pipeline copies the coefficients out before thresholding, so its
    # decision_function is still the full ridge classifier. The thresholded
    # AUC is recorded alongside under its own name.
    auc_full = (float(roc_auc_score(y, clf.decision_function(Xz)))
                if len(np.unique(y)) == 2 else 0.5)

    kappa = 10.0 * np.sqrt(np.log(Z_SU.shape[1]) / (len(Z_SU) + len(Z_TU)))
    keep = np.abs(coef) >= kappa
    coef[~keep] = 0.0

    # The fitted intercept belonged to the untruncated vector; refit it.
    eta_no_int = Xz @ coef
    if keep.any():
        intercept = _refit_intercept(eta_no_int, y)
    else:
        intercept = float(np.log(len(Z_TU) / len(Z_SU)))

    auc_thresholded = (float(roc_auc_score(y, eta_no_int + intercept))
                       if keep.any() else 0.5)

    log_w = (sc.transform(Z_SL) @ coef + intercept
             + np.log(len(Z_SU) / len(Z_TU)))
    # Centre so that E_S[w] = 1 BEFORE clipping, then renormalise.
    m = log_w.max()
    log_w = log_w - (m + np.log(np.mean(np.exp(log_w - m))))
    lo, hi = np.log(cfg.clip[0]), np.log(cfg.clip[1])
    frac_clipped = float(np.mean((log_w < lo) | (log_w > hi)))
    w = np.exp(np.clip(log_w, lo, hi))
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return w, {"domain_auc": auc_full,
               "domain_auc_thresholded": auc_thresholded,
               "n_retained": int(keep.sum()),
               "frac_clipped": frac_clipped, "ess": ess}


def _refit_intercept(linear_predictor, y) -> float:
    """Unpenalised intercept given fixed slopes.

    Byte-for-byte the pipeline's _refit_intercept: 200 damped Newton steps
    with the step clipped to +/- 5 and a floor on the Hessian. The clipping is
    what makes it safe on a saturated offset, where an undamped step can
    overshoot into a region the next iterate cannot return from.

    Every coalition splices slope blocks from two different fits, so the
    intercept it inherits belonged to neither. Refitting it is what keeps a
    coalition comparison about slopes rather than about calibration offsets.
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


def fit_delta_fista(Z, y, offset, cfg: Config, seed: int):
    """Offset logistic with an L1 penalty on the slopes only, solved by FISTA.
    The intercept is unpenalised so a pure calibration shift between the two
    populations does not consume the L1 budget.

    The penalty path and the solver budget are the pipeline's: the same seven
    ratios of lambda_max (the smallest penalty that zeroes every slope at the
    refitted intercept), the same 5000-iteration cap and the same relative
    stopping rule, with lambda chosen by stratified CV. The step 1/L, with
    L = ||A||_2^2 / (4n), is computed once on the full design and reused for
    the CV subproblems, where the pipeline recomputes it per subproblem; the
    paired re-solve with a hardened solver (fista_paired.py, the paper's
    "correction solve" check) leaves rho and c_hat unchanged.

    Returns (delta_intercept, delta_slopes, lambda); fit_theta_pair adds the
    intercept to theta_meta, as the pipeline's run_strifle does."""
    n, d = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    L = 0.25 * np.linalg.norm(A, 2) ** 2 / n
    step = 1.0 / max(L, 1e-12)

    a0 = _refit_intercept(offset, y)
    grad0 = A.T @ (1.0 / (1.0 + np.exp(-(offset + a0))) - y) / n
    lam_max = max(float(np.abs(grad0[1:]).max()), 1e-6)
    lambdas = lam_max * CORRECTION_LAMBDA_RATIOS

    counts = np.bincount(np.asarray(y, int), minlength=2)
    k = min(cfg.cv_folds, int(counts.min()))
    if k >= 2:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
        losses = np.zeros(len(lambdas))
        for tr, te in cv.split(Z, y):
            for j, lam in enumerate(lambdas):
                b = _fista(A[tr], y[tr], offset[tr], lam, step)
                losses[j] += _weighted_log_loss_from_logits(
                    y[te], offset[te] + A[te] @ b)
        lam = float(lambdas[int(np.argmin(losses))])
    else:
        lam = float(lambdas[len(lambdas) // 2])

    b = _fista(A, y, offset, lam, step)
    return float(b[0]), b[1:], lam


def _fista(A, y, offset, lam, step, iters: int = CORRECTION_MAX_ITER,
           tol: float = CORRECTION_TOL):
    """The pipeline's FISTA: at most 5000 iterations, stopping when
    ||b_new - b||_inf <= 1e-7 (1 + ||b||_inf). Solves that reach the cap are
    counted in _FISTA_NONCONVERGENCE rather than warned about."""
    n, m = A.shape
    b = np.zeros(m)
    z = b.copy()
    t = 1.0
    converged = False
    for _ in range(iters):
        eta = offset + A @ z
        grad = A.T @ (1.0 / (1.0 + np.exp(-eta)) - y) / n
        b_new = z - step * grad
        thr = step * lam
        b_new[1:] = np.sign(b_new[1:]) * np.maximum(np.abs(b_new[1:]) - thr, 0.0)
        scale = 1.0 + np.max(np.abs(b))
        if np.max(np.abs(b_new - b)) <= tol * scale:
            b = b_new
            converged = True
            break
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = b_new + ((t - 1) / t_new) * (b_new - b)
        b, t = b_new, t_new
    _FISTA_NONCONVERGENCE["total"] += 1
    if not converged:
        _FISTA_NONCONVERGENCE["count"] += 1
    return b


# ======================================================================
# Shapley decomposition over the two parameter blocks
# ======================================================================

COALITIONS = ("none", "beta", "gamma", "both")


def _theta_of(coalition: str, theta_T, theta_meta, p: int):
    """theta^(c): each slope block from the meta fit if it is in c, from the
    target-only fit otherwise.

    theta is [intercept, slopes], so the image block X is theta[1:1+p] and the
    surrogate block S is theta[1+p:]. Every caller refits the intercept, so
    out[0] is a placeholder: the intercept belongs to neither block and is
    not a player.
    """
    out = theta_T.copy()
    if coalition in ("beta", "both"):
        out[1:1 + p] = theta_meta[1:1 + p]
    if coalition in ("gamma", "both"):
        out[1 + p:] = theta_meta[1 + p:]
    return out


def _check_theta_of():
    """Regression test for the block boundary: beta (X) and gamma (S) must
    cover the slopes with no overlap and no gap, and the intercept must belong
    to neither."""
    p = 4
    tT = np.array([100., 1, 2, 3, 4, 5, 6, 7])
    tM = np.array([900., -1, -2, -3, -4, -5, -6, -7])
    assert np.array_equal(_theta_of("none", tT, tM, p), tT)
    assert np.array_equal(_theta_of("beta", tT, tM, p),
                          np.array([100., -1, -2, -3, -4, 5, 6, 7]))
    assert np.array_equal(_theta_of("gamma", tT, tM, p),
                          np.array([100., 1, 2, 3, 4, -5, -6, -7]))
    assert np.array_equal(_theta_of("both", tT, tM, p)[1:], tM[1:])
    b = _theta_of("beta", tT, tM, p)[1:] - tT[1:]
    g = _theta_of("gamma", tT, tM, p)[1:] - tT[1:]
    assert np.all(b * g == 0), "beta and gamma overlap"
    assert np.array_equal(b + g, tM[1:] - tT[1:]), "beta and gamma leave a gap"
    # p = 0 is the plasmode mode with no image block: beta must be a no-op.
    assert np.array_equal(_theta_of("beta", tT, tM, 0), tT)
    assert np.array_equal(_theta_of("gamma", tT, tM, 0)[1:], tM[1:])


def fit_theta_pair(Z_TL, y_TL, Z_SL, y_SL, w_SL, cfg: Config, seed: int):
    """FitPair of Algorithm 1: the target-only fit, the pooled fit
    theta_tilde and its correction delta, which give the meta fit.

    Algorithm 1 uses I_{T,L} for BOTH the pooled fit and the correction, so
    theta_tilde has already absorbed those labels, and the TL residuals the
    correction is read from are smaller than the population bias they are
    meant to reveal. cfg.delta_holdout (the paper's held-out correction)
    splits TL so the correction sees rows the pooled fit did not, which
    separates "there is no bias to correct" from "the bias is invisible from
    the rows we are looking at".
    """
    Z_meta, y_meta = Z_TL, y_TL
    Z_corr, y_corr = Z_TL, y_TL
    if cfg.delta_holdout and len(y_TL) >= 4:
        half = len(y_TL) // 2
        Z_meta, y_meta = Z_TL[:half], y_TL[:half]
        Z_corr, y_corr = Z_TL[half:], y_TL[half:]

    m_T, _ = fit_l1_cv(Z_TL, y_TL, cfg, seed)
    theta_T = np.r_[m_T.intercept_[0], m_T.coef_[0]]

    Z_pool = np.vstack([Z_SL, Z_meta])
    y_pool = np.r_[y_SL, y_meta]
    w_pool = np.r_[w_SL, np.ones(len(y_meta))]
    cv_w = (np.r_[np.zeros(len(y_SL)), np.ones(len(y_meta))]
            if cfg.meta_cv_target_only else None)
    m_tilde, _ = fit_l1_cv(Z_pool, y_pool, cfg, seed + 1,
                           sample_weight=w_pool, cv_weight=cv_w)
    theta_tilde = np.r_[m_tilde.intercept_[0], m_tilde.coef_[0]]

    offset = theta_tilde[0] + Z_corr @ theta_tilde[1:]
    delta0, delta, _ = fit_delta_fista(
        Z_corr, np.asarray(y_corr, float), offset, cfg, seed + 2)
    theta_meta = theta_tilde.copy()
    # As in the pipeline's run_strifle, the correction's intercept is added
    # too; every consumer refits the intercept anyway.
    theta_meta[0] = theta_tilde[0] + delta0
    theta_meta[1:] = theta_tilde[1:] + delta
    return theta_T, theta_tilde, theta_meta, delta


def _score_predictions(y, eta, n_bins: int = 10):
    """AUC, Brier and ECE from a linear predictor, binned as the pipeline bins.

    The top bin is closed on both sides: the pipeline uses np.digitize with a
    clip, so a probability of exactly 1 lands in the last bin. A half-open top
    bin would drop those rows from the numerator while leaving them in the
    denominator and bias ECE down, and saturated predictions at large
    magnitude are precisely the rows that should count.
    """
    y = np.asarray(y, dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()
    p_hat = expit(eta)
    out = {
        "auc": (float(roc_auc_score(y.astype(int), eta))
                if len(np.unique(y)) > 1 else float("nan")),
        "brier": float(np.mean((p_hat - y) ** 2)),
    }
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc = 0.0
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = ((p_hat >= lo) & (p_hat <= hi) if k == n_bins - 1
                else (p_hat >= lo) & (p_hat < hi))
        if not mask.any():
            continue
        acc += mask.sum() * abs(float(y[mask].mean()) - float(p_hat[mask].mean()))
    out["ece"] = float(acc / len(y))
    return out


def fit_beta_imputed(theta, Z_TL, y_TL, Z_TU, cfg: Config, seed: int,
                     theta_oracle=None):
    """Algorithm 1's final stage, the deployed layer: an image-only Lasso on
    the labelled and the soft-imputed target rows.

    This is the estimator the application evaluates, and it is not the
    imputation model. theta reads both blocks; beta reads only the image
    block, so everything the surrogate contributes has to pass through the
    imputed label m = expit(theta' Z_TU) before it can reach a prediction.
    Scoring both layers is what lets the paper separate what phi measures at
    the imputation layer from what reaches the deployed model.

    Ported from the pipeline's fit_beta: the same soft-label-by-duplication
    trick (weights m and 1 - m, total weight one per original row, so the
    penalty scale matches an ordinary fit), the same fold construction
    (stratified on the labelled rows, plain K-fold on the imputed ones) and
    the same C grid.
    """
    if cfg.p == 0:
        raise ValueError("the final beta stage is image-only and needs p > 0")
    X_TL, X_TU = Z_TL[:, :cfg.p], Z_TU[:, :cfg.p]
    # theta_oracle (the oracle-imputation ablation) replaces the coalition's
    # own soft labels with those of the true theta_bar. Everything downstream,
    # the fold construction, the C grid and the final fit, is untouched, so a
    # difference between the two arms is attributable to the imputation.
    src = theta if theta_oracle is None else theta_oracle
    m = expit(src[0] + Z_TU @ src[1:])
    X_soft = np.vstack([X_TL, X_TU])
    y_soft = np.r_[np.asarray(y_TL, dtype=float), m]
    n_tl = len(y_TL)

    counts = np.bincount(np.asarray(y_TL, int), minlength=2)
    k = min(cfg.cv_folds, int(counts.min()), max(2, len(X_TU)))
    if k < 2:
        raise RuntimeError("the final beta fit needs two labelled rows per class")

    fold_id = np.full(len(X_soft), -1, dtype=int)
    for j, (_, va) in enumerate(StratifiedKFold(
            n_splits=k, shuffle=True, random_state=int(seed)).split(X_TL, y_TL)):
        fold_id[va] = j
    if len(X_TU) >= k:
        for j, (_, va) in enumerate(KFold(
                n_splits=k, shuffle=True,
                random_state=int(seed) + 1).split(X_TU)):
            fold_id[n_tl + va] = j
    else:
        fold_id[n_tl:] = np.arange(len(X_TU)) % k
    if np.any(fold_id < 0):
        raise RuntimeError("failed to assign every row to a beta CV fold")

    def _duplicated(X, y_soft_block):
        Xd = np.vstack([X, X])
        yd = np.r_[np.ones(len(X), int), np.zeros(len(X), int)]
        wd = np.r_[y_soft_block, 1.0 - y_soft_block]
        return Xd, yd, wd

    losses = np.zeros(len(cfg.C_grid))
    for j in range(k):
        tr = fold_id != j
        Xd, yd, wd = _duplicated(X_soft[tr], y_soft[tr])
        for c_idx, C in enumerate(cfg.C_grid):
            fit = _lasso(C, seed + j).fit(Xd, yd, sample_weight=wd)
            losses[c_idx] += _weighted_log_loss_from_logits(
                y_soft[~tr], fit.decision_function(X_soft[~tr]))
    best = float(cfg.C_grid[int(np.argmin(losses))])

    Xd, yd, wd = _duplicated(X_soft, y_soft)
    fit = _lasso(best, seed + 7919).fit(Xd, yd, sample_weight=wd)
    return float(fit.intercept_[0]), fit.coef_[0], best


def shapley_blocks(Z_TL, y_TL, Z_SL, y_SL, w_SL, cfg: Config, seed: int):
    """The paper's value function, attribution and both rules.

    TL is split into two stratified folds. On each fold the pair
    (theta_hat(T), theta_hat(meta)) is refitted, with the source rows and the
    density ratio unchanged, every coalition is assembled from that one pair
    with its intercept refitted on the fold, and the log loss is evaluated on
    the other fold. L(c) averages the two, v(c) = L(empty) - L(c), phi_beta
    and phi_gamma are the two-player Shapley values phi_X and phi_S, rho is
    the binary rule and selected the coalition rule c_hat, both at cfg.eps0.
    """
    y_TL = np.asarray(y_TL, int)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(seed))
    folds = list(skf.split(Z_TL, y_TL))

    per_fold = {c: [] for c in COALITIONS}
    for fold_id, (fit_idx, eval_idx) in enumerate(folds):
        # The pipeline offsets the seed by fold (seed + 100 * fold_id), so the
        # two folds' internal CV splits are independent rather than identical.
        th_T, _, th_meta, _ = fit_theta_pair(
            Z_TL[fit_idx], y_TL[fit_idx], Z_SL, y_SL, w_SL, cfg,
            seed + 100 * fold_id)
        for c in COALITIONS:
            th = _theta_of(c, th_T, th_meta, cfg.p)
            off_fit = Z_TL[fit_idx] @ th[1:]
            a = _refit_intercept(off_fit, y_TL[fit_idx])    # slopes, not offsets
            eta = a + Z_TL[eval_idx] @ th[1:]
            # From logits, as the pipeline does. Going through a probability
            # is not merely a different route to the same number: since
            # scikit-learn 1.5 log_loss does not clip, and a saturated eta on
            # a misclassified evaluation row returns inf, which would take a
            # coalition loss, and with it phi, rho and the selection, out of
            # the reals for the whole replicate.
            per_fold[c].append(
                _weighted_log_loss_from_logits(y_TL[eval_idx], eta))

    L = {c: float(np.mean(per_fold[c])) for c in COALITIONS}
    v = {c: L["none"] - L[c] for c in COALITIONS}
    phi_beta = 0.5 * ((v["beta"] - v["none"]) + (v["both"] - v["gamma"]))
    phi_gamma = 0.5 * ((v["gamma"] - v["none"]) + (v["both"] - v["beta"]))

    # phi_beta, phi_gamma and phi_total are differences of the four coalition
    # losses, so they are identical at every epsilon_0. Only rho and the
    # selected coalition move with the threshold.
    eps0 = cfg.eps0
    rho = int(L["both"] <= L["none"] - eps0)
    eligible = ["none"] + [c for c in COALITIONS
                           if c != "none" and L[c] <= L["none"] - eps0]
    selected = min(eligible, key=lambda c: L[c])
    return {"L": L, "v": v, "phi_beta": float(phi_beta),
            "phi_gamma": float(phi_gamma),
            "phi_total": float(phi_beta + phi_gamma),
            "efficiency_gap": float(abs(phi_beta + phi_gamma - v["both"])),
            "rho": rho, "selected": selected, "eps0": float(eps0)}


# ======================================================================
# Data generation
# ======================================================================

def _a_transform(t):
    return np.floor(np.log1p(np.exp(np.clip(t, -30, 30))))


def generate_parametric(cfg: Config, rng):
    """Cai et al. (2025, Sec. 4) generation of Z and of domain membership.
    Y is added by the caller so that Delta can be injected with a known
    support.

    Domain membership is drawn rather than assigned, so the pool is
    deliberately oversized: a pool of exactly n_S + N_S + n_T + N_T + n_valid
    rows would almost never split into enough of both."""
    need_s = cfg.n_S + cfg.N_S
    need_t = cfg.n_T + cfg.N_T + cfg.n_valid
    n_total = int(2.5 * (need_s + need_t))
    d = cfg.d

    # AR(1) latent draw, realised by an AR recursion rather than a d x d
    # Cholesky: at d = 221 the factorisation dominates the runtime.
    W = np.empty((n_total, d), dtype=np.float64)
    W[:, 0] = rng.standard_normal(n_total)
    if cfg.nu > 0:
        s = np.sqrt(1 - cfg.nu ** 2)
        for j in range(1, d):
            W[:, j] = cfg.nu * W[:, j - 1] + s * rng.standard_normal(n_total)
    else:
        W[:, 1:] = rng.standard_normal((n_total, d - 1))

    X = _a_transform(W[:, :cfg.p])
    S = np.empty((n_total, cfg.q))
    S[:, 0] = _a_transform(X[:, :5].sum(1) + W[:, cfg.p])
    S[:, 1] = _a_transform(X[:, 2:7].sum(1) + W[:, cfg.p + 1])
    S[:, 2] = _a_transform(X[:, 5:10].sum(1) + W[:, cfg.p + 2])
    S[:, 3:] = _a_transform(W[:, cfg.p + 3:cfg.d])

    Z = np.hstack([X, S])
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-8)

    # Domain membership, drawn from a logistic model in Z. Here R = 1 marks a
    # source row, the reverse of the paper's convention (R = 1 for target).
    if cfg.misspecify_w:
        lin = (1.8 * Z[:, cfg.p + 1] - 2.0 * Z[:, cfg.p + 2]
               + Z[:, cfg.p] * (1 + (Z[:, 0] + Z[:, 1]) /
                                (1 + np.exp(-Z[:, 2:10].sum(1)))))
    else:
        lin = Z[:, 0] - Z[:, 1] - Z[:, 2] + Z[:, cfg.p]
    R = (rng.uniform(size=n_total) > 1 / (1 + np.exp(-lin))).astype(int)
    return Z, R


def outcome_logit(Z, theta, cfg: Config):
    """Linear predictor for Y given Z. In the paper's runs the working model
    is correctly specified, so theta is the generating parameter. Under the
    unreported misspecified variant the working logistic form cannot
    represent the truth and theta_bar becomes a projection; Delta still
    enters linearly, which keeps the scenario label meaningful."""
    lin = theta[0] + Z @ theta[1:]
    if not cfg.misspecify_m:
        return lin
    j = cfg.p                                   # first surrogate column
    return (lin + 0.7 * Z[:, 2] ** 2 - 0.7 * Z[:, j + 1] ** 2
            + 1.2 * Z[:, j] / (1 + np.exp(-Z[:, 0] * Z[:, j] ** 2)))


BLOCK_STRENGTH_LEVELS = {
    # name              -> (beta multiplier, gamma multiplier)
    #
    # The rungs of the strength ladder, indexed by the gap in held-out AUC
    # between the surrogate block alone and the image block alone (at the
    # true parameter, on the validation rows), ordered from image-dominant to
    # surrogate-dominant, which is the order strength_reversal reads them in.
    # The paper's ladder uses six of them; its realised gaps (signal_base =
    # 0.85, reference p, q and nu) are
    #
    #   beta_strong_2x    image-dominant       -0.344
    #   balanced          balanced             -0.004
    #   gamma_soft        surrogate, soft      +0.075
    #   gamma_med         surrogate, medium    +0.124
    #   gamma_strong      surrogate, strong    +0.186
    #   gamma_strong_2x   surrogate-dominant   +0.331
    #
    # The four middle rungs span the range the application's three branches
    # occupy on the same axis. The other levels are not used in the paper.
    # The gaps are not scale-free: they move if p, q, nu or signal_base
    # change, which --operating-point checks without running the grid.
    "beta_strong_2x":     (2.0, 0.5),
    "beta_strong":        (1.4, 0.7),
    "beta_med":           (1.25, 0.8),
    "beta_soft":          (1.15, 0.87),
    "beta_mild":          (1.07, 0.93),
    "balanced":           (1.0, 1.0),
    "gamma_mild":         (0.93, 1.07),
    "gamma_soft":         (0.87, 1.15),
    "gamma_med":          (0.8, 1.25),
    "gamma_strong":       (0.7, 1.4),
    "gamma_strong_2x":    (0.5, 2.0),
}


def make_theta(cfg: Config, rng, s_common: int = 7, base: float | None = None):
    """The target parameter theta_bar, with the two blocks made symmetric
    except for one deliberate asymmetry.

    Both blocks receive s_common = 7 non-zero slopes, each signal_base *
    U(0.7, 1.3) with a random sign, times the block multipliers of
    cfg.block_strength. Holding the support and the distribution equal keeps
    "which block is stronger" from being confounded with "which block is
    which", and leaves block width (128 against 93) as the only structural
    difference. The realised single-block AUCs are recorded so the intended
    asymmetry can be checked against the achieved one. The intercept is set
    afterwards by calibrate_intercept.
    """
    mult_b, mult_g = BLOCK_STRENGTH_LEVELS[cfg.block_strength]
    base = cfg.signal_base if base is None else float(base)
    theta = np.zeros(cfg.d + 1)
    s_b = min(s_common, cfg.p)
    s_g = min(s_common, cfg.q)
    ib = rng.choice(cfg.p, s_b, replace=False)
    ig = cfg.p + rng.choice(cfg.q, s_g, replace=False)
    theta[1 + ib] = (mult_b * base * rng.choice([-1.0, 1.0], s_b)
                     * rng.uniform(0.7, 1.3, s_b))
    theta[1 + ig] = (mult_g * base * rng.choice([-1.0, 1.0], s_g)
                     * rng.uniform(0.7, 1.3, s_g))
    return theta, ib, ig


def single_block_auc(Z, y, theta, cfg: Config):
    """Discrimination achievable from each block alone at the true parameter.

    This is the realised counterpart of the intended strength level, and the
    ladder's AUC gap is read off it. It uses the true theta rather than a
    fit, so it measures the signal placed in each block rather than what a
    finite sample recovers from it; the application's block-strength
    diagnostic places its branches on the same axis with fitted models."""
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return {"auc_image_only": np.nan, "auc_surrogate_only": np.nan,
                "auc_joint": np.nan}
    eta_b = Z[:, :cfg.p] @ theta[1:1 + cfg.p] if cfg.p else np.zeros(len(Z))
    eta_g = Z[:, cfg.p:] @ theta[1 + cfg.p:]
    return {"auc_image_only": (roc_auc_score(y, eta_b) if cfg.p else np.nan),
            "auc_surrogate_only": roc_auc_score(y, eta_g),
            "auc_joint": roc_auc_score(y, eta_b + eta_g)}


def make_delta(cfg: Config, rng, scenario: str, ib, ig, magnitude: float):
    """Delta = theta_source - theta_target, with support fixed by the scenario.
    magnitude is in multiples of the radius sqrt(n_T / log d), the anchor's
    radius when one is set.

    In the parametric arm the first ten image columns feed the first three
    surrogates by construction, so perturbing them contaminates both blocks
    and blurs the scenario label. Delta is therefore drawn away from that
    coupling. Coupled columns still carry signal through theta; they are
    merely excluded from the perturbation, which is what keeps delta_0
    concentrated where the scenario says it is.

    Real features carry no such coupling, so cfg.block_coupling is cleared for
    the plasmode arm and the whole width of each block is available. Applying
    the parametric exclusion there would drop the leading image components for
    a reason that does not exist in the data.
    """
    Delta = np.zeros(cfg.d + 1)
    if scenario == "S0" or magnitude == 0:
        return Delta
    if cfg.block_coupling:
        n_couple_x, n_couple_s = min(10, cfg.p // 2), min(3, cfg.q // 2)
    else:
        n_couple_x = n_couple_s = 0
    free_x = np.arange(n_couple_x, cfg.p)
    free_s = cfg.p + np.arange(n_couple_s, cfg.q)

    if scenario == "S4":
        # D_dense: Delta spread over every free coordinate of both blocks,
        # with weights decaying as 1/sqrt(rank).
        #
        # S1-S3 place Delta on seven or eight coordinates, the regime where
        # the correction step can in principle recover it: with lambda_delta
        # at its assigned order sqrt(log d / n_T) = 0.164 (d = 221,
        # n_T = 200), a defect of L1 norm R spread over k coordinates needs
        # R > k * lambda, which at k = 8 is 1.31, or 0.22 radii. Over all 221
        # coordinates the same threshold is 36.3, or 5.97 radii, so no
        # correction engages at any magnitude the design covers (the paper's
        # dense-threshold equation). Recovering a dense delta_0 at one radius
        # would need n_T of order d log d, about 1200 here.
        #
        # This is the regime the application sits in: its features are
        # learned representations in which signal is distributed rather than
        # sparse. Cai et al. (2025, Remark 3) give the condition for negative
        # transfer there in terms of rates; this scenario is its
        # finite-sample counterpart.
        wx = 1.0 / np.sqrt(1.0 + np.arange(len(free_x)))
        ws = 1.0 / np.sqrt(1.0 + np.arange(len(free_s)))
        Delta[1 + free_x] = rng.choice([-1.0, 1.0], len(free_x)) * wx / wx.sum()
        Delta[1 + free_s] = rng.choice([-1.0, 1.0], len(free_s)) * ws / ws.sum()
        return _scale_delta(Delta, cfg, magnitude)

    if scenario == "S1":                      # image block only
        k = min(max(4, len(ib)), len(free_x))
        idx = 1 + rng.choice(free_x, k, replace=False)
    elif scenario == "S2":                    # surrogate block only
        k = min(max(4, len(ig)), len(free_s))
        idx = 1 + rng.choice(free_s, k, replace=False)
    else:                                     # both blocks
        k_x, k_s = min(4, len(free_x)), min(4, len(free_s))
        if k_x == 0 or k_s == 0:
            raise ValueError(
                f"S3 needs perturbable columns in BOTH blocks, but only "
                f"{k_x} image and {k_s} surrogate columns are free "
                f"(p={cfg.p}, q={cfg.q}, block_coupling={cfg.block_coupling}). "
                "S3 on one block is S1 or S2 under the wrong label."
            )
        idx = np.r_[1 + rng.choice(free_x, k_x, replace=False),
                    1 + rng.choice(free_s, k_s, replace=False)]
    if len(idx) == 0:
        # Raise rather than return a zero shift under a label that asserts a
        # block does not transfer; the surrogate-only plasmode mode (p = 0)
        # has no image column for S1.
        raise ValueError(
            f"{scenario} has no perturbable column: p={cfg.p}, q={cfg.q}, "
            f"block_coupling={cfg.block_coupling}. Drop this scenario from "
            "--scenarios rather than generating a zero shift under its label."
        )
    Delta[idx] = rng.choice([-1.0, 1.0], len(idx))
    return _scale_delta(Delta, cfg, magnitude)


def _scale_delta(Delta, cfg: Config, magnitude: float):
    """Scale so ||delta_0||_1 ~= magnitude * radius, using delta_0 = -a * Delta.

    anchor_budget equals the cell's budget unless an anchor is set, in which
    case ||delta_0||_1 is held at one absolute value across an n_T sweep. This
    is only the linearised starting point; calibrate_delta_to_budget then
    solves for the scale that hits the requested norm exactly.
    """
    a = cfg.n_S / (cfg.n_S + cfg.n_T)
    l1 = a * np.abs(Delta).sum()
    if l1 > 0:
        Delta = Delta * (magnitude * cfg.anchor_budget / l1)
    return Delta


def calibrate_delta_to_budget(cfg: Config, Z_T, theta_T, Delta, target_l1,
                              max_iter: int = 6, tol: float = 0.005):
    """Rescale Delta so the REALISED ||delta_0||_1 equals its requested value.

    make_delta scales Delta on the linearisation delta_0 = -a * Delta, but
    delta_0 is theta_bar - theta_star, and theta_star is a logistic projection
    of a mixture of two logistic means. The linearisation therefore misses the
    request, and not by a constant that could be divided out: for the sparse
    scenarios the realised norm runs up to 28% above it, peaking near one
    radius and falling as the logit saturates (to below the request at three
    radii), and only under D_dense is it close to exact (the "before" column
    of the paper's calibration table). It also varies by replicate, since the
    supports of Delta and theta_T are drawn.

    Solving for the scale makes the axis mean what it says, and gives D_X and
    D_S the same ||delta_0||_1 in every replicate, so a row of the strength
    ladder changes only WHERE delta_0 sits, not how big it is.

    g(c) = ||delta_0(theta_T, theta_T + c*Delta)||_1 is monotone in c, so the
    fixed point c <- c * target / g(c) converges. Nothing here reads a label:
    the iteration sees only the unlabelled target rows and the two parameter
    vectors, so it is a calibration of the design, not an estimate.

    Returns (Delta_scaled, delta_0, info); the caller reuses the returned
    delta_0 rather than paying for it twice.
    """
    d0 = oracle_delta0(Z_T, theta_T, theta_T + Delta, cfg)
    info = {"delta0_calibration_iters": 0,
            "delta0_calibration_ratio_initial": np.nan,
            "delta0_calibration_ratio_final": np.nan,
            "delta0_calibration_scale": 1.0,
            "delta0_calibration_converged": True}
    if target_l1 <= 0 or not np.any(Delta):
        return Delta, d0, info                      # S0, or magnitude 0

    # The requested norm is not always attainable. delta_0 is a difference of
    # two logistic projections, so once the linear predictor saturates, g(c)
    # flattens and no scale reaches the target: 18 of the paper's 4250
    # shifted replicates stop short, seventeen of them D_XS at three radii,
    # and at a small p, magnitude 2 can sit above the ceiling entirely.
    # Chasing it would inflate Delta several fold and move the generating
    # process into near-deterministic labels, a qualitatively different DGP,
    # which is a worse failure than missing the number. So the step is
    # bounded, the closest scale seen is what gets returned, and the
    # shortfall is recorded rather than hidden.
    c, ratios = 1.0, []
    best = (float("inf"), 1.0, d0)
    previous_gap = float("inf")
    for _ in range(max_iter):
        got = float(np.abs(d0[1:]).sum())
        ratios.append(got / target_l1)
        gap = abs(ratios[-1] - 1.0)
        if gap < best[0]:
            best = (gap, c, d0)
        if got <= 0 or gap <= tol:
            break
        # Saturation guard. If the previous rescale bought almost no
        # improvement, g has flattened and no further scale reaches the
        # target. Stopping here keeps Delta near the size the scenario asked
        # for; without it the loop runs the scale up several fold for a few
        # percent of norm and quietly swaps the DGP for a near-deterministic
        # one, which corrupts far more than the axis label it was chasing.
        if gap > 0.99 * previous_gap:
            break
        previous_gap = gap
        c *= float(np.clip(1.0 / ratios[-1], 0.5, 2.0))
        d0 = oracle_delta0(Z_T, theta_T, theta_T + c * Delta, cfg)
    else:
        ratios.append(float(np.abs(d0[1:]).sum()) / target_l1)
        if abs(ratios[-1] - 1.0) < best[0]:
            best = (abs(ratios[-1] - 1.0), c, d0)

    gap, c, d0 = best
    info.update({
        "delta0_calibration_iters": len(ratios),
        "delta0_calibration_ratio_initial": float(ratios[0]),
        "delta0_calibration_ratio_final": float(1.0 + np.sign(
            ratios[-1] - 1.0) * gap) if gap < float("inf") else np.nan,
        "delta0_calibration_scale": float(c),
        "delta0_calibration_converged": bool(gap <= tol),
    })
    return c * Delta, d0, info


def calibrate_intercept(Z, theta, cfg: Config):
    """Shift the intercept so the event rate on the ROWS PASSED IN matches the
    application's target prevalence. Callers pass target rows only: pinning
    the pooled rate leaves the target rate free, and the target rate is the
    one the application's 10.5% refers to."""
    lo, hi = -20.0, 20.0
    th = theta.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        th[0] = mid
        if np.mean(1 / (1 + np.exp(-outcome_logit(Z, th, cfg)))) > cfg.prevalence:
            hi = mid
        else:
            lo = mid
    th[0] = 0.5 * (lo + hi)
    return th


def oracle_delta0(Z_T, theta_T, theta_S, cfg: Config):
    """delta_0 = theta_bar - theta_star, both solved numerically.

    theta_bar is the logistic projection of the true target conditional mean
    onto the working model, which equals theta_T only when that model is
    correctly specified; under misspecification it must be computed. theta_star
    is the maximiser of the w-tilted objective, which under a correct density
    ratio reduces to a logistic fit on the target covariate distribution
    against the mixture pseudo-outcome a*m_S + (1-a)*m_T."""
    a = cfg.n_S / (cfg.n_S + cfg.n_T)
    m_S = 1 / (1 + np.exp(-outcome_logit(Z_T, theta_S, cfg)))
    m_T = 1 / (1 + np.exp(-outcome_logit(Z_T, theta_T, cfg)))
    theta_bar = _soft_label_logistic(Z_T, m_T)
    theta_star = _soft_label_logistic(Z_T, a * m_S + (1 - a) * m_T)
    return theta_bar - theta_star


def _soft_label_logistic(Z, m, max_iter: int = 4000):
    """Unpenalised logistic projection onto the working model, fitted against
    a soft label by row duplication.

    Convergence is asserted rather than assumed. ConvergenceWarning is
    silenced module-wide at import, and this solve produces delta_0, which is
    the ground truth every scenario label and every identification rate rests
    on: a silent failure here would not corrupt one number, it would corrupt
    the definition of correct. Comfortable at the reference grid (about 14
    iterations at d = 221 on 5000 rows), but a dimension sweep is exactly the
    setting that would strain it.
    """
    X = np.hstack([np.ones((len(Z), 1)), Z])
    Xd = np.vstack([X, X])
    yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
    wd = np.r_[m, 1 - m]
    clf = LogisticRegression(C=1e6, max_iter=max_iter, solver="lbfgs",
                             fit_intercept=False)
    clf.fit(Xd, yd, sample_weight=wd)
    if int(np.max(clf.n_iter_)) >= max_iter:
        raise RuntimeError(
            f"the oracle logistic projection hit {max_iter} iterations without "
            f"converging at d={Z.shape[1]} on {len(Z)} rows. delta_0 would be "
            "wrong, and delta_0 is the ground truth, so the replicate is "
            "discarded rather than reported."
        )
    return clf.coef_[0]


# ======================================================================
# One replicate
# ======================================================================

def run_replicate(cfg: Config, scenario: str, magnitude: float, rep: int,
                  Z_pool=None, R_pool=None):
    seed = cfg.seed + 1000 * rep

    # Separate streams. theta_T, the split and the outcome noise are seeded
    # from the replicate alone and never touched by anything scenario
    # dependent, so every scenario at this replicate sees the SAME target
    # parameter, the SAME row assignment and the SAME uniform draws (common
    # random numbers): scenarios differ only in Delta, which is what pairs
    # them. make_delta consumes a scenario-dependent number of variates, which
    # is why Delta has a stream of its own.
    rng_pool = np.random.default_rng(seed)             # covariates, if needed
    rng_param = np.random.default_rng(seed + 100_001)  # theta_T, ib, ig
    rng_delta = np.random.default_rng(seed + 200_002)  # Delta, per scenario
    rng_split = np.random.default_rng(seed + 300_003)  # row assignment
    rng_noise = np.random.default_rng(seed + 400_004)  # Bernoulli draws

    if Z_pool is None:
        Z, R = generate_parametric(cfg, rng_pool)
    else:
        Z, R = Z_pool, R_pool

    src = np.flatnonzero(R == 1)
    tgt = np.flatnonzero(R == 0)

    theta_T, ib, ig = make_theta(cfg, rng_param)
    Delta = make_delta(cfg, rng_delta, scenario, ib, ig, magnitude)
    # Intercept set by bisection on TARGET rows, not on the pooled draw:
    # theta_T is the target parameter and cfg.prevalence the application's
    # target event rate (0.105), while domain membership depends on Z, so the
    # two populations carry different event rates at one shared intercept.
    # Delta has no intercept component, so the source rate is left free: the
    # source is a shifted population, not a recalibrated one.
    theta_T = calibrate_intercept(Z[tgt], theta_T, cfg)

    rng_split.shuffle(src)
    rng_split.shuffle(tgt)
    need_s, need_t = cfg.n_S + cfg.N_S, cfg.n_T + cfg.N_T + cfg.n_valid
    if len(src) < need_s or len(tgt) < need_t:
        raise RuntimeError(f"pool too small: {len(src)} source, {len(tgt)} target")
    i_SL, i_SU = src[:cfg.n_S], src[cfg.n_S:need_s]
    i_TL = tgt[:cfg.n_T]
    i_val = tgt[cfg.n_T:cfg.n_T + cfg.n_valid]
    i_TU = tgt[cfg.n_T + cfg.n_valid:need_t]

    # Delta is calibrated here, after the split fixes which unlabelled rows
    # delta_0 is defined on and before any label is drawn from theta_S. The
    # iteration consumes no random numbers, so the common-random-number
    # property established above survives it untouched.
    Z_oracle = Z[i_TU[:5000]]
    if cfg.calibrate_delta0:
        Delta, d0, delta0_cal = calibrate_delta_to_budget(
            cfg, Z_oracle, theta_T, Delta, magnitude * cfg.anchor_budget)
    else:
        d0 = oracle_delta0(Z_oracle, theta_T, theta_T + Delta, cfg)
        target = magnitude * cfg.anchor_budget
        ratio = (float(np.abs(d0[1:]).sum()) / target if target > 0 else np.nan)
        delta0_cal = {"delta0_calibration_iters": 0,
                      "delta0_calibration_ratio_initial": ratio,
                      "delta0_calibration_ratio_final": ratio,
                      "delta0_calibration_scale": 1.0,
                      "delta0_calibration_converged": False}
    theta_S = theta_T + Delta

    # Realised event rates, recorded so the calibration is visible per
    # replicate rather than assumed to have taken.
    prevalence_diag = {
        "prevalence_target": float(np.mean(
            expit(outcome_logit(Z[tgt], theta_T, cfg)))),
        "prevalence_source": float(np.mean(
            expit(outcome_logit(Z[src], theta_S, cfg)))),
        "prevalence_requested": float(cfg.prevalence),
    }

    # Common random numbers: the uniforms are drawn in a fixed order and a
    # fixed size, so row i receives the same variate under every scenario and
    # the difference in y is attributable to theta alone.
    def draw(idx, theta):
        pr = 1 / (1 + np.exp(-outcome_logit(Z[idx], theta, cfg)))
        return (rng_noise.uniform(size=len(idx)) < pr).astype(int)

    y_SL = draw(i_SL, theta_S)
    y_TL = draw(i_TL, theta_T)
    y_val = draw(i_val, theta_T)

    # Realised class composition. The application's split fixes these; here
    # they are drawn, so recording them is what lets a reader tell design
    # noise from estimator behaviour when reading the replicate-to-replicate
    # spread.
    prevalence_diag.update({
        "n_tl_positive": int(y_TL.sum()),
        "n_val_positive": int(y_val.sum()),
        "n_sl_positive": int(y_SL.sum()),
        "n_tl_positive_expected": int(round(cfg.n_T * cfg.prevalence)),
    })

    # Ground truth: the delta_0 the calibration above already solved for, not
    # a second solve at the same arguments.
    truth = {
        "delta0_l1": float(np.abs(d0[1:]).sum()),
        "delta0_beta_l1": float(np.abs(d0[1:1 + cfg.p]).sum()),
        "delta0_gamma_l1": float(np.abs(d0[1 + cfg.p:]).sum()),
    }
    truth["delta0_l1_over_budget"] = truth["delta0_l1"] / cfg.budget
    # Measured against the anchor too. delta0_l1_over_budget divides by the
    # cell's own radius, which is deliberate (the anchor changes what is
    # generated, never how it is measured), but under an anchored n_T sweep
    # one fixed delta_0 then reads as a different multiple in every cell (1.41
    # radii at n_T = 100, 0.50 at 800 in the paper). The anchored version is
    # recorded alongside, for reading such a sweep at fixed delta_0.
    truth["delta0_l1_over_anchor_budget"] = truth["delta0_l1"] / cfg.anchor_budget
    # delta_0 measured against theta itself, not against the radius. The
    # radius is absolute while theta is not, so this is the column that says
    # whether a scenario label still describes the design: well above one, the
    # source parameter is no longer a perturbation of the target one and
    # "block X does not transfer" stops being an accurate summary (the last
    # column of the paper's calibration table).
    truth["theta_l1"] = float(np.abs(theta_T[1:]).sum())
    truth["delta0_l1_over_theta_l1"] = (
        truth["delta0_l1"] / truth["theta_l1"]
        if truth["theta_l1"] > 1e-12 else np.nan)
    # theta_star is a logistic projection of a mixture of two logistic means,
    # and a mixture of links is not the link of a mixture, so delta_0 never
    # lands exactly on the support of Delta: some of it always leaks into the
    # block the scenario left alone. The block label is therefore assigned by
    # dominance, and the realised share is reported so the purity of each
    # scenario is visible rather than assumed.
    share = (truth["delta0_beta_l1"] / truth["delta0_l1"]
             if truth["delta0_l1"] > 1e-8 else np.nan)
    truth["delta0_beta_share"] = float(share) if share == share else np.nan
    # The label of the paper's dominance check: the block carrying more than
    # 65% of the realised ||delta_0||_1, "both" in between, and "none" when
    # ||delta_0||_1 is below 2% of the radius (D_empty, where delta_0 is zero
    # to machine precision). The cut is judged against the anchor, the scale
    # Delta was generated on; against the cell's own radius it would tighten
    # with n_T. Under S3 and S4 the share sits near one half, so they are
    # labelled "both" and not scored for identification.
    truth["nontransferable_block"] = (
        "none" if truth["delta0_l1"] < 0.02 * cfg.anchor_budget else
        "beta" if share > 0.65 else "gamma" if share < 0.35 else "both")

    if len(np.unique(y_TL)) < 2 or len(np.unique(y_SL)) < 2:
        raise RuntimeError("degenerate labels in a labelled block")

    # Realised strength, measured at the true parameter on held-out target
    # rows. Reported so the intended level can be checked against what the
    # design actually produced, and so the application's measured values can
    # be located on the same axis.
    realised = single_block_auc(Z[i_val], y_val, theta_T, cfg)

    w_SL, dr = fit_density_ratio(Z[i_SU], Z[i_TU], Z[i_SL], cfg, seed)
    out = shapley_blocks(Z[i_TL], y_TL, Z[i_SL], y_SL, w_SL, cfg, seed)

    # Full-sample fit, for the delta diagnostics the application reports.
    th_T, th_tilde, th_meta, delta = fit_theta_pair(
        Z[i_TL], y_TL, Z[i_SL], y_SL, w_SL, cfg, seed)

    # The RSC tolerance term at the realised norm (the paper's assessment of
    # epsilon0). The schedule epsilon0 = log d / n_T takes the prefactor of
    # the tolerance component of the sufficient condition and fixes the
    # multiplier at one, without claiming that the condition holds. Here
    # theta_bar is known, so the norm the condition is stated in can be
    # evaluated: T_RSC = (log d / n_T) ||theta_hat(meta) - theta_bar||_1^2,
    # the unit-weight proxy for that component, over the slopes. An
    # application cannot compute it, since theta_bar is unknown there.
    #
    # Under the unreported misspecified imputation model theta_bar is the
    # projection of the target law onto the working model and no longer
    # equals theta_T, so the column is exact only for m_cor.
    if th_meta.shape != theta_T.shape:
        raise RuntimeError(
            f"meta {th_meta.shape} and truth {theta_T.shape} are not aligned; "
            "the l1 gap is only meaningful for matching coefficient vectors"
        )
    gap_l1 = float(np.abs(th_meta[1:] - theta_T[1:]).sum())   # slopes only
    eps0_required = float(np.log(cfg.d) / cfg.n_T * gap_l1 ** 2)

    # Held-out performance per coalition, at both layers of the paper:
    #
    #   imp   theta^(c), the imputation model, reading image and surrogate;
    #         phi is defined on this layer
    #   pred  beta_hat(theta^(c)), Algorithm 1's image-only outcome model,
    #         fitted on TL plus the soft-imputed TU rows; the endpoints of
    #         the paper and of the application are scored on this layer
    #
    # Each coalition gets its own beta fit rather than a re-scoring of a
    # shared one. Scoring both layers is what lets the paper separate what
    # phi measures from what reaches the deployed model.
    imp, pred, beta_info = {}, {}, {}
    fit_pred = bool(cfg.fit_prediction_layer) and cfg.p > 0
    for c in COALITIONS:
        th = _theta_of(c, th_T, th_meta, cfg.p)
        # The intercept is refitted on TL at the coalition's slopes, as in the
        # paper's coalition definition and in the pipeline: a spliced
        # coalition inherits an intercept that belonged to neither fit, and
        # left alone it would make the Brier and ECE comparisons partly a
        # comparison of calibration offsets. AUC, a rank statistic, would not
        # show it.
        a = _refit_intercept(Z[i_TL] @ th[1:], y_TL)
        imp[c] = _score_predictions(y_val, a + Z[i_val] @ th[1:])

        if not fit_pred:
            pred[c] = {k: float("nan") for k in ("auc", "brier", "ece")}
            beta_info[c] = (float("nan"), -1)
            continue
        # The refitted intercept drives the imputation, as run_strifle passes
        # coalition_theta's intercept into fit_beta: the soft labels inherit
        # the calibration of theta^(c), so the refit propagates to beta.
        th_c = th.copy()
        th_c[0] = a
        b0, b, bC = fit_beta_imputed(
            th_c, Z[i_TL], y_TL, Z[i_TU], cfg,
            seed + 500 + COALITIONS.index(c),
            theta_oracle=theta_T if cfg.oracle_imputation else None)
        pred[c] = _score_predictions(y_val, b0 + Z[i_val][:, :cfg.p] @ b)
        beta_info[c] = (float(bC), int(np.count_nonzero(b)))
    row = {"scenario": scenario, "magnitude": magnitude, "repeat": rep,
           "p": cfg.p, "q": cfg.q, "d_total": cfg.d,
           "n_T": cfg.n_T, "n_S": cfg.n_S, "nu": cfg.nu,
           "signal_base": cfg.signal_base,
           "dgp": cfg.dgp_label, "block_strength": cfg.block_strength,
           **prevalence_diag,
           **delta0_cal,
           **realised,
           **truth,
           "phi_beta": out["phi_beta"], "phi_gamma": out["phi_gamma"],
           "phi_total": out["phi_total"],
           "efficiency_gap": out["efficiency_gap"],
           "rho": out["rho"], "selected": out["selected"],
           "eps0_applied": out["eps0"],
           "theta_meta_minus_truth_l1": gap_l1,
           "eps0_required": eps0_required,
           # T_RSC / epsilon0. Under the schedule it equals gap_l1^2, which
           # is why the paper's table has its last column the square of the
           # third; it is stored separately so that an epsilon_0 override
           # does not silently change its meaning.
           "eps0_conservatism": (eps0_required / out["eps0"]
                                 if out["eps0"] > 0 else float("inf")),
           "delta_l1": float(np.abs(delta).sum()),
           "delta_beta_l1": float(np.abs(delta[:cfg.p]).sum()),
           "delta_gamma_l1": float(np.abs(delta[cfg.p:]).sum()),
           "budget": cfg.budget,
           "anchor_budget": cfg.anchor_budget,
           "delta0_anchor_n_T": (cfg.delta0_anchor_n_T
                                 if cfg.delta0_anchor_n_T is not None
                                 else cfg.n_T),
           "dr_domain_auc": dr["domain_auc"],
           "dr_domain_auc_thresholded": dr["domain_auc_thresholded"],
           "dr_n_retained": dr["n_retained"],
           "dr_ess": dr["ess"],
           "dr_frac_clipped": dr["frac_clipped"]}
    
    row.update({f"L_{c}": out["L"][c] for c in COALITIONS})
    for layer, scores in (("imp", imp), ("pred", pred)):
        for metric in ("auc", "brier", "ece"):
            row.update({f"{metric}_{layer}_{c}": scores[c][metric]
                        for c in COALITIONS})
    row["prediction_layer_fitted"] = bool(fit_pred)
    row["oracle_imputation"] = bool(cfg.oracle_imputation)
    row["delta_holdout"] = bool(cfg.delta_holdout)
    row.update({f"beta_C_{c}": beta_info[c][0] for c in COALITIONS})
    row.update({f"beta_nnz_{c}": beta_info[c][1] for c in COALITIONS})
    return row


# ======================================================================
# Scenario driver and reporting
# ======================================================================

def scenario_grid(scenarios, magnitudes):
    """(scenario, magnitude) points, with the degenerate duplicates removed.

    make_delta returns a zero Delta whenever magnitude == 0, so S1 and S2 at
    magnitude 0 are identical to S0 in every respect: same theta_S, same
    delta_0, same labels. Running them spends replicates on a cell already
    covered and, worse, files them under a scenario whose label asserts that a
    named block does not transfer while delta_0 is exactly zero. The truth
    column then reads "none", the identification rate is undefined, and any
    table that pools over magnitudes silently mixes those rows in.

    Magnitude 0 is therefore collapsed to one S0 cell, which is emitted if S0
    was requested or if a zero magnitude was requested alongside a shifted
    scenario.
    """
    magnitudes = [float(m) for m in magnitudes]
    known = {"S0", "S1", "S2", "S3", "S4"}
    unknown = [s for s in scenarios if s not in known]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; choose from "
                         f"{sorted(known)}")
    shifted = [s for s in scenarios if s != "S0"]
    zero_requested = any(m == 0.0 for m in magnitudes)

    grid = []
    if "S0" in scenarios or (zero_requested and shifted):
        grid.append(("S0", 0.0))
    for s in shifted:
        for m in magnitudes:
            if m == 0.0:
                continue
            grid.append((s, m))
    return grid


def run_cell(cfg: Config, scenarios, magnitudes, reps: int, arm: str,
             Z_pool=None, R_pool=None, progress: Callable | None = None):
    """All replicates for one grid point."""
    rows = []
    grid = scenario_grid(scenarios, magnitudes)
    # Replicate-outer, so the covariate pool is generated once per replicate
    # and reused across scenarios. Scenarios then differ only in Delta, which
    # pairs them and removes between-pool variance from every contrast.
    for rep in range(reps):
        if Z_pool is None:
            pool_rng = np.random.default_rng(cfg.seed + 7_000_000 + rep)
            Z_rep, R_rep = generate_parametric(cfg, pool_rng)
        else:
            Z_rep, R_rep = Z_pool, R_pool
        for scen, mag in grid:
            try:
                rows.append(run_replicate(cfg, scen, mag, rep, Z_rep, R_rep))
            except (RuntimeError, ValueError) as exc:
                print(f"    [skip] {scen} mag={mag} rep={rep}: {exc}")
            if progress:
                progress()
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["arm"] = arm
    return frame


# ======================================================================
# Grid specification
# ======================================================================

SWEEPABLE = {"p", "q_text", "q_meta", "n_T", "n_S", "N_T", "N_S",
             "nu", "prevalence", "n_valid", "seed", "signal_base"}


def parse_sweep(spec: str) -> dict:
    """'n_T=100,200,500' -> {'n_T': [100, 200, 500]}. Repeatable across
    several --sweep flags; the cross product is taken."""
    field_name, _, values = spec.partition("=")
    field_name = field_name.strip()
    if field_name not in SWEEPABLE:
        raise argparse.ArgumentTypeError(
            f"'{field_name}' is not sweepable; choose from "
            f"{sorted(SWEEPABLE)}")
    if not values:
        raise argparse.ArgumentTypeError(f"no values given for '{field_name}'")
    cast = float if field_name in ("nu", "prevalence", "signal_base") else int
    return {field_name: [cast(v) for v in values.split(",")]}


def build_grid(base: Config, sweeps: list[dict], dgps: list[str],
               strengths: list[str]):
    """Cross product of every swept field with every DGP variant and every
    block-strength level."""
    from itertools import product
    merged: dict = {}
    for s in sweeps:
        merged.update(s)
    names = sorted(merged)
    combos = list(product(*[merged[n] for n in names])) if names else [()]

    cells = []
    for strength in strengths:
        if strength not in BLOCK_STRENGTH_LEVELS:
            raise SystemExit(
                f"unknown block strength '{strength}'; choose from "
                f"{list(BLOCK_STRENGTH_LEVELS)}")
        for dgp in dgps:
            m_mis = dgp.startswith("m_mis")
            w_mis = dgp.endswith("w_mis")
            for combo in combos:
                kw = dict(zip(names, combo))
                cfg = Config(**{**base.__dict__, **kw,
                                "misspecify_m": m_mis, "misspecify_w": w_mis,
                                "block_strength": strength})
                label = ", ".join(f"{n}={v}" for n, v in kw.items()) or "base"
                cells.append((f"{label} | {dgp} | {strength}", cfg))
    return cells


# ======================================================================
# Reporting
# ======================================================================

GRID_KEYS = ["p", "q", "d_total", "n_T", "n_S", "nu", "signal_base", "dgp",
             "block_strength"]


def identification(cell: pd.DataFrame) -> dict:
    """Identification rate, scored replicate by replicate, and label purity.

    Identification is the paper's: the share of a cell's replicates in which
    phi is smaller for the block carrying delta_0. A mixture of links is not
    the link of a mixture, so some of delta_0 always leaks into the block the
    scenario left alone, and each replicate is scored against the dominance
    label of its own realised delta_0 (nontransferable_block) rather than
    against its scenario. label_purity, the share of replicates carrying the
    cell's modal label, shows whether that leakage ever changes a label; in
    the paper's runs it never does. Replicates whose delta_0 names no block
    (below 2% of the radius, or split between the blocks as under D_XS and
    D_dense) are left out of the denominator and counted in n_unscorable.
    """
    truth = cell["nontransferable_block"].to_numpy()
    called = np.where(cell["phi_beta"].to_numpy() < cell["phi_gamma"].to_numpy(),
                      "beta", "gamma")
    scorable = np.isin(truth, ("beta", "gamma"))
    modal = (cell["nontransferable_block"].mode().iat[0]
             if len(cell) else "none")
    return {
        "truth_block": modal,
        "label_purity": float(np.mean(truth == modal)) if len(cell) else np.nan,
        "n_scorable": int(scorable.sum()),
        "n_unscorable": int((~scorable).sum()),
        "identification_rate": (float(np.mean(called[scorable] == truth[scorable]))
                                if scorable.any() else np.nan),
    }


def strength_reversal(frame: pd.DataFrame, out_dir: str):
    """The paper's block-strength ladder: identification under D_X (S1) and
    D_S (S2) at each rung, with the realised AUC gap, surrogate block alone
    minus image block alone, that indexes the rung.

    The gap phi_S - phi_X is a sum of two terms, one set by the strength
    imbalance and unrelated to delta_0, the other carrying the position and
    size of delta_0, and identification reads only the sign of the sum. The
    two rates should therefore run in opposite directions along the ladder
    and cross where the blocks are equally strong, which is what the paper
    finds. The verdict is "reversed" when S1 - S2 has opposite signs at the
    first and the last rung of the ladder requested; if S1 or S2 stays ahead
    at every rung, the asymmetry is not explained by strength.
    """
    if frame["block_strength"].nunique() < 2:
        return None
    sub = frame[frame.scenario.isin(["S1", "S2"])]
    if sub.empty:
        return None

    print("\n--- Identification against block strength (the ladder) ---")
    print("phi_S - phi_X carries a strength term unrelated to delta_0, so the "
          "S1 and S2 rates\nshould run in opposite directions along the "
          "ladder and cross where the blocks\nare equally strong.\n")
    order = [s for s in BLOCK_STRENGTH_LEVELS if s in
             set(sub.block_strength.unique())]
    rows = []
    for level in order:
        cell = sub[sub.block_strength == level]
        rec = {"block_strength": level,
               "auc_image_only": cell.auc_image_only.mean(),
               "auc_surrogate_only": cell.auc_surrogate_only.mean()}
        rec["auc_gap_surrogate_minus_image"] = (rec["auc_surrogate_only"]
                                                - rec["auc_image_only"])
        for scen in ("S1", "S2"):
            s = cell[cell.scenario == scen]
            if s.empty:
                rec[f"{scen}_identification"] = np.nan
                continue
            ident = identification(s)
            rec[f"{scen}_identification"] = ident["identification_rate"]
            rec[f"{scen}_truth"] = ident["truth_block"]
            rec[f"{scen}_label_purity"] = ident["label_purity"]
            rec[f"{scen}_n_scorable"] = ident["n_scorable"]
        rows.append(rec)
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    table.to_csv(os.path.join(out_dir, "simulation_strength_reversal.csv"),
                 index=False)

    # The table above pools the four magnitudes of each rung, as the paper's
    # ladder table does. Resolved by magnitude it gives the paper's table of
    # the surrogate rungs by ||delta_0||_1, where the gap phi_S - phi_X
    # starts positive under D_S and turns negative as the norm grows.
    if sub["magnitude"].nunique() > 1:
        mrows = []
        for level in order:
            for mag, cell in sub[sub.block_strength == level].groupby(
                    "magnitude"):
                rec = {"block_strength": level, "magnitude": mag,
                       "delta0_over_budget":
                           float(cell["delta0_l1_over_budget"].mean())}
                for scen in ("S1", "S2"):
                    s = cell[cell.scenario == scen]
                    if s.empty:
                        rec[f"{scen}_identification"] = np.nan
                        continue
                    ident = identification(s)
                    rec[f"{scen}_identification"] = ident["identification_rate"]
                    rec[f"{scen}_label_purity"] = ident["label_purity"]
                    rec[f"{scen}_phi_beta"] = float(s.phi_beta.mean())
                    rec[f"{scen}_phi_gamma"] = float(s.phi_gamma.mean())
                mrows.append(rec)
        by_mag = pd.DataFrame(mrows)
        print("\n--- Identification against block strength, by magnitude ---")
        print("delta_0 has to cancel the strength imbalance before it shows: "
              "identification\ncrosses 0.5 a little after the gap phi_S - "
              "phi_X crosses zero, at a radius that\nrises with the strength "
              "gap.\n")
        print(by_mag.round(4).to_string(index=False))
        by_mag.to_csv(
            os.path.join(out_dir, "simulation_strength_reversal_by_"
                                  "magnitude.csv"), index=False)

    s1 = table["S1_identification"].to_numpy(float)
    s2 = table["S2_identification"].to_numpy(float)
    ok = np.isfinite(s1) & np.isfinite(s2)
    verdict = "inconclusive"
    if ok.sum() >= 2:
        diff = s1[ok] - s2[ok]
        if diff[0] * diff[-1] < 0:
            verdict = "reversed"
            print("\n  The two rates cross along the ladder: the ordering of "
                  "phi follows which block\n  is stronger as well as where "
                  "delta_0 sits, and is interpretable only where the\n  "
                  "blocks are comparably strong.")
        elif np.all(diff > 0):
            verdict = "no_reversal_favours_beta_detection"
            print("\n  [WATCH] S1 exceeds S2 at every rung, including where "
                  "the image block is the\n  stronger one. The asymmetry is "
                  "not explained by strength: the ordering of phi\n  finds "
                  "an image-block defect more readily for some other reason.")
        else:
            verdict = "no_reversal_favours_gamma_detection"
            print("\n  [WATCH] S2 exceeds S1 at every rung, including where "
                  "the surrogate block is\n  the stronger one. The asymmetry "
                  "is not explained by strength.")
    return {"verdict": verdict, "table": table.to_dict("records")}


def report(frame: pd.DataFrame, out_dir: str = ".") -> dict:
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    # Proposition (efficiency): phi_X + phi_S = v(XS) by construction, so a
    # residual above rounding is an implementation error, not a finding.
    gap = frame["efficiency_gap"].max()
    print(f"\nEfficiency, max |phi_beta + phi_gamma - v(both)| = {gap:.3e}  "
          f"{'OK' if gap < 1e-9 else 'FAIL: inconsistent value function'}")

    keys = [k for k in GRID_KEYS if frame[k].nunique() > 1]
    varying = keys or ["d_total"]
    if keys:
        print(f"\nGrid varies: {', '.join(keys)}")

    # ---- attribution against known ground truth -------------------------
    print("\n--- Attribution against known ground truth ---")
    print("identification_rate is the share of replicates in which phi is "
          "smaller for the\nblock carrying delta_0, scored against each "
          "replicate's own dominance label.\nphi_gap is phi_S - phi_X "
          "(phi_gamma - phi_beta) with a paired 95% t-interval.")
    print("label_purity is the share of replicates carrying the cell's modal "
          "label: below\none means the leakage of delta_0 changed a label, "
          "and n_unscorable counts\nreplicates whose delta_0 names no "
          "block.\n")
    rows = []
    for gvals, cell in frame.groupby(varying + ["scenario", "magnitude"],
                                     dropna=False):
        pb, pg = cell["phi_beta"].values, cell["phi_gamma"].values
        d = pg - pb
        ci = (stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
              if len(d) > 1 and d.std() > 0 else (np.nan, np.nan))
        rec = dict(zip(varying + ["scenario", "magnitude"], gvals))
        rec.update({"n": len(cell)})
        rec.update(identification(cell))
        rec.update({"auc_image_only": cell["auc_image_only"].mean(),
                    "auc_surrogate_only": cell["auc_surrogate_only"].mean(),
                    "delta0_beta_share": cell["delta0_beta_share"].mean(),
                    "phi_beta": pb.mean(), "phi_gamma": pg.mean(),
                    "phi_gap": d.mean(), "ci_low": ci[0], "ci_high": ci[1],
                    "delta0_over_budget":
                        cell["delta0_l1_over_budget"].mean(),
                    "rho_rate": cell["rho"].mean()})
        rows.append(rec)
    attrib = pd.DataFrame(rows)
    print(attrib.round(4).to_string(index=False))
    attrib.to_csv(os.path.join(out_dir, "simulation_attribution.csv"),
                  index=False)

    reversal = strength_reversal(frame, out_dir)

    # ---- how much design noise is in the class composition? -------------
    if "n_tl_positive" in frame.columns:
        comp = frame.groupby(varying, dropna=False).agg(
            n=("n_tl_positive", "size"),
            expected=("n_tl_positive_expected", "first"),
            tl_pos_mean=("n_tl_positive", "mean"),
            tl_pos_sd=("n_tl_positive", "std"),
            tl_pos_min=("n_tl_positive", "min"),
            tl_pos_max=("n_tl_positive", "max"),
            val_pos_mean=("n_val_positive", "mean"),
            val_pos_sd=("n_val_positive", "std"),
        ).reset_index()
        print("\n--- Class composition (design noise) ---")
        print("The application's patient split fixes the positive counts by "
              "construction: 21\nin the 200 validation rows, and "
              "round(n_T x prevalence) in TL, 21 at n_T = 200.\nHere outcomes "
              "are drawn, so the labelled positive count is "
              "Binomial(n_T, prevalence),\na layer of design noise the "
              "application does not have (the paper's supplementary\ntable "
              "of positive counts).\n")
        print(comp.round(2).to_string(index=False))
        comp.to_csv(os.path.join(out_dir, "simulation_class_composition.csv"),
                    index=False)

    # ---- did the delta_0 calibration take? ------------------------------
    if "delta0_calibration_converged" in frame.columns:
        cal = frame.groupby(varying + ["scenario", "magnitude"],
                            dropna=False).agg(
            n=("delta0_calibration_converged", "size"),
            converged=("delta0_calibration_converged", "mean"),
            iters=("delta0_calibration_iters", "mean"),
            ratio_before=("delta0_calibration_ratio_initial", "mean"),
            ratio_after=("delta0_calibration_ratio_final", "mean"),
            scale=("delta0_calibration_scale", "mean"),
            d0_over_theta=("delta0_l1_over_theta_l1", "mean"),
        ).reset_index()
        shifted = cal[cal["scenario"] != "S0"]
        if len(shifted):
            print("\n--- delta_0 calibration ---")
            print("The scale of Delta is solved for until the realised "
                  "||delta_0||_1 equals\nmagnitude x radius (the anchor's "
                  "radius when one is set). ratio_before is what\nthe "
                  "linearisation delta_0 = -a Delta delivers, ratio_after "
                  "what was achieved\n(the paper's calibration table). Cells "
                  "with converged < 1 mostly sit where the\nlogit saturates "
                  "and the requested norm is not attainable at any scale; "
                  "read\nthem on delta0_l1_over_anchor_budget rather than on "
                  "magnitude.\nd0_over_theta is ||delta_0||_1 / "
                  "||theta_bar||_1, the table's last column: it says\nwhether "
                  "the source parameter is still a perturbation of the target "
                  "one.\n")
            print(shifted.round(4).to_string(index=False))
            cal.to_csv(os.path.join(out_dir, "simulation_delta0_calibration.csv"),
                       index=False)
            bad = shifted[shifted["converged"] < 1.0]
            if len(bad):
                print(f"\n  [WATCH] {len(bad)} of {len(shifted)} cells did not "
                      "reach the requested norm in every\n  replicate. Mostly "
                      "a property of the design point rather than a solver "
                      "failure:\n  the scenario asks for more shift than the "
                      "link can express (in the paper,\n  17 of the 18 "
                      "shortfalls are D_XS at three radii).")

    # ---- does phi order coalitions within a design point? ----------------
    print("\n--- Does phi order coalitions within a design point? ---")
    print("Within a cell the design is fixed and only the sample changes, so "
          "a Spearman\ncorrelation between phi_S - phi_X and the realised "
          "gain of S over X asks whether\nthe replicate on which phi favours "
          "S most is the one on which S does best.\nMagnitude is held fixed "
          "within a cell: pooling it would let between-cell\nvariation in "
          "delta_0 stand in for the replicate-level agreement at issue.\n")
    print("The realised gain is read in AUC and in the Brier score, never in "
          "log loss: by\nthe paper's proposition phi_S - phi_X = L(X) - L(S), "
          "so a log-loss contrast would\nbe a second estimate of the same "
          "functional rather than a check on it. Brier is\nthe proper "
          "scoring rule closest to the loss phi is built on.\n")
    print("Both layers are scored. imp is theta^(c), the layer phi is "
          "defined on; pred is\nthe image-only beta_hat(theta^(c)) the "
          "endpoints are scored on, which a coalition\nreaches only through "
          "the imputed labels. Sign agreement is reported but is the\nwrong "
          "column to read: away from a zero gap the two signs agree whatever "
          "phi does\nwithin a cell. Across an n_T sweep the table separates "
          "the two readings a null\nresult admits: finite-sample noise, or a "
          "property of the decomposition.\n")
    rows = []
    for gvals, cell in frame.groupby(varying + ["scenario", "magnitude"],
                                     dropna=False):
        dphi = (cell["phi_gamma"] - cell["phi_beta"]).values
        rec = dict(zip(varying + ["scenario", "magnitude"], gvals))
        kept = False
        # Every comparison is oriented as "gain of gamma over beta", so a
        # positive dphi should meet a positive gain. Brier is a loss, hence
        # the reversed difference.
        targets = []
        for layer in ("imp", "pred"):
            for metric in ("auc", "brier"):
                cb, cg = f"{metric}_{layer}_beta", f"{metric}_{layer}_gamma"
                if cb not in cell.columns or cg not in cell.columns:
                    continue
                dm = ((cell[cg] - cell[cb]).values if metric == "auc"
                      else (cell[cb] - cell[cg]).values)
                targets.append((f"{metric}_{layer}", dm))
        for name, dmetric in targets:
            ok = np.isfinite(dphi) & np.isfinite(dmetric)
            if ok.sum() < 6:
                continue
            rho, pv = stats.spearmanr(dphi[ok], dmetric[ok])
            rec.update({f"n_{name}": int(ok.sum()),
                        f"spearman_{name}": rho, f"p_{name}": pv,
                        f"sign_agreement_{name}": float(np.mean(
                            np.sign(dphi[ok]) == np.sign(dmetric[ok])))})
            kept = True
        if kept:
            rows.append(rec)
    track, tracking_medians = None, {}
    if rows:
        track = pd.DataFrame(rows)
        print(track.round(4).to_string(index=False))
        track.to_csv(os.path.join(out_dir, "simulation_tracking.csv"),
                     index=False)
        for name in ("auc_imp", "brier_imp", "auc_pred", "brier_pred"):
            col = f"spearman_{name}"
            if col not in track.columns:
                continue
            r = track[col].dropna()
            pv = track[f"p_{name}"].dropna()
            if not len(r):
                continue
            tracking_medians[name] = float(r.abs().median())
            print(f"\n    {name:11s}: median |spearman| = "
                  f"{float(r.abs().median()):.3f}, "
                  f"{int((pv < 0.05).sum())}/{len(pv)} cells at p < 0.05")
        if {"brier_imp", "brier_pred"} <= set(tracking_medians):
            print(f"\n    Brier, imp {tracking_medians['brier_imp']:.3f} vs "
                  f"pred {tracking_medians['brier_pred']:.3f}. Near zero at "
                  "imp, the layer phi is\n    defined on, phi does not order "
                  "coalitions within a design point; a departure\n    at "
                  "pred alone reflects the imputed-label channel rather than "
                  "the\n    decomposition.")

    # ---- the RSC tolerance term at the realised norm --------------------
    thresh = None
    if "eps0_conservatism" in frame.columns:
        print("\n--- The RSC tolerance term at the realised norm ---")
        print("eps0_applied is the schedule log d / n_T: the prefactor of "
              "the tolerance\ncomponent of the sufficient condition, with the "
              "multiplier fixed at one.")
        print("eps0_required is T_RSC = (log d / n_T) ||theta_hat(meta) - "
              "theta_bar||_1^2 over\nthe slopes, the unit-weight proxy for "
              "that component; conservatism is\nT_RSC / eps0_applied.")
        print("A norm stable in n_T means the factor multiplying log d / n_T "
              "is close to\nconstant, which is what the schedule needs; its "
              "distance from one says how far\nthe unit multiplier is from "
              "satisfying the condition. Exact for m_cor only.\n")
        thresh = frame.groupby(varying, dropna=False).agg(
            n=("eps0_applied", "size"),
            eps0_applied=("eps0_applied", "first"),
            norm_median=("theta_meta_minus_truth_l1", "median"),
            norm_q90=("theta_meta_minus_truth_l1",
                      lambda s: float(np.percentile(s.dropna(), 90))),
            eps0_required_median=("eps0_required", "median"),
            conservatism_median=("eps0_conservatism", "median"),
        ).reset_index()
        print(thresh.round(4).to_string(index=False))
        thresh.to_csv(os.path.join(out_dir, "simulation_threshold.csv"),
                      index=False)
        if frame["rho"].nunique() > 1:
            by_rho = frame.groupby("rho")["eps0_conservatism"].median()
            print(f"\n    conservatism by the binary rule rho: "
                  f"{by_rho.round(3).to_dict()}")

    # ---- coalition selection --------------------------------------------
    print("\n--- Coalition selection ---")
    print("Counts of the coalition rule c_hat. Coalitions are named as in the "
          "code:\nnone = empty set, beta = X, gamma = S, both = XS.\n")
    sel = pd.crosstab([frame[k] for k in varying + ["scenario", "magnitude"]],
                      frame["selected"])
    print(sel.to_string())
    sel.to_csv(os.path.join(out_dir, "simulation_selection.csv"))

    # ---- negative-transfer sweep ----------------------------------------
    sweep = None
    if (frame.scenario == "S3").any():
        print("\n--- The negative-transfer axis (S3, D_XS) ---")
        print("Where transfer stops paying and where the binary rule stops "
              "accepting it:\nrho_rate is its acceptance rate, and "
              "meta_minus_target = L(XS) - L(empty) is\nthe margin the "
              "criterion reads.\n")
        s3 = frame[frame.scenario == "S3"].copy()

        # Three policies at every point of the axis, from columns already
        # recorded per replicate: forced_none is Target-Only, forced_both is
        # Corrected-Meta, and gate is the coalition c_hat actually selected,
        # i.e. the paper's Gate column, Gate-Block. Reading them on one axis
        # separates "the gate transferred under negative transfer" from "the
        # gate transferred where transfer still paid", which a pooled rate
        # cannot. brier and ece are losses, auc is a gain; every derived
        # column below is oriented so that positive always means "refusing
        # the source was the right call", which for auc means flipping the
        # difference. Each runs at both layers, so the column stem is
        # metric_layer.
        SWEEP_METRICS = [(f"{m}_{layer}", kind)
                         for layer in ("imp", "pred")
                         for m, kind in (("brier", "loss"), ("ece", "loss"),
                                         ("auc", "gain"))]
        for stem, kind in SWEEP_METRICS:
            cols = [f"{stem}_{c}" for c in COALITIONS]
            if not all(c in s3.columns for c in cols):
                continue
            if s3[cols].isna().all().all():        # layer not fitted
                continue
            s3[f"{stem}_gate"] = [
                row[f"{stem}_{row['selected']}"]
                for _, row in s3.iterrows()
            ]
            # What forcing the source would have cost relative to the
            # target-only fit, positive when refusal helped: it turns positive
            # where Corrected-Meta falls behind Target-Only, the crossing the
            # paper tabulates for each metric and layer.
            s3[f"{stem}_harm_avoided"] = (
                s3[f"{stem}_none"] - s3[f"{stem}_both"] if kind == "gain"
                else s3[f"{stem}_both"] - s3[f"{stem}_none"]
            )

        agg = {
            "delta0_over_budget": ("delta0_l1_over_budget", "mean"),
            "delta0_over_anchor_budget":
                ("delta0_l1_over_anchor_budget", "mean"),
            "L_none": ("L_none", "mean"),
            "L_both": ("L_both", "mean"),
            "rho_rate": ("rho", "mean"),
            "phi_total": ("phi_total", "mean"),
        }
        for stem, _kind in SWEEP_METRICS:
            if f"{stem}_gate" not in s3.columns:
                continue
            agg.update({
                f"{stem}_forced_none": (f"{stem}_none", "mean"),
                f"{stem}_forced_both": (f"{stem}_both", "mean"),
                f"{stem}_gate": (f"{stem}_gate", "mean"),
                f"{stem}_harm_avoided": (f"{stem}_harm_avoided", "mean"),
            })

        sweep = s3.groupby(varying + ["magnitude"], dropna=False).agg(
            **agg).reset_index()
        sweep["meta_minus_target"] = sweep.L_both - sweep.L_none

        # The gap from the gate to the better of the two fixed policies at
        # each magnitude, the envelope a single threshold is measured against.
        for stem, kind in SWEEP_METRICS:
            gcol = f"{stem}_gate"
            if gcol not in sweep.columns:
                continue
            pair = sweep[[f"{stem}_forced_none", f"{stem}_forced_both"]]
            # Signed so that positive is always "the gate did worse than the
            # better of the two fixed policies", whichever direction the
            # metric runs.
            sweep[f"{stem}_gate_regret"] = (
                pair.max(axis=1) - sweep[gcol] if kind == "gain"
                else sweep[gcol] - pair.min(axis=1)
            )

        sweep.to_csv(os.path.join(out_dir, "simulation_sweep.csv"), index=False)
        # The full table is too wide to read with both layers, so the CSV
        # carries everything and the console shows one layer at a time.
        shared = varying + ["magnitude", "delta0_over_budget",
                            "delta0_over_anchor_budget", "rho_rate",
                            "meta_minus_target"]
        for layer in ("imp", "pred"):
            cols = [c for c in shared if c in sweep.columns]
            cols += [c for c in sweep.columns
                     if c.split("_")[1:2] == [layer] and c not in cols]
            if len(cols) == len(shared):
                continue
            print(f"\n    layer = {layer}"
                  + ("  (imputation layer, theta^(c))" if layer == "imp"
                     else "  (deployed layer, image-only beta_hat(theta^(c)))"))
            print(sweep[cols].round(4).to_string(index=False))

        if "brier_pred_gate" in sweep.columns or "brier_imp_gate" in sweep.columns:
            print("\n    *_forced_none is Target-Only, *_forced_both is "
                  "Corrected-Meta (it always\n"
                  "    takes the source), and *_gate follows the coalition "
                  "rule c_hat (Gate-Block).\n"
                  "    *_harm_avoided is what forcing the source would have "
                  "cost; it turns positive\n"
                  "    where Corrected-Meta falls behind Target-Only. Every "
                  "*_harm_avoided and\n"
                  "    *_gate_regret column is signed the same way for losses "
                  "and for auc: positive\n"
                  "    harm_avoided means refusing helped, and positive "
                  "gate_regret means the gate\n"
                  "    did worse than the better fixed policy. Read the axis "
                  "as\n"
                  "    delta0_over_anchor_budget whenever an n_T anchor is "
                  "set.")

    ident = attrib[attrib["n_scorable"] > 0]
    headline = {
        "max_efficiency_gap": float(gap),
        "fista_nonconvergence": dict(_FISTA_NONCONVERGENCE),
        "label_purity_min": (float(attrib["label_purity"].min())
                             if len(attrib) else None),
        "strength_reversal": (reversal or {}).get("verdict"),
        "eps0_conservatism_median": (
            float(frame["eps0_conservatism"].median())
            if "eps0_conservatism" in frame.columns else None),
        "theta_meta_minus_truth_l1_median": (
            float(frame["theta_meta_minus_truth_l1"].median())
            if "theta_meta_minus_truth_l1" in frame.columns else None),
        "threshold_by_cell": (thresh.to_dict("records")
                              if thresh is not None else []),
        "n_grid_cells": int(frame.groupby(varying, dropna=False).ngroups),
        "n_replicates": int(len(frame)),
        # The sweep goes in whole: a single pooled false-transfer rate mixes
        # magnitudes at which transfer still pays with ones at which it does
        # not, so the per-magnitude rows are the reportable form.
        "sweep_by_magnitude": (sweep.to_dict("records")
                               if sweep is not None else []),
        "gate_never_worse_than_best_fixed": {
            stem: (bool((sweep[f"{stem}_gate_regret"] <= 1e-9).all())
                   if sweep is not None
                   and f"{stem}_gate_regret" in sweep.columns else None)
            for stem in (f"{m}_{layer}" for layer in ("imp", "pred")
                         for m in ("brier", "ece", "auc"))},
        "tracking_median_abs_spearman": tracking_medians,
        "n_tl_positive_by_cell": (
            frame.groupby(varying, dropna=False).agg(
                expected=("n_tl_positive_expected", "first"),
                mean=("n_tl_positive", "mean"),
                sd=("n_tl_positive", "std"),
                min=("n_tl_positive", "min"),
                max=("n_tl_positive", "max"),
            ).reset_index().to_dict("records")
            if "n_tl_positive" in frame.columns else []),
        "delta0_calibration_converged_rate": (
            float(frame.loc[frame["scenario"] != "S0",
                            "delta0_calibration_converged"].mean())
            if "delta0_calibration_converged" in frame.columns
            and (frame["scenario"] != "S0").any() else None),
        "delta0_over_anchor_budget_by_cell": (
            frame[frame["scenario"] != "S0"]
            .groupby(varying + ["scenario", "magnitude"], dropna=False)
            ["delta0_l1_over_anchor_budget"].mean().reset_index()
            .to_dict("records")
            if "delta0_l1_over_anchor_budget" in frame.columns else []),
        "prediction_layer_fitted": (
            bool(frame["prediction_layer_fitted"].all())
            if "prediction_layer_fitted" in frame.columns else None),
        "identification_rate_overall": float(ident.identification_rate.mean())
        if len(ident) else None,
        "identification_rate_by_cell": ident[
            varying + ["scenario", "magnitude", "truth_block", "label_purity",
                       "n_scorable", "n_unscorable", "identification_rate"]
        ].to_dict("records") if len(ident) else [],
    }
    with open(os.path.join(out_dir, "simulation_summary.json"), "w") as h:
        json.dump(headline, h, indent=2, default=float)
    return headline


# ======================================================================
# Operating point: where a grid cell sits on the signal axis
# ======================================================================

# Single-block and joint AUCs of the application's ssl (SimCLR) branch on the
# target, the reference --operating-point prints its cells against.
APPLICATION_OPERATING_POINT = {"image": 0.545, "surrogate": 0.611,
                               "joint": 0.630}


def report_operating_point(cells, Z_pool=None, R_pool=None, reps: int = 6):
    """Realised single-block AUCs at the true theta_bar, per grid cell.

    The strength ladder is indexed by the gap between them, surrogate block
    alone minus image block alone, and the gap is not scale-free: it moves
    when p, q, nu or signal_base change. This locates a cell on that axis
    without running the grid. It measures at the true parameter, so it is the
    signal placed in each block rather than what a finite sample recovers.
    Not used in the paper.
    """
    print("=" * 78)
    print("OPERATING POINT")
    print("=" * 78)
    print("Application, ssl branch on the target: image "
          f"{APPLICATION_OPERATING_POINT['image']:.3f}, surrogate "
          f"{APPLICATION_OPERATING_POINT['surrogate']:.3f}, joint "
          f"{APPLICATION_OPERATING_POINT['joint']:.3f}, gap "
          f"{APPLICATION_OPERATING_POINT['surrogate'] - APPLICATION_OPERATING_POINT['image']:+.3f}\n")
    rows = []
    for label, cfg in cells:
        if Z_pool is None:
            Z, R = generate_parametric(
                cfg, np.random.default_rng(cfg.seed + 7_000_000))
        else:
            Z, R = Z_pool, R_pool
        tgt = np.flatnonzero(R == 0)
        idx = tgt[:min(4000, len(tgt))]
        rng = np.random.default_rng(cfg.seed + 400_004)
        acc = []
        for rep in range(reps):
            th, _, _ = make_theta(
                cfg, np.random.default_rng(cfg.seed + 1000 * rep + 100_001))
            th = calibrate_intercept(Z[tgt], th, cfg)
            y = (rng.uniform(size=len(idx))
                 < expit(outcome_logit(Z[idx], th, cfg))).astype(int)
            acc.append(single_block_auc(Z[idx], y, th, cfg))
        rec = {"cell": label, "signal_base": cfg.signal_base,
               "block_strength": cfg.block_strength, "p": cfg.p, "q": cfg.q,
               "nu": cfg.nu,
               "image": float(np.nanmean([a["auc_image_only"] for a in acc])),
               "surrogate": float(np.mean([a["auc_surrogate_only"] for a in acc])),
               "joint": float(np.mean([a["auc_joint"] for a in acc]))}
        rec["gap"] = rec["surrogate"] - rec["image"]
        rec["joint_minus_application"] = (
            rec["joint"] - APPLICATION_OPERATING_POINT["joint"])
        rows.append(rec)
    frame = pd.DataFrame(rows)
    print(frame.round(4).to_string(index=False))
    print("\ngap places the cell on the strength ladder. The simulation does "
          "not match the\napplication on the absolute discrimination either "
          "block affords, as the paper\nstates, so joint_minus_application "
          "is expected to be well above zero at\nsignal_base = 0.85.")
    return frame


# ======================================================================
# Plasmode: real features, regenerated outcome
# ======================================================================

def load_plasmode(prep_dir: str, cfg: Config, image_features: str | None):
    """Assemble Z from the application's prepared cohort (SL.npz, SU.npz and
    TARGET.npz written by its preparation script). The image block is read
    from a cached feature file when one is supplied; otherwise the surrogate
    block is used alone and cfg.p is set to zero, which still exercises the
    gamma side of the decomposition but cannot speak to phi_beta. The
    plasmode arm is not used in the paper."""
    def read(split):
        with np.load(os.path.join(prep_dir, f"{split}.npz"),
                     allow_pickle=False) as f:
            return np.hstack([f["S_text_bow"], f["S_meta"]]).astype(np.float64)

    S = {k: read(k) for k in ("SL", "SU", "TARGET")}
    if image_features and os.path.exists(image_features):
        with np.load(image_features, allow_pickle=False) as f:
            X = {k: f[k].astype(np.float64) for k in ("SL", "SU", "TARGET")}
        Z = {k: np.hstack([X[k], S[k]]) for k in S}
        cfg.p = X["SL"].shape[1]
    else:
        print("  [note] no cached image features; running on the surrogate "
              "block alone. phi_beta is not interpretable in this mode, and "
              "S1 and S3 are unavailable: with no image block there is no "
              "image-side perturbation for them to place.")
        Z = S
        cfg.p = 0

    # Read the metadata width from the cohort rather than trusting the default.
    # q is right either way, since q_text is derived to make it right, but the
    # q_text and q_meta columns recorded per replicate would otherwise be a
    # guess carried over from the parametric arm.
    meta_path = os.path.join(prep_dir, "metadata_feature_names.csv")
    if os.path.exists(meta_path):
        cfg.q_meta = int(len(pd.read_csv(meta_path)))
    else:
        print(f"  [warn] {meta_path} absent; keeping q_meta={cfg.q_meta}. "
              "q is still correct; only the reported text/metadata split is "
              "a guess.")
    cfg.q_text = S["SL"].shape[1] - cfg.q_meta
    if cfg.q_text <= 0:
        raise ValueError(
            f"q_meta={cfg.q_meta} is not smaller than the surrogate width "
            f"{S['SL'].shape[1]}; the text/metadata split cannot be recovered."
        )

    # Real features carry none of the parametric arm's block coupling, so the
    # column exclusion built for it must not be applied here.
    cfg.block_coupling = False

    Zc = np.vstack([Z["SL"], Z["SU"], Z["TARGET"]])
    R = np.r_[np.ones(len(Z["SL"]) + len(Z["SU"]), int),
              np.zeros(len(Z["TARGET"]), int)]
    return StandardScaler().fit_transform(Zc), R


# ======================================================================
# Main
# ======================================================================

EXAMPLES = """
examples, run from the repository root:

  # smoke test, seconds
  python simulation/simulation_shapley_2.py --demo

The paper's seven batches, fifty replicates per scenario-magnitude point:

  # negtransfer: D_empty, and D_XS from 0.5 to 3 radii (400)
  python simulation/simulation_shapley_2.py --scenarios S0,S3 \\
      --magnitudes 0,0.5,0.75,1,1.25,1.5,2,3 --reps 50 \\
      --out-dir results/negtransfer

  # dense: D_dense from 0.5 to 2 radii (200)
  python simulation/simulation_shapley_2.py --scenarios S4 \\
      --magnitudes 0.5,1,1.5,2 --reps 50 \\
      --out-dir results/dense

  # dense_holdout: the held-out correction under D_XS and D_dense (400)
  python simulation/simulation_shapley_2.py --delta-holdout \\
      --scenarios S3,S4 --magnitudes 0.5,1,1.5,2 --reps 50 \\
      --out-dir results/dense_holdout

  # strength: four rungs of the strength ladder, D_X and D_S (1600)
  python simulation/simulation_shapley_2.py \\
      --block-strength beta_strong_2x,balanced,gamma_strong,gamma_strong_2x \\
      --scenarios S1,S2 --magnitudes 0.5,1,1.5,2 --reps 50 \\
      --out-dir results/strength

  # strength_fine: the two finer surrogate rungs, D_X and D_S (800)
  python simulation/simulation_shapley_2.py \\
      --block-strength gamma_soft,gamma_med \\
      --scenarios S1,S2 --magnitudes 0.5,1,1.5,2 --reps 50 \\
      --out-dir results/strength_fine

  # grid: n_T sweep, ||delta_0||_1 held at the reference radius 6.09 (600)
  python simulation/simulation_shapley_2.py \\
      --sweep n_T=100,200,300,400,500,800 --delta0-anchor-nt 200 \\
      --scenarios S1,S2 --magnitudes 1 --reps 50 \\
      --out-dir results/grid

  # oracle: oracle imputation, paired with negtransfer (350)
  python simulation/simulation_shapley_2.py --oracle-imputation \\
      --scenarios S0,S3 --magnitudes 0,0.5,0.75,1,1.25,1.5,2 --reps 50 \\
      --out-dir results/oracle

Not used in the paper:

  # misspecified nuisance models, all four combinations
  python simulation/simulation_shapley_2.py \\
      --dgp m_cor/w_cor,m_mis/w_cor,m_cor/w_mis,m_mis/w_mis \\
      --scenarios S0,S3 --reps 50 --out-dir results/misspecified

  # where grid cells sit on the strength ladder, without running them
  python simulation/simulation_shapley_2.py --operating-point \\
      --block-strength beta_strong_2x,balanced,gamma_soft,gamma_med,gamma_strong,gamma_strong_2x

  # real covariates from the application's prepared cohort, regenerated outcome
  python simulation/simulation_shapley_2.py --arm plasmode \\
      --prep-dir data/prepared_historycombined_canonical_blank0_source1200_bow \\
      --image-features biovil_cached.npz --reps 50 \\
      --out-dir results/plasmode
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["parametric", "plasmode"],
                    default="parametric",
                    help="parametric, as in every run of the paper, or "
                         "plasmode: real covariates from the application's "
                         "prepared cohort with a regenerated outcome (not "
                         "used in the paper)")
    ap.add_argument("--prep-dir", default=None,
                    help="plasmode only: the folder the application's "
                         "preparation script writes (see the example "
                         "below)")
    ap.add_argument("--image-features", default=None,
                    help="plasmode only: npz holding SL/SU/TARGET image "
                         "feature arrays; without it the surrogate block is "
                         "used alone")
    ap.add_argument("--reps", type=int, default=100,
                    help="replicates per scenario-magnitude point (default "
                         "100); the paper uses 50")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="where the raw CSV and the summary tables are "
                         "written (default <repo>/results/simulation_output, "
                         "or $STRIFLE_RESULTS_DIR/simulation_output when that "
                         "is set)")
    ap.add_argument("--sweep", action="append", type=parse_sweep, default=[],
                    metavar="FIELD=v1,v2",
                    help="sweep a Config field; repeatable, cross product. "
                         "The paper's sweep is n_T=100,200,300,400,500,800")
    ap.add_argument("--dgp", default="m_cor/w_cor",
                    help="comma-separated specification variants: "
                         "m_cor/w_cor, m_mis/w_cor, m_cor/w_mis, m_mis/w_mis. "
                         "The paper reports m_cor/w_cor only")
    ap.add_argument("--block-strength", default="balanced",
                    help="comma-separated rungs of the strength ladder: "
                         + ", ".join(BLOCK_STRENGTH_LEVELS)
                         + ". The paper's ladder is beta_strong_2x, balanced, "
                           "gamma_soft, gamma_med, gamma_strong, "
                           "gamma_strong_2x")
    ap.add_argument("--scenarios", default="S0,S1,S2,S3",
                    help="comma-separated subset of S0,S1,S2,S3,S4, the "
                         "paper's D_empty, D_X, D_S, D_XS and D_dense")
    ap.add_argument("--magnitudes", default="0,0.25,0.5,1,2",
                    help="comma-separated magnitudes for the shifted "
                         "scenarios, in multiples of the radius "
                         "sqrt(n_T / log d) (the anchor's radius when "
                         "--delta0-anchor-nt is set) at which the realised "
                         "||delta_0||_1 is set. Magnitude 0 is run once, "
                         "as S0")
    ap.add_argument("--epsilon-0", type=float, default=None, dest="epsilon_0",
                    help="override the dead zone; default is the paper's "
                         "schedule log(d)/n_T, recomputed per cell. No run "
                         "of the paper overrides it: its assessment of "
                         "epsilon0 replays the stored coalition losses "
                         "instead of refitting")
    ap.add_argument("--delta0-anchor-nt", type=int, default=None,
                    dest="delta0_anchor_n_T",
                    help="reference n_T for the radius Delta is scaled "
                         "against; default is each cell's own. The paper's "
                         "n_T sweep sets 200, which holds ||delta_0||_1 at "
                         "6.09 across the sweep instead of letting it grow "
                         "with n_T")
    ap.add_argument("--oracle-imputation", action="store_true",
                    dest="oracle_imputation",
                    help="the paper's oracle imputation: the unlabelled "
                         "target rows get soft labels from the true "
                         "theta_bar instead of each coalition's own fit, the "
                         "rest of the deployed stage unchanged. Paired with "
                         "the reference run it leaves the imputation layer "
                         "bit-identical and isolates the imputed-label "
                         "channel. Reads the truth, so not a runnable "
                         "estimator")
    ap.add_argument("--delta-holdout", action="store_true",
                    dest="delta_holdout",
                    help="the paper's held-out correction: fit the correction "
                         "on target rows held out of the meta fit instead of "
                         "reusing I_{T,L} as Algorithm 1 does. Halves n_T in "
                         "both steps, so compare the fraction of zero "
                         "corrections rather than the estimation error")
    ap.add_argument("--no-delta0-calibration", action="store_true",
                    dest="no_delta0_calibration",
                    help="keep the linearisation delta_0 = -a Delta instead "
                         "of solving for the realised ||delta_0||_1. "
                         "Sensitivity arm, not used in the paper: the "
                         "realised norm then runs up to 28%% above the "
                         "request on the sparse scenarios, and below it at "
                         "three radii")
    ap.add_argument("--skip-pred", action="store_true",
                    help="score only the imputation layer and skip Algorithm "
                         "1's image-only outcome model. Roughly 2.5x faster; "
                         "none of the runs in the paper's table uses it")
    ap.add_argument("--meta-cv-target-only", action="store_true",
                    dest="meta_cv_target_only",
                    help="select theta-tilde's penalty on target rows only. "
                         "The default, as in the paper and the pipeline, "
                         "selects it under the fitting weights")
    ap.add_argument("--operating-point", action="store_true",
                    dest="operating_point",
                    help="print the realised single-block AUCs and their gap "
                         "for every grid cell, then exit. The ladder is "
                         "indexed by that gap, which moves when p, q, nu or "
                         "signal_base change. Not used in the paper")
    ap.add_argument("--self-test", action="store_true",
                    help="run the block-boundary regression test and exit")
    ap.add_argument("--demo", action="store_true",
                    help="tiny configuration for a smoke test")
    args = ap.parse_args(argv)

    _check_theta_of()
    if args.self_test:
        print("block-boundary regression test passed")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    base = Config(epsilon_0=args.epsilon_0,
                  delta0_anchor_n_T=args.delta0_anchor_n_T,
                  meta_cv_target_only=args.meta_cv_target_only,
                  fit_prediction_layer=not args.skip_pred,
                  calibrate_delta0=not args.no_delta0_calibration,
                  oracle_imputation=args.oracle_imputation,
                  delta_holdout=args.delta_holdout)
    reps = args.reps
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    magnitudes = [float(m) for m in args.magnitudes.split(",")]
    dgps = [d.strip() for d in args.dgp.split(",") if d.strip()]
    strengths = [b.strip() for b in args.block_strength.split(",") if b.strip()]

    if args.demo:
        base = Config(p=16, q_text=12, q_meta=4, n_T=200, n_S=600,
                      N_T=1500, N_S=1500, n_valid=200,
                      epsilon_0=args.epsilon_0,
                      delta0_anchor_n_T=args.delta0_anchor_n_T,
                      meta_cv_target_only=args.meta_cv_target_only,
                      fit_prediction_layer=not args.skip_pred,
                      calibrate_delta0=not args.no_delta0_calibration,
                      oracle_imputation=args.oracle_imputation,
                      delta_holdout=args.delta_holdout)
        reps, magnitudes = 4, [0.0, 1.0, 2.0]

    cells = build_grid(base, args.sweep, dgps, strengths)

    print("=" * 78)
    print("SIMULATION: Shapley block attribution under known ground truth")
    print("=" * 78)
    print(f"arm={args.arm}  scenarios={','.join(scenarios)}  reps={reps}  "
          f"grid cells={len(cells)}")
    if args.no_delta0_calibration:
        print("[note] delta_0 calibration OFF: the S3 axis is the requested "
              "magnitude, not\n       the realised one. Read "
              "delta0_l1_over_anchor_budget instead of magnitude.")
    if args.oracle_imputation:
        print("[ABLATION] oracle imputation: TU soft labels come from the "
              "true theta_bar.\n           Not a runnable estimator; a "
              "diagnostic only.")
    if args.delta_holdout:
        print("[ABLATION] held-out correction: the correction is fitted on "
              "target rows the\n           meta fit never saw. Departs from "
              "Algorithm 1 and halves n_T in\n           both steps.")
    print("layers scored: imputation theta^(c)"
          + ("" if args.skip_pred else " + deployed image-only beta")
          + ("   [--skip-pred: the deployed layer is NOT scored]"
             if args.skip_pred else ""))
    points = scenario_grid(scenarios, magnitudes)
    n_points = len(points)
    print(f"scenario points per cell = {n_points}: "
          + ", ".join(f"{s}@{m:g}" for s, m in points))
    if any(float(m) == 0.0 for m in magnitudes) and \
            any(s != "S0" for s in scenarios):
        print("  [note] magnitude 0 collapsed to a single S0 cell; a shifted "
              "scenario at\n         magnitude 0 is S0 by construction and "
              "carries no block label.")
    print(f"total replicates = {len(cells)} x {n_points} x {reps} "
          f"= {len(cells) * n_points * reps}")

    if args.operating_point and args.arm == "parametric":
        report_operating_point(cells)
        return 0

    Z_pool = R_pool = None
    if args.arm == "plasmode":
        if not args.prep_dir:
            print("plasmode requires --prep-dir")
            return 2
        if args.sweep:
            swept = sorted({k for s in args.sweep for k in s})
            frozen = [k for k in swept if k in ("p", "q_text", "q_meta", "nu")]
            live = [k for k in swept if k not in ("p", "q_text", "q_meta", "nu")]
            if frozen:
                print(f"[note] plasmode holds the real covariates fixed, so "
                      f"{frozen} cannot be swept; the loaded dimensions win.")
            if live:
                print(f"[note] {live} are swept normally: they change how many "
                      "rows are drawn\n       from the fixed pool, not the "
                      "pool itself. The n_T sweep in particular\n       is "
                      "fully meaningful here.")
        Z_pool, R_pool = load_plasmode(args.prep_dir, base, args.image_features)
        cells = build_grid(base, args.sweep, dgps, strengths)
        print(f"  loaded {Z_pool.shape[0]} rows, d={Z_pool.shape[1]}")
        if args.operating_point:
            report_operating_point(cells, Z_pool, R_pool)
            return 0

    frames, t0 = [], time.time()
    total = len(cells) * n_points * reps
    state = {"done": 0, "last": t0, "cell": ""}

    # Progress on whichever comes first: 1/60th of the run, or two minutes of
    # wall clock. A pure count throttle can stay silent for long stretches on
    # a grid whose cells differ in cost, long enough that a hang and a slow
    # cell look alike.
    PROGRESS_EVERY_SECONDS = 120.0

    def tick():
        state["done"] += 1
        now = time.time()
        step = max(1, total // 60)
        due = (state["done"] % step == 0
               or now - state["last"] >= PROGRESS_EVERY_SECONDS
               or state["done"] == total)
        if not due:
            return
        state["last"] = now
        el = now - t0
        left = el / state["done"] * (total - state["done"])
        print(f"    {state['done']}/{total}  [{state['cell']}]  "
              f"{el / 60:.0f}m elapsed, ~{left / 60:.0f}m left "
              f"({el / state['done']:.1f}s per replicate)", flush=True)

    for i, (label, cfg) in enumerate(cells, 1):
        state["cell"] = f"cell {i}/{len(cells)}"
        print(f"\n[cell {i}/{len(cells)}] {label}  "
              f"(d={cfg.d}, nT={cfg.n_T}, nS={cfg.n_S}, "
              f"budget={cfg.budget:.2f})")
        f = run_cell(cfg, scenarios, magnitudes, reps, args.arm,
                     Z_pool, R_pool, tick)
        if not f.empty:
            f["cell"] = label
            frames.append(f)

    if not frames:
        print("no replicates completed")
        return 1
    frame = pd.concat(frames, ignore_index=True)
    if _FISTA_NONCONVERGENCE["total"]:
        hit = _FISTA_NONCONVERGENCE["count"]
        tot = _FISTA_NONCONVERGENCE["total"]
        print(f"\n[delta solver] {hit}/{tot} FISTA solves hit the "
              f"{CORRECTION_MAX_ITER}-iteration cap ({hit / tot:.1%}). In "
              "the paper's runs\n               they all sit at the "
              "small-lambda end of the path, and the selected\n"
              "               lambda never falls at its two smallest ratios "
              "(the correction-solve\n               check).")
    path = os.path.join(args.out_dir, f"simulation_{args.arm}_raw.csv")
    frame.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(frame)} replicates)")
    report(frame, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
