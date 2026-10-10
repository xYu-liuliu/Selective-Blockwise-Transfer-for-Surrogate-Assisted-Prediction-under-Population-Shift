"""Negative control of the application: permuted source labels.

What this tests
---------------
Two things could produce the rescues of the coalition rule: transferable
structure in the source, or the mechanics of choosing from four candidates
rather than two. This script separates them by permuting the source labels
within SL, which breaks the source law of Y given Z as completely as the design
allows, while every split and every label-free nuisance quantity stays as in
the real-source run it is paired with.

Permuting rather than resampling keeps the source prevalence, the class counts
and every covariate exactly as they were, so the only thing destroyed is the
association between Z and the source outcome. Nothing in the source is then
worth borrowing, and the ideal behaviour is

    rho_hat = 0      the binary rule of Cai et al. (2025, eq. 9) declines
    c_hat   = none   the coalition rule keeps the target-only fit
    phi_X, phi_S     no positive gain to attribute (phi_beta, phi_gamma here)

Y_TARGET is untouched. TL, VALID and TU keep their real labels, so Target-Only
remains a working estimator and every endpoint is still measured against a
model that actually predicts.

Why the permutation is redrawn every repeat
-------------------------------------------
A single fixed permutation shared by all repeats can by chance induce a
spurious source association, and that one accident would then be replicated in
every repeat. The repeat-to-repeat variation would no longer contain it, so the
observed acceptance rate would be biased and its variance understated.
Redrawing per repeat makes each repeat a fresh draw from the null.

What is reused from the real-source run, and why that is legitimate
-------------------------------------------------------------------
Every artefact whose construction never reads Y_SL is reused verbatim, which
holds the representation fixed and makes this a controlled contrast in Y_SL
alone rather than a second independent experiment:

    SimCLR reduced features         self-supervised on target images, label-free
    BioViL features                 frozen public weights, label-free
    image PCA for those branches    fit on SL+SU+TL+TU features, label-free
    text and metadata standardisers label-free
    patient splits                  derived from seeds and Y_TARGET, unchanged

Supervised (code name no_ssl) is excluded, as in the paper, because its
encoder is fine-tuned on Y_SL: the labels enter the representation before they
enter the estimator, and its cached features are invalid under the null. Set
INCLUDE_NO_SSL = True to refit it anyway; that requires the raw image arrays
and costs roughly as much as that branch of the real-source run.

The density ratios are deliberately NOT reused even though they are also
label-free. Recomputing them is a cheap self-check: the diagnostics written
here must match the real-source run's row for row, and the report flags it if
they do not. A mismatch would mean the cached features are not the ones that
run actually used.

The threshold
-------------
This run fits at STRIFLE_GATE_EPSILON = 0, as the real-source run does, and
stores the four coalition losses of every repeat. With no dead zone, under an
exact null the comparison of two cross-fitted losses is close to a coin flip,
and the coalition rule, which accepts whenever any of three comparisons
clears the bar, accepts more often still. Acceptances at epsilon0 = 0 are a
property of the threshold, not evidence that the rules are uninformative.

The paper compares the real and permuted sources at the applied epsilon0 =
log(d) / n_T. replay_gate_epsilon.py makes that comparison, reading this run's
directory next to the real one (its docstring lists what it computes). The
report below summarises this run at epsilon0 = 0 and, as a diagnostic only,
the smallest grid threshold at which the null acceptance rate falls below a
nominal level. The threshold is not calibrated against this control, since
that would build its false-acceptance rate into epsilon0.

Usage
-----
    python -u run_negative_control_shuffled_source.py
    python -u run_negative_control_shuffled_source.py --report-only

The second form rebuilds every summary from artefacts already on disk.
"""

import os
import sys
import gc
import json
import argparse
import hashlib
import importlib.util

import numpy as np
import pandas as pd

from project_paths import NEGCTRL_SUFFIX


# ============================================================
# Configuration
# ============================================================

# The unmodified pipeline, from this folder. Every estimator is imported from
# it, so the null run and the main run share one implementation and cannot
# drift apart. The result directories are taken from it too (see below).
BASE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_patient_split_nT200_valid200_history_bow.py",
)

# Repeat count. None means "inherit whatever the pipeline is set to", which is
# the safe default: the two arms must run the same number of repeats on the
# same splits for the paired contrast to mean anything, and inheriting removes
# the chance of the two drifting apart. Set an integer only to run a shorter
# pilot, and expect the paired section of the report to cover fewer repeats.
NEG_REPEATS = None  # resolved to P.REPEATS after the import below

# The Supervised encoder (code name no_ssl) is fine-tuned on Y_SL, so its cached
# features are invalid under the null. The paper excludes it; including it
# forces a refit and requires the raw images.
INCLUDE_NO_SSL = False
NEG_BRANCHES = ("ssl", "biovil") + (("no_ssl",) if INCLUDE_NO_SSL else ())

# Offset keeps the permutation stream disjoint from every seed the pipeline
# already uses for splits, folds and solvers.
PERMUTATION_SEED_OFFSET = 7_700_000

# Grid for the null-calibration diagnostic of the report, in nats of
# cross-fitted log loss. The null-calibrated thresholds the paper reports come
# from replay_gate_epsilon.py and its finer grid.
EPSILON_GRID = np.array(
    [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
)
# Nominal null acceptance rates the diagnostic reports a threshold for.
NOMINAL_LEVELS = (0.10, 0.05)


# ============================================================
# Import the pipeline and repoint its output globals
# ============================================================

def load_pipeline(path):
    """Import the pipeline by file path without requiring it on sys.path.

    Importing executes the module's top-level code, which prints the device
    banner and creates the MAIN run's output directories. Both are harmless:
    the directories already exist, and nothing is written to them.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BASE_SCRIPT not found: {path}\n"
            "Point BASE_SCRIPT at the pipeline file on this machine."
        )
    spec = importlib.util.spec_from_file_location("strifle_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strifle_pipeline"] = module
    spec.loader.exec_module(module)
    return module


P = load_pipeline(BASE_SCRIPT)

# The null run reads the main run's artefacts and writes next to them. Both
# directories follow the pipeline's RESULT_DIR, captured here before the
# rebinding below points it at the null run.
MAIN_RESULT_DIR = P.RESULT_DIR
NEG_RESULT_DIR = MAIN_RESULT_DIR + NEGCTRL_SUFFIX

# Inherit the repeat count from the pipeline unless it was overridden above.
# P.REPEATS is captured here, before the rebinding below overwrites it.
MAIN_REPEATS = int(P.REPEATS)
if NEG_REPEATS is None:
    NEG_REPEATS = MAIN_REPEATS
NEG_REPEATS = int(NEG_REPEATS)
if NEG_REPEATS > MAIN_REPEATS:
    print(f"[config] WARNING: {NEG_REPEATS} null repeats requested but the main "
          f"run has {MAIN_REPEATS}. The excess repeats have no counterpart, so "
          "they contribute to the null distribution but not to the paired "
          "contrast against the real run.")

MAIN_FEATURE_DIR = os.path.join(MAIN_RESULT_DIR, "repeat_features")
MAIN_SPLIT_DIR = os.path.join(MAIN_RESULT_DIR, "selected_indices")
MAIN_GATE_CSV = os.path.join(MAIN_RESULT_DIR, "gate_and_shapley_audit.csv")
MAIN_DR_CSV = os.path.join(MAIN_RESULT_DIR, "density_ratio_diagnostics.csv")

# Functions in the pipeline read these at call time, so rebinding them here
# redirects every write into the null run's directory.
P.RESULT_DIR = NEG_RESULT_DIR
P.STANDARDIZER_DIR = os.path.join(NEG_RESULT_DIR, "standardizers")
P.WEIGHT_DIR = os.path.join(NEG_RESULT_DIR, "density_ratio")
P.FEATURE_DIR = os.path.join(NEG_RESULT_DIR, "repeat_features")
P.BACKBONE_DIR = os.path.join(NEG_RESULT_DIR, "backbones")
P.REPEATS = NEG_REPEATS
P.IMAGE_BRANCHES = NEG_BRANCHES
# The image features are reused, so re-saving them would duplicate gigabytes
# for no gain. The permutations and splits are saved separately below.
P.SAVE_FEATURES = INCLUDE_NO_SSL
P.SAVE_BACKBONE = False

PERMUTATION_DIR = os.path.join(NEG_RESULT_DIR, "permutations")
for _d in (NEG_RESULT_DIR, P.STANDARDIZER_DIR, P.WEIGHT_DIR, P.FEATURE_DIR,
           P.BACKBONE_DIR, PERMUTATION_DIR):
    os.makedirs(_d, exist_ok=True)

NT = P.NT
SEED = P.SEED


# ============================================================
# The null: permute the source labels
# ============================================================

def permute_source_labels(Y_SL, seed):
    """Permute Y_SL, preserving class counts exactly.

    A permutation is the right null here rather than an independent Bernoulli
    redraw: it holds the source prevalence at its observed value, so no loss
    change can come from a shift in prevalence, and what moves is the
    association between Z and the outcome. Under a Bernoulli redraw the
    prevalence itself would fluctuate and part of any observed loss change
    would be a calibration artefact rather than a signal artefact.
    """
    Y_SL = np.asarray(Y_SL)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(Y_SL))
    Y_permuted = Y_SL[order]

    n = len(Y_SL)
    n_pos = int(np.sum(Y_SL == 1))
    n_neg = n - n_pos
    n_changed = int(np.sum(Y_permuted != Y_SL))
    # Each position marginally receives a uniform draw from the label multiset,
    # so linearity gives E[changed] = 2 * n_pos * n_neg / n.
    expected_changed = 2.0 * n_pos * n_neg / max(n, 1)

    digest = hashlib.sha256(
        np.ascontiguousarray(Y_permuted.astype(np.int8)).tobytes()
    ).hexdigest()[:16]

    audit = {
        "permutation_seed": int(seed),
        "n_source_rows": int(n),
        "n_source_positive": n_pos,
        "source_prevalence": float(n_pos / max(n, 1)),
        "n_labels_changed": n_changed,
        "frac_labels_changed": float(n_changed / max(n, 1)),
        "expected_frac_changed": float(expected_changed / max(n, 1)),
        "permuted_label_sha16": digest,
    }
    if int(np.sum(Y_permuted == 1)) != n_pos:
        raise AssertionError("Permutation changed the source class counts.")
    return Y_permuted.astype(Y_SL.dtype, copy=False), order, audit


def source_cv_auc(X, y, seed, n_splits=5):
    """Five-fold CV AUC of a source-only linear model.

    This is the assay that the null actually took: on the real labels it should
    sit clearly above 0.5, and on the permuted labels it should sit at 0.5
    within Monte Carlo error. If the permuted value is not near 0.5, the
    permutation did not destroy what it was supposed to destroy and nothing
    downstream is interpretable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    scores, truths = [], []
    for train_idx, val_idx in cv.split(X, y):
        model = LogisticRegression(
            C=P.OUTCOME_LR_C, solver="lbfgs", max_iter=2000, random_state=int(seed)
        ).fit(X[train_idx], y[train_idx])
        scores.append(model.decision_function(X[val_idx]))
        truths.append(y[val_idx])
    scores = np.concatenate(scores)
    truths = np.concatenate(truths)
    if len(np.unique(truths)) < 2:
        return float("nan")
    return float(roc_auc_score(truths, scores))


# ============================================================
# Cached artefacts from the main run
# ============================================================

def cache_path(run_key, name):
    return os.path.join(MAIN_FEATURE_DIR, run_key, f"{name}.npy")


def load_cached(run_key, name):
    path = cache_path(run_key, name)
    return np.load(path) if os.path.exists(path) else None


BRANCH_CACHE_NAMES = {
    "ssl": ("X_SL_simclr", "X_SU_simclr", "X_TARGET_simclr"),
    "biovil": ("X_SL_biovil", "X_SU_biovil", "X_TARGET_biovil"),
}


def required_cache_names(branches):
    names = ["S_SL", "S_SU", "S_TARGET"]
    for branch in branches:
        names.extend(BRANCH_CACHE_NAMES.get(branch, ()))
    return names


def preflight_cache(run_keys, branches):
    """Decide before any heavy work whether the raw images are needed.

    Raw CheXpert arrays are the largest object in the run by far. If every
    reusable artefact is on disk and the Supervised branch (no_ssl) is off,
    they are never touched, which removes both the load time and the peak
    memory.
    """
    names = required_cache_names(branches)
    missing = {}
    for run_key in run_keys:
        absent = [n for n in names if not os.path.exists(cache_path(run_key, n))]
        if absent:
            missing[run_key] = absent

    need_images = bool(missing) or ("no_ssl" in branches)
    print(f"\n[cache] main-run feature directory: {MAIN_FEATURE_DIR}")
    if not missing:
        print(f"[cache] all reusable artefacts present for {len(run_keys)} repeats")
    else:
        print(f"[cache] incomplete for {len(missing)} of {len(run_keys)} repeats; "
              "those repeats will be rebuilt from raw images")
        for run_key, absent in list(missing.items())[:5]:
            print(f"    {run_key}: missing {absent}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    if "no_ssl" in branches:
        print("[cache] no_ssl branch is on: its extractor reads Y_SL, so it is "
              "refit under the null and raw images are required")
    print(f"[cache] raw images needed: {need_images}")
    return missing, need_images


def load_split_light(split_name, text_dim, metadata_dim, need_images):
    """Load a prepared split, skipping the image array when it is not needed.

    NPZ member access is lazy, so not touching X_image_uint8 avoids
    decompressing it entirely. The validations mirror the pipeline's loader for
    everything that is actually read.
    """
    path = os.path.join(P.PREP_DIR, f"{split_name}.npz")
    with np.load(path, allow_pickle=False) as data:
        required = {"S_text_bow", "S_meta", "prepared_row_index"}
        if need_images:
            required.add("X_image_uint8")
        if split_name in {"SL", "TARGET"}:
            required.add("Y")
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing NPZ keys: {missing}")

        text = data["S_text_bow"].astype(np.float32, copy=False)
        metadata = data["S_meta"].astype(np.float32, copy=False)
        prepared_row_index = data["prepared_row_index"].astype(np.int64)
        labels = data["Y"].astype(np.float32) if "Y" in data.files else None

        images = None
        if need_images:
            images = data["X_image_uint8"]
            if images.ndim != 3 or images.shape[1:] != (224, 224):
                raise ValueError(
                    f"{split_name} images must be (N,224,224); got {images.shape}."
                )
            if images.dtype != np.uint8:
                raise ValueError(
                    f"{split_name} images must be uint8; got {images.dtype}."
                )
            images = images.reshape(len(images), -1)

    n_rows = len(text)
    if text.shape != (n_rows, text_dim):
        raise ValueError(f"{split_name} text must be (N,{text_dim}); got {text.shape}.")
    if metadata.shape != (n_rows, metadata_dim):
        raise ValueError(f"{split_name} metadata shape mismatch: {metadata.shape}.")
    if len(prepared_row_index) != n_rows:
        raise ValueError(f"{split_name} prepared_row_index length mismatch.")
    if labels is not None and len(labels) != n_rows:
        raise ValueError(f"{split_name} Y length mismatch.")
    if images is not None and len(images) != n_rows:
        raise ValueError(f"{split_name} image/text row-count mismatch.")
    if not np.isfinite(text).all() or not np.isfinite(metadata).all():
        raise ValueError(f"{split_name} text/metadata contains NaN or infinity.")
    return {"X": images, "text": text, "meta": metadata, "Y": labels,
            "prepared_row_index": prepared_row_index, "n_rows": n_rows}


def verify_split_matches_main(rep, split):
    """Assert the null repeat sits on the main run's patient split.

    The splits are a deterministic function of the seed, the patient ids and
    Y_TARGET, none of which the null modifies, so they must reproduce exactly.
    Checking it makes the pairing in the report a verified fact rather than an
    assumption: any per-repeat contrast against the main run is only meaningful
    if both ran on the same patients.
    """
    path = os.path.join(MAIN_SPLIT_DIR, f"repeat{rep}_nT{NT}_patient_split.npz")
    if not os.path.exists(path):
        print(f"  [split check] no stored main-run split for repeat {rep}; skipped")
        return False
    with np.load(path, allow_pickle=False) as stored:
        for key in ("tl_idx", "valid_idx", "tu_idx", "target_fit_idx"):
            if not np.array_equal(np.asarray(stored[key]), np.asarray(split[key])):
                raise AssertionError(
                    f"repeat {rep}: {key} differs from the main run. The null "
                    "would not be paired with the real run and the contrast "
                    "would be uninterpretable."
                )
    return True


# ============================================================
# The null run
# ============================================================

def run_negative_control():
    P.seed_everything(SEED)
    print("=" * 88)
    print("NEGATIVE CONTROL: STRIFLE with permuted source labels")
    print("Y_SL is permuted independently every repeat; Y_TARGET is untouched")
    print("=" * 88)
    print(f"repeats={NEG_REPEATS} (main run: {MAIN_REPEATS}), nT={NT}, "
          f"nVALID={P.N_VALID}, seed={SEED}")
    print(f"gate epsilon0 at fit time = {P.STRIFLE_GATE_EPSILON} "
          "(the paper's log(d)/nT is applied afterwards by "
          "replay_gate_epsilon.py)")

    with open(os.path.join(P.PREP_DIR, "prepare_config.json"), encoding="utf-8") as h:
        prepare_config = json.load(h)
    if prepare_config.get("target_label") != "Pneumothorax":
        raise ValueError("This control requires target_label='Pneumothorax'.")

    vocabulary = pd.read_csv(os.path.join(P.PREP_DIR, "surrogate_vocabulary.csv"))
    metadata_spec = pd.read_csv(
        os.path.join(P.PREP_DIR, "metadata_feature_names.csv")
    )
    q_text = int(len(vocabulary))
    q_meta = int(len(metadata_spec))
    q_total = q_text + q_meta
    print(f"Surrogate: q_text={q_text}, q_metadata={q_meta}, q={q_total}")
    print(f"Branches under test: {NEG_BRANCHES}")

    run_keys = [f"rep{rep}_nT{NT}" for rep in range(NEG_REPEATS)]
    missing_cache, need_images = preflight_cache(run_keys, NEG_BRANCHES)

    print("\nLoading prepared NPZ files ...")
    sl = load_split_light("SL", q_text, q_meta, need_images)
    su = load_split_light("SU", q_text, q_meta, need_images)
    target = load_split_light("TARGET", q_text, q_meta, need_images)

    Y_SL_true = sl["Y"]
    Y_TARGET = target["Y"]
    X_SL_raw, X_SU_raw, X_TARGET_raw = sl["X"], su["X"], target["X"]
    T_SL_raw, T_SU_raw, T_TARGET_raw = sl["text"], su["text"], target["text"]
    M_SL_raw, M_SU_raw, M_TARGET_raw = sl["meta"], su["meta"], target["meta"]

    audit = pd.read_csv(os.path.join(P.PREP_DIR, "row_audit.csv"))
    ordered = []
    for name, prepared in (("SL", sl), ("SU", su), ("TARGET", target)):
        block = audit[audit["split_group"] == name].reset_index(drop=True)
        if len(block) != prepared["n_rows"]:
            raise ValueError(f"{name} NPZ/audit row-count mismatch.")
        if not np.array_equal(block["prepared_row_index"].to_numpy(np.int64),
                              prepared["prepared_row_index"]):
            raise ValueError(f"{name} NPZ/audit row order mismatch.")
        ordered.append(block)
    audit = pd.concat(ordered, ignore_index=True)
    patient_ids_all = audit["patient_id_parsed"].astype(str).to_numpy()
    patient_ids_target = patient_ids_all[audit["split_group"].to_numpy() == "TARGET"]

    observed_prevalence = float(np.mean(Y_TARGET))
    print(f"Source rows={len(Y_SL_true)}, source prevalence="
          f"{float(np.mean(Y_SL_true)):.4f}")
    print(f"Target prevalence={observed_prevalence:.6f}")

    print("\nRebuilding the main run's splits ...")
    repeat_splits = []
    for rep in range(NEG_REPEATS):
        run_seed = SEED + rep * 1000 + NT
        repeat_splits.append(P.patient_split_tl_valid_tu(
            patient_ids=patient_ids_target, target_labels=Y_TARGET,
            n_tl=NT, n_valid=P.N_VALID,
            max_images_per_patient=P.MAX_IMAGES_PER_PATIENT,
            positive_ratio=observed_prevalence, seed=run_seed,
            min_tl_positive_count=P.MIN_TL_POSITIVES,
            min_tl_negative_count=P.MIN_TL_NEGATIVES,
            min_valid_positive_count=P.MIN_VALID_POSITIVES,
            min_valid_negative_count=P.MIN_VALID_NEGATIVES,
        ))
    print(f"  {NEG_REPEATS} splits rebuilt")

    biovil_features = None
    if "biovil" in NEG_BRANCHES and any(
        "X_SL_biovil" in absent for absent in missing_cache.values()
    ):
        print("\nSome BioViL caches are missing; loading frozen BioViL features ...")
        biovil_features = P.load_or_extract_biovil_features(audit)

    config = {
        "experiment": "negative_control_permuted_source_labels",
        "base_script": BASE_SCRIPT,
        "main_result_dir": MAIN_RESULT_DIR,
        "null": ("Y_SL permuted within the source labelled rows, redrawn "
                 "independently each repeat; class counts and every covariate "
                 "preserved; Y_TARGET untouched"),
        "permutation_seed_rule": "SEED + rep*1000 + NT + PERMUTATION_SEED_OFFSET",
        "expected_behaviour": {
            "rho_hat": "0 at a calibrated epsilon0",
            "c_hat": "none",
            "phi_beta_phi_gamma": "centred on zero",
        },
        "branches": list(NEG_BRANCHES),
        "no_ssl_excluded_reason": (
            "the no-SSL extractor is finetuned on Y_SL, so cached features are "
            "invalid under the null and must be refit"
            if not INCLUDE_NO_SSL else "included and refit"
        ),
        "reused_from_main_run": [
            "SimCLR backbones and reduced features (self-supervised, label-free)",
            "BioViL features (frozen public weights, label-free)",
            "image PCA and scalers for those branches (label-free fits)",
            "text and metadata standardised blocks (label-free)",
            "patient splits (function of seed and Y_TARGET only)",
        ],
        "recomputed": [
            "density ratios (label-free, recomputed as a cache self-check)",
            "theta_T, theta_tilde, delta, the gate and every final beta",
            "all baselines that read Y_SL",
        ],
        "repeats": NEG_REPEATS,
        "nT": NT,
        "nVALID": P.N_VALID,
        "gate_epsilon0_at_fit_time": P.STRIFLE_GATE_EPSILON,
        "epsilon_grid_for_post_hoc_calibration": [float(v) for v in EPSILON_GRID],
    }
    with open(os.path.join(NEG_RESULT_DIR, "config.json"), "w", encoding="utf-8") as h:
        json.dump(config, h, indent=2)

    rows, gate_rows, weight_records, all_results = [], [], [], {}
    permutation_rows = []
    completed = P.load_completed_repeats(rows, gate_rows, weight_records, all_results)

    perm_csv = os.path.join(NEG_RESULT_DIR, "permutation_audit.csv")
    if os.path.exists(perm_csv):
        try:
            permutation_rows.extend(pd.read_csv(perm_csv).to_dict("records"))
        except (OSError, pd.errors.ParserError):
            pass

    for rep in range(NEG_REPEATS):
        if rep in completed:
            print(f"\nRepeat {rep + 1}/{NEG_REPEATS} already on disk, skipping")
            continue

        run_seed = SEED + rep * 1000 + NT
        perm_seed = run_seed + PERMUTATION_SEED_OFFSET
        P.seed_everything(run_seed)
        run_key = f"rep{rep}_nT{NT}"

        print("\n" + "#" * 90)
        print(f"NULL repeat {rep + 1}/{NEG_REPEATS}, nT={NT}, seed={run_seed}, "
              f"permutation seed={perm_seed}")
        print("#" * 90)

        split = repeat_splits[rep]
        verify_split_matches_main(rep, split)
        tl_idx, tu_idx = split["tl_idx"], split["tu_idx"]
        valid_idx, target_fit_idx = split["valid_idx"], split["target_fit_idx"]
        Y_TL, Y_TU, Y_VALID = (Y_TARGET[tl_idx], Y_TARGET[tu_idx],
                               Y_TARGET[valid_idx])

        # ---- the null ----------------------------------------------------
        Y_SL, perm_order, perm_audit = permute_source_labels(Y_SL_true, perm_seed)
        perm_audit.update({"repeat": int(rep), "seed": int(run_seed)})
        np.savez_compressed(
            os.path.join(PERMUTATION_DIR, f"repeat{rep}_source_permutation.npz"),
            permutation=perm_order.astype(np.int32),
            Y_SL_permuted=np.asarray(Y_SL, dtype=np.int8),
            Y_SL_true=np.asarray(Y_SL_true, dtype=np.int8),
            permutation_seed=perm_seed,
        )
        print(f"  [null] permuted {perm_audit['n_labels_changed']} of "
              f"{perm_audit['n_source_rows']} source labels "
              f"({perm_audit['frac_labels_changed']:.1%}; expected "
              f"{perm_audit['expected_frac_changed']:.1%}), prevalence held at "
              f"{perm_audit['source_prevalence']:.4f}")

        split_info = {
            "n_tl_labeled": len(tl_idx),
            "n_tl_patients": len(np.unique(patient_ids_target[tl_idx])),
            "n_tl_positive": int(Y_TL.sum()),
            "n_target_unlabeled_total": len(tu_idx),
            "n_excluded_selected_patient_siblings":
                len(split["excluded_sibling_idx"]),
            "n_valid_images": len(valid_idx),
            "n_valid_positive": int(Y_VALID.sum()),
            "n_tu_images": len(tu_idx),
            "n_tu_positive": int(Y_TU.sum()),
        }
        print(f"TL={len(tl_idx)} images, positives={int(Y_TL.sum())}; "
              f"VALID={len(valid_idx)}, positives={int(Y_VALID.sum())}; "
              f"TU={len(tu_idx)}")
        all_results[run_key] = {"split_info": split_info, "seed": run_seed,
                                "permutation": perm_audit}

        # ---- surrogate block: reuse, or rebuild identically ---------------
        S_SL = load_cached(run_key, "S_SL")
        S_SU = load_cached(run_key, "S_SU")
        S_TARGET = load_cached(run_key, "S_TARGET")
        if S_SL is None or S_SU is None or S_TARGET is None:
            print("  [surrogate] cache miss; restandardising from the prepared blocks")
            T_SL, T_SU, T_TARGET = P.standardize_block(
                reference_arrays=[T_SL_raw, T_SU_raw, T_TARGET_raw[target_fit_idx]],
                transform_arrays=[T_SL_raw, T_SU_raw, T_TARGET_raw],
                tag=f"BoW text {run_key}", file_stem=f"text_bow_scaler_{run_key}",
            )
            M_SL, M_SU, M_TARGET = P.standardize_block(
                reference_arrays=[M_SL_raw, M_SU_raw, M_TARGET_raw[target_fit_idx]],
                transform_arrays=[M_SL_raw, M_SU_raw, M_TARGET_raw],
                tag=f"metadata {run_key}", file_stem=f"metadata_scaler_{run_key}",
            )
            S_SL = np.hstack([T_SL, M_SL]).astype(np.float32)
            S_SU = np.hstack([T_SU, M_SU]).astype(np.float32)
            S_TARGET = np.hstack([T_TARGET, M_TARGET]).astype(np.float32)
        else:
            print(f"  [surrogate] reused from main run, q={S_SL.shape[1]}")
        if S_SL.shape[1] != q_total:
            raise AssertionError(f"Surrogate dimension {S_SL.shape[1]} != {q_total}.")

        # ---- image blocks -------------------------------------------------
        branch_inputs = {}
        for branch in NEG_BRANCHES:
            if branch == "no_ssl":
                continue
            names = BRANCH_CACHE_NAMES[branch]
            blocks = [load_cached(run_key, n) for n in names]
            if all(b is not None for b in blocks):
                print(f"  [{branch}] reused from main run, p={blocks[0].shape[1]}")
                branch_inputs[branch] = tuple(blocks)
                continue

            if branch == "ssl":
                print(f"  [{branch}] cache miss; retraining SimCLR "
                      "(self-supervised, so the null does not affect it)")
                backbone, ssl_info = P.train_simclr(
                    X_TARGET_raw[target_fit_idx], seed=run_seed,
                    n_epochs=P.SIMCLR_EPOCHS, batch_size=P.SIMCLR_BATCH_SIZE,
                )
                all_results[run_key]["simclr"] = ssl_info
                raw_sl = P.extract_features(X_SL_raw, backbone, tag=f"SSL SL {run_key}")
                raw_su = P.extract_features(X_SU_raw, backbone, tag=f"SSL SU {run_key}")
                raw_tg = P.extract_features(X_TARGET_raw, backbone,
                                            tag=f"SSL TARGET {run_key}")
                P.free_memory(backbone)
            else:
                print(f"  [{branch}] cache miss; using frozen BioViL features")
                if biovil_features is None:
                    biovil_features = P.load_or_extract_biovil_features(audit)
                raw_sl, raw_su, raw_tg = biovil_features

            branch_inputs[branch] = P.reduce_image_representation(
                raw_sl, raw_su, raw_tg, target_fit_idx,
                tag=f"{branch} {run_key}", file_stem=f"{branch}_image_pca_{run_key}",
            )
            del raw_sl, raw_su, raw_tg
            gc.collect()

        # ---- Supervised (no_ssl): refit, its encoder reads Y_SL ----------
        if "no_ssl" in NEG_BRANCHES:
            print("  [no_ssl] refitting the extractor on PERMUTED source labels")
            X_TL_raw = np.asarray(X_TARGET_raw[tl_idx])
            w_pixel, diag_pixel = P.compute_pixel_density_ratio_image_only(
                X_SU_raw, X_TARGET_raw[tu_idx], X_SL_raw, seed=run_seed
            )
            weight_records.append({"repeat": rep, "branch": "no_ssl_pixel",
                                   "kind": "pixel", **diag_pixel})
            predictor, extractor = P.joint_finetune_cnn_dr(
                X_SL_raw, Y_SL, w_pixel, X_TL_raw, Y_TL, nT=NT, seed=run_seed
            )
            for eval_name, idx in (("valid", valid_idx), ("tu_transductive", tu_idx)):
                p = P.predict_proba_densenet(
                    np.asarray(X_TARGET_raw[idx]), predictor,
                    tag=f"noSSL {eval_name} {run_key}",
                )
                metrics = P.evaluate_predictions(
                    Y_TARGET[idx], p, patient_ids_target[idx]
                )
                P.append_metric_rows(rows, rep, run_seed, "cnn_direct", "dr_finetune",
                                     {f"dr_finetune|{eval_name}": metrics},
                                     split_info, q_text, q_meta)
            P.free_memory(predictor)

            ns_sl = P.extract_features(X_SL_raw, extractor, tag=f"noSSL SL {run_key}")
            ns_su = P.extract_features(X_SU_raw, extractor, tag=f"noSSL SU {run_key}")
            ns_tg = P.extract_features(X_TARGET_raw, extractor, tag=f"noSSL TG {run_key}")
            P.free_memory(extractor)
            branch_inputs["no_ssl"] = P.reduce_image_representation(
                ns_sl, ns_su, ns_tg, target_fit_idx,
                tag=f"noSSL CNN {run_key}", file_stem=f"no_ssl_image_pca_{run_key}",
            )
            del ns_sl, ns_su, ns_tg, X_TL_raw, w_pixel
            gc.collect()

        # ---- did the permutation actually destroy the source signal? ------
        for branch in NEG_BRANCHES:
            X_sl_b = branch_inputs[branch][0]
            auc_null = source_cv_auc(X_sl_b, Y_SL, run_seed)
            auc_true = source_cv_auc(X_sl_b, Y_SL_true, run_seed)
            print(f"  [null check {branch}] source-only 5-fold CV AUC: "
                  f"real labels={auc_true:.4f}, permuted={auc_null:.4f}")
            if np.isfinite(auc_null) and abs(auc_null - 0.5) > 0.08:
                print("    [WARN] the permuted source still carries apparent "
                      "signal well away from 0.5; treat this repeat's null "
                      "as suspect")
            permutation_rows.append({
                **perm_audit, "branch": branch,
                "source_cv_auc_permuted": auc_null,
                "source_cv_auc_real": auc_true,
            })

        # ---- the estimators, under the null -------------------------------
        for branch in NEG_BRANCHES:
            X_sl_b, X_su_b, X_tg_b = branch_inputs[branch]
            n_gate_before = len(gate_rows)
            strifle, baselines, _ = P.run_branch(
                branch, X_sl_b, X_su_b, X_tg_b, S_SL, S_SU, S_TARGET,
                Y_SL, Y_TARGET, split, patient_ids_target, run_seed, rep,
                split_info, rows, gate_rows, weight_records, q_text, q_meta,
            )
            # Stamp the null's provenance onto the rows run_branch just wrote,
            # so the gate audit is self-describing without a join.
            for record in gate_rows[n_gate_before:]:
                record.update({
                    "arm": "negative_control_permuted_source",
                    "permutation_seed": perm_audit["permutation_seed"],
                    "frac_labels_changed": perm_audit["frac_labels_changed"],
                    "permuted_label_sha16": perm_audit["permuted_label_sha16"],
                })
            all_results[run_key][branch] = {
                "metrics": strifle["metrics"],
                "baseline_metrics": baselines["metrics"],
                "phi_beta": strifle["gate"]["phi_beta"],
                "phi_gamma": strifle["gate"]["phi_gamma"],
                "rho": strifle["rho"],
                "selected_endpoint_shapley": strifle["selected_endpoint_shapley"],
            }
            P.free_memory(strifle["predictions"], baselines["predictions"])

        # ---- persist after every repeat -----------------------------------
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(NEG_RESULT_DIR, "raw_results.csv"), index=False)
        P.summarize_raw(frame).to_csv(
            os.path.join(NEG_RESULT_DIR, "summary_raw.csv"), index=False
        )
        paired = P.paired_differences(frame)
        paired.to_csv(os.path.join(NEG_RESULT_DIR, "paired_differences.csv"),
                      index=False)
        P.summarize_paired(paired).to_csv(
            os.path.join(NEG_RESULT_DIR, "summary_paired.csv"), index=False
        )
        pd.DataFrame(gate_rows).to_csv(
            os.path.join(NEG_RESULT_DIR, "gate_and_shapley_audit.csv"), index=False
        )
        pd.DataFrame(weight_records).to_csv(
            os.path.join(NEG_RESULT_DIR, "density_ratio_diagnostics.csv"), index=False
        )
        pd.DataFrame(permutation_rows).to_csv(perm_csv, index=False)
        with open(os.path.join(NEG_RESULT_DIR, "results.json"), "w",
                  encoding="utf-8") as h:
            json.dump(all_results, h, indent=2, default=float)

        gate_view = pd.DataFrame(gate_rows)
        print("\n  Null gate behaviour so far:")
        print(gate_view.groupby("branch").agg(
            n=("rho", "size"), rho_rate=("rho", "mean"),
            c_none_rate=("selected_endpoint_shapley",
                         lambda s: float((s == "none").mean())),
            phi_beta=("phi_beta", "mean"), phi_gamma=("phi_gamma", "mean"),
        ).to_string())

        P.free_memory(*[b for blocks in branch_inputs.values() for b in blocks],
                      S_SL, S_SU, S_TARGET)
        branch_inputs.clear()

    print("\nNull run complete. Building the report ...")
    report_negative_control()


# ============================================================
# Report and threshold calibration
# ============================================================

def recompute_decision(loss_none, loss_beta, loss_gamma, loss_both, epsilon0):
    """Replay both decision rules at an arbitrary epsilon0.

    Every coalition loss is stored, so the binary rule and the coalition rule
    are pure functions of four numbers and a threshold. Nothing has to be
    refit to ask what either rule would have decided at a different margin.
    """
    losses = {"none": float(loss_none), "beta": float(loss_beta),
              "gamma": float(loss_gamma), "both": float(loss_both)}
    cutoff = losses["none"] - float(epsilon0)
    rho = int(losses["both"] <= cutoff)
    eligible = {c: v for c, v in losses.items() if c == "none" or v <= cutoff}
    return rho, min(eligible, key=eligible.get)


def calibrate_epsilon(gate_frame, levels=NOMINAL_LEVELS, grid=EPSILON_GRID):
    """Smallest grid epsilon0 whose null acceptance rate is at or below a level.

    A diagnostic: the paper does not choose epsilon0 this way (see the module
    docstring). The two rules are reported separately. The binary rule makes
    one comparison; the coalition rule accepts whenever the binary rule does
    and also when a single block clears the bar, so its null acceptance rate
    is at least as high at every epsilon0 and its threshold is never smaller.
    """
    records, chosen = [], []
    for branch, group in gate_frame.groupby("branch"):
        for epsilon0 in grid:
            decisions = [
                recompute_decision(r["loss_none"], r["loss_beta"],
                                   r["loss_gamma"], r["loss_both"], epsilon0)
                for _, r in group.iterrows()
            ]
            rho_rate = float(np.mean([d[0] for d in decisions]))
            c_none = float(np.mean([d[1] == "none" for d in decisions]))
            records.append({
                "branch": branch, "epsilon0": float(epsilon0),
                "n_repeats": int(len(group)),
                "null_rho_rate": rho_rate,
                "null_false_transfer_rate_gate": rho_rate,
                "null_c_none_rate": c_none,
                "null_false_selection_rate_shapley": 1.0 - c_none,
            })
        branch_records = [r for r in records if r["branch"] == branch]
        for level in levels:
            gate_ok = [r for r in branch_records
                       if r["null_false_transfer_rate_gate"] <= level]
            shap_ok = [r for r in branch_records
                       if r["null_false_selection_rate_shapley"] <= level]
            chosen.append({
                "branch": branch, "nominal_level": float(level),
                "epsilon0_for_gate": (min(r["epsilon0"] for r in gate_ok)
                                      if gate_ok else np.nan),
                "epsilon0_for_shapley_selection": (
                    min(r["epsilon0"] for r in shap_ok) if shap_ok else np.nan),
            })
    return pd.DataFrame(records), pd.DataFrame(chosen)


def report_negative_control():
    gate_path = os.path.join(NEG_RESULT_DIR, "gate_and_shapley_audit.csv")
    if not os.path.exists(gate_path):
        print(f"No gate audit at {gate_path}; nothing to report.")
        return
    gate = pd.read_csv(gate_path)
    if gate.empty:
        print("Gate audit is empty; nothing to report.")
        return

    print("\n" + "=" * 88)
    print("NEGATIVE CONTROL REPORT")
    print("=" * 88)

    # ---- 1. did the null take? ------------------------------------------
    perm_path = os.path.join(NEG_RESULT_DIR, "permutation_audit.csv")
    if os.path.exists(perm_path):
        perm = pd.read_csv(perm_path)
        check = perm.groupby("branch").agg(
            n=("source_cv_auc_permuted", "size"),
            cv_auc_real=("source_cv_auc_real", "mean"),
            cv_auc_permuted=("source_cv_auc_permuted", "mean"),
            cv_auc_permuted_sd=("source_cv_auc_permuted", "std"),
            frac_labels_changed=("frac_labels_changed", "mean"),
        ).reset_index()
        check.to_csv(os.path.join(NEG_RESULT_DIR, "negctrl_null_validity.csv"),
                     index=False)
        print("\n[1] Did the permutation destroy the source signal?")
        print(check.to_string(index=False))
        print("    The permuted CV AUC should sit at 0.5; the real one above it. "
              "If they are close, the source never carried signal in the first "
              "place and the control has nothing to detect.")

    # ---- 2. gate behaviour at the fitted epsilon0 ------------------------
    summary = gate.groupby("branch").agg(
        n_repeats=("rho", "size"),
        rho_rate=("rho", "mean"),
        c_none_rate=("selected_endpoint_shapley",
                     lambda s: float((s == "none").mean())),
        phi_beta_mean=("phi_beta", "mean"), phi_beta_sd=("phi_beta", "std"),
        phi_gamma_mean=("phi_gamma", "mean"), phi_gamma_sd=("phi_gamma", "std"),
        phi_total_mean=("phi_total", "mean"),
        margin_mean=("transfer_margin", "mean"),
        margin_sd=("transfer_margin", "std"),
        margin_q90=("transfer_margin", lambda s: float(np.percentile(s, 90))),
        margin_max=("transfer_margin", "max"),
    ).reset_index()
    summary.to_csv(os.path.join(NEG_RESULT_DIR, "negctrl_gate_summary.csv"),
                   index=False)
    print(f"\n[2] Null gate behaviour at the fitted epsilon0 = "
          f"{P.STRIFLE_GATE_EPSILON}:")
    print(summary.to_string(index=False))

    coalition = (gate.groupby(["branch", "selected_endpoint_shapley"]).size()
                 .rename("n_repeats").reset_index())
    coalition.to_csv(
        os.path.join(NEG_RESULT_DIR, "negctrl_coalition_counts.csv"), index=False
    )
    print("\n    Selected coalition under the null:")
    print(coalition.to_string(index=False))

    # A one-sided exact test that the null rho rate is below one half. Not
    # below zero: at epsilon0 = 0 the rule has no dead zone, so one half is the
    # honest reference for "the binary rule does better than a coin flip".
    try:
        from scipy import stats
        print("\n    Is the null rho rate below 0.5 (binomial, one-sided)?")
        for _, row in summary.iterrows():
            k = int(round(row["rho_rate"] * row["n_repeats"]))
            n = int(row["n_repeats"])
            p_value = stats.binomtest(k, n, 0.5, alternative="less").pvalue
            print(f"      {row['branch']:>8}: {k}/{n} transfers, p={p_value:.4f}")
    except Exception as exc:
        print(f"    [binomial test skipped: {exc}]")

    # ---- 3. threshold calibration ---------------------------------------
    curve, chosen = calibrate_epsilon(gate)
    curve.to_csv(os.path.join(NEG_RESULT_DIR, "negctrl_epsilon_curve.csv"),
                 index=False)
    chosen.to_csv(os.path.join(NEG_RESULT_DIR, "negctrl_epsilon_calibrated.csv"),
                  index=False)
    print("\n[3] Null decision rates as a function of epsilon0:")
    for branch, group in curve.groupby("branch"):
        print(f"\n    {branch}:")
        print(group[["epsilon0", "null_rho_rate", "null_c_none_rate"]]
              .to_string(index=False))
    print("\n    Smallest epsilon0 reaching each nominal null rate:")
    print(chosen.to_string(index=False))
    print("    The coalition rule accepts whenever the binary rule does, and also "
          "when a single block clears the bar, so its threshold is never below "
          "the binary rule's. A diagnostic only: the reported threshold is "
          "log(d)/nT.")

    # ---- 4. contrast with the real run ----------------------------------
    if os.path.exists(MAIN_GATE_CSV):
        main = pd.read_csv(MAIN_GATE_CSV)
        merged = main.merge(
            gate, on=["repeat", "branch"], suffixes=("_real", "_null")
        )
        if not merged.empty:
            print(f"\n[4] Paired contrast against the real run "
                  f"({len(merged)} paired branch-repeats):")
            contrast = []
            for branch, group in merged.groupby("branch"):
                record = {
                    "branch": branch, "n_paired": int(len(group)),
                    "rho_rate_real": float(group["rho_real"].mean()),
                    "rho_rate_null": float(group["rho_null"].mean()),
                    "margin_real": float(group["transfer_margin_real"].mean()),
                    "margin_null": float(group["transfer_margin_null"].mean()),
                    "phi_gamma_real": float(group["phi_gamma_real"].mean()),
                    "phi_gamma_null": float(group["phi_gamma_null"].mean()),
                    "phi_beta_real": float(group["phi_beta_real"].mean()),
                    "phi_beta_null": float(group["phi_beta_null"].mean()),
                }
                try:
                    from scipy import stats
                    differences = (group["transfer_margin_real"]
                                   - group["transfer_margin_null"]).dropna()
                    if len(differences) >= 3 and differences.std(ddof=1) > 0:
                        record["margin_wilcoxon_p"] = float(
                            stats.wilcoxon(differences,
                                           alternative="greater").pvalue
                        )
                except Exception:
                    pass
                contrast.append(record)
            contrast = pd.DataFrame(contrast)
            contrast.to_csv(
                os.path.join(NEG_RESULT_DIR, "negctrl_vs_main.csv"), index=False
            )
            print(contrast.to_string(index=False))
            print("    The splits are identical repeat by repeat, so this is a "
                  "paired comparison and the Wilcoxon test is one-sided for "
                  "the real margin exceeding the null margin.")

        # The density ratios never read Y_SL, so identical inputs must give
        # identical diagnostics. This is the check that the reused features
        # really are the ones the main run used.
        if os.path.exists(MAIN_DR_CSV):
            dr_null = pd.read_csv(
                os.path.join(NEG_RESULT_DIR, "density_ratio_diagnostics.csv")
            )
            dr_main = pd.read_csv(MAIN_DR_CSV)
            keys = ["repeat", "branch", "kind"]
            if all(k in dr_null.columns for k in keys) and \
                    all(k in dr_main.columns for k in keys):
                joined = dr_main.merge(dr_null, on=keys, suffixes=("_real", "_null"))
                if not joined.empty and "domain_auc_real" in joined.columns:
                    gap = (joined["domain_auc_real"]
                           - joined["domain_auc_null"]).abs().max()
                    print(f"\n    Cache self-check: largest density-ratio domain "
                          f"AUC discrepancy = {gap:.2e}")
                    if gap > 1e-8:
                        print("      [WARN] the density ratios differ between the "
                              "two arms. They are label-free, so identical inputs "
                              "must give identical output: the reused features are "
                              "NOT the ones the main run used.")

    # ---- 5. discrimination under the null -------------------------------
    paired_path = os.path.join(NEG_RESULT_DIR, "summary_paired.csv")
    if os.path.exists(paired_path):
        paired = pd.read_csv(paired_path)
        view = paired[paired["evaluation_split"] == "valid"]
        columns = [c for c in ["branch", "endpoint", "n_repeats",
                               "delta_auc_mean", "delta_auc_ci_low",
                               "delta_auc_ci_high", "delta_auc_wilcoxon_p"]
                   if c in view.columns]
        if not view.empty:
            print("\n[5] Paired AUC difference against target-only, VALID split:")
            print(view[columns].to_string(index=False))
            print("    Under the null every transferring endpoint should sit at "
                  "or below target-only. A confidently positive delta here would "
                  "mean the estimator extracts apparent gain from labels that "
                  "carry no information, which is a leakage finding, not a "
                  "transfer finding.")

    print(f"\nArtefacts written to: {NEG_RESULT_DIR}")
    print("=" * 88)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Negative control of the application: permuted source labels."
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Rebuild every summary from artefacts already on disk.",
    )
    args = parser.parse_args()

    if args.report_only:
        report_negative_control()
    else:
        run_negative_control()
