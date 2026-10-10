"""Where each branch sits on the paper's block-strength ladder.

Why this exists
---------------
phi_X and phi_S say what borrowing each block from the source fit is worth,
but their ordering also tracks how strong each block is to begin with: the gap
phi_S - phi_X sums a term set by the strength imbalance between the blocks and
a term carrying the defect delta_0. The simulation's strength ladder is indexed
by the gap in held-out AUC between a model on the surrogate block alone and
one on the image block alone. Along it, a defect in the stronger block is
rarely identified and one in the weaker block almost always, so the two
identification rates sum highest at the balanced rung, and the ordering of
phi across blocks is interpretable only where the blocks are comparably
strong. This script measures that gap in the application, so each branch and
arm can be placed on the ladder (the paper's supplementary tables on locating
the branches on the strength ladder).

What is measured
----------------
Per repeat, with the pipeline's own splits, saved representations and L1
logistic tuning, every model is scored on VALID:

  1. Single-block and joint discrimination: L1 logistic fits on X alone, on S
     alone and on Z = (X, S). The gap is the AUC on S minus the AUC on X,
     summarised over repeats by its mean and a paired 95% t interval; this is
     the quantity the ladder is indexed by.

  2. Incremental value: what S adds on top of X, and what X adds on top of S.

  3. Coefficient mass: the L1 norm and support size of each block inside the
     joint fit, which says where the penalty is spending its budget.

  4. A block-inclusion Shapley decomposition, using the same four-coalition
     lattice as the estimator but with coalitions over which block is
     *present* rather than which block is *borrowed*. It is in the same units
     as phi_X and phi_S (nats of log loss) and separates "this block carries
     signal" from "this block's source coefficients transfer".

Every quantity is computed twice, from fits on TL and from fits on SL, both
scored on the same VALID rows. The TL row is the target-only half of every
coalition. The SL row stands in for the meta fit, which is estimated on SL and
TL with six source rows to every target row at n_T = 200; both rows have to be
balanced for the ordering of phi_X and phi_S to be read. At n_T = 400 the
ratio is three to one, TL has enough say in the meta fit that the SL row no
longer stands in for it, and the paper reports that arm's placement without
reading it. The S-only AUC does not vary by branch, since S does not depend on
the image encoder.

Usage
-----
    python block_strength_diagnostic.py                  # every repeat of the main run
    python block_strength_diagnostic.py --repeats 5      # the first five only
    python block_strength_diagnostic.py --branches biovil,ssl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score

# This script calls the pipeline's tuned Lasso several hundred times. Its
# convergence and deprecation chatter would bury the report, and neither
# affects which penalty is selected.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# The pipeline in this folder, wherever the script is launched from.
DEFAULT_PIPELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_patient_split_nT200_valid200_history_bow.py",
)


def load_pipeline(path: str):
    """Import the pipeline by path so its constants, splitting rule, PCA and
    Lasso tuning are reused verbatim rather than reimplemented. It has a
    __main__ guard, so importing runs no experiment."""
    if not os.path.exists(path):
        raise SystemExit(f"pipeline not found: {path}\nPass --pipeline PATH.")
    spec = importlib.util.spec_from_file_location("strifle_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strifle_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Measurement on one design matrix
# ======================================================================

def _auc(y, s):
    y = np.asarray(y, int)
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def fit_and_score(P, X_tr, y_tr, X_ev, y_ev, seed, label):
    """Fit the pipeline's tuned L1 logistic and score it on held-out rows."""
    fit = P.fit_l1_logistic_cv(X_tr, y_tr, seed, label=label)
    intercept, coef = _unpack(fit)
    eta = intercept + X_ev @ coef
    return {
        "auc": _auc(y_ev, eta),
        "logloss": log_loss(np.asarray(y_ev, int), 1 / (1 + np.exp(-eta)),
                            labels=[0, 1]),
        "coef": coef, "intercept": float(intercept),
    }


def _unpack(fit):
    """fit_l1_logistic_cv returns either an estimator or a tuple containing
    one; accept both so this keeps working if the pipeline's return signature
    is adjusted."""
    obj = fit
    if isinstance(fit, (tuple, list)):
        for item in fit:
            if hasattr(item, "coef_"):
                obj = item
                break
        else:
            obj = fit[0]
    if hasattr(obj, "coef_"):
        return float(np.ravel(obj.intercept_)[0]), np.ravel(obj.coef_)
    if isinstance(obj, dict):
        return float(obj.get("intercept", 0.0)), np.ravel(obj["coef"])
    raise TypeError(f"cannot unpack fit result of type {type(fit)}")


def block_inclusion_shapley(P, X_tr, S_tr, y_tr, X_ev, S_ev, y_ev, seed):
    """Shapley over block *presence*, on the same four-coalition lattice the
    estimator uses for block *borrowing*.

    The value function is the held-out log-loss reduction relative to an
    intercept-only model, which mirrors v(c) = L(none) - L(c) with the
    target-only fit replaced by the null model. The two decompositions are
    therefore on the same scale and can be read side by side.
    """
    y_tr = np.asarray(y_tr, int)
    y_ev = np.asarray(y_ev, int)
    p_hat = float(np.clip(y_tr.mean(), 1e-6, 1 - 1e-6))
    L_null = log_loss(y_ev, np.full(len(y_ev), p_hat), labels=[0, 1])

    designs = {
        "none": (None, None),
        "beta": (X_tr, X_ev),
        "gamma": (S_tr, S_ev),
        "both": (np.hstack([X_tr, S_tr]), np.hstack([X_ev, S_ev])),
    }
    L = {}
    for name, (tr, ev) in designs.items():
        if tr is None:
            L[name] = L_null
            continue
        L[name] = fit_and_score(P, tr, y_tr, ev, y_ev, seed,
                                f"strength_{name}")["logloss"]
    v = {c: L["none"] - L[c] for c in L}
    phi_b = 0.5 * ((v["beta"] - v["none"]) + (v["both"] - v["gamma"]))
    phi_g = 0.5 * ((v["gamma"] - v["none"]) + (v["both"] - v["beta"]))
    return {"strength_phi_beta": float(phi_b),
            "strength_phi_gamma": float(phi_g),
            "strength_v_both": float(v["both"]),
            "strength_efficiency_gap": float(abs(phi_b + phi_g - v["both"]))}


def measure(P, X_tr, S_tr, y_tr, X_ev, S_ev, y_ev, seed, population):
    """All block-strength quantities for one training population."""
    Z_tr = np.hstack([X_tr, S_tr])
    Z_ev = np.hstack([X_ev, S_ev])
    p = X_tr.shape[1]

    only_x = fit_and_score(P, X_tr, y_tr, X_ev, y_ev, seed, "strength_x")
    only_s = fit_and_score(P, S_tr, y_tr, S_ev, y_ev, seed, "strength_s")
    joint = fit_and_score(P, Z_tr, y_tr, Z_ev, y_ev, seed, "strength_xs")

    beta, gamma = joint["coef"][:p], joint["coef"][p:]
    l1_b, l1_g = float(np.abs(beta).sum()), float(np.abs(gamma).sum())
    row = {
        "population": population,
        "p": int(p), "q": int(S_tr.shape[1]),
        "auc_image_only": only_x["auc"],
        "auc_surrogate_only": only_s["auc"],
        "auc_joint": joint["auc"],
        # What each block adds on top of the other, in AUC.
        "increment_of_surrogate": joint["auc"] - only_x["auc"],
        "increment_of_image": joint["auc"] - only_s["auc"],
        "logloss_image_only": only_x["logloss"],
        "logloss_surrogate_only": only_s["logloss"],
        "logloss_joint": joint["logloss"],
        "l1_beta": l1_b, "l1_gamma": l1_g,
        "l1_gamma_share": l1_g / max(l1_b + l1_g, 1e-12),
        "nnz_beta": int((beta != 0).sum()), "nnz_gamma": int((gamma != 0).sum()),
    }
    row.update(block_inclusion_shapley(P, X_tr, S_tr, y_tr,
                                       X_ev, S_ev, y_ev, seed))
    # Share of the block-inclusion total held by the surrogate. Because the
    # two values sum to v(both) rather than to a positive constant, this
    # leaves [0, 1] whenever a block contributes negatively: a block that
    # actively hurts once the other is present pushes the other above one.
    # That is informative and is reported unclipped. When the total is itself
    # near zero neither block carries signal and the ratio is meaningless, so
    # it is withheld rather than allowed to explode.
    total = row["strength_phi_beta"] + row["strength_phi_gamma"]
    row["strength_gamma_share"] = (row["strength_phi_gamma"] / total
                                   if abs(total) > 1e-4 else np.nan)
    return row


# ======================================================================
# Per-repeat driver
# ======================================================================

def load_cached_repeat(feature_dir: str, rep: int, nT: int, branch: str):
    """Read one repeat's saved representation.

    The pipeline writes each branch's post-PCA features and the standardised
    surrogate block to RESULT_DIR/repeat_features/rep{N}_nT{NT}/ when
    SAVE_FEATURES is on, so the branches that train a network per repeat need
    not be retrained here. Reading the saved arrays also guarantees the
    diagnostic describes the exact representation the reported results came
    from rather than a re-derivation of it. Nothing is rebuilt here: a missing
    repeat raises FileNotFoundError, which run() reports as no cache.
    """
    d = os.path.join(feature_dir, f"rep{rep}_nT{nT}")
    if not os.path.isdir(d):
        raise FileNotFoundError(d)

    def get(name):
        path = os.path.join(d, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path).astype(np.float64)

    # The pipeline names the SimCLR arrays after the method rather than after
    # the branch key, so 'ssl' is stored as 'simclr'. The other two match.
    file_branch = {"ssl": "simclr"}.get(branch, branch)
    X = tuple(get(f"X_{s}_{file_branch}") for s in ("SL", "SU", "TARGET"))
    S = tuple(get(f"S_{s}") for s in ("SL", "SU", "TARGET"))
    return X, S


def run(args):
    P = load_pipeline(args.pipeline)
    feature_dir = args.features_dir or os.path.join(P.RESULT_DIR,
                                                    "repeat_features")
    args.out_dir = args.out_dir or os.path.join(P.RESULT_DIR, "block_strength")
    # Same repeats as the main run unless asked otherwise, so the diagnostic
    # covers exactly the splits the reported results came from.
    n_repeats = int(P.REPEATS if args.repeats is None else args.repeats)
    have_cache = os.path.isdir(feature_dir)
    print(f"pipeline : {args.pipeline}")
    print(f"prep dir : {P.PREP_DIR}")
    print(f"features : {feature_dir}  "
          f"{'(found)' if have_cache else '(not found)'}")
    print(f"out dir  : {args.out_dir}")
    print(f"nT={P.NT}  n_valid={P.N_VALID}  seed={P.SEED}  repeats={n_repeats}")
    if not have_cache:
        raise FileNotFoundError(
            f"no cache: {feature_dir} does not exist. Run the main pipeline "
            "with SAVE_FEATURES = True first, or pass --features-dir.")

    audit = pd.read_csv(os.path.join(P.PREP_DIR, "row_audit.csv"))
    vocab = pd.read_csv(os.path.join(P.PREP_DIR, "surrogate_vocabulary.csv"))
    meta = pd.read_csv(os.path.join(P.PREP_DIR, "metadata_feature_names.csv"))
    text_dim, meta_dim = len(vocab), len(meta)

    splits = {k: P.load_prepared_split(k, text_dim, meta_dim)
              for k in ("SL", "SU", "TARGET")}
    for k in splits:                       # pixels are never needed here
        splits[k]["X"] = None
    S_raw = {k: np.hstack([splits[k]["text"], splits[k]["meta"]])
             for k in splits}
    Y_SL = splits["SL"]["Y"].astype(int)
    Y_TARGET = splits["TARGET"]["Y"].astype(int)

    target_audit = audit[audit.split_group == "TARGET"].reset_index(drop=True)
    pids = target_audit["patient_id_parsed"].astype(str).to_numpy()
    prevalence = float(Y_TARGET.mean())
    print(f"cohort   : SL={len(Y_SL)}  TARGET={len(Y_TARGET)}  "
          f"prevalence={prevalence:.4f}  q={S_raw['SL'].shape[1]}")

    rows = []
    for rep in range(n_repeats):
        seed = P.SEED + rep * 1000 + P.NT
        try:
            sp = P.patient_split_tl_valid_tu(
                pids, Y_TARGET, P.NT, P.N_VALID, P.MAX_IMAGES_PER_PATIENT,
                prevalence, seed)
        except Exception as exc:                      # noqa: BLE001
            print(f"  repeat {rep}: split failed ({exc})")
            continue
        tl, val, tu = sp["tl_idx"], sp["valid_idx"], sp["tu_idx"]

        for branch in args.branches:
            source = "cache"
            try:
                (X_SL, X_SU, X_TARGET), (S_SL, S_SU, S_TARGET) = \
                    load_cached_repeat(feature_dir, rep, P.NT, branch)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"no cache for branch {branch}, repeat {rep}: {exc} does "
                    "not exist. Run the main pipeline with SAVE_FEATURES = True "
                    "for every repeat measured here, or pass --features-dir."
                ) from None

            if len(X_TARGET) != len(Y_TARGET) or len(X_SL) != len(Y_SL):
                print(f"  [skip {branch} rep {rep}] feature rows "
                      f"{len(X_SL)}/{len(X_TARGET)} do not match labels "
                      f"{len(Y_SL)}/{len(Y_TARGET)}")
                continue

            common = dict(X_ev=X_TARGET[val], S_ev=S_TARGET[val],
                          y_ev=Y_TARGET[val], seed=seed)
            for population, (Xtr, Str, ytr) in {
                "target_TL": (X_TARGET[tl], S_TARGET[tl], Y_TARGET[tl]),
                "source_SL": (X_SL, S_SL, Y_SL),
            }.items():
                r = measure(P, Xtr, Str, ytr, population=population, **common)
                r.update({"repeat": rep, "seed": seed, "branch": branch,
                          "feature_source": source})
                rows.append(r)

            last = rows[-2]
            print(f"  rep {rep:>2} {branch:<7} TL: "
                  f"AUC image={last['auc_image_only']:.3f} "
                  f"surrogate={last['auc_surrogate_only']:.3f} "
                  f"joint={last['auc_joint']:.3f}  "
                  f"gamma share of strength={last['strength_gamma_share']:+.3f}")

    if not rows:
        print("no repeats completed")
        return 1

    frame = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    raw = os.path.join(args.out_dir, "block_strength_raw.csv")
    frame.to_csv(raw, index=False)
    report(frame, args.out_dir)
    print(f"\nraw per-repeat values: {raw}")
    return 0


# ======================================================================
# Reporting
# ======================================================================

def report(frame: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 78)
    print("BLOCK STRENGTH INSIDE THE IMPUTATION MODEL")
    print("=" * 78)

    cols = ["auc_image_only", "auc_surrogate_only", "auc_joint",
            "increment_of_surrogate", "increment_of_image",
            "l1_beta", "l1_gamma", "l1_gamma_share",
            "nnz_beta", "nnz_gamma",
            "strength_phi_beta", "strength_phi_gamma", "strength_gamma_share"]
    summary = (frame.groupby(["branch", "population"])[cols]
               .agg(["mean", "std"]).round(4))
    print("\n" + summary.to_string())
    summary.to_csv(os.path.join(out_dir, "block_strength_summary.csv"))

    print("\n--- Which block is stronger? ---")
    verdicts = {}
    for (branch, pop), s in frame.groupby(["branch", "population"]):
        d = (s.auc_surrogate_only - s.auc_image_only).values
        ci = (stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
              if len(d) > 1 and d.std() > 0 else (np.nan, np.nan))
        stronger = ("surrogate" if ci[0] > 0 else
                    "image" if ci[1] < 0 else "indistinguishable")
        print(f"\n{branch} / {pop}")
        print(f"  AUC surrogate - image = {d.mean():+.4f}  "
              f"CI({ci[0]:+.4f}, {ci[1]:+.4f})   -> {stronger}")
        print(f"  surrogate adds {s.increment_of_surrogate.mean():+.4f} AUC "
              f"on top of the image")
        print(f"  image adds     {s.increment_of_image.mean():+.4f} AUC "
              f"on top of the surrogate")
        print(f"  gamma holds {s.l1_gamma_share.mean():.1%} of the L1 mass "
              f"and {s.strength_gamma_share.mean():.1%} of the block-inclusion "
              f"Shapley total")
        verdicts[f"{branch}/{pop}"] = {
            "auc_gap_surrogate_minus_image": float(d.mean()),
            "ci": [float(ci[0]), float(ci[1])],
            "stronger_block": stronger,
            "gamma_share_of_l1": float(s.l1_gamma_share.mean()),
            "gamma_share_of_strength": float(s.strength_gamma_share.mean()),
        }

    print("\n--- Consequences ---")
    tl = frame[frame.population == "target_TL"]
    share = tl.strength_gamma_share.mean()
    if share > 0.6:
        print(f"  The surrogate block carries {share:.0%} of the block-"
              "inclusion Shapley total, so it is the stronger block.")
        print("  On the simulation's strength ladder a defect in the stronger")
        print("  block is identified less often: on the surrogate rungs, in 65")
        print("  down to 10 percent of replicates under D_S, against 93 to 100")
        print("  under D_X. The ordering of phi_X and phi_S is read only where")
        print("  the blocks are comparably strong.")
    elif share < 0.4:
        print(f"  The image block is the stronger one ({1 - share:.0%} of the")
        print("  block-inclusion total). On the ladder's image-dominant rung a")
        print("  defect in the weaker surrogate block is identified in every")
        print("  replicate, one in the stronger image block in 9 percent.")
    else:
        print(f"  The two blocks are comparable ({share:.0%} to the "
              f"surrogate): the balanced rung of the ladder, where a defect")
        print("  in either block is identified in 82 percent of replicates.")

    print("\n--- Coefficient mass in the TL fits ---")
    print("  L1 mass and support of each block in the target-only joint fit,")
    print("  for reference. The simulation does not take its block multipliers")
    print("  from these: its ladder is indexed by the AUC gap, and the gap")
    print("  reported above is what places each branch on it.\n")
    for (branch, pop), s in frame.groupby(["branch", "population"]):
        if pop != "target_TL":
            continue
        ratio = s.l1_gamma.mean() / max(s.l1_beta.mean(), 1e-12)
        per_beta = s.l1_beta.mean() / max(s.nnz_beta.mean(), 1)
        per_gamma = s.l1_gamma.mean() / max(s.nnz_gamma.mean(), 1)
        print(f"  {branch}: gamma/beta L1 ratio = {ratio:.2f}; "
              f"mean |coefficient| beta={per_beta:.3f}, gamma={per_gamma:.3f}; "
              f"support {s.nnz_beta.mean():.0f} vs {s.nnz_gamma.mean():.0f}")

    with open(os.path.join(out_dir, "block_strength_verdict.json"), "w") as h:
        json.dump(verdicts, h, indent=2, default=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    ap.add_argument("--repeats", type=int, default=None,
                    help="measure repeats 0..N-1; defaults to the pipeline's "
                         "REPEATS (20), the repeats the main run reports")
    ap.add_argument("--features-dir", default=None,
                    help="per-repeat features saved by the pipeline; "
                         "defaults to RESULT_DIR/repeat_features. Nothing is "
                         "rebuilt: a missing repeat stops the run with a "
                         "no cache error")
    ap.add_argument("--branches", default="biovil,ssl,no_ssl",
                    help="comma-separated; every branch reads the saved "
                         "features")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to RESULT_DIR/block_strength")
    args = ap.parse_args(argv)
    args.branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
