#!/usr/bin/env python3
"""Offline B_infty: the moment imbalance the classification density ratio leaves
unenforced.

Nothing is refitted. Every input is already on disk from the main run:

  selected_indices/repeat{r}_nT{NT}_patient_split.npz   -> tu_idx
  repeat_features/rep{r}_nT{NT}/{X_SL_*, X_TARGET_*, S_SL, S_TARGET}.npy
  density_ratio/repeat{r}_{branch}_weights.npz          -> w_paper_xs, w_image_only

For each repeat and branch we form Z = [X, S] and report

    B_inf = || Ehat_S[w Z] - Ehat_T[Z] ||_inf

against the target rows the domain classifier was fitted on (TU by default),
which is the empirical counterpart of E_S[w_0(Z) h(Z)] = E_T[h(Z)] at h(z) = z.

WHY THE UNWEIGHTED COLUMN MATTERS. B_inf on its own is not interpretable: a
value of 0.3 could be a triumph or a failure depending on where the two domains
started. The script therefore also reports the same statistic at w == 1, so the
paper can quote what the reweighting removed rather than only what it left. The
reduction column is the ratio of the two.

Units are those of the stored features, which the pipeline standardises and
whitens on the pooled rows, so an imbalance of 0.12 reads as 0.12 pooled SD on
the worst coordinate.

Run once per arm, pointing --result-dir at that arm's RESULT_DIR: the weight
filenames carry the repeat and branch but not nT, so the two arms are only kept
apart by living in separate result directories.

Usage
-----
    python -u application/b_infty_offline.py --result-dir results/pneumothorax_history_bow_nT200_valid200_strifle_shapley --nT 200
    python -u application/b_infty_offline.py --result-dir ... --nT 200 --target-rows tu

(from the repository root, where the results/ path above resolves)
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# branch name in IMAGE_BRANCHES -> suffix used in the saved feature filenames
BRANCH_FEATURE_SUFFIX = {
    "ssl": "simclr",
    "biovil": "biovil",
    "no_ssl": "no_ssl",
}


def imbalance(weights, Z_source, Z_target):
    """Return the L-infinity, L2 and mean-absolute coordinate imbalance.

    ``weights`` are the density ratio values on the source rows, normalised by
    the pipeline to mean one over those same rows, so a plain column mean of
    ``weights[:, None] * Z_source`` already estimates E_S[w Z].
    """
    weights = np.asarray(weights, dtype=np.float64).ravel()
    Z_source = np.asarray(Z_source, dtype=np.float64)
    Z_target = np.asarray(Z_target, dtype=np.float64)
    if len(weights) != len(Z_source):
        raise ValueError(
            f"weights carry {len(weights)} rows but the source block has "
            f"{len(Z_source)}; the weight file and the feature file do not "
            "come from the same repeat."
        )
    gap = (weights[:, None] * Z_source).mean(axis=0) - Z_target.mean(axis=0)
    return gap


def summarise_gap(gap, p_image):
    """Split one imbalance vector into the two blocks the paper reasons about."""
    out = {
        "b_infty": float(np.max(np.abs(gap))),
        "b_l2": float(np.sqrt(np.sum(gap ** 2))),
        "b_mean_abs": float(np.mean(np.abs(gap))),
        "worst_coordinate": int(np.argmax(np.abs(gap))),
    }
    if p_image is not None and 0 < p_image < len(gap):
        out["b_infty_image"] = float(np.max(np.abs(gap[:p_image])))
        out["b_infty_surrogate"] = float(np.max(np.abs(gap[p_image:])))
        out["worst_block"] = ("image" if out["b_infty_image"] >= out["b_infty_surrogate"]
                              else "surrogate")
    return out


def load_repeat(result_dir, rep, n_t):
    """Return (feature dict, tu_idx) for one repeat, or None if incomplete."""
    run_key = f"rep{rep}_nT{n_t}"
    feature_dir = os.path.join(result_dir, "repeat_features", run_key)
    split_path = os.path.join(
        result_dir, "selected_indices", f"repeat{rep}_nT{n_t}_patient_split.npz"
    )
    if not os.path.isdir(feature_dir) or not os.path.exists(split_path):
        return None

    with np.load(split_path, allow_pickle=False) as data:
        idx = {k: data[k] for k in ("tl_idx", "tu_idx", "valid_idx")}

    features = {}
    for name in os.listdir(feature_dir):
        if name.endswith(".npy"):
            features[name[:-4]] = np.load(
                os.path.join(feature_dir, name), mmap_mode="r"
            )
    return features, idx


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", required=True,
                        help="RESULT_DIR of one arm of the main run.")
    parser.add_argument("--nT", type=int, default=None,
                        help="Labelled target fitting size. Inferred from the "
                             "repeat_features directory names when omitted.")
    parser.add_argument("--target-rows", choices=("tu", "tu_tl"), default="tu",
                        help="Which target rows form Ehat_T[Z]. 'tu' matches the "
                             "rows the domain classifier was fitted on and is "
                             "the default; 'tu_tl' is a robustness variant.")
    parser.add_argument("--branches", nargs="+", default=list(BRANCH_FEATURE_SUFFIX),
                        help="Subset of branches to score.")
    parser.add_argument("--out", default=None,
                        help="CSV path. Defaults to <result-dir>/b_infty.csv")
    args = parser.parse_args()

    feature_root = os.path.join(args.result_dir, "repeat_features")
    if not os.path.isdir(feature_root):
        sys.exit(f"No repeat_features/ under {args.result_dir}. Was the run "
                 "executed with SAVE_FEATURES = True?")

    keys = sorted(os.listdir(feature_root))
    parsed = [re.match(r"rep(\d+)_nT(\d+)$", k) for k in keys]
    parsed = [m for m in parsed if m]
    if not parsed:
        sys.exit(f"No rep*_nT* directories under {feature_root}.")

    n_t = args.nT
    if n_t is None:
        found = sorted({int(m.group(2)) for m in parsed})
        if len(found) != 1:
            sys.exit(f"Several nT values present ({found}); pass --nT.")
        n_t = found[0]
    repeats = sorted(int(m.group(1)) for m in parsed if int(m.group(2)) == n_t)
    print(f"[b_infty] {args.result_dir}")
    print(f"[b_infty] nT={n_t}, repeats={repeats}, "
          f"target rows = {args.target_rows}\n")

    records = []
    for rep in repeats:
        loaded = load_repeat(args.result_dir, rep, n_t)
        if loaded is None:
            print(f"  [skip] repeat {rep}: features or split missing")
            continue
        features, idx = loaded
        target_idx = (idx["tu_idx"] if args.target_rows == "tu"
                      else np.concatenate([idx["tu_idx"], idx["tl_idx"]]))

        S_SL = np.asarray(features["S_SL"], dtype=np.float64)
        S_TARGET = np.asarray(features["S_TARGET"], dtype=np.float64)

        for branch in args.branches:
            suffix = BRANCH_FEATURE_SUFFIX[branch]
            weight_path = os.path.join(
                args.result_dir, "density_ratio", f"repeat{rep}_{branch}_weights.npz"
            )
            if not os.path.exists(weight_path):
                print(f"  [skip] repeat {rep}, {branch}: no weight file")
                continue
            if f"X_SL_{suffix}" not in features:
                print(f"  [skip] repeat {rep}, {branch}: no X_SL_{suffix}.npy")
                continue

            X_SL = np.asarray(features[f"X_SL_{suffix}"], dtype=np.float64)
            X_TARGET = np.asarray(features[f"X_TARGET_{suffix}"], dtype=np.float64)
            p_image = X_SL.shape[1]

            Z_SL = np.hstack([X_SL, S_SL])
            Z_T = np.hstack([X_TARGET[target_idx], S_TARGET[target_idx]])

            with np.load(weight_path, allow_pickle=False) as wdata:
                available = set(wdata.files)
                w_xs = np.asarray(wdata["w_paper_xs"], dtype=np.float64) \
                    if "w_paper_xs" in available else None
                w_img = np.asarray(wdata["w_image_only"], dtype=np.float64) \
                    if "w_image_only" in available else None

            # --- the paper's ratio, fitted and applied on Z ------------------
            if w_xs is not None:
                gap_w = imbalance(w_xs, Z_SL, Z_T)
                gap_0 = imbalance(np.ones(len(Z_SL)), Z_SL, Z_T)
                row = {
                    "repeat": rep, "nT": n_t, "branch": branch,
                    "ratio": "paper_x_plus_s",
                    "p_image": p_image, "q": S_SL.shape[1],
                    "n_source_rows": len(Z_SL), "n_target_rows": len(Z_T),
                    "w_mean": float(w_xs.mean()), "w_max": float(w_xs.max()),
                    "ess": float(w_xs.sum() ** 2 / (w_xs ** 2).sum()),
                }
                row.update(summarise_gap(gap_w, p_image))
                unweighted = summarise_gap(gap_0, p_image)
                row.update({f"unweighted_{k}": v for k, v in unweighted.items()
                            if not isinstance(v, str)})
                row["reduction"] = (row["b_infty"] / unweighted["b_infty"]
                                    if unweighted["b_infty"] > 0 else np.nan)
                records.append(row)

            # --- the image-only ratio the CS comparator is fed ---------------
            if w_img is not None:
                gap_w = imbalance(w_img, X_SL, X_TARGET[target_idx])
                gap_0 = imbalance(np.ones(len(X_SL)), X_SL, X_TARGET[target_idx])
                row = {
                    "repeat": rep, "nT": n_t, "branch": branch,
                    "ratio": "image_only",
                    "p_image": p_image, "q": 0,
                    "n_source_rows": len(X_SL), "n_target_rows": len(Z_T),
                    "w_mean": float(w_img.mean()), "w_max": float(w_img.max()),
                    "ess": float(w_img.sum() ** 2 / (w_img ** 2).sum()),
                }
                row.update(summarise_gap(gap_w, None))
                unweighted = summarise_gap(gap_0, None)
                row.update({f"unweighted_{k}": v for k, v in unweighted.items()
                            if not isinstance(v, str)})
                row["reduction"] = (row["b_infty"] / unweighted["b_infty"]
                                    if unweighted["b_infty"] > 0 else np.nan)
                records.append(row)

    if not records:
        sys.exit("Nothing scored. Check --result-dir and --nT.")

    frame = pd.DataFrame.from_records(records)
    out_path = args.out or os.path.join(args.result_dir, "b_infty.csv")
    frame.to_csv(out_path, index=False)

    cols = ["b_infty", "unweighted_b_infty", "reduction",
            "b_infty_image", "b_infty_surrogate", "ess"]
    summary = (frame.groupby(["ratio", "branch"])[
        [c for c in cols if c in frame.columns]
    ].agg(["mean", "min", "max"]).round(4))
    print(summary.to_string())
    print(f"\n[b_infty] {len(frame)} rows written to {out_path}")

    high = frame.loc[frame["w_mean"].sub(1.0).abs() > 1e-6]
    if len(high):
        print("\n[warn] some weight vectors do not have mean one on the source "
              "rows; the pipeline renormalises them, so check that the weight "
              "file and the feature file belong to the same repeat:")
        print(high[["repeat", "branch", "ratio", "w_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
