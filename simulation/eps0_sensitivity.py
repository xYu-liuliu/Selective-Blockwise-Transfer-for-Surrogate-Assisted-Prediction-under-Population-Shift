"""
Sensitivity of the gate's deployed-layer performance to the choice of eps0.

The four coalition losses are stored per replicate, so the selection rule can be
replayed at any threshold without rerunning the simulation.  Decisions are made
on the imputation-layer losses L_c; the deployed-layer metric only scores the
result and never enters the decision.

Replicates are split by the realised defect size rather than pooled, because a
pooled optimum would be an artefact of how many cells the magnitude grid happens
to place on each side.  The claim the figure supports is that the threshold is
not sharply optimal in either direction, which does not depend on that mixture.

Outputs: eps0_sensitivity.png, eps0_sensitivity.csv
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# EDIT THESE
# --------------------------------------------------------------------------
# The simulation's batches are read from results/<batch>, where
# simulation_shapley_2.py writes them; STRIFLE_RESULTS_DIR overrides the root,
# as it does there.
_RESULTS_OVERRIDE = os.environ.get("STRIFLE_RESULTS_DIR", "").strip()
RESULTS = Path(os.path.abspath(os.path.expanduser(_RESULTS_OVERRIDE))
               if _RESULTS_OVERRIDE
               else os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "results"))
RUNS = ["negtransfer", "dense"]      # raws to pool; overlapping rows are dropped
OUTDIR = RESULTS                     # eps0_sensitivity.csv and .png go here

METRIC = "brier_pred"                # brier_pred | auc_pred | ece_pred | *_imp
LOWER_IS_BETTER = True               # False for auc_*

BENIGN_MAX = 1.0                     # radii; <= this counts as benign
ADVERSE_MIN = 2.0                    # radii; >= this counts as adverse

LOG_D = np.log(221.0)                # log d at the reference cell
N_T = 200
EPS0_APPLIED = LOG_D / N_T           # 0.02699

COALITIONS = ["none", "beta", "gamma", "both"]
KEY = ["scenario", "magnitude", "repeat"]


# --------------------------------------------------------------------------
def load_pooled(results: Path, runs) -> pd.DataFrame:
    """Concatenate raws, dropping rows that appear in more than one run.

    Runs sharing a design point share paired random numbers, so the same
    replicate can appear twice; the earlier run in `runs` wins.
    """
    frames = []
    for name in runs:
        d = pd.read_csv(results / name / "simulation_parametric_raw.csv")
        d["run"] = name
        frames.append(d)
    pooled = pd.concat(frames, ignore_index=True)

    before = len(pooled)
    pooled = pooled.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)
    dropped = before - len(pooled)
    if dropped:
        print(f"dropped {dropped} duplicated replicates across runs")
    return pooled


def replay(df: pd.DataFrame, eps0: float, metric: str) -> dict:
    """Re-run the selection rule of eq:rules at one threshold.

    A coalition is eligible when it beats the target-only fit by at least eps0
    on the imputation-layer loss; the eligible coalition with the smallest loss
    is taken.  `none` is always eligible.
    """
    L = df[[f"L_{c}" for c in COALITIONS]].to_numpy()
    M = df[[f"{metric}_{c}" for c in COALITIONS]].to_numpy()
    base = df["L_none"].to_numpy()

    eligible = L <= (base - eps0)[:, None]
    eligible[:, 0] = True                       # `none` always available

    masked = np.where(eligible, L, np.inf)
    pick = masked.argmin(axis=1)

    return {
        "eps0": eps0,
        "metric_mean": M[np.arange(len(df)), pick].mean(),
        "transfer_rate": (pick != 0).mean(),     # share not falling back to none
        "rho": (L[:, 3] <= base - eps0).mean(),  # XS admissible (the binary rho)
    }


def curve(df: pd.DataFrame, grid, metric: str) -> pd.DataFrame:
    return pd.DataFrame([replay(df, e, metric) for e in grid])


def fixed_policies(df: pd.DataFrame, metric: str) -> dict:
    return {
        "always_none": df[f"{metric}_none"].mean(),
        "always_both": df[f"{metric}_both"].mean(),
    }


def plateau(c: pd.DataFrame, tol: float, lower_is_better: bool):
    """Widest eps0 interval containing EPS0_APPLIED within `tol` of the best."""
    best = c.metric_mean.min() if lower_is_better else c.metric_mean.max()
    ok = (c.metric_mean <= best * (1 + tol)) if lower_is_better \
        else (c.metric_mean >= best * (1 - tol))
    inside = c.loc[ok, "eps0"]
    if inside.empty:
        return np.nan, np.nan
    return inside.min(), inside.max()


# --------------------------------------------------------------------------
def main() -> None:
    d = load_pooled(RESULTS, RUNS)

    # Group on the realised defect, not the requested magnitude: the two differ
    # where logit saturation kept the calibration from reaching its target.
    # The radius is averaged within a design cell first, or replicate-level
    # jitter would split a cell across both groups.
    cell = ["scenario", "magnitude"]
    d["radii_cell"] = d.groupby(cell)["delta0_l1_over_budget"].transform("mean")
    groups = {
        "benign": d[d.radii_cell <= BENIGN_MAX],
        "adverse": d[d.radii_cell >= ADVERSE_MIN],
    }
    for name, g in groups.items():
        cells = g.groupby(["scenario", "magnitude"]).size()
        print(f"\n{name}: {len(g)} replicates over {len(cells)} cells")
        print(cells.to_string())

    grid = np.concatenate([[0.0], np.logspace(-3.2, -0.4, 40)])
    curves = {k: curve(g, grid, METRIC) for k, g in groups.items()}
    fixed = {k: fixed_policies(g, METRIC) for k, g in groups.items()}

    out = pd.concat([c.assign(group=k) for k, c in curves.items()], ignore_index=True)
    out.to_csv(OUTDIR / "eps0_sensitivity.csv", index=False)

    for k, c in curves.items():
        lo, hi = plateau(c, 0.01, LOWER_IS_BETTER)
        at = float(np.interp(EPS0_APPLIED, c.eps0, c.metric_mean))
        best = c.metric_mean.min() if LOWER_IS_BETTER else c.metric_mean.max()
        print(f"\n[{k}] {METRIC}")
        print(f"  always-none {fixed[k]['always_none']:.5f} | "
              f"always-both {fixed[k]['always_both']:.5f}")
        print(f"  best {best:.5f} | at log d/n_T {at:.5f} | "
              f"excess {100 * abs(at - best) / best:.2f}%")
        print(f"  within 1% of best over eps0 in [{lo:.4f}, {hi:.4f}]")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    titles = {
        "benign": rf"benign: $\|\delta_0\|_1 \leq {BENIGN_MAX:g}$ radii",
        "adverse": rf"adverse: $\|\delta_0\|_1 \geq {ADVERSE_MIN:g}$ radii",
    }

    for ax, (k, c) in zip(axes, curves.items()):
        ax.plot(c.eps0, c.metric_mean, color="0.15", lw=1.6, zorder=3)
        ax.axhline(fixed[k]["always_both"], color="0.55", ls="--", lw=1.0,
                   label="always transfer")
        ax.axhline(fixed[k]["always_none"], color="0.55", ls=":", lw=1.0,
                   label="never transfer")
        ax.axvline(EPS0_APPLIED, color="firebrick", lw=1.2, zorder=2)
        ax.annotate(r"$\log d / n_T$", xy=(EPS0_APPLIED, 0.97),
                    xycoords=("data", "axes fraction"), color="firebrick",
                    fontsize=8, ha="right", va="top", rotation=90)

        lo, hi = plateau(c, 0.01, LOWER_IS_BETTER)
        if np.isfinite(lo):
            ax.axvspan(max(lo, grid[1] / 2), hi, color="firebrick", alpha=0.07, zorder=0)

        ax.set_xscale("symlog", linthresh=grid[1])
        ax.set_title(titles[k], fontsize=9)
        ax.set_xlabel(r"$\epsilon_0$")
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel(METRIC.replace("_", " "))
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "eps0_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUTDIR / 'eps0_sensitivity.png'}")


if __name__ == "__main__":
    main()
