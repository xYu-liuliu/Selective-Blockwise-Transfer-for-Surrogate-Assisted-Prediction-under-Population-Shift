"""
Table A: what each coalition buys at each layer, with and without the
imputation pass.

The aggregate outputs only carry none / both / gate, so the single-block
coalitions have to be read off the per-replicate files.

D_none is the cell to read.  With no defect, any difference between the blocks
comes from how much of each one reaches the deployed model, not from a defect
sitting in one of them.

The oracle branch replaces the soft labels of eq:beta-loss with ones drawn from
the true target parameter and leaves everything upstream alone, so the two runs
are paired replicate by replicate and the imputation layer is bit-identical
between them.  The script asserts that before using the pairing.

Outputs a LaTeX body and the paired confidence intervals.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# The simulation's batches are read from results/<batch>, where
# simulation_shapley_2.py writes them; STRIFLE_RESULTS_DIR overrides the root,
# as it does there.
_RESULTS_OVERRIDE = os.environ.get("STRIFLE_RESULTS_DIR", "").strip()
RESULTS = Path(os.path.abspath(os.path.expanduser(_RESULTS_OVERRIDE))
               if _RESULTS_OVERRIDE
               else os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "results"))
BASELINE = RESULTS / "negtransfer" / "simulation_parametric_raw.csv"
ORACLE = RESULTS / "oracle" / "simulation_parametric_raw.csv"

CELL = {"scenario": "S0"}          # the no-defect cell
COALITIONS = ["none", "beta", "gamma", "both"]
LABEL = {"none": r"$\varnothing$", "beta": r"$\mathsf{X}$",
         "gamma": r"$\mathsf{S}$", "both": r"$\mathsf{XS}$"}
METRIC = "auc_pred"                # deployed-layer metric for the two right columns
HIGHER_IS_BETTER = True


def subset(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    for k, v in CELL.items():
        d = d[d[k] == v]
    return d.set_index("repeat").sort_index()


def ci(x: pd.Series, level: float = 0.95):
    """Paired interval for a per-replicate difference."""
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def fmt(v: float, lo: float, hi: float, signed: bool = True) -> str:
    s = "+" if (signed and v >= 0) else ("-" if signed else "")
    body = f"{s}{abs(v):.4f}" if signed else f"{v:.4f}"
    return f"${body}$ {{\\scriptsize$[{lo:+.4f}, {hi:+.4f}]$}}".replace("0.", ".")


def main() -> None:
    b, o = subset(BASELINE), subset(ORACLE)
    common = b.index.intersection(o.index)
    b, o = b.loc[common], o.loc[common]
    print(f"{len(common)} paired replicates in {CELL}")

    # The ablation must not have touched anything upstream of the imputation.
    gap = max((b[f"L_{c}"] - o[f"L_{c}"]).abs().max() for c in COALITIONS)
    assert gap < 1e-10, f"imputation layer differs between runs (max {gap:.2e})"
    print(f"imputation layer identical between runs (max |diff| {gap:.1e})\n")

    rows = []
    for c in COALITIONS:
        dL = b["L_none"] - b[f"L_{c}"]                              # loss improvement
        dB = b[f"{METRIC}_{c}"] - b[f"{METRIC}_none"]               # deployed, baseline
        dO = o[f"{METRIC}_{c}"] - o[f"{METRIC}_none"]               # deployed, oracle
        rows.append({
            "coalition": c,
            "dL": dL.mean(), "dL_ci": ci(dL),
            "dB": dB.mean(), "dB_ci": ci(dB),
            "dO": dO.mean(), "dO_ci": ci(dO),
            "abs_base": b[f"{METRIC}_{c}"].mean(),
            "abs_orac": o[f"{METRIC}_{c}"].mean(),
        })
    t = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print(t[["coalition", "dL", "dB", "dO", "abs_base", "abs_orac"]]
          .round(4).to_string(index=False))

    print("\n% --- LaTeX body (Table A) ---")
    for _, r in t.iterrows():
        if r.coalition == "none":
            print(rf"{LABEL[r.coalition]} & --- & --- & --- \\")
            continue
        print(f"{LABEL[r.coalition]} & "
              f"{fmt(r.dL, *r.dL_ci)} & "
              f"{fmt(r.dB, *r.dB_ci)} & "
              f"{fmt(r.dO, *r.dO_ci)} " + r"\\")

    # ---- the two numbers the prose needs ----
    x = t[t.coalition == "beta"].iloc[0]
    s = t[t.coalition == "gamma"].iloc[0]
    xs = t[t.coalition == "both"].iloc[0]

    print(f"\nimputation contributions X vs S: {x.dL:+.4f} vs {s.dL:+.4f} "
          f"({100 * (x.dL / s.dL - 1):.0f}% apart)")
    print(f"deployed gains, baseline:        {x.dB:+.4f} vs {s.dB:+.4f} "
          f"(factor {x.dB / s.dB:.0f})")
    print(f"deployed gains, oracle:          {x.dO:+.4f} vs {s.dO:+.4f}")
    print(f"oracle spread across coalitions: "
          f"{t.abs_orac.max() - t.abs_orac.min():.4f}")

    ceiling = o[f"{METRIC}_none"].mean()      # oracle labels, no borrowing
    floor = b[f"{METRIC}_none"].mean()        # real labels, no borrowing
    reached = b[f"{METRIC}_both"].mean()      # real labels, borrowing
    print(f"\nfloor {floor:.4f} -> XS {reached:.4f} -> oracle ceiling {ceiling:.4f}")
    print(f"transfer closes {100 * (reached - floor) / (ceiling - floor):.0f}% "
          f"of the gap to the ceiling")


if __name__ == "__main__":
    main()