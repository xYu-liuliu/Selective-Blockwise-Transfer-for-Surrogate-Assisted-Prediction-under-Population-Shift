"""fista_paired.py: the test that settles it.

The census locates the non-convergence. It does not prove the results are
unaffected, because there is one channel through which a mis-solved lambda
can still reach the estimate: a lambda solved badly gets an inflated CV
loss and loses the selection it might otherwise have won. Solving it
properly can therefore change the selected lambda, hence delta, hence
theta_meta, hence every coalition loss.

So: run the same cells twice with identical seeds, once with the shipped
solver and once with a hardened one, and compare replicate by replicate.
_fista draws no random numbers, so replacing it shifts no RNG stream and
the pairing is exact: every difference observed is attributable to the
solver and nothing else.

The hardened solver fixes three things:
  * step size computed for the matrix it is actually given, instead of
    inherited from the full design (that is what makes 97% of the CV
    subproblems run at step > 1/L, with no descent guarantee)
  * adaptive restart, which is what stops FISTA oscillating at the sparse
    end of the path
  * a stopping rule on the subgradient KKT residual rather than on the
    iterate change, so "converged" means optimal

Report the table this prints in the paper. If the movement is at 1e-6 and
no selection flips, one sentence retires the issue.

Run:
    python fista_paired.py --reps 40 --scenarios S1,S2 --magnitudes 0.5,1.0,2.0

The simulation is loaded from simulation_shapley_2.py next to this file, and
the two arms are written to <out-prefix>_baseline.csv and
<out-prefix>_hardened.csv, results/paired by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# The simulation file sits next to this one; output goes under results/, or
# under STRIFLE_RESULTS_DIR when that is set, as for simulation_shapley_2.py.
HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_OVERRIDE = os.environ.get("STRIFLE_RESULTS_DIR", "").strip()
RESULTS_DIR = (os.path.abspath(os.path.expanduser(_RESULTS_OVERRIDE))
               if _RESULTS_OVERRIDE
               else os.path.join(os.path.dirname(HERE), "results"))


def load_sim(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# The hardened solver
# --------------------------------------------------------------------------

def make_hardened(sim, max_iter=50_000, kkt_tol=1e-8, check_every=50):
    """Drop-in replacement for sim._fista with a genuine stopping rule."""

    def _kkt_rel(A, y, offset, lam, b, n):
        eta = offset + A @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        grad = A.T @ (p - y) / n
        res = abs(grad[0])
        g, bb = grad[1:], b[1:]
        act = bb != 0.0
        if act.any():
            res = max(res, float(np.max(np.abs(g[act] + lam * np.sign(bb[act])))))
        if (~act).any():
            res = max(res, float(np.max(np.maximum(np.abs(g[~act]) - lam, 0.0))))
        return res / max(lam, 1e-300)

    def hardened(A, y, offset, lam, step, iters=None, tol=None):
        n, m = A.shape
        y = np.asarray(y, float)
        # Step for THIS matrix, not the one the caller happened to measure.
        L = 0.25 * np.linalg.norm(A, 2) ** 2 / n
        step = 1.0 / max(L, 1e-12)

        b = np.zeros(m)
        z = b.copy()
        t = 1.0
        converged = False
        for it in range(max_iter):
            eta = offset + A @ z
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
            grad = A.T @ (p - y) / n
            b_new = z - step * grad
            thr = step * lam
            b_new[1:] = np.sign(b_new[1:]) * np.maximum(
                np.abs(b_new[1:]) - thr, 0.0)

            if (it + 1) % check_every == 0 and \
                    _kkt_rel(A, y, offset, lam, b_new, n) <= kkt_tol:
                b = b_new
                converged = True
                break

            # O'Donoghue-Candes gradient restart. Reuses vectors already in
            # hand, so it costs nothing, and it is what stops the oscillation
            # at the sparse end of the path.
            if float((z - b_new) @ (b_new - b)) > 0.0:
                t, z = 1.0, b_new.copy()
            else:
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                z = b_new + ((t - 1) / t_new) * (b_new - b)
                t = t_new
            b = b_new

        sim._FISTA_NONCONVERGENCE["total"] += 1
        if not converged:
            sim._FISTA_NONCONVERGENCE["count"] += 1
        return b

    return hardened


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

KEY = ["scenario", "magnitude", "repeat"]

CONTINUOUS = ["phi_beta", "phi_gamma", "phi_total", "efficiency_gap",
              "L_none", "L_beta", "L_gamma", "L_both",
              "delta_l1", "delta_beta_l1", "delta_gamma_l1",
              "theta_meta_minus_truth_l1",
              "auc_imp_none", "auc_imp_both", "auc_pred_none", "auc_pred_both",
              "brier_imp_both", "ece_imp_both"]

DISCRETE = ["selected", "rho"]


def compare(base: pd.DataFrame, hard: pd.DataFrame) -> None:
    m = base.merge(hard, on=KEY, suffixes=("_base", "_hard"))
    print(f"paired replicates: {len(m)} of {len(base)} baseline rows\n")

    print("=" * 78)
    print("CONTINUOUS QUANTITIES  (absolute movement, baseline -> hardened)")
    print("=" * 78)
    rows = []
    for c in CONTINUOUS:
        a, b = f"{c}_base", f"{c}_hard"
        if a not in m or b not in m:
            continue
        d = (m[a] - m[b]).abs()
        # Scaled by the max, not the median: delta_l1 is exactly 0 in most
        # replicates, and a median of 0 makes the relative column nonsense.
        scale = m[a].abs().max()
        rows.append({"quantity": c, "median_abs_diff": d.median(),
                     "p90": d.quantile(0.9), "max_abs_diff": d.max(),
                     "max_rel_to_range": (d.max() / scale) if scale > 0
                     else np.nan})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3e"))

    print("\n" + "=" * 78)
    print("DISCRETE OUTCOMES  (any flip is a real change)")
    print("=" * 78)
    for c in DISCRETE:
        a, b = f"{c}_base", f"{c}_hard"
        if a not in m or b not in m:
            continue
        flip = (m[a] != m[b])
        print(f"{c:10s} disagreement: {flip.mean():.4f}  ({int(flip.sum())}/{len(m)})")
        if flip.any():
            print(pd.crosstab(m.loc[flip, a], m.loc[flip, b]).to_string())

    print("\n" + "=" * 78)
    print("DOWNSTREAM CONCLUSIONS")
    print("=" * 78)
    for tag, fr in (("baseline", base), ("hardened", hard)):
        sub = fr[fr.nontransferable_block.isin(["beta", "gamma"])] \
            if "nontransferable_block" in fr else fr.iloc[0:0]
        if len(sub):
            hit = np.where(sub.nontransferable_block == "beta",
                           sub.phi_beta < sub.phi_gamma,
                           sub.phi_gamma < sub.phi_beta)
            print(f"{tag:9s} identification rate: {hit.mean():.4f}  (n={len(sub)})")
    for tag, fr in (("baseline", base), ("hardened", hard)):
        print(f"{tag:9s} mean phi_beta {fr.phi_beta.mean(): .6f}   "
              f"mean phi_gamma {fr.phi_gamma.mean(): .6f}   "
              f"rho=1 rate {fr.rho.mean():.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(HERE, "simulation_shapley_2.py"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--scenarios", default="S1,S2")
    ap.add_argument("--magnitudes", default="1.0")
    ap.add_argument("--block-strength", default="balanced")
    ap.add_argument("--skip-pred", action="store_true",
                    help="imputation layer only; ~2.5x faster and enough for "
                         "phi, L and the selection")
    ap.add_argument("--out-prefix", default=os.path.join(RESULTS_DIR, "paired"))
    args = ap.parse_args(argv)

    scen = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    mags = [float(m) for m in args.magnitudes.split(",")]

    def run(hardened: bool) -> pd.DataFrame:
        sim = load_sim(args.sim, "sim_hard" if hardened else "sim_base")
        if hardened:
            sim._fista = make_hardened(sim)
        cfg = sim.Config(block_strength=args.block_strength,
                         fit_prediction_layer=not args.skip_pred)
        fr = sim.run_cell(cfg, scen, mags, args.reps, "parametric")
        c = sim._FISTA_NONCONVERGENCE
        print(f"[{'hardened' if hardened else 'baseline'}] "
              f"FISTA flag rate {c['count']}/{c['total']} = "
              f"{c['count']/max(c['total'],1):.4f}")
        return fr

    base = run(False)
    hard = run(True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    base.to_csv(f"{args.out_prefix}_baseline.csv", index=False)
    hard.to_csv(f"{args.out_prefix}_hardened.csv", index=False)
    print()
    compare(base, hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
