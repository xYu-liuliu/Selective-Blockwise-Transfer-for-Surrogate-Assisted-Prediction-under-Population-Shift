#!/usr/bin/env python3
"""Build the two n_T = 200 endpoint tables from the three runs on disk.

Reads three artefacts and emits Table~\\ref{tab:app-endpoints-valid} (paired
contrasts against target-only) and Table~\\ref{tab:app-endpoints-levels}
(levels), as LaTeX.

    raw_results.csv               run_patient_split_nT200_valid200_history_bow.py
    gate_and_shapley_audit.csv    same run; carries the four coalition losses
    suptrans_three_step_*.csv     recompute_suptrans_three_step.py

Two corrections are applied to raw_results.csv before anything is tabulated.

1.  The gates. The pipeline fits at epsilon0 = 0; the reported operating point
    is epsilon0 = log(d)/n_T per branch. Replaying the rule only reselects a
    coalition, and every coalition is already scored in raw_results.csv, so the
    gate rows are rebuilt by relabelling the selected endpoint's metric row.
    This is the same map replay_gate_epsilon.py uses, and it is re-verified at
    epsilon0 = 0 against the stored gate rows before it is trusted.

2.  SUPTRANS. The stored column is a two-step fit; the three-step procedure of
    Li et al. (1)-(3) comes from the recompute, endpoint 'suptrans'. The
    aggregation of their Section 2.2 is a separate step and is not part of the
    three-step procedure, so 'suptrans_agg' is not used. Pass --suptrans-agg to
    override.

All three default to the pipeline's own RESULT_DIR, so with an unmodified run

    python make_endpoint_tables.py

is enough. Override --result-dir for another arm (nT=400, the negative control),
together with --nT for the n_T = 400 arm, from the repository root:

    python application/make_endpoint_tables.py --nT 400 --result-dir results/pneumothorax_history_bow_nT400_valid200_strifle_shapley
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from project_paths import run_dir

# ============================================================
# Configuration
# ============================================================

# Must match RESULT_DIR in run_patient_split_nT200_valid200_history_bow.py and
# MAIN_RESULT_DIR in replay_gate_epsilon.py: the n_T = 200 arm's run directory,
# results/pneumothorax_history_bow_nT200_valid200_strifle_shapley by default
# (see project_paths.py).
RESULT_DIR = run_dir(200, 200)
# Where recompute_suptrans_three_step.py writes when --out-dir is not given.
SUPTRANS_SUBDIR = "suptrans_recompute"
SUPTRANS_FILE = "suptrans_three_step_results.csv"
# Written under whichever --result-dir is in use, never the constant above.
TABLE_SUBDIR = "tables"


# ============================================================
# Layout
# ============================================================

# branch key in the CSVs -> the \textsc{} name in the table
BRANCHES = [("biovil", "biovil"), ("ssl", "simclr"), ("no_ssl", "supervised")]

# endpoint key -> table name, in row order. Above the rule: the construction.
ENDPOINTS_ABOVE = [
    ("to", "target-only"),
    ("beta_only", "borrow-x"),
    ("gamma_only", "borrow-s"),
    ("corrected_meta", "corrected-meta"),
    ("paper_gate", "gate-binary"),
    ("shapley_gate", "gate-block"),
]
# Below the rule: the external comparators.
ENDPOINTS_BELOW = [("lr_to", "sup"), ("lr_dr", "cs"), ("suptrans", "suptrans")]
CEILING = ("target_ceiling", "ceiling")

BASELINE = "to"                 # what the contrasts are taken against
CNN_BRANCH, CNN_ENDPOINT = "cnn_direct", "dr_finetune"

# Excluded from the contrasts: CEILING sees labels no other row sees, CNN is
# outside the linear model class. Both appear as levels.
CONTRAST_EXCLUDED = {BASELINE, CEILING[0]}

# The gate rules and the coalition -> endpoint map, as in replay_gate_epsilon.py.
GATE_RULES = {"paper_gate": "paper", "shapley_gate": "shapley"}
COALITION_TO_ENDPOINT = {"none": "to", "beta": "beta_only",
                         "gamma": "gamma_only", "both": "corrected_meta"}
EPSILON_AT_FIT_TIME = 0.0

METRICS = ["auc", "brier", "ece"]
DECIMALS = {"auc": 3, "brier": 4, "ece": 4}

# How each split is written in the captions.
SPLIT_NAMES = {"valid": "valid", "tu_transductive": "tu"}


def long_nT(raw):
    """The single n_T the run was fitted at, for captions when --nT is unset."""
    values = sorted(raw["nT"].dropna().unique())
    if len(values) != 1:
        raise ValueError(f"raw_results mixes nT={values}; pass --nT.")
    return int(values[0])


# ============================================================
# Loading
# ============================================================

def load_inputs(result_dir, suptrans_path, nT):
    raw_path = os.path.join(result_dir, "raw_results.csv")
    gate_path = os.path.join(result_dir, "gate_and_shapley_audit.csv")
    for path, flag in ((raw_path, "--result-dir"), (gate_path, "--result-dir"),
                       (suptrans_path, "--suptrans")):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path}\nNot found. Point {flag} at the directory the run "
                "actually wrote to; the default is RESULT_DIR of the pipeline.")

    raw = pd.read_csv(raw_path)
    gate = pd.read_csv(gate_path)
    sup = pd.read_csv(suptrans_path)

    if nT is not None:
        for name, frame in (("raw_results", raw), ("gate audit", gate),
                            ("suptrans", sup)):
            if "nT" in frame.columns:
                keep = frame["nT"] == nT
                if not keep.any():
                    raise ValueError(f"{name}: no rows at nT={nT}")
                frame.drop(frame.index[~keep], inplace=True)

    print(f"[load] raw_results: {len(raw)} rows, "
          f"{raw['repeat'].nunique()} repeats, "
          f"branches={sorted(raw['branch'].unique())}")
    print(f"[load] gate audit:  {len(gate)} rows")
    print(f"[load] suptrans:    {len(sup)} rows, "
          f"endpoints={sorted(sup['endpoint'].unique())}")
    return raw, gate, sup


# ============================================================
# The gate replay
# ============================================================

def epsilon0_map(gate):
    """epsilon0(branch) = log(d)/n_T, read off the audit. Both must be
    constant within a branch or the formula has no single value."""
    eps, rows = {}, []
    for branch, group in gate.groupby("branch"):
        d = sorted(group["d_total"].unique())
        n = sorted(group["nT"].unique())
        if len(d) != 1 or len(n) != 1:
            raise ValueError(f"{branch}: d_total={d}, nT={n}; both must be "
                             "constant within a branch.")
        eps[branch] = float(np.log(int(d[0])) / int(n[0]))
        rows.append({"branch": branch, "d": int(d[0]), "n_T": int(n[0]),
                     "epsilon0": eps[branch]})
    print("\n[epsilon0] log(d) / n_T")
    print(pd.DataFrame(rows).to_string(index=False))
    return eps


def replay(gate, eps):
    """Reselect a coalition per repeat at the given epsilon0."""
    if not isinstance(eps, dict):
        eps = {b: float(eps) for b in gate["branch"].unique()}
    out = []
    for row in gate.itertuples():
        e = eps[row.branch]
        losses = {"none": float(row.loss_none), "beta": float(row.loss_beta),
                  "gamma": float(row.loss_gamma), "both": float(row.loss_both)}
        cutoff = losses["none"] - e
        rho = int(losses["both"] <= cutoff)
        eligible = {c: v for c, v in losses.items()
                    if c == "none" or v <= cutoff}
        out.append({
            "repeat": int(row.repeat), "branch": row.branch, "rho": rho,
            "paper": "both" if rho else "none",
            "shapley": min(eligible, key=eligible.get),
        })
    return pd.DataFrame(out)


def verify_replay(gate, raw):
    """Replaying at the fitted epsilon0 must reproduce the stored gate rows.

    If it does not, the coalition map or the rule has drifted and no replayed
    number can be trusted, so this raises rather than warns.
    """
    decisions = replay(gate, EPSILON_AT_FIT_TIME)
    stored = gate.set_index(["repeat", "branch"])["rho"]
    got = decisions.set_index(["repeat", "branch"])["rho"].reindex(stored.index)
    if not (got == stored).all():
        raise AssertionError(
            f"replayed rho disagrees with the stored rho at epsilon0="
            f"{EPSILON_AT_FIT_TIME} for {int((got != stored).sum())} rows.")

    indexed = raw.set_index(["repeat", "branch", "endpoint", "evaluation_split"])
    checked = bad = 0
    for row in decisions.itertuples():
        for endpoint, rule in GATE_RULES.items():
            src = COALITION_TO_ENDPOINT[getattr(row, rule)]
            for split in raw["evaluation_split"].dropna().unique():
                ka = (row.repeat, row.branch, endpoint, split)
                kb = (row.repeat, row.branch, src, split)
                if ka not in indexed.index or kb not in indexed.index:
                    continue
                a, b = indexed.loc[ka], indexed.loc[kb]
                checked += 1
                if not all(np.isclose(float(a[m]), float(b[m]), rtol=0,
                                      atol=1e-9, equal_nan=True)
                           for m in METRICS if m in raw.columns):
                    bad += 1
                    print(f"  [MISMATCH] repeat {row.repeat} {row.branch} "
                          f"{endpoint} <- {src} ({split})")
    print(f"[selfcheck] {checked} endpoint comparisons, {bad} mismatched")
    if bad:
        raise AssertionError("the coalition-to-endpoint map does not reproduce "
                             "the stored gate endpoints.")


def gate_rows_at(raw, decisions, split):
    """Metric rows for the two gates, taken from the reselected endpoints."""
    src = raw[raw["evaluation_split"] == split]
    indexed = src.set_index(["repeat", "branch", "endpoint"])
    out = []
    for row in decisions.itertuples():
        for endpoint, rule in GATE_RULES.items():
            key = (row.repeat, row.branch,
                   COALITION_TO_ENDPOINT[getattr(row, rule)])
            if key not in indexed.index:
                continue
            picked = indexed.loc[key]
            rec = {"repeat": row.repeat, "branch": row.branch,
                   "endpoint": endpoint}
            rec.update({m: float(picked[m]) for m in METRICS})
            out.append(rec)
    return pd.DataFrame(out)


# ============================================================
# Assembling one long frame
# ============================================================

def build_long(raw, gate, sup, split, suptrans_endpoint, gate_source):
    keep = raw[raw["evaluation_split"] == split].copy()

    # 1. everything except the gates and suptrans, straight off raw_results
    passthrough = [e for e, _ in ENDPOINTS_ABOVE + ENDPOINTS_BELOW + [CEILING]
                   if e not in GATE_RULES and e != "suptrans"]
    frame = keep[keep["endpoint"].isin(passthrough)][
        ["repeat", "branch", "endpoint"] + METRICS].copy()

    # 2. the gates, replayed at log(d)/n_T
    if gate_source == "replay":
        verify_replay(gate, raw)
        decisions = replay(gate, epsilon0_map(gate))
        rates = (decisions.groupby("branch")["rho"].mean().round(3).to_dict())
        picks = (decisions.groupby(["branch", "shapley"]).size()
                 .unstack(fill_value=0))
        print(f"\n[gates] rho rate by branch: {rates}")
        print("[gates] Shapley coalition counts")
        print(picks.to_string())
        frame = pd.concat([frame, gate_rows_at(raw, decisions, split)],
                          ignore_index=True)
    else:
        print("\n[gates] WARNING: using the stored gate rows, which were fitted "
              "at epsilon0=0. These are NOT the reported operating point.")
        frame = pd.concat(
            [frame, keep[keep["endpoint"].isin(GATE_RULES)][
                ["repeat", "branch", "endpoint"] + METRICS]],
            ignore_index=True)

    # 3. suptrans, from the recompute
    new = sup[(sup["evaluation_split"] == split)
              & (sup["endpoint"] == suptrans_endpoint)][
        ["repeat", "branch"] + METRICS].copy()
    if new.empty:
        raise ValueError(f"no '{suptrans_endpoint}' rows on split={split}")
    new["endpoint"] = "suptrans"
    frame = pd.concat([frame, new], ignore_index=True)

    # 4. CNN, its own branch
    cnn = keep[(keep["branch"] == CNN_BRANCH)
               & (keep["endpoint"] == CNN_ENDPOINT)][
        ["repeat", "branch", "endpoint"] + METRICS]
    frame = pd.concat([frame[frame["branch"] != CNN_BRANCH], cnn],
                      ignore_index=True)

    dup = frame.duplicated(["repeat", "branch", "endpoint"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (repeat, branch, endpoint) rows.")
    return frame


def report_counts(long, expected=None):
    counts = long.groupby(["branch", "endpoint"]).size()
    if expected is None:
        expected = int(counts.max())
    short = counts[counts != expected]
    if len(short):
        print(f"\n[counts] WARNING: cells not at {expected} repeats")
        print(short.to_string())
    else:
        print(f"\n[counts] every cell has {expected} repeats")
    return expected


# ============================================================
# The two tables
# ============================================================

def levels(long):
    return (long.groupby(["branch", "endpoint"])[METRICS].mean()
            .reset_index())


def contrasts(long, alpha=0.05):
    """Paired mean and t-interval against target-only, within a branch."""
    rows = []
    for branch, group in long.groupby("branch"):
        if branch == CNN_BRANCH:
            continue
        wide = group.pivot(index="repeat", columns="endpoint", values=METRICS)
        if BASELINE not in wide[METRICS[0]].columns:
            raise ValueError(f"{branch}: no '{BASELINE}' rows to contrast against")
        for endpoint in group["endpoint"].unique():
            if endpoint in CONTRAST_EXCLUDED:
                continue
            rec = {"branch": branch, "endpoint": endpoint}
            for m in METRICS:
                d = (wide[m][endpoint] - wide[m][BASELINE]).dropna().values
                n = len(d)
                mu = float(d.mean())
                se = float(d.std(ddof=1)) / np.sqrt(n)
                half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
                rec[f"{m}_n"] = n
                rec[f"{m}_mean"] = mu
                rec[f"{m}_lo"] = mu - half
                rec[f"{m}_hi"] = mu + half
            rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================
# LaTeX
# ============================================================

def signed(v, nd):
    r = round(float(v), nd)
    sign = "$-$" if r < 0 else "$+$"
    return sign + f"{abs(r):.{nd}f}".lstrip("0")


def plain(v, nd):
    return f"{round(float(v), nd):.{nd}f}".lstrip("0")


def cell(row, metric):
    nd = DECIMALS[metric]
    return (f"{signed(row[f'{metric}_mean'], nd)}~"
            f"[{signed(row[f'{metric}_lo'], nd)}, "
            f"{signed(row[f'{metric}_hi'], nd)}]")


CONTRAST_PREAMBLE = r"""\begin{table}[H]
\centering
\caption{Paired contrasts against \textsc{target-only} on \textsc{%(split)s} at
$n_{\mathcal{T}}=%(nT)d$: mean over %(n)d repeats with a 95%% paired $t$-interval.
Within a repeat every row is scored on the same \textsc{%(split)s} rows, and the
five endpoints above the rule are built from the same $\hat{w}$, the same folds
and the same pair
$(\hat\theta^{(\mathcal{T})},\hat\theta^{(\mathrm{meta})})$, so a difference
between them is attributable to the coalition. The three comparators below the
rule share none of those stages, so their contrasts are evidence that the
construction helps or does not, never an attribution to a block.
\textsc{ceiling} and \textsc{cnn} are excluded and appear as levels only, in
Table~\ref{tab:app-endpoints-levels%(suffix)s}. Negative $\Delta$Brier and $\Delta$ECE
are improvements. The two gates read
$\epsilon_0=\log d/n_{\mathcal{T}}$ of Table~\ref{tab:dimensions}; no other row
reads $\epsilon_0$.}
\label{tab:app-endpoints-valid%(suffix)s}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccc}
\toprule
Branch & Endpoint & $\Delta$AUC & $\Delta$Brier & $\Delta$ECE \\
\midrule"""

LEVELS_PREAMBLE = r"""\begin{table}[t]
\centering
\caption{Endpoint levels on \textsc{%(split)s} at $n_{\mathcal{T}}=%(nT)d$, means over %(n)d repeats.
The six endpoints of Table~\ref{tab:endpoints} are above the rule, the external comparators of
Table~\ref{tab:comparators} below it. By Section~\ref{sec:eval-principles} these levels situate the
contrasts of Table~\ref{tab:app-endpoints-valid%(suffix)s} and carry no claim of their own: within a branch
they share \textsc{%(split)s} rows, but the level of any one of them is governed largely by which rows
fell into the held-out set. \textsc{ceiling} fits on labels every other row treats as absent and
\textsc{cnn} sits outside the linear model class of~\eqref{eq:outcome}; both are references.
\textsc{cnn} reads pixels and so has no branch.}
\label{tab:app-endpoints-levels%(suffix)s}
\small
\begin{tabular}{llccc}
\toprule
Branch & Endpoint & AUC & Brier & ECE \\
\midrule"""

CLOSER = r"""\bottomrule
\end{tabular}
\end{table}"""


def emit_contrasts(table, ctx):
    idx = table.set_index(["branch", "endpoint"])
    out = [CONTRAST_PREAMBLE % ctx]
    for bi, (branch, name) in enumerate(BRANCHES):
        if bi:
            out.append(r"\addlinespace")
        head = rf"\textsc{{{name}}}"
        first = True
        for endpoint, label in ENDPOINTS_ABOVE:
            if endpoint == BASELINE or (branch, endpoint) not in idx.index:
                continue
            row = idx.loc[(branch, endpoint)]
            lead = head if first else ""
            first = False
            out.append(f"{lead} & \\textsc{{{label}}} & "
                       f"{cell(row, 'auc')} & {cell(row, 'brier')} & "
                       f"{cell(row, 'ece')} \\\\")
        out.append(r"\cmidrule(l){2-5}")
        for endpoint, label in ENDPOINTS_BELOW:
            if (branch, endpoint) not in idx.index:
                continue
            row = idx.loc[(branch, endpoint)]
            out.append(f" & \\textsc{{{label}}} & "
                       f"{cell(row, 'auc')} & {cell(row, 'brier')} & "
                       f"{cell(row, 'ece')} \\\\")
    out.append(CLOSER)
    return "\n".join(out)


def emit_levels(table, ctx):
    idx = table.set_index(["branch", "endpoint"])

    def line(lead, label, row):
        return (f"{lead} & \\textsc{{{label}}} & {plain(row['auc'], 3)} & "
                f"{plain(row['brier'], 4)} & {plain(row['ece'], 4)} \\\\")

    out = [LEVELS_PREAMBLE % ctx]
    for bi, (branch, name) in enumerate(BRANCHES):
        if bi:
            out.append(r"\addlinespace")
        head = rf"\textsc{{{name}}}"
        first = True
        for endpoint, label in ENDPOINTS_ABOVE:
            if (branch, endpoint) not in idx.index:
                continue
            out.append(line(head if first else "", label,
                            idx.loc[(branch, endpoint)]))
            first = False
        out.append(r"\cmidrule(l){2-5}")
        for endpoint, label in ENDPOINTS_BELOW + [CEILING]:
            if (branch, endpoint) not in idx.index:
                continue
            out.append(line("", label, idx.loc[(branch, endpoint)]))
    key = (CNN_BRANCH, CNN_ENDPOINT)
    if key in idx.index:
        row = idx.loc[key]
        out.append(r"\midrule")
        out.append(rf"\textsc{{cnn}} & --- & {plain(row['auc'], 3)} & "
                   f"{plain(row['brier'], 4)} & {plain(row['ece'], 4)} \\\\")
    out.append(CLOSER)
    return "\n".join(out)


# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result-dir", default=RESULT_DIR,
                   help="holds raw_results.csv and gate_and_shapley_audit.csv "
                        f"(default: {RESULT_DIR})")
    p.add_argument("--suptrans", default=None,
                   help="recompute results CSV (default: <result-dir>/"
                        f"{SUPTRANS_SUBDIR}/{SUPTRANS_FILE})")
    p.add_argument("--split", default="valid")
    p.add_argument("--nT", type=int, default=200)
    p.add_argument("--out-dir", default=None,
                   help="write the two .tex files here "
                        "(default: <result-dir>/tables)")
    p.add_argument("--label-suffix", default=None,
                   help="bare tag appended to both \\label keys and to the "
                        ".tex filenames, so two arms can live in one document; "
                        "give it without a leading dash, e.g. nT400 "
                        "(default: empty at nT=200 on valid, else nT<value>)")
    p.add_argument("--suptrans-agg", action="store_true",
                   help="use the aggregated endpoint instead of the three-step one")
    p.add_argument("--gate-source", choices=["replay", "stored"], default="replay",
                   help="'stored' takes the gate rows as fitted at epsilon0=0; "
                        "diagnostic only, not the reported operating point")
    args = p.parse_args()

    suptrans_path = args.suptrans or os.path.join(
        args.result_dir, SUPTRANS_SUBDIR, SUPTRANS_FILE)
    out_dir = args.out_dir or os.path.join(args.result_dir, TABLE_SUBDIR)
    raw, gate, sup = load_inputs(args.result_dir, suptrans_path, args.nT)

    endpoint = "suptrans_agg" if args.suptrans_agg else "suptrans"
    long = build_long(raw, gate, sup, args.split, endpoint, args.gate_source)
    n_repeats = report_counts(long)

    lv = levels(long)
    ct = contrasts(long)

    print("\n[levels]")
    print(lv.round(4).to_string(index=False))
    print("\n[contrasts vs target-only]")
    show = ct[["branch", "endpoint"] + [f"{m}_mean" for m in METRICS]]
    print(show.round(4).to_string(index=False))

    nT = args.nT or long_nT(raw)
    tag = args.label_suffix
    if tag is None:
        parts = ([f"nT{nT}"] if nT != 200 else []) + \
                (["tu"] if args.split != "valid" else [])
        tag = "-".join(parts)
    tag = tag.strip("-_")
    ctx = {"n": n_repeats, "nT": nT,
           "split": SPLIT_NAMES.get(args.split, args.split),
           "suffix": f"-{tag}" if tag else ""}
    tex_contrasts = emit_contrasts(ct, ctx)
    tex_levels = emit_levels(lv, ctx)
    print("\n" + "=" * 72 + "\n" + tex_contrasts)
    print("\n" + "=" * 72 + "\n" + tex_levels)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = f"_{tag.replace('-', '_')}" if tag else ""
        for name, body in ((f"app_endpoints_valid{stem}.tex", tex_contrasts),
                           (f"app_endpoints_levels{stem}.tex", tex_levels)):
            with open(os.path.join(out_dir, name), "w") as fh:
                fh.write(body + "\n")
            print(f"\nwrote {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    sys.exit(main())
