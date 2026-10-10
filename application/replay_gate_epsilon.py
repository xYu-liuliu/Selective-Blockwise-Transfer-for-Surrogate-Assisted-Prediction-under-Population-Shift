"""Apply the paper's dead zone epsilon0 = log(d) / n_T to the stored losses.

The pipeline and the negative control fit at epsilon0 = 0 and store the four
cross-fitted coalition losses of every repeat. epsilon0 changes only the two
decisions, not any fitted estimator: the losses, the four outcome models and
phi are the same at every threshold. This script applies epsilon0 per branch to
the stored losses and rebuilds Gate-Binary (code name paper_gate) and
Gate-Block (shapley_gate) from the outcome models of the coalitions the two
rules select, which is what a fit at that epsilon0 gives.

What it writes
--------------
* The reported Gate-Binary and Gate-Block rows and their paired contrasts
  against Target-Only, for the real arm and the permuted-source arm.
* The threshold replay on a uniform grid (separation_table): real and permuted
  acceptance counts for both rules, the Clopper-Pearson interval for the
  permuted binary rate, the discordant pairs of the binary rule, and exact
  two-sided McNemar p-values for both rules.
* The null-calibrated thresholds of both rules, as a diagnostic.
* At the applied thresholds (headline_table), the binary rule only: both
  acceptance counts, the Clopper-Pearson interval, the discordance and the
  McNemar p. The per-repeat decisions of both rules at the applied thresholds
  are saved for both arms (main_decisions_*.csv, null_decisions_*.csv). The
  coalition rule's interval, discordance and McNemar p at the applied
  thresholds are not computed by this script; they follow directly from
  those two files.

The threshold
-------------
For a logistic link, Cai et al. (2025, Supplement S.4) give no
application-computable cutoff. epsilon0 = 0 is allowed for the exact-sparse
linear model with identity link, which does not apply here, and the sufficient
condition used in the proof of their Theorem S2 involves the unknown target
projection thetabar, the error of the meta fit and restricted-strong-convexity
constants, none of which an application can evaluate. The paper therefore
prespecifies the schedule

    epsilon0 = log(d) / n_T,    d = p + q,

which takes the explicit log(d) / n_T prefactor of the RSC tolerance term and
fixes the multiplier at one. It is not claimed to satisfy that condition. d
differs across branches, because the image block keeps a different number of
PCA components (d = 130 for BioViL and 221 for SimCLR and Supervised in the
paper), and n_T across arms, so every branch and arm has its own value: .0243
and .0270 at n_T = 200, .0122 and .0135 at n_T = 400. The formula mode reads d
and n_T from the gate audit; it uses no labels, no losses and no evaluation
data, so no result can move it.

Why the negative control does not choose epsilon0
-------------------------------------------------
If epsilon0 were chosen so that the permuted acceptance rate falls below a
nominal level, reporting that rate would be circular: the number would be an
artefact of the choice rather than an observation. The null-calibrated
thresholds are still computed and printed, as the paper reports them, but as
a diagnostic. The threshold that drives the reported results is the formula.

What epsilon0 does not affect
-----------------------------
phi_beta, phi_gamma and phi_total (phi_X, phi_S and v(XS) in the paper) are
differences of the four coalition losses and are identical at every epsilon0.
Only the binary decision rho and the selected coalition c_hat move.

Self-check
----------
Before reporting anything the script replays both arms at the epsilon0 the runs
were fitted with and requires the rebuilt paper_gate and shapley_gate metrics
to reproduce the stored rows exactly. A mismatch means the coalition-to-endpoint
mapping is wrong, so the script stops rather than reporting the wrong coalition.

Usage
-----
    python -u replay_gate_epsilon.py --epsilon formula
    python -u replay_gate_epsilon.py --epsilon biovil=0.0243,ssl=0.0270,no_ssl=0.0270
    python -u replay_gate_epsilon.py --epsilon 0.02
    python -u replay_gate_epsilon.py --epsilon formula --main-only

The first form gives the paper's results. The n_T = 200 arm and its negative
control are read by default. For the n_T = 400 arm, point --result-dir at that
arm's directory (its negative control and output follow it), and use the
n_T = 400 grid noted at EPSILON_GRID. From the repository root:

    python -u application/replay_gate_epsilon.py --epsilon formula \\
        --result-dir results/pneumothorax_history_bow_nT400_valid200_strifle_shapley
"""

import os
import re
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from project_paths import NEGCTRL_SUFFIX, run_dir


# ============================================================
# Configuration
# ============================================================

# Default arm: the main run at n_T = 200, as named by the pipeline. The null arm
# defaults to the same directory plus NEGCTRL_SUFFIX, the output to a subfolder.
MAIN_RESULT_DIR = run_dir(200, 200)

# The epsilon0 both runs were executed at. The self-check replays this value
# and requires an exact reproduction of the stored gate endpoints.
EPSILON_AT_FIT_TIME = 0.0

# Grid for the threshold replay: the paper's grid for n_T = 200, denser between
# 0.005 and 0.05, where the decision rates move at that budget. At n_T = 400 the
# formula value halves, and the paper's n_T = 400 tables use a grid extended
# downwards instead: 0, .0005, .001, .0015, .002, .003, .004, .005, .0075, .01,
# .0125, .015, .02, .03, .05, .10, .20. Set EPSILON_GRID to that when replaying
# the n_T = 400 arm.
EPSILON_GRID = np.array([
    0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02,
    0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20,
])

# Nominal level for the null-only calibration. Reported as a diagnostic only.
NOMINAL_LEVEL = 0.10

# Significance level for the robustness interval.
ALPHA = 0.05

# Verified empirically against the stored gate endpoints; the self-check
# re-verifies it on every run rather than trusting this constant.
COALITION_TO_ENDPOINT = {
    "none": "to",
    "beta": "beta_only",
    "gamma": "gamma_only",
    "both": "corrected_meta",
}

DELTA_METRICS = ["auc", "pr_auc", "f1", "brier", "ece", "calibration_slope"]
SUMMARY_METRICS = ["auc", "pr_auc", "brier", "ece"]

BASELINE_ENDPOINT = "to"
GATE_ENDPOINTS = {"paper_gate": "paper", "shapley_gate": "shapley"}


# ============================================================
# Loading
# ============================================================

def load_arm(result_dir, name):
    """Load one arm's gate audit and per-endpoint metric rows."""
    gate_path = os.path.join(result_dir, "gate_and_shapley_audit.csv")
    raw_path = os.path.join(result_dir, "raw_results.csv")
    for path in (gate_path, raw_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: missing {path}")

    gate = pd.read_csv(gate_path)
    raw = pd.read_csv(raw_path)

    required = ["repeat", "branch", "loss_none", "loss_beta", "loss_gamma",
                "loss_both", "rho", "selected_endpoint_shapley"]
    missing = [c for c in required if c not in gate.columns]
    if missing:
        raise ValueError(f"{name}: gate audit lacks {missing}")

    print(f"[load] {name}: {len(gate)} gate rows, "
          f"{gate['repeat'].nunique()} repeats, "
          f"branches={sorted(gate['branch'].unique())}")
    return gate, raw


# ============================================================
# epsilon0 as a per-branch map
# ============================================================

def formula_epsilon(gate):
    """epsilon0(branch) = log(d_branch) / n_T, read from the gate audit.

    d and n_T are recorded per row and must be constant within a branch: d is
    the design dimension the branch was fitted at and n_T the labelled target
    budget. A violation means the audit mixes runs, in which case the formula
    has no single value and silently averaging would be wrong.
    """
    if "d_total" not in gate.columns or "nT" not in gate.columns:
        raise ValueError(
            "Formula mode needs d_total and nT in the gate audit. Pass explicit "
            "values instead, e.g. --epsilon biovil=0.0243,ssl=0.0270,no_ssl=0.0270"
        )
    records, eps_map = [], {}
    for branch, group in gate.groupby("branch"):
        d_values = sorted(group["d_total"].unique())
        n_values = sorted(group["nT"].unique())
        if len(d_values) != 1 or len(n_values) != 1:
            raise ValueError(
                f"{branch}: d_total={d_values}, nT={n_values}. Both must be "
                "constant within a branch for the formula to have one value."
            )
        d, n_t = int(d_values[0]), int(n_values[0])
        epsilon0 = float(np.log(d) / n_t)
        eps_map[branch] = epsilon0
        records.append({"branch": branch, "d": d, "n_T": n_t,
                        "log_d": float(np.log(d)), "epsilon0": epsilon0})
    return eps_map, pd.DataFrame(records)


def parse_epsilon(spec, gate):
    """Resolve --epsilon into a branch -> value map.

    Accepts 'formula', a single number applied to every branch, or a
    comma-separated list of branch=value pairs.
    """
    branches = sorted(gate["branch"].unique())
    spec = (spec or "").strip()

    if spec.lower() == "formula":
        eps_map, table = formula_epsilon(gate)
        print("\n[epsilon0] formula mode: log(d) / n_T")
        print(table.to_string(index=False))
        return eps_map, table

    if "=" in spec:
        eps_map = {}
        for piece in spec.split(","):
            if "=" not in piece:
                raise ValueError(f"Cannot parse epsilon fragment: {piece!r}")
            branch, value = piece.split("=", 1)
            eps_map[branch.strip()] = float(value)
        unknown = [b for b in eps_map if b not in branches]
        if unknown:
            raise ValueError(f"Unknown branch(es) in --epsilon: {unknown}. "
                             f"Available: {branches}")
        absent = [b for b in branches if b not in eps_map]
        if absent:
            raise ValueError(
                f"No epsilon0 given for {absent}. Give every branch a value, "
                "or pass a single number, or use 'formula'."
            )
        table = pd.DataFrame([{"branch": b, "epsilon0": v}
                              for b, v in sorted(eps_map.items())])
        print("\n[epsilon0] explicit per-branch values")
        print(table.to_string(index=False))
        return eps_map, table

    value = float(spec)
    eps_map = {b: value for b in branches}
    table = pd.DataFrame([{"branch": b, "epsilon0": value} for b in branches])
    print(f"\n[epsilon0] uniform value {value} applied to every branch")
    return eps_map, table


def epsilon_tag(eps_map):
    """Filesystem-safe tag describing the epsilon0 map."""
    values = set(round(v, 6) for v in eps_map.values())
    if len(values) == 1:
        body = f"{values.pop():g}"
    else:
        body = "_".join(f"{b}{eps_map[b]:g}" for b in sorted(eps_map))
    return "eps" + re.sub(r"[^0-9A-Za-z._-]", "", body.replace(".", "p"))


# ============================================================
# The decision rules
# ============================================================

def recompute_decision(loss_none, loss_beta, loss_gamma, loss_both, epsilon0):
    """Replay both rules at one epsilon0."""
    losses = {"none": float(loss_none), "beta": float(loss_beta),
              "gamma": float(loss_gamma), "both": float(loss_both)}
    cutoff = losses["none"] - float(epsilon0)
    rho = int(losses["both"] <= cutoff)
    eligible = {c: v for c, v in losses.items() if c == "none" or v <= cutoff}
    c_shapley = min(eligible, key=eligible.get)
    # The binary rule (Gate-Binary) keeps XS when rho = 1 and falls back to the
    # target-only coalition otherwise.
    c_paper = "both" if rho else "none"
    return rho, c_paper, c_shapley


def replay_arm(gate, eps_map):
    """Per-repeat decisions, each row thresholded at its own branch's value.

    eps_map may be a dict or a scalar; a scalar is broadcast to every branch so
    the grid sweep and the self-check can share this function.
    """
    if not isinstance(eps_map, dict):
        eps_map = {b: float(eps_map) for b in gate["branch"].unique()}
    records = []
    for row in gate.itertuples():
        if row.branch not in eps_map:
            continue
        epsilon0 = eps_map[row.branch]
        rho, c_paper, c_shapley = recompute_decision(
            row.loss_none, row.loss_beta, row.loss_gamma, row.loss_both, epsilon0
        )
        records.append({
            "repeat": int(row.repeat), "branch": row.branch,
            "epsilon0": float(epsilon0), "rho": rho,
            "c_paper": c_paper, "c_shapley": c_shapley,
            "transfer_margin": float(row.loss_none - row.loss_both),
        })
    return pd.DataFrame(records)


# ============================================================
# Self-check
# ============================================================

def verify_mapping(gate, raw, arm_name, epsilon0=EPSILON_AT_FIT_TIME):
    """Rebuild the gate endpoints at the fitted epsilon0 and demand a match."""
    decisions = replay_arm(gate, epsilon0)

    stored_rho = gate.set_index(["repeat", "branch"])["rho"]
    replayed_rho = decisions.set_index(["repeat", "branch"])["rho"]
    aligned = replayed_rho.reindex(stored_rho.index)
    if not (aligned == stored_rho).all():
        bad = stored_rho[aligned != stored_rho]
        raise AssertionError(
            f"{arm_name}: replayed rho disagrees with the stored rho at the "
            f"fitted epsilon0={epsilon0} for {len(bad)} rows. Either "
            f"EPSILON_AT_FIT_TIME is wrong or the decision rule changed.\n"
            f"{bad.head()}"
        )

    if "selected_endpoint_shapley" in gate.columns:
        stored_c = gate.set_index(["repeat", "branch"])["selected_endpoint_shapley"]
        replayed_c = decisions.set_index(["repeat", "branch"])["c_shapley"]
        agree = (replayed_c.reindex(stored_c.index) == stored_c)
        if not agree.all():
            raise AssertionError(
                f"{arm_name}: replayed coalition-rule selection disagrees with the "
                f"stored selection for {int((~agree).sum())} rows."
            )

    indexed = raw.set_index(["repeat", "branch", "endpoint", "evaluation_split"])
    checked = mismatched = skipped = 0
    for row in decisions.itertuples():
        for gate_endpoint, rule in GATE_ENDPOINTS.items():
            coalition = row.c_paper if rule == "paper" else row.c_shapley
            source_endpoint = COALITION_TO_ENDPOINT[coalition]
            for split in raw["evaluation_split"].dropna().unique():
                key_gate = (row.repeat, row.branch, gate_endpoint, split)
                key_src = (row.repeat, row.branch, source_endpoint, split)
                if key_gate not in indexed.index or key_src not in indexed.index:
                    skipped += 1
                    continue
                a, b = indexed.loc[key_gate], indexed.loc[key_src]
                same = all(
                    np.isclose(float(a[m]), float(b[m]), rtol=0, atol=1e-9,
                               equal_nan=True)
                    for m in ("auc", "brier") if m in raw.columns
                )
                checked += 1
                mismatched += (not same)
                if not same:
                    print(f"  [MISMATCH] repeat {row.repeat} {row.branch} "
                          f"{gate_endpoint} <- {source_endpoint} ({split}): "
                          f"auc {float(a['auc']):.8f} vs {float(b['auc']):.8f}")

    print(f"[selfcheck] {arm_name}: {checked} endpoint comparisons, "
          f"{mismatched} mismatched, {skipped} skipped")
    if mismatched:
        raise AssertionError(
            f"{arm_name}: the coalition-to-endpoint mapping does not reproduce "
            "the stored gate endpoints. Fix COALITION_TO_ENDPOINT before "
            "trusting any replayed number."
        )
    return True


# ============================================================
# Separation between the arms
# ============================================================

def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    low = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def mcnemar(real, null):
    """Exact McNemar for paired binary decisions, one- and two-sided.

    The arms sit on identical patient splits repeat by repeat, so the decisions
    are paired and only the discordant repeats carry information. Concordant
    repeats are conditioned out: a repeat where both arms fire says nothing
    about whether the gate can tell the arms apart.
    """
    real = np.asarray(real, dtype=int)
    null = np.asarray(null, dtype=int)
    b = int(np.sum((real == 1) & (null == 0)))
    c = int(np.sum((real == 0) & (null == 1)))
    if b + c == 0:
        return b, c, float("nan"), float("nan")
    one = float(stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue)
    two = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return b, c, one, two


def separation_table(main_gate, null_gate, grid=EPSILON_GRID):
    """Real and null decision rates side by side across a uniform grid.

    The sweep is deliberately uniform across branches: it answers "would any
    threshold overturn the conclusion", which is a question about the whole
    range, not about the per-branch operating point. The paper's
    threshold-sensitivity counts and its McNemar p-values across the grid come
    from this table (mcnemar_p for the binary rule, mcnemar_p_selection for
    the coalition rule).
    """
    records = []
    branches = sorted(set(main_gate["branch"]) | set(null_gate["branch"]))
    for epsilon0 in grid:
        main_dec = replay_arm(main_gate, float(epsilon0))
        null_dec = replay_arm(null_gate, float(epsilon0))
        for branch in branches:
            m = main_dec[main_dec["branch"] == branch].set_index("repeat")
            n = null_dec[null_dec["branch"] == branch].set_index("repeat")
            record = {"branch": branch, "epsilon0": float(epsilon0)}

            if len(m):
                record["n_real"] = int(len(m))
                record["real_rho_rate"] = float(m["rho"].mean())
                record["real_select_rate"] = float((m["c_shapley"] != "none").mean())
            if len(n):
                k = int(n["rho"].sum())
                low, high = clopper_pearson(k, len(n))
                record.update({
                    "n_null": int(len(n)),
                    "null_rho_rate": float(k / len(n)),
                    "null_rho_ci_low": low, "null_rho_ci_high": high,
                    "null_select_rate": float((n["c_shapley"] != "none").mean()),
                })

            if len(m) and len(n):
                shared = m.index.intersection(n.index)
                if len(shared):
                    b, c, p1, p2 = mcnemar(m.loc[shared, "rho"],
                                           n.loc[shared, "rho"])
                    record.update({
                        "n_paired": int(len(shared)),
                        "discordant_real_only": b, "discordant_null_only": c,
                        "mcnemar_p_one_sided": p1, "mcnemar_p": p2,
                    })
                    sb, sc, sp1, sp2 = mcnemar(
                        (m.loc[shared, "c_shapley"] != "none").astype(int),
                        (n.loc[shared, "c_shapley"] != "none").astype(int),
                    )
                    record["mcnemar_p_selection"] = sp2
                record["separation_gate"] = (record["real_rho_rate"]
                                             - record["null_rho_rate"])
            records.append(record)
    return pd.DataFrame(records)


def robustness_interval(table, alpha=ALPHA):
    """Longest run of grid values, starting at zero, with two-sided p < alpha.

    Binary rule only. Stops at the first grid point that fails, so the
    interval is contiguous rather than a union of lucky points. Reported per
    branch, with the smallest upper end as the interval common to all; the
    paper reports the p-values across the grid themselves.
    """
    records = []
    for branch, group in table.groupby("branch"):
        group = group.sort_values("epsilon0")
        if "mcnemar_p" not in group.columns:
            continue
        upper, worst = np.nan, np.nan
        for row in group.itertuples():
            p = getattr(row, "mcnemar_p", np.nan)
            if not np.isfinite(p) or p >= alpha:
                break
            upper = float(row.epsilon0)
            worst = p if not np.isfinite(worst) else max(worst, p)
        records.append({"branch": branch, "alpha": alpha,
                        "robust_from": 0.0, "robust_to": upper,
                        "max_two_sided_p": worst})
    frame = pd.DataFrame(records)
    if not frame.empty and frame["robust_to"].notna().any():
        frame.attrs["common_upper"] = float(frame["robust_to"].min())
    return frame


def choose_epsilon_from_null(null_gate, level=NOMINAL_LEVEL, grid=EPSILON_GRID):
    """Smallest grid epsilon0 whose null acceptance rate is at or below the level.

    The paper's null-calibrated thresholds (level 0.10), for each rule. A
    diagnostic only: choosing epsilon0 this way would make the reported null
    acceptance rate an artefact of the choice rather than an observation, so
    the reported threshold comes from the formula instead.
    """
    records = []
    for branch, group in null_gate.groupby("branch"):
        n_repeats = int(group["repeat"].nunique())
        gate_pick = select_pick = np.nan
        for epsilon0 in grid:
            dec = replay_arm(group, float(epsilon0))
            if np.isnan(gate_pick) and float(dec["rho"].mean()) <= level:
                gate_pick = float(epsilon0)
            if np.isnan(select_pick) and \
                    float((dec["c_shapley"] != "none").mean()) <= level:
                select_pick = float(epsilon0)
        records.append({
            "branch": branch, "n_null_repeats": n_repeats,
            "nominal_level": float(level),
            "level_resolvable": bool(level >= 1.0 / max(n_repeats, 1)),
            "finest_resolvable_level": 1.0 / max(n_repeats, 1),
            "epsilon0_for_gate": gate_pick,
            "epsilon0_for_shapley_selection": select_pick,
        })
    return pd.DataFrame(records)


# ============================================================
# Rebuilding the downstream metrics
# ============================================================

def rebuild_gate_rows(raw, decisions):
    """Gate-Binary (paper_gate) and Gate-Block (shapley_gate) rows at the
    replayed epsilon0.

    The outcome model of every coalition is already fitted and scored in
    raw_results.csv, so each gate row is the stored row of the coalition its
    rule selects, relabelled.
    """
    indexed = raw.set_index(["repeat", "branch", "endpoint", "evaluation_split"])
    splits = list(raw["evaluation_split"].dropna().unique())
    rebuilt, missing = [], 0
    for row in decisions.itertuples():
        for gate_endpoint, rule in GATE_ENDPOINTS.items():
            coalition = row.c_paper if rule == "paper" else row.c_shapley
            source_endpoint = COALITION_TO_ENDPOINT[coalition]
            for split in splits:
                key = (row.repeat, row.branch, source_endpoint, split)
                if key not in indexed.index:
                    missing += 1
                    continue
                record = indexed.loc[key].to_dict()
                record.update({
                    "repeat": row.repeat, "branch": row.branch,
                    "endpoint": gate_endpoint, "evaluation_split": split,
                    "model": f"{row.branch}_{gate_endpoint}",
                    "epsilon0": row.epsilon0,
                    "selected_coalition": coalition,
                    "source_endpoint": source_endpoint,
                })
                rebuilt.append(record)
    if missing:
        print(f"  [warn] {missing} coalition rows absent from raw_results")
    return pd.DataFrame(rebuilt)


def paired_differences(raw, rebuilt):
    """Per-repeat differences against target-only, for the rebuilt endpoints."""
    baseline = raw[raw["endpoint"] == BASELINE_ENDPOINT].set_index(
        ["repeat", "branch", "evaluation_split"]
    )
    records = []
    for row in rebuilt.itertuples():
        key = (row.repeat, row.branch, row.evaluation_split)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        record = {
            "repeat": row.repeat, "branch": row.branch,
            "evaluation_split": row.evaluation_split, "endpoint": row.endpoint,
            "model": f"{row.branch}_{row.endpoint}",
            "epsilon0": row.epsilon0,
            "selected_coalition": row.selected_coalition,
        }
        for metric in DELTA_METRICS:
            if metric in rebuilt.columns and metric in base.index:
                record[f"delta_{metric}"] = (float(getattr(row, metric))
                                             - float(base[metric]))
        records.append(record)
    return pd.DataFrame(records)


def summarize_paired(paired):
    """Mean, Student t interval, number of positive differences and a
    two-sided Wilcoxon p.

    The mean and the t interval match the pipeline's summary and are what the
    paper reports. The Wilcoxon test here drops zero differences, whereas the
    pipeline's zsplit keeps them, so the two p-values can differ for the gate
    endpoints, which equal Target-Only in every repeat the rule declines.
    """
    records = []
    for (branch, endpoint, split), group in paired.groupby(
            ["branch", "endpoint", "evaluation_split"]):
        record = {"branch": branch, "endpoint": endpoint,
                  "evaluation_split": split, "n_repeats": int(len(group))}
        if "epsilon0" in group.columns:
            record["epsilon0"] = float(group["epsilon0"].iloc[0])
        for metric in SUMMARY_METRICS:
            column = f"delta_{metric}"
            if column not in group.columns:
                continue
            values = group[column].dropna().to_numpy(dtype=float)
            n = len(values)
            record[f"{column}_mean"] = float(values.mean()) if n else np.nan
            record[f"{column}_n_positive"] = int((values > 0).sum())
            if n >= 2 and values.std(ddof=1) > 0:
                half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
                record[f"{column}_ci_low"] = float(values.mean() - half)
                record[f"{column}_ci_high"] = float(values.mean() + half)
                record[f"{column}_wilcoxon_p"] = float(stats.wilcoxon(values).pvalue)
            else:
                record[f"{column}_ci_low"] = np.nan
                record[f"{column}_ci_high"] = np.nan
                record[f"{column}_wilcoxon_p"] = np.nan
        records.append(record)
    return pd.DataFrame(records)


# ============================================================
# Driver
# ============================================================

def replay_at(gate, raw, eps_map, arm_name):
    decisions = replay_arm(gate, eps_map)
    rebuilt = rebuild_gate_rows(raw, decisions)
    paired = paired_differences(raw, rebuilt)
    summary = summarize_paired(paired)
    coalitions = (decisions.groupby(["branch", "c_shapley"]).size()
                  .rename("n_repeats").reset_index())
    rates = decisions.groupby("branch").agg(
        epsilon0=("epsilon0", "first"),
        n_repeats=("rho", "size"), rho_rate=("rho", "mean"),
        c_none_rate=("c_shapley", lambda s: float((s == "none").mean())),
        margin_mean=("transfer_margin", "mean"),
    ).reset_index()
    rates.insert(0, "arm", arm_name)
    return {"decisions": decisions, "rebuilt": rebuilt, "paired": paired,
            "summary": summary, "coalitions": coalitions, "rates": rates}


def headline_table(main_out, null_out, eps_table):
    """Real against permuted acceptance at the applied thresholds, binary rule.

    Per branch: real and permuted acceptance counts, the Clopper-Pearson
    interval for the permuted rate, the discordance (real-only : permuted-only)
    and the exact two-sided McNemar p.
    """
    records = []
    main_dec, null_dec = main_out["decisions"], null_out["decisions"]
    for branch in sorted(main_dec["branch"].unique()):
        m = main_dec[main_dec["branch"] == branch].set_index("repeat")
        n = null_dec[null_dec["branch"] == branch].set_index("repeat")
        record = {"branch": branch, "epsilon0": float(m["epsilon0"].iloc[0])}
        info = eps_table[eps_table["branch"] == branch]
        for column in ("d", "n_T"):
            if column in eps_table.columns and len(info):
                record[column] = info[column].iloc[0]
        record["real_rho"] = f"{int(m['rho'].sum())}/{len(m)}"
        if len(n):
            k = int(n["rho"].sum())
            low, high = clopper_pearson(k, len(n))
            record["null_rho"] = f"{k}/{len(n)}"
            record["null_rate_ci"] = f"[{low:.3f}, {high:.3f}]"
            shared = m.index.intersection(n.index)
            b, c, p1, p2 = mcnemar(m.loc[shared, "rho"], n.loc[shared, "rho"])
            record["discordant"] = f"{b}:{c}"
            record["mcnemar_p_two_sided"] = p2
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Replay the STRIFLE gate at a per-branch epsilon0."
    )
    parser.add_argument(
        "--epsilon", default="formula",
        help=("'formula' for log(d)/n_T per branch (the paper), a single number "
              "for a uniform threshold, or "
              "'biovil=0.0243,ssl=0.0270,no_ssl=0.0270'."),
    )
    parser.add_argument("--level", type=float, default=NOMINAL_LEVEL,
                        help="Nominal level for the null-only diagnostic.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Significance level for the robustness interval.")
    parser.add_argument("--main-only", action="store_true",
                        help="Skip the null arm.")
    parser.add_argument("--result-dir", default=MAIN_RESULT_DIR,
                        help="RESULT_DIR of the main run (default: the n_T = 200 "
                             f"arm, {MAIN_RESULT_DIR}).")
    parser.add_argument("--null-dir", default=None,
                        help="Negative-control directory (default: "
                             f"<result-dir>{NEGCTRL_SUFFIX}).")
    parser.add_argument("--out-dir", default=None,
                        help="Default: <result-dir>/epsilon_replay.")
    args = parser.parse_args()
    args.null_dir = args.null_dir or args.result_dir + NEGCTRL_SUFFIX
    args.out_dir = args.out_dir or os.path.join(args.result_dir, "epsilon_replay")

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 88)
    print("EPSILON REPLAY: re-thresholding the gate without refitting")
    print("=" * 88)
    print(f"main arm : {args.result_dir}")
    print(f"null arm : {'(skipped)' if args.main_only else args.null_dir}")
    print(f"out dir  : {args.out_dir}")

    main_gate, main_raw = load_arm(args.result_dir, "main")
    verify_mapping(main_gate, main_raw, "main")

    null_gate = null_raw = None
    if not args.main_only:
        try:
            null_gate, null_raw = load_arm(args.null_dir, "null")
            verify_mapping(null_gate, null_raw, "null")
        except FileNotFoundError as exc:
            print(f"[load] null arm unavailable ({exc}); continuing main-only")
            null_gate = null_raw = None

    eps_map, eps_table = parse_epsilon(args.epsilon, main_gate)
    tag = epsilon_tag(eps_map)
    eps_table.to_csv(os.path.join(args.out_dir, f"epsilon_applied_{tag}.csv"),
                     index=False)

    # ---- 1. robustness sweep --------------------------------------------
    if null_gate is not None:
        table = separation_table(main_gate, null_gate)
        table.to_csv(os.path.join(args.out_dir, "epsilon_separation.csv"),
                     index=False)
        print("\n[1] Real and null decision rates across the grid:")
        for branch, group in table.groupby("branch"):
            columns = [c for c in ["epsilon0", "real_rho_rate", "null_rho_rate",
                                   "discordant_real_only", "discordant_null_only",
                                   "mcnemar_p", "mcnemar_p_selection"]
                       if c in group.columns]
            print(f"\n    {branch}:")
            print(group[columns].to_string(index=False))

        robust = robustness_interval(table, alpha=args.alpha)
        robust.to_csv(os.path.join(args.out_dir, "epsilon_robustness.csv"),
                      index=False)
        print(f"\n[2] Robustness interval (two-sided McNemar p < {args.alpha}):")
        print(robust.to_string(index=False))
        if "common_upper" in robust.attrs:
            upper = robust.attrs["common_upper"]
            worst = robust["max_two_sided_p"].max()
            print(f"\n    Holds for every branch on [0, {upper:g}]; "
                  f"largest two-sided p in that range = {worst:.2e}")
            for branch, epsilon0 in sorted(eps_map.items()):
                inside = "inside" if epsilon0 <= upper else "OUTSIDE"
                print(f"      {branch}: epsilon0={epsilon0:.4f} is {inside} "
                      "the robustness interval")

        chosen = choose_epsilon_from_null(null_gate, level=args.level)
        chosen.to_csv(os.path.join(args.out_dir, "epsilon_null_diagnostic.csv"),
                      index=False)
        print(f"\n[3] Null-only calibration at level {args.level} "
              "(diagnostic; NOT the reported threshold):")
        print(chosen.to_string(index=False))
        if not chosen["level_resolvable"].all():
            print("    [WARN] the level is finer than 1/R for some branch, so "
                  "the observed null rate cannot resolve it.")

    # ---- 2. rebuild the endpoints ---------------------------------------
    print(f"\n[4] Rebuilding endpoints at {dict(sorted(eps_map.items()))}")
    main_out = replay_at(main_gate, main_raw, eps_map, "main")
    print("\n    Main-arm gate behaviour:")
    print(main_out["rates"].to_string(index=False))
    print("\n    Main-arm selected coalition:")
    print(main_out["coalitions"].to_string(index=False))

    for name, frame in (("decisions", main_out["decisions"]),
                        ("raw_rebuilt", main_out["rebuilt"]),
                        ("paired_differences", main_out["paired"]),
                        ("summary_paired", main_out["summary"])):
        frame.to_csv(os.path.join(args.out_dir, f"main_{name}_{tag}.csv"),
                     index=False)

    columns = ["branch", "endpoint", "epsilon0", "n_repeats", "delta_auc_mean",
               "delta_auc_ci_low", "delta_auc_ci_high", "delta_auc_wilcoxon_p"]
    view = main_out["summary"]
    view = view[view["evaluation_split"] == "valid"]
    print("\n[5] Main arm, paired AUC difference against target-only (VALID):")
    print(view[[c for c in columns if c in view.columns]].to_string(index=False))

    if null_gate is not None:
        null_out = replay_at(null_gate, null_raw, eps_map, "null")
        for name, frame in (("decisions", null_out["decisions"]),
                            ("paired_differences", null_out["paired"]),
                            ("summary_paired", null_out["summary"])):
            frame.to_csv(os.path.join(args.out_dir, f"null_{name}_{tag}.csv"),
                         index=False)
        print("\n    Null-arm gate behaviour at the same thresholds:")
        print(null_out["rates"].to_string(index=False))

        null_view = null_out["summary"]
        null_view = null_view[null_view["evaluation_split"] == "valid"]
        print("\n[6] Null arm, paired AUC difference against target-only (VALID):")
        print(null_view[[c for c in columns if c in null_view.columns]]
              .to_string(index=False))
        print("    Under the null these should sit at or below zero. A "
              "confidently positive delta would be a leakage finding.")

        headline = headline_table(main_out, null_out, eps_table)
        headline.to_csv(os.path.join(args.out_dir, f"headline_{tag}.csv"),
                        index=False)
        print("\n[7] Headline table:")
        print(headline.to_string(index=False))

    with open(os.path.join(args.out_dir, f"replay_config_{tag}.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "epsilon0_applied": {b: float(v) for b, v in eps_map.items()},
            "epsilon_spec": args.epsilon,
            "epsilon0_at_fit_time": EPSILON_AT_FIT_TIME,
            "alpha": float(args.alpha),
            "nominal_level_diagnostic": float(args.level),
            "threshold_source": (
                "log(d)/n_T, pre-specified from d and n_T only"
                if args.epsilon.strip().lower() == "formula"
                else "user-supplied"
            ),
            "coalition_to_endpoint": COALITION_TO_ENDPOINT,
            "grid": [float(v) for v in EPSILON_GRID],
            "note": ("phi_beta, phi_gamma and phi_total are differences of the "
                     "four coalition losses and do not depend on epsilon0"),
        }, handle, indent=2)

    print(f"\nArtefacts written to: {args.out_dir}  (tag: {tag})")
    print("=" * 88)


if __name__ == "__main__":
    main()
