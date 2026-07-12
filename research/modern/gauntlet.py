"""P4: the pre-registered gauntlet + verdict (HYP-090).

Criteria (locked verbatim in the prereg — verdict_criteria):
 CONFIRMED iff some run R in {A1,A2}x{90,180,365} passes ALL of
  (1) block-bootstrap one-sided p < 0.05 vs A0 AND survives BH across the 6 runs
  (2) Sharpe(R) > 95th pct of the 500-seed A3 envelope for its window
  (3) deflated Sharpe > 0 at n_trials = 5,775
  (4) per-year non-degrade 2017-2025: Sharpe_R(y) >= Sharpe_A0(y) - 0.05
  (5) switching-costed R still beats A0 (point estimate)
  and the reconcile band held.
 NOT_ROBUST if (1) passes somewhere but (2)/(3)/(5) fail there, or (4) fails.
 NOT_SIGNIFICANT if no run passes (1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.modern._lib import OUT_DIR, block_bootstrap_sharpe_diff_p, daily_sharpe

NON_DEGRADE_TOL = 0.05
FULL_YEARS = list(range(2017, 2026))
DSR_PRIMARY_N = 5775
DSR_SECONDARY = [6, 17325]


def per_year_sharpes(returns: np.ndarray, eval_dates) -> dict:
    years = eval_dates.year
    return {int(y): daily_sharpe(returns[years == y]) for y in FULL_YEARS}


def adjudicate(runs: list, a0: np.ndarray, eval_dates, placebo_p95: dict,
               reconcile_in_band: bool, n_boot: int = 10_000) -> dict:
    from sovereign.discovery.gate import benjamini_hochberg, deflated_sharpe_ratio

    a0_sharpe = daily_sharpe(a0)
    a0_years = per_year_sharpes(a0, eval_dates)

    rows = []
    for run in runs:
        r = run["returns"]
        s = daily_sharpe(r)
        boot = block_bootstrap_sharpe_diff_p(r, a0, n_boot=n_boot)
        dsr, _ = deflated_sharpe_ratio(s, DSR_PRIMARY_N)
        dsr_sec = {str(n): round(deflated_sharpe_ratio(s, n)[0], 4) for n in DSR_SECONDARY}
        ry = per_year_sharpes(r, eval_dates)
        c4 = all(ry[y] >= a0_years[y] - NON_DEGRADE_TOL for y in FULL_YEARS)
        s_costed = daily_sharpe(run["returns_costed"])
        rows.append({
            "run": f"{run['arm']}_W{run['window']}", "arm": run["arm"], "window": run["window"],
            "sharpe": round(s, 4), "sharpe_costed": round(s_costed, 4),
            "a0_sharpe": round(a0_sharpe, 4),
            "p_vs_a0": boot["p_one_sided"], "boot": boot,
            "placebo_p95": round(float(placebo_p95[run["window"]]), 4),
            "c1_p_lt_05": boot["p_one_sided"] < 0.05,
            "c2_beats_placebo": s > placebo_p95[run["window"]],
            "c3_dsr_positive": dsr > 0, "dsr": round(dsr, 4), "dsr_secondary": dsr_sec,
            "c4_per_year_nondegrade": c4,
            "per_year": {str(y): round(ry[y], 3) for y in FULL_YEARS},
            "c5_costed_beats_a0": s_costed > a0_sharpe,
            "n_switches": run["n_switches"],
        })

    bh = benjamini_hochberg([row["p_vs_a0"] for row in rows], alpha=0.05)
    for row, ok in zip(rows, bh):
        row["c1_bh_survives"] = bool(ok)
        row["c1"] = row["c1_p_lt_05"] and bool(ok)
        row["all_pass"] = all(row[c] for c in
                              ("c1", "c2_beats_placebo", "c3_dsr_positive",
                               "c4_per_year_nondegrade", "c5_costed_beats_a0"))

    any_c1 = any(row["c1"] for row in rows)
    confirmed = reconcile_in_band and any(row["all_pass"] for row in rows)
    if confirmed:
        verdict = "CONFIRMED"
    elif any_c1:
        verdict = "NOT_ROBUST"
    else:
        verdict = "NOT_SIGNIFICANT"

    best = max(rows, key=lambda r: r["sharpe"])
    return {"verdict": verdict, "reconcile_in_band": reconcile_in_band,
            "a0_sharpe": round(a0_sharpe, 4),
            "a0_per_year": {str(y): round(a0_years[y], 3) for y in FULL_YEARS},
            "best_run": best["run"], "rows": rows,
            "primary_p_min": min(r["p_vs_a0"] for r in rows)}
