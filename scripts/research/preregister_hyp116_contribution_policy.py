#!/usr/bin/env python3
"""HYP-116 — the fade slowed down to a contribution policy. Sealed before any policy path is computed.

Proposal (forwarded 2026-09-03): the one thing that replicated across 2020-2026 was 'the momentum side
after a shock is the wrong side'; at a session it cannot clear costs (HYP-114). Slowed to a rebalance /
contribution rule — add on the days the crowd is selling — the costs vanish. Prior evidence against:
HYP-114 found no next-session fade in 2016-2019 or on 20 other ETFs; HYP-113 found the largest
down-days carry no fade. This tests the policy form honestly on 2007-2026 with the GFC inside.

Not run at sealing. Operator decides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.incumbent.data import CORE  # noqa: E402

HYP_ID = "HYP-116"


def build_doc() -> dict:
    n = prereg.mined_total() + 14
    return {
        "id": HYP_ID, "slug": "shock_deferred_contribution_vs_dca",
        "name": "Shock-deferred contributions vs dollar-cost averaging into the EW-10 basket, 2007-2026",
        "status": "PREREGISTERED", "frozen_at": "2026-09-03T00:00:00Z", "family": "INCUMBENT-2026-09",
        "lineage": "HYP-111/114 fade (null out of regime); HYP-115 basket (FRAGILE) as the vehicle — the vehicle's quality is not what is tested here",
        "policies": {
            "DCA": "1 unit invested into the EW-10 basket (daily-rebalanced, research/incumbent) at the close of the first session of every month",
            "DEFERRED": ("the same 1 unit per month accrues in cash at 0% and the whole cash balance is invested at the close of the "
                         "session AFTER the first SPY down-shock (SPY close-to-close log return <= −p90 of trailing 252 |r|, t excluded); "
                         "if no down-shock for 126 sessions the balance is invested at the next month start (cap, declared)"),
            "why_next_session_close": "one-session lag is what a retail contributor can actually do after seeing the shock; conservative",
            "cash_at_zero": "conservative against DEFERRED",
        },
        "window": ["2007-06-01", "2026-07-16"],
        "data": "yfinance auto-adjusted closes (research/incumbent/data.py); SPY shock rule from the same series",
        "statistics": {
            "primary": "ratio of terminal wealth DEFERRED / DCA",
            "significance": ("stationary block bootstrap of the joint daily (basket return, SPY shock flag) sequence, L=20, 2000 draws, seed 42; "
                             "both policies re-run on each resampled path; 95% CI of the ratio"),
            "secondary": "share of rolling 5-year windows (monthly start) in which DEFERRED terminal wealth > DCA",
            "descriptive": ["average months of cash drag", "number of deployment events", "IRR of both", "same test with 60/40 as the vehicle"],
        },
        "verdict_ladder": {"POLICY_HOLDS": "ratio CI excludes 1 from above AND rolling share >= 0.6",
                           "POLICY_FAILS": "ratio CI excludes 1 from below OR rolling share <= 0.4",
                           "NO_DIFFERENCE": "otherwise"},
        "prior_expectation": "NO_DIFFERENCE",
        "priors": {"operator": {"prior": "not stated (forwarded proposal)"},
                   "claude": {"prior": "NO_DIFFERENCE to POLICY_FAILS — cash drag at 0% over months without a shock costs more than buying a −2% day gains; the 2016-2019 null says the dip has no extra return",
                              "most_likely_failure": "cash drag in 2013-2019 and 2023-2024"}},
        "frozen_parameters": {"percentile": 0.90, "trailing": 252, "cap_sessions": 126, "block_L": 20, "draws": 2000, "seed": 42, "n_trials": n},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across percentiles, caps, lags or vehicles"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-03 before any policy path was computed. The fade as a contribution policy. NOT RUN at sealing.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
