#!/usr/bin/env python3
"""HYP-111a — post-shock intraday retrace-then-continuation, FREE-TIER WINDOW. Sealed before any
event-session minute bar is read.

Why the 'a': research/HYP-111_SCOPE.md froze a 2020-2026 window and a sealed data-gate rule.
The gate failed (ThetaData STOCK.FREE first-access date 2023-06-01, probe.json). The operator
chose to run the scoped hypothesis on the window the tier serves, as a SEPARATELY REGISTERED
pilot with the power cut stated. Every other definition is byte-for-byte the scoped one.

  .venv313/bin/python scripts/research/preregister_hyp111a_postshock_intraday.py --write|--verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-111a"
PROBE_DATES = ["2020-03-16", "2021-06-15", "2022-06-13", "2024-01-03", "2026-08-03"]


def build_doc() -> dict:
    n_trials = prereg.mined_total() + 4          # HYP-109, regime test, HYP-110, this
    return {
        "id": HYP_ID, "slug": "postshock_intraday_retrace_continuation_free_window",
        "name": "Post-shock intraday retrace-then-continuation (ten ETFs, 1-min, 2023-06 → 2026-07 pilot)",
        "status": "PREREGISTERED", "frozen_at": "2026-09-02T00:00:00Z",
        "family": "POSTSHOCK-INTRADAY-2026-09", "parent_scope": "research/HYP-111_SCOPE.md (commit 0925bd9)",
        "why_a": ("scoped window 2020-2026 untestable on STOCK.FREE (first access 2023-06-01, "
                  "data/research/hyp111/probe.json); operator-instructed pilot on the served window. "
                  "Does NOT answer the scoped question; a CONFIRMED here is a reason to buy the "
                  "STANDARD tier and run the scoped test, not a result about 2020-2022."),
        "thesis": ("After a top-decile |return| day, session t+1 shows a retrace against the shock to "
                   "0.382 of the shock-day range followed by a reclaim of the shock-day close and "
                   "continuation in the shock direction; trading that path beats naive continuation."),
        "instrument_set": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"],
        "data": {
            "daily": "data/cache/daily_universe/<SYM>.parquet (shock rule, levels, C1, C5)",
            "intraday": "ThetaData v2 /v2/hist/stock/ohlc ivl=60000 rth=true on 127.0.0.1:25510, cached data/cache/theta_1m/",
            "entitlement": "STOCK.FREE, first access 2023-06-01 (probe.json)",
            "window_t1": ["2023-06-02", "2026-07-16"],
            "probe_dates_excluded": PROBE_DATES,
            "abort": "any instrument with < 80% of event sessions having >= 370 RTH bars, or pooled triggered n < 100",
        },
        "power_statement": ("798 instrument-events on 329 unique dates (counted from the daily cache, no "
                            "outcomes read). At an assumed 30-50% trigger rate: ~240-400 trades on ~120-180 "
                            "independent dates; SE(mean) ≈ 0.06%/trade → 80% power only at ≈0.17%/trade. "
                            "The floor (0.05%/day) is far below detectability. No 2020, no 2022 — the two "
                            "highest-shock years are absent; ex-2020 component is vacuous here and is "
                            "replaced by ex-2025 (the year with the most events in this window)."),
        "shock": {"rule": "abs(r_t) >= p90 of trailing 252 |r|, t excluded, per instrument; s = sign(r_t)"},
        "levels": {"range": "high_t - low_t", "C": "close_t", "L": "C - s*0.382*range", "T": "C + s*0.382*range", "stop": "L"},
        "path": {"retrace": "first bar reaching L against s (open beyond L counts at 09:30)",
                 "reclaim": "first later bar closing back on the shock side of C, at or before 14:30 ET",
                 "entry": "open of the next bar, direction s, one trade per instrument-event",
                 "exit": "stop at L (both in one bar -> stop) | target at T | time exit 15:55 bar close",
                 "no_entry": "return 0"},
        "costs": {"round_trip_bp": 3.0, "note": "2bp spread + 1bp slip, every executed trade, both series; break-even reported only"},
        "incumbent": {"name": "naive post-shock continuation",
                      "rule": "direction s from the 09:30 open to the 15:55 close on every instrument-event, same costs",
                      "why": "what the momentum trader does without the path; the delta isolates the path"},
        "delta_series": "per event date: mean over shocked instruments of (structure_net - naive_net); annualised by sqrt(event dates per year)",
        "confluence": {
            "c1": "shock-day volume >= 1.5x median of trailing 20 sessions",
            "c2": "session t+1 open on the shock side of C",
            "c3": "close at tau2 on the shock side of session-to-date VWAP",
            "c4": "proxy (SPY; QQQ for SPY) 09:30->tau2 return has sign s",
            "c5": "close_t in the shock-side 25% of range_t",
            "buckets": "<=1 / 2 / >=3; MONOTONIC iff bucket means weakly increase AND OLS slope on count has date-block CI > 0; any bucket < 30 trades -> INCONCLUSIVE",
        },
        "statistics": {
            "bootstrap": "date-block stationary, L=5, 10000 draws, seed 42, all same-date rows co-resampled (research/hyp111/date_bootstrap.py)",
            "cpcv": "sovereign/discovery/cpcv.py 6/2 over event dates, embargo 1 date",
            "dsr": {"n_trials": n_trials, "note": "mined 1543 + HYP-109 + regime + HYP-110 + this", "on": "Sharpe of the delta series, n_obs = event dates"},
            "components": {
                "b1": "Sharpe(delta) date-block 95% CI excludes 0 from above",
                "b2": "DSR prob >= 0.95",
                "c": ">=12/15 folds Sharpe(delta) > 0 (null <=7, inconclusive 8-11)",
                "g": ">=7/10 instruments Sharpe(structure-naive) > 0 (null <=5, inconclusive 6)",
                "x": "ex-2025 Sharpe(delta) > 0 (replaces ex-2020 in this window)",
                "d": "mean structure_net per event-day vs 0.0005",
            },
        },
        "verdict_ladder": {
            "INCONCLUSIVE": "data abort OR fold error OR (c 8-11) OR (g == 6) OR b2 fails alone OR x fails alone",
            "NULL": "b1 fails OR c <= 7 OR g <= 5",
            "CONFIRMED": "b1 & b2 & c>=12 & g>=7 & x & d >= 0.0005",
            "VALID_BUT_BELOW_FLOOR": "b1 & b2 & c>=12 & g>=7 & x & d < 0.0005",
            "confluence": "separate line MONOTONIC / STORY / INCONCLUSIVE, no weight on the primary",
        },
        "prior_expectation": "CONFIRMED",
        "priors": {"operator": {"prior": "CONFIRMED", "most_likely_failure": "data gate fails (it did, for the scoped window)"},
                   "claude": {"prior": "NOT_SIGNIFICANT", "most_likely_failure": "carried by the equity-index names, absent in TLT/GLD; power too thin"}},
        "frozen_parameters": {"percentile": 0.90, "trailing": 252, "retrace_frac": 0.382, "reclaim_deadline_et": "14:30",
                              "time_exit_et": "15:55", "round_trip_bp": 3.0, "block_L": 5, "draws": 10000, "seed": 42,
                              "cpcv": "6/2", "embargo_dates": 1, "n_trials": n_trials, "floor_per_day": 0.0005},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across fractions, deadlines, costs or subsets"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-02 before any event-session minute bar was read. Free-tier pilot of research/HYP-111_SCOPE.md; power cut stated in the doc.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
