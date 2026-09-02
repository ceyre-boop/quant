#!/usr/bin/env python3
"""HYP-112 — post-shock ATM straddle vs matched control. Sealed before any option chain for an
event is read (one schema probe: TLT 2022-06-17 chain on 2022-06-13, declared, format only).

The instrument question: HYP-109 proved next-week realized vol is 1.36x after a shock and that
direction is null. UVXY was the wrong instrument (roll). The right one is the straddle on the
shocked ETF itself, bought at real NBBO on the data the desk already pays for (OPTION.VALUE,
first access 2020-01). The question is whether implied vol already prices the clustering.

  .venv313/bin/python scripts/research/preregister_hyp112_postshock_straddle.py --write|--verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-112"


def build_doc() -> dict:
    n_trials = prereg.mined_total() + 5          # HYP-109, regime, HYP-110, HYP-111a, this
    return {
        "id": HYP_ID, "slug": "postshock_atm_straddle_vs_control",
        "name": "Post-shock ATM straddle (5-session hold) vs matched non-shock control, ten ETFs, 2020-2026",
        "status": "PREREGISTERED", "frozen_at": "2026-09-02T00:00:00Z",
        "family": "MAGNITUDE-INSTRUMENT-2026-09",
        "lineage": "HYP-109(a) magnitude PASS 1.36x p~0; UVXY companion descriptive negative; this is the direct instrument",
        "thesis": ("After a top-decile |return| day, buying the ATM straddle on the shocked ETF and holding "
                   "five sessions earns more (on premium) than the same straddle bought on a matched "
                   "non-shock day, because realized vol rises more than implied vol has already repriced."),
        "instrument_set": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"],
        "data": {
            "daily": "data/cache/daily_universe/<SYM>.parquet (shock rule, spot, session calendar)",
            "options": "ThetaData v2 /v2/bulk_hist/option/eod on 127.0.0.1:25510 (OPTION.VALUE, first access 2020-01), cached data/cache/theta_opt_eod/",
            "quote": "last EOD NBBO tick of the date (ThetaData EOD snapshot, ~17:20 ET) — an approximation of a next-open fill, stated",
            "schema_probe_declared": "TLT exp 2022-06-17 on 2022-06-13, format only",
            "window_t": ["2020-01-02", "2026-07-08"], "hold_sessions": 5,
            "abort": "pooled events with a priced straddle on BOTH legs (event and control) < 100, or any instrument < 30",
        },
        "power_statement": ("~1,798 instrument-events on ~670 dates before option-quote availability; expect "
                            "some loss to unquoted strikes. Straddle returns on premium have sd ~40-60%; with "
                            "~500 independent dates SE ≈ 2-3% of premium → detects a ~6% premium-return "
                            "difference. 2020 is 22% of dates and the largest IV regime; ex-2020 is a component."),
        "shock": {"rule": "abs(r_t) >= p90 of trailing 252 |r|, t excluded, per instrument (HYP-109 rule)"},
        "straddle": {
            "expiration": "nearest listed expiration >= t + 7 calendar days",
            "strike": "listed strike nearest close_t on date t",
            "entry": "buy call + put at the ASK, EOD NBBO of date t",
            "exit": "sell call + put at the BID, EOD NBBO of session t+5; bid 0 -> 0 proceeds",
            "commission": "$0.65 per contract per leg (4 legs)",
            "skip": "either leg unquoted (ask <= 0 or bid <= 0) at entry -> event dropped (both legs)",
            "unit": "return on premium paid; return on spot notional reported",
        },
        "incumbent": {"name": "matched non-shock control straddle",
                      "rule": "same instrument, control date c = t - 10 sessions (skipped if c is itself a shock day), same construction, its own nearest expiration >= c+7d",
                      "why": "isolates the shock conditioning; the unconditional straddle premium (negative VRP) is netted out"},
        "delta_series": "per event date: mean over shocked instruments of (straddle_ret_on_premium(t) - straddle_ret_on_premium(c)); annualised by sqrt(event dates per year)",
        "descriptive": ["implied move = straddle mid / spot vs realized |5-session log move|, shock vs control",
                        "mean straddle return on premium, shock and control separately, per instrument and per year",
                        "share of events lost to unquoted strikes"],
        "statistics": {
            "bootstrap": "date-block stationary, L=5, 10000 draws, seed 42 (research/hyp111/date_bootstrap.py)",
            "cpcv": "sovereign/discovery/cpcv.py 6/2 over event dates, embargo 5 dates",
            "dsr": {"n_trials": n_trials, "note": "mined 1543 + HYP-109 + regime + HYP-110 + HYP-111a + this", "on": "Sharpe of the delta series"},
            "components": {
                "b1": "Sharpe(delta) date-block 95% CI excludes 0 from above",
                "b2": "DSR prob >= 0.95",
                "c": ">=12/15 folds Sharpe(delta) > 0 (null <=7, inconclusive 8-11)",
                "g": ">=7/10 instruments mean delta > 0 (null <=5, inconclusive 6)",
                "x": "ex-2020 Sharpe(delta) > 0",
                "d": "mean shock-straddle return on SPOT per event vs 5 x 0.0005 = 0.0025 (five-session hold)",
            },
        },
        "verdict_ladder": {
            "INCONCLUSIVE": "data abort OR fold error OR (c 8-11) OR (g == 6) OR b2 fails alone OR x fails alone",
            "NULL": "b1 fails OR c <= 7 OR g <= 5",
            "CONFIRMED": "b1 & b2 & c>=12 & g>=7 & x & d >= 0.0025",
            "VALID_BUT_BELOW_FLOOR": "b1 & b2 & c>=12 & g>=7 & x & d < 0.0025",
        },
        "prior_expectation": "NOT_SIGNIFICANT",
        "priors": {"operator": {"prior": "not stated (instructed: run all of them)", "most_likely_failure": "not stated"},
                   "claude": {"prior": "NOT_SIGNIFICANT",
                              "most_likely_failure": "IV has already repriced by the shock-day close: the ask you pay embeds the 1.36x; delta ≈ 0 or negative on premium after the spread"}},
        "frozen_parameters": {"percentile": 0.90, "trailing": 252, "hold_sessions": 5, "min_dte_days": 7, "control_offset_sessions": 10,
                              "commission_per_contract": 0.65, "block_L": 5, "draws": 10000, "seed": 42, "cpcv": "6/2",
                              "embargo_dates": 5, "n_trials": n_trials, "floor_per_event_on_spot": 0.0025},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across DTE, strike offset, hold or control offset"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-02 before any event option chain was read. The magnitude instrument, tested on real NBBO.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
