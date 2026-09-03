#!/usr/bin/env python3
"""HYP-114 — deploying the unconditional post-shock fade. Sealed before any 2016-2019 core session
or any wider-universe session is read.

What stands after 2026-09-02 (HYP-109/110/111/111a/112/113): after a top-decile |return| day the
next-session continuation loses; the fade is +0.147%/event-day on ten ETFs 2020-2026, 10/10, holds in
March 2020, not improvable by size/path/confluence, not buyable through vol. This registers how a $2k
retail account would DEPLOY it, and tests the deployment on data no fade test has seen (2016-2019,
and 20 new instruments).

THE SIZING DENOMINATOR — sealed here, in the form that could have been written before any number:
  A selective strategy that sits in cash is measured on deployed capital; an always-on overlay on
  calendar time. The fade is selective (deployed ~25% of calendar days). Its yield is therefore
  return per unit of capital deployed per day, and the constitutional floor (0.05%/day) applies to
  THAT quantity. Calendar-time CAGR, Sharpe and max drawdown are reported beside it so the always-on
  comparison is never hidden. The position rule that defines "deployed" is frozen below and is not a
  sizing decision: the same fraction of equity in every shocked instrument, every time.

  .venv313/bin/python scripts/research/preregister_hyp114_fade_deployment.py --write|--verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-114"
WIDER = ["XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "HYG", "LQD", "USO", "SLV", "VNQ", "XBI", "SMH", "KRE", "FXI", "EWZ", "EWJ", "GDX", "UNG"]


def build_doc() -> dict:
    base = prereg.mined_total() + 9              # HYP-109, regime, HYP-110, HYP-111a, HYP-112, HYP-113×2, HYP-111×2
    return {
        "id": HYP_ID, "slug": "postshock_fade_deployment",
        "name": "Deploying the unconditional post-shock next-session fade: denominator, unseen years, wider universe, exit",
        "status": "PREREGISTERED", "frozen_at": "2026-09-02T00:00:00Z", "family": "POSTSHOCK-FADE-DEPLOY-2026-09",
        "lineage": "HYP-111 secondary FADE_HOLDS (d3d5258285a74cf9); HYP-113 floor_note_standing",
        "rule_frozen": {
            "shock": "abs(r_t) >= p90 of trailing 252 |r| (t excluded), per instrument, close-to-close log return",
            "trade": "session t+1: direction −sign(r_t), 09:30 open → 15:55 bar close, 3.0 bp round trip",
            "position": "10% of current equity per shocked instrument (fractional shares; k shocked → 10k% deployed, capped 100%), compounding daily",
            "no_selection": "no size filter (HYP-113), no path (HYP-111), no confluence, no regime gate",
        },
        "denominator_sealed": {
            "statement": ("A selective strategy that sits in cash is measured on deployed capital; an always-on overlay on "
                          "calendar time. Yield = mean net fade per instrument-day of deployment; floor 0.0005 applies to it."),
            "always_on_comparison": "calendar CAGR, calendar Sharpe, max drawdown of the compounding account, mean deployed fraction, event-dates/yr — all reported",
        },
        "core_instruments": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"],
        "wider_universe": WIDER,
        "data": {"core_daily": "data/cache/daily_universe (2014+), shocks valid from 2015-01; events with t+1 in core_window_t1",
                 "wider_daily": "Alpaca SIP 1Day adjustment=all from 2016-01-04 (research/hyp111/alpaca_1m.py::daily_bars); shocks valid from 2017-01",
                 "intraday": "Alpaca SIP 1-min RTH (alpaca_1m.py::stock_1m), cache data/cache/alpaca_1m_rth/",
                 "already_seen": "core 2020-01→2026-07 sessions were read by HYP-111 (fade known +0.147%); 2016-2019 core and ALL wider sessions are unseen",
                 "abort": "< 95% of event sessions with >= 370 bars → INCONCLUSIVE"},
        "claims": {
            "claim1_denominator_unseen": {
                "n_trials": base + 1,
                "pass": ("deployed yield date-block CI > 0 AND yield >= 0.0005 AND unseen-years (t+1 in 2016-01-04..2019-12-31) mean fade CI > 0 "
                         "AND >= 7/10 core instruments mean > 0 in unseen years AND account max drawdown better than −15%"),
                "valid_but_below_floor": "all of the above except yield < 0.0005",
                "fail": "otherwise",
            },
            "claim2_universe": {"n_trials": base + 2,
                                "pass": "wider-20 mean fade date-block CI > 0 AND >= 14/20 instruments mean > 0 AND ex-2020 mean > 0",
                                "descriptive": "30-ETF account vs 10-ETF account on the same 2017+ window: CAGR, maxDD, deployed fraction, event-dates/yr"},
            "claim3_exit": {"n_trials": base + 3, "alternative": "exit at the 12:00 bar close instead of 15:55 (one alternative, no sweep)",
                            "verdict": "EARLY_EXIT_BETTER if delta CI > 0; LATE_EXIT_BETTER if CI < 0; NO_DIFFERENCE otherwise"},
        },
        "statistics": {"bootstrap": "date-block stationary L=5, 10000 draws, seed 42, same-date rows co-resampled",
                       "note": "no CPCV here: the claims are means on disjoint unseen samples, not a fitted model"},
        "ledger_verdict": "claim 1's verdict (the deployment test); claims 2 and 3 recorded in result",
        "prior_expectation": "CONFIRMED",
        "priors": {"operator": {"claim1": "PASS", "claim3": "no prior — 'test what the difference is'"},
                   "claude": {"claim1": "VALID_BUT_BELOW_FLOOR — 2016-2019 is a low-vol regime; expect it to look like 2024 (+0.02%)",
                              "claim2": "PASS — short-term reversal is pervasive in liquid ETFs; expect commodities/EM to carry it",
                              "claim3": "NO_DIFFERENCE"}},
        "frozen_parameters": {"percentile": 0.90, "trailing": 252, "round_trip_bp": 3.0, "fraction_per_instrument": 0.10,
                              "floor_per_deployed_day": 0.0005, "max_dd_abort": 0.15, "early_exit_et": "12:00",
                              "core_window_t1": ["2016-01-04", "2026-07-16"], "wider_window_t1": ["2017-01-03", "2026-07-16"],
                              "block_L": 5, "draws": 10000, "seed": 42, "n_trials_base": base},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across fractions, exits, universes or windows"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-02 before any 2016-2019 or wider-universe session was read. Sizing denominator sealed in the doc.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
