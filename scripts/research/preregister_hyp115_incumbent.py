#!/usr/bin/env python3
"""HYP-115 — the incumbent as the hypothesis. Sealed before any pre-2014 close is read.

Every overlay tested on 2026-09-02 was measured against 'equal-weight, ten ETFs, buy-and-hold'
(+131.7%, Sharpe 0.496, maxDD −33.2% on 2015-01→2026-07). Nothing beat it. It has never itself been
specified as a construction, tested out of sample, or asked whether it is a good portfolio or a lucky
basket over one regime. This registers exactly that. First study in the program where the incumbent
is the hypothesis rather than the hurdle.

  .venv313/bin/python scripts/research/preregister_hyp115_incumbent.py --write|--verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.incumbent.data import CORE, WIDER  # noqa: E402

HYP_ID = "HYP-115"


def build_doc() -> dict:
    n = prereg.mined_total() + 13         # 12 claims through HYP-114 + this
    return {
        "id": HYP_ID, "slug": "incumbent_ew10_validation",
        "name": "The incumbent: equal-weight ten-ETF basket — construction, out-of-sample 2007-2014, lucky-basket test, stress",
        "status": "PREREGISTERED", "frozen_at": "2026-09-03T00:00:00Z", "family": "INCUMBENT-2026-09",
        "construction": {
            "instruments": CORE,
            "weights": "equal, 10% each",
            "rebalance_A": "daily to equal weight (the series every 2026-09-02 overlay was measured against)",
            "rebalance_B": "monthly, first session, 3.0 bp on turnover (the retail-implementable version) — reported alongside",
            "data": "yfinance auto-adjusted daily closes (research/incumbent/data.py), cached; close-to-close log returns",
            "why_yfinance": "daily_universe (2014+) and Alpaca (2016+) cannot reach the out-of-sample window",
        },
        "windows": {
            "in_sample": ["2015-01-02", "2026-07-16"], "in_sample_note": "the window the basket was 'found' on (HYP-109 incumbent)",
            "out_of_sample": ["2007-06-01", "2014-12-31"], "oos_note": "all 30 pool ETFs exist from 2007-05 (UNG/HYG last); includes GFC",
            "stress": {"GFC": ["2007-10-01", "2009-03-31"], "COVID": ["2020-01-01", "2020-12-31"], "2022": ["2022-01-01", "2022-12-31"]},
        },
        "random_basket_pool": CORE + WIDER,
        "claims": {
            "c1_oos_existence": {"stat": "annualised Sharpe of rebalance_A on out_of_sample", "pass": "stationary block bootstrap (L=20, 10000 draws, seed 42) 95% CI excludes 0 from above"},
            "c2_not_lucky": {"stat": ("percentile rank of the basket's Sharpe among 10,000 random equal-weight baskets of 10 drawn without "
                                      "replacement from the 30-ETF pool (seed 42), computed separately in_sample and out_of_sample"),
                             "LUCKY": "in_sample pct >= 95 AND out_of_sample pct <= 50", "not_lucky": "out_of_sample pct >= 50"},
            "c3_stress": {"stat": "max drawdown of rebalance_A vs SPY and vs 60/40 SPY/TLT (daily rebalanced) in each stress window",
                          "pass": "basket maxDD less severe than SPY in all three windows"},
            "descriptive": ["calendar-year returns 2007-2026, basket vs SPY vs 60/40", "rebalance_B numbers", "per-ETF contribution to OOS return"],
        },
        "verdict_ladder": {
            "VALIDATED_CORE": "c1 pass AND not_lucky AND c3 pass",
            "ORDINARY": "c1 pass AND not_lucky AND c3 fails — a real but unremarkable diversified basket",
            "LUCKY": "c2 LUCKY (regardless of c1)",
            "FRAGILE": "c1 fails",
        },
        "prior_expectation": "ORDINARY",
        "priors": {"operator": {"prior": "not stated (forwarded a proposal: 'validated core or dodged a fitted basket')"},
                   "claude": {"prior": "ORDINARY — OOS Sharpe ~0.3-0.5 (GFC crushes EEM/XLF/XLE, TLT/GLD offset), percentile ~50-70; maxDD in GFC will NOT beat 60/40 and may not beat SPY by much",
                              "most_likely_failure": "c3 in the GFC window"}},
        "frozen_parameters": {"block_L": 20, "draws": 10000, "seed": 42, "n_random_baskets": 10000, "basket_size": 10, "turnover_bp": 3.0, "n_trials": n},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across weights, pools, windows or rebalance rules"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-03 before any pre-2014 close was read. The incumbent as hypothesis.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
