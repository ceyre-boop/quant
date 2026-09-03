#!/usr/bin/env python3
"""HYP-118 — M6 (characteristic ranking + mechanical crash management) on an UNSEEN universe:
five EM currencies × three funding currencies, FRED only, 1997-2026. Sealed before any EM row is read
(data guard EM_HOLDOUT_UNLOCK). Answers Colin's question: is there a more advanced way to carry-trade
whose drawdown is not 15× its average year — and does 'stronger prediction' exist, or only stronger
risk management?"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-118"


def build_doc() -> dict:
    n = 1640 + 2                        # after HYP-117; M6 and M6b are two claims
    return {
        "id": HYP_ID, "slug": "em_carry_m6_unseen_universe",
        "name": "M6 characteristic ranking + mechanical crash management, unseen EM universe 1997-2026",
        "status": "PREREGISTERED", "frozen_at": "2026-09-03T00:00:00Z", "family": "CARRY-CROSSSECTION-2026-09",
        "lineage": "HYP-117 IC_ONLY: parameter-free carry+mom+value ranking real on 1990-2005; every fitted model negative",
        "universe": {"assets": ["MXN", "ZAR", "BRL", "KRW", "INR"], "funding": ["USD", "JPY", "EUR"], "positions": 15,
                     "data": "FRED DEX*US spot, IRSTCI01/IR3TIB01 rates, OECD CPI (research/carry_model/em.py)", "window": ["1997-01-31", "2026-06-30"],
                     "unseen": "no EM row read before sealing; guard EM_HOLDOUT_UNLOCK set only by the test after gate zero",
                     "return": "Δlog spot + carry/12·(1−0.30) − 12bp (6bp/leg, EM spreads)"},
        "model_frozen": "research/carry_model/m6.py as committed: score z(carry)+z(mom12)+z(value)−0.5·z(rvol3); top-5 of 15; risk-parity 1/rvol; vol-target 10% cap 1.5×; ×0.5 if VIX(t−1)>30. M6b adds ×1.5 when the universe's carry factor is >10% off peak. No fitting.",
        "comparators": {"plain_em_carry": "top-5 by carry, equal weight, same universe/costs", "hyp117_g10": "M1 top-5 Sh 0.80, factor Sh 0.82, IC 0.15 on 1990-2005; G10 factor 2006-26 Sh 0.19 maxDD −33.5%"},
        "claims": {
            "c1_ranking": "monthly IC of the M6 score across 15 positions: block-bootstrap CI (L=6) excludes 0 from above",
            "c2_return": "Sharpe(M6 managed) − Sharpe(plain EM carry), jointly resampled: CI excludes 0 from above",
            "c3_drawdown": "maxDD(M6 managed) ≤ (2/3)·maxDD(plain EM carry)  [the crash-management claim]",
            "c4_perm": "cross-sectional permutation (targets shuffled across positions within month), 2000 draws, p<0.05 for the M6 top-5 raw mean",
            "descriptive": ["1997-98 Asian/Russia, 2008, 2013 taper, 2015, 2020, 2022 month-by-month", "M6b", "per-currency", "exposure path", "IC by year"],
        },
        "verdict_ladder": {"ADVANCED_HOLDS": "c1 & c2 & c3 & c4", "RISK_ONLY": "c1 & c3 & c4 but not c2 — ranking + crash management cut the drawdown without beating plain carry's Sharpe",
                           "IC_ONLY": "c1 & c4 only", "NULL": "c1 fails"},
        "n_trials": n, "prior_expectation": "RISK_ONLY",
        "priors": {"operator": {"prior": "wants 'stronger prediction'; not stated formally"},
                   "claude": {"prior": "RISK_ONLY — EM carry IC will be positive (well documented, larger differentials), vol targeting will cut the 1998/2008 drawdowns by more than a third mechanically, but the Sharpe will not be distinguishable from plain EM carry; there is no stronger prediction, only stronger risk management",
                              "most_likely_failure": "c2"}},
        "abort": {"no_rerun": "one run", "no_scan": "no weight, k, vol target, cap, VIX threshold or universe change after the run"},
        "verdict": None, "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write: return prereg.write(build_doc(), "Pre-registered 2026-09-03 before any EM row was read.")
    if a.verify: prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
