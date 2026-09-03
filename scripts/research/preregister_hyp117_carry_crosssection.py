#!/usr/bin/env python3
"""HYP-117 — cross-sectional G10 carry/momentum/value selection, sealed HOLDOUT 1990-01 → 2005-12.

Everything is frozen: research/carry_model/{data_fred,panel,models,evaluate}.py as committed at sealing,
the six models, top-5 equal-notional, retail haircut 0.30 + 3bp/leg. The development window (2006-2026,
walk-forward) has been run and its numbers are declared below — none of the six beat the plain carry
factor there. The holdout has never been read (data guard: CARRY_HOLDOUT_UNLOCK). Fitted models are
trained ONCE on all of 2006-2026 and applied to 1990-2005 (reverse out-of-sample, deliberately: the
holdout contains the 1992 ERM and 1998 LTCM/JPY crashes the models never saw).

  .venv313/bin/python scripts/research/preregister_hyp117_carry_crosssection.py --write|--verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-117"


def build_doc() -> dict:
    dev = json.loads((ROOT / "data" / "research" / "carry_model" / "dev_results.json").read_text())
    n_models = dev.pop("_n_models"); dev.pop("_oracle_ann", None)
    n = prereg.mined_total() + 14 + 76 + n_models          # ledger claims through HYP-116 + carry-map cells + dev models
    return {
        "id": HYP_ID, "slug": "g10_crosssection_carry_mom_value_holdout_1990_2005",
        "name": "Cross-sectional G10 pair selection (carry/momentum/value ± derivatives × vol) — sealed holdout 1990-2005",
        "status": "PREREGISTERED", "frozen_at": "2026-09-03T00:00:00Z", "family": "CARRY-CROSSSECTION-2026-09",
        "design_doc": "research/carry_map/MODEL_DESIGN.md", "map": "research/carry_map/REPORT.md (76 cells, in-sample 2006-2026)",
        "data": {"source": "FRED only: DEX* spot, IRSTCI01* rates, OECD CPI, EUR spliced from DEM pre-1999 (research/carry_model/data_fred.py)",
                 "holdout": ["1990-01-31", "2005-12-31"], "guard": "panel refuses rows <= 2005-12-31 unless CARRY_HOLDOUT_UNLOCK=1 (set only by the sealed test after gate zero)",
                 "dev_window": ["2006-01-31", "2026-06-30"], "return": "Δlog spot + carry/12·(1−0.30) − 6bp; long a / short b convention, 45 pairs"},
        "models_frozen": ["M1 baseline z(carry)+z(mom12)+z(value), no parameters",
                          "M2 ridge(α=10) on 8 features", "M3 ridge + carry×{factor_dd, spread_chg12, fed6}", "M4 HistGBM depth3/200/lr0.05 monotone(carry+,mom12+,value+,rvol3−)",
                          "M5 SHOT: ridge on features+first/second differences × trailing-3m realized vol (operator's statistics×calculus)",
                          "M5b SHOT-pure: sign(M1) × realized vol"],
        "portfolio": "top-5 by |score|, sign = sign(score), equal notional, monthly; top-1 reported descriptively",
        "dev_results_declared": dev,
        "dev_read": "no model's IC CI excludes 0 (M1 lower bound −0.001); no top-5 beats the carry factor's Sharpe; M4/M5 top-1 (+4%/yr, Sh 0.46-0.48) is the noise slot on 126 months",
        "statistics": {"ic": "monthly Spearman IC across the 45 signed pairs; mean with block-bootstrap CI (L=6, 5000)",
                       "portfolio": "top-5 Sharpe vs carry factor Sharpe on the same holdout months, jointly resampled",
                       "perm": "cross-sectional permutation (targets shuffled across pairs within month), 2000 draws, BH across the 6 models",
                       "crashes": "1992-09 (ERM) and 1998-08..10 (LTCM/JPY) reported month by month, descriptive"},
        "verdict_ladder": {"MODEL_HOLDS": "for a model: IC CI > 0 AND top-5 Sharpe − factor Sharpe CI > 0 AND BH-adjusted perm p < 0.05; ledger verdict MODEL_HOLDS if any model holds",
                           "IC_ONLY": "IC CI > 0 for some model but no portfolio improvement after costs",
                           "NULL": "no model has IC CI > 0"},
        "n_trials": n, "n_trials_note": "1543 mined + 14 ledger claims + 76 carry-map cells + 6 dev models",
        "prior_expectation": "NULL",
        "priors": {"operator": {"prior": "not stated; wanted to 'see those numbers' for the statistics×calculus shot"},
                   "claude": {"prior": "NULL to IC_ONLY — IC ≈ 0.03 on 192 holdout months has a CI of ±0.05; the 1992/1998 crashes have different signatures; value is the only feature with holdout content and it is published",
                              "most_likely_failure": "M1 IC positive but CI includes 0; every top-5 below the factor"}},
        "abort": {"no_rerun": "one run", "no_scan": "no model, feature, k, haircut or window change after the holdout is read"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-03 with dev-window results declared; holdout 1990-2005 unread (data guard).")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
