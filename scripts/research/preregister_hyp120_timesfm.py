#!/usr/bin/env python3
"""HYP-120 — TimesFM 3.0 zero-shot on the desk's own questions. Three claims, one run, formal-attack sealed.
 (1) MAGNITUDE: does the model's own quantile sigma forecast next-5-session realized vol better than
     persistence (trailing-21 RV) and EWMA(λ=0.94)? — the proven ground (HYP-109/119: magnitude is predictable)
 (2) DIRECTION: does sign(point forecast, step 1) beat a coin on next-day returns? — the ledger's strongest null
 (3) CASCADE: on the HYP-119 liquid events + matched non-state minutes, does TimesFM's 30-min sigma rank the
     realized 30-min range better than the bar-derived cascade z? — does a foundation model see the blowup?"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.formal.report import attack, to_dict  # noqa: E402
from research.formal.dsl import Rule, win  # noqa: E402

HYP_ID = "HYP-120"
RULE = Rule("TimesFM zero-shot", features={"ctx_daily": win("logret", -512, -1), "ctx_minute": win("logret_1m", -512, -1)}, entry_offset=1)


def build_doc() -> dict:
    doc = {
        "id": HYP_ID, "slug": "timesfm3_zero_shot_magnitude_direction_cascade",
        "name": "TimesFM 3.0 zero-shot: realized-vol forecast vs persistence, next-day direction, cascade magnitude vs the bar ratio",
        "status": "PREREGISTERED", "frozen_at": "2026-09-16T00:00:00Z", "family": "FOUNDATION-MODEL-2026-09",
        "model": {"checkpoint": "google/timesfm-3.0-pytorch (MLX backend, numerically matched)", "license": "weights non-commercial — research only; nothing here ships",
                  "context": 512, "horizons": {"daily": 5, "minute": 30}, "sigma": "(q0.9 − q0.1)/(2·1.2816) from the model's quantile head", "no_finetune": "zero-shot only; no fitting anywhere"},
        "data": {"daily_panel": "10 ETFs (daily_universe, 2014+) + 9 G10 vs USD (FRED DEX*), log returns; windows t ≥ 512, every session, 50,182 windows 2016-01 → 2026-08",
                 "cascade": "HYP-119 liquid events (7,640) + one matched non-state minute per event (seed 42), 512-minute causal context from the minute cache"},
        "claims": {
            "c1_magnitude": {"target": "std of the next 5 daily log returns", "stat": "ΔQLIKE = QLIKE(TimesFM σ) − QLIKE(baseline), QLIKE = ln σ² + RV²/σ², pooled by date",
                             "baselines": ["trailing-21 RV", "EWMA λ=0.94"], "pass": "date-block 95% CI of ΔQLIKE < 0 against BOTH baselines", "descriptive": "Spearman(σ̂, RV) for each"},
            "c2_direction": {"target": "sign of next-day log return", "stat": "hit rate of sign(step-1 point forecast), pooled by date; Spearman(forecast, realized) per series",
                             "pass": "hit-rate date-block CI > 0.50 AND mean per-series Spearman CI > 0"},
            "c3_cascade": {"target": "realized 30-min log range after the minute", "stat": "Spearman(TimesFM 30-min σ, range) − Spearman(cascade z, range) on state ∪ non-state minutes (z for non-state = its raw ratio z)",
                           "pass": "date-block CI of the difference > 0", "descriptive": "each Spearman alone; range ratio state/non-state as the model sees it"},
        },
        "verdict_ladder": {"per claim": "PASS / FAIL", "ledger": "MAGNITUDE if c1 passes; DIRECTION if c2 passes; CASCADE if c3 passes; NULL if none — combined as a list"},
        "n_trials": 1648, "n_trials_note": "1645 after HYP-119 + 3 claims",
        "prior_expectation": "MAGNITUDE_ONLY",
        "priors": {"operator": {"prior": "try all three, see which performs best"},
                   "claude": {"c1": "modest PASS vs RV21, closer vs EWMA", "c2": "FAIL — hit ≈ 0.50, IC ≈ 0 (118 verdicts)", "c3": "FAIL — the bar ratio uses volume and range the return series cannot see",
                              "most_likely_failure": "c2 and c3"}},
        "abort": {"no_rerun": "one run", "no_scan": "no context, horizon, quantile, baseline or panel change after the run"},
        "verdict": None, "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }
    doc["attack_report"] = to_dict(attack(RULE, doc))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write: return prereg.write(build_doc(), "Pre-registered 2026-09-16 through the formal attack stage; zero-shot only; not run at sealing.")
    if a.verify: prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
