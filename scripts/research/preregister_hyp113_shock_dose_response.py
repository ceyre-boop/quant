#!/usr/bin/env python3
"""HYP-113 — magnitude→reversal dose-response within p90+ shocks. Sealed before any bin is computed.

HYP-111a found naive next-session continuation after a shock LOSES (−0.128%/event-day, CI excludes 0,
10/10 instruments, 2023-06→2026-07). The fade is the finding. This asks whether the fade scales with
shock size inside the p90+ set. Same 798 events, same cached minute bars, no new data. A pass makes
the fade selectable; a null kills "fade harder when bigger" cleanly.

  .venv313/bin/python scripts/research/preregister_hyp113_shock_dose_response.py --write|--verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402

HYP_ID = "HYP-113"


def build_doc() -> dict:
    n_primary = prereg.mined_total() + 6          # HYP-109, regime, HYP-110, HYP-111a, HYP-112, this
    return {
        "id": HYP_ID, "slug": "postshock_fade_dose_response",
        "name": "Post-shock next-session fade: dose-response in shock size within p90+ (ten ETFs, 2023-06 → 2026-07)",
        "status": "PREREGISTERED", "frozen_at": "2026-09-02T00:00:00Z",
        "family": "POSTSHOCK-INTRADAY-2026-09", "lineage": "HYP-111a incumbent series (naive continuation, negated)",
        "thesis": "Within top-decile shocks, the next-session intraday fade return increases with shock size.",
        "instrument_set": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"],
        "data": {"events": "the 798 HYP-111a events (data/research/hyp111/hyp111a_trades.parquet: sym, t, t1, s, naive_net)",
                 "intraday": "cached data/cache/theta_1m — no new fetch",
                 "daily": "data/cache/daily_universe — for the size measure only",
                 "window_t1": ["2023-06-02", "2026-07-16"], "regime_caveat": "one regime; says nothing about 2020-2022"},
        "size_measure": {"x": "trailing-252 percentile rank of |r_t| among the prior 252 sessions' |r| (t excluded), in [0.90, 1.0]",
                         "bins": {"B1": "[0.90, 0.95)", "B2": "[0.95, 0.99)", "B3": "[0.99, 1.0]"},
                         "why_rank": "comparable across instruments and vol levels; the raw |r| is not"},
        "outcome": {"fade": "−s × (close_15:55 − open_09:30)/open_09:30 on session t+1, minus 3.0 bp — i.e. exactly −(HYP-111a naive_net)",
                    "unit": "% of notional per event"},
        "claims": {
            "primary": {"population": "all 798 events",
                        "pass": "bin means weakly increasing B1 ≤ B2 ≤ B3 AND OLS slope of fade on x has date-block 95% CI excluding 0 from above",
                        "n_trials": n_primary},
            "secondary": {"population": "down-shocks only (s = −1)", "pass": "same rule", "n_trials": n_primary + 1,
                          "note": "the operator's stated form — 'after a sharp move down, fade harder when bigger'"},
            "descriptive": ["per-bin mean, CI, n, per instrument", "up-shocks-only bins", "fade %/event-day per bin vs 0.05% (no gate)"],
        },
        "statistics": {"bootstrap": "date-block stationary L=5, 10000 draws, seed 42 (research/hyp111/date_bootstrap.py)",
                       "abort": "any bin with < 40 events (primary) / < 25 (secondary) → INCONCLUSIVE for that claim"},
        "verdict_ladder": {"DOSE_RESPONSE": "pass", "FLAT": "slope CI includes 0 or bins not monotone", "INCONCLUSIVE": "abort"},
        "floor_note_standing": ("Written before any floor-gated fade prereg is run: a selective strategy that sits in cash is "
                                "measured on deployed capital; an always-on overlay on calendar time. HYP-111a's 0.05%/event-day "
                                "floor was the always-on form. Any fade prereg that adopts per-deployed-capital must cite this "
                                "note and seal the denominator before its run. This hypothesis has no floor gate."),
        "prior_expectation": "NOT_SIGNIFICANT",
        "priors": {"operator": {"prior": "expects a dose-response (message 2026-09-02, not formally stated)", "most_likely_failure": "not stated"},
                   "claude": {"prior": "NOT_SIGNIFICANT", "most_likely_failure": "B3 (p99+) has ~60-80 events and its CI swallows the slope; B1 vs B2 flat"}},
        "frozen_parameters": {"cost_bp": 3.0, "bins": [0.90, 0.95, 0.99, 1.0], "block_L": 5, "draws": 10000, "seed": 42,
                              "n_trials_primary": n_primary, "n_trials_secondary": n_primary + 1},
        "abort": {"no_rerun": "one run, one verdict", "no_scan": "no search across bin edges, size measures or outcomes"},
        "verdict": None,
        "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return prereg.write(build_doc(), "Pre-registered 2026-09-02 before any size bin was computed. Reuses HYP-111a events; no new data.")
    if a.verify:
        prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
