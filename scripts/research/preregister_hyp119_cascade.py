#!/usr/bin/env python3
"""HYP-119 — the liquidity-cascade STATE (Navier–Stokes ratio) on liquid names and microcap event-days.
First hypothesis to pass through all four stages: declared in the formal DSL, attacked (Z3 causality
proof, LULD halt bound, cost lemma, static multiplicity), sealed, then run once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.hyp111 import prereg  # noqa: E402
from research.formal.report import attack, to_dict  # noqa: E402
from research.cascade.state import RULE, W, Z_THR, HORIZON  # noqa: E402

HYP_ID = "HYP-119"


def build_doc() -> dict:
    doc = {
        "id": HYP_ID, "slug": "liquidity_cascade_state",
        "name": "Liquidity-cascade state (|kinetic|/dampening z>3): magnitude, continuation with realistic fills, proven halt bound",
        "status": "PREREGISTERED", "frozen_at": "2026-09-13T00:00:00Z", "family": "CASCADE-2026-09",
        "plan": "Plans/quant-repo-floofy-swan.md (The Blowup Loop)", "lessons_consulted": "research/lessons.jsonl (128 entries; HYP-093/107/109/112 directly relevant)",
        "state": {"K": "RVOL(t−5..t−1, vs same-clock median of prior 60 sessions) × signed 5-min log return (close t−1 / close t−5)",
                  "D": "Σ dollar volume(t−5..t−1) / mean((high−low)/close, t−5..t−1)", "z": "log(|K|/D) standardised on a causal baseline",
                  "baseline_liquid": "same name, prior 60 sessions", "baseline_microcap": "pooled minutes of all prior event-days (expanding)",
                  "STATE": f"z > {Z_THR}", "one_event_per": f"{HORIZON} minutes", "direction": "sign(5-min return)"},
        "universes": [{"name": "liquid", "what": "23 continuous names, data/cache/minute_bars, 2024-07 → 2026-09 (eval from session 61)"},
                      {"name": "microcap", "what": "single-event-day names in the same cache (HYP-107 harvest), ~230 symbol-days 2025–2026; LONG-side cascades only (no borrow model)"}],
        "claims": {
            "a_magnitude": "median forward-30-min log range (high/low) after STATE ÷ median over non-STATE minutes of the same sessions; date-block bootstrap (L=5 dates) 95% CI > 1",
            "b_continuation": "enter in direction at open of m+1 paying the measured half-spread (TICK-039 model at the event's price/range/$vol); exit at min(30 min, first LULD halt-resume bar open − 2% slip, 15:55) receiving the half-spread; mean net return date-block CI > 0 AND > the mean of 1000 random-direction placebos on the same events (p<0.05) AND cross-event permutation p<0.05; BH across the two universes",
            "c_bound": "for every entry, the Z3 max_loss_before_halt(entry, clock, tier 2, trailing 5 closes) is the declared worst case; realized worst trade reported against the proven bound; any realized loss beyond it = model failure (data or mechanics), reported as such",
        },
        "verdict_ladder": {"CASCADE_TRADEABLE": "a ∧ b ∧ c on a universe (per universe)", "MAGNITUDE_ONLY": "a only", "NULL": "a fails"},
        "priors": {"operator": {"prior": "MAGNITUDE_ONLY (2026-09-13)"}, "claude": {"prior": "MAGNITUDE_ONLY on liquid; microcap possibly tradeable-but-below-floor; fills decide", "most_likely_failure": "b on liquid: direction null at every resolution tried"}},
        "frozen_parameters": {"W": W, "z_threshold": Z_THR, "horizon_min": HORIZON, "baseline_sessions": 60, "halt_slip": 0.02, "min_events_per_universe": 40, "placebo_draws": 1000, "block_L": 5, "draws": 5000, "seed": 42},
        "abort": {"no_rerun": "one run", "no_scan": "no z threshold, window, horizon or universe change after the run", "data": "< 40 events in a universe → INCONCLUSIVE for that universe"},
        "n_trials": 1645, "n_trials_note": "1642 after HYP-118 + this doc's 3 claims (counted once per universe pair by BH)",
        "prior_expectation": "MAGNITUDE_ONLY", "verdict": None, "hash_method": "sha256(json.dumps(doc minus hash_lock, sort_keys=True, separators=(',',':')))",
    }
    doc["attack_report"] = to_dict(attack(RULE, doc, entry_price=5.0, entry_clock="10:00", tier=2, typical_price=5.0, bar_range_pct=0.02, minute_dollar_vol=2e5))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--verify", action="store_true"); ap.add_argument("--attack", action="store_true")
    a = ap.parse_args()
    if a.attack: print(json.dumps(build_doc()["attack_report"], indent=1)); return 0
    if a.write: return prereg.write(build_doc(), "Pre-registered 2026-09-13 — first hypothesis through the formal attack stage (Z3 causality proof attached). Not run at sealing.")
    if a.verify: prereg.verify(HYP_ID); return 0
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
