#!/usr/bin/env python3
"""
Combinatorial hypothesis feeder (Edge Factory, Track A).
========================================================

Enumerates PARAM-DELTA hypotheses (the only kind EdgePipeline auto-tests), prioritizes
small perturbations of the current green config (touching already-validated conditions
beats wild swings), dedups against rejection memory, and appends to the queue.

Single-change deltas only — so a pass/fail is interpretable (which one knob moved).

Usage:  python3 scripts/generate_combinatorial.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUEUE = ROOT / "data" / "research" / "hypothesis_queue.jsonl"

# Current (green, post-rollback) config — perturbations near these get priority.
FOREX_VIX_NOW = {"USDJPY=X": 15, "AUDNZD=X": 15, "EURUSD=X": 18, "GBPUSD=X": 18, "AUDUSD=X": 20}
FOREX_VIX_GRID = [12, 14, 16, 18, 20, 22]
ICT_WEIGHTS_NOW = {"kill_zone": 2.0, "sweep": 4.0, "fvg_tap": 3.0, "displacement": 1.0,
                   "market_structure": 0.0, "pd_alignment": 0.0}
ICT_WEIGHT_GRID = {"sweep": [3.0, 3.5, 4.5, 5.0], "fvg_tap": [2.0, 2.5, 3.5, 4.0],
                   "displacement": [0.5, 1.5, 2.0], "kill_zone": [1.5, 2.5, 3.0],
                   "market_structure": [0.5, 1.0], "pd_alignment": [0.5, 1.0]}
ICT_MIN_SCORE_NOW = 6.0
ICT_MIN_SCORE_GRID = [5.5, 6.5, 7.0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidates() -> list[dict]:
    """One-change param-delta candidates with a priority = closeness to current config."""
    out = []
    # Forex VIX gate, one pair at a time
    for pair, cur in FOREX_VIX_NOW.items():
        for thr in FOREX_VIX_GRID:
            if thr == cur:
                continue
            out.append({"subsystem": "forex", "param_delta": {"PAIR_VIX_GATES": {pair: float(thr)}},
                        "priority": round(1.0 / (1 + abs(thr - cur)), 3),
                        "label": f"VIX gate {pair} {cur}->{thr}"})
    # ICT scoring weights, one component at a time
    for comp, grid in ICT_WEIGHT_GRID.items():
        cur = ICT_WEIGHTS_NOW[comp]
        for val in grid:
            out.append({"subsystem": "ict", "param_delta": {"weights": {comp: float(val)}},
                        "priority": round(1.0 / (1 + abs(val - cur)), 3),
                        "label": f"ICT weight {comp} {cur}->{val}"})
    # ICT min score to trade
    for val in ICT_MIN_SCORE_GRID:
        out.append({"subsystem": "ict", "param_delta": {"min_score_to_trade": float(val)},
                    "priority": round(1.0 / (1 + abs(val - ICT_MIN_SCORE_NOW)), 3),
                    "label": f"ICT min_score {ICT_MIN_SCORE_NOW}->{val}"})
    out.sort(key=lambda c: c["priority"], reverse=True)
    return out


def _already_queued() -> set:
    if not QUEUE.exists():
        return set()
    keys = set()
    for line in QUEUE.read_text().splitlines():
        if line.strip():
            try:
                keys.add(json.dumps(json.loads(line).get("param_delta"), sort_keys=True))
            except Exception:
                continue
    return keys


def _sensitivity_variants(base_delta: dict) -> list[dict]:
    """Stress-test a VALIDATED seed: small neighbor perturbations of its params, to confirm
    the edge isn't knife-edge dependent on one number (the user's correct reframe of
    combinatorial search — stress survivors, don't fish)."""
    out = []
    sw = base_delta.get("signal_weights")
    if sw:
        for d in (-0.2, -0.1, 0.1, 0.2):
            v = dict(sw)
            if "rate_weight" in v:
                v["rate_weight"] = round(min(max(v["rate_weight"] + d, 0.0), 1.0), 2)
            if "irp_weight" in v:
                v["irp_weight"] = round(min(max(v.get("irp_weight", 0.0) - d, 0.0), 1.0), 2)
            out.append({"subsystem": "forex", "param_delta": {"signal_weights": v},
                        "priority": 1.0, "label": f"sensitivity rate={v.get('rate_weight')}/irp={v.get('irp_weight')}"})
    gates = base_delta.get("PAIR_VIX_GATES")
    if gates:
        for pair, thr in gates.items():
            for d in (-2, -1, 1, 2):
                out.append({"subsystem": "forex", "param_delta": {"PAIR_VIX_GATES": {pair: float(thr + d)}},
                            "priority": 1.0, "label": f"sensitivity {pair} VIX {thr}->{thr+d}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="max hypotheses to append this run")
    ap.add_argument("--sensitivity", metavar="HYP_ID",
                    help="stress-test a validated hypothesis: queue neighbor perturbations of its params")
    args = ap.parse_args()

    if args.sensitivity:
        items = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()] if QUEUE.exists() else []
        base = next((q for q in items if q.get("id") == args.sensitivity), None)
        if not base or not base.get("param_delta"):
            raise SystemExit(f"{args.sensitivity} not found / has no param_delta.")
        variants = _sensitivity_variants(base["param_delta"])
        with open(QUEUE, "a") as f:
            for i, v in enumerate(variants, 1):
                rec = {"id": f"SENS-{args.sensitivity}-{i:02d}", "subsystem": v["subsystem"],
                       "param_delta": v["param_delta"], "source": "sensitivity",
                       "of_seed": args.sensitivity, "priority": v["priority"], "label": v["label"],
                       "status": "QUEUED", "queued_at": _now()}
                f.write(json.dumps(rec) + "\n")
        print(f"Sensitivity: queued {len(variants)} neighbor variant(s) of {args.sensitivity}.")
        return

    from sovereign.oracle.edge_pipeline import EdgePipeline
    ep = EdgePipeline()
    queued = _already_queued()

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    existing_n = sum(1 for _ in QUEUE.read_text().splitlines()) if QUEUE.exists() else 0

    appended, skipped_dup, skipped_rej = 0, 0, 0
    with open(QUEUE, "a") as f:
        for c in _candidates():
            if appended >= args.n:
                break
            key = json.dumps(c["param_delta"], sort_keys=True)
            if key in queued:
                skipped_dup += 1
                continue
            # Rejection memory: don't queue a known-dead idea.
            skip, why = ep.should_skip_hypothesis(c)
            if skip:
                skipped_rej += 1
                continue
            hid = f"FACT-{existing_n + appended + 1:05d}"
            rec = {"id": hid, "subsystem": c["subsystem"], "param_delta": c["param_delta"],
                   "source": "combinatorial", "priority": c["priority"], "label": c["label"],
                   "status": "QUEUED", "queued_at": _now()}
            f.write(json.dumps(rec) + "\n")
            queued.add(key)
            appended += 1

    print(f"Combinatorial feeder: appended {appended}, skipped {skipped_dup} dup / "
          f"{skipped_rej} rejected-memory. Queue now {existing_n + appended} total.")


if __name__ == "__main__":
    main()
