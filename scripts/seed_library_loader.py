#!/usr/bin/env python3
"""
Seed library loader — feed pre-registered academic factors into the factory queue.
==================================================================================

Reads data/research/seed_library.jsonl. Testable-now seeds (expressible as a param_delta
on the existing backtester — carry/IRP-value re-weightings) are appended to the factory
hypothesis_queue flagged `preregistered=True` with their citation. Seeds needing factors the
backtester doesn't compute (PPP/vol/dollar/term-spread/cross-asset/microstructure) are NOT
queued — they're the prioritized NEEDS_FACTOR backlog.

The `preregistered` flag is the whole point: when results come in, the claim is "of N
pre-registered academic factors, X survived BH on the 2025 holdout" — defensible science,
unlike "of 10,000 combos, X survived" (which multiple-testing math kills).

Usage:  python3 scripts/seed_library_loader.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "research" / "seed_library.jsonl"
QUEUE = ROOT / "data" / "research" / "hypothesis_queue.jsonl"
LEDGER = ROOT / "data" / "research" / "factory_ledger.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    if not SEEDS.exists():
        raise SystemExit(f"No seed library at {SEEDS}")
    seeds = _jsonl(SEEDS)
    tested = {r.get("id") for r in _jsonl(LEDGER)}
    queued = _jsonl(QUEUE)
    queued_ids = {q.get("id") for q in queued}
    queued_keys = {json.dumps(q.get("param_delta"), sort_keys=True) for q in queued}

    loaded, parked, skipped = 0, 0, 0
    with open(QUEUE, "a") as f:
        for s in seeds:
            sid = f"SEED_{s['id']}"
            if not s.get("testable_now"):
                parked += 1
                continue
            key = json.dumps(s["preregistered_params"], sort_keys=True)
            if sid in tested or sid in queued_ids or key in queued_keys:
                skipped += 1
                continue
            rec = {
                "id": sid,
                "subsystem": "forex" if s.get("family") in ("carry", "value", "blend") else s.get("family"),
                "param_delta": s["preregistered_params"],
                "source": "seed_library",
                "preregistered": True,
                "citation": s["source"],
                "family": s["family"],
                "expected_direction": s.get("expected_direction"),
                "expected_sharpe_range": s.get("expected_sharpe_range"),
                "label": s["name"],
                "status": "QUEUED",
                "queued_at": _now(),
            }
            f.write(json.dumps(rec) + "\n")
            loaded += 1

    print(f"Seed loader: queued {loaded} pre-registered testable factor(s); "
          f"parked {parked} NEEDS_FACTOR (backlog); skipped {skipped} already-known.")
    print(f"  Queue now {len(queued) + loaded} total. These will report as PRE-REGISTERED "
          f"(defensible: 'of N academic factors, X survived BH on holdout').")


if __name__ == "__main__":
    main()
