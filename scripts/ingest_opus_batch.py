#!/usr/bin/env python3
"""
ingest_opus_batch.py — Normalize and queue the Opus 2026-06-01 hypothesis batch.
==================================================================================

Two output paths:
  1. data/research/seed_library.jsonl  — archival record (all 100, full schema)
  2. data/research/hypothesis_queue.jsonl — factory queue

Status routing:
  QUEUED          — type=="param_delta": factory will test; may return INSUFFICIENT_DATA
  NEEDS_BACKTESTER — type=="strategy": requires new code; NOT written to factory queue

Why the split: edge_factory_worker.py calls it["param_delta"] directly — strategy-type
entries have no param_delta and would KeyError-crash the worker mid-batch.

BH note: family field is preserved on every queue entry. The factory's
derive_hypothesis_pvalues.py applies BH across all factory tests. Within-family
grouping is deferred until derive_hypothesis_pvalues.py gains a family-groupby path.

Usage:
    python3 scripts/ingest_opus_batch.py [--batch data/research/opus_batch_2026_06_01.jsonl]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SEEDS  = ROOT / "data" / "research" / "seed_library.jsonl"
QUEUE  = ROOT / "data" / "research" / "hypothesis_queue.jsonl"
LEDGER = ROOT / "data" / "research" / "factory_ledger.jsonl"

_FOREX_FAMILIES = {"carry", "momentum", "value", "combination", "calendar", "volatility"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _to_subsystem(hyp: dict) -> str:
    family = hyp.get("family", "")
    return "forex" if family in _FOREX_FAMILIES else family


def _to_queue_entry(hyp: dict) -> dict:
    """Convert a param_delta hypothesis to factory queue format."""
    return {
        "id":               "SEED_" + hyp["id"],
        "subsystem":        _to_subsystem(hyp),
        "param_delta":      {"research_params": hyp.get("params", {})},
        "source":           "seed_library",
        "source_batch":     "opus_2026_06_01",
        "preregistered":    True,
        "citation":         hyp["source"],
        "family":           hyp["family"],
        "expected_direction": hyp.get("expected_direction"),
        "label":            hyp["name"],
        "type":             hyp["type"],
        "status":           "QUEUED",
        "queued_at":        _now(),
    }


def _to_seed_entry(hyp: dict) -> dict:
    """Normalize to seed_library schema."""
    is_testable = hyp.get("type") == "param_delta"
    return {
        **hyp,
        "testable_now":          is_testable,
        "expected_failure_modes": [hyp["fails_in"]],
        "preregistered_params":  hyp.get("params", {}),
        "needs_factor":          None if is_testable else "new backtester strategy code",
        "status":                "QUEUED" if is_testable else "NEEDS_BACKTESTER",
        "source_batch":          "opus_2026_06_01",
        "ingested_at":           _now(),
    }


def main(batch_file: str) -> None:
    batch_path = ROOT / batch_file
    if not batch_path.exists():
        raise SystemExit(f"Batch file not found: {batch_path}")

    batch = _jsonl(batch_path)
    print(f"Loaded {len(batch)} hypotheses from {batch_path.name}")

    # Build dedup sets
    existing_seed_ids  = {s["id"] for s in _jsonl(SEEDS)}
    existing_queue_ids = {q["id"] for q in _jsonl(QUEUE)}
    existing_ledger_ids = {r["id"] for r in _jsonl(LEDGER)}
    already_known_queue = existing_queue_ids | existing_ledger_ids

    seed_added    = 0
    seed_skipped  = 0
    queued        = 0
    parked        = 0

    family_queued:  dict[str, int] = {}
    family_parked:  dict[str, int] = {}

    with SEEDS.open("a") as sf, QUEUE.open("a") as qf:
        for hyp in batch:
            hid    = hyp["id"]
            family = hyp.get("family", "?")

            # ── Seed library (archival) ──────────────────────────────────
            if hid in existing_seed_ids:
                seed_skipped += 1
            else:
                sf.write(json.dumps(_to_seed_entry(hyp)) + "\n")
                seed_added += 1

            # ── Factory queue (param_delta only) ─────────────────────────
            queue_id = "SEED_" + hid
            if hyp.get("type") == "param_delta":
                if queue_id in already_known_queue:
                    parked += 1
                    family_parked[family] = family_parked.get(family, 0) + 1
                else:
                    qf.write(json.dumps(_to_queue_entry(hyp)) + "\n")
                    queued += 1
                    family_queued[family] = family_queued.get(family, 0) + 1
            else:
                parked += 1
                family_parked[family] = family_parked.get(family, 0) + 1

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nOpus batch 2026-06-01: ingested {len(batch)} hypotheses")
    print(f"  → seed_library.jsonl:     {seed_added} appended "
          f"({seed_skipped} dupes skipped)")
    print(f"  → hypothesis_queue.jsonl: {queued} QUEUED (param_delta types, factory will test)")
    print(f"  → parked NEEDS_BACKTESTER: {parked - (queued - queued)} "
          f"({len(batch) - queued} total not queued)")

    all_families = sorted(set(list(family_queued) + list(family_parked)))
    print("\nFamily breakdown:")
    for fam in all_families:
        q = family_queued.get(fam, 0)
        p = family_parked.get(fam, 0)
        total = q + p
        note = ""
        if fam in ("macro", "cross_asset", "commodity", "positioning"):
            note = " (→ INSUFFICIENT_DATA expected)"
        print(f"  {fam} ({total}): {q} QUEUED{note}, {p} NEEDS_BACKTESTER")

    print(f"\nBH note: family field preserved on all {queued} queue entries.")
    print("  Within-family BH grouping is deferred (derive_hypothesis_pvalues.py "
          "currently applies global BH — family-groupby is a future enhancement).")
    print(f"Replication expectation: ~10-15% of {queued} factory tests survive "
          "BH on 2025+ holdout (FX literature base rate).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest Opus hypothesis batch into seed library and factory queue"
    )
    ap.add_argument(
        "--batch",
        default="data/research/opus_batch_2026_06_01.jsonl",
        help="Path to batch JSONL file (relative to repo root)",
    )
    args = ap.parse_args()
    main(args.batch)
