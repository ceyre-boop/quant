#!/usr/bin/env python3
"""
Edge Factory morning report (Track C).
=======================================

Your baseline of trust: a glance that says the scientific method ran N times yesterday and
surfaced the few candidates worth your attention. Reads the factory ledger, the queue, and the
review queue; writes data/research/factory_report.json for the dashboard.

Usage:  python3 scripts/factory_morning_report.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "research" / "factory_ledger.jsonl"
QUEUE = ROOT / "data" / "research" / "hypothesis_queue.jsonl"
REVIEW = ROOT / "data" / "oracle" / "edge_review_queue.json"
TEST_COUNT = ROOT / "data" / "research" / "factory_test_count.json"
OUT = ROOT / "data" / "research" / "factory_report.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    ledger = _jsonl(LEDGER)
    queue = _jsonl(QUEUE)
    cutoff = _now() - timedelta(hours=24)

    def _recent(r):
        try:
            return datetime.fromisoformat(str(r.get("tested_at", "")).replace("Z", "+00:00")) >= cutoff
        except Exception:
            return False

    last24 = [r for r in ledger if _recent(r)]
    by_status_24 = Counter(r.get("status") for r in last24)
    queue_depth = sum(1 for q in queue if q.get("status") == "QUEUED")
    candidates = [q for q in queue if q.get("status") == "FACTORY_CANDIDATE"]
    total_tests = (json.loads(TEST_COUNT.read_text()).get("total", len(ledger))
                   if TEST_COUNT.exists() else len(ledger))
    try:
        pending_review = json.loads(REVIEW.read_text()).get("pending", []) if REVIEW.exists() else []
    except Exception:
        pending_review = []

    report = {
        "generated_at": _now().isoformat(),
        "tested_last_24h": len(last24),
        "by_status_last_24h": dict(by_status_24),
        "validated_candidates_total": len(candidates),
        "queue_depth": queue_depth,
        "factory_tests_all_time": total_tests,
        "candidates_pending_your_review": [
            {"id": c["id"], "label": c.get("label"), "p_value": c.get("p_value"),
             "holdout": c.get("holdout"), "reason": c.get("reason")}
            for c in candidates
        ],
        "review_queue_pending_approval": len(pending_review),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print("=" * 56)
    print("EDGE FACTORY — MORNING REPORT")
    print("=" * 56)
    print(f"  Tested last 24h: {len(last24)}  {dict(by_status_24)}")
    print(f"  Queue depth: {queue_depth}  |  All-time factory tests: {total_tests}")
    print(f"  FACTORY_CANDIDATEs (survived BH + 2025 holdout): {len(candidates)}")
    for c in candidates:
        print(f"    → {c['id']} {c.get('label')}  p={c.get('p_value')}  {c.get('holdout','')}")
    print(f"  Pending YOUR approval (approve_edge.py): {len(pending_review)}")
    if not candidates:
        print("  No validated candidates — that is the factory working, not failing.")
    print("=" * 56)


if __name__ == "__main__":
    main()
