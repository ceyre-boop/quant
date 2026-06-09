#!/usr/bin/env python3
"""
One-shot integrity migration: tag historical futures trade-log entries with data_quality.

Background: until the three integrity guards shipped, scripts/futures_monitor.py logged "shadow"
entries while IB Gateway was disconnected and in learning mode — including impossible values
(stop == entry, expected_r = 5031). Those records have no data_quality field, so the learning loop
(nightly/weekly review → Oracle) would ingest them as real. This script quarantines them.

Rule (only touches records MISSING data_quality):
  - a real fill (notes contain "order_id=")            → "LIVE_PAPER"
  - everything else (IB disconnected / dry-run /
    learning rep / macro hold / no order id / REJECTED) → "SIMULATED"

Records that already carry data_quality are left untouched. Backs up the log first.

Usage:  python3 scripts/futures_quarantine_simulated.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"


def _classify(rec: dict) -> str:
    notes = str(rec.get("notes", ""))
    if "order_id=" in notes:
        return "LIVE_PAPER"
    return "SIMULATED"


def main() -> int:
    ap = argparse.ArgumentParser(description="Quarantine untagged simulated futures entries")
    ap.add_argument("--dry-run", action="store_true", help="Report only; don't rewrite the log")
    args = ap.parse_args()

    if not TRADE_LOG.exists():
        print(f"no trade log at {TRADE_LOG} — nothing to do")
        return 0

    records, tagged_sim, tagged_live, already = [], 0, 0, 0
    for line in TRADE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "data_quality" in rec and rec["data_quality"]:
            already += 1
        else:
            dq = _classify(rec)
            rec["data_quality"] = dq
            if dq == "SIMULATED":
                tagged_sim += 1
            else:
                tagged_live += 1
        records.append(rec)

    print(f"  records: {len(records)} | already tagged: {already} | "
          f"newly SIMULATED: {tagged_sim} | newly LIVE_PAPER: {tagged_live}")

    if args.dry_run:
        print("  [dry-run] no changes written")
        return 0

    if tagged_sim or tagged_live:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = TRADE_LOG.with_name(f"trade_log.bak-{stamp}.jsonl")
        shutil.copy(TRADE_LOG, backup)
        TRADE_LOG.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"  backed up → {backup.name}")
        print(f"  rewrote {TRADE_LOG.relative_to(ROOT)} with data_quality on every record")
    else:
        print("  nothing to tag — all records already carry data_quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
