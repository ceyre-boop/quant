#!/usr/bin/env python3
"""Stage 4 of the loop, machine-readable: research/lessons.jsonl from HYPOTHESIS_LESSONS.md + the ledger.
One line per hypothesis: {id, tested, verdict, lesson, ledger_status, ledger_verdict, family}. Re-run
after every adjudication. The in-session generator reads this before proposing anything."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research" / "HYPOTHESIS_LESSONS.md"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
OUT = ROOT / "research" / "lessons.jsonl"


def main() -> int:
    ledger = {e.get("id"): e for e in json.loads(LEDGER.read_text())}
    rows = []
    for line in SRC.read_text().splitlines():
        if not line.startswith("| HYP-") and not line.startswith("| H1") and not re.match(r"^\| [A-Z0-9-]+ \|", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("id",):
            continue
        hid, tested, verdict, lesson = cells[0], cells[1], cells[2], cells[3]
        led = ledger.get(hid, {})
        rows.append({"id": hid, "tested": tested, "verdict": verdict, "lesson": lesson,
                     "ledger_status": led.get("status"), "ledger_verdict": led.get("verdict"),
                     "family": led.get("family") or led.get("source"), "correction": led.get("correction_note")})
    # ledger entries not yet in the lessons file (post-116)
    for hid, e in ledger.items():
        if hid and hid not in {r["id"] for r in rows} and str(hid).startswith("HYP-1"):
            rows.append({"id": hid, "tested": e.get("name"), "verdict": e.get("verdict"), "lesson": e.get("methodology_note"),
                         "ledger_status": e.get("status"), "ledger_verdict": e.get("verdict"), "family": e.get("source"), "correction": e.get("correction_note")})
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} lessons → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
