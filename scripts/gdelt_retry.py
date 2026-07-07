#!/usr/bin/env python3
"""Off-peak GDELT backfill retry (TICK-016 / Day-3 E1).

The paced GDELT fill has failed 3 consecutive daytime/evening attempts (Jul-3 19:30,
Jul-3 21:00, Jul-6 preflight — 8/8 throttled, latterly raw 30s ReadTimeouts). This
job retries in the 02:30 ET off-peak window. Semantics:

  - If the done-marker exists: exit 0 immediately (never re-fetch a filled feed).
  - Run the paced feeder (gdelt_feed.update — 5s spacing, 3 retries).
  - If ANY rows landed for ALL four pairs: rebuild the board, run the look-ahead
    auditor (fail LOUD on violations), write the done-marker, escalate SUCCESS to
    messages_to_colin (family BH is now unblocked: run
    `run_positioning_family_options.py --only HYP-080` then `--adjudicate`).
  - Otherwise: append one dated line to logs/gdelt_retry.log and exit 0 (tomorrow
    retries; no burst, no hammering — one paced pass per night).

Loaded via scripts/com.alta.gdelt_retry.plist (02:30 ET daily) — operator installs.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MARKER = ROOT / "data" / "system" / ".gdelt_backfill_done"
LOG = ROOT / "logs" / "gdelt_retry.log"
MESSAGES = ROOT / "data" / "agent" / "messages_to_colin.json"


def _log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {line}\n")


def _escalate_success(rows_by_pair: dict) -> None:
    try:
        doc = json.loads(MESSAGES.read_text()) if MESSAGES.exists() else []
        doc.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "severity": "IMPORTANT",
            "source": "gdelt_retry",
            "message": (f"GDELT backfill LANDED off-peak ({ {p: c.get('rows') for p, c in rows_by_pair.items()} }). "
                        "Board rebuilt, look-ahead audit green, done-marker written. Family BH is unblocked: "
                        "run scripts/research/run_positioning_family_options.py --only HYP-080, review "
                        "--adjudicate --dry-run, then --adjudicate."),
        })
        MESSAGES.write_text(json.dumps(doc, indent=2))
    except Exception as e:  # noqa: BLE001 — escalation must not mask the successful fill
        _log(f"escalation-write failed: {type(e).__name__}: {e}")


def main() -> int:
    if MARKER.exists():
        return 0
    from sovereign.sentiment import store, gdelt_feed, board_state
    con = store.connect()
    try:
        cov = gdelt_feed.update(con=con)
        landed = {p: c for p, c in cov.items() if (c.get("rows") or 0) > 0}
        if len(landed) < len(cov):
            _log(f"throttled again: {len(landed)}/{len(cov)} pairs landed rows — retry tomorrow")
            return 0
        rows = board_state.rebuild(con=con)
        from scripts.audit_look_ahead import audit
        viol = sum(r["violations"] for r in audit(con))
        if viol:
            _log(f"FILL LANDED but look-ahead audit FAILED ({viol} violations) — NOT marking done; investigate")
            return 1
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(datetime.now(timezone.utc).isoformat() + "\n")
        _log(f"SUCCESS: all pairs landed, board {rows} rows, audit 0 violations — marker written")
        _escalate_success(cov)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
