#!/usr/bin/env python3
"""
Oracle session-open briefing — runs ~15 min after the forex scan (08:15 UTC weekdays).
=====================================================================================

Oracle's live-trading role is NOT to place trades (DecisionChain/forex_live_scan do that).
It OBSERVES and CONTEXTUALIZES: what the scanner saw, open positions, loop health, concerns.

Deterministic by design — NO LLM/API-key dependency, so a scheduled run can't die on a
missing key. Reads the latest forex_scan.log entry, open OANDA positions (fail-safe), and
loop_health_status.json; composes a one-paragraph briefing → data/agent/oracle_briefing_morning.json
(the dashboard reads it). LLM enrichment can be layered later.

Usage:  python3 scripts/oracle_session_open.py
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.ERROR)
for lib in ("oandapyV20", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

FOREX_LOG = ROOT / "logs" / "forex_scan.log"
LOOP_HEALTH = ROOT / "data" / "oracle" / "loop_health_status.json"
OUT = ROOT / "data" / "agent" / "oracle_briefing_morning.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_scan() -> list[dict]:
    """All log records from the most recent forex scan run (same timestamp prefix)."""
    if not FOREX_LOG.exists():
        return []
    recs = []
    for line in FOREX_LOG.read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    if not recs:
        return []
    latest_ts = recs[-1].get("timestamp")
    return [r for r in recs if r.get("timestamp") == latest_ts]


def _open_positions() -> list[dict]:
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        return OandaBridge().get_open_trades()  # fail-safe in bridge; returns [] on error
    except Exception:
        return []


def _loops_down() -> list[str]:
    try:
        return json.loads(LOOP_HEALTH.read_text()).get("down", [])
    except Exception:
        return []


def build_briefing() -> dict:
    scan = _latest_scan()
    positions = _open_positions()
    down = _loops_down()

    # What the scanner saw
    if not scan:
        scan_line = "Forex scan has not produced output yet."
    else:
        verdicts = [r.get("verdict") for r in scan]
        if verdicts == ["SIT"]:
            scan_line = f"Scanner SIT — {scan[0].get('reason')}."
        elif verdicts == ["NO_SIGNALS"]:
            scan_line = "No qualifying forex setup today (rate differentials within normal range)."
        else:
            would = [r for r in scan if r.get("verdict") in ("WOULD_PLACE", "PLACED")]
            denied = [r for r in scan if r.get("verdict") == "DENIED"]
            parts = []
            for r in would:
                parts.append(f"{r['pair']} {r['direction']} ({r['verdict']}, risk {r.get('risk_pct',0):.2%})")
            line = "; ".join(parts) if parts else "no actionable signals"
            scan_line = f"Scanner: {line}." + (f" {len(denied)} denied by risk gate." if denied else "")

    pos_line = (f"{len(positions)} open position(s)." if positions else "No open positions.")
    health_line = (f"⚠️ loops down: {', '.join(down)}." if down else "All monitored loops alive.")

    briefing = (f"Session open {_now()[:16]}Z. {scan_line} {pos_line} {health_line} "
                f"Forex macro is a swing edge (~1 trade/pair/month) — a no-signal day is the strategy "
                f"working, not failing. Watching for rate-differential divergence to widen past threshold.")

    payload = {
        "generated_at": _now(),
        "briefing": briefing,
        "scan_verdicts": [r.get("verdict") for r in scan],
        "open_positions": len(positions),
        "loops_down": down,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    p = build_briefing()
    print(p["briefing"])
