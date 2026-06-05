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
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make `scripts.*` and `sovereign.*` importable regardless of how launchd invokes us
# (the plist runs `python3 scripts/oracle_session_open.py`, so sys.path[0] is scripts/, not ROOT).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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


def _refresh_analyst_briefing() -> None:
    """Refresh the analyst-narrative briefing BEFORE composing the session summary.

    morning_market_briefing.build() rewrites data/oracle/market_briefings/latest.json — the
    store reflect_cycle._load_market_briefing() feeds to Oracle's REFLECT — and stamps
    logs/.heartbeat_morning_briefing, which is the ONLY signal loop_health uses for the
    morning_briefing loop. Nothing else schedules that engine, so without this call Oracle
    reflects on a stale market read and loop_health false-alarms RED every 2h.

    Fail-safe: build() is fully _safe-wrapped internally (deterministic fallback, no API key
    required), but we still guard the call so a briefing-engine error can never crash the
    session-open summary — and we stamp the heartbeat ourselves as a last resort so the
    monitor never goes dark on us."""
    try:
        from scripts.morning_market_briefing import build as _build_briefing
        _build_briefing()
    except Exception as exc:
        logging.getLogger("oracle_session_open").warning(
            "analyst briefing refresh failed (session summary continues): %s", exc)
        try:
            hb = ROOT / "logs" / ".heartbeat_morning_briefing"
            hb.parent.mkdir(parents=True, exist_ok=True)
            hb.write_text(_now())
        except Exception:
            pass


def _read_regime() -> dict:
    try:
        d = json.loads((ROOT / "data" / "research" / "nqes_regime.json").read_text())
        return {"regime": d.get("regime"), "nq_last": d.get("nq_last"), "es_last": d.get("es_last")}
    except Exception:
        return {}


def build_briefing() -> dict:
    # Refresh Oracle's analyst briefing + morning_briefing heartbeat FIRST, so the loop_health
    # read below reflects this run and Oracle's next REFLECT consumes a fresh market read.
    _refresh_analyst_briefing()

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

    # NQ/ES regime (research input — what kind of market, for sizing/trust; not a trade signal).
    try:
        nq = json.loads((ROOT / "data" / "research" / "nqes_regime.json").read_text())
        nqes_line = f"NQ/ES regime: {nq.get('regime')} ({nq.get('read','')[:80]})."
    except Exception:
        nqes_line = "NQ/ES regime: not computed yet."

    briefing = (f"Session open {_now()[:16]}Z. {scan_line} {pos_line} {health_line} {nqes_line} "
                f"Forex macro is a swing edge (~1 trade/pair/month) — a no-signal day is the strategy "
                f"working, not failing. Watching for rate-differential divergence to widen past threshold.")

    payload = {
        "generated_at": _now(),
        "briefing": briefing,
        "scan_verdicts": [r.get("verdict") for r in scan],
        "open_positions": len(positions),
        "loops_down": down,
        "nqes_regime": _read_regime(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    p = build_briefing()
    print(p["briefing"])
