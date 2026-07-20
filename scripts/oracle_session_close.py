#!/usr/bin/env python3
"""
Oracle session-close briefing — runs on the closing bell (16:00 ET weekdays).
=============================================================================

The end-of-day mirror of oracle_session_open.py. Where the open brief sets the day up
(what the scanner saw, open positions, loop health), this one closes the day out: what
traded, what closed and how it resolved, the NAV move on the day, what is carried
overnight, and — critically — whether the closed loop actually learned today.

Deterministic by design — NO LLM/API-key dependency, so a scheduled run can't die on a
missing key. Reads the latest forex_scan.log entry, OANDA open/closed trades (fail-safe),
the live equity curve, and loop_health_status.json; composes a one-paragraph briefing →
data/agent/oracle_briefing_evening.json (the dashboard reads it). Stamps a heartbeat so
loop_health can monitor this loop the same way it monitors the morning brief.

Usage:  python3 scripts/oracle_session_close.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make `scripts.*` and `sovereign.*` importable regardless of how launchd invokes us
# (the plist runs `python3 scripts/oracle_session_close.py`, so sys.path[0] is scripts/, not ROOT).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env with an explicit path before any imports that need OANDA credentials.
# Without this, load_dotenv() in oanda_bridge uses CWD, which is wrong under launchd.
from dotenv import load_dotenv  # noqa: E402 (import after path setup is intentional)
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.ERROR)
for lib in ("oandapyV20", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

FOREX_LOG = ROOT / "logs" / "forex_scan.log"
LOOP_HEALTH = ROOT / "data" / "oracle" / "loop_health_status.json"
EQUITY_LIVE = ROOT / "data" / "agent" / "equity_curve_live.jsonl"
OUT = ROOT / "data" / "agent" / "oracle_briefing_evening.json"
HEARTBEAT = ROOT / "logs" / ".heartbeat_session_close"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def _closed_today() -> list[dict]:
    """OANDA trades whose closeTime falls on today's UTC date. Fail-safe → []."""
    try:
        from sovereign.execution.oanda_bridge import OandaBridge
        closed = OandaBridge().get_closed_trades(limit=100)
    except Exception:
        return []
    today = _today()
    return [t for t in closed if str(t.get("closeTime", ""))[:10] == today]


def _day_nav_change() -> dict:
    """First vs last NAV snapshot on today's UTC date from the live equity curve.

    Returns {} when there are not at least two snapshots today (e.g. a fresh day).
    """
    if not EQUITY_LIVE.exists():
        return {}
    today = _today()
    snaps = []
    for line in EQUITY_LIVE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("t", ""))[:10] == today and obj.get("nav") is not None:
            snaps.append(obj)
    if len(snaps) < 2:
        return {}
    first, last = snaps[0], snaps[-1]
    nav0, nav1 = float(first["nav"]), float(last["nav"])
    return {
        "nav_open": nav0,
        "nav_close": nav1,
        "delta_usd": round(nav1 - nav0, 2),
        "delta_pct": round((nav1 - nav0) / nav0 * 100, 3) if nav0 else 0.0,
    }


def _loops_down() -> list[str]:
    try:
        return json.loads(LOOP_HEALTH.read_text()).get("down", [])
    except Exception:
        return []


def _classify_realized(realized_pl: float) -> str:
    if realized_pl > 0:
        return "WIN"
    if realized_pl < 0:
        return "LOSS"
    return "SCRATCH"


def _stamp_heartbeat() -> None:
    """So loop_health can monitor this loop the same way it monitors the morning brief."""
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(_now())
    except Exception:
        pass


def build_briefing() -> dict:
    scan = _latest_scan()
    positions = _open_positions()
    closed = _closed_today()
    nav = _day_nav_change()
    down = _loops_down()

    # What the scanner concluded today (its last word).
    if not scan:
        scan_line = "Forex scan produced no output today."
    else:
        verdicts = [r.get("verdict") for r in scan]
        if verdicts == ["SIT"]:
            scan_line = f"Scanner sat out — {scan[0].get('reason')}."
        elif verdicts == ["NO_SIGNALS"]:
            scan_line = "No qualifying forex setup today (rate differentials within normal range)."
        else:
            placed = [r for r in scan if r.get("verdict") in ("WOULD_PLACE", "PLACED")]
            denied = [r for r in scan if r.get("verdict") == "DENIED"]
            line = "; ".join(f"{r['pair']} {r['direction']} ({r['verdict']})" for r in placed) or "no actionable signals"
            scan_line = f"Scanner: {line}." + (f" {len(denied)} denied by risk gate." if denied else "")

    # What closed today, and how it resolved.
    if closed:
        wins = sum(1 for t in closed if _classify_realized(float(t.get("realizedPL") or 0.0)) == "WIN")
        losses = sum(1 for t in closed if _classify_realized(float(t.get("realizedPL") or 0.0)) == "LOSS")
        realized = round(sum(float(t.get("realizedPL") or 0.0) for t in closed), 2)
        closed_line = (f"{len(closed)} trade(s) closed today: {wins}W/{losses}L, "
                       f"realized {realized:+.2f}. Oracle has these outcomes to learn from.")
    else:
        closed_line = "No trades closed today — nothing new for the loop to learn from."

    # NAV move on the day.
    if nav:
        nav_line = (f"NAV {nav['nav_open']:,.0f} → {nav['nav_close']:,.0f} "
                    f"({nav['delta_usd']:+,.2f}, {nav['delta_pct']:+.3f}%).")
    else:
        nav_line = "NAV: not enough snapshots today to compute a day change."

    pos_line = (f"{len(positions)} position(s) carried overnight." if positions
                else "Flat into the close — no overnight exposure.")
    health_line = (f"⚠️ loops down: {', '.join(down)}." if down else "All monitored loops alive.")

    briefing = (f"Session close {_now()[:16]}Z. {scan_line} {closed_line} {nav_line} {pos_line} "
                f"{health_line} Forex macro is a swing edge (~1 trade/pair/month) — a quiet day is the "
                f"strategy working, not failing. The day's closed outcomes feed Oracle's next reflection.")

    payload = {
        "generated_at": _now(),
        "briefing": briefing,
        "scan_verdicts": [r.get("verdict") for r in scan],
        "closed_today": len(closed),
        "closed_realized_pl": round(sum(float(t.get("realizedPL") or 0.0) for t in closed), 2),
        "nav_day": nav,
        "open_overnight": len(positions),
        "loops_down": down,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    _stamp_heartbeat()
    return payload


if __name__ == "__main__":
    p = build_briefing()
    print(p["briefing"])
