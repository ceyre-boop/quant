"""
LoopHealth — sovereign/oracle/loop_health.py  (Gap 2: watch the loops)

A self-improving system must monitor whether its own machinery is running. The
49-hour scanner silence was caught by luck (a message bubbled up), not monitoring.
This turns that into a ~6-hour automated alert.

Design: heartbeats are INFERRED from existing artifacts (newest decision-log entry,
pulse state, reflection file) — no new instrumentation needed. Periodic loops that
go silent past their threshold raise a RED message; event-driven loops (edge_pipeline)
are reported as INFO, never alerted on (they're correctly idle most of the time).

Read-only + messaging + a status artifact. Touches no config. Hosted in the 2h pulse.
"""
from __future__ import annotations

import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISION_LOG_DIR = ROOT / "data" / "decision_logs"
HEARTBEAT_DIR = ROOT / "logs"   # per-loop heartbeat files: .heartbeat_<loop>
PULSE_STATE = ROOT / "data" / "oracle" / ".pulse_state.json"
REFLECTIONS_DIR = ROOT / "data" / "oracle" / "reflections"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
MESSAGES = ROOT / "data" / "agent" / "messages_to_colin.json"
STATUS = ROOT / "data" / "oracle" / "loop_health_status.json"

log = logging.getLogger("oracle.loop_health")

# during: 'always' | 'market_hours'. kind: 'periodic' (alert on silence) | 'event' (info only).
# IMPORTANT: heartbeats measure EXECUTION (the loop ran), NOT output (a signal/decision
# fired). A scanner that runs every 5 min but finds no signal is HEALTHY — using the
# decision log as its heartbeat is a false-alarm bug (fixed: scanners write a heartbeat
# file every invocation, before any session/signal gate).
HEARTBEAT_EXPECTATIONS = {
    "ict_scanner":       {"max_silence_hours": 0.5, "during": "market_hours", "kind": "periodic"},
    "forex_scan":        {"max_silence_hours": 30,  "during": "market_hours", "kind": "periodic"},
    "pulse_check":       {"max_silence_hours": 3,   "during": "always",       "kind": "periodic"},
    "decision_backfill": {"max_silence_hours": 3,   "during": "always",       "kind": "periodic"},
    "oracle_reflection": {"max_silence_hours": 26,  "during": "always",       "kind": "periodic"},
    "morning_briefing":  {"max_silence_hours": 26,  "during": "always",       "kind": "periodic"},
    "edge_pipeline":     {"max_silence_hours": 999, "during": "always",       "kind": "event"},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    except Exception:
        return None


def _is_market_hours(now: datetime | None = None) -> bool:
    """FX is open ~Sun 21:00 UTC → Fri 21:00 UTC. Coarse weekend check."""
    now = now or _now()
    wd = now.weekday()  # Mon=0 .. Sun=6
    if wd == 5:  # Saturday
        return False
    if wd == 6 and now.hour < 21:  # Sunday before open
        return False
    if wd == 4 and now.hour >= 21:  # Friday after close
        return False
    return True


# ── heartbeat inference (from existing artifacts) ──────────────────────────────
def _newest_decision_ts(system: str | None = None) -> datetime | None:
    newest = None
    for fp in glob.glob(str(DECISION_LOG_DIR / "decisions_*.jsonl")):
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if system and r.get("system") != system:
                continue
            ts = _parse(r.get("entry_timestamp"))
            if ts and (newest is None or ts > newest):
                newest = ts
    return newest


def _pulse_last() -> datetime | None:
    try:
        return _parse(json.loads(PULSE_STATE.read_text()).get("last_pulse_time"))
    except Exception:
        return None


def _reflection_last() -> datetime | None:
    files = list(REFLECTIONS_DIR.glob("*.json")) if REFLECTIONS_DIR.exists() else []
    if not files:
        return None
    return datetime.fromtimestamp(max(f.stat().st_mtime for f in files), tz=timezone.utc)


def _edge_pipeline_last() -> datetime | None:
    try:
        led = json.loads(LEDGER.read_text())
        stamps = []
        for h in led.get("ledger", []) + led.get("hypotheses", []):
            if isinstance(h, dict) and h.get("edge_pipeline_status"):
                # evaluated_at is on the verdict; fall back to any timestamp present
                ts = _parse(h.get("edge_pipeline_evaluated_at") or h.get("date_confirmed"))
                if ts:
                    stamps.append(ts)
        return max(stamps) if stamps else None
    except Exception:
        return None


def _heartbeat_file(loop: str) -> datetime | None:
    """Mtime of an explicit per-loop heartbeat file (logs/.heartbeat_<loop>), written
    by the loop on EVERY run. This is the true execution signal."""
    hb = HEARTBEAT_DIR / f".heartbeat_{loop}"
    if hb.exists():
        return datetime.fromtimestamp(hb.stat().st_mtime, tz=timezone.utc)
    return None


def _last_heartbeat(loop: str) -> datetime | None:
    # Prefer an explicit heartbeat file (true execution signal) when present.
    hb = _heartbeat_file(loop)
    if hb is not None:
        return hb
    # Fallbacks per loop (execution-based, NOT decision/output-based):
    if loop in ("pulse_check", "decision_backfill"):
        return _pulse_last()
    if loop == "oracle_reflection":
        return _reflection_last()
    if loop == "edge_pipeline":
        return _edge_pipeline_last()
    # Scanners with no heartbeat file yet → unknown (NOT "down"). Avoid the
    # decision-log conflation: absence of signals is not absence of execution.
    return None


def _message(subject: str, body: str, priority: str = "RED") -> None:
    try:
        data = json.loads(MESSAGES.read_text()) if MESSAGES.exists() else {"messages": []}
        data.setdefault("messages", []).insert(0, {
            "id": f"loophealth-{_now().isoformat()[:19].replace(':','').replace('-','')}",
            "timestamp": _now().isoformat(), "priority": priority, "source": "LOOP_HEALTH",
            "subject": subject, "message": body, "action_required": priority == "RED",
        })
        data["messages"] = data["messages"][:80]
        MESSAGES.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        log.warning("message write failed: %s", exc)


def check_all_loops(message: bool = True) -> dict:
    now = _now()
    loops, dead = {}, []
    for loop, expect in HEARTBEAT_EXPECTATIONS.items():
        last = _last_heartbeat(loop)
        silence = round((now - last).total_seconds() / 3600, 1) if last else None
        applicable = expect["during"] == "always" or _is_market_hours(now)
        status = "UNKNOWN"
        if expect["kind"] == "event":
            status = "EVENT_DRIVEN"
        elif last is None:
            status = "NO_DATA"
        elif silence > expect["max_silence_hours"] and applicable:
            status = "DOWN"
            dead.append((loop, silence))
        else:
            status = "ALIVE"
        loops[loop] = {"status": status, "last": last.isoformat() if last else None,
                       "silence_hours": silence, "threshold": expect["max_silence_hours"],
                       "kind": expect["kind"]}

    if dead and message:
        for loop, silence in dead:
            thr = HEARTBEAT_EXPECTATIONS[loop]["max_silence_hours"]
            _message(f"LOOP DOWN: {loop} silent {silence:.0f}h",
                     f"{loop} has not run for {silence:.0f}h (expected < {thr}h). "
                     f"The self-improvement machinery is degraded — investigate the scheduler "
                     f"(execute_daily.py / launchd) before relying on downstream loops.",
                     priority="RED")

    result = {"checked_at": now.isoformat(), "market_hours": _is_market_hours(now),
              "down": [d[0] for d in dead], "loops": loops}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = check_all_loops()
    print(f"Loop health @ {r['checked_at']} (market_hours={r['market_hours']})")
    for loop, s in r["loops"].items():
        flag = "🔴" if s["status"] == "DOWN" else ("⚪" if s["status"] in ("NO_DATA", "EVENT_DRIVEN") else "🟢")
        print(f"  {flag} {loop:18s} {s['status']:12s} silence={s['silence_hours']}h (< {s['threshold']}h)")
    if r["down"]:
        print(f"\nDOWN: {r['down']}")
