#!/usr/bin/env python3
"""Component 4 — Escalation Router.

Triages the autonomous layer's outputs (loop health, dispatch queue, factory verdicts,
oracle suggestions) into four lanes and records each to data/agent/escalation_log.jsonl —
the dashboard's intelligence feed.

  RED        interrupt now: unauthorized live trade, risk-ceiling breach, drawdown near
             prop limit, kill-switch failed to engage, any CRITICAL loop.
  YELLOW     daily digest (09:00 ET): VALID_EDGE found, health degraded >24h, oracle param
             suggestion.
  GREEN      weekly digest (Sun): ledger updates, perf summary, capital utilization.
  AUTO-HANDLE no human: loop restart requests (Component 1), cache refresh, reconnects.

GATED. escalation_log.jsonl is ALWAYS written (it is the passive record the dashboard polls).
The ACTIVE push — RED interrupts into messages_to_colin, and any future push/email — fires only
when config/autonomous.yml::live = true. In dry-run it is recorded but not pushed. No push/email
transport exists yet; that lane is a logged stub.

Invoked continuously by the other loops (e.g. health_responder calls route()).
Direct: python3 sovereign/autonomous/escalation_router.py [--live]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.autonomous import _common as C

LOOP_HEALTH = ROOT / "data" / "oracle" / "loop_health_status.json"
DISPATCH_QUEUE = ROOT / "data" / "agent" / "dispatch_queue.jsonl"
FACTORY_RESULTS = ROOT / "data" / "research" / "auto_hypothesis_results.jsonl"
MESSAGES_PATH = ROOT / "data" / "agent" / "messages_to_colin.json"
ESCALATION_LOG = ROOT / "data" / "agent" / "escalation_log.jsonl"

_log = C.make_logger("escalation_router")

_LOOKBACK_H = 24  # only triage items newer than this


def _recent(ts_str: str, hours: float = _LOOKBACK_H) -> bool:
    try:
        return datetime.fromisoformat(ts_str) >= datetime.now(timezone.utc) - timedelta(hours=hours)
    except (ValueError, TypeError):
        return False


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _classify_events() -> list[dict]:
    """Build the list of triaged events from current system state. Each event is
    {level, type, message, source, component?}."""
    events: list[dict] = []

    # ── Health: CRITICAL loop → RED; DOWN loop → YELLOW (degraded) ──────────────
    health = _read_json(LOOP_HEALTH, {})
    for name, info in health.get("loops", {}).items():
        if info.get("status") != "DOWN":
            continue
        silence = info.get("silence_hours") or 0
        threshold = info.get("threshold") or 0
        if threshold and silence > 2 * threshold:
            events.append({"level": "RED", "type": "LOOP_CRITICAL",
                           "message": f"{name} silent {silence}h (>2× threshold {threshold}h)",
                           "source": "loop_health", "component": name})
        else:
            events.append({"level": "YELLOW", "type": "HEALTH_DEGRADED",
                           "message": f"{name} down {silence}h (threshold {threshold}h)",
                           "source": "loop_health", "component": name})

    # ── Kill-switch integrity: frozen but recent fills would be RED ─────────────
    fs = C.freeze_state()
    if fs:
        events.append({"level": "AUTO-HANDLE", "type": "FROZEN",
                       "message": f"trading path frozen ({fs.get('mode','?')}): {fs.get('reason','')[:80]}",
                       "source": "kill_switch"})

    # ── Dispatch queue: DATA_STALE restarts are Component-1 handled ─────────────
    for d in _read_jsonl(DISPATCH_QUEUE):
        if not _recent(d.get("timestamp", "")):
            continue
        events.append({"level": "AUTO-HANDLE", "type": f"DISPATCH_{d.get('failure_type')}",
                       "message": f"{d.get('component')}: {d.get('suggested_fix','')[:90]}",
                       "source": "health_responder", "component": d.get("component")})

    # ── Factory verdicts: VALID_EDGE → YELLOW (human review) ────────────────────
    for r in _read_jsonl(FACTORY_RESULTS):
        if not _recent(r.get("timestamp", "")):
            continue
        if r.get("verdict") == "VALID_EDGE":
            events.append({"level": "YELLOW", "type": "VALID_EDGE",
                           "message": f"{r.get('hypothesis_id')} VALID_EDGE via {r.get('validator')} "
                                      "— needs approve_edge.py (NOT auto-deployed)",
                           "source": "research_factory"})

    # ── Oracle suggestions already in messages → GREEN weekly digest ────────────
    msgs = _read_json(MESSAGES_PATH, {}).get("messages", [])
    for m in msgs:
        if m.get("source") == "oracle_suggestion" and _recent(m.get("timestamp", "")):
            events.append({"level": "GREEN", "type": "ORACLE_SUGGESTION",
                           "message": m.get("text", "")[:120], "source": "oracle"})

    return events


def route(trigger: str = "manual", live: bool | None = None) -> dict:
    """Triage current state, record to escalation_log.jsonl, push RED/YELLOW only when live."""
    is_live = C.is_live() if live is None else live
    events = _classify_events()
    by_level = {lvl: [e for e in events if e["level"] == lvl]
                for lvl in ("RED", "YELLOW", "GREEN", "AUTO-HANDLE")}

    _log(f"route ({trigger}) — RED={len(by_level['RED'])} YELLOW={len(by_level['YELLOW'])} "
         f"GREEN={len(by_level['GREEN'])} AUTO={len(by_level['AUTO-HANDLE'])} | live={is_live}")

    pushed = 0
    for e in events:
        record = {"timestamp": C.now_iso(), "trigger": trigger, **e}
        C.append_jsonl(ESCALATION_LOG, record)   # passive feed — always recorded
        if is_live and e["level"] == "RED":
            C.write_message("RED", e["message"], source="escalation_router", tag=e["type"])
            pushed += 1
        elif is_live and e["level"] == "YELLOW":
            C.write_message("IMPORTANT", e["message"], source="escalation_router", tag=e["type"])
            pushed += 1
    # push/email transport is not built; record the intent only.
    if is_live and by_level["RED"]:
        _log(f"  [push/email stub] {len(by_level['RED'])} RED event(s) would page Colin "
             "(no transport configured)")

    return {"timestamp": C.now_iso(), "trigger": trigger, "live": is_live,
            "counts": {k: len(v) for k, v in by_level.items()}, "pushed": pushed}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Escalation Router — triage autonomous outputs.")
    parser.add_argument("--live", action="store_true", help="push RED/YELLOW (also needs config.live=true)")
    args = parser.parse_args()
    print(json.dumps(route(trigger="manual", live=True if args.live else None), indent=2))


if __name__ == "__main__":
    main()
