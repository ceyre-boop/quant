#!/usr/bin/env python3
"""Component 1 — Health Responder.

Watches data/oracle/loop_health_status.json. On a DOWN/CRITICAL loop it classifies
the failure and writes a structured fix request to data/agent/dispatch_queue.jsonl
for a human or a Claude Code worker to action.

DISPATCH + NOTIFY ONLY. This module NEVER restarts a loop or runs a fix itself
(honors constraints #1 never-auto-deploy-live and #3 never-touch-forex). Every
dispatch record carries `auto_dispatched: false`.

Schedule: launchd com.alta.health.responder, every 30 min, 24/7.
Direct:   python3 sovereign/autonomous/health_responder.py [--dry-run]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.autonomous import _common as C

LOOP_HEALTH = ROOT / "data" / "oracle" / "loop_health_status.json"
DISPATCH_QUEUE = ROOT / "data" / "agent" / "dispatch_queue.jsonl"
RESPONDER_LOG = ROOT / "data" / "agent" / "responder_log.jsonl"
LOG_GLOB_DIR = ROOT / "logs"

_log = C.make_logger("health_responder")

# Map a loop name to the log file(s) we scan for its real error, and the component
# a Claude Code worker would actually fix. Loops not listed fall back to <name>.log.
_LOOP_LOGS = {
    "forex_scan": ["forex_scan.err", "forex_scan.log", "forex_scan_launchd.log"],
    "morning_briefing": ["oracle_briefing.log", "launchd_err.log"],
    "ict_scanner": ["ict_scanner.log"],
    "pulse_check": ["pulse.log", "pulse.err"],
}


def _read_status() -> dict:
    """Parsed loop_health_status.json. Raises if absent/malformed — we must not
    silently treat a missing health file as 'all green'."""
    if not LOOP_HEALTH.exists():
        raise FileNotFoundError(f"loop health status missing: {LOOP_HEALTH}")
    import json
    return json.loads(LOOP_HEALTH.read_text())


def _scan_logs_for_error(loop: str) -> str | None:
    """Return the most recent ImportError/ModuleNotFound line from this loop's logs,
    or None. Bounded to the tail so a huge log never blows memory."""
    for fname in _LOOP_LOGS.get(loop, [f"{loop}.log"]):
        p = LOG_GLOB_DIR / fname
        if not p.exists():
            continue
        try:
            tail = p.read_text(errors="replace").splitlines()[-400:]
        except OSError:
            continue
        for line in reversed(tail):
            if "ImportError" in line or "ModuleNotFoundError" in line or "cannot import name" in line:
                return line.strip()
    return None


def _scan_logs_for_connection(loop: str) -> str | None:
    """Return a recent IB/OANDA connection-failure line for this loop, or None."""
    needles = ("Connectivity between IBKR", "connection refused", "Connection refused",
               "OANDA", "ConnectionError", "Max retries", "Errno 61")
    for fname in _LOOP_LOGS.get(loop, [f"{loop}.log"]):
        p = LOG_GLOB_DIR / fname
        if not p.exists():
            continue
        try:
            tail = p.read_text(errors="replace").splitlines()[-400:]
        except OSError:
            continue
        for line in reversed(tail):
            if any(n in line for n in needles):
                return line.strip()
    return None


def _classify(loop: str, info: dict) -> dict:
    """Return {failure_type, error, suggested_fix, priority} for one down loop."""
    import_err = _scan_logs_for_error(loop)
    if import_err:
        return {
            "failure_type": "IMPORT_ERROR",
            "error": import_err,
            "suggested_fix": f"Resolve the import in the module behind '{loop}'. "
                             "Check the package __all__/exports and the import path; "
                             "run the module directly to reproduce.",
            "priority": "HIGH",
        }
    conn_err = _scan_logs_for_connection(loop)
    if conn_err:
        return {
            "failure_type": "CONNECTION_DOWN",
            "error": conn_err,
            "suggested_fix": f"Upstream feed for '{loop}' is down. Verify IB Gateway / "
                             "OANDA session, then let the loop reconnect with backoff. "
                             "No restart dispatched — connection issues self-heal.",
            "priority": "IMPORTANT",
        }
    silence = info.get("silence_hours")
    threshold = info.get("threshold")
    if isinstance(silence, (int, float)) and isinstance(threshold, (int, float)) and silence > threshold:
        return {
            "failure_type": "DATA_STALE",
            "error": f"{loop} silent {silence}h (threshold {threshold}h); last "
                     f"heartbeat {info.get('last')}.",
            "suggested_fix": f"Restart the launchd job for '{loop}' "
                             f"(launchctl kickstart -k gui/$(id -u)/com.alta.<job>); "
                             "verify it writes a fresh heartbeat within one interval.",
            "priority": "HIGH",
        }
    return {
        "failure_type": "UNKNOWN",
        "error": f"{loop} status={info.get('status')} but no import/connection/stale "
                 "signal matched.",
        "suggested_fix": f"Manual triage required for '{loop}' — inspect its logs.",
        "priority": "IMPORTANT",
    }


def _recent_dispatches(within_hours: float) -> set[str]:
    """Components dispatched within the window — used to dedup."""
    if not DISPATCH_QUEUE.exists():
        return set()
    import json
    cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
    seen: set[str] = set()
    for line in DISPATCH_QUEUE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            ts = datetime.fromisoformat(rec["timestamp"])
            if ts >= cutoff:
                seen.add(rec.get("component", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return seen


def run(dry_run: bool = False) -> dict:
    """One responder pass. Returns a summary dict (also logged)."""
    cfg = C.load_config()
    dedup_hours = cfg.get("health_responder", {}).get("dedup_hours", 6)
    status = _read_status()
    down = status.get("down", [])
    frozen = bool(status.get("frozen") or C.freeze_state())

    _log(f"pass start — {len(down)} loop(s) down: {down or 'none'} | "
         f"frozen={frozen} | dry_run={dry_run}")

    recent = _recent_dispatches(dedup_hours)
    dispatched, skipped, escalated = [], [], []

    for loop in down:
        info = status.get("loops", {}).get(loop, {})
        verdict = _classify(loop, info)
        component = loop  # the loop name is the component a worker would fix/restart

        if component in recent:
            skipped.append(component)
            _log(f"  skip {component}: already dispatched within {dedup_hours}h")
            continue

        record = {
            "timestamp": C.now_iso(),
            "failure_type": verdict["failure_type"],
            "component": component,
            "error": verdict["error"],
            "suggested_fix": verdict["suggested_fix"],
            "auto_dispatched": False,   # dispatch + notify only — never True
            "priority": verdict["priority"],
        }

        if verdict["failure_type"] == "UNKNOWN":
            escalated.append(component)
            if not dry_run:
                C.write_message(verdict["priority"],
                                f"{component} DOWN — {verdict['error']}",
                                source="health_responder", tag="HEALTH")

        if dry_run:
            _log(f"  [dry-run] would dispatch {verdict['failure_type']} for {component}")
        else:
            C.append_jsonl(DISPATCH_QUEUE, record)
            C.append_jsonl(RESPONDER_LOG, {"action": "dispatch", **record})
            dispatched.append(component)
            _log(f"  dispatched {verdict['failure_type']} for {component} "
                 f"(priority {verdict['priority']})")

    summary = {
        "timestamp": C.now_iso(),
        "down": down,
        "dispatched": dispatched,
        "skipped_dedup": skipped,
        "escalated_unknown": escalated,
        "frozen": frozen,
        "dry_run": dry_run,
    }
    if not dry_run:
        C.append_jsonl(RESPONDER_LOG, {"action": "pass_summary", **summary})

    # Hand off to the escalation router (continuous triage). Best-effort.
    if not dry_run:
        try:
            from sovereign.autonomous import escalation_router
            escalation_router.route(trigger="health_responder")
        except Exception as e:
            _log(f"  WARN escalation_router handoff failed: {e}")

    _log(f"pass done — dispatched {len(dispatched)}, skipped {len(skipped)}, "
         f"escalated {len(escalated)}")
    return summary


def main() -> None:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Health Responder — dispatch fix requests for down loops.")
    parser.add_argument("--dry-run", action="store_true", help="classify and log but write nothing")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
