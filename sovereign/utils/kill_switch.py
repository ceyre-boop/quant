"""
System Kill Switch — master freeze for the live trading path.
sovereign/utils/kill_switch.py

A single file `data/system/KILL_SWITCH` (JSON) is the master freeze. When present,
the TRADING/SIGNAL path self-skips while monitoring (pulse_check/loop_health) and
Oracle cognition (reflect/harvest/codify/briefing — all propose-only) keep running.

Two tiers:
  soft — freeze the trading/signal path (forex_live_scan placement, DecisionChain.evaluate)
  hard — ALSO block approve_edge.py (the only live-config mutator)

This is DISTINCT from capital_allocator's per-system HEALTH freeze (sizing→0 on
HEALTH_UNRELIABLE). This switch is a global, operator-controlled freeze.

Persists across reboots (file on disk). launchd keeps firing; loops self-skip — no
launchctl changes needed. Operate via `python3 scripts/alta.py {freeze,thaw,status}`.

Protection envelope: this cannot stop a human editing source in an editor. It stops
edited code from EXECUTING in the live path, and (hard) blocks live-config approval.
It contains execution and commitment, not editing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
KILL_SWITCH = ROOT / "data" / "system" / "KILL_SWITCH"
AUDIT_LOG = ROOT / "data" / "agent" / "param_change_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state() -> Optional[dict]:
    """Parsed freeze state, or None if not frozen. Never raises.

    A malformed switch file is treated as a HARD freeze (fail-safe): a corrupt
    KILL_SWITCH must never silently un-freeze the system."""
    if not KILL_SWITCH.exists():
        return None
    try:
        return json.loads(KILL_SWITCH.read_text())
    except Exception:
        return {"frozen_at": None, "by": "unknown", "mode": "hard",
                "reason": "UNPARSEABLE KILL_SWITCH file — treated as hard freeze (fail-safe)"}


def trading_frozen() -> Optional[dict]:
    """State dict if the trading/signal path is frozen (soft OR hard), else None."""
    return state()


def config_frozen() -> Optional[dict]:
    """State dict only under a HARD freeze (blocks approve_edge.py), else None."""
    s = state()
    return s if (s and s.get("mode") == "hard") else None


def _audit(change: str, reason: str, by: str) -> None:
    """Append to the existing param-change audit trail (Non-negotiable #4). Never raises."""
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps({
                "timestamp": _now(), "change": change,
                "rationale": reason, "approved_by": by,
            }) + "\n")
    except Exception:
        pass


def freeze(reason: str, *, hard: bool = False, by: str = "manual") -> dict:
    """Create the kill switch + log it. Returns the written state."""
    mode = "hard" if hard else "soft"
    payload = {"frozen_at": _now(), "by": by, "mode": mode, "reason": reason}
    KILL_SWITCH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH.write_text(json.dumps(payload, indent=2))
    _audit(f"SYSTEM FREEZE ({mode})", reason, by)
    return payload


def thaw(*, by: str = "manual") -> Optional[dict]:
    """Remove the kill switch + log it. Returns the prior state, or None if not frozen."""
    prior = state()
    if KILL_SWITCH.exists():
        try:
            KILL_SWITCH.unlink()
        except Exception:
            pass
    if prior:
        _audit("SYSTEM THAW", f"unfreeze (was: {prior.get('reason', '')})", by)
    return prior


def skip_if_frozen(component: str, *, logger=None) -> bool:
    """Loop convenience: True if the trading path is frozen (caller should exit cleanly).

    Call this AFTER the loop writes its heartbeat, so loop_health reports EXECUTION
    (FROZEN) rather than a false DOWN."""
    s = trading_frozen()
    if not s:
        return False
    msg = (f"{component}: SYSTEM FROZEN ({s.get('mode')}) — "
           f"{s.get('reason', '')}. Skipping trading path.")
    try:
        (logger or print)(msg)
    except Exception:
        print(msg)
    return True
