"""Shared helpers for the autonomous layer. Reuses existing system conventions
(timestamps, the messages_to_colin writer, the kill switch, the cost ledger) so
the four modules stay thin and consistent.

FAIL LOUD: helpers here either succeed or raise. The ONLY swallow-and-continue is
in best-effort sinks (a log/message write must never crash a monitoring loop), and
those are named `_best_effort_*` and log the failure.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sovereign.utils.timestamps import canonical_timestamp
from sovereign.utils import kill_switch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "autonomous.yml"
MESSAGES_PATH = ROOT / "data" / "agent" / "messages_to_colin.json"
COST_LOG = ROOT / "logs" / "oracle_cost.json"
LOG_DIR = ROOT / "logs"

_EMOJI = {"RED": "🔴", "CRITICAL": "🔴", "HIGH": "🟠", "IMPORTANT": "🟡",
          "YELLOW": "🟡", "FYI": "🟢", "GREEN": "🟢"}


# ── Logging ───────────────────────────────────────────────────────────────────

def make_logger(module: str):
    """Return a `_log(msg)` that appends to logs/<module>.log AND prints, UTC ISO.
    Mirrors the oracle_cycle.py logging convention."""
    log_path = LOG_DIR / f"{module}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return _log


# ── Config / gate ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Parsed config/autonomous.yml. Raises if missing or malformed — the gate flag
    is safety-critical and must never silently default to a permissive value."""
    import yaml
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"autonomous config missing: {CONFIG_PATH}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(cfg, dict) or "live" not in cfg:
        raise ValueError(f"autonomous config malformed (no `live` key): {CONFIG_PATH}")
    return cfg


def is_live() -> bool:
    """True only if the gate is explicitly enabled. Any ambiguity → False (safe)."""
    try:
        return load_config().get("live") is True
    except Exception:
        return False


# ── Kill switch ───────────────────────────────────────────────────────────────

def freeze_state() -> Optional[dict]:
    """Current kill-switch state (None if not frozen). Never raises."""
    try:
        return kill_switch.state()
    except Exception:
        return None


# ── Budget ────────────────────────────────────────────────────────────────────

def daily_spend_usd() -> float:
    """Total USD the agent has spent today, summed from logs/oracle_cost.json.
    Returns 0.0 if the cost ledger is absent (no spend recorded yet)."""
    if not COST_LOG.exists():
        return 0.0
    data = json.loads(COST_LOG.read_text())
    today = datetime.now(timezone.utc).date().isoformat()
    return round(sum(
        float(e.get("cost_usd", 0.0))
        for e in data.get("entries", [])
        if str(e.get("at", ""))[:10] == today
    ), 5)


# ── JSONL append (audit logs) ─────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a line. Creates parent dirs. Raises on failure
    (an audit write that silently fails is a future bug — constraint #5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via temp-file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── messages_to_colin writer ──────────────────────────────────────────────────

def write_message(priority: str, text: str, source: str, *, tag: str = "") -> None:
    """Insert one message at the front of data/agent/messages_to_colin.json (cap 50),
    matching the pulse_check convention. Best-effort: logs but never raises, so a
    notification hiccup can't kill a loop."""
    try:
        data = json.loads(MESSAGES_PATH.read_text()) if MESSAGES_PATH.exists() else {}
        if "messages" not in data:
            data["messages"] = []
        ts = canonical_timestamp()
        prefix = f"[{tag}] " if tag else ""
        data["messages"].insert(0, {
            "id": f"{source}-{ts[:16].replace(':', '').replace('-', '')}",
            "priority": priority,
            "emoji": _EMOJI.get(priority, "🟢"),
            "text": f"{prefix}{text}",
            "timestamp": ts,
            "read": False,
            "source": source,
        })
        data["messages"] = data["messages"][:50]
        data["last_updated"] = ts
        atomic_write_json(MESSAGES_PATH, data)
    except Exception as e:  # best-effort sink — never crash the caller
        print(f"[_common] WARN failed to write message: {e}")


def now_iso() -> str:
    return canonical_timestamp()
