"""ES/NQ session logger — the Oracle's harvest source for this system.

Appends one JSON line per session (or brief) to data/es_nq/session_log.jsonl.
Own file, zero collision with data/futures/trade_log.jsonl. Oracle reads this
file during nightly HARVEST; this package never imports oracle.

Schema (brief Component 4): session_date, mode (BACKTEST|PAPER|BRIEF), bias
block, levels, structure trigger, trades[], session_R_total, bias_was_correct,
structure_improved_outcome, adaptive_vs_flat_delta, notes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SESSION_LOG = ROOT / "data" / "es_nq" / "session_log.jsonl"


def log_session(record: dict) -> None:
    """Append one session record. Adds ts; requires session_date and mode."""
    for key in ("session_date", "mode"):
        if key not in record:
            raise ValueError(f"session record missing required key: {key}")
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_brief(date: str, bias: dict, levels: Optional[dict], plan_text: str,
              prior_session: Optional[dict] = None) -> None:
    log_session({
        "session_date": date, "mode": "BRIEF", "bias": bias,
        "levels": levels, "plan": plan_text, "prior_session": prior_session,
    })


def read_sessions(since: Optional[str] = None, mode: Optional[str] = None) -> list[dict]:
    """All session records, optionally filtered by session_date >= since and mode."""
    if not SESSION_LOG.exists():
        return []
    out = []
    for line in SESSION_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)   # malformed line = loud failure, by design
        if since and rec.get("session_date", "") < since:
            continue
        if mode and rec.get("mode") != mode:
            continue
        out.append(rec)
    return out
