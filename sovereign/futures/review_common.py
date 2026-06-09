"""Shared helpers for the nightly + weekly review cycles (learning agent).

Pure, null-safe. Loads trade_log records, computes costed $ P&L, and groups by reasoning fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from sovereign.futures.config import contract_spec, round_turn_cost_usd

ET = ZoneInfo("America/New_York")


def load_trades(path: Path, include_simulated: bool = False) -> list[dict]:
    """Load trade-log records. By DEFAULT excludes data_quality=="SIMULATED" entries (dry-run,
    IB-disconnected, learning reps with no fill, REJECTED) so fabricated trades never reach the
    learning loop (Guard 3). Pass include_simulated=True to inspect them explicitly."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not include_simulated and rec.get("data_quality") == "SIMULATED":
                continue
            out.append(rec)
    return out


def session_date(rec: dict) -> Optional[str]:
    ts = rec.get("ts")
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).astimezone(ET).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


def trade_pnl_usd(rec: dict) -> Optional[float]:
    """Costed $ P&L for a CLOSED trade; None if still open or un-priceable."""
    entry, exit_p = rec.get("entry"), rec.get("exit")
    if entry is None or exit_p is None:
        return None
    direction = 1 if rec.get("direction") == "LONG" else -1
    n = int(rec.get("size_contracts", 1) or 1)
    inst = rec.get("instrument", "MES")
    try:
        dpp = contract_spec(inst)["dollars_per_point"]
        gross = (float(exit_p) - float(entry)) * direction * dpp * n
        return round(gross - round_turn_cost_usd(inst, n), 2)
    except Exception:
        return None


def is_win(rec: dict) -> Optional[bool]:
    r = rec.get("r_realized")
    return (r > 0) if isinstance(r, (int, float)) else None


def reasoning_field(rec: dict, key: str):
    """A field from the entry reasoning block (null-safe)."""
    return (rec.get("reasoning") or {}).get(key)


def winrate(recs: list[dict]) -> Optional[float]:
    scored = [is_win(r) for r in recs if is_win(r) is not None]
    return round(sum(scored) / len(scored), 3) if scored else None


def group_by(recs: list[dict], key_fn) -> dict:
    """Group closed/scored records by key_fn(rec) -> bucket; drop None keys."""
    out: dict = {}
    for r in recs:
        k = key_fn(r)
        if k is None:
            continue
        out.setdefault(k, []).append(r)
    return out
