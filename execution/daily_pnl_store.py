"""Reconstructs today's realized daily_pnl_frac / consecutive_losses from fill_log.jsonl.

STAGED FOR TICK-044 — dormant until execution/harness.py's staged patch
(research/TICK-044_staged_patch.diff) is applied. Nothing imports this module yet.

Root cause it closes: `risk.AccountState.daily_pnl_frac` defaults to 0.0
(execution/risk.py:102) and is never populated from real fills anywhere in the
codebase — `execution/harness.py:401` constructs a fresh `AccountState` on every
invocation and nothing writes to `daily_pnl_frac` afterward, even within the same
run. `DAILY_LOSS_HALT` (execution/risk.py:171) is therefore permanently inert on
the standard 09:25 ET launchd path. See TICK-044 (tickets/backlog.md).

Reconstruction depends on `FillRecord.effective_risk_frac`, a field added by the
staged patch — it is the only way to turn a fill's `net_return` back into a
fraction of session-start equity, since notional/equity dollars are never
persisted. Fills logged BEFORE the patch lands have no `effective_risk_frac` and
are skipped, not guessed — no-silent-mocking (CLAUDE.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

FILL_LOG = "fill_log.jsonl"


@dataclass
class DailyState:
    daily_pnl_frac: float
    consecutive_losses: int
    n_fills_considered: int


def reconstruct_daily_state(day: date, out_dir: Path) -> DailyState:
    """Sum today's already-persisted fills into a daily_pnl_frac + loss streak.

    `SKIP_*` rows never reach here — they didn't move P&L. Rows missing
    `effective_risk_frac` or `net_return` are skipped rather than guessed.
    """
    fp = out_dir / FILL_LOG
    if not fp.exists():
        return DailyState(0.0, 0, 0)

    pnl_frac = 0.0
    streak = 0
    considered = 0
    day_str = str(day)
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("date") != day_str:
            continue
        if str(row.get("signal_type", "")).startswith("SKIP_"):
            continue
        net_return = row.get("net_return")
        eff_risk = row.get("effective_risk_frac")
        if net_return is None or eff_risk is None:
            continue
        contribution = float(net_return) * float(eff_risk)
        pnl_frac += contribution
        considered += 1
        streak = streak + 1 if contribution < 0 else 0

    return DailyState(daily_pnl_frac=pnl_frac, consecutive_losses=streak,
                      n_fills_considered=considered)
