"""ES/NQ live scanner — intraday state machine shared by the paper runner.

Time is INJECTED (never read from the clock) so the exact live logic is
replay-testable, mirroring the futures sandbox's scalp_strategy discipline.

State flow per session:
  WAIT_SWEEP → WAIT_CONFIRM → ENTER (emit TradePlan) → IN_TRADE → (exit) →
  WAIT_SWEEP (next setup, if the ladder allows) ... until 15:55 flat / deadline.

The scanner only DETECTS and PLANS — order placement, fills, and the ladder
live in scripts/es_nq_paper_runner.py. This module stays pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from sovereign.es_nq.config import es_nq_params
from sovereign.es_nq.structure_gate import (
    Levels, Sweep, TradePlan, detect_confirmation, detect_sweep, plan_trade,
)


@dataclass
class ScannerState:
    bias_dir: str                      # UP | DOWN (NEUTRAL → scanner never built)
    levels: Levels
    instrument: str
    trades_done: int = 0
    in_trade: bool = False
    scan_from_idx: int = 0             # bars before this are consumed by prior trades
    last_plan: Optional[TradePlan] = None
    params: Optional[dict] = None


def scan_step(now: datetime, bars5: pd.DataFrame, state: ScannerState) -> Optional[TradePlan]:
    """One scan pass over session-to-date 5-min bars. Returns a TradePlan when a
    fresh setup has fully confirmed and the NEXT bar is open for entry, else None.

    The caller must set state.in_trade while a position is open and bump
    state.scan_from_idx past the exit bar after it closes.
    """
    p = state.params or es_nq_params()
    if state.bias_dir not in ("UP", "DOWN"):
        raise ValueError(f"scanner requires a directional bias, got {state.bias_dir}")
    if state.in_trade or state.trades_done >= p["sizing"]["max_trades_per_session"]:
        return None
    et = now.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    deadline_h, deadline_m = map(int, p["structure"]["entry_deadline_et"].split(":"))
    if (et.hour, et.minute) >= (deadline_h, deadline_m):
        return None
    window = bars5.iloc[state.scan_from_idx:]
    if len(window) < 3:
        return None
    sweep = detect_sweep(window, state.levels, state.bias_dir, p)
    if sweep is None:
        return None
    confirm = detect_confirmation(window, sweep, state.bias_dir, p)
    if confirm is None:
        return None
    # Entry bar must be the bar that is FORMING now (confirm bar fully closed).
    if confirm + 1 != len(window):
        # Confirmation happened earlier and the entry bar already passed — stale.
        return None
    plan = plan_trade_live(window, confirm, sweep, state, p)
    state.last_plan = plan
    return plan


def plan_trade_live(window: pd.DataFrame, confirm_idx: int, sweep: Sweep,
                    state: ScannerState, p: dict) -> Optional[TradePlan]:
    """Live variant of plan_trade: the entry bar has no Open yet, so the plan is
    priced off the confirmation bar Close (the market order goes out now)."""
    from sovereign.es_nq.config import contract_spec
    s = p["structure"]
    spec = contract_spec(state.instrument)
    tick = spec["tick"]
    raw_entry = float(window["Close"].iloc[confirm_idx])
    slip = p["costs"]["slippage_ticks_entry"] * tick
    buffer = s["stop_buffer_ticks"] * tick
    if state.bias_dir == "UP":
        entry = raw_entry + slip
        stop = sweep.extreme - buffer
        stop_pts = entry - stop
        direction = "LONG"
        t1, t2 = entry + s["t1_r"] * stop_pts, entry + s["t2_r"] * stop_pts
    else:
        entry = raw_entry - slip
        stop = sweep.extreme + buffer
        stop_pts = stop - entry
        direction = "SHORT"
        t1, t2 = entry - s["t1_r"] * stop_pts, entry - s["t2_r"] * stop_pts
    if stop_pts <= 0:
        return None
    return TradePlan(direction=direction, entry=round(entry, 4), stop=round(stop, 4),
                     t1=round(t1, 4), t2=round(t2, 4), stop_points=round(stop_pts, 4),
                     sweep=sweep, confirm_bar_idx=confirm_idx,
                     entry_bar_idx=confirm_idx + 1)
