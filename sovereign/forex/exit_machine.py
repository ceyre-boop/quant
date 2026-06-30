"""sovereign/forex/exit_machine.py — the shared deterministic exit decision.

Extracted VERBATIM from `fast_backtester._simulate_forex_core`'s per-bar in-trade logic so the
backtester AND the live position-manager (L2) call ONE function — live == backtest by construction,
not by re-implementation. Behavior-preserving: the same (state, bar, cfg) yields the same decision
as the original inline logic (proven byte-identical by tests/test_exit_machine.py).

Canonical v015 runs `strict_mode=False` → DONCHIAN is unreached live; pyramiding is off. Those code
paths are preserved here for backtest parity but are not exercised by the live config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class ExitDecision(IntEnum):
    """Member values EQUAL fast_backtester.EXIT_REASON_* so `int(decision)` feeds the reasons array
    byte-identically. HOLD=0 is not a reason (no exit)."""
    HOLD = 0
    INITIAL_STOP = 1      # EXIT_REASON_STOP
    REVERSAL = 2          # EXIT_REASON_REVERSAL
    CB_REFRESH = 3        # EXIT_REASON_CB_REFRESH
    TIME = 4              # EXIT_REASON_TIME
    TRAILING_ATR = 5      # EXIT_REASON_TRAILING
    DONCHIAN = 6          # EXIT_REASON_DONCHIAN (strict-mode only; unreached under live v015)


@dataclass
class PositionState:
    """Per-trade state the exit decision reads. `stop_price` is the fixed initial ATR stop —
    decide_exit NEVER moves it (the backtest only checks it; the live manager amends the broker
    stop separately in L2 Step 3)."""
    direction: int            # +1 long / -1 short
    stop_price: float
    best_price: float
    worst_price: float
    hold_count: int
    hold_limit: int


@dataclass(frozen=True)
class BarContext:
    """Per-bar inputs (one trading day)."""
    close: float
    atr_pct: float            # today's ATR% (raw; decide_exit applies the 1e-6 floor, as the original did)
    signal: int               # today's carry signal (-1 / 0 / +1)
    hold_today: int           # the signal's hold_days for this bar
    donchian_exit_low: float  # NaN when N/A (non-strict / out of range)


@dataclass(frozen=True)
class ExitConfig:
    """Per-run config (constant across a backtest / a live session)."""
    stop_atr_mult: float
    trailing_atr_mult: float
    strict_mode: bool
    enable_cb_refresh: bool


@dataclass(frozen=True)
class ExitResult:
    decision: ExitDecision
    state: PositionState          # best_price / worst_price / hold_count advanced for this bar
    reentry_signal: int          # signal to re-enter with (0 = none) — set whenever (reversal OR
    #                              cb_refresh) fired, INDEPENDENT of which reason won the priority order
    #                              (a bar can be stop_hit AND reversal: reason=STOP yet re-enters).


def decide_exit(state: PositionState, bar: BarContext, cfg: ExitConfig) -> ExitResult:
    """One bar of the deterministic exit machine. Mirrors `_simulate_forex_core` lines 86-134 exactly."""
    hold_count = state.hold_count + 1
    best_price = max(state.best_price, bar.close)
    worst_price = min(state.worst_price, bar.close)
    atr_pct_now = max(bar.atr_pct, 1e-6)
    direction = state.direction
    price = bar.close

    trail_hit = False
    if cfg.trailing_atr_mult > 0:
        if direction == 1:
            trail_stop = best_price - (cfg.trailing_atr_mult * atr_pct_now * best_price)
            trail_hit = price <= trail_stop
        else:
            trail_stop = worst_price + (cfg.trailing_atr_mult * atr_pct_now * worst_price)
            trail_hit = price >= trail_stop

    donchian_hit = False
    if cfg.strict_mode and direction == 1 and not math.isnan(bar.donchian_exit_low):
        donchian_hit = price < float(bar.donchian_exit_low)

    stop_hit = (price <= state.stop_price) if direction == 1 else (price >= state.stop_price)
    reversal = bar.signal != 0 and bar.signal != direction
    time_exit = (hold_count >= state.hold_limit) and not cfg.strict_mode
    cb_refresh = (
        cfg.enable_cb_refresh and bar.signal == direction and bar.hold_today < 30 and hold_count >= 20
    )

    if stop_hit:
        decision = ExitDecision.INITIAL_STOP
    elif trail_hit:
        decision = ExitDecision.TRAILING_ATR
    elif donchian_hit:
        decision = ExitDecision.DONCHIAN
    elif reversal:
        decision = ExitDecision.REVERSAL
    elif cb_refresh:
        decision = ExitDecision.CB_REFRESH
    elif time_exit:
        decision = ExitDecision.TIME
    else:
        decision = ExitDecision.HOLD

    reentry_signal = bar.signal if (reversal or cb_refresh) and bar.signal != 0 else 0
    new_state = PositionState(direction, state.stop_price, best_price, worst_price, hold_count, state.hold_limit)
    return ExitResult(decision, new_state, reentry_signal)
