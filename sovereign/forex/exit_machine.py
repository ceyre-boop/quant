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

import numpy as np


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


def decide_exit_vec(
    direction, stop_price, best_price, worst_price, hold_count, hold_limit,
    close, atr_pct, signal, hold_today, donchian_exit_low, cfg: ExitConfig,
):
    """Vectorized BYTE-IDENTICAL port of `decide_exit` over N independent positions stepping one bar.

    Every input is a length-N array (`direction`/`hold_count`/`hold_limit`/`signal`/`hold_today` integer,
    the rest float); `cfg` is a single ExitConfig shared across the batch. Returns
    `(decision, best_price, worst_price, hold_count, reentry_signal)` as length-N arrays — the same
    quantities the scalar returns, computed with identical arithmetic and the identical priority order.

    This exists ONLY for speed (stepping 10k rollouts in lockstep). It is gated by
    tests/test_exit_machine.py::test_decide_exit_vec_parity, which fuzzes it against the scalar; the
    scalar `decide_exit` remains the single source of truth for the re-trace / parity path. If parity
    ever breaks, callers must fall back to the scalar.
    """
    direction = np.asarray(direction, dtype=np.int64)
    stop_price = np.asarray(stop_price, dtype=np.float64)
    best_price = np.maximum(np.asarray(best_price, dtype=np.float64), close)
    worst_price = np.minimum(np.asarray(worst_price, dtype=np.float64), close)
    hold_count = np.asarray(hold_count, dtype=np.int64) + 1
    hold_limit = np.asarray(hold_limit, dtype=np.int64)
    price = np.asarray(close, dtype=np.float64)
    atr_pct_now = np.maximum(np.asarray(atr_pct, dtype=np.float64), 1e-6)
    signal = np.asarray(signal, dtype=np.int64)
    hold_today = np.asarray(hold_today, dtype=np.int64)
    donchian_exit_low = np.asarray(donchian_exit_low, dtype=np.float64)
    n = price.shape[0]
    is_long = direction == 1

    trail_hit = np.zeros(n, dtype=bool)
    if cfg.trailing_atr_mult > 0:
        trail_stop_long = best_price - (cfg.trailing_atr_mult * atr_pct_now * best_price)
        trail_stop_short = worst_price + (cfg.trailing_atr_mult * atr_pct_now * worst_price)
        trail_hit = np.where(is_long, price <= trail_stop_long, price >= trail_stop_short)

    donchian_hit = np.zeros(n, dtype=bool)
    if cfg.strict_mode:
        donchian_hit = is_long & ~np.isnan(donchian_exit_low) & (price < donchian_exit_low)

    stop_hit = np.where(is_long, price <= stop_price, price >= stop_price)
    reversal = (signal != 0) & (signal != direction)
    time_exit = (hold_count >= hold_limit) if not cfg.strict_mode else np.zeros(n, dtype=bool)
    if cfg.enable_cb_refresh:
        cb_refresh = (signal == direction) & (hold_today < 30) & (hold_count >= 20)
    else:
        cb_refresh = np.zeros(n, dtype=bool)

    # Priority: stop > trail > donchian > reversal > cb_refresh > time > HOLD.
    # Assign lowest-priority first so higher priority overwrites (mirrors the scalar elif chain).
    decision = np.full(n, int(ExitDecision.HOLD), dtype=np.int64)
    decision = np.where(time_exit, int(ExitDecision.TIME), decision)
    decision = np.where(cb_refresh, int(ExitDecision.CB_REFRESH), decision)
    decision = np.where(reversal, int(ExitDecision.REVERSAL), decision)
    decision = np.where(donchian_hit, int(ExitDecision.DONCHIAN), decision)
    decision = np.where(trail_hit, int(ExitDecision.TRAILING_ATR), decision)
    decision = np.where(stop_hit, int(ExitDecision.INITIAL_STOP), decision)

    reentry_signal = np.where((reversal | cb_refresh) & (signal != 0), signal, 0)
    return decision, best_price, worst_price, hold_count, reentry_signal
