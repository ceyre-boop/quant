"""
Fast simulation kernel for forex backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from sovereign.forex.exit_machine import (
    BarContext, ExitConfig, ExitDecision, PositionState, decide_exit,
)

HOLD_DAYS = 60
STOP_PCT = 0.04
EXIT_REASON_STOP = 1
EXIT_REASON_REVERSAL = 2
EXIT_REASON_CB_REFRESH = 3
EXIT_REASON_TIME = 4
EXIT_REASON_TRAILING = 5
EXIT_REASON_DONCHIAN = 6


def _rolling_min_prev(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum over the *previous* `window` bars (excludes current bar)."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for i in range(window, len(values)):
        out[i] = float(np.min(values[i - window:i]))
    return out


def _default_atr_pct(closes: np.ndarray, stop_pct: float, stop_atr_mult: float) -> np.ndarray:
    if len(closes) == 0:
        return np.zeros(0, dtype=np.float64)
    if stop_atr_mult <= 0:
        stop_atr_mult = 1.0
    fallback = max(stop_pct / stop_atr_mult, 1e-6)
    arr = np.full(len(closes), fallback, dtype=np.float64)
    if len(closes) < 15:
        return arr
    tr = np.abs(np.diff(closes, prepend=closes[0]))
    for i in range(14, len(closes)):
        atr = float(np.mean(tr[i - 13:i + 1]))
        arr[i] = max(atr / max(closes[i], 1e-9), 1e-6)
    return arr


def _simulate_forex_core(
    opens: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    hold_days: np.ndarray,
    stop_pct: float,
    atr_pcts: np.ndarray,
    stop_atr_mult: float,
    trailing_atr_mult: float,
    donchian_exit_lows: np.ndarray,
    strict_mode: bool,
    allow_pyramiding: bool,
    pyramid_step_atr: float,
    max_units: int,
    enable_cb_refresh: bool,
):
    """Core loop for trade simulation with ATR stops, trailing exits, and optional pyramiding."""
    entries, exits, dirs, pnls, holds, reasons, units_arr = [], [], [], [], [], [], []

    in_trade = False
    direction = 0
    entry_idx = 0
    hold_count = 0
    hold_limit = 0
    units = 1
    avg_entry = 0.0
    stop_price = 0.0
    best_price = 0.0
    worst_price = 0.0
    next_pyramid_price = 0.0

    _cfg = ExitConfig(stop_atr_mult, trailing_atr_mult, strict_mode, enable_cb_refresh)

    for i in range(len(closes)):
        signal_today = int(signals[i])
        hold_today = int(hold_days[i]) if i < len(hold_days) else HOLD_DAYS
        price = float(closes[i])
        atr_pct_now = float(atr_pcts[i]) if i < len(atr_pcts) else 0.0
        atr_pct_now = max(atr_pct_now, 1e-6)

        if in_trade:
            _res = decide_exit(
                PositionState(direction, stop_price, best_price, worst_price, hold_count, hold_limit),
                BarContext(
                    price,
                    float(atr_pcts[i]) if i < len(atr_pcts) else 0.0,
                    signal_today,
                    hold_today,
                    float(donchian_exit_lows[i]) if i < len(donchian_exit_lows) else float("nan"),
                ),
                _cfg,
            )
            best_price = _res.state.best_price
            worst_price = _res.state.worst_price
            hold_count = _res.state.hold_count

            if _res.decision != ExitDecision.HOLD:
                pnl = direction * (price / max(avg_entry, 1e-9) - 1.0) * units
                entries.append(entry_idx)
                exits.append(i)
                dirs.append(direction)
                pnls.append(float(pnl))
                holds.append(hold_count)
                units_arr.append(units)
                reasons.append(int(_res.decision))
                in_trade = False

                reenter_dir = _res.reentry_signal
                if reenter_dir != 0 and i + 1 < len(opens):
                    direction = int(reenter_dir)
                    entry_price = float(opens[i + 1])
                    entry_atr = max(atr_pct_now, 1e-6)
                    stop_dist = (
                        entry_price * stop_atr_mult * entry_atr
                        if stop_atr_mult > 0
                        else entry_price * stop_pct
                    )
                    stop_price = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
                    entry_idx = i + 1
                    hold_count = 0
                    hold_limit = max(hold_today, 1)
                    units = 1
                    avg_entry = entry_price
                    best_price = entry_price
                    worst_price = entry_price
                    next_pyramid_price = entry_price + (
                        direction * pyramid_step_atr * entry_price * entry_atr
                    )
                    in_trade = True
                continue

            if allow_pyramiding and units < max_units and pyramid_step_atr > 0:
                if direction == 1 and price >= next_pyramid_price:
                    units += 1
                    avg_entry = ((avg_entry * (units - 1)) + price) / units
                    next_pyramid_price = price + (pyramid_step_atr * atr_pct_now * price)
                elif direction == -1 and price <= next_pyramid_price:
                    units += 1
                    avg_entry = ((avg_entry * (units - 1)) + price) / units
                    next_pyramid_price = price - (pyramid_step_atr * atr_pct_now * price)

        if not in_trade and signal_today != 0 and i + 1 < len(opens):
            direction = int(signal_today)
            entry_price = float(opens[i + 1])
            entry_atr = max(atr_pct_now, 1e-6)
            stop_dist = (
                entry_price * stop_atr_mult * entry_atr
                if stop_atr_mult > 0
                else entry_price * stop_pct
            )
            stop_price = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
            entry_idx = i + 1
            hold_count = 0
            hold_limit = max(hold_today, 1)
            units = 1
            avg_entry = entry_price
            best_price = entry_price
            worst_price = entry_price
            next_pyramid_price = entry_price + (direction * pyramid_step_atr * entry_price * entry_atr)
            in_trade = True

    return (
        np.asarray(entries, dtype=np.int32),
        np.asarray(exits, dtype=np.int32),
        np.asarray(dirs, dtype=np.int8),
        np.asarray(pnls, dtype=np.float64),
        np.asarray(holds, dtype=np.int32),
        np.asarray(reasons, dtype=np.int8),
        np.asarray(units_arr, dtype=np.int16),
    )


def simulate_forex_trades_arrays(
    opens: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    hold_days: np.ndarray,
    stop_pct: float,
    index: Optional[Sequence] = None,
    *,
    atr_pcts: Optional[np.ndarray] = None,
    stop_atr_mult: float = 2.0,
    trailing_atr_mult: float = 1.0,
    strict_mode: bool = False,
    donchian_exit_days: int = 10,
    allow_pyramiding: bool = False,
    pyramid_step_atr: float = 0.5,
    max_pyramid_units: int = 4,
    risk_pct: float = 0.01,
    max_risk_pct: float = 0.01,
    enable_cb_refresh: bool = True,
) -> List[dict]:
    opens = np.asarray(opens, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.int8)
    hold_days = np.asarray(hold_days, dtype=np.int32)
    atr_pcts = np.asarray(atr_pcts, dtype=np.float64) if atr_pcts is not None else _default_atr_pct(
        closes, stop_pct, stop_atr_mult
    )
    donchian_exit_lows = (
        _rolling_min_prev(closes, donchian_exit_days)
        if strict_mode and donchian_exit_days > 0
        else np.full(len(closes), np.nan, dtype=np.float64)
    )

    entry_idx, exit_idx, directions, pnl_pcts, hold_counts, exit_reasons, unit_counts = _simulate_forex_core(
        opens=opens,
        closes=closes,
        signals=signals,
        hold_days=hold_days,
        stop_pct=float(stop_pct),
        atr_pcts=atr_pcts,
        stop_atr_mult=float(stop_atr_mult),
        trailing_atr_mult=float(trailing_atr_mult),
        donchian_exit_lows=donchian_exit_lows,
        strict_mode=bool(strict_mode),
        allow_pyramiding=bool(allow_pyramiding),
        pyramid_step_atr=float(pyramid_step_atr),
        max_units=int(max(1, max_pyramid_units)),
        enable_cb_refresh=bool(enable_cb_refresh),
    )

    reason_map = {
        EXIT_REASON_STOP: 'stop',
        EXIT_REASON_REVERSAL: 'reversal',
        EXIT_REASON_CB_REFRESH: 'cb_refresh',
        EXIT_REASON_TIME: 'time',
        EXIT_REASON_TRAILING: 'trailing_stop',
        EXIT_REASON_DONCHIAN: 'donchian_exit',
    }
    effective_risk_pct = min(float(risk_pct), float(max_risk_pct))
    trades = []
    for j in range(len(entry_idx)):
        e_idx = int(entry_idx[j])
        x_idx = int(exit_idx[j])
        if e_idx < 0 or x_idx < 0:
            continue
        entry_date = index[e_idx] if index is not None else e_idx
        exit_date = index[x_idx] if index is not None else x_idx
        pnl = float(pnl_pcts[j])
        trades.append(
            {
                'entry_date': entry_date,
                'exit_date': exit_date,
                'direction': int(directions[j]),
                'entry': float(opens[e_idx]),
                'exit': float(closes[x_idx]),
                'pnl_pct': pnl,
                'hold_days': int(hold_counts[j]),
                'exit_reason': reason_map.get(int(exit_reasons[j]), 'time'),
                'units': int(unit_counts[j]),
                'risk_pct': effective_risk_pct,
                'risk_adjusted_pnl_pct': pnl * effective_risk_pct,
            }
        )
    return trades


def simulate_forex_trades(
    df,
    signal_frame,
    stop_pct: float,
    *,
    atr_series=None,
    stop_atr_mult: float = 2.0,
    trailing_atr_mult: float = 1.0,
    strict_mode: bool = False,
    donchian_exit_days: int = 10,
    allow_pyramiding: bool = False,
    pyramid_step_atr: float = 0.5,
    max_pyramid_units: int = 4,
    risk_pct: float = 0.01,
    max_risk_pct: float = 0.01,
    enable_cb_refresh: bool = True,
) -> List[dict]:
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    opens = df['Open'] if 'Open' in df.columns else close
    hold_col = 'hold_days' if 'hold_days' in signal_frame.columns else 'hold'

    # Apply size_mult from signal frame if present (Session 3 latent feature sizing)
    effective_risk = risk_pct
    if 'size_mult' in signal_frame.columns:
        # Scale risk_pct by size_mult for each bar; clamp to [0.5×, 1.5×] × risk_pct
        size_mults = signal_frame['size_mult'].fillna(1.0).clip(0.5, 1.5)
        signals_arr = signal_frame['signal'].to_numpy(dtype=np.int8)
        # Build per-bar effective risk: only varies on signal bars, 1.0× on flat bars
        per_bar_risk = np.where(signals_arr != 0,
                                np.clip(size_mults.to_numpy() * risk_pct, 0.0, max_risk_pct),
                                0.0)
        # simulate_forex_trades_arrays uses a fixed risk_pct; run with the median
        # adjusted risk as the scalar, and annotate trades post-hoc below
        effective_risk = float(np.median(per_bar_risk[per_bar_risk > 0])) if np.any(per_bar_risk > 0) else risk_pct

    return simulate_forex_trades_arrays(
        opens=opens.to_numpy(dtype=np.float64),
        closes=close.to_numpy(dtype=np.float64),
        signals=signal_frame['signal'].to_numpy(dtype=np.int8),
        hold_days=signal_frame[hold_col].to_numpy(dtype=np.int32),
        stop_pct=stop_pct,
        index=close.index,
        atr_pcts=None if atr_series is None else np.asarray(atr_series, dtype=np.float64),
        stop_atr_mult=stop_atr_mult,
        trailing_atr_mult=trailing_atr_mult,
        strict_mode=strict_mode,
        donchian_exit_days=donchian_exit_days,
        allow_pyramiding=allow_pyramiding,
        pyramid_step_atr=pyramid_step_atr,
        max_pyramid_units=max_pyramid_units,
        risk_pct=effective_risk,
        max_risk_pct=max_risk_pct,
        enable_cb_refresh=enable_cb_refresh,
    )


def warmup_forex_kernel() -> None:
    opens = np.array([100.0, 100.0, 99.0, 98.0], dtype=np.float64)
    closes = np.array([100.0, 99.0, 98.0, 97.0], dtype=np.float64)
    signals = np.array([1, 0, 0, 0], dtype=np.int8)
    hold_days = np.array([2, 60, 60, 60], dtype=np.int32)
    _simulate_forex_core(
        opens=opens,
        closes=closes,
        signals=signals,
        hold_days=hold_days,
        stop_pct=0.04,
        atr_pcts=_default_atr_pct(closes, 0.04, 2.0),
        stop_atr_mult=2.0,
        trailing_atr_mult=1.0,
        donchian_exit_lows=np.full(len(closes), np.nan),
        strict_mode=False,
        allow_pyramiding=False,
        pyramid_step_atr=0.5,
        max_units=4,
        enable_cb_refresh=True,
    )


def simulate_kernel(
    close: np.ndarray,
    opens: np.ndarray,
    signal: np.ndarray,
    hold_days: int = HOLD_DAYS,
    stop_pct: float = STOP_PCT,
):
    hold_arr = np.full(len(close), int(hold_days), dtype=np.int32)
    trades = simulate_forex_trades_arrays(
        opens=np.asarray(opens, dtype=np.float64),
        closes=np.asarray(close, dtype=np.float64),
        signals=np.asarray(signal, dtype=np.int8),
        hold_days=hold_arr,
        stop_pct=float(stop_pct),
    )
    pnl = np.asarray([t['pnl_pct'] for t in trades], dtype=np.float64)
    holds = np.asarray([t['hold_days'] for t in trades], dtype=np.int32)
    return pnl, holds


@dataclass
class ForexArrayDataset:
    pair: str
    close: np.ndarray
    opens: np.ndarray
    signal: np.ndarray
    n_bars: int


@dataclass
class FastBacktestResult:
    pair: str
    win_rate: float
    total_trades: int
    avg_hold_days: float
    sharpe: float


class ForexFastBacktester:
    def __init__(self, hold_days: int = HOLD_DAYS, stop_pct: float = STOP_PCT):
        self.hold_days = hold_days
        self.stop_pct = stop_pct

    def run(self, dataset: ForexArrayDataset) -> Optional[FastBacktestResult]:
        pnl, hold = simulate_kernel(
            close=dataset.close,
            opens=dataset.opens,
            signal=dataset.signal,
            hold_days=self.hold_days,
            stop_pct=self.stop_pct,
        )
        if len(pnl) == 0:
            return None
        wins = float(np.sum(pnl > 0))
        win_rate = wins / len(pnl)
        sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-9)) * np.sqrt(252 / max(float(np.mean(hold)), 1.0))
        return FastBacktestResult(
            pair=dataset.pair,
            win_rate=float(win_rate),
            total_trades=int(len(pnl)),
            avg_hold_days=float(np.mean(hold)),
            sharpe=sharpe,
        )
