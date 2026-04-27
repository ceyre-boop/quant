"""
Fast simulation kernel for forex backtests.

This does not make the whole forex stack as fast as the equity engine yet,
but it removes the Python bar loop from the hot path and gives the strategy
one compiled simulation core to build on.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    def _njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap if args and callable(args[0]) else _wrap


@_njit(cache=True)
def _simulate_forex_core(
    opens: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    hold_days: np.ndarray,
    stop_pct: float,
):
    n = len(closes)
    max_trades = n

    entry_idx = np.full(max_trades, -1, dtype=np.int32)
    exit_idx = np.full(max_trades, -1, dtype=np.int32)
    directions = np.zeros(max_trades, dtype=np.int8)
    pnl_pcts = np.zeros(max_trades, dtype=np.float64)
    hold_counts = np.zeros(max_trades, dtype=np.int32)
    exit_reasons = np.zeros(max_trades, dtype=np.int8)  # 1 stop, 2 reversal, 3 cb_refresh, 4 time

    in_trade = False
    direction = 0
    entry_price = 0.0
    current_entry_idx = 0
    hold_count = 0
    current_hold_days = 0
    trade_count = 0

    for i in range(n):
        signal_today = int(signals[i])
        sig_hold_today = int(hold_days[i])

        if in_trade:
            hold_count += 1
            price = closes[i]
            ret = direction * (price / entry_price - 1.0)
            stop_hit = ret <= -stop_pct
            time_exit = hold_count >= current_hold_days
            reversal = signal_today != 0 and signal_today != direction
            cb_refresh = signal_today == direction and sig_hold_today < 30 and hold_count >= 20

            if stop_hit or time_exit or reversal or cb_refresh:
                entry_idx[trade_count] = current_entry_idx
                exit_idx[trade_count] = i
                directions[trade_count] = np.int8(direction)
                pnl_pcts[trade_count] = direction * (price / entry_price - 1.0)
                hold_counts[trade_count] = hold_count
                if stop_hit:
                    exit_reasons[trade_count] = 1
                elif reversal:
                    exit_reasons[trade_count] = 2
                elif cb_refresh:
                    exit_reasons[trade_count] = 3
                else:
                    exit_reasons[trade_count] = 4
                trade_count += 1
                in_trade = False

                reenter_dir = signal_today if (reversal or cb_refresh) and signal_today != 0 else 0
                if reenter_dir != 0 and i + 1 < n:
                    direction = int(reenter_dir)
                    current_hold_days = int(sig_hold_today)
                    entry_price = opens[i + 1]
                    current_entry_idx = i + 1
                    hold_count = 0
                    in_trade = True

        if not in_trade and signal_today != 0 and i + 1 < n:
            direction = signal_today
            current_hold_days = sig_hold_today
            entry_price = opens[i + 1]
            current_entry_idx = i + 1
            hold_count = 0
            in_trade = True

    return (
        entry_idx[:trade_count],
        exit_idx[:trade_count],
        directions[:trade_count],
        pnl_pcts[:trade_count],
        hold_counts[:trade_count],
        exit_reasons[:trade_count],
    )


def simulate_forex_trades_arrays(
    opens: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    hold_days: np.ndarray,
    stop_pct: float,
    index: Optional[Sequence] = None,
) -> List[dict]:
    entry_idx, exit_idx, directions, pnl_pcts, hold_counts, exit_reasons = _simulate_forex_core(
        np.asarray(opens, dtype=np.float64),
        np.asarray(closes, dtype=np.float64),
        np.asarray(signals, dtype=np.int8),
        np.asarray(hold_days, dtype=np.int32),
        float(stop_pct),
    )

    reason_map = {1: 'stop', 2: 'reversal', 3: 'cb_refresh', 4: 'time'}
    trades = []
    for j in range(len(entry_idx)):
        e_idx = int(entry_idx[j])
        x_idx = int(exit_idx[j])
        if e_idx < 0 or x_idx < 0:
            continue
        entry_date = index[e_idx] if index is not None else e_idx
        exit_date = index[x_idx] if index is not None else x_idx
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': int(directions[j]),
            'entry': float(opens[e_idx]),
            'exit': float(closes[x_idx]),
            'pnl_pct': float(pnl_pcts[j]),
            'hold_days': int(hold_counts[j]),
            'exit_reason': reason_map.get(int(exit_reasons[j]), 'time'),
        })
    return trades


def simulate_forex_trades(df, signal_frame, stop_pct: float) -> List[dict]:
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    opens = df['Open'] if 'Open' in df.columns else close
    return simulate_forex_trades_arrays(
        opens=opens.to_numpy(dtype=np.float64),
        closes=close.to_numpy(dtype=np.float64),
        signals=signal_frame['signal'].to_numpy(dtype=np.int8),
        hold_days=signal_frame['hold_days'].to_numpy(dtype=np.int32),
        stop_pct=stop_pct,
        index=close.index,
    )


def warmup_forex_kernel() -> None:
    opens = np.array([100.0, 100.0, 99.0, 98.0], dtype=np.float64)
    closes = np.array([100.0, 99.0, 98.0, 97.0], dtype=np.float64)
    signals = np.array([1, 0, 0, 0], dtype=np.int8)
    hold_days = np.array([2, 60, 60, 60], dtype=np.int32)
    _simulate_forex_core(opens, closes, signals, hold_days, 0.04)
