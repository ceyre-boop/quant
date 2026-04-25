"""
Forex fast backtester — numba-backed array simulation kernel.

The simulation kernel is compiled with numba when available; otherwise it
falls back to an equivalent pure-numpy/Python loop so the module is always
importable even without numba installed.

Typical usage:
    from sovereign.forex.fast_backtester import ForexArrayDataset, ForexFastBacktester

    dataset = ForexArrayDataset(
        pair="EURUSD=X",
        close=close_arr,
        opens=opens_arr,
        signal=signal_arr,
        n_bars=len(close_arr),
    )
    bt = ForexFastBacktester()
    result = bt.run(dataset)  # ForexBacktestResult or None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """Identity decorator used when numba is unavailable."""
        if args and callable(args[0]):
            return args[0]

        def _wrap(f):
            return f

        return _wrap


# Shared strategy constants — kept in sync with ForexBacktester
HOLD_DAYS: int = 60
STOP_PCT: float = 0.04


@dataclass
class ForexArrayDataset:
    """Preloaded price/signal arrays for one pair."""

    pair: str
    close: np.ndarray   # shape (n_bars,), dtype float64
    opens: np.ndarray   # shape (n_bars,), dtype float64
    signal: np.ndarray  # shape (n_bars,), dtype float64: +1 / 0 / -1 per bar
    n_bars: int


@_njit(cache=True)
def simulate_kernel(
    close: np.ndarray,
    opens: np.ndarray,
    signal: np.ndarray,
    hold_days: int,
    stop_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core trade simulation loop operating on bare arrays.

    Rules (matches ForexBacktester._simulate_trades exactly):
      - Enter at next-bar open when a non-zero signal appears.
      - Exit after hold_days bars, on stop_pct loss, or on reversal.
      - One position at a time; reversal immediately opens the opposing trade.

    Returns:
        pnl_arr : float64 array, P&L pct for each completed trade.
        hold_arr: int32 array, bars held for each completed trade.
    """
    n = len(close)
    # Upper bound on trades: one entry per bar
    pnl_out = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    n_trades = 0

    in_trade = False
    direction = 0
    entry_price = 0.0
    hold_count = 0

    i = 0
    while i < n:
        if in_trade:
            hold_count += 1
            price = close[i]
            sig_today = signal[i]
            ret = direction * (price / entry_price - 1.0)

            stop_hit = ret <= -stop_pct
            time_exit = hold_count >= hold_days
            reversal = (sig_today != 0.0) and (int(sig_today) != direction)

            if stop_hit or time_exit or reversal:
                pnl_out[n_trades] = direction * (price / entry_price - 1.0)
                hold_out[n_trades] = hold_count
                n_trades += 1
                in_trade = False
                hold_count = 0

                # Reversal: immediately open opposing trade at next-bar open
                if reversal and i + 1 < n:
                    in_trade = True
                    direction = int(sig_today)
                    entry_price = opens[min(i + 1, n - 1)]

        # New trade from signal (also runs after non-reversal exit on same bar)
        if not in_trade:
            sig_today = signal[i]
            if sig_today != 0.0 and i + 1 < n:
                in_trade = True
                direction = int(sig_today)
                entry_price = opens[i + 1]
                hold_count = 0

        i += 1

    return pnl_out[:n_trades], hold_out[:n_trades]


def _compute_stats(
    pair: str,
    pnl_arr: np.ndarray,
    hold_arr: np.ndarray,
    n_bars: int,
):
    """Build a ForexBacktestResult from compact trade arrays."""
    from sovereign.forex.forex_backtester import ForexBacktestResult

    n = len(pnl_arr)
    if n == 0:
        return None

    years = n_bars / 252.0
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]

    win_rate = len(wins) / n
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (20.0 if gross_win > 0 else 1.0)
    avg_hold = float(hold_arr.mean()) if n else 0.0

    equity = np.cumprod(1.0 + pnl_arr)
    log_eq = np.log(np.maximum(equity, 1e-12))
    returns = np.diff(log_eq, prepend=0.0)
    ann_factor = np.sqrt(252 / max(avg_hold, 1))
    sharpe = (
        (np.mean(returns) / (np.std(returns) + 1e-9)) * ann_factor
        if n > 1
        else 0.0
    )

    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0

    return ForexBacktestResult(
        pair=pair,
        win_rate=round(win_rate, 3),
        profit_factor=round(min(profit_factor, 20.0), 3),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd, 3),
        avg_hold_days=round(avg_hold, 1),
        trades_per_year=round(n / max(years, 1), 1),
        total_trades=n,
        years=round(years, 1),
    )


class ForexFastBacktester:
    """Run the array kernel on a preloaded ForexArrayDataset."""

    HOLD_DAYS: int = HOLD_DAYS
    STOP_PCT: float = STOP_PCT

    def run(self, dataset: ForexArrayDataset):
        """Simulate and return a ForexBacktestResult, or None if no trades."""
        pnl_arr, hold_arr = simulate_kernel(
            dataset.close,
            dataset.opens,
            dataset.signal,
            self.HOLD_DAYS,
            self.STOP_PCT,
        )
        if len(pnl_arr) == 0:
            return None
        return _compute_stats(dataset.pair, pnl_arr, hold_arr, dataset.n_bars)
