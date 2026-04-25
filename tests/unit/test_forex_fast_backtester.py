"""Unit tests for sovereign/forex/fast_backtester.py"""
from __future__ import annotations

import numpy as np
import pytest

from sovereign.forex.fast_backtester import (
    ForexArrayDataset,
    ForexFastBacktester,
    simulate_kernel,
    HOLD_DAYS,
    STOP_PCT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_dataset(n_bars: int = 100, price: float = 1.2, pair: str = "EURUSD=X"):
    """Dataset where prices never move — no stop, no time exit until hold expires."""
    close = np.full(n_bars, price, dtype=np.float64)
    opens = np.full(n_bars, price, dtype=np.float64)
    signal = np.zeros(n_bars, dtype=np.float64)
    return ForexArrayDataset(pair=pair, close=close, opens=opens, signal=signal, n_bars=n_bars)


# ---------------------------------------------------------------------------
# simulate_kernel tests
# ---------------------------------------------------------------------------

def test_simulate_kernel_no_signals_no_trades():
    """Zero signals → no trades."""
    ds = _make_flat_dataset(200)
    pnl, hold = simulate_kernel(ds.close, ds.opens, ds.signal, HOLD_DAYS, STOP_PCT)
    assert len(pnl) == 0
    assert len(hold) == 0


def test_simulate_kernel_single_long_time_exit():
    """Single +1 signal held to time exit produces one trade."""
    n = 200
    close = np.ones(n, dtype=np.float64) * 1.0
    opens = close.copy()
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0  # fire long at bar 0

    pnl, hold = simulate_kernel(close, opens, signal, HOLD_DAYS, STOP_PCT)

    assert len(pnl) == 1
    assert hold[0] == HOLD_DAYS
    # Flat price → 0 P&L
    assert pnl[0] == pytest.approx(0.0)


def test_simulate_kernel_stop_hit():
    """Price drops below stop → trade exits early with negative P&L."""
    n = 100
    close = np.ones(n, dtype=np.float64) * 1.0
    opens = close.copy()
    # Drop price to trigger stop on bar 5
    close[5:] = 1.0 * (1.0 - STOP_PCT - 0.01)
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0  # long

    pnl, hold = simulate_kernel(close, opens, signal, HOLD_DAYS, STOP_PCT)

    assert len(pnl) == 1
    assert pnl[0] < 0.0
    assert hold[0] < HOLD_DAYS


def test_simulate_kernel_reversal():
    """A reversal signal exits current trade and opens the opposite immediately."""
    n = 200
    close = np.ones(n, dtype=np.float64) * 1.0
    opens = close.copy()
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0    # initial long
    signal[10] = -1.0  # reversal to short

    pnl, hold = simulate_kernel(close, opens, signal, HOLD_DAYS, STOP_PCT)

    # First trade exits on reversal at bar 10; second trade exits at time exit
    assert len(pnl) >= 2
    assert hold[0] == 10  # held exactly 10 bars before reversal


def test_simulate_kernel_returns_numpy_arrays():
    """Output dtypes must be float64 and int32."""
    n = 150
    close = np.ones(n, dtype=np.float64)
    opens = close.copy()
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0

    pnl, hold = simulate_kernel(close, opens, signal, HOLD_DAYS, STOP_PCT)
    assert pnl.dtype == np.float64
    assert hold.dtype == np.int32


# ---------------------------------------------------------------------------
# ForexFastBacktester tests
# ---------------------------------------------------------------------------

def test_fast_backtester_no_trades_returns_none():
    ds = _make_flat_dataset(200)
    bt = ForexFastBacktester()
    result = bt.run(ds)
    assert result is None


def test_fast_backtester_returns_result_with_trades():
    n = 300
    close = np.ones(n, dtype=np.float64)
    opens = close.copy()
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0

    ds = ForexArrayDataset(pair="TEST=X", close=close, opens=opens, signal=signal, n_bars=n)
    bt = ForexFastBacktester()
    result = bt.run(ds)

    assert result is not None
    assert result.pair == "TEST=X"
    assert result.total_trades >= 1
    assert 0.0 <= result.win_rate <= 1.0
