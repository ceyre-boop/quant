"""Unit tests for sovereign/forex/batch_backtester.py"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.fast_backtester import ForexArrayDataset, HOLD_DAYS, STOP_PCT
from sovereign.forex.batch_backtester import (
    ForexBatchBacktester,
    _make_synthetic_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_bars: int = 200, pair: str = "EURUSD=X") -> ForexArrayDataset:
    close = np.ones(n_bars, dtype=np.float64)
    opens = close.copy()
    signal = np.zeros(n_bars, dtype=np.float64)
    signal[0] = 1.0
    return ForexArrayDataset(pair=pair, close=close, opens=opens, signal=signal, n_bars=n_bars)


# ---------------------------------------------------------------------------
# _make_synthetic_dataset tests
# ---------------------------------------------------------------------------

def test_make_synthetic_dataset_shapes():
    rng = np.random.default_rng(0)
    ds = _make_synthetic_dataset(rng, n_bars=500)

    assert ds.close.shape == (500,)
    assert ds.opens.shape == (500,)
    assert ds.signal.shape == (500,)
    assert ds.n_bars == 500


def test_make_synthetic_dataset_signal_values():
    rng = np.random.default_rng(1)
    ds = _make_synthetic_dataset(rng, n_bars=500)
    unique = set(np.unique(ds.signal))
    assert unique.issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# run_synthetic_benchmark tests
# ---------------------------------------------------------------------------

def test_run_synthetic_benchmark_returns_dict():
    bt = ForexBatchBacktester()
    stats = bt.run_synthetic_benchmark(n_pairs=2, n_bars=100, n_iterations=5)

    assert isinstance(stats, dict)
    for key in ("n_pairs", "n_bars", "n_iterations", "total_runs", "elapsed_s", "runs_per_sec"):
        assert key in stats


def test_run_synthetic_benchmark_correct_total_runs():
    bt = ForexBatchBacktester()
    stats = bt.run_synthetic_benchmark(n_pairs=3, n_bars=50, n_iterations=10)
    assert stats["total_runs"] == 30


def test_run_synthetic_benchmark_positive_throughput():
    bt = ForexBatchBacktester()
    stats = bt.run_synthetic_benchmark(n_pairs=2, n_bars=100, n_iterations=5)
    assert stats["runs_per_sec"] > 0
    assert stats["elapsed_s"] > 0


# ---------------------------------------------------------------------------
# run_serial tests
# ---------------------------------------------------------------------------

def test_run_serial_with_trades():
    bt = ForexBatchBacktester()
    datasets = [_make_dataset(300, pair="EURUSD=X")]
    results = bt.run_serial(datasets)
    assert len(results) == 1
    assert results[0].total_trades >= 1


def test_run_serial_no_trades_dataset_excluded():
    bt = ForexBatchBacktester()
    # No signals → no trades → result is None → should be excluded
    n = 200
    ds = ForexArrayDataset(
        pair="FLAT=X",
        close=np.ones(n),
        opens=np.ones(n),
        signal=np.zeros(n),
        n_bars=n,
    )
    results = bt.run_serial([ds])
    assert results == []


# ---------------------------------------------------------------------------
# Preload does NOT call build_signal_frame
# ---------------------------------------------------------------------------

def test_preload_uses_array_builder_not_frame_builder():
    """
    Verify preload calls build_signal_arrays and never build_signal_frame.
    This ensures the batch path stays off the pandas frame builder.
    """
    bt = ForexBatchBacktester()

    # Build a minimal price DataFrame (enough bars to pass the 252-bar guard)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame({
        "Close": np.ones(n) * 1.2,
        "Open": np.ones(n) * 1.2,
        "High": np.ones(n) * 1.21,
        "Low": np.ones(n) * 1.19,
        "Volume": np.zeros(n),
    }, index=idx)

    with patch(
        "sovereign.forex.batch_backtester.yf.download",
        return_value=prices,
    ), patch(
        "sovereign.forex.batch_backtester.build_signal_arrays",
        return_value=(np.zeros(n), np.full(n, HOLD_DAYS, dtype=np.int32)),
    ) as mock_arrays, patch(
        "sovereign.forex.signal_engine.build_signal_frame",
    ) as mock_frame:
        bt.preload(pairs=["EURUSD=X"])

    assert mock_arrays.called
    mock_frame.assert_not_called()
