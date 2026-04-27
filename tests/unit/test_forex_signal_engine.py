"""Unit tests for sovereign/forex/signal_engine.py"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.signal_engine import (
    build_signal_arrays,
    build_signal_frame,
    HOLD_DAYS,
    SIGNAL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(n: int = 500, start: str = "2018-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    close = 1.05 + np.linspace(0.0, 0.15, n)
    return pd.DataFrame({"Close": close, "Open": close * 0.9995}, index=idx)


def _flat_macro_series(start: str = "2014-01-01", value: float = 2.0) -> pd.Series:
    idx = pd.date_range(start, periods=3000, freq="B")
    return pd.Series(value, index=idx)


def _patched_fetcher():
    """Return a mock ForexDataFetcher with flat fallback rate/CPI histories."""
    fetcher = MagicMock()
    fetcher.get_rate_history.side_effect = lambda country, start="2014-01-01": _flat_macro_series(start, 2.0)
    fetcher.get_cpi_history.side_effect = lambda country, start="2014-01-01": _flat_macro_series(start, 2.0)
    return fetcher


# ---------------------------------------------------------------------------
# build_signal_arrays tests
# ---------------------------------------------------------------------------

def test_build_signal_arrays_returns_numpy_arrays():
    prices = _make_prices()
    fetcher = _patched_fetcher()
    signal_arr, hold_arr = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    assert isinstance(signal_arr, np.ndarray)
    assert isinstance(hold_arr, np.ndarray)


def test_build_signal_arrays_correct_shape():
    prices = _make_prices(400)
    fetcher = _patched_fetcher()
    signal_arr, hold_arr = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    assert signal_arr.shape == (len(prices),)
    assert hold_arr.shape == (len(prices),)


def test_build_signal_arrays_valid_signal_values():
    """Signal values must only be -1, 0, or +1."""
    prices = _make_prices()
    fetcher = _patched_fetcher()
    signal_arr, _ = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    unique = set(np.unique(signal_arr))
    assert unique.issubset({-1.0, 0.0, 1.0})


def test_build_signal_arrays_hold_arr_constant():
    """hold_arr should equal HOLD_DAYS at every position."""
    prices = _make_prices()
    fetcher = _patched_fetcher()
    _, hold_arr = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    assert np.all(hold_arr == HOLD_DAYS)


def test_build_signal_arrays_empty_prices():
    """Empty price input should return zero-length arrays without error."""
    idx = pd.date_range("2020-01-01", periods=0, freq="B")
    prices = pd.DataFrame({"Close": []}, index=idx)
    fetcher = _patched_fetcher()
    signal_arr, hold_arr = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)
    assert len(signal_arr) == 0
    assert len(hold_arr) == 0


# ---------------------------------------------------------------------------
# build_signal_frame tests
# ---------------------------------------------------------------------------

def test_build_signal_frame_returns_dataframe():
    prices = _make_prices()
    fetcher = _patched_fetcher()
    frame = build_signal_frame("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    assert isinstance(frame, pd.DataFrame)
    assert "signal" in frame.columns
    assert "hold" in frame.columns


def test_build_signal_frame_index_matches_prices():
    prices = _make_prices(300)
    fetcher = _patched_fetcher()
    frame = build_signal_frame("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    pd.testing.assert_index_equal(frame.index, prices.index)


# ---------------------------------------------------------------------------
# Consistency: array builder must match frame builder
# ---------------------------------------------------------------------------

def test_array_builder_matches_frame_output():
    """
    build_signal_arrays and build_signal_frame must produce identical values
    (frame is just a thin wrapper over the arrays).
    """
    prices = _make_prices(600)
    fetcher = _patched_fetcher()

    signal_arr, hold_arr = build_signal_arrays("EURUSD=X", prices, "EU", "US", fetcher=fetcher)
    frame = build_signal_frame("EURUSD=X", prices, "EU", "US", fetcher=fetcher)

    np.testing.assert_array_equal(signal_arr, frame["signal"].values)
    np.testing.assert_array_equal(hold_arr, frame["hold"].values)
