"""Unit tests for sovereign/discovery/block_length.py — the locked HYP-071 block length L (v2).

Network-free: the pure numerics (_crossing_lag, _innovations, _aggregate_clamp) are exercised directly,
and compute_atr_block_length is driven with an INJECTED synthetic loader (no yfinance, no network).

v2 correction: L is derived from the ATR% INNOVATION (first difference), which strips the level's
mechanical 14-day-SMA autocorrelation. The clamp is [5, 60] (wide, degenerate-catcher only); if the
innovation crossing still overshoots on most pairs the helper RAISES BlockLengthDegenerate.

Synthetic vol kinds:
  - "rw":  daily range is a slow random walk  → ATR% level ~persistent; its innovation is MA-shaped by
           the SMA window (triangular acf out to ~lag 14) → crossing lands mid-range.
  - "iid": daily range is iid                 → innovation telescopes to disjoint terms → crossing ~1 → floor.
  - "i2":  daily range is a DOUBLE random walk → innovation stays a random walk → crossing overshoots → degenerate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign.discovery.block_length import (
    BLOCK_MAX,
    BLOCK_MIN,
    DEFAULT_MAX_LAG,
    SERIES_LOCKED,
    BlockLengthDegenerate,
    _aggregate_clamp,
    _crossing_lag,
    _innovations,
    compute_atr_block_length,
)


def _synthetic_loader(seed: int, kind: str):
    """Return a deterministic (pair, start, end) -> OHLC DataFrame loader built from seeded vol."""

    def loader(pair, start, end):
        rng = np.random.default_rng(seed + sum(ord(c) for c in pair))
        n = 700
        if kind == "rw":           # slow random walk → persistent level
            walk = rng.normal(0.0, 0.02, n).cumsum()
            band = 1.0 + (walk - walk.min()) + 0.5
        elif kind == "iid":        # iid daily range
            band = np.abs(rng.normal(1.0, 0.3, n)) + 0.1
        elif kind == "i2":         # double random walk → persistent even after one difference
            walk2 = rng.normal(0.0, 0.02, n).cumsum().cumsum()
            band = 1.0 + (walk2 - walk2.min()) + 0.5
        else:
            raise ValueError(kind)
        close = np.full(n, 100.0)
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {"Open": close, "High": close + band / 2, "Low": close - band / 2, "Close": close},
            index=idx,
        )

    return loader


class TestAggregateClamp:
    def test_floor_engages(self):
        assert _aggregate_clamp([1, 1, 2, 1]) == BLOCK_MIN

    def test_ceiling_engages(self):
        assert _aggregate_clamp([99, 250, 120, 200]) == BLOCK_MAX

    def test_in_range_passthrough(self):
        assert _aggregate_clamp([8, 9, 10, 9]) == 9

    def test_returns_int(self):
        assert isinstance(_aggregate_clamp([7, 8, 9, 8]), int)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _aggregate_clamp([])


class TestCrossingLag:
    def test_white_noise_crosses_at_lag_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 4000)
        assert _crossing_lag(x) == 1

    def test_persistent_series_never_crosses_within_window(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, 4000).cumsum()
        assert _crossing_lag(x, max_lag=DEFAULT_MAX_LAG) == DEFAULT_MAX_LAG

    def test_deterministic(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, 2000)
        assert _crossing_lag(x) == _crossing_lag(x)

    def test_constant_series_safe(self):
        assert _crossing_lag(np.full(100, 3.0)) >= 1


class TestInnovations:
    def test_first_difference(self):
        np.testing.assert_array_equal(_innovations(np.array([1.0, 2.0, 4.0, 7.0])), [1.0, 2.0, 3.0])

    def test_length_is_n_minus_one(self):
        assert _innovations(np.arange(50.0)).size == 49

    def test_nan_safe(self):
        # non-finite values are dropped before differencing
        out = _innovations(np.array([1.0, np.nan, 3.0, 6.0]))
        np.testing.assert_array_equal(out, [2.0, 3.0])


class TestComputeBlockLength:
    def test_default_series_is_innovations(self):
        loader = _synthetic_loader(3, "rw")
        assert compute_atr_block_length(["EURUSD=X", "GBPUSD=X"], loader=loader) == \
            compute_atr_block_length(["EURUSD=X", "GBPUSD=X"], loader=loader, series="innovations")
        assert SERIES_LOCKED == "innovations"

    def test_returns_int_in_range(self):
        L = compute_atr_block_length(["EURUSD=X", "GBPUSD=X"], loader=_synthetic_loader(3, "rw"))
        assert isinstance(L, int)
        assert BLOCK_MIN <= L <= BLOCK_MAX

    def test_deterministic_on_fixed_input(self):
        loader = _synthetic_loader(11, "rw")
        a = compute_atr_block_length(["EURUSD=X", "USDJPY=X"], loader=loader)
        b = compute_atr_block_length(["EURUSD=X", "USDJPY=X"], loader=loader)
        assert a == b

    def test_innovation_floors_on_iid(self):
        # iid range → innovation telescopes to disjoint terms → crossing ~1 → clamp floor
        L = compute_atr_block_length(
            ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"], loader=_synthetic_loader(8, "iid")
        )
        assert L == BLOCK_MIN

    def test_level_crossings_exceed_innovation_crossings(self):
        # the whole point of v2: differencing strips the SMA-mechanical persistence, so the level
        # crossing is much larger than the innovation crossing for every pair.
        _, detail = compute_atr_block_length(
            ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
            loader=_synthetic_loader(5, "rw"),
            return_detail=True,
        )
        for pair in detail["level_crossings"]:
            assert detail["level_crossings"][pair] > detail["innovation_crossings"][pair]

    def test_level_path_raises_when_degenerate(self):
        # explicitly inspecting the audit-only level path on persistent data overshoots [5,60] → raises
        with pytest.raises(BlockLengthDegenerate):
            compute_atr_block_length(
                ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
                loader=_synthetic_loader(5, "rw"),
                series="level",
            )

    def test_stop_and_report_on_degenerate(self):
        # double random walk → innovation stays persistent → overshoots BLOCK_MAX on all pairs → raise
        with pytest.raises(BlockLengthDegenerate):
            compute_atr_block_length(
                ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"], loader=_synthetic_loader(4, "i2")
            )

    def test_detail_reports_both_crossings(self):
        L, detail = compute_atr_block_length(
            ["EURUSD=X"], loader=_synthetic_loader(2, "rw"), return_detail=True
        )
        assert detail["series_locked"] == "innovations"
        assert set(detail["innovation_crossings"]) == {"EURUSD=X"}
        assert set(detail["level_crossings"]) == {"EURUSD=X"}
        assert detail["clamp"] == [BLOCK_MIN, BLOCK_MAX]
        assert BLOCK_MIN <= L <= BLOCK_MAX
