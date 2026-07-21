"""ATR must be measured or absent — never substituted.

Until 2026-07-20 `execute_daily.py` sized every trade off
`atr = latest['close'] * 0.02`, a constant regardless of realised volatility, on a
script that runs live daily via com.sovereign.papertrading. Measured against real
bars the placeholder was wrong by 2.34x on META.

The contract these tests defend: `_compute_atr` returns a real number or None, and
None means skip the symbol. It must never return a guess, because a fabricated ATR
reaching sizing is the FAKE_DATA class that forex_data_health.py exists to catch.
"""
from __future__ import annotations

import pandas as pd
import pytest

import execute_daily as ed


class _DeadFeed:
    """Broker outage."""
    def get_bars(self, *a, **k):
        raise RuntimeError("simulated Alpaca outage")


class _ShortFeed:
    """Fewer bars than the ATR window needs."""
    def get_bars(self, *a, **k):
        return pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2],
                             "close": [1, 2], "volume": [1, 1]})


class _EmptyFeed:
    def get_bars(self, *a, **k):
        return None


class _FlatFeed:
    """Real bar count, zero range — ATR of 0 is not a usable stop distance."""
    def get_bars(self, *a, **k):
        n = ed._ATR_LOOKBACK_BARS
        return pd.DataFrame({"open": [100.0] * n, "high": [100.0] * n,
                             "low": [100.0] * n, "close": [100.0] * n,
                             "volume": [1] * n})


class _GoodFeed:
    """Bars with a known, constant true range of 2.0."""
    def get_bars(self, *a, **k):
        n = ed._ATR_LOOKBACK_BARS
        return pd.DataFrame({"open": [100.0] * n, "high": [101.0] * n,
                             "low": [99.0] * n, "close": [100.0] * n,
                             "volume": [1] * n})


@pytest.mark.parametrize("feed,label", [
    (_DeadFeed(), "broker outage"),
    (_ShortFeed(), "insufficient history"),
    (_EmptyFeed(), "no data returned"),
    (_FlatFeed(), "zero-range bars"),
])
def test_failure_returns_none_never_a_placeholder(feed, label):
    """Every failure mode must yield None so the caller skips the symbol."""
    atr = ed._compute_atr(feed, "TEST", 100.0)
    assert atr is None, f"{label} produced {atr} instead of None"


def test_never_returns_the_old_two_percent_constant():
    """The specific regression: 2% of price must not reappear as a fallback."""
    for feed in (_DeadFeed(), _ShortFeed(), _EmptyFeed()):
        assert ed._compute_atr(feed, "TEST", 500.0) != 500.0 * 0.02


def test_real_bars_produce_a_real_atr():
    """True range is high-low = 2.0 on every bar, so ATR ~ 2.0 absolute."""
    atr = ed._compute_atr(_GoodFeed(), "TEST", 100.0)
    assert atr is not None
    assert atr == pytest.approx(2.0, abs=0.01)


def test_result_is_absolute_not_a_fraction():
    """_compute_atr_pct returns a FRACTION of price. Forgetting the multiply
    would under-size every position by two orders of magnitude."""
    atr = ed._compute_atr(_GoodFeed(), "TEST", 100.0)
    assert atr > 1.0, "looks like a fraction (0.02) rather than absolute (2.0)"


def test_atr_window_is_long_enough_for_the_period():
    assert ed._ATR_LOOKBACK_BARS > ed._ATR_PERIOD + 1
