"""Unit tests for the market data adapter + parquet cache (TICK-043).

No network. Backends are stubbed; the contract under test is the seam itself:
column normalisation, fallback-on-primary-failure, and cache invalidation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sovereign.data import adapter as adapter_mod
from sovereign.data.adapter import (
    BAR_COLUMNS,
    DataUnavailable,
    MarketDataAdapter,
    VendorNotSupported,
    _Backend,
    _normalise,
)
from sovereign.data.cache import DataCache, date_range


def _bars(n=3, start="2026-01-05T14:30:00Z"):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": 1.0, "high": 2.0,
        "low": 0.5, "close": 1.5, "volume": 100.0,
    })


class StubBackend(_Backend):
    name = "stub"

    def __init__(self, frame=None, fail=False):
        self.frame = frame if frame is not None else _bars()
        self.fail = fail
        self.calls = 0

    def get_bars(self, symbol, start, end, timeframe):
        self.calls += 1
        if self.fail:
            raise RuntimeError("primary vendor down")
        return self.frame

    def get_snapshot(self, symbols):
        if self.fail:
            raise RuntimeError("primary vendor down")
        return {s: {"symbol": s, "price": 10.0, "source": self.name} for s in symbols}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Adapter with two stub vendors registered as primary/fallback."""
    def _make(primary: StubBackend, fallback: StubBackend):
        monkeypatch.setitem(adapter_mod._BACKENDS, "p", lambda: primary)
        monkeypatch.setitem(adapter_mod._BACKENDS, "f", lambda: fallback)
        return MarketDataAdapter(primary="p", fallback="f",
                                 cache=DataCache(tmp_path), use_cache=False)
    return _make


# ── normalisation ────────────────────────────────────────────────────────────

def test_normalise_polygon_short_keys():
    raw = pd.DataFrame({"t": pd.to_datetime([1704470400000], unit="ms", utc=True),
                        "o": [1.0], "h": [2.0], "l": [0.5], "c": [1.5], "v": [10]})
    out = _normalise(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert len(out) == 1


def test_normalise_yfinance_multiindex():
    ts = pd.date_range("2026-01-05", periods=2, freq="1D", tz="UTC")
    raw = pd.DataFrame(
        [[1.0, 2.0, 0.5, 1.5, 10], [1.1, 2.1, 0.6, 1.6, 11]],
        index=ts,
        columns=pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]),
    )
    out = _normalise(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert len(out) == 2


def test_normalise_empty_returns_contract_columns():
    assert list(_normalise(pd.DataFrame()).columns) == BAR_COLUMNS


def test_normalise_rejects_incomplete_frame():
    with pytest.raises(ValueError, match="missing required columns"):
        _normalise(pd.DataFrame({"open": [1.0], "close": [1.5]}))


def test_normalise_sorts_by_timestamp():
    df = _bars(3).iloc[::-1]
    out = _normalise(df)
    assert out["timestamp"].is_monotonic_increasing


# ── fallback ─────────────────────────────────────────────────────────────────

def test_primary_used_when_healthy(wired):
    primary, fallback = StubBackend(), StubBackend()
    out = wired(primary, fallback).get_bars("SPY", "2026-01-05", "2026-01-06")
    assert len(out) == 3
    assert primary.calls == 1 and fallback.calls == 0


def test_fallback_fires_when_primary_raises(wired):
    primary, fallback = StubBackend(fail=True), StubBackend()
    out = wired(primary, fallback).get_bars("SPY", "2026-01-05", "2026-01-06")
    assert len(out) == 3
    assert primary.calls == 1 and fallback.calls == 1


def test_both_failing_raises_data_unavailable(wired):
    a = wired(StubBackend(fail=True), StubBackend(fail=True))
    with pytest.raises(DataUnavailable):
        a.get_bars("SPY", "2026-01-05", "2026-01-06")


def test_unsupported_call_falls_through_to_vendor_that_supports_it(wired):
    """A VendorNotSupported from the primary is a fallback trigger, not a crash."""
    class NoOptions(StubBackend):
        def get_options_chain(self, symbol, expiry):
            raise VendorNotSupported("stub: get_options_chain")

    class HasOptions(StubBackend):
        def get_options_chain(self, symbol, expiry):
            return pd.DataFrame({"strike": [100.0], "option_type": ["call"]})

    out = wired(NoOptions(), HasOptions()).get_options_chain("SPY")
    assert len(out) == 1


def test_unknown_vendor_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown vendor"):
        MarketDataAdapter(primary="nasdaq_direct", fallback="yfinance")


def test_env_configures_vendors(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PRIMARY", "polygon")
    monkeypatch.setenv("DATA_FALLBACK", "yfinance")
    a = MarketDataAdapter(cache=DataCache(tmp_path))
    assert (a.primary_name, a.fallback_name) == ("polygon", "yfinance")


# ── cache ────────────────────────────────────────────────────────────────────

def test_cache_hit_avoids_second_fetch(tmp_path):
    cache, calls = DataCache(tmp_path), []

    def fetcher():
        calls.append(1)
        return _bars()

    a = cache.get_or_fetch("SPY", "2026-01-05", fetcher)
    b = cache.get_or_fetch("SPY", "2026-01-05", fetcher)
    assert len(calls) == 1
    assert len(a) == len(b) == 3
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_empty_result_is_not_cached(tmp_path):
    """An empty frame is usually a transient failure — caching it makes it permanent."""
    cache, calls = DataCache(tmp_path), []

    def fetcher():
        calls.append(1)
        return pd.DataFrame(columns=BAR_COLUMNS)

    cache.get_or_fetch("SPY", "2026-01-05", fetcher)
    cache.get_or_fetch("SPY", "2026-01-05", fetcher)
    assert len(calls) == 2


def test_historical_day_never_stale(tmp_path):
    cache = DataCache(tmp_path)
    cache.put("SPY", "2020-01-02", _bars())
    assert not cache._is_stale(cache.path_for("SPY", "2020-01-02"), "2020-01-02")


def test_today_written_intraday_is_stale(tmp_path):
    cache = DataCache(tmp_path)
    today = datetime.now(timezone.utc).date().isoformat()
    cache.put("SPY", today, _bars())
    path = cache.path_for("SPY", today)
    # Stamp the write to 14:00 UTC — mid-session, before the 21:00 close boundary.
    intraday = datetime.now(timezone.utc).replace(hour=14, minute=0).timestamp()
    import os
    os.utime(path, (intraday, intraday))
    assert cache._is_stale(path, today)


def test_corrupt_parquet_is_refetched(tmp_path):
    cache = DataCache(tmp_path)
    path = cache.path_for("SPY", "2026-01-05")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not parquet")
    out = cache.get_or_fetch("SPY", "2026-01-05", _bars)
    assert len(out) == 3
    assert cache.stats.corrupt == 1


def test_invalidate_symbol_drops_all_days(tmp_path):
    cache = DataCache(tmp_path)
    for d in ("2026-01-05", "2026-01-06"):
        cache.put("SPY", d, _bars())
    assert cache.invalidate("SPY") == 2
    assert not cache.path_for("SPY", "2026-01-05").exists()


def test_date_range_inclusive():
    assert date_range("2026-01-05", "2026-01-07") == [
        "2026-01-05", "2026-01-06", "2026-01-07"]


# ── isolation (NN#1) ─────────────────────────────────────────────────────────

def test_adapter_does_not_import_ict():
    import sovereign.data.adapter as m
    import sovereign.data.cache as c
    for mod in (m, c):
        src = open(mod.__file__).read()
        assert "import ict" not in src and "from ict" not in src
