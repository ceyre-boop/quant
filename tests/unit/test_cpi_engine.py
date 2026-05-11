"""Tests for CPISurpriseEngine — mocks FRED data, tests surprise detection and fade logic."""
from __future__ import annotations
import pandas as pd
import pytest
from unittest.mock import patch
from sovereign.forex.cpi_engine import CPISurpriseEngine, MIN_SURPRISE_PCT, FADE_HOLD_DAYS


def _make_cpi_series(actual: float, consensus_avg: float, n: int = 24) -> pd.Series:
    """Build a monthly CPI series where the last value is `actual` and prior mean is `consensus_avg`."""
    dates = pd.date_range('2022-01-01', periods=n, freq='MS')
    values = [consensus_avg] * (n - 1) + [actual]
    return pd.Series(values, index=dates, name='cpi')


class TestCPISurpriseEngineGetLatestSurprise:
    def test_returns_none_below_threshold(self):
        engine = CPISurpriseEngine()
        series = _make_cpi_series(actual=2.50, consensus_avg=2.49)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            assert engine.get_latest_surprise('US') is None

    def test_returns_short_on_beat(self):
        # CPI beat → initial FX move up → FADE = SHORT
        engine = CPISurpriseEngine()
        series = _make_cpi_series(actual=3.50, consensus_avg=2.80)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            result = engine.get_latest_surprise('US')
        assert result is not None
        assert result['direction'] == 'SHORT'
        assert result['surprise_pct'] > MIN_SURPRISE_PCT

    def test_returns_long_on_miss(self):
        # CPI miss → initial FX move down → FADE = LONG
        engine = CPISurpriseEngine()
        series = _make_cpi_series(actual=2.00, consensus_avg=3.20)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            result = engine.get_latest_surprise('US')
        assert result is not None
        assert result['direction'] == 'LONG'

    def test_conviction_capped_at_0_65(self):
        engine = CPISurpriseEngine()
        series = _make_cpi_series(actual=10.0, consensus_avg=2.0)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            result = engine.get_latest_surprise('US')
        assert result['conviction'] <= 0.65

    def test_hold_days_is_correct(self):
        engine = CPISurpriseEngine()
        series = _make_cpi_series(actual=3.5, consensus_avg=2.8)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            result = engine.get_latest_surprise('US')
        assert result['hold_days'] == FADE_HOLD_DAYS

    def test_returns_none_with_no_data(self):
        engine = CPISurpriseEngine()
        with patch.object(engine, '_get_cpi_series', return_value=None):
            assert engine.get_latest_surprise('US') is None


class TestCPISurpriseEngineHistorical:
    def test_returns_list(self):
        engine = CPISurpriseEngine()
        dates = pd.date_range('2015-01-01', periods=60, freq='MS')
        # Introduce obvious surprises every 12 months
        values = [2.0] * 60
        for i in range(12, 60):
            if i % 12 == 0:
                values[i] = 4.0
        series = pd.Series(values, index=dates)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            results = engine.get_historical_surprises(
                'US', pd.Timestamp('2016-01-01'), pd.Timestamp('2020-01-01')
            )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_results_have_required_keys(self):
        engine = CPISurpriseEngine()
        dates = pd.date_range('2015-01-01', periods=60, freq='MS')
        values = [2.0] * 59 + [4.0]
        series = pd.Series(values, index=dates)
        with patch.object(engine, '_get_cpi_series', return_value=series):
            results = engine.get_historical_surprises(
                'US', pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-01')
            )
        for r in results:
            for key in ('signal_date', 'release_date', 'surprise_pct', 'direction', 'conviction'):
                assert key in r
