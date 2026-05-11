"""Tests for DXYEngine — mocks yfinance, tests trend detection and smile modifier."""
from __future__ import annotations
import pandas as pd
import pytest
from unittest.mock import patch
from sovereign.forex.dxy_engine import DXYEngine


def _dxy(last: float, baseline: float, n: int = 300) -> pd.Series:
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    values = [baseline] * (n - 20) + [last] * 20
    return pd.Series(values, index=dates, name='DXY')


class TestDXYEngineGetTrend:
    def test_returns_required_keys(self):
        engine = DXYEngine()
        with patch.object(engine, '_load_dxy', return_value=_dxy(105.0, 100.0)), \
             patch.object(engine, '_latest_vix', return_value=18.0):
            result = engine.get_trend()
        for key in ('trend', 'z_score', 'smile_regime', 'vix'):
            assert key in result

    def test_strong_bull_high_z(self):
        engine = DXYEngine()
        # last=115 vs baseline=100, std~0 → large z
        series = _dxy(115.0, 100.0)
        with patch.object(engine, '_load_dxy', return_value=series), \
             patch.object(engine, '_latest_vix', return_value=18.0):
            result = engine.get_trend()
        assert result['trend'] in ('STRONG_BULL', 'BULL')
        assert result['z_score'] > 0

    def test_bear_low_z(self):
        engine = DXYEngine()
        series = _dxy(85.0, 100.0)
        with patch.object(engine, '_load_dxy', return_value=series), \
             patch.object(engine, '_latest_vix', return_value=18.0):
            result = engine.get_trend()
        assert result['trend'] in ('STRONG_BEAR', 'BEAR')

    def test_neutral_when_flat(self):
        engine = DXYEngine()
        series = _dxy(100.0, 100.0)
        with patch.object(engine, '_load_dxy', return_value=series), \
             patch.object(engine, '_latest_vix', return_value=18.0):
            result = engine.get_trend()
        assert result['trend'] == 'NEUTRAL'
        assert result['smile_regime'] == 'WEAK'

    def test_growth_driven_when_bull_and_low_vix(self):
        engine = DXYEngine()
        series = _dxy(115.0, 100.0)
        with patch.object(engine, '_load_dxy', return_value=series), \
             patch.object(engine, '_latest_vix', return_value=14.0):
            result = engine.get_trend()
        if result['trend'] in ('STRONG_BULL', 'BULL'):
            assert result['smile_regime'] == 'GROWTH_DRIVEN'

    def test_safety_driven_when_bull_and_high_vix(self):
        engine = DXYEngine()
        series = _dxy(115.0, 100.0)
        with patch.object(engine, '_load_dxy', return_value=series), \
             patch.object(engine, '_latest_vix', return_value=35.0):
            result = engine.get_trend()
        if result['trend'] in ('STRONG_BULL', 'BULL'):
            assert result['smile_regime'] == 'SAFETY_DRIVEN'

    def test_returns_neutral_with_no_data(self):
        engine = DXYEngine()
        with patch.object(engine, '_load_dxy', return_value=None), \
             patch.object(engine, '_latest_vix', return_value=18.0):
            result = engine.get_trend()
        assert result['trend'] == 'NEUTRAL'


class TestDXYEngineGetModifier:
    def test_usd_base_long_boosted_in_strong_bull_growth(self):
        engine = DXYEngine()
        with patch.object(engine, 'get_trend',
                          return_value={'trend': 'STRONG_BULL', 'smile_regime': 'GROWTH_DRIVEN',
                                        'z_score': 2.5, 'vix': 15.0}):
            mult = engine.get_modifier('USDJPY=X', 'LONG')
        assert mult > 1.0

    def test_usd_base_long_reduced_in_bear(self):
        engine = DXYEngine()
        with patch.object(engine, 'get_trend',
                          return_value={'trend': 'BEAR', 'smile_regime': 'WEAK',
                                        'z_score': -1.5, 'vix': 18.0}):
            mult = engine.get_modifier('USDJPY=X', 'LONG')
        assert mult < 1.0

    def test_neutral_trend_returns_one(self):
        engine = DXYEngine()
        with patch.object(engine, 'get_trend',
                          return_value={'trend': 'NEUTRAL', 'smile_regime': 'WEAK',
                                        'z_score': 0.1, 'vix': 18.0}):
            mult = engine.get_modifier('GBPUSD=X', 'LONG')
        assert mult == 1.0

    def test_unknown_pair_returns_one(self):
        engine = DXYEngine()
        with patch.object(engine, 'get_trend',
                          return_value={'trend': 'STRONG_BULL', 'smile_regime': 'GROWTH_DRIVEN',
                                        'z_score': 2.5, 'vix': 15.0}):
            mult = engine.get_modifier('XAUUSD=X', 'LONG')
        assert mult == 1.0
