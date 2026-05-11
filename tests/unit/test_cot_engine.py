"""Tests for COTEngine — mocks CFTC download, tests z-score and gate logic."""
from __future__ import annotations
import io, zipfile
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from sovereign.forex.cot_engine import COTEngine, FUTURES_CODES, CROWDED_Z


def _make_fake_zip(currency: str = 'EUR', net: float = 50000.0) -> bytes:
    """Build a minimal fake CFTC disaggregated CSV inside a zip."""
    code = FUTURES_CODES[currency]
    rows = []
    for i in range(160):
        date = pd.Timestamp('2021-01-05') + pd.Timedelta(weeks=i)
        # net varies so z-score calculation is non-trivial
        row_net = net + (i - 80) * 500
        rows.append(f'{code},{date.strftime("%Y-%m-%d")},{row_net + 10000},{10000},0,0,0')

    header = 'CFTC_Contract_Market_Code,Report_Date_as_YYYY-MM-DD,NonComm_Positions_Long_All,NonComm_Positions_Short_All,x,y,z'
    content = (header + '\n' + '\n'.join(rows)).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('c_disagg.txt', content)
    return buf.getvalue()


class TestCOTEngineGetPositioning:
    def test_unknown_currency_returns_none(self):
        engine = COTEngine()
        assert engine.get_positioning('XYZ') is None

    def test_returns_dict_with_required_keys(self):
        engine = COTEngine()
        dates = pd.date_range('2021-01-01', periods=160, freq='W')
        series = pd.Series([float(i * 100) for i in range(160)], index=dates, name='EUR_cot_net')
        with patch.object(engine, '_load_or_fetch', return_value=series):
            result = engine.get_positioning('EUR')
        assert result is not None
        for key in ('currency', 'net_position', 'z_score', 'signal', 'as_of'):
            assert key in result

    def test_crowded_long_when_z_above_threshold(self):
        engine = COTEngine()
        # Build series where last value is far above mean
        dates = pd.date_range('2021-01-01', periods=160, freq='W')
        values = [0.0] * 156 + [100_000.0, 100_000.0, 100_000.0, 100_000.0]
        series = pd.Series(values, index=dates, name='EUR_cot_net')
        with patch.object(engine, '_load_or_fetch', return_value=series):
            result = engine.get_positioning('EUR')
        assert result['signal'] == 'CROWDED_LONG'
        assert result['z_score'] > CROWDED_Z

    def test_crowded_short_when_z_below_threshold(self):
        engine = COTEngine()
        dates = pd.date_range('2021-01-01', periods=160, freq='W')
        values = [0.0] * 156 + [-100_000.0, -100_000.0, -100_000.0, -100_000.0]
        series = pd.Series(values, index=dates, name='EUR_cot_net')
        with patch.object(engine, '_load_or_fetch', return_value=series):
            result = engine.get_positioning('EUR')
        assert result['signal'] == 'CROWDED_SHORT'

    def test_neutral_when_z_within_band(self):
        engine = COTEngine()
        dates = pd.date_range('2021-01-01', periods=160, freq='W')
        # Flat series → std near 0 treated as 0 → z=0 → NEUTRAL
        values = [10000.0] * 160
        series = pd.Series(values, index=dates, name='EUR_cot_net')
        with patch.object(engine, '_load_or_fetch', return_value=series):
            result = engine.get_positioning('EUR')
        assert result['signal'] == 'NEUTRAL'


class TestCOTEngineGateSignal:
    def test_returns_one_when_no_data(self):
        engine = COTEngine()
        with patch.object(engine, 'get_positioning', return_value=None):
            assert engine.gate_signal('LONG', 'EUR') == 1.0

    def test_returns_half_when_crowded_in_direction(self):
        engine = COTEngine()
        with patch.object(engine, 'get_positioning',
                          return_value={'signal': 'CROWDED_LONG', 'z_score': 2.0,
                                        'net_position': 80000, 'currency': 'EUR', 'as_of': '2024-01-01'}):
            assert engine.gate_signal('LONG', 'EUR') == 0.5

    def test_returns_one_when_crowded_opposite(self):
        engine = COTEngine()
        with patch.object(engine, 'get_positioning',
                          return_value={'signal': 'CROWDED_SHORT', 'z_score': -2.0,
                                        'net_position': -80000, 'currency': 'EUR', 'as_of': '2024-01-01'}):
            assert engine.gate_signal('LONG', 'EUR') == 1.0

    def test_returns_one_when_neutral(self):
        engine = COTEngine()
        with patch.object(engine, 'get_positioning',
                          return_value={'signal': 'NEUTRAL', 'z_score': 0.5,
                                        'net_position': 10000, 'currency': 'EUR', 'as_of': '2024-01-01'}):
            assert engine.gate_signal('LONG', 'EUR') == 1.0
