from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
from sovereign.forex.compliance import (
    ForexComplianceConfig,
    score_compliance,
    block_live_mode_if_needed,
)
from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
from sovereign.forex.forex_backtester import ForexBacktester


class _DummyFetcher:
    def get_rate_history(self, country, start='2014-01-01'):
        idx = pd.date_range(start, periods=800, freq='B')
        return pd.Series(2.0, index=idx)

    def get_cpi_history(self, country, start='2014-01-01'):
        idx = pd.date_range(start, periods=800, freq='B')
        return pd.Series(2.0, index=idx)


class _DummyCB:
    def check_historical(self, **kwargs):
        return []


def test_strict_signal_engine_produces_donchian_breakouts():
    idx = pd.date_range('2020-01-01', periods=120, freq='B')
    close = pd.Series(np.linspace(1.0, 1.5, len(idx)), index=idx)
    prices = pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close}, index=idx)
    eng = ForexSignalEngine(
        fetcher=_DummyFetcher(),
        cb_trigger=_DummyCB(),
        config=SignalConfig(strict_mode=True, use_macro_overlay=False),
    )
    sig, hold = eng.build_signal_arrays(
        close=close,
        base_country='EU',
        quote_country='US',
        start='2020-01-01',
        end='2020-12-31',
        pair='EURUSD=X',
        prices_df=prices,
    )
    assert np.any(sig == 1)
    assert np.all(hold == eng.config.hold_days)


def test_trade_risk_pct_is_capped_to_one_percent():
    n = 80
    closes = np.linspace(1.0, 1.2, n)
    opens = closes.copy()
    signals = np.zeros(n, dtype=np.int8)
    signals[0] = 1
    hold = np.full(n, 20, dtype=np.int32)
    trades = simulate_forex_trades_arrays(
        opens=opens,
        closes=closes,
        signals=signals,
        hold_days=hold,
        stop_pct=0.04,
        risk_pct=0.03,
        max_risk_pct=0.01,
    )
    assert trades
    assert all(t['risk_pct'] <= 0.01 for t in trades)


def test_compliance_validation_rejects_risk_above_one_percent():
    cfg = ForexComplianceConfig(strict_mode=True, max_risk_per_trade_pct=0.02)
    with pytest.raises(ValueError):
        cfg.validate_startup()


def test_compliance_scoring_can_reach_100():
    report = score_compliance(ForexComplianceConfig(strict_mode=True))
    assert report['score'] == 100
    assert report['status'] == 'pass'


def test_live_mode_blocked_when_compliance_fails():
    report = {'status': 'fail', 'score': 70}
    with pytest.raises(RuntimeError):
        block_live_mode_if_needed('live', report)


def test_jpy_correlation_cap_limits_concurrent_exposure():
    bt = ForexBacktester(strict_mode=True)
    sample = {
        'USDJPY=X': [{'entry_date': pd.Timestamp('2024-01-01'), 'exit_date': pd.Timestamp('2024-02-01')}],
        'EURJPY=X': [{'entry_date': pd.Timestamp('2024-01-01'), 'exit_date': pd.Timestamp('2024-02-01')}],
        'GBPJPY=X': [{'entry_date': pd.Timestamp('2024-01-01'), 'exit_date': pd.Timestamp('2024-02-01')}],
    }
    capped = bt._apply_correlation_caps(sample)
    total_kept = sum(len(v) for v in capped.values())
    assert total_kept == bt.MAX_SHARED_JPY_POSITIONS
