"""
tests/unit/test_funderpro_executor.py
======================================
Unit tests for the FunderPro executor and pipeline GO guard.

All cTrader network calls are mocked — no real connection required.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from execution.funderpro_executor import (
    FunderProExecutor,
    OrderResult,
    record_pipeline_go,
    DAILY_LOSS_LIMIT,
)


# ── Helpers ─────────────────────────────────────────────────────────────────── #

def _mock_scan(
    pair: str = 'GBPUSD',
    signal: str = 'LONG',
    entry: float = 1.27000,
    stop: float = 1.26000,
) -> SimpleNamespace:
    return SimpleNamespace(pair=pair, signal=signal, entry_level=entry, stop=stop)


def _off_executor(account_size: float = 10_000.0) -> FunderProExecutor:
    """Return an executor in OFF routing mode (no env vars, no network)."""
    with patch.dict(os.environ, {'FUNDERPRO_LIVE': 'off'}):
        return FunderProExecutor(account_size=account_size)


# ── OFF routing ─────────────────────────────────────────────────────────────── #

class TestOffRouting:

    def test_routing_is_off_by_default(self):
        ex = _off_executor()
        assert ex._routing == 'OFF'

    def test_submit_logs_and_returns_simulated_id(self):
        ex = _off_executor()
        result = ex.submit(_mock_scan())
        assert isinstance(result, OrderResult)
        assert result.submitted is False
        assert result.order_id.startswith('SIMULATED_')
        assert result.error == ''

    def test_submit_long_computes_tp1_correctly(self):
        ex = _off_executor()
        result = ex.submit(_mock_scan(entry=1.27, stop=1.26), tp1_r=2.0)
        # entry + 2 * stop_dist = 1.27 + 2 * 0.01 = 1.29
        assert abs(result.tp1 - 1.29) < 1e-4

    def test_submit_short_computes_tp1_correctly(self):
        ex = _off_executor()
        result = ex.submit(
            _mock_scan(signal='SHORT', entry=1.27, stop=1.28),
            tp1_r=2.0,
        )
        # entry - 2 * stop_dist = 1.27 - 2 * 0.01 = 1.25
        assert abs(result.tp1 - 1.25) < 1e-4

    def test_blocked_by_daily_loss_limit(self):
        ex = _off_executor()
        ex._daily_pnl = -(DAILY_LOSS_LIMIT + 0.001)
        result = ex.submit(_mock_scan())
        assert result.submitted is False
        assert 'Daily loss limit' in result.error

    def test_blocked_when_pair_already_open(self):
        ex = _off_executor()
        ex._open['GBPUSD'] = 'SIMULATED_...'
        result = ex.submit(_mock_scan())
        assert result.submitted is False
        assert 'open position' in result.error

    def test_rejected_when_no_entry_level(self):
        ex = _off_executor()
        result = ex.submit(_mock_scan(entry=0.0))
        assert result.submitted is False
        assert 'Missing' in result.error

    def test_close_position_off_routing(self):
        ex = _off_executor()
        ex._open['GBPUSD'] = 'SIMULATED_POS'
        assert ex.close_position('GBPUSD') is True
        assert 'GBPUSD' not in ex._open

    def test_close_position_missing_pair(self):
        ex = _off_executor()
        assert ex.close_position('EURUSD') is False

    def test_get_status_reflects_state(self):
        ex = _off_executor()
        s = ex.get_status()
        assert s.routing == 'OFF'
        assert s.connected is False
        assert s.blocked is False

    def test_lot_sizing_floor(self):
        ex = _off_executor()
        # Tiny account, big stop → lots clamped to 0.01
        lots = ex._size_lots('GBPUSD', risk_dollars=1.0, stop_dist=0.5)
        assert lots == 0.01

    def test_lot_sizing_gbpusd(self):
        ex = _off_executor()
        # 1% of $10k = $100 risk; 10-pip stop (0.001) = $100/$10 per pip/lot => 10/10 = 1 lot
        lots = ex._size_lots('GBPUSD', risk_dollars=100.0, stop_dist=0.001)
        # pips = 0.001 / 0.0001 = 10; lots = 100 / (10 * 10) = 1.0
        assert abs(lots - 1.0) < 0.01


# ── Pipeline GO guard ────────────────────────────────────────────────────────── #

class TestPipelineGoGuard:

    def test_no_verdict_file_returns_false(self, tmp_path):
        verdict_path = tmp_path / 'pipeline_verdict.json'
        config_path  = tmp_path / 'parameters.yml'
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is False
        assert 'No pipeline verdict' in reason

    def test_no_go_verdict_returns_false(self, tmp_path):
        verdict_path = tmp_path / 'pipeline_verdict.json'
        config_path  = tmp_path / 'parameters.yml'
        verdict_path.write_text(json.dumps({'verdict': 'NO-GO', 'timestamp': '2026-05-01', 'config_hash': ''}))
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is False
        assert 'NO-GO' in reason

    def test_go_verdict_without_config_returns_true(self, tmp_path):
        verdict_path = tmp_path / 'pipeline_verdict.json'
        config_path  = tmp_path / 'does_not_exist.yml'
        verdict_path.write_text(json.dumps({
            'verdict': 'GO', 'timestamp': '2026-05-01T00:00:00+00:00', 'config_hash': '',
        }))
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is True
        assert 'GO' in reason

    def test_go_verdict_config_hash_matches(self, tmp_path):
        import hashlib
        config_path  = tmp_path / 'parameters.yml'
        config_path.write_text('risk:\n  base_risk_pct: 0.005\n')
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
        verdict_path = tmp_path / 'pipeline_verdict.json'
        verdict_path.write_text(json.dumps({
            'verdict': 'GO', 'timestamp': '2026-05-01', 'config_hash': config_hash,
        }))
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is True

    def test_go_verdict_config_hash_mismatch_returns_false(self, tmp_path):
        config_path  = tmp_path / 'parameters.yml'
        config_path.write_text('risk:\n  base_risk_pct: 0.005\n')
        verdict_path = tmp_path / 'pipeline_verdict.json'
        verdict_path.write_text(json.dumps({
            'verdict': 'GO', 'timestamp': '2026-05-01', 'config_hash': 'stalehashabcd1234',
        }))
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is False
        assert 'changed' in reason

    def test_live_routing_refused_without_go(self, tmp_path):
        with patch.dict(os.environ, {'FUNDERPRO_LIVE': 'live'}):
            with patch(
                'execution.funderpro_executor.PIPELINE_VERDICT_FILE',
                tmp_path / 'missing.json',
            ):
                with pytest.raises(RuntimeError, match='LIVE mode refused'):
                    FunderProExecutor._detect_routing()

    def test_live_routing_allowed_with_go(self, tmp_path):
        verdict_path = tmp_path / 'verdict.json'
        verdict_path.write_text(json.dumps({
            'verdict': 'GO', 'timestamp': '2026-05-01', 'config_hash': '',
        }))
        with patch.dict(os.environ, {'FUNDERPRO_LIVE': 'live'}):
            with patch(
                'execution.funderpro_executor.FunderProExecutor._check_pipeline_go',
                staticmethod(lambda *a, **kw: (True, 'Pipeline GO confirmed')),
            ):
                routing = FunderProExecutor._detect_routing()
        assert routing == 'LIVE'

    def test_demo_routing_does_not_check_go(self, tmp_path):
        with patch.dict(os.environ, {'FUNDERPRO_LIVE': 'demo'}):
            # No verdict file — demo should still succeed
            routing = FunderProExecutor._detect_routing()
        assert routing == 'DEMO'


# ── record_pipeline_go utility ────────────────────────────────────────────────── #

class TestRecordPipelineGo:

    def test_writes_go_verdict(self, tmp_path):
        verdict_path = tmp_path / 'verdict.json'
        config_path  = tmp_path / 'parameters.yml'
        config_path.write_text('risk:\n  base_risk_pct: 0.005\n')
        record_pipeline_go(config_path=config_path, verdict_path=verdict_path)
        payload = json.loads(verdict_path.read_text())
        assert payload['verdict'] == 'GO'
        assert len(payload['config_hash']) == 16

    def test_creates_parent_dirs(self, tmp_path):
        verdict_path = tmp_path / 'deep' / 'nested' / 'verdict.json'
        record_pipeline_go(
            config_path=tmp_path / 'does_not_exist.yml',
            verdict_path=verdict_path,
        )
        assert verdict_path.exists()

    def test_round_trip_check_pipeline_go(self, tmp_path):
        config_path  = tmp_path / 'parameters.yml'
        config_path.write_text('risk:\n  base_risk_pct: 0.005\n')
        verdict_path = tmp_path / 'verdict.json'
        record_pipeline_go(config_path=config_path, verdict_path=verdict_path)
        ok, reason = FunderProExecutor._check_pipeline_go(verdict_path, config_path)
        assert ok is True


# ── cTrader delegation (bridge mocked) ──────────────────────────────────────── #

class TestCTraderDelegation:
    """Verify _send_ctrader_order and _close_ctrader_order delegate correctly."""

    def _demo_executor_with_mock_bridge(self):
        """Return DEMO executor with a mock CTraderBridge injected."""
        bridge_mock = MagicMock()
        bridge_mock.wait_for_ready.return_value = True
        bridge_mock.send_bracket_order.return_value = '99001'
        bridge_mock.close_position.return_value = True

        with patch.dict(os.environ, {
            'FUNDERPRO_LIVE': 'demo',
            'CTRADER_CLIENT_ID':     'test_id',
            'CTRADER_CLIENT_SECRET': 'test_secret',
            'CTRADER_ACCOUNT_ID':    '12345',
            'CTRADER_ACCESS_TOKEN':  'test_token',
        }):
            with patch('execution.funderpro_executor.FunderProExecutor._init_ctrader'):
                ex = FunderProExecutor(account_size=10_000.0)

        ex._routing  = 'DEMO'
        ex._ctrader  = bridge_mock
        return ex, bridge_mock

    def test_send_delegates_to_bridge(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        order = OrderResult(
            submitted=False, order_id='', pair='GBPUSD', direction='LONG',
            entry=1.27, stop=1.26, tp1=1.29, tp2=1.31, lots=0.10,
            routing='DEMO', timestamp='2026-05-13T00:00:00+00:00',
        )
        position_id = ex._send_ctrader_order(order)
        bridge.send_bracket_order.assert_called_once_with(
            direction='LONG',
            size_lots=0.10,
            entry=1.27,
            stop=1.26,
            target=1.29,
            conviction=0.0,
            pred_p50=0.0,
            pred_p90=0.0,
            timeout=15.0,
        )
        assert position_id == '99001'

    def test_send_raises_when_bridge_times_out(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        bridge.send_bracket_order.return_value = None  # timeout
        order = OrderResult(
            submitted=False, order_id='', pair='GBPUSD', direction='LONG',
            entry=1.27, stop=1.26, tp1=1.29, tp2=1.31, lots=0.10,
            routing='DEMO', timestamp='2026-05-13T00:00:00+00:00',
        )
        with pytest.raises(RuntimeError, match='timed out'):
            ex._send_ctrader_order(order)

    def test_send_raises_when_bridge_not_connected(self):
        ex = _off_executor()
        ex._routing  = 'DEMO'
        ex._ctrader  = None
        order = OrderResult(
            submitted=False, order_id='', pair='GBPUSD', direction='LONG',
            entry=1.27, stop=1.26, tp1=1.29, tp2=1.31, lots=0.10,
            routing='DEMO', timestamp='2026-05-13T00:00:00+00:00',
        )
        with pytest.raises(RuntimeError, match='not connected'):
            ex._send_ctrader_order(order)

    def test_close_delegates_to_bridge(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        ex._close_ctrader_order('99001')
        bridge.close_position.assert_called_once_with(position_id='99001', timeout=10.0)

    def test_close_raises_when_bridge_times_out(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        bridge.close_position.return_value = False
        with pytest.raises(RuntimeError, match='timed out'):
            ex._close_ctrader_order('99001')

    def test_close_raises_when_bridge_not_connected(self):
        ex = _off_executor()
        ex._routing  = 'DEMO'
        ex._ctrader  = None
        with pytest.raises(RuntimeError, match='not connected'):
            ex._close_ctrader_order('99001')

    def test_submit_end_to_end_demo(self):
        """submit() → _send_ctrader_order() → bridge.send_bracket_order() → position_id stored."""
        ex, bridge = self._demo_executor_with_mock_bridge()
        result = ex.submit(_mock_scan())
        assert result.submitted is True
        assert result.order_id == '99001'
        assert ex._open['GBPUSD'] == '99001'

    def test_close_position_end_to_end_demo(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        ex._open['GBPUSD'] = '99001'
        ok = ex.close_position('GBPUSD', reason='TP1_HIT')
        assert ok is True
        assert 'GBPUSD' not in ex._open
        bridge.close_position.assert_called_once()

    def test_submit_logs_error_when_bridge_fails(self):
        ex, bridge = self._demo_executor_with_mock_bridge()
        bridge.send_bracket_order.return_value = None  # causes _send to raise
        result = ex.submit(_mock_scan())
        assert result.submitted is False
        assert result.error != ''
