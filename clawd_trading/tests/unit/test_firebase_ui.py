"""Unit tests for Firebase UI writer and broadcaster."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from integration.firebase_ui_writer import (
    format_direction_for_ui,
    format_position_status_for_ui,
    format_signal_for_ui,
    format_position_for_ui,
    format_account_state_for_ui,
    format_bias_for_ui,
    format_risk_for_ui,
    format_game_output_for_ui,
    format_connection_status,
    format_regime_for_ui,
    create_live_state_update,
    create_session_control_update,
    ConnectionStatus,
    SignalStatus
)
from integration.firebase_broadcaster import FirebaseBroadcaster
from contracts.types import (
    EntrySignal, PositionState, AccountState,
    BiasOutput, RiskOutput, GameOutput,
    Direction, AdversarialRisk,
    FeatureSnapshot, ThreeLayerContext, RegimeState,
    VolRegime, TrendRegime, RiskAppetite, MomentumRegime, EventRisk
)


class TestFormatDirection:
    """Test direction formatting for UI."""
    
    def test_long_direction(self):
        result = format_direction_for_ui(Direction.LONG)
        assert result == 'LONG'
    
    def test_short_direction(self):
        result = format_direction_for_ui(Direction.SHORT)
        assert result == 'SHORT'


class TestFormatPositionStatus:
    """Test position status formatting for UI."""
    
    def test_open_position(self):
        result = format_position_status_for_ui('OPEN')
        assert result == 'ACTIVE'
    
    def test_closed_position(self):
        result = format_position_status_for_ui('CLOSED')
        assert result == 'CLOSED'
    
    def test_closed_win_position(self):
        result = format_position_status_for_ui('CLOSED_WIN')
        assert result == 'CLOSED'
    
    def test_pending_position(self):
        result = format_position_status_for_ui('PENDING')
        assert result == 'PENDING'


class TestFormatSignalForUI:
    """Test signal formatting for UI."""
    
    def test_format_signal(self):
        timestamp = datetime.now(timezone.utc)
        
        # Create mock layer context
        layer_context = MagicMock(spec=ThreeLayerContext)
        layer_context.to_dict.return_value = {'bias': {}, 'risk': {}, 'game': {}}
        
        signal = EntrySignal(
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=18500.50,
            stop_loss=18450.00,
            tp1=18550.00,
            tp2=18600.00,
            position_size=1000.0,
            confidence=0.72,
            rationale=['TREND_STRENGTH'],
            timestamp=timestamp,
            layer_context=layer_context
        )
        
        result = format_signal_for_ui(signal)
        
        assert result['symbol'] == 'NAS100'
        assert result['direction'] == 'LONG'
        assert result['confidence'] == 72.0  # Percentage
        assert result['entry_price'] == 18500.50
        assert result['stop_loss'] == 18450.00
        assert result['tp1'] == 18550.00
        assert result['tp2'] == 18600.00
        assert result['status'] == 'ACTIVE'
        assert 'timestamp' in result
    
    def test_format_signal_with_custom_status(self):
        timestamp = datetime.now(timezone.utc)
        layer_context = MagicMock(spec=ThreeLayerContext)
        layer_context.to_dict.return_value = {}
        
        signal = EntrySignal(
            symbol='NAS100',
            direction=Direction.SHORT,
            entry_price=18500.00,
            stop_loss=18550.00,
            tp1=18450.00,
            tp2=18400.00,
            position_size=1000.0,
            confidence=0.65,
            rationale=[],
            timestamp=timestamp,
            layer_context=layer_context
        )
        
        result = format_signal_for_ui(signal, SignalStatus.CLOSED)
        
        assert result['direction'] == 'SHORT'
        assert result['status'] == 'CLOSED'


class TestFormatPositionForUI:
    """Test position formatting for UI."""
    
    def test_format_position(self):
        timestamp = datetime.now(timezone.utc)
        position = PositionState(
            trade_id='trade_001',
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=18500.00,
            position_size=1000.0,
            stop_loss=18450.00,
            tp1=18550.00,
            tp2=18600.00,
            current_price=18525.00,
            unrealized_pnl=25.00,
            realized_pnl=0.0,
            status='OPEN',
            opened_at=timestamp
        )
        
        result = format_position_for_ui(position)
        
        assert result['symbol'] == 'NAS100'
        assert result['direction'] == 'LONG'
        assert result['current_price'] == 18525.00
        assert result['unrealized_pnl'] == 25.00
        assert result['status'] == 'ACTIVE'
        assert 'trade_id' in result


class TestFormatAccountStateForUI:
    """Test account state formatting for UI."""
    
    def test_format_account_state(self):
        timestamp = datetime.now(timezone.utc)
        account = AccountState(
            account_id='acc_001',
            equity=100000.0,
            balance=95000.0,
            open_positions=2,
            daily_pnl=500.0,
            daily_loss_pct=0.005,
            margin_used=10000.0,
            margin_available=40000.0,
            timestamp=timestamp
        )
        
        result = format_account_state_for_ui(account)
        
        assert result['account_id'] == 'acc_001'
        assert result['equity'] == 100000.0
        assert result['daily_pnl_pct'] == 0.5  # Percentage
        assert 'utilization_pct' in result


class TestFormatBiasForUI:
    """Test bias formatting for UI."""
    
    def test_format_bias(self):
        timestamp = datetime.now(timezone.utc)
        feature_snapshot = FeatureSnapshot(
            raw_features={'adx_14': 30.0, 'rsi_14': 60.0, 'atr_percent_14': 0.005},
            feature_group_tags={},
            regime_at_inference={'vol_regime': 2},
            inference_timestamp=timestamp
        )
        
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=2,
            confidence=0.75,
            regime_override=False,
            rationale=['TREND_STRENGTH', 'MOMENTUM_SHIFT'],
            model_version='v1.0',
            feature_snapshot=feature_snapshot,
            timestamp=timestamp
        )
        
        result = format_bias_for_ui('NAS100', bias)
        
        assert result['direction'] == 'LONG'
        assert result['confidence'] == 75.0
        assert result['magnitude'] == 2
        assert 'feature_summary' in result
        assert result['feature_summary']['trend_strength'] == 30.0


class TestFormatRiskForUI:
    """Test risk structure formatting for UI."""
    
    def test_format_risk(self):
        timestamp = datetime.now(timezone.utc)
        risk = RiskOutput(
            timestamp=timestamp,
            position_size=1000.0,
            kelly_fraction=0.15,
            stop_price=18450.00,
            stop_method='atr',
            tp1_price=18550.00,
            tp2_price=18600.00,
            trail_config={'atr_multiple': 1.5},
            expected_value=0.05,
            ev_positive=True,
            size_breakdown={'base_size': 1000}
        )
        
        result = format_risk_for_ui('NAS100', risk)
        
        assert result['symbol'] == 'NAS100'
        assert result['kelly_fraction'] == 15.0  # Percentage
        assert result['ev_positive'] is True


class TestFormatGameOutputForUI:
    """Test game output formatting for UI."""
    
    def test_format_game_output(self):
        timestamp = datetime.now(timezone.utc)
        game = GameOutput(
            liquidity_map={'equal_highs': [], 'equal_lows': []},
            nearest_unswept_pool=None,
            trapped_positions={},
            forced_move_probability=0.3,
            nash_zones=[],
            kyle_lambda=0.5,
            game_state_aligned=True,
            game_state_summary='Test summary',
            adversarial_risk=AdversarialRisk.LOW,
            timestamp=timestamp
        )
        
        result = format_game_output_for_ui('NAS100', game)
        
        assert result['symbol'] == 'NAS100'
        assert result['forced_move_probability'] == 30.0  # Percentage
        assert result['adversarial_risk'] == 'LOW'
        assert result['game_state_aligned'] is True


class TestFormatConnectionStatus:
    """Test connection status formatting."""
    
    def test_format_live_status(self):
        result = format_connection_status(ConnectionStatus.LIVE)
        
        assert result['status'] == 'live'
        assert result['is_live'] is True
        assert result['is_demo'] is False
        assert result['is_error'] is False
    
    def test_format_demo_status(self):
        result = format_connection_status(ConnectionStatus.DEMO)
        
        assert result['status'] == 'demo'
        assert result['is_demo'] is True
    
    def test_format_error_status(self):
        result = format_connection_status(ConnectionStatus.ERROR)
        
        assert result['status'] == 'error'
        assert result['is_error'] is True


class TestFormatRegimeForUI:
    """Test regime formatting for UI."""
    
    def test_format_regime_normal(self):
        result = format_regime_for_ui(vol_regime=2, vix_level=20.0, event_risk='CLEAR')
        
        assert result['volatility_regime'] == 'NORMAL'
        assert result['volatility_color'] == 'blue'
        assert result['vix_level'] == 20.0
        assert result['event_risk'] == 'CLEAR'
    
    def test_format_regime_extreme(self):
        result = format_regime_for_ui(vol_regime=4, vix_level=40.0, event_risk='HIGH')
        
        assert result['volatility_regime'] == 'EXTREME'
        assert result['volatility_color'] == 'red'
    
    def test_format_regime_low(self):
        result = format_regime_for_ui(vol_regime=1, vix_level=12.0)
        
        assert result['volatility_regime'] == 'LOW'
        assert result['volatility_color'] == 'green'


class TestCreateLiveStateUpdate:
    """Test live state update creation."""
    
    def test_create_minimal_state(self):
        result = create_live_state_update('NAS100')
        
        assert result['symbol'] == 'NAS100'
        assert 'last_updated' in result
    
    def test_create_full_state(self):
        timestamp = datetime.now(timezone.utc)
        layer_context = MagicMock(spec=ThreeLayerContext)
        layer_context.to_dict.return_value = {}
        
        signal = EntrySignal(
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=18500.00,
            stop_loss=18450.00,
            tp1=18550.00,
            tp2=18600.00,
            position_size=1000.0,
            confidence=0.72,
            rationale=[],
            timestamp=timestamp,
            layer_context=layer_context
        )
        
        result = create_live_state_update(
            'NAS100',
            signal=signal,
            regime=2,
            vix_level=20.0,
            event_risk='CLEAR'
        )
        
        assert 'current_signal' in result
        assert 'regime' in result
        assert result['current_signal']['direction'] == 'LONG'


class TestCreateSessionControlUpdate:
    """Test session control update creation."""
    
    def test_create_session_control(self):
        result = create_session_control_update(
            trading_enabled=True,
            hard_logic_status='ACTIVE'
        )
        
        assert result['trading_enabled'] is True
        assert result['hard_logic_status'] == 'ACTIVE'
        assert 'last_updated' in result


class TestFirebaseBroadcaster:
    """Test Firebase broadcaster."""
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_init(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        
        assert broadcaster.client is mock_client
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_set_connection_status(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        broadcaster.set_connection_status(ConnectionStatus.LIVE)
        
        mock_client.rtdb_update.assert_called()
        call_args = mock_client.rtdb_update.call_args
        assert call_args[0][0] == '/connection_status'
        assert call_args[0][1]['status'] == 'live'
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_broadcast_session_control(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        broadcaster.broadcast_session_control(trading_enabled=True)
        
        mock_client.rtdb_update.assert_called()
        call_args = mock_client.rtdb_update.call_args
        assert call_args[0][0] == '/session_controls'
        assert call_args[0][1]['trading_enabled'] is True
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_broadcast_live_state(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        broadcaster.broadcast_live_state('NAS100', regime=2, vix_level=20.0)
        
        mock_client.rtdb_update.assert_called()
        call_args = mock_client.rtdb_update.call_args
        assert call_args[0][0] == '/live_state/NAS100'
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_broadcast_regime_state(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        broadcaster.broadcast_regime_state(vol_regime=2, vix_level=20.0)
        
        mock_client.rtdb_update.assert_called()
        call_args = mock_client.rtdb_update.call_args
        assert call_args[0][0] == '/regime_state'
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_broadcast_entry_signal(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        
        timestamp = datetime.now(timezone.utc)
        layer_context = MagicMock(spec=ThreeLayerContext)
        layer_context.to_dict.return_value = {}
        
        signal = EntrySignal(
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=18500.00,
            stop_loss=18450.00,
            tp1=18550.00,
            tp2=18600.00,
            position_size=1000.0,
            confidence=0.72,
            rationale=[],
            timestamp=timestamp,
            layer_context=layer_context
        )
        
        broadcaster.broadcast_entry_signal(signal)
        
        # Should call both firestore_set and rtdb_update
        assert mock_client.firestore_set.called
        assert mock_client.rtdb_update.called
    
    @patch('integration.firebase_broadcaster.FirebaseClient')
    def test_initialize_ui_state(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        broadcaster = FirebaseBroadcaster(mock_client)
        broadcaster.initialize_ui_state(['NAS100', 'SPY'])
        
        # Should initialize connection status, session controls, and live states
        assert mock_client.rtdb_update.call_count >= 4
