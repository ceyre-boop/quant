"""Unit tests for firebase/client.py"""

import pytest
import sys
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Mock firebase_admin before importing the client
sys.modules['firebase_admin'] = Mock()
sys.modules['firebase_admin.credentials'] = Mock()
sys.modules['firebase_admin.firestore'] = Mock()
sys.modules['firebase_admin.db'] = Mock()
sys.modules['firebase_admin.exceptions'] = Mock()

from contracts.types import (
    Direction, Magnitude, VolRegime, TrendRegime, RiskAppetite,
    MomentumRegime, EventRisk, AdversarialRisk, FeatureGroup,
    RegimeState, BiasOutput, RiskOutput, GameOutput, TrappedPositions,
    FeatureRecord, PositionState, AccountState
)


class TestFirebaseClient:
    """Test FirebaseClient with mocked Firebase Admin."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton state before each test."""
        # Import here to ensure mocks are in place
        from firebase.client import FirebaseClient
        FirebaseClient._instance = None
        FirebaseClient._initialized = False
        yield
        # Cleanup
        FirebaseClient._instance = None
        FirebaseClient._initialized = False
    
    @pytest.fixture
    def mock_firebase(self):
        """Mock Firebase Admin SDK."""
        import firebase_admin
        
        mock_db_instance = Mock()
        mock_collection = Mock()
        mock_document = Mock()
        mock_collection.return_value = mock_document
        mock_db_instance.collection = mock_collection
        
        # Mock document methods
        mock_doc_ref = Mock()
        mock_doc_ref.id = 'mock_doc_id'
        mock_document.document.return_value = mock_doc_ref
        mock_document.return_value = mock_doc_ref
        
        firebase_admin.get_app.side_effect = ValueError("Not initialized")
        firebase_admin.initialize_app = Mock()
        
        # Mock firestore client
        import firebase_admin.firestore as mock_firestore
        mock_firestore.client.return_value = mock_db_instance
        mock_firestore.SERVER_TIMESTAMP = 'SERVER_TIMESTAMP'
        
        # Mock db
        import firebase_admin.db as mock_db
        mock_rtdb_ref = Mock()
        mock_db.reference.return_value = mock_rtdb_ref
        
        yield {
            'db_instance': mock_db_instance,
            'collection': mock_collection,
            'document': mock_document,
            'doc_ref': mock_doc_ref,
            'firestore': mock_firestore,
            'rtdb': mock_rtdb_ref
        }
    
    @pytest.fixture
    def firebase_client(self, mock_firebase):
        """Create FirebaseClient instance with mocked Firebase."""
        from firebase.client import FirebaseClient
        
        with patch.dict('os.environ', {
            'FIREBASE_PROJECT_ID': 'test-project',
            'FIREBASE_SERVICE_ACCOUNT_PATH': '',
            'FIREBASE_RTDB_URL': 'https://test-project.firebaseio.com'
        }):
            client = FirebaseClient()
            client.db = mock_firebase['db_instance']
            return client
    
    def test_client_singleton(self, mock_firebase):
        """Test that FirebaseClient is a singleton."""
        from firebase.client import FirebaseClient
        
        with patch.dict('os.environ', {
            'FIREBASE_PROJECT_ID': 'test-project',
            'FIREBASE_RTDB_URL': 'https://test.firebaseio.com'
        }):
            client1 = FirebaseClient()
            client2 = FirebaseClient()
            assert client1 is client2
    
    def test_write_feature_record(self, firebase_client, mock_firebase):
        """Test writing feature record."""
        record = FeatureRecord(
            symbol='NAS100',
            timestamp=datetime.utcnow(),
            timeframe='1h',
            features={'f1': 1.0},
            raw_data={},
            is_valid=True,
            validation_errors=[],
            metadata={}
        )
        
        doc_id = firebase_client.write_feature_record(record)
        assert 'NAS100' in doc_id
        
        # Verify Firestore was called
        mock_firebase['db_instance'].collection.assert_called_with('feature_records')
    
    def test_write_invalid_feature_record_raises(self, firebase_client):
        """Test that invalid feature records raise error."""
        record = FeatureRecord(
            symbol='NAS100',
            timestamp=datetime.utcnow(),
            timeframe='1h',
            features={},
            raw_data={},
            is_valid=False,
            validation_errors=['Missing required features'],
            metadata={}
        )
        
        with pytest.raises(ValueError, match='Cannot write invalid feature record'):
            firebase_client.write_feature_record(record)
    
    def test_write_bias_output(self, firebase_client, mock_firebase):
        """Test writing bias output."""
        bias = BiasOutput(
            direction=Direction.LONG,
            magnitude=Magnitude.NORMAL,
            confidence=0.75,
            regime_override=False,
            rationale=[FeatureGroup.TREND_STRENGTH.value],
            model_version='v1.0',
            feature_snapshot={}
        )
        
        doc_id = firebase_client.write_bias_output('NAS100', bias)
        assert 'NAS100' in doc_id
        
        mock_firebase['db_instance'].collection.assert_called_with('bias_outputs')
    
    def test_write_risk_structure(self, firebase_client, mock_firebase):
        """Test writing risk structure."""
        risk = RiskOutput(
            position_size=1000.0,
            kelly_fraction=0.15,
            stop_price=19500.0,
            stop_method='atr',
            tp1_price=20000.0,
            tp2_price=20500.0,
            trail_config={},
            expected_value=0.05,
            ev_positive=True,
            size_breakdown={}
        )
        
        doc_id = firebase_client.write_risk_structure('NAS100', risk, 'bias_001')
        assert 'NAS100' in doc_id
        
        mock_firebase['db_instance'].collection.assert_called_with('risk_structures')
    
    def test_write_game_output(self, firebase_client, mock_firebase):
        """Test writing game output."""
        trapped = TrappedPositions([], [], 0, 0, 0)
        game = GameOutput(
            liquidity_map={'equal_highs': [], 'equal_lows': []},
            nearest_unswept_pool=None,
            trapped_positions=trapped,
            forced_move_probability=0.3,
            nash_zones=[],
            kyle_lambda=0.5,
            game_state_aligned=True,
            game_state_summary='Test',
            adversarial_risk=AdversarialRisk.LOW
        )
        
        doc_id = firebase_client.write_game_output('NAS100', game)
        assert 'NAS100' in doc_id
        
        mock_firebase['db_instance'].collection.assert_called_with('game_outputs')
    
    def test_write_position(self, firebase_client, mock_firebase):
        """Test writing position state."""
        position = PositionState(
            trade_id='trade_001',
            symbol='NAS100',
            direction=Direction.LONG,
            entry_price=20000.0,
            position_size=1000.0,
            stop_loss=19500.0,
            tp1=20500.0,
            tp2=21000.0,
            current_price=20100.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            status='OPEN',
            opened_at=datetime.utcnow()
        )
        
        doc_id = firebase_client.write_position(position)
        assert doc_id == 'trade_001'
        
        mock_firebase['db_instance'].collection.assert_called_with('positions')
    
    def test_write_account_state(self, firebase_client, mock_firebase):
        """Test writing account state."""
        account = AccountState(
            account_id='acc_001',
            equity=100000.0,
            balance=95000.0,
            open_positions=1,
            daily_pnl=500.0,
            daily_loss_pct=0.005,
            margin_used=10000.0,
            margin_available=40000.0,
            timestamp=datetime.utcnow()
        )
        
        doc_id = firebase_client.write_account_state(account)
        assert doc_id == 'acc_001'
        
        mock_firebase['db_instance'].collection.assert_called_with('account_state')
    
    def test_write_system_log(self, firebase_client, mock_firebase):
        """Test writing system log."""
        doc_id = firebase_client.write_system_log(
            level='INFO',
            message='Test message',
            module='test_module',
            metadata={'key': 'value'}
        )
        
        assert doc_id is not None
        mock_firebase['db_instance'].collection.assert_called_with('system_logs')
    
    def test_batch_write_bias_outputs(self, firebase_client, mock_firebase):
        """Test batch writing bias outputs."""
        outputs = {
            'NAS100': BiasOutput(
                Direction.LONG, Magnitude.NORMAL, 0.75, False,
                [FeatureGroup.TREND_STRENGTH.value], 'v1', {}
            ),
            'SPY': BiasOutput(
                Direction.SHORT, Magnitude.SMALL, 0.6, False,
                [FeatureGroup.MOMENTUM_SHIFT.value], 'v1', {}
            )
        }
        
        doc_ids = firebase_client.batch_write_bias_outputs(outputs)
        assert len(doc_ids) == 2
        assert all('NAS100' in id or 'SPY' in id for id in doc_ids)


class TestFirebaseInitialization:
    """Test Firebase initialization scenarios."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton state before each test."""
        from firebase.client import FirebaseClient
        FirebaseClient._instance = None
        FirebaseClient._initialized = False
        yield
        FirebaseClient._instance = None
        FirebaseClient._initialized = False
    
    def test_missing_project_id_raises(self):
        """Test that missing project ID raises ValueError."""
        from firebase.client import FirebaseClient
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match='FIREBASE_PROJECT_ID not set'):
                FirebaseClient()
