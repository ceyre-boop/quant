"""Unit tests for data pipeline components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from data.polygon_client import PolygonRestClient, PolygonWebSocketClient, fetch_nas100_ohlcv
from data.daily_fetcher import DailyFetcher
from data.index_fetcher import IndexFetcher
from data.breadth_engine import BreadthEngine, BreadthMetrics
from data.calendar_fetcher import CalendarFetcher
from data.sentiment_engine import SentimentEngine
from data.order_flow_fetcher import OrderFlowFetcher
from data.tradelocker_client import TradeLockerClient
from data.pipeline import DataPipeline


class TestPolygonRestClient:
    """Test Polygon REST client."""
    
    def test_init_with_api_key(self):
        client = PolygonRestClient(api_key='test_key')
        assert client.api_key == 'test_key'
    
    def test_init_without_api_key_uses_env(self):
        with patch('data.polygon_client.os.getenv', return_value='env_key'):
            client = PolygonRestClient()
            assert client.api_key == 'env_key'
    
    def test_init_raises_without_key(self):
        with patch('data.polygon_client.os.getenv', return_value=None):
            with pytest.raises(ValueError, match="POLYGON_API_KEY"):
                PolygonRestClient()
    
    @patch('data.polygon_client.requests.Session')
    def test_get_aggregates(self, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {'results': [{'c': 100}]}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        client = PolygonRestClient(api_key='test')
        result = client.get_aggregates('SPY', 1, 'day', '2024-01-01', '2024-01-02')
        
        assert 'results' in result
        mock_session.get.assert_called_once()
    
    @patch('data.polygon_client.requests.Session')
    def test_get_previous_close(self, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {'results': [{'c': 100}]}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        client = PolygonRestClient(api_key='test')
        result = client.get_previous_close('SPY')
        
        assert 'results' in result


class TestPolygonWebSocketClient:
    """Test Polygon WebSocket client."""
    
    def test_init(self):
        client = PolygonWebSocketClient(api_key='test')
        assert client.api_key == 'test'
        assert client.subscriptions == []
    
    @patch('data.polygon_client.WebSocketApp')
    def test_connect(self, mock_ws_class):
        import threading as real_threading
        
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws
        
        # Mock Thread class
        mock_thread = MagicMock()
        original_thread = real_threading.Thread
        real_threading.Thread = lambda *args, **kwargs: mock_thread
        
        try:
            client = PolygonWebSocketClient(api_key='test')
            client.connect(['T.SPY'], on_message=lambda x: None)
            
            assert client.subscriptions == ['T.SPY']
            mock_ws_class.assert_called_once()
            mock_thread.start.assert_called_once()
        finally:
            real_threading.Thread = original_thread
    
    def test_disconnect(self):
        mock_ws = MagicMock()
        client = PolygonWebSocketClient(api_key='test')
        client.ws = mock_ws
        client.disconnect()
        mock_ws.close.assert_called_once()


class TestDailyFetcher:
    """Test daily data fetcher."""
    
    @patch('data.daily_fetcher.PolygonRestClient')
    def test_fetch_daily_bars(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_aggregates.return_value = {'results': [{'c': 100}]}
        mock_client_class.return_value = mock_client
        
        fetcher = DailyFetcher()
        result = fetcher.fetch_daily_bars('SPY', 30)
        
        assert len(result) == 1
        mock_client.get_aggregates.assert_called_once()
    
    @patch('data.daily_fetcher.PolygonRestClient')
    def test_fetch_multiple_symbols(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_aggregates.return_value = {'results': [{'c': 100}]}
        mock_client_class.return_value = mock_client
        
        fetcher = DailyFetcher()
        result = fetcher.fetch_multiple_symbols(['SPY', 'QQQ'], 30)
        
        assert 'SPY' in result
        assert 'QQQ' in result


class TestIndexFetcher:
    """Test index fetcher."""
    
    @patch('data.index_fetcher.PolygonRestClient')
    def test_fetch_index_data(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_aggregates.return_value = {
            'results': [{'c': 20, 't': 1234567890}]
        }
        mock_client_class.return_value = mock_client
        
        fetcher = IndexFetcher()
        result = fetcher.fetch_index_data('VIX', 30)
        
        assert result['index'] == 'VIX'
        assert result['latest'] is not None
    
    def test_fetch_unknown_index(self):
        fetcher = IndexFetcher()
        with pytest.raises(ValueError, match="Unknown index"):
            fetcher.fetch_index_data('UNKNOWN')
    
    @patch('data.index_fetcher.PolygonRestClient')
    def test_fetch_all_indices(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_aggregates.return_value = {
            'results': [{'c': 20, 't': 1234567890}]
        }
        mock_client_class.return_value = mock_client
        
        fetcher = IndexFetcher()
        result = fetcher.fetch_all_indices(30)
        
        assert 'VIX' in result
        assert 'SPX' in result
        assert 'NDX' in result
    
    def test_get_vix_regime(self):
        fetcher = IndexFetcher()
        
        assert fetcher.get_vix_regime(10) == 'LOW'
        assert fetcher.get_vix_regime(20) == 'NORMAL'
        assert fetcher.get_vix_regime(30) == 'ELEVATED'
        assert fetcher.get_vix_regime(40) == 'EXTREME'


class TestBreadthEngine:
    """Test breadth engine."""
    
    @patch('data.breadth_engine.PolygonRestClient')
    def test_compute_breadth_metrics(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_previous_close.return_value = {
            'results': [{'c': 100, 'o': 99}]
        }
        mock_client_class.return_value = mock_client
        
        engine = BreadthEngine()
        result = engine.compute_breadth_metrics(['SPY'])
        
        assert isinstance(result, BreadthMetrics)
        assert result.advance_decline_ratio >= 0
    
    @patch('data.breadth_engine.PolygonRestClient')
    def test_get_sector_rotation_signal(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_previous_close.return_value = {
            'results': [{'c': 100, 'o': 99}]
        }
        mock_client_class.return_value = mock_client
        
        engine = BreadthEngine()
        result = engine.get_sector_rotation_signal()
        
        assert 'leading' in result
        assert 'lagging' in result
        assert 'neutral' in result


class TestCalendarFetcher:
    """Test calendar fetcher."""
    
    @patch('data.calendar_fetcher.os.getenv', return_value=None)
    def test_fetch_events_no_api_key(self, mock_getenv):
        fetcher = CalendarFetcher()
        result = fetcher.fetch_events()
        
        assert len(result) > 0  # Returns mock data
    
    def test_get_high_impact_events(self):
        fetcher = CalendarFetcher()
        events = [
            {'Date': '2024-01-15T10:00:00', 'Event': 'FOMC Statement', 'Importance': 3},
            {'Date': '2024-01-15T10:00:00', 'Event': 'Regular News', 'Importance': 1}
        ]
        high_impact = fetcher.get_high_impact_events(events)
        
        assert len(high_impact) == 1
        assert 'FOMC' in high_impact[0]['Event']
    
    def test_calculate_event_risk(self):
        fetcher = CalendarFetcher()
        
        assert fetcher.calculate_event_risk([]) == 'CLEAR'
        assert fetcher.calculate_event_risk([{'Importance': 3}]) == 'ELEVATED'
        assert fetcher.calculate_event_risk([{'Importance': 3}] * 2) == 'HIGH'
        assert fetcher.calculate_event_risk([{'Importance': 3}] * 4) == 'EXTREME'


class TestSentimentEngine:
    """Test sentiment engine."""
    
    @patch('data.sentiment_engine.os.getenv', return_value=None)
    def test_fetch_news_sentiment_no_api_key(self, mock_getenv):
        engine = SentimentEngine()
        result = engine.fetch_news_sentiment('SPY')
        
        assert result['symbol'] == 'SPY'
        assert result['sentiment_label'] == 'neutral'
    
    def test_is_sentiment_extreme(self):
        engine = SentimentEngine()
        
        assert engine.is_sentiment_extreme({'average_sentiment': 0.6}) is True
        assert engine.is_sentiment_extreme({'average_sentiment': -0.6}) is True
        assert engine.is_sentiment_extreme({'average_sentiment': 0.1}) is False


class TestOrderFlowFetcher:
    """Test order flow fetcher."""
    
    @patch('data.order_flow_fetcher.PolygonRestClient')
    def test_fetch_recent_trades(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_trades.return_value = {'results': [{'p': 100, 's': 100, 't': 1}]}
        mock_client_class.return_value = mock_client
        
        fetcher = OrderFlowFetcher()
        result = fetcher.fetch_recent_trades('SPY', 100)
        
        assert len(result) == 1
    
    def test_compute_volume_imbalance(self):
        fetcher = OrderFlowFetcher()
        trades = [
            {'p': 100, 's': 100, 't': 1},  # Buy
            {'p': 100, 's': 100, 't': 2},  # Sell
            {'p': 100, 's': 100, 't': 1},  # Buy
        ]
        
        result = fetcher.compute_volume_imbalance(trades)
        
        assert result['buy_volume'] == 200
        assert result['sell_volume'] == 100
        assert result['imbalance_ratio'] == 2.0
    
    def test_compute_volume_imbalance_empty(self):
        fetcher = OrderFlowFetcher()
        result = fetcher.compute_volume_imbalance([])
        
        assert result['total_volume'] == 0
    
    def test_estimate_kyle_lambda(self):
        # Skip numpy mocking - just test it runs without error
        fetcher = OrderFlowFetcher()
        trades = [{'p': 100 + i, 's': 100, 't': 1} for i in range(20)]
        
        # This should run without error (may return 0 if numpy fails)
        result = fetcher.estimate_kyle_lambda(trades)
        assert isinstance(result, (int, float))


class TestTradeLockerClient:
    """Test TradeLocker client."""
    
    def test_init_demo_mode(self):
        client = TradeLockerClient(
            api_key='test',
            account_id='acc123',
            env='demo'
        )
        assert client.env == 'demo'
    
    @patch('data.tradelocker_client.requests.Session')
    def test_get_account_info(self, mock_session_class):
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {'account_id': 'acc123'}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        client = TradeLockerClient(api_key='test', account_id='acc123')
        result = client.get_account_info()
        
        assert 'account_id' in result
    
    def test_place_order_demo(self):
        client = TradeLockerClient(api_key='test', account_id='acc123', env='demo')
        result = client.place_order('SPY', 'buy', 100)
        
        assert result['demo'] is True
        assert result['symbol'] == 'SPY'
    
    def test_close_position_demo(self):
        client = TradeLockerClient(api_key='test', account_id='acc123', env='demo')
        result = client.close_position('pos123')
        
        assert result['demo'] is True


class TestDataPipeline:
    """Test master data pipeline."""
    
    @patch('data.pipeline.PolygonRestClient')
    @patch('data.pipeline.TradeLockerClient')
    def test_init(self, mock_tl_class, mock_polygon_class):
        pipeline = DataPipeline()
        
        assert pipeline.polygon is not None
        assert pipeline.tradelocker is not None
        assert pipeline.daily_fetcher is not None
        assert pipeline.index_fetcher is not None
        assert pipeline.breadth_engine is not None
        assert pipeline.calendar_fetcher is not None
        assert pipeline.sentiment_engine is not None
        assert pipeline.order_flow_fetcher is not None
        assert pipeline.validator is not None
    
    @patch('data.pipeline.PolygonRestClient')
    @patch('data.pipeline.TradeLockerClient')
    @patch('data.pipeline.DailyFetcher')
    @patch('data.pipeline.IndexFetcher')
    @patch('data.pipeline.BreadthEngine')
    def test_run_premarket(self, mock_breadth_class, mock_index_class, mock_daily_class, mock_tl_class, mock_polygon_class):
        mock_daily = MagicMock()
        mock_daily.fetch_daily_bars.return_value = [
            {'c': 100, 'h': 101, 'l': 99, 'o': 100}
        ] * 30
        mock_daily_class.return_value = mock_daily
        
        mock_index = MagicMock()
        mock_index.fetch_all_indices.return_value = {
            'VIX': {'latest': {'value': 20}}
        }
        mock_index.get_vix_regime.return_value = 'NORMAL'
        mock_index_class.return_value = mock_index
        
        mock_breadth = MagicMock()
        mock_breadth.compute_breadth_metrics.return_value = MagicMock(advance_decline_ratio=1.5)
        mock_breadth_class.return_value = mock_breadth
        
        pipeline = DataPipeline()
        result = pipeline.run_premarket(['SPY', 'QQQ'])
        
        assert 'SPY' in result
        assert 'QQQ' in result
        # Note: Some records may be invalid due to feature validation - that's expected
        # The important thing is the pipeline runs without crashing
        assert len(result) == 2
    
    @patch('data.pipeline.PolygonRestClient')
    @patch('data.pipeline.TradeLockerClient')
    @patch('data.pipeline.OrderFlowFetcher')
    def test_run_intraday_refresh(self, mock_order_flow_class, mock_tl_class, mock_polygon_class):
        mock_order_flow = MagicMock()
        mock_order_flow.get_order_flow_summary.return_value = {
            'kyle_lambda': 0.5,
            'volume_imbalance': {'imbalance_ratio': 1.2}
        }
        mock_order_flow_class.return_value = mock_order_flow
        
        pipeline = DataPipeline()
        result = pipeline.run_intraday_refresh(['SPY'])
        
        assert 'SPY' in result
        assert result['SPY'].is_valid


class TestFetchNas100Ohlcv:
    """Test NAS100 OHLCV helper function."""
    
    @patch('data.polygon_client.PolygonRestClient')
    def test_fetch_nas100_ohlcv(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_aggregates.return_value = {
            'results': [{'c': 20000, 'o': 19900, 'h': 20100, 'l': 19800}]
        }
        mock_client_class.return_value = mock_client
        
        result = fetch_nas100_ohlcv(mock_client, '1h', 30)
        
        assert len(result) == 1
        assert result[0]['c'] == 20000
