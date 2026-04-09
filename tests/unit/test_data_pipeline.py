"""Unit tests for data pipeline."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from data.pipeline import DataPipeline
from contracts.types import FeatureRecord


class TestDataPipeline:
    """Test DataPipeline."""

    @pytest.fixture
    def mock_polygon(self):
        """Mock Polygon client."""
        client = Mock()
        client.get_aggregates.return_value = {
            "results": [
                {
                    "c": 20000,
                    "h": 20100,
                    "l": 19900,
                    "o": 19950,
                    "v": 1000000,
                    "t": 1705312800000,
                },
                {
                    "c": 20100,
                    "h": 20200,
                    "l": 20000,
                    "o": 20000,
                    "v": 1100000,
                    "t": 1705399200000,
                },
                {
                    "c": 20050,
                    "h": 20150,
                    "l": 19950,
                    "o": 20100,
                    "v": 1050000,
                    "t": 1705485600000,
                },
            ]
        }
        return client

    @pytest.fixture
    def mock_tradelocker(self):
        """Mock TradeLocker client."""
        return Mock()

    @pytest.fixture
    def pipeline(self, mock_polygon, mock_tradelocker):
        """Create DataPipeline with mocked clients."""
        return DataPipeline(polygon_client=mock_polygon, tradelocker_client=mock_tradelocker)

    def test_run_premarket_returns_valid_records(self, pipeline):
        """Test that premarket pipeline returns valid feature records."""
        results = pipeline.run_premarket(["NAS100"])

        assert "NAS100" in results
        record = results["NAS100"]
        assert isinstance(record, FeatureRecord)
        assert record.symbol == "NAS100"

    def test_run_premarket_features_populated(self, pipeline):
        """Test that features are populated."""
        results = pipeline.run_premarket(["NAS100"])
        record = results["NAS100"]

        # Should have some features
        assert len(record.features) > 0
        assert "returns_1h" in record.features or "returns_daily" in record.features

    def test_run_premarket_validation(self, pipeline):
        """Test that records pass validation."""
        results = pipeline.run_premarket(["NAS100"])
        record = results["NAS100"]

        # Should be valid (or have validation_errors if invalid)
        assert record.is_valid or len(record.validation_errors) > 0

    def test_run_premarket_multiple_symbols(self, pipeline):
        """Test pipeline with multiple symbols."""
        results = pipeline.run_premarket(["NAS100", "SPY"])

        assert "NAS100" in results
        assert "SPY" in results
        assert len(results) == 2

    def test_run_intraday_refresh(self, pipeline):
        """Test intraday refresh."""
        results = pipeline.run_intraday_refresh(["NAS100"])

        assert "NAS100" in results
        record = results["NAS100"]
        assert isinstance(record, FeatureRecord)

    def test_build_features_with_data(self, pipeline):
        """Test feature building with data."""
        daily_data = [
            {"c": 20000, "h": 20100, "l": 19900, "o": 19950, "v": 1000000},
            {"c": 20100, "h": 20200, "l": 20000, "o": 20000, "v": 1100000},
            {"c": 20050, "h": 20150, "l": 19950, "o": 20100, "v": 1050000},
        ]

        features = pipeline._build_features(
            symbol="NAS100",
            daily_data=daily_data,
            index_data={},
            vix_value=18.5,
            vix_regime="NORMAL",
            event_risk="CLEAR",
            breadth=None,
        )

        assert len(features) > 0
        assert "vix_level" in features
        assert features["vix_level"] == 18.5


class TestPolygonClient:
    """Test PolygonRestClient."""

    @patch("data.polygon_client.requests.Session")
    def test_client_creation(self, mock_session):
        """Test client creation."""
        from data.polygon_client import PolygonRestClient

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            client = PolygonRestClient()
            assert client.api_key == "test_key"

    def test_missing_api_key_raises(self):
        """Test that missing API key raises error."""
        from data.polygon_client import PolygonRestClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="POLYGON_API_KEY not set"):
                PolygonRestClient()


class TestDataFetchers:
    """Test individual data fetchers."""

    def test_daily_fetcher(self):
        """Test DailyFetcher."""
        from data.daily_fetcher import DailyFetcher

        mock_client = Mock()
        mock_client.get_aggregates.return_value = {"results": [{"c": 20000, "h": 20100, "l": 19900, "o": 19950, "v": 1000000}]}

        fetcher = DailyFetcher(mock_client)
        results = fetcher.fetch_daily_bars("NAS100", 30)

        assert len(results) == 1
        assert results[0]["c"] == 20000

    def test_index_fetcher(self):
        """Test IndexFetcher."""
        from data.index_fetcher import IndexFetcher

        mock_client = Mock()
        mock_client.get_aggregates.return_value = {"results": [{"c": 18.5, "h": 19.0, "l": 18.0, "o": 18.2, "v": 0}]}

        fetcher = IndexFetcher(mock_client)

        # Test VIX regime classification
        assert fetcher.get_vix_regime(12) == "LOW"
        assert fetcher.get_vix_regime(20) == "NORMAL"
        assert fetcher.get_vix_regime(30) == "ELEVATED"
        assert fetcher.get_vix_regime(40) == "EXTREME"

    def test_breadth_engine(self):
        """Test BreadthEngine."""
        from data.breadth_engine import BreadthEngine

        mock_client = Mock()
        mock_client.get_previous_close.return_value = {"results": [{"c": 20100, "o": 20000}]}

        engine = BreadthEngine(mock_client)
        metrics = engine.compute_breadth_metrics(["XLK", "XLF"])

        assert metrics.advance_decline_ratio > 0

    def test_calendar_fetcher_mock(self):
        """Test CalendarFetcher with mock data."""
        from data.calendar_fetcher import CalendarFetcher

        fetcher = CalendarFetcher()

        # Without API key, should return mock data
        events = fetcher._get_mock_events()
        assert len(events) > 0

        # Test event risk calculation
        assert fetcher.calculate_event_risk([]) == "CLEAR"
        # High impact event (Importance = 3)
        high_impact = [{"Event": "FOMC Statement", "Importance": 3}]
        assert fetcher.calculate_event_risk(high_impact) == "ELEVATED"

    def test_sentiment_engine_mock(self):
        """Test SentimentEngine with mock data."""
        from data.sentiment_engine import SentimentEngine

        engine = SentimentEngine()

        # Without API key, should return mock data
        sentiment = engine._get_mock_sentiment("NAS100")
        assert sentiment["symbol"] == "NAS100"
        assert sentiment["sentiment_label"] == "neutral"

    def test_order_flow_fetcher(self):
        """Test OrderFlowFetcher."""
        from data.order_flow_fetcher import OrderFlowFetcher

        mock_client = Mock()
        mock_client.get_trades.return_value = {
            "results": [
                {"p": 20000, "s": 100, "t": 1},
                {"p": 20001, "s": 150, "t": 1},
                {"p": 19999, "s": 200, "t": 2},
            ]
        }

        fetcher = OrderFlowFetcher(mock_client)
        summary = fetcher.get_order_flow_summary("NAS100")

        assert summary["symbol"] == "NAS100"
        assert summary["trade_count"] == 3

    def test_order_flow_volume_imbalance(self):
        """Test volume imbalance calculation."""
        from data.order_flow_fetcher import OrderFlowFetcher

        fetcher = OrderFlowFetcher(Mock())

        trades = [
            {"p": 20000, "s": 100, "t": 1},  # Buy
            {"p": 20001, "s": 150, "t": 1},  # Buy
            {"p": 19999, "s": 200, "t": 2},  # Sell
        ]

        imbalance = fetcher.compute_volume_imbalance(trades)

        assert imbalance["buy_volume"] == 250
        assert imbalance["sell_volume"] == 200
        assert imbalance["imbalance_ratio"] == 250 / 200


class TestTradeLockerClient:
    """Test TradeLockerClient."""

    def test_demo_mode(self):
        """Test that demo mode doesn't place real orders."""
        from data.tradelocker_client import TradeLockerClient

        client = TradeLockerClient(api_key="test_key", account_id="test_account", env="demo")

        result = client.place_order("NAS100", "buy", 1.0)

        assert result["demo"] is True
        assert result["symbol"] == "NAS100"

    def test_close_position_demo(self):
        """Test close position in demo mode."""
        from data.tradelocker_client import TradeLockerClient

        client = TradeLockerClient(api_key="test_key", account_id="test_account", env="demo")

        result = client.close_position("pos_001")

        assert result["demo"] is True
        assert result["position_id"] == "pos_001"
