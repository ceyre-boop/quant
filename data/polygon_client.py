"""Polygon.io Client

REST and WebSocket client for Polygon.io market data.
"""

import json
import os
import logging
from typing import Callable, Optional, Dict, List, Any
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from websocket import WebSocketApp

logger = logging.getLogger(__name__)


class PolygonRestClient:
    """Polygon.io REST API client."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key: Optional[str] = api_key or os.getenv("POLYGON_API_KEY")
        # os.getenv with a default always returns str, but we annotate explicitly for mypy
        self.base_url: str = base_url or os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")  # type: ignore[assignment]

        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set")

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to Polygon API."""
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polygon API error: {e}")
            raise

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,  # minute, hour, day, week, month, quarter, year
        from_date: str,  # YYYY-MM-DD
        to_date: str,  # YYYY-MM-DD
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """Get aggregate bars (OHLCV) for a ticker.

        Args:
            ticker: Stock symbol (e.g., 'SPY', 'QQQ')
            multiplier: Size of timespan multiplier
            timespan: Size of time window
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Adjust for splits
            sort: Sort order (asc/desc)
            limit: Max results
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": str(adjusted).lower(), "sort": sort, "limit": limit}
        return self._get(endpoint, params)

    def get_daily_open_close(self, ticker: str, date: str) -> Dict[str, Any]:  # YYYY-MM-DD
        """Get daily open/close for a ticker."""
        endpoint = f"/v1/open-close/{ticker}/{date}"
        return self._get(endpoint)

    def get_previous_close(self, ticker: str) -> Dict[str, Any]:
        """Get previous day's close for a ticker."""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        return self._get(endpoint)

    def get_snapshot(self, ticker: str) -> Dict[str, Any]:
        """Get current snapshot for a ticker."""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        return self._get(endpoint)

    def get_grouped_daily(self, date: str, adjusted: bool = True) -> Dict[str, Any]:  # YYYY-MM-DD
        """Get daily aggregates for all tickers in a group."""
        endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params = {"adjusted": str(adjusted).lower()}
        return self._get(endpoint, params)

    def get_trades(
        self,
        ticker: str,
        timestamp: Optional[str] = None,
        order: str = "asc",
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Get trades for a ticker."""
        endpoint = f"/v3/trades/{ticker}"
        params = {"order": order, "limit": limit}
        if timestamp:
            params["timestamp"] = timestamp
        return self._get(endpoint, params)


class PolygonWebSocketClient:
    """Polygon.io WebSocket client for real-time data."""

    WEBSOCKET_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.ws: Optional[WebSocketApp] = None
        self.subscriptions: List[str] = []
        self.on_message_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_close_callback: Optional[Callable] = None

    def on_message(self, ws, message):
        """Handle incoming message."""
        if self.on_message_callback:
            self.on_message_callback(message)
        else:
            logger.debug(f"Received: {message}")

    def on_error(self, ws, error):
        """Handle error."""
        logger.error(f"WebSocket error: {error}")
        if self.on_error_callback:
            self.on_error_callback(error)

    def on_close(self, ws, close_status_code, close_msg):
        """Handle connection close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.on_close_callback:
            self.on_close_callback(close_status_code, close_msg)

    def on_open(self, ws):
        """Handle connection open."""
        logger.info("WebSocket connected")
        # Authenticate
        ws.send(json.dumps({"action": "auth", "params": self.api_key}))
        # Subscribe to channels
        if self.subscriptions:
            ws.send(json.dumps({"action": "subscribe", "params": ",".join(self.subscriptions)}))

    def connect(
        self,
        subscriptions: List[str],
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
    ):
        """Connect to WebSocket.

        Args:
            subscriptions: List of channels (e.g., ['T.NAS100', 'Q.SPY'])
            on_message: Message callback
            on_error: Error callback
            on_close: Close callback
        """
        self.subscriptions = subscriptions
        self.on_message_callback = on_message
        self.on_error_callback = on_error
        self.on_close_callback = on_close

        self.ws = WebSocketApp(
            self.WEBSOCKET_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        # Run in separate thread
        import threading

        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()

    def subscribe(self, channels: List[str]):
        """Subscribe to additional channels."""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps({"action": "subscribe", "params": ",".join(channels)}))
            self.subscriptions.extend(channels)

    def unsubscribe(self, channels: List[str]):
        """Unsubscribe from channels."""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps({"action": "unsubscribe", "params": ",".join(channels)}))
            self.subscriptions = [s for s in self.subscriptions if s not in channels]


def fetch_nas100_ohlcv(client: PolygonRestClient, timeframe: str = "1h", lookback_days: int = 30) -> List[Dict[str, Any]]:
    """Fetch NAS100 OHLCV data.

    Args:
        client: Polygon REST client
        timeframe: '1m', '5m', '15m', '1h', '4h', 'daily'
        lookback_days: Number of days to fetch

    Returns:
        List of OHLCV bars
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    timeframe_map = {
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "1h": (1, "hour"),
        "4h": (4, "hour"),
        "daily": (1, "day"),
    }

    multiplier, timespan = timeframe_map.get(timeframe, (1, "hour"))

    try:
        response = client.get_aggregates(
            ticker="NQ",  # NAS100 futures symbol
            multiplier=multiplier,
            timespan=timespan,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )
        return response.get("results", [])
    except Exception as e:
        logger.error(f"Failed to fetch NAS100 data: {e}")
        return []
