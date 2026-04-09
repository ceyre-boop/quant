"""
Unified Data Provider for Clawd Trading

Primary: Yahoo Finance (yfinance) - free, reliable, great for backtesting
Secondary: Polygon.io - backup for symbols Yahoo doesn't cover

Usage:
    from data.providers import DataProvider

    provider = DataProvider()

    # Live data
    price = provider.get_current_price("SPY")

    # Historical data for backtesting
    history = provider.get_historical_data("SPY", period="1y", interval="1h")
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

import yfinance as yf
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Symbol mappings between our symbols and provider tickers
SYMBOL_MAP = {
    # Our symbol -> Yahoo Finance ticker
    "NAS100": "QQQ",  # NASDAQ-100 ETF
    "US30": "DIA",  # Dow Jones ETF
    "SPX500": "SPY",  # S&P 500 ETF
    "XAUUSD": "GLD",  # Gold ETF (USD)
    "EURUSD": "EURUSD=X",  # EUR/USD forex
    "GBPUSD": "GBPUSD=X",  # GBP/USD forex
    "USDJPY": "JPY=X",  # USD/JPY forex
    "BTCUSD": "BTC-USD",  # Bitcoin
    "ETHUSD": "ETH-USD",  # Ethereum
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "TSLA": "TSLA",
    "AMZN": "AMZN",
    "GOOGL": "GOOGL",
    "NVDA": "NVDA",
    "META": "META",
}

# Symbols that work better with Polygon
POLYGON_ONLY_SYMBOLS = {
    "VIX": "VIX",  # Volatility index
    "DX-Y.NYB": "DXY",  # Dollar index
}


@dataclass
class MarketData:
    """Standardized market data structure."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Derived metrics
    change: float = 0.0
    change_percent: float = 0.0

    # ICT features (populated by feature builder)
    fvg_detected: bool = False
    liquidity_sweep: bool = False
    order_block: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "change": self.change,
            "change_percent": self.change_percent,
        }


class YahooFinanceProvider:
    """Yahoo Finance data provider - PRIMARY source."""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".Yahoo")
        self.logger.info("Yahoo Finance provider initialized")

    def get_ticker(self, symbol: str) -> str:
        """Map our symbol to Yahoo ticker."""
        return SYMBOL_MAP.get(symbol, symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol."""
        try:
            ticker = self.get_ticker(symbol)
            stock = yf.Ticker(ticker)

            # Get today's data
            hist = stock.history(period="1d", interval="1m")

            if hist.empty:
                self.logger.warning(f"No data from Yahoo for {symbol} ({ticker})")
                return None

            price = hist["Close"].iloc[-1]
            self.logger.debug(f"Yahoo: {symbol} = ${price:.2f}")
            return float(price)

        except Exception as e:
            self.logger.error(f"Yahoo price error for {symbol}: {e}")
            return None

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get full market data (OHLCV) for a symbol."""
        try:
            ticker = self.get_ticker(symbol)
            stock = yf.Ticker(ticker)

            # Get today's data with some history for context
            hist = stock.history(period="5d", interval="1h")

            if hist.empty or len(hist) < 2:
                self.logger.warning(f"Insufficient data from Yahoo for {symbol}")
                return None

            latest = hist.iloc[-1]
            prev = hist.iloc[-2]

            change = latest["Close"] - prev["Close"]
            change_pct = change / prev["Close"] if prev["Close"] != 0 else 0

            return MarketData(
                symbol=symbol,
                timestamp=latest.name.to_pydatetime(),
                open=float(latest["Open"]),
                high=float(latest["High"]),
                low=float(latest["Low"]),
                close=float(latest["Close"]),
                volume=int(latest["Volume"]),
                change=float(change),
                change_percent=float(change_pct),
            )

        except Exception as e:
            self.logger.error(f"Yahoo market data error for {symbol}: {e}")
            return None

    def get_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for backtesting.

        Args:
            symbol: Trading symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            DataFrame with OHLCV columns
        """
        try:
            ticker = self.get_ticker(symbol)
            stock = yf.Ticker(ticker)

            self.logger.info(
                f"Fetching {period} {interval} data for {symbol} ({ticker})"
            )

            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                self.logger.warning(f"No historical data from Yahoo for {symbol}")
                return None

            # Rename columns to standard format
            hist = hist.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            self.logger.info(f"Got {len(hist)} bars of {interval} data for {symbol}")
            return hist

        except Exception as e:
            self.logger.error(f"Yahoo historical error for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Get intraday (1m or 5m) data for recent days."""
        try:
            ticker = self.get_ticker(symbol)
            stock = yf.Ticker(ticker)

            # Yahoo limits: 7 days for 1m, 60 days for 5m
            if days <= 7:
                interval = "1m"
            else:
                interval = "5m"

            hist = stock.history(period=f"{days}d", interval=interval)

            if hist.empty:
                return None

            hist = hist.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            return hist

        except Exception as e:
            self.logger.error(f"Yahoo intraday error for {symbol}: {e}")
            return None


class PolygonProvider:
    """Polygon.io data provider - SECONDARY/backup source."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            self.logger = logging.getLogger(__name__ + ".Polygon")
            self.logger.warning("No Polygon API key provided!")
        else:
            self.logger = logging.getLogger(__name__ + ".Polygon")
            self.logger.info("Polygon provider initialized")

        self.base_url = "https://api.polygon.io/v2"

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Polygon API."""
        if not self.api_key:
            return None

        url = f"{self.base_url}{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f"Polygon API error {response.status_code}: {response.text}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Polygon request error: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Polygon."""
        try:
            # Map symbol if needed
            ticker = SYMBOL_MAP.get(symbol, symbol)

            # Get previous close (most reliable for current price)
            data = self._make_request(f"/aggs/ticker/{ticker}/prev")

            if data and "results" in data and len(data["results"]) > 0:
                price = data["results"][0].get("c")
                self.logger.debug(f"Polygon: {symbol} = ${price:.2f}")
                return float(price)

            return None

        except Exception as e:
            self.logger.error(f"Polygon price error for {symbol}: {e}")
            return None

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get full market data from Polygon."""
        try:
            ticker = SYMBOL_MAP.get(symbol, symbol)

            # Get previous close data
            data = self._make_request(f"/aggs/ticker/{ticker}/prev")

            if not data or "results" not in data or len(data["results"]) == 0:
                return None

            result = data["results"][0]

            return MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(result.get("t", 0) / 1000),
                open=float(result.get("o", 0)),
                high=float(result.get("h", 0)),
                low=float(result.get("l", 0)),
                close=float(result.get("c", 0)),
                volume=int(result.get("v", 0)),
                change=float(result.get("c", 0) - result.get("o", 0)),
                change_percent=float(
                    (result.get("c", 0) - result.get("o", 0)) / result.get("o", 1)
                ),
            )

        except Exception as e:
            self.logger.error(f"Polygon market data error for {symbol}: {e}")
            return None

    def get_historical_data(
        self, symbol: str, days: int = 365, multiplier: int = 1, timespan: str = "hour"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data from Polygon.

        Args:
            symbol: Trading symbol
            days: Number of days of history
            multiplier: Aggregation multiplier
            timespan: minute, hour, day, week, month, quarter, year
        """
        try:
            ticker = SYMBOL_MAP.get(symbol, symbol)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Format dates as YYYY-MM-DD
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            self.logger.info(
                f"Fetching {days} days of {timespan} data from Polygon for {symbol}"
            )

            # Build endpoint
            endpoint = f"/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_str}/{end_str}"

            data = self._make_request(
                endpoint, params={"adjusted": "true", "sort": "asc"}
            )

            if not data or "results" not in data:
                self.logger.warning(f"No data from Polygon for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data["results"])

            if df.empty:
                return None

            # Rename columns to standard format
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "t": "timestamp",
                }
            )

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            self.logger.info(f"Got {len(df)} bars from Polygon for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Polygon historical error for {symbol}: {e}")
            return None


class DataProvider:
    """
    Unified data provider that tries Yahoo first, falls back to Polygon.
    This is the main interface the rest of the system should use.
    """

    def __init__(self, polygon_api_key: Optional[str] = None):
        self.yahoo = YahooFinanceProvider()
        self.polygon = PolygonProvider(api_key=polygon_api_key)
        self.logger = logging.getLogger(__name__)

        # Symbols that should use Polygon directly
        self.polygon_symbols = set(POLYGON_ONLY_SYMBOLS.keys())

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (Yahoo primary, Polygon backup)."""
        # Try Polygon first for special symbols
        if symbol in self.polygon_symbols:
            price = self.polygon.get_current_price(symbol)
            if price:
                return price

        # Try Yahoo
        price = self.yahoo.get_current_price(symbol)
        if price:
            return price

        # Fallback to Polygon
        self.logger.info(f"Falling back to Polygon for {symbol}")
        return self.polygon.get_current_price(symbol)

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get full market data (Yahoo primary, Polygon backup)."""
        # Try Polygon first for special symbols
        if symbol in self.polygon_symbols:
            data = self.polygon.get_market_data(symbol)
            if data:
                return data

        # Try Yahoo
        data = self.yahoo.get_market_data(symbol)
        if data:
            return data

        # Fallback to Polygon
        self.logger.info(f"Falling back to Polygon for {symbol}")
        return self.polygon.get_market_data(symbol)

    def get_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for backtesting.

        Args:
            symbol: Trading symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
        """
        # Map period to days for Polygon fallback
        period_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }

        # Map interval to Polygon timespan
        interval_map = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "30m": ("minute", 30),
            "1h": ("hour", 1),
            "1d": ("day", 1),
        }

        # Try Yahoo first
        df = self.yahoo.get_historical_data(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            return df

        # Fallback to Polygon
        self.logger.info(f"Falling back to Polygon for {symbol} historical data")

        days = period_days.get(period, 365)
        timespan_info = interval_map.get(interval, ("hour", 1))

        return self.polygon.get_historical_data(
            symbol, days=days, multiplier=timespan_info[1], timespan=timespan_info[0]
        )

    def get_intraday_data(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Get intraday data (Yahoo primary, Polygon backup)."""
        df = self.yahoo.get_intraday_data(symbol, days=days)
        if df is not None and not df.empty:
            return df

        # Polygon fallback
        self.logger.info(f"Falling back to Polygon for {symbol} intraday")

        if days <= 7:
            return self.polygon.get_historical_data(
                symbol, days=days, multiplier=1, timespan="minute"
            )
        else:
            return self.polygon.get_historical_data(
                symbol, days=min(days, 30), multiplier=5, timespan="minute"
            )

    def test_connection(self, symbol: str = "SPY") -> Dict[str, Any]:
        """Test data provider connections."""
        results = {
            "symbol": symbol,
            "yahoo": {},
            "polygon": {},
        }

        # Test Yahoo
        try:
            yahoo_price = self.yahoo.get_current_price(symbol)
            results["yahoo"]["status"] = "OK" if yahoo_price else "FAILED"
            results["yahoo"]["price"] = yahoo_price
        except Exception as e:
            results["yahoo"]["status"] = "ERROR"
            results["yahoo"]["error"] = str(e)

        # Test Polygon
        try:
            poly_price = self.polygon.get_current_price(symbol)
            results["polygon"]["status"] = "OK" if poly_price else "FAILED"
            results["polygon"]["price"] = poly_price
        except Exception as e:
            results["polygon"]["status"] = "ERROR"
            results["polygon"]["error"] = str(e)

        return results


# Convenience functions
def get_provider(polygon_key: Optional[str] = None) -> DataProvider:
    """Get a configured DataProvider instance."""
    # Use provided key or environment variable
    key = polygon_key or os.getenv("POLYGON_API_KEY")
    return DataProvider(polygon_api_key=key)
