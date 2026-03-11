"""Daily Fetcher - Daily OHLCV data fetching."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from data.polygon_client import PolygonRestClient

logger = logging.getLogger(__name__)


class DailyFetcher:
    """Fetches daily OHLCV data for symbols."""
    
    def __init__(self, polygon_client: Optional[PolygonRestClient] = None):
        self.client = polygon_client or PolygonRestClient()
    
    def fetch_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 252
    ) -> List[Dict[str, Any]]:
        """Fetch daily bars for a symbol.
        
        Args:
            symbol: Ticker symbol
            lookback_days: Number of trading days to fetch (default ~1 year)
        
        Returns:
            List of daily OHLCV bars
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)  # Extra for weekends
        
        try:
            response = self.client.get_aggregates(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d'),
                limit=500
            )
            
            results = response.get('results', [])
            logger.info(f"Fetched {len(results)} daily bars for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch daily bars for {symbol}: {e}")
            return []
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch daily bars for multiple symbols.
        
        Returns:
            Dict mapping symbol to list of bars
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_daily_bars(symbol, lookback_days)
        return results
