"""Index Fetcher - VIX, SPX, NDX, RTY index data."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from data.polygon_client import PolygonRestClient

logger = logging.getLogger(__name__)


class IndexFetcher:
    """Fetches major market index data."""

    # Polygon.io ticker symbols
    INDICES = {
        "VIX": "I:VIX",  # Volatility Index
        "SPX": "I:SPX",  # S&P 500
        "NDX": "I:NDX",  # Nasdaq 100
        "RTY": "I:RUT",  # Russell 2000
        "DJI": "I:DJI",  # Dow Jones
    }

    def __init__(self, polygon_client: Optional[PolygonRestClient] = None):
        self.client = polygon_client or PolygonRestClient()

    def fetch_index_data(self, index: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Fetch data for a specific index.

        Args:
            index: Index name (VIX, SPX, NDX, RTY, DJI)
            lookback_days: Number of days to fetch

        Returns:
            Dict with index data including latest value and history
        """
        ticker = self.INDICES.get(index.upper())
        if not ticker:
            raise ValueError(f"Unknown index: {index}. Available: {list(self.INDICES.keys())}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        try:
            # Fetch daily aggregates
            response = self.client.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )

            results = response.get("results", [])

            if not results:
                logger.warning(f"No data returned for {index}")
                return {"index": index, "latest": None, "history": []}

            latest = results[-1] if results else None

            return {
                "index": index,
                "ticker": ticker,
                "latest": {
                    "value": latest["c"] if latest else None,
                    "timestamp": latest.get("t") if latest else None,
                },
                "history": results,
                "change_1d": self._calculate_change(results, 1),
                "change_5d": self._calculate_change(results, 5),
                "change_20d": self._calculate_change(results, 20),
            }

        except Exception as e:
            logger.error(f"Failed to fetch {index} data: {e}")
            return {"index": index, "latest": None, "history": [], "error": str(e)}

    def fetch_all_indices(self, lookback_days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Fetch data for all major indices.

        Returns:
            Dict mapping index name to data
        """
        results = {}
        for index in self.INDICES.keys():
            results[index] = self.fetch_index_data(index, lookback_days)
        return results

    def _calculate_change(self, history: list, days: int) -> Optional[float]:
        """Calculate percentage change over N days."""
        if len(history) < days + 1:
            return None

        current = history[-1]["c"]
        previous = history[-(days + 1)]["c"]

        return (current - previous) / previous if previous else None

    def get_vix_regime(self, vix_value: float) -> str:
        """Classify VIX value into regime.

        Returns:
            'LOW' (< 15), 'NORMAL' (15-25), 'ELEVATED' (25-35), 'EXTREME' (> 35)
        """
        if vix_value < 15:
            return "LOW"
        elif vix_value < 25:
            return "NORMAL"
        elif vix_value < 35:
            return "ELEVATED"
        else:
            return "EXTREME"
