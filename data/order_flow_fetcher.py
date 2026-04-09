"""Order Flow Fetcher - Trade tick order flow analysis."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from data.polygon_client import PolygonRestClient

logger = logging.getLogger(__name__)


class OrderFlowFetcher:
    """Fetches and analyzes order flow data."""

    def __init__(self, polygon_client: Optional[PolygonRestClient] = None):
        self.client = polygon_client or PolygonRestClient()

    def fetch_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch recent trade ticks.

        Args:
            symbol: Ticker symbol
            limit: Max trades to fetch

        Returns:
            List of trade records
        """
        try:
            response = self.client.get_trades(symbol, limit=limit)
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Failed to fetch trades for {symbol}: {e}")
            return []

    def compute_volume_imbalance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute buy/sell volume imbalance.

        Returns:
            Dict with imbalance metrics
        """
        if not trades:
            return {
                "buy_volume": 0,
                "sell_volume": 0,
                "total_volume": 0,
                "imbalance_ratio": 1.0,
            }

        buy_volume = 0
        sell_volume = 0

        for trade in trades:
            size = trade.get("s", 0)
            # Tick direction: 1 = uptick, 2 = downtick, 3 = zero tick
            tick_direction = trade.get("t", 3)

            if tick_direction == 1:
                buy_volume += size
            elif tick_direction == 2:
                sell_volume += size
            else:
                # Zero tick - split evenly
                buy_volume += size / 2
                sell_volume += size / 2

        total_volume = buy_volume + sell_volume

        # Imbalance ratio: > 1 means more buying, < 1 means more selling
        imbalance_ratio = buy_volume / max(sell_volume, 1)

        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "imbalance_ratio": imbalance_ratio,
            "buy_percentage": buy_volume / max(total_volume, 1) * 100,
        }

    def estimate_kyle_lambda(self, trades: List[Dict[str, Any]]) -> float:
        """Estimate Kyle's lambda (price impact per unit volume).

        Kyle's lambda measures the price impact of informed trading.
        High lambda = informed traders are active.

        Args:
            trades: List of trade records

        Returns:
            Kyle lambda estimate
        """
        if len(trades) < 10:
            return 0.0

        # Calculate signed volume and price changes
        signed_volumes = []
        price_changes = []

        prev_price = None
        for trade in trades:
            price = trade.get("p", 0)
            size = trade.get("s", 0)
            tick_direction = trade.get("t", 3)

            # Sign volume based on tick direction
            if tick_direction == 1:
                signed_vol = size
            elif tick_direction == 2:
                signed_vol = -size
            else:
                signed_vol = 0

            if prev_price is not None:
                price_change = price - prev_price
                signed_volumes.append(signed_vol)
                price_changes.append(price_change)

            prev_price = price

        if not signed_volumes:
            return 0.0

        # Simple OLS: price_change = lambda * signed_volume + noise
        # lambda = Cov(price_change, signed_volume) / Var(signed_volume)

        import numpy as np

        x = np.array(signed_volumes)
        y = np.array(price_changes)

        # Add small regularization to avoid division by zero
        variance = np.var(x) + 1e-10
        covariance = np.cov(x, y)[0, 1] if len(x) > 1 else 0

        lambda_estimate = covariance / variance

        return float(lambda_estimate)

    def get_order_flow_summary(self, symbol: str) -> Dict[str, Any]:
        """Get complete order flow summary for a symbol.

        Returns:
            Dict with all order flow metrics
        """
        trades = self.fetch_recent_trades(symbol)

        if not trades:
            return {
                "symbol": symbol,
                "trade_count": 0,
                "volume_imbalance": None,
                "kyle_lambda": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
            }

        imbalance = self.compute_volume_imbalance(trades)
        kyle_lambda = self.estimate_kyle_lambda(trades)

        return {
            "symbol": symbol,
            "trade_count": len(trades),
            "volume_imbalance": imbalance,
            "kyle_lambda": kyle_lambda,
            "timestamp": datetime.utcnow().isoformat(),
        }
