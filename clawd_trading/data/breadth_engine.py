"""Breadth Engine - Market breadth metric computation."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from data.polygon_client import PolygonRestClient

logger = logging.getLogger(__name__)


@dataclass
class BreadthMetrics:
    """Market breadth metrics."""
    advance_decline_ratio: float
    percent_stocks_above_sma20: Optional[float]
    percent_stocks_above_sma50: Optional[float]
    percent_stocks_above_sma200: Optional[float]
    new_highs_lows_ratio: float
    bullish_percent_index: Optional[float]


class BreadthEngine:
    """Computes market breadth metrics."""
    
    # Key sector ETFs for breadth analysis
    SECTOR_ETFS = [
        'XLK',  # Technology
        'XLF',  # Financials
        'XLE',  # Energy
        'XLI',  # Industrials
        'XLP',  # Consumer Staples
        'XLY',  # Consumer Discretionary
        'XLB',  # Materials
        'XLU',  # Utilities
        'XLRE', # Real Estate
        'XBI'   # Biotech
    ]
    
    def __init__(self, polygon_client: Optional[PolygonRestClient] = None):
        self.client = polygon_client or PolygonRestClient()
    
    def compute_breadth_metrics(
        self,
        symbols: Optional[List[str]] = None
    ) -> BreadthMetrics:
        """Compute breadth metrics for a universe of stocks.
        
        Args:
            symbols: List of symbols (defaults to sector ETFs)
        
        Returns:
            BreadthMetrics
        """
        symbols = symbols or self.SECTOR_ETFS
        
        try:
            # Fetch previous close for all symbols
            advances = 0
            declines = 0
            unchanged = 0
            
            for symbol in symbols:
                try:
                    data = self.client.get_previous_close(symbol)
                    results = data.get('results', [])
                    if results:
                        bar = results[0]
                        change = bar.get('c', 0) - bar.get('o', 0)
                        if change > 0:
                            advances += 1
                        elif change < 0:
                            declines += 1
                        else:
                            unchanged += 1
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    continue
            
            total = advances + declines + unchanged
            
            ad_ratio = advances / max(declines, 1)
            
            return BreadthMetrics(
                advance_decline_ratio=ad_ratio,
                percent_stocks_above_sma20=None,  # Would need historical data
                percent_stocks_above_sma50=None,
                percent_stocks_above_sma200=None,
                new_highs_lows_ratio=0.0,  # Would need historical data
                bullish_percent_index=None
            )
            
        except Exception as e:
            logger.error(f"Failed to compute breadth metrics: {e}")
            return BreadthMetrics(
                advance_decline_ratio=1.0,
                percent_stocks_above_sma20=None,
                percent_stocks_above_sma50=None,
                percent_stocks_above_sma200=None,
                new_highs_lows_ratio=0.0,
                bullish_percent_index=None
            )
    
    def get_sector_rotation_signal(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns.
        
        Returns:
            Dict with leading/lagging sectors
        """
        sector_performance = {}
        
        for sector in self.SECTOR_ETFS:
            try:
                data = self.client.get_previous_close(sector)
                results = data.get('results', [])
                if results:
                    bar = results[0]
                    performance = (bar.get('c', 0) - bar.get('o', 0)) / bar.get('o', 1)
                    sector_performance[sector] = performance
            except Exception as e:
                logger.debug(f"Failed to fetch {sector}: {e}")
        
        if not sector_performance:
            return {'leading': [], 'lagging': [], 'neutral': []}
        
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        n = len(sorted_sectors)
        leading = [s[0] for s in sorted_sectors[:max(1, n//3)]]
        lagging = [s[0] for s in sorted_sectors[-max(1, n//3):]]
        neutral = [s[0] for s in sorted_sectors[max(1, n//3):-max(1, n//3)]]
        
        return {
            'leading': leading,
            'lagging': lagging,
            'neutral': neutral,
            'performance': sector_performance
        }
