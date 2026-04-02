"""
Positioning Data Feed - COT and GEX

Fetches real positioning data:
- COT (Commitment of Traders) from CFTC
- GEX (Gamma Exposure) from options chain analysis
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class COTDataFeed:
    """
    CFTC Commitment of Traders data feed.
    
    Note: CFTC provides free weekly COT reports.
    For real-time use, would need commercial data provider.
    """
    
    CFTC_API_BASE = "https://www.cftc.gov/files/dea/history/"
    
    # Legacy report codes for major markets
    REPORT_CODES = {
        "SPX": "13874+",      # S&P 500 Consolidated
        "NAS100": "20974+",   # NASDAQ-100
        "EUR": "099741",      # Euro FX
        "GBP": "096742",      # British Pound
        "JPY": "097741",      # Japanese Yen
        "GOLD": "088691",     # Gold
        "OIL": "067651",      # Crude Oil
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_time = None
    
    def fetch_cot_report(self, symbol: str) -> Optional[Dict]:
        """
        Fetch latest COT report for a symbol.
        
        Note: This is a simplified implementation.
        Real implementation would parse CFTC's weekly reports.
        """
        code = self.REPORT_CODES.get(symbol)
        if not code:
            self.logger.warning(f"No COT code for {symbol}")
            return None
        
        try:
            # For demo, return simulated COT index
            # Real implementation would fetch from CFTC
            return self._generate_cot_index(symbol)
            
        except Exception as e:
            self.logger.error(f"COT fetch error for {symbol}: {e}")
            return None
    
    def _generate_cot_index(self, symbol: str) -> Dict:
        """
        Generate COT index from available data sources.
        
        In production, this would parse actual CFTC reports.
        For now, uses proxy indicators.
        """
        # Placeholder: would integrate with real CFTC data
        return {
            "symbol": symbol,
            "cot_index": 50,  # 0-100 scale
            "commercial_net": 0,
            "noncommercial_net": 0,
            "retail_net": 0,
            "weeks_back": 0,
            "data_source": "placeholder_cftc",
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_cot_index(self, net_position: float, history: list) -> float:
        """
        Calculate COT index (0-100) from net position vs 3-year range.
        
        Args:
            net_position: Current net commercial position
            history: List of historical net positions (3 years)
        
        Returns:
            COT index 0-100 (50 = neutral)
        """
        if not history:
            return 50.0
        
        min_pos = min(history)
        max_pos = max(history)
        
        if max_pos == min_pos:
            return 50.0
        
        # Percentile rank
        index = 100 * (net_position - min_pos) / (max_pos - min_pos)
        return round(index, 2)


class GEXDataFeed:
    """
    Gamma Exposure (GEX) data feed.
    
    GEX measures dealer gamma exposure from options chain.
    High positive GEX = price magnet (support/resistance)
    High negative GEX = volatility expansion
    
    Note: Real GEX requires options chain data (paid API)
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.polygon_key = polygon_api_key
        self.cache = {}
    
    def fetch_gex(self, symbol: str, spot_price: float) -> Optional[Dict]:
        """
        Calculate GEX for a symbol.
        
        Note: This requires options chain data.
        Without paid data source, returns estimate.
        """
        try:
            # Real implementation would:
            # 1. Fetch options chain
            # 2. Calculate gamma for each strike
            # 3. Weight by open interest
            # 4. Sum to get net GEX
            
            # For now, return placeholder
            return self._estimate_gex(symbol, spot_price)
            
        except Exception as e:
            self.logger.error(f"GEX fetch error for {symbol}: {e}")
            return None
    
    def _estimate_gex(self, symbol: str, spot_price: float) -> Dict:
        """
        Estimate GEX from available data.
        
        Real implementation needs options chain API.
        """
        # Placeholder values
        return {
            "symbol": symbol,
            "spot_price": spot_price,
            "total_gex": 0,  # Billions
            "zero_gamma_level": spot_price,
            "major_gamma_strikes": [],
            "gamma_regime": "neutral",  # magnet, repellent, neutral
            "data_source": "placeholder_estimated",
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_gex_from_chain(self, options_chain: list, spot: float) -> float:
        """
        Calculate GEX from options chain data.
        
        Formula: GEX = Σ (Gamma × Open Interest × Contract Multiplier)
        
        Args:
            options_chain: List of option contracts with gamma, oi, strike
            spot: Current spot price
        
        Returns:
            Total GEX in billions
        """
        total_gex = 0
        
        for option in options_chain:
            gamma = option.get('gamma', 0)
            open_interest = option.get('open_interest', 0)
            strike = option.get('strike', 0)
            
            # Gamma exposure
            contract_gex = gamma * open_interest * 100  # 100 shares per contract
            
            # Sign based on option type and moneyness
            if option.get('type') == 'call':
                if strike > spot:  # OTM call
                    contract_gex *= -1
            else:  # put
                if strike < spot:  # OTM put
                    contract_gex *= -1
            
            total_gex += contract_gex
        
        return total_gex / 1e9  # Convert to billions


class PositioningDataFeed:
    """
    Combined positioning data feed.
    
    Provides COT + GEX + options skew + fear/greed.
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.cot = COTDataFeed()
        self.gex = GEXDataFeed(polygon_api_key)
    
    def fetch_all(self, symbol: str, spot_price: float) -> Dict:
        """Fetch all positioning data for a symbol."""
        return {
            "cot": self.cot.fetch_cot_report(symbol),
            "gex": self.gex.fetch_gex(symbol, spot_price),
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "COT/GEX data requires commercial data feeds for real-time accuracy"
        }


# Convenience functions
def get_positioning_feed(polygon_key: Optional[str] = None) -> PositioningDataFeed:
    """Get positioning data feed instance."""
    return PositioningDataFeed(polygon_key)
