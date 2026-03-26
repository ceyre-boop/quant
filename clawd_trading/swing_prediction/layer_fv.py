"""
LAYER A — Fair Value Deviation
Produces z-score measuring how far asset price is from statistical fair value.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FairValueResult:
    """Result from fair value calculation."""
    z_score: float
    direction: str  # "long", "short", or "neutral"
    score: float  # Normalized score for composite (-3 to +3)
    methods_used: list
    raw_values: Dict[str, float]
    

class FairValueLayer:
    """
    Layer A: Fair Value Deviation
    Routes to correct model based on asset class.
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_a_fair_value"]
    
    async def compute(
        self, 
        symbol: str, 
        asset_class: str, 
        data: Dict[str, Any]
    ) -> FairValueResult:
        """
        Compute fair value z-score based on asset class.
        """
        self.logger.debug(f"Computing FV for {symbol} ({asset_class})")
        
        if asset_class == "equities":
            return self._compute_equity_fv(symbol, data)
        elif asset_class == "forex":
            return self._compute_forex_fv(symbol, data)
        elif asset_class == "commodities":
            return self._compute_commodity_fv(symbol, data)
        elif asset_class == "crypto":
            return self._compute_crypto_fv(symbol, data)
        else:
            self.logger.warning(f"Unknown asset class {asset_class}, using equity model")
            return self._compute_equity_fv(symbol, data)
    
    def _compute_equity_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Equity Fair Value Models:
        1. Price vs 200-week MA deviation (primary)
        2. Earnings yield vs bond yield spread (Fed Model)
        3. P/E z-score vs 5yr sector average
        """
        methods_used = []
        z_scores = []
        weights = []
        raw_values = {}
        
        # Method 1: Price vs 200-week MA
        if all(k in data for k in ["current_price", "ma_200w", "std_200w"]):
            deviation = (data["current_price"] - data["ma_200w"]) / data["std_200w"]
            z_scores.append(deviation)
            weights.append(self.config["equity_weights"]["ma200w_deviation"])
            methods_used.append("ma200w_deviation")
            raw_values["ma200w_zscore"] = deviation
        
        # Method 2: Fed Model (Earnings yield vs bond yield)
        if all(k in data for k in ["forward_pe", "treasury_10y_yield"]):
            earnings_yield = 1 / data["forward_pe"]
            spread = earnings_yield - data["treasury_10y_yield"]
            
            # Z-score of spread vs 5-year history
            if "spread_history_mean" in data and "spread_history_std" in data:
                spread_z = (spread - data["spread_history_mean"]) / data["spread_history_std"]
                z_scores.append(spread_z)
                weights.append(self.config["equity_weights"]["fed_model_spread"])
                methods_used.append("fed_model_spread")
                raw_values["fed_spread"] = spread
                raw_values["fed_spread_zscore"] = spread_z
        
        # Method 3: P/E z-score
        if all(k in data for k in ["current_pe", "sector_pe_mean_5yr", "sector_pe_std_5yr"]):
            pe_z = (data["current_pe"] - data["sector_pe_mean_5yr"]) / data["sector_pe_std_5yr"]
            z_scores.append(pe_z)
            weights.append(self.config["equity_weights"]["pe_zscore"])
            methods_used.append("pe_zscore")
            raw_values["pe_zscore"] = pe_z
        
        # Calculate weighted average z-score
        if not z_scores:
            self.logger.warning(f"No FV methods available for {symbol}")
            return FairValueResult(
                z_score=0,
                direction="neutral",
                score=0,
                methods_used=[],
                raw_values={}
            )
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))
        
        # Determine direction
        if weighted_z >= self.config["z_score_thresholds"]["notable"]:
            direction = "short"  # Overvalued = short for mean reversion
        elif weighted_z <= -self.config["z_score_thresholds"]["notable"]:
            direction = "long"  # Undervalued = long for mean reversion
        else:
            direction = "neutral"
        
        # Convert to score (-3 to +3 scale)
        score = np.clip(weighted_z, -3, 3)
        
        return FairValueResult(
            z_score=weighted_z,
            direction=direction,
            score=score,
            methods_used=methods_used,
            raw_values=raw_values
        )
    
    def _compute_forex_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Forex Fair Value Models:
        1. Real interest rate differential
        2. PPP deviation
        """
        methods_used = []
        z_scores = []
        weights = []
        raw_values = {}
        
        # Method 1: Real rate differential
        if "real_rate_diff" in data:
            diff = data["real_rate_diff"]
            if "rate_diff_history_mean" in data and "rate_diff_history_std" in data:
                z = (diff - data["rate_diff_history_mean"]) / data["rate_diff_history_std"]
                z_scores.append(z)
                weights.append(self.config["forex_weights"]["real_rate_differential"])
                methods_used.append("real_rate_differential")
                raw_values["rate_diff_zscore"] = z
        
        # Method 2: PPP deviation
        if "ppp_deviation" in data:
            dev = data["ppp_deviation"]
            if "ppp_history_mean" in data and "ppp_history_std" in data:
                z = (dev - data["ppp_history_mean"]) / data["ppp_history_std"]
                z_scores.append(z)
                weights.append(self.config["forex_weights"]["ppp_deviation"])
                methods_used.append("ppp_deviation")
                raw_values["ppp_zscore"] = z
        
        if not z_scores:
            return FairValueResult(z_score=0, direction="neutral", score=0, 
                                 methods_used=[], raw_values={})
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))
        
        direction = "short" if weighted_z >= 2 else "long" if weighted_z <= -2 else "neutral"
        score = np.clip(weighted_z, -3, 3)
        
        return FairValueResult(
            z_score=weighted_z,
            direction=direction,
            score=score,
            methods_used=methods_used,
            raw_values=raw_values
        )
    
    def _compute_commodity_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Commodity Fair Value Models:
        1. Cost of production vs spot price
        2. Inventory deviation from seasonal
        """
        methods_used = []
        z_scores = []
        weights = []
        raw_values = {}
        
        # Method 1: Cost of production
        if all(k in data for k in ["spot_price", "cost_of_production"]):
            margin = (data["spot_price"] - data["cost_of_production"]) / data["cost_of_production"]
            if "margin_history_mean" in data and "margin_history_std" in data:
                z = (margin - data["margin_history_mean"]) / data["margin_history_std"]
                z_scores.append(z)
                weights.append(self.config["commodity_weights"]["cost_of_production"])
                methods_used.append("cost_of_production")
                raw_values["margin_zscore"] = z
        
        # Method 2: Inventory deviation
        if "inventory_zscore" in data:
            z_scores.append(data["inventory_zscore"])
            weights.append(self.config["commodity_weights"]["inventory_deviation"])
            methods_used.append("inventory_deviation")
            raw_values["inventory_zscore"] = data["inventory_zscore"]
        
        if not z_scores:
            return FairValueResult(z_score=0, direction="neutral", score=0,
                                 methods_used=[], raw_values={})
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))
        
        direction = "short" if weighted_z >= 2 else "long" if weighted_z <= -2 else "neutral"
        score = np.clip(weighted_z, -3, 3)
        
        return FairValueResult(
            z_score=weighted_z,
            direction=direction,
            score=score,
            methods_used=methods_used,
            raw_values=raw_values
        )
    
    def _compute_crypto_fv(self, symbol: str, data: Dict) -> FairValueResult:
        """
        Crypto Fair Value Models:
        1. MVRV ratio (Market Cap / Realized Cap)
        2. NVT ratio (Network Value / Transactions)
        3. Price vs Realized Price
        """
        methods_used = []
        z_scores = []
        weights = []
        raw_values = {}
        
        # Method 1: MVRV
        if "mvrv_ratio" in data:
            mvrv = data["mvrv_ratio"]
            # Historical mean ~1.0, std varies by cycle
            mvrv_mean = data.get("mvrv_history_mean", 1.0)
            mvrv_std = data.get("mvrv_history_std", 0.5)
            z = (mvrv - mvrv_mean) / mvrv_std
            z_scores.append(z)
            weights.append(self.config["crypto_weights"]["mvrv_ratio"])
            methods_used.append("mvrv_ratio")
            raw_values["mvrv_zscore"] = z
        
        # Method 2: NVT
        if "nvt_ratio" in data:
            nvt = data["nvt_ratio"]
            nvt_mean = data.get("nvt_history_mean", 50)
            nvt_std = data.get("nvt_history_std", 20)
            z = (nvt - nvt_mean) / nvt_std
            z_scores.append(z)
            weights.append(self.config["crypto_weights"]["nvt_ratio"])
            methods_used.append("nvt_ratio")
            raw_values["nvt_zscore"] = z
        
        # Method 3: Price vs Realized Price
        if "price_vs_realized_zscore" in data:
            z_scores.append(data["price_vs_realized_zscore"])
            weights.append(self.config["crypto_weights"]["price_vs_realized"])
            methods_used.append("price_vs_realized")
            raw_values["price_vs_realized_zscore"] = data["price_vs_realized_zscore"]
        
        if not z_scores:
            return FairValueResult(z_score=0, direction="neutral", score=0,
                                 methods_used=[], raw_values={})
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        weighted_z = sum(z * w for z, w in zip(z_scores, normalized_weights))
        
        direction = "short" if weighted_z >= 2 else "long" if weighted_z <= -2 else "neutral"
        score = np.clip(weighted_z, -3, 3)
        
        return FairValueResult(
            z_score=weighted_z,
            direction=direction,
            score=score,
            methods_used=methods_used,
            raw_values=raw_values
        )
