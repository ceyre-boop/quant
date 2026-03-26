"""
LAYER C — Regime Classification
Hurst Exponent, ADX, Volatility — determines if mean-reversion signals are valid
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RegimeResult:
    score: float  # -3 to +3
    regime: str  # "mean_reverting", "random_walk", "trending"
    hurst: float
    adx: float
    vol_percentile: float
    is_valid_for_mr: bool  # Mean-reversion signals valid?
    regime_classification: str  # 'MEAN_REVERT' | 'TRANSITIONAL' | 'TRENDING' | 'BREAKOUT'


class RegimeLayer:
    """
    Layer C: Regime Classification
    A mean-reversion signal in a trending market is a death trap.
    """
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config["layer_c_regime"]
    
    def compute_hurst_exponent(self, prices: np.ndarray, window: int = 252) -> float:
        """
        Hurst Exponent via R/S analysis on rolling window.
        
        Algorithm:
        1. Compute log returns: r = log(p[t] / p[t-1])
        2. For each sub-period length n in [10, 20, 40, 80, 160]:
           a. Divide series into chunks of length n
           b. For each chunk: compute range R (max - min of cumulative deviations from mean)
           c. For each chunk: compute std S of returns
           d. Compute mean(R/S) for this n
        3. Regress log(R/S) on log(n) → slope = Hurst exponent H
        
        Interpretation:
        H < 0.4: strongly mean-reverting
        0.4–0.5: mildly mean-reverting
        0.5: random walk
        0.5–0.6: mildly trending
        H > 0.6: strongly trending
        
        Args:
            prices: numpy array of price data
            window: rolling window (default 252 trading days = 1 year)
        
        Returns:
            Hurst exponent H
        """
        if len(prices) < window:
            self.logger.warning(f"Insufficient data for Hurst: {len(prices)} < {window}")
            return 0.5
        
        # Use last 'window' prices
        series = prices[-window:]
        
        # Compute log returns
        log_returns = np.diff(np.log(series))
        
        # Sub-period lengths
        lags = [10, 20, 40, 80, 160]
        rs_values = []
        
        for lag in lags:
            if lag >= len(log_returns):
                continue
                
            # Divide into chunks
            n_chunks = len(log_returns) // lag
            rs_chunks = []
            
            for i in range(n_chunks):
                chunk = log_returns[i*lag:(i+1)*lag]
                
                # Mean of chunk
                mean_chunk = np.mean(chunk)
                
                # Cumulative deviations from mean
                cumsum = np.cumsum(chunk - mean_chunk)
                
                # Range R
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation S
                S = np.std(chunk)
                
                if S > 0:
                    rs_chunks.append(R / S)
            
            if rs_chunks:
                rs_values.append((np.log(lag), np.log(np.mean(rs_chunks))))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression of log(R/S) on log(lag)
        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        
        # Slope = Hurst exponent
        A = np.vstack([x, np.ones(len(x))]).T
        H, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return round(H, 4)
    
    def compute_adx_regime(self, ohlcv: Dict) -> Tuple[str, float, float]:
        """
        ADX (Average Directional Index, period=14) + Volatility Regime Classification.
        
        ADX levels:
        - adx < 20: no trend, ranging
        - 20–25: weak trend forming
        - 25–40: established trend
        - > 40: strong trend
        
        Volatility percentile:
        hist_vol = std(log_returns, 20) * sqrt(252)
        vol_percentile = percentile_rank(hist_vol, 1yr)
        
        Regime classification:
        - MEAN_REVERT: adx < 25 AND vol_percentile < 35
        - TRANSITIONAL: adx 20–30 OR vol_percentile 30–50
        - TRENDING: adx > 30 AND vol_percentile > 50
        - BREAKOUT: adx > 40 AND vol_percentile > 75
        
        Only MEAN_REVERT and TRANSITIONAL support swing reversion trades.
        
        Returns:
            (regime_classification, adx_value, vol_percentile)
        """
        adx = ohlcv.get("adx", 20)
        vol_pct = ohlcv.get("volatility_percentile", 50)
        
        # Classify regime
        if adx > 40 and vol_pct > 75:
            regime = "BREAKOUT"
        elif adx > 30 and vol_pct > 50:
            regime = "TRENDING"
        elif adx < 25 and vol_pct < 35:
            regime = "MEAN_REVERT"
        else:
            regime = "TRANSITIONAL"
        
        return regime, adx, vol_pct
    
    async def compute(self, symbol: str, data: Dict[str, Any]) -> RegimeResult:
        """Classify current market regime."""
        self.logger.debug(f"Classifying regime for {symbol}")
        
        # Calculate Hurst Exponent if price data available
        hurst = 0.5
        if "prices" in data:
            prices = np.array(data["prices"])
            window = self.config.get("hurst_window", 252)
            hurst = self.compute_hurst_exponent(prices, window)
        else:
            hurst = data.get("hurst_exponent", 0.5)
        
        # ADX Regime
        ohlcv = data.get("ohlcv", {})
        adx_regime, adx, vol_pct = self.compute_adx_regime(ohlcv)
        
        # Determine regime from Hurst and ADX
        if hurst < 0.4 or (hurst < 0.5 and adx_regime == "MEAN_REVERT"):
            regime = "mean_reverting"
            score = self.config["regime_scores"]["mean_reverting"]
            is_valid = True
        elif hurst > 0.6 or adx_regime in ["TRENDING", "BREAKOUT"]:
            regime = "trending"
            score = self.config["regime_scores"]["trending"]
            is_valid = False
        else:
            regime = "random_walk"
            score = self.config["regime_scores"]["random_walk"]
            is_valid = adx_regime in ["MEAN_REVERT", "TRANSITIONAL"]
        
        # ADX confirmation - mean reversion valid in compression
        if adx < 25 and vol_pct < 35:
            if regime != "trending":
                is_valid = True
                score = max(score, 1.0)
        
        return RegimeResult(
            score=score,
            regime=regime,
            hurst=hurst,
            adx=adx,
            vol_percentile=vol_pct,
            is_valid_for_mr=is_valid,
            regime_classification=adx_regime
        )
