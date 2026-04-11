"""
Hurst Diagnostic (H-Gate Verification)
Pillar 11: Regime Stability Analysis

Verifies the hypothesis that a drift in Hurst Exponent (H) was the unseen 
driver behind the 2025-2026 Daily model collapse.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_hurst(price_series, window=100):
    """
    Hurst Exponent via R/S analysis.
    H < 0.5 = mean reverting 
    H > 0.5 = trending
    H = 0.5 = random walk (efficient)
    """
    def hurst_rs(ts):
        n = len(ts)
        if n < 30: # Higher requirement for R/S stability
            return 0.5
        
        # Calculate returns for stationarity
        returns = np.diff(np.log(ts))
        n_ret = len(returns)
        
        mean = np.mean(returns)
        deviation = np.cumsum(returns - mean)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(returns)
        
        if s == 0:
            return 0.5
            
        # Log(R/S) / Log(n)
        res = np.log(r/s) / np.log(n_ret)
        return res
    
    return price_series.rolling(window).apply(hurst_rs, raw=True)

def run_hurst_diagnostic():
    symbols = ['SPY', 'NVDA']
    # 2024 (Success era) vs 2025 (Failure era)
    start_date = "2023-01-01"
    end_date = "2026-04-10"
    
    results = {}
    
    for s in symbols:
        logger.info(f"Analyzing Hurst drift for {s}...")
        df = yf.download(s, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        df['hurst'] = calculate_hurst(df['close'], window=100) # ~5 months of daily data
        
        # Compare 2024 vs 2025
        h_2024 = df.loc['2024-01-01':'2024-12-31', 'hurst']
        h_2025 = df.loc['2025-01-01':'2026-04-10', 'hurst']
        
        results[s] = {
            'mean_2024': h_2024.mean(),
            'mean_2025': h_2025.mean(),
            'max_2025': h_2025.max(),
            'pct_above_50_2024': (h_2024 > 0.50).mean() * 100,
            'pct_above_50_2025': (h_2025 > 0.50).mean() * 100
        }
        
        logger.info(f"--- {s} HURST RESULTS ---")
        logger.info(f"Avg H (2024): {h_2024.mean():.3f} | % Trending: {(h_2024 > 0.50).mean()*100:.1f}%")
        logger.info(f"Avg H (2025): {h_2025.mean():.3f} | % Trending: {(h_2025 > 0.50).mean()*100:.1f}%")

    logger.info("========================================")
    if any(res['mean_2025'] > res['mean_2024'] for res in results.values()):
        logger.info("DIAGNOSTIC POSITIVE: Hurst drift detected in 2025.")
        logger.info("Proceeding to V3.0 Rebuild with H-Gate.")
    else:
        logger.warning("DIAGNOSTIC NEGATIVE: Hurst did not catch the shift.")

if __name__ == "__main__":
    run_hurst_diagnostic()
