"""
Intraday Hurst Diagnostic (V3.0 Verification)
Tests if the 1-Hour timeframe provides the 'Anti-Persistence' (H < 0.45) 
that the Daily timeframe is currently lacking.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_hurst(price_series, window=100):
    def hurst_rs(ts):
        n = len(ts)
        if n < 30: return 0.5
        returns = np.diff(np.log(ts))
        n_ret = len(returns)
        mean = np.mean(returns)
        deviation = np.cumsum(returns - mean)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(returns)
        if s == 0: return 0.5
        return np.log(r/s) / np.log(n_ret)
    return price_series.rolling(window).apply(hurst_rs, raw=True)

def run_intraday_diagnostic():
    symbols = ['SPY', 'NVDA', 'AMD', 'TSLA']
    # Last 6 months of Intraday data
    logger.info("Downloading Intraday 1-Hour data for last 2 years...")
    
    results = {}
    
    for s in symbols:
        # yfinance limited to 730 days of 1h data
        df = yf.download(s, period="2y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        df['hurst'] = calculate_hurst(df['close'], window=100)
        
        # Latest Snapshot (March-April 2026)
        recent_h = df['hurst'].tail(500)
        mean_h = recent_h.mean()
        min_h = recent_h.min()
        pct_reverting = (recent_h < 0.45).mean() * 100
        
        logger.info(f"--- {s} INTRADAY RESULTS ---")
        logger.info(f"Avg H (1h): {mean_h:.3f}")
        logger.info(f"Min H (1h): {min_h:.3f}")
        logger.info(f"% Reverting (< 0.45): {pct_reverting:.1f}%")
        
        results[s] = pct_reverting

    if any(pct > 15 for pct in results.values()):
        logger.info("========================================")
        logger.info("INTRADAY DIAGNOSTIC POSITIVE: Anti-persistence found in Hourly bars.")
        logger.info("Proceeding to V3.0 Intraday Rebuild.")
    else:
        logger.error("INTRADAY DIAGNOSTIC NEGATIVE: Even hourly bars are noise/trending.")

if __name__ == "__main__":
    run_intraday_diagnostic()
