"""
V4.0 Momentum Diagnostic - Hurst Distribution Analysis
Pillar 12: Trend Persistence Validation

Determines the population size of the 'Trending Regime' (H > 0.55) in 
the 2025-2026 dataset.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from training.engine_v3 import calculate_hurst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_momentum_diagnostic():
    symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'AMZN', 'AAPL', 'GLD', 'SLV']
    start_date = "2024-10-01"
    end_date = "2026-04-10"
    
    total_samples = 0
    trending_samples = 0 # H > 0.55
    aggressive_trending = 0 # H > 0.60
    
    logger.info("Analyzing Hurst Distribution (2025-2026)...")

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date, interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        h = calculate_hurst(df['close'], window=100)
        
        # OOS 2025-2026
        h_oos = h[h.index >= '2025-01-01'].dropna()
        
        total = len(h_oos)
        trend = (h_oos > 0.55).sum()
        agg_trend = (h_oos > 0.60).sum()
        
        total_samples += total
        trending_samples += trend
        aggressive_trending += agg_trend
        
        logger.info(f"{s}: Total Hours: {total:4} | Trending (>0.55): {trend:4} ({(trend/total)*100:.1f}%)")

    logger.info("========================================")
    logger.info("V4.0 MOMENTUM DISTRIBUTION RESULTS")
    logger.info(f"Global Sample Size: {total_samples}")
    logger.info(f"Trending Population (>0.55): {trending_samples} ({(trending_samples/total_samples)*100:.1f}%)")
    logger.info(f"Hyper-Trending Pop (>0.60): {aggressive_trending} ({(aggressive_trending/total_samples)*100:.1f}%)")
    logger.info("========================================")

if __name__ == "__main__":
    run_momentum_diagnostic()
