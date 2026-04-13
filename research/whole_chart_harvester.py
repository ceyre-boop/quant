"""
V6.2 Planetary Harvester (The Whole-Chart Account)
Harvests 10 years of Daily data to map Fractal Corridors and Biological Anomalies.
Architecture: Decadal Memory for Regime Identification.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_planetary_corridors(price_series, window_years=2):
    """
    Identifies the Logarithmic Growth Corridor across a multi-year window.
    High R-squared = The 'Era' is being defined by a structural geometric constant.
    """
    window = window_years * 252 # Trading days
    def _get_era_slope(ts):
        if len(ts) < 100: return 0
        t = np.arange(len(ts))
        log_p = np.log(ts)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_p)
        return slope # The 'Expansion Constant' of the era
    
    return price_series.rolling(window).apply(_get_era_slope, raw=True)

def run_planetary_harvest():
    # 1. THE PLANETARY UNIVERSE
    # Broad sentinels (SPY, QQQ) + High-Beta Leaders (NVDA, TSLA, MSFT)
    symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'MSFT', 'AAPL', 'AMD']
    
    logger.info("============================================")
    logger.info("V6.2 PLANETARY HARVEST (10-YEAR DAILY)")
    logger.info("============================================")
    
    all_data = []

    for s in symbols:
        logger.info(f"Harvesting Era Data for {s}...")
        df = yf.download(s, period="10y", interval="1d")
        if df.empty: continue
        
        # Flatten columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # 2. CALCULATE ERA-METRICS
        # The 'Expansion Constant': How fast the company is structurally growing.
        df['era_slope'] = map_planetary_corridors(df['close'], window_years=2)
        
        # The 'Geometric Integrity': How well it respects its own corridor.
        df['era_integrity'] = df['close'].rolling(504).apply(lambda x: stats.linregress(np.arange(len(x)), np.log(x))[2]**2, raw=True)
        
        # 3. BIOLOGICAL ANOMALY DETECTION (Astrophage)
        # 10-year rolling volatility Z-score
        ann_vol = np.log(df['close'] / df['close'].shift(1)).rolling(252).std() * np.sqrt(252)
        df['vol_zscore'] = (ann_vol - ann_vol.rolling(2520, min_periods=252).mean()) / ann_vol.rolling(2520, min_periods=252).std()
        
        df['ticker'] = s
        all_data.append(df[['ticker', 'close', 'era_slope', 'era_integrity', 'vol_zscore']])

    # 4. AGGREGATE THE PLANETARY TRUTH
    planetary_df = pd.concat(all_data)
    os.makedirs("data/planetary", exist_ok=True)
    planetary_df.to_csv("data/planetary/whole_chart_corridors.csv")
    
    logger.info("============================================")
    logger.info("PLANETARY HARVEST COMPLETE")
    logger.info(f"Total Era-Samples: {len(planetary_df)}")
    
    # Identify the 'Comet Alignment'
    # High era_integrity (> 0.90) + High era_slope
    comet_alignment = planetary_df[planetary_df['era_integrity'] > 0.95].tail(5)
    if not comet_alignment.empty:
        logger.info("Recent 'Comet Alignments' (Perfect Structural Corridors):")
        print(comet_alignment[['ticker', 'era_integrity', 'era_slope']])
    
    logger.info("============================================")

if __name__ == "__main__":
    run_planetary_harvest()
