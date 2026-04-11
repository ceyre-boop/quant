"""
V4.2 CSD Falsification Test
Tests if Critical Slowing Down predicted the 2025 regime collapses.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from research.critical_slowing_detector import CriticalSlowingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_falsification():
    # 1. LOAD THE FAILURE TRUTH SET (V2.2 Ledger)
    ledger = pd.read_csv("data/paper_trading/ledger_v2_2.csv")
    failures = ledger[ledger['outcome'] == 'STOPPED'].copy()
    
    # 2. INITIALIZE DETECTOR
    detector = CriticalSlowingDetector(window=60)
    
    results = []
    
    # Symbols with the most failures
    symbols = failures['ticker'].unique()
    
    logger.info(f"Running Falsification Test on {len(failures)} historical failures...")

    for s in symbols:
        # Download historical 1h data for the target period
        df = yf.download(s, start="2025-01-01", end="2025-12-31", interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate CSD for the entire year
        csd_df = detector.compute(df['close'])
        csd_df.index = csd_df.index.tz_localize(None) # Remove TZ for matching
        
        sym_failures = failures[failures['ticker'] == s]
        
        for _, fail in sym_failures.iterrows():
            fail_date = pd.to_datetime(fail['date']).normalize()
            # Find the closest hourly matches for that day
            mask = (csd_df.index >= fail_date) & (csd_df.index < fail_date + pd.Timedelta(days=1))
            
            if mask.any():
                # Take the CSD reading at the END of that day (the failure signal day)
                target_window = csd_df[mask]
                avg_score = target_window['csd_score'].mean()
                max_score = target_window['csd_score'].max()
                
                results.append({
                    'symbol': s,
                    'date': fail_date,
                    'avg_csd': avg_score,
                    'max_csd': max_score,
                    'predicted': max_score > 0.65
                })

    res_df = pd.DataFrame(results)

    
    if not res_df.empty:
        logger.info("========================================")
        logger.info("V4.2 CSD FALSIFICATION RESULTS (2025)")
        logger.info(f"Total Failures Tested: {len(res_df)}")
        logger.info(f"True Positive Rate (CSD > 0.65): {res_df['predicted'].mean():.1%}")
        logger.info(f"Avg CSD Before Stop: {res_df['avg_csd'].mean():.3f}")
        logger.info("========================================")
        
        print("\nDetail of top failures:")
        print(res_df.sort_values('avg_csd', ascending=False).head(10))

if __name__ == "__main__":
    run_falsification()
