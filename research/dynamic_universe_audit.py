"""
V4.4 Dynamic Universe Audit - The Rule of Persistence
Tests the hypothesis: Does a pre-entry Hurst filter (H > 0.52) 
select for superior forward performance?
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from training.engine_v4 import calculate_hurst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dynamic_universe_audit():
    symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'MSFT', 'AAPL', 'GOOGL', 'META']
    
    # Analyze the full 2025-2026 period
    start_date = "2024-11-01" # Buffer for Hurst
    end_date = "2026-04-10"
    
    all_signals = []
    
    logger.info("Initializing Dynamic Regime Audit (2025-2026)...")

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date, interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # 1. THE RULE (Calculate Hurst on 90-bar window)
        h_series = calculate_hurst(df['close'], window=90)
        
        # 2. IDENTIFY POTENTIAL ENTRIES (Using $|Z| > 2.0$ as a proxy for 'setup')
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        zscore = (df['close'] - mean_20) / std_20
        atr = (df['high'] - df['low']).rolling(14).mean()
        
        test_df = df[df.index >= '2025-01-01'].copy()
        
        for i in range(len(test_df)-15):
            idx = test_df.index[i]
            if pd.isna(h_series.loc[idx]): continue
            
            # Setup: Price is at an extreme (Potential momentum initiation)
            if abs(zscore.loc[idx]) > 2.0:
                h_at_t = h_series.loc[idx]
                is_eligible = h_at_t > 0.52 # THE RULE
                
                # Check outcome (1.5R target vs 1.5R stop)
                entry_p = test_df['open'].iloc[i+1]
                is_long = zscore.loc[idx] > 0
                a = atr.loc[idx]
                stop_dist = 1.5 * a
                stop_p = entry_p - stop_dist if is_long else entry_p + stop_dist
                target_p = entry_p + stop_dist if is_long else entry_p - stop_dist
                
                fwd = test_df.iloc[i+2 : i+17]
                win = 0
                for _, bar in fwd.iterrows():
                    if is_long:
                        if bar['low'] <= stop_p: break
                        if bar['high'] >= target_p:
                            win = 1
                            break
                    else:
                        if bar['high'] >= stop_p: break
                        if bar['low'] <= target_p:
                            win = 1
                            break
                
                all_signals.append({
                    'symbol': s,
                    'date': idx,
                    'hurst': h_at_t,
                    'is_eligible': is_eligible,
                    'win': win
                })

    res_df = pd.DataFrame(all_signals)
    if not res_df.empty:
        eligible_group = res_df[res_df['is_eligible'] == True]
        vetoed_group = res_df[res_df['is_eligible'] == False]
        
        logger.info("========================================")
        logger.info("V4.4 DYNAMIC UNIVERSE AUDIT RESULTS")
        logger.info("========================================")
        logger.info(f"ELIGIBLE (H > 0.52): {len(eligible_group)} trades | Win Rate: {eligible_group['win'].mean():.1%}")
        logger.info(f"VETOED   (H < 0.52): {len(vetoed_group)} trades | Win Rate: {vetoed_group['win'].mean():.1%}")
        logger.info("-" * 40)
        
        diff = eligible_group['win'].mean() - vetoed_group['win'].mean()
        logger.info(f"PERFORMANCE LIFT: {diff:+.1%}")
        
        if diff > 0.05:
            logger.info("VERDICT: THE DYNAMIC REGIME RULE IS SCIENTIFICALLY VALID.")
        else:
            logger.info("VERDICT: RULE LACKS SIGNIFICANT LIFT. RE-EVALUATE REGIME SIGNATURE.")
        logger.info("========================================")

if __name__ == "__main__":
    run_dynamic_universe_audit()
