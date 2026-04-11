"""
V3.1 Simple Truth Audit - Pure Hurst-Z Strategy
Tests the hypothesis that $|Z| > 2.0$ + $H < 0.45$ is the core alpha.
No XGBoost. No overfitting. Just structural snaps.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from training.engine_v3 import calculate_hurst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simple_truth_audit():
    symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'AMZN', 'AAPL', 'GLD', 'SLV']
    
    total_trades = 0
    total_hits = 0
    total_pnl_r = 0

    logger.info("Starting Pure Hurst-Z Audit (2025-2026)...")

    for s in symbols:
        # Full 2 years of 1h data
        df = yf.download(s, period="2y", interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # 1. Structural Engines
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        zscore = (df['close'] - mean_20) / std_20
        h = calculate_hurst(df['close'], window=100)
        atr = (df['high'] - df['low']).rolling(14).mean()
        
        # OOS Window
        df_oos = df[df.index >= '2025-01-01']
        
        for i in range(len(df_oos)-9):
            idx = df_oos.index[i]
            if pd.isna(h.loc[idx]) or pd.isna(zscore.loc[idx]): continue
            
            # THE SIMPLE TRUTH SIGNAL
            # 1. Abs Z-Score > 2.0 (Distance)
            # 2. Hurst < 0.45 (Anti-persistence)
            if abs(zscore.loc[idx]) > 2.0 and h.loc[idx] < 0.45:
                
                entry_p = df_oos['open'].iloc[i+1] # T+1 Open
                is_long = zscore.loc[idx] < -2.0 # Revert UP if below mean
                
                # Risk/Reward
                stop_dist = 1.5 * atr.loc[idx]
                target_p = mean_20.loc[idx]
                stop_p = (entry_p - stop_dist) if is_long else (entry_p + stop_dist)
                
                # Rule: Minimal RR check
                if abs(target_p - entry_p) / stop_dist < 1.0: continue
                
                # Check outcome (8 hour window)
                fwd = df_oos.iloc[i+2 : i+10]
                success = False
                exit_p = fwd['close'].iloc[-1]
                
                for _, bar in fwd.iterrows():
                    if is_long:
                        if bar['low'] <= stop_p:
                            exit_p = stop_p
                            break
                        if bar['high'] >= target_p:
                            success = True
                            exit_p = target_p
                            break
                    else:
                        if bar['high'] >= stop_p:
                            exit_p = stop_p
                            break
                        if bar['low'] <= target_p:
                            success = True
                            exit_p = target_p
                            break
                
                total_trades += 1
                if success: total_hits += 1
                
                # Just use R-multiple for audit

                r_multiple = (exit_p - entry_p) / stop_dist if is_long else (entry_p - exit_p) / stop_dist
                total_pnl_r += r_multiple

    if total_trades > 0:
        logger.info("========================================")
        logger.info("V3.1 SIMPLE TRUTH AUDIT (2025-2026)")
        logger.info(f"Total Signals: {total_trades}")
        logger.info(f"Win Rate: {total_hits/total_trades:.1%}")
        logger.info(f"Avg R-Multiple: {total_pnl_r/total_trades:.2f}R")
        logger.info(f"Total Expectancy: {total_pnl_r:.2f}R")
        logger.info("========================================")

if __name__ == "__main__":
    run_simple_truth_audit()
