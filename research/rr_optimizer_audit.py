"""
V4.6 R:R Optimizer Audit - The Expectancy Game
Focus: Converting a 50.6% Information Edge into Institutional Expectancy 
through Convexity and Dynamic Exit Management.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from training.engine_v4 import build_v4_features
from research.critical_slowing_detector import CriticalSlowingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rr_optimization():
    # 1. LOAD THE BASE REALITY MODEL (V4.1/4.2)
    with open('training/xgb_model_v4.pkl', 'rb') as f:
        payload = pickle.load(f)
        model = payload['model']
        feature_cols = payload['features']

    symbols = ['NVDA', 'AMZN', 'TSLA', 'AAPL', 'MSFT', 'AMD']
    results = []
    
    detector = CriticalSlowingDetector(window=60)
    
    logger.info("Starting V4.6 R:R Optimization Audit (Path A Execution)...")

    for s in symbols:
        df = yf.download(s, start="2025-01-01", end="2026-04-10", interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        f = build_v4_features(df)
        csd_res = detector.compute(df['close'])
        atr = (df['high'] - df['low']).rolling(14).mean()
        
        # OOS Window
        test_df = df[df.index >= '2025-01-01']
        probs = model.predict_proba(f.loc[test_df.index, feature_cols])[:, 1]
        
        for i in range(len(test_df)-20):
            idx = test_df.index[i]
            
            # --- V4.1/4.5 BASE ENTRY (THE 2.3% EDGE) ---
            if probs[i] < 0.60: continue
            
            # TRADE INITIALIZATION (Path A - High Convexity Settings)
            entry_p = test_df['open'].iloc[i+1]
            zscore = (df['close'].loc[idx] - df['close'].rolling(20).mean().loc[idx]) / df['close'].rolling(20).std().loc[idx]
            is_long = zscore > 0
            
            a = atr.loc[idx]
            stop_dist = 2.0 * a # RELAXED STOP - Give it room to work
            initial_stop = entry_p - stop_dist if is_long else entry_p + stop_dist
            
            # NO FIXED TARGET - We are letting it run!
            
            # --- EXECUTION MONITORING ---
            fwd = test_df.iloc[i+2 : i+22] # Up to 20 hours
            exit_p = fwd['close'].iloc[-1]
            outcome = 'TIME'
            
            for f_idx, bar in fwd.iterrows():
                current_h = f.loc[f_idx, 'hurst']
                current_csd = csd_res.loc[f_idx, 'csd_score']
                
                # RULE 1: STRUCTURAL CSD EXIT (The Pull-Cord)
                if current_csd > 0.65:
                    outcome = 'CSD_EXIT'
                    exit_p = bar['open']
                    break
                
                # RULE 2: REGIME DECAY EXIT (Hurst falls < 0.48)
                if current_h < 0.48:
                    outcome = 'REGIME_DECAY'
                    exit_p = bar['open']
                    break
                
                # RULE 3: THE HARD STOP (Final safety)
                if is_long:
                    if bar['low'] <= initial_stop:
                        outcome = 'HARD_STOP'
                        exit_p = initial_stop
                        break
                else:
                    if bar['high'] >= initial_stop:
                        outcome = 'HARD_STOP'
                        exit_p = initial_stop
                        break
            
            r_multiple = (exit_p - entry_p) / stop_dist if is_long else (entry_p - exit_p) / stop_dist
            
            results.append({
                'ticker': s,
                'r_multiple': r_multiple,
                'outcome': outcome
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        logger.info("========================================")
        logger.info("V4.6 EXECUTION OPTIMIZATION (PATH A)")
        logger.info("========================================")
        logger.info(f"Total Trades: {len(res_df)}")
        logger.info(f"Win Rate (>0R): {(res_df['r_multiple'] > 0).mean():.1%}")
        logger.info(f"Avg R-Multiple: {res_df['r_multiple'].mean():.2f}R")
        logger.info(f"Max Win (R): {res_df['r_multiple'].max():.2f}R")
        logger.info(f"Profit Factor: {abs(res_df[res_df['r_multiple'] > 0]['r_multiple'].sum() / res_df[res_df['r_multiple'] < 0]['r_multiple'].sum()):.2f}")
        logger.info("========================================")
        
        print("\nExit Reason Distribution:")
        print(res_df['outcome'].value_counts())

if __name__ == "__main__":
    run_rr_optimization()
