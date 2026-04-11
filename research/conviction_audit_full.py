"""
V4.5 Model-Confidence Audit - The Rule of Conviction
Tests if XGBoost Prob > 0.60 is the only valid dynamic universe selector.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from training.engine_v4 import build_v4_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_conviction_audit():
    # 1. LOAD MODEL V4.1
    with open('training/xgb_model_v4.pkl', 'rb') as f:
        payload = pickle.load(f)
        model = payload['model']
        feature_cols = payload['features']

    symbols = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'MSFT', 'AAPL', 'GOOGL', 'META']
    start_date = "2024-11-01"
    end_date = "2026-04-10"
    
    all_signals = []
    
    logger.info("Initializing Model Conviction Audit (2025-2026)...")

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date, interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Features
        f_df = build_v4_features(df)
        atr = (df['high'] - df['low']).rolling(14).mean()
        
        test_df = df[df.index >= '2025-01-01'].copy()
        
        # Predict all bars
        probs = model.predict_proba(f_df.loc[test_df.index, feature_cols])[:, 1]
        
        for i in range(len(test_df)-15):
            idx = test_df.index[i]
            prob = probs[i]
            
            # Setup: Probability > 0.50 (Base line)
            # We compare High Confidence (>=0.60) vs Low Confidence (<0.60)
            is_high_conf = prob >= 0.60
            
            # Outcome (1:1 RR)
            entry_p = test_df['open'].iloc[i+1]
            zscore = (df['close'].loc[idx] - df['close'].rolling(20).mean().loc[idx]) / df['close'].rolling(20).std().loc[idx]
            is_long = zscore > 0
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
                'prob': prob,
                'is_high_conf': is_high_conf,
                'win': win
            })

    res_df = pd.DataFrame(all_signals)
    if not res_df.empty:
        high_group = res_df[res_df['is_high_conf'] == True]
        low_group = res_df[res_df['is_high_conf'] == False]
        
        logger.info("========================================")
        logger.info("V4.5 MODEL CONVICTION AUDIT RESULTS")
        logger.info("========================================")
        logger.info(f"HIGH CONF (>= 0.60): {len(high_group)} trades | Win Rate: {high_group['win'].mean():.1%}")
        logger.info(f"LOW CONF  (< 0.60): {len(low_group)} trades | Win Rate: {low_group['win'].mean():.1%}")
        
        diff = high_group['win'].mean() - low_group['win'].mean()
        logger.info(f"CONVICTION LIFT: {diff:+.1%}")
        logger.info("========================================")

if __name__ == "__main__":
    run_conviction_audit()
