"""
V4.2 Virgin Territory Backtest (Jan - Apr 2026)
Institutional Validation of the Momentum + CSD + Logistic Substrate.
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

def run_v4_2_backtest():
    # 1. LOAD MODEL V4.1/4.2
    with open('training/xgb_model_v4.pkl', 'rb') as f:
        payload = pickle.load(f)
        model = payload['model']
        feature_cols = payload['features']

    symbols = ['NVDA', 'TSLA', 'AMD', 'AMZN', 'MSFT', 'AAPL', 'GOOGL', 'META']

    results = []
    
    detector = CriticalSlowingDetector(window=60)
    
    logger.info("Starting V4.2 Institutional Backtest (Jan-Apr 2026)...")

    for s in symbols:
        df = yf.download(s, start="2025-11-01", end="2026-04-10", interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Features and CSD
        f = build_v4_features(df)
        csd_res = detector.compute(df['close'])
        
        # Filter for the 'Virgin' OOS Period
        test_mask = df.index >= '2026-01-01'
        df_oos = df[test_mask]
        
        for i in range(len(df_oos)-15):
            idx = df_oos.index[i]
            
            # --- V4.2 ENTRY RULES ---
            current_f = f.loc[idx]
            current_csd = csd_res.loc[idx, 'csd_score']
            
            # 1. Hurst Trigger (Persistence)
            if current_f['hurst'] <= 0.52: continue
            
            # 2. CSD Veto (Stability)
            if current_csd > 0.65: continue
            
            # 3. Model Prob (Conviction)
            prob = model.predict_proba(f.loc[[idx], feature_cols])[0, 1]
            if prob < 0.60: continue
            
            # --- TRADE INITIALIZATION ---
            entry_p = df_oos['open'].iloc[i+1] # T+1 Open
            is_long = current_f['zscore_20'] > 0 # Follow the current Z-deviation
            
            a = (df['high'] - df['low']).rolling(14).mean().loc[idx]
            stop_dist = 1.5 * a
            stop_p = entry_p - stop_dist if is_long else entry_p + stop_dist
            
            # Target: Mirror the Stop (1:1 R-Multiple)
            target_p = entry_p + stop_dist if is_long else entry_p - stop_dist

            
            # --- MONITORING LOOP (15 hour max) ---
            fwd = df_oos.iloc[i+2 : i+17]
            outcome = 'TIME'
            exit_p = fwd['close'].iloc[-1]
            
            for f_idx, bar in fwd.iterrows():
                # Check PREDICTIVE EXIT (CSD)
                if csd_res.loc[f_idx, 'csd_score'] > 0.65:
                    outcome = 'CSD_EXIT'
                    exit_p = bar['open'] # Exit at next open following CSD spike
                    break
                    
                # Check STOP/TARGET
                if is_long:
                    if bar['low'] <= stop_p:
                        outcome = 'STOP'
                        exit_p = stop_p
                        break
                    if bar['high'] >= target_p:
                        outcome = 'TARGET'
                        exit_p = target_p
                        break
                else:
                    if bar['high'] >= stop_p:
                        outcome = 'STOP'
                        exit_p = stop_p
                        break
                    if bar['low'] <= target_p:
                        outcome = 'TARGET'
                        exit_p = target_p
                        break
            
            r_multiple = (exit_p - entry_p) / stop_dist if is_long else (entry_p - exit_p) / stop_dist
            
            results.append({
                'ticker': s,
                'date': idx,
                'prob': prob,
                'outcome': outcome,
                'r_multiple': r_multiple
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        win_rate = (res_df['r_multiple'] > 0).mean()
        logger.info("========================================")
        logger.info("V4.2 VIRGIN BACKTEST RESULTS (2026)")
        logger.info(f"Total Trades: {len(res_df)}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Average R: {res_df['r_multiple'].mean():.2f}R")
        logger.info(f"Profit Factor: {abs(res_df[res_df['r_multiple'] > 0]['r_multiple'].sum() / res_df[res_df['r_multiple'] < 0]['r_multiple'].sum()):.2f}")
        logger.info("========================================")
        
        print("\nOutcome Breakdown:")
        print(res_df['outcome'].value_counts())
        
        print("\nTicker Performance:")
        print(res_df.groupby('ticker')['r_multiple'].mean())

if __name__ == "__main__":
    run_v4_2_backtest()
