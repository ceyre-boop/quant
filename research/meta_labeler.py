"""
V5.0 Meta-Labeler: Generating the Specialist Performance Matrix
Fuses Momentum (V4.8) and Reversion (V2.2) results for Router Training.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
import os
from training.engine_v3 import build_v3_features
from training.engine_v4 import build_v4_features
from research.critical_slowing_detector import CriticalSlowingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_meta_labeling():
    symbols = ['NVDA', 'AMZN', 'TSLA', 'AAPL', 'MSFT', 'AMD']
    
    with open('training/xgb_model_v4_8.pkl', 'rb') as f_in:
        mom_payload = pickle.load(f_in)
        mom_model = mom_payload['model']
        mom_features = mom_payload['features']
        
    # Harvest IV Data for V4.8 Model
    vix = yf.download("^VIX", period="2y", interval="1d")
    vix9d = yf.download("^VIX9D", period="2y", interval="1d")
    vix.columns = [c.lower() for c in (vix.columns.get_level_values(0) if isinstance(vix.columns, pd.MultiIndex) else vix.columns)]
    vix9d.columns = [c.lower() for c in (vix9d.columns.get_level_values(0) if isinstance(vix9d.columns, pd.MultiIndex) else vix9d.columns)]
    iv_df = pd.DataFrame(index=vix.index)
    iv_df['iv_spread'] = vix9d['close'] / vix['close']
    iv_df.index = iv_df.index.normalize()

    detector = CriticalSlowingDetector(window=60)
    all_rows = []

    for s in symbols:
        try:
            df = yf.download(s, start="2024-05-01", end="2026-04-10", interval="1h")
            if df.empty: continue
            df.columns = [c.lower() for c in (df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns)]
            
            # 1. INTEGRATED FEATURES (V4.8 Requirements)
            f_mom = build_v4_features(df)
            csd_res = detector.compute(df['close'])
            f_mom['csd_score'] = csd_res['csd_score']
            # Expectation Indicator (IV Spread - Daily aligned to Hourly)
            f_mom['iv_spread'] = pd.Series(df.index.normalize()).map(iv_df['iv_spread']).values
            f_mom['iv_spread'] = f_mom['iv_spread'].ffill().fillna(1.0)

            
            # Specialist B features
            f_rev = build_v3_features(df)
            
            # Calculate Mom Probs
            mom_probs = mom_model.predict_proba(f_mom[mom_features])[:, 1]
            m_df = pd.DataFrame({'mom_prob': mom_probs}, index=f_mom.index)
            
            # 2. ALIGNMENT
            active_df = df.join(m_df, how='inner')
            active_df = active_df.join(f_mom[['zscore_20', 'hurst', 'adx', 'csd_score', 'iv_spread']], how='inner')
            active_df = active_df.join(f_rev[['zscore_20', 'hurst']], how='inner', rsuffix='_rev')
            
            # 3. LABELING LOOP
            for i in range(len(active_df)-15):
                idx = active_df.index[i]
                fwd_idx = active_df.index[min(i+15, len(active_df)-1)]
                fwd_ret = (df.loc[fwd_idx, 'close'] - df.loc[idx, 'close']) / df.loc[idx, 'close']
                
                # MOMENTUM EV
                mom_ev = 0
                if active_df.loc[idx, 'mom_prob'] > 0.60:
                    is_long = active_df.loc[idx, 'zscore_20'] > 0
                    mom_ev = fwd_ret if is_long else -fwd_ret
                
                # REVERSION EV
                rev_ev = 0
                if abs(active_df.loc[idx, 'zscore_20_rev']) > 2.0 and active_df.loc[idx, 'hurst_rev'] < 0.45:
                    is_long = active_df.loc[idx, 'zscore_20_rev'] < 0
                    rev_ev = fwd_ret if is_long else -fwd_ret
                    
                label = 0
                if mom_ev > 0.005 and mom_ev > rev_ev:
                    label = 1
                elif rev_ev > 0.005 and rev_ev > mom_ev:
                    label = 2
                    
                all_rows.append({
                    'timestamp': idx,
                    'ticker': s,
                    'label': label,
                    'hurst': active_df.loc[idx, 'hurst'],
                    'zscore': active_df.loc[idx, 'zscore_20'],
                    'csd': active_df.loc[idx, 'csd_score'],
                    'iv_spread': active_df.loc[idx, 'iv_spread'],
                    'adx': active_df.loc[idx, 'adx'],
                    'volatility': df['close'].rolling(20).std().loc[idx] / df['close'].loc[idx]
                })

        except Exception as e:
            logger.error(f"Ticker {s} failed: {e}")

    final_df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    final_df.to_csv("data/router_labels_v5.csv", index=False)
    
    logger.info("========================================")
    logger.info("V5.0 SPECIALIST MAPPING COMPLETE")
    logger.info(f"Total Labeled Bars: {len(final_df)}")
    print(final_df['label'].value_counts(normalize=True))
    logger.info("========================================")

if __name__ == "__main__":
    run_meta_labeling()
