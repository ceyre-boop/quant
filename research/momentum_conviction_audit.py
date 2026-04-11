"""
V4.0 Momentum Conviction Audit
Searching for high-probability continuation pockets in the H > 0.52 regime.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from training.engine_v4 import build_v4_features, build_v4_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_momentum_conviction_audit():
    # 1. LOAD V4.0
    with open('training/xgb_model_v4.pkl', 'rb') as f:
        payload = pickle.load(f)
        model = payload['model']
        feature_cols = payload['features']

    # 2. EVALUATE ON VIRGIN OOS (AAPL, AMZN, SPY)
    symbols = ['AAPL', 'AMZN', 'SPY']
    all_res = []

    for s in symbols:
        df = yf.download(s, period="1y", interval="1h")
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        f = build_v4_features(df)
        l = build_v4_labels(df)
        
        res = pd.concat([f, l.rename('target')], axis=1).dropna()
        res = res[res['hurst'] > 0.52] # Apply Momentum Gate
        
        if len(res) == 0: continue
        
        res['prob'] = model.predict_proba(res[feature_cols])[:, 1]
        all_res.append(res)

    full_res = pd.concat(all_res)
    
    logger.info("========================================")
    logger.info("V4.0 MOMENTUM CONVICTION BUCKETS")
    logger.info("========================================")
    
    buckets = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    for b in buckets:
        mask = full_res['prob'] > b
        if mask.any():
            rate = full_res.loc[mask, 'target'].mean()
            count = mask.sum()
            logger.info(f"Prob > {b:.2f}: {count:3} trades | Actual Success: {rate:.1%}")
    logger.info("========================================")

if __name__ == "__main__":
    run_momentum_conviction_audit()
