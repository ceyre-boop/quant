"""
V3.0 Conviction Audit
Checks if the 'Noisy' 0.52 AUC model has high-probability pockets.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from training.engine_v3 import build_v3_features, build_v3_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_conviction_audit():
    # 1. LOAD V3.0
    with open('training/xgb_model_v3.pkl', 'rb') as f:
        payload = pickle.load(f)
        model = payload['model']
        feature_cols = payload['features']

    # 2. EVALUATE ON RECENT NVDA/SPY (OOS)
    symbols = ['NVDA', 'SPY']
    all_res = []

    for s in symbols:
        df = yf.download(s, period="1y", interval="1h")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        f = build_v3_features(df)
        l = build_v3_labels(df)
        
        res = pd.concat([f, l.rename('target')], axis=1).dropna()
        res = res[res['hurst'] < 0.45] # Apply H-Gate
        
        if len(res) == 0: continue
        
        res['prob'] = model.predict_proba(res[feature_cols])[:, 1]
        all_res.append(res)

    full_res = pd.concat(all_res)
    
    logger.info("========================================")
    logger.info("V3.0 INTRADAY CONVICTION BUCKETS")
    logger.info("========================================")
    
    buckets = [0.4, 0.45, 0.5, 0.55, 0.6]
    for b in buckets:
        mask = full_res['prob'] > b
        if mask.any():
            rate = full_res.loc[mask, 'target'].mean()
            count = mask.sum()
            logger.info(f"Prob > {b:.2f}: {count:3} trades | Actual Success: {rate:.1%}")
    logger.info("========================================")

if __name__ == "__main__":
    run_conviction_audit()
