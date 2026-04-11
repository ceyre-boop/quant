"""
Probability Bucket Analysis (V2.2)
Pillar 9: Verification & Validation

Analyzes the actual reversion rate per probability bucket on Fold 1-4 data.
This empirical analysis determines the 'Institutional Floor' for trade deployment.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import yfinance as yf
from training.mean_reversion_engine import build_mean_reversion_features, build_mean_reversion_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bucket_analysis():
    # 1. LOAD MODEL
    try:
        with open('training/xgb_model.pkl', 'rb') as f:
            payload = pickle.load(f)
            model = payload['model']
            feature_cols = payload['features']
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. GATHER 2024 LABELED DATA
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    start_date = "2023-10-01"
    end_date = "2024-12-31"
    
    logger.info("Gathering 2024 labels and predictions...")
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix_df = pd.DataFrame({'close': vix.squeeze()}, index=vix.index)


    all_results = []

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        features = build_mean_reversion_features(df, vix_df)
        labels = build_mean_reversion_label(df) # Surgical Labels
        
        # OOS Window (2024)
        oos_mask = (df.index >= '2024-01-01')
        X = features.loc[oos_mask].dropna(subset=feature_cols)
        y = labels.loc[X.index]
        
        if len(X) == 0: continue
        
        probs = model.predict_proba(X[feature_cols])[:, 1]
        
        symbol_res = pd.DataFrame({'prob': probs, 'label': y}, index=X.index)
        all_results.append(symbol_res)

    results = pd.concat(all_results)
    
    # 3. BUCKET ANALYSIS
    buckets = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]
    
    logger.info("========================================")
    logger.info("PROBABILITY BUCKET ANALYSIS (2024 OOS)")
    logger.info("========================================")
    
    for low, high in buckets:
        mask = (results['prob'] >= low) & (results['prob'] < high)
        if mask.sum() == 0:
            continue
        
        actual_rate = results.loc[mask, 'label'].mean()
        count = mask.sum()
        logger.info(f"Prob {low:.2f}-{high:.2f}: {count:4} trades | Actual reversion: {actual_rate:.1%}")
    logger.info("========================================")

if __name__ == "__main__":
    run_bucket_analysis()
