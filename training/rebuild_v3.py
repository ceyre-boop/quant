"""
V3.0 Rebuild - Hurst-Anchored Intraday Pipeline
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from training.engine_v3 import build_v3_features, build_v3_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_v3_rebuild():
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'AAPL', 'MSFT', 'AMZN', 'TSLA']
    
    logger.info("Downloading 2 years of 1-Hour Intraday data...")
    all_dfs = []
    
    for s in symbols:
        try:
            df = yf.download(s, period="2y", interval="1h")
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            # 1. Feature & Label Generation
            features = build_v3_features(df)
            labels = build_v3_labels(df)
            
            combined = pd.concat([features, labels.rename('target')], axis=1)
            combined['symbol'] = s
            combined['date'] = combined.index
            
            # 2. THE H-GATE FILTER
            # Only train/evaluate on bars where Hurst confirms mean reversion
            pre_filter_len = len(combined)
            combined = combined[combined['hurst'] < 0.45].dropna()
            
            logger.info(f"{s}: Filtered {pre_filter_len} -> {len(combined)} ({(len(combined)/pre_filter_len)*100:.1f}% tradeable)")
            all_dfs.append(combined)
        except Exception as e:
            logger.error(f"Failed {s}: {e}")

    full_df = pd.concat(all_dfs).sort_values('date')
    
    # 3. Features
    feature_cols = [c for c in full_df.columns if c not in ['target', 'symbol', 'date']]
    
    # 4. TIME-BASED SPLIT (70/30)
    split_idx = int(len(full_df) * 0.7)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    logger.info(f"Training V3.0 on {len(X_train)} instances...")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    logger.info("========================================")
    logger.info("V3.0 INTRADAY REBUILD RESULTS")
    logger.info(f"Validation AUC: {auc:.4f}")
    logger.info(f"Validation ACC: {acc:.4f}")
    logger.info("========================================")
    
    # Save V3.0
    with open('training/xgb_model_v3.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols, 'timeframe': '1h'}, f)
    logger.info("V3.0 Model Saved to training/xgb_model_v3.pkl")

if __name__ == "__main__":
    run_v3_rebuild()
