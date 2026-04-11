"""
V4.0 Rebuild - The Momentum Pivot Pipeline
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from training.engine_v4 import build_v4_features, build_v4_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_v4_rebuild():
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'AAPL', 'MSFT', 'AMZN', 'TSLA']
    
    logger.info("Initializing V4.0 Momentum Rebuild (1-Hour Bars)...")
    all_dfs = []
    
    for s in symbols:
        try:
            df = yf.download(s, period="2y", interval="1h")
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            # 1. Feature & Label Generation (Momentum Logic)
            features = build_v4_features(df)
            labels = build_v4_labels(df)
            
            combined = pd.concat([features, labels.rename('target')], axis=1)
            combined['symbol'] = s
            combined['date'] = combined.index
            
            # 2. THE INVERTED H-GATE
            # Now we ONLY train on the Trending Regime (H > 0.52)
            pre_filter_len = len(combined)
            combined = combined[combined['hurst'] > 0.52].dropna()
            
            logger.info(f"{s}: Trend Window Captured: {len(combined)} ({(len(combined)/pre_filter_len)*100:.1f}%)")
            all_dfs.append(combined)
        except Exception as e:
            logger.error(f"Failed {s}: {e}")

    if not all_dfs:
        logger.error("No trending data found for rebuild.")
        return

    full_df = pd.concat(all_dfs).sort_values('date')
    
    # 3. Features
    feature_cols = [c for c in full_df.columns if c not in ['target', 'symbol', 'date']]
    
    # 4. TIME-BASED SPLIT (70/30)
    split_idx = int(len(full_df) * 0.7)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    logger.info(f"Training V4.0 Momentum on {len(X_train)} trending instances...")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]), # Handle imbalance
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    logger.info("========================================")
    logger.info("V4.0 MOMENTUM REBUILD RESULTS")
    logger.info(f"OOS AUC (Trending-Only): {auc:.4f}")
    logger.info(f"Win Rate (Model Preds): {acc:.1%}")
    logger.info("========================================")
    
    # Save V4.0
    with open('training/xgb_model_v4.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols, 'timeframe': '1h', 'type': 'momentum'}, f)
    logger.info("V4.0 Model Saved to training/xgb_model_v4.pkl")

if __name__ == "__main__":
    run_v4_rebuild()
