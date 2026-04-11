"""
XGBoost Mean Reversion Rebuild V2.2 (Institutional Protocol)
Implements Root Cause 3: Walk-Forward Expanding Window Validation.
Locks Fold 5 (2025) for Final Honest Evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import logging
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from training.mean_reversion_engine import build_mean_reversion_label, build_mean_reversion_features, calculate_atr
from data.alpaca_client import AlpacaDataClient
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rebuild():
    client = AlpacaDataClient()
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    # 1. DOWNLOAD RAW DATA (2022 - 2026) via yfinance (No historical limit)
    logger.info("Downloading historical data (2022-2026) via yfinance...")
    raw_data = {}
    for s in symbols:
        df = yf.download(s, start='2021-01-01', interval='1d')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        raw_data[s] = df

    
    # 2. PULL VIX VIA YFINANCE
    logger.info("Pulling VIX via yfinance...")
    vix_df = yf.download('^VIX', start='2021-01-01', interval='1d')
    # Flatten multi-index if necessary
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_df.columns = [c.lower() for c in vix_df.columns]

    
    # 3. FEATURE & LABEL GENERATION
    logger.info("Generating V2.2 Features and Surgical Labels...")
    all_dfs = []
    for symbol, df in raw_data.items():
        if len(df) < 200: continue
        
        # Features (4-Dimensions)
        features = build_mean_reversion_features(df, vix_df)
        
        # Labels (Surgical)
        labels = build_mean_reversion_label(df)
        
        # Combine
        df_combined = features.copy()
        df_combined['target'] = labels
        df_combined['date'] = df.index
        df_combined['symbol'] = symbol
        
        # RULE: GAP ENTROPY FILTER
        # If gap > 0.5 ATR, signal is evaporated/dead.
        df_combined = df_combined[df_combined['gap_entropy'] < 0.5]
        
        all_dfs.append(df_combined.dropna())

        
    full_df = pd.concat(all_dfs).sort_values('date')
    
    # 4. WALK-FORWARD ARCHITECTURE (Expanding Window)
    # Fold 1: Train Jan 2022 – Dec 2023 │ Test Jan 2024 – Mar 2024
    folds = [
        ('2022-01-01', '2023-12-31', '2024-01-01', '2024-03-31'),
        ('2022-01-01', '2024-03-31', '2024-04-01', '2024-06-30'),
        ('2022-01-01', '2024-06-30', '2024-07-01', '2024-09-30'),
        ('2022-01-01', '2024-09-30', '2024-10-01', '2024-12-31'),
        ('2022-01-01', '2024-12-31', '2025-01-01', '2025-12-31') # FOLD 5: LOCKED
    ]
    
    feature_cols = [c for c in full_df.columns if c not in ['target', 'date', 'symbol']]
    fold_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        if i == 4:
            logger.info("FOLD 5 DETECTED: LOCKING 2025 DATA UNTIL FINAL VALIDATION.")
            continue
            
        train_df = full_df[(full_df['date'] >= train_start) & (full_df['date'] <= train_end)]
        test_df = full_df[(full_df['date'] >= test_start) & (full_df['date'] <= test_end)]
        
        if len(train_df) < 50 or len(test_df) < 10:
            logger.warning(f"Fold {i+1} insufficient samples. Skipping.")
            continue
            
        logger.info(f"Processing Fold {i+1}... Train: {len(train_df)} | Test: {len(test_df)}")
        
        X_train, y_train = train_df[feature_cols], train_df['target']
        X_test, y_test = test_df[feature_cols], test_df['target']
        
        if len(y_train.unique()) < 2:
            logger.warning(f"Fold {i+1} single class. Skipping.")
            continue

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            eval_metric='auc',
            early_stopping_rounds=30,
            random_state=42
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        fold_results.append({'fold': i+1, 'auc': auc, 'acc': acc})
        logger.info(f"Fold {i+1} Result: AUC {auc:.4f} | ACC {acc:.4f}")

        
    logger.info("Walk-Forward complete. Validation AUC: {:.4f}".format(np.mean([r['auc'] for r in fold_results])))

    # 5. SAVE PRODUCTION MODEL (Folds 1-4 Training)
    logger.info("Training Production Model on full Folds 1-4 data...")
    final_df = full_df[full_df['date'] <= '2024-12-31'].dropna(subset=['target'])
    X_final = final_df[feature_cols]
    y_final = final_df['target']
    
    prod_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric='auc',
        random_state=42
    )
    prod_model.fit(X_final, y_final)
    
    import pickle
    with open('training/xgb_model.pkl', 'wb') as f:
        pickle.dump({'model': prod_model, 'features': feature_cols}, f)
    logger.info("V2.2 Production Model Saved to training/xgb_model.pkl")

if __name__ == "__main__":
    run_rebuild()

