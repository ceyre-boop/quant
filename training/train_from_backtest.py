"""
Train XGBoost from Backtest Signal CSV
Usage: python training/train_from_backtest.py --signals data/backtest_results/signals_YYYYMMDD_HHMMSS.csv
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from governance.policy_engine import GOVERNANCE
import argparse
import joblib
import sys
import os
from pathlib import Path

def train_from_csv(csv_path: str):
    print(f"Loading signals from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Prepare Target
    # We need a target. In the signals CSV, we might need to join with outcomes.
    # If the signals CSV has 'pnl' or 'win' from the backtest, we use that.
    # Note: signals_*.csv usually contains features + metadata.
    # trades_*.csv contains the outcome.
    
    # Actually, let's look for 'win' or 'pnl' in the dataframe.
    # If not present, we assume the user wants us to use the 'win' column from a matched trade.
    
    # For now, let's assume we are training on FeatureVector fields to predict 'direction' 
    # (This is just a test of the pipeline).
    # REAL TRAINING: We need Features (t=0) and Outcome (t+1).
    
    if 'win' not in df.columns:
        print("ERROR: CSV must contain 'win' column (True/False) for training.")
        print("Tip: Use the trades_*.csv or ensure backtest_lifecycle saves outcomes in signals.")
        return

    # 2. Identify Feature Columns
    # Exclude non-feature columns
    exclude = ['symbol', 'timestamp', 'date', 'win', 'pnl', 'entry_price', 'exit_price', 'target', 'direction']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32]]
    
    print(f"Training on {len(feature_cols)} features: {feature_cols[:5]}...")
    
    X = df[feature_cols].values
    y = df['win'].astype(int).values
    
    if len(X) < 50:
        print(f"ERROR: Not enough samples ({len(X)})")
        return

    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Pillar 1 & 6: Deterministic and Governed Training
        seed = GOVERNANCE.parameters.get('random_seed', 42)
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            random_state=seed
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        print(f"Fold {fold+1} Accuracy: {accuracy_score(y_val, y_pred):.3f}")

    # Final Train
    print("Training final model...")
    final_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    final_model.fit(X, y)
    
    # Save feature names for inference parity
    final_model.feature_names_ = feature_cols
    
    model_path = Path("training/xgb_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"SUCCESS: Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True, help="Path to backtest signals CSV")
    args = parser.parse_args()
    train_from_csv(args.signals)
