"""
Backtest - Check if 57.8% accuracy translates to profit
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.feature_generator import FeatureGenerator
from xgboost import XGBClassifier


def backtest_signals():
    """Backtest: did the signals make money?"""
    print("="*70)
    print("SIGNAL PROFITABILITY BACKTEST")
    print("="*70)
    
    # Load model
    model_path = Path(__file__).parent / 'xgb_model.pkl'
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        feature_cols = data['features']
    
    # Generate fresh data
    print("\n[1/3] Generating validation data...")
    gen = FeatureGenerator()
    df, _ = gen.build_training_dataset(timeframe='1D', days=180)
    
    X = df[feature_cols].values
    y = df['target'].values
    returns = df['next_day_return'].values
    symbols = df['symbol'].values
    
    print(f"Samples: {len(X)}")
    
    # Simulate walk-forward backtest
    print("\n[2/3] Running walk-forward backtest...")
    
    # Split into train/val like TimeSeriesSplit
    split_point = int(len(X) * 0.7)
    X_train, X_val = X[:split_point], X[split_point:]
    y_train, y_val = y[:split_point], y[split_point:]
    returns_train, returns_val = returns[:split_point], returns[split_point:]
    
    # Train on first 70%, test on last 30%
    model_val = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_val.fit(X_train, y_train)
    
    # Get predictions
    signals = model_val.predict(X_val)
    proba = model_val.predict_proba(X_val)[:, 1]  # Probability of UP
    
    print("\n[3/3] Calculating returns...")
    print(f"\nValidation period: {len(X_val)} days")
    
    # Baseline: buy and hold every day
    baseline_return = returns_val.mean()
    print(f"Baseline (buy every day): {baseline_return:.4f} ({baseline_return*100:.2f}% avg daily)")
    
    # Strategy: only trade when model predicts UP
    signal_mask = signals == 1
    signal_returns = returns_val[signal_mask]
    
    if len(signal_returns) > 0:
        signal_mean = signal_returns.mean()
        signal_std = signal_returns.std()
        signal_sharpe = signal_mean / signal_std if signal_std > 0 else 0
        hit_rate = (signal_returns > 0).mean()
        
        print(f"\nStrategy (model says BUY):")
        print(f"  Trades taken: {signal_mask.sum()}/{len(X_val)} ({signal_mask.mean()*100:.1f}%)")
        print(f"  Avg return: {signal_mean:.4f} ({signal_mean*100:.2f}%)")
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Sharpe (daily): {signal_sharpe:.2f}")
        
        # Compare to baseline
        edge = signal_mean - baseline_return
        print(f"\n  Edge over baseline: {edge:.4f} ({edge*100:.2f}%)")
        
        if signal_mean > baseline_return:
            print(f"  [OK] SIGNAL IS PROFITABLE")
        else:
            print(f"  [WARN] Signal underperforms baseline")
    
    # High confidence signals only (proba > 0.6)
    high_conf_mask = proba > 0.6
    if high_conf_mask.sum() > 10:
        high_conf_returns = returns_val[high_conf_mask]
        print(f"\nHigh confidence (>60% proba):")
        print(f"  Trades: {high_conf_mask.sum()}")
        print(f"  Avg return: {high_conf_returns.mean():.4f}")
        print(f"  Hit rate: {(high_conf_returns > 0).mean():.1%}")
    
    # Annualized estimates
    trading_days = 252
    annual_baseline = baseline_return * trading_days
    annual_signal = signal_mean * trading_days
    
    print(f"\nAnnualized estimates (252 trading days):")
    print(f"  Baseline: {annual_baseline:.1%}")
    print(f"  Strategy: {annual_signal:.1%}")
    print(f"  Edge: {annual_signal - annual_baseline:.1%}")
    
    return signal_mean, baseline_return


if __name__ == '__main__':
    backtest_signals()
