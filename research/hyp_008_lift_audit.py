"""
HYP-008 Accuracy Lift Audit
Objective: Measure the out-of-sample accuracy delta after adding decadal_divergence_score.
Constraint: Inclusion requires Lift >= 0.5% (0.005).
"""

import pandas as pd
import numpy as np
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lift_audit():
    # 1. DATA PREP
    df = pd.read_csv("data/router_labels_v5.csv")
    
    # Feature Sets
    baseline_features = ['hurst', 'zscore', 'csd', 'iv_spread', 'adx', 'volatility']
    # Decadal divergence is mapped as the 'era_slope' and 'era_integrity' from the Planetary Harvest.
    # We will use the 'era_slope' as the best proxy for decadal divergence direction.
    
    # 2. ALIGNMENT (PULLING FROM PLANETARY DATA)
    planetary_path = "data/planetary/whole_chart_corridors.csv"
    planetary_df = pd.read_csv(planetary_path)
    planetary_df['Date'] = pd.to_datetime(planetary_df['Date'])
    df['date_only'] = pd.to_datetime(pd.to_datetime(df['timestamp']).dt.date)
    
    all_aligned = []
    for s in df['ticker'].unique():
        s_pulse = df[df['ticker'] == s].copy()
        s_plane = planetary_df[planetary_df['ticker'] == s].copy()
        # Adjusted to 5-year (1260 day) window for sufficient overlap
        s_plane['decadal_divergence'] = (s_plane['close'] - s_plane['close'].rolling(1260).mean()) / s_plane['close'].rolling(1260).std()
        s_plane['decadal_divergence'] = s_plane['decadal_divergence'].shift(1) # Ensure NO LOOKAHEAD

        
        merged = pd.merge(s_pulse, s_plane[['Date', 'decadal_divergence']], 
                          left_on='date_only', right_on='Date', how='left')
        all_aligned.append(merged)
        
    final_df = pd.concat(all_aligned).dropna()
    X_base = final_df[baseline_features]
    X_challenge = final_df[baseline_features + ['decadal_divergence']]
    y = final_df['label']
    
    # 3. TRAIN-TEST SPLIT (Last 20% of time)
    split_idx = int(len(final_df) * 0.8)
    X_base_train, X_base_test = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
    X_chall_train, X_chall_test = X_challenge.iloc[:split_idx], X_challenge.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 4. TRAINING
    model_base = XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_chall = XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
    
    model_base.fit(X_base_train, y_train)
    model_chall.fit(X_chall_train, y_train)
    
    # 5. RESULTS
    acc_base = accuracy_score(y_test, model_base.predict(X_base_test))
    acc_chall = accuracy_score(y_test, model_chall.predict(X_chall_test))
    lift = acc_chall - acc_base
    
    print("\n" + "="*40)
    print("HYP-008 ACCURACY LIFT AUDIT")
    print("="*40)
    print(f"Baseline OOS Accuracy: {acc_base:.4%}")
    print(f"Challenger OOS Accuracy: {acc_chall:.4%}")
    print(f"ABS ACCURACY LIFT: {lift:.4%}")
    
    if lift >= 0.005:
        print("\nVERDICT: LIFT IS SIGNIFICANT (>= 0.5%).")
        print("ACTION: Integrate decadal_divergence into V6.5 Production.")
    else:
        print("\nVERDICT: LIFT IS NEGLIGIBLE (< 0.5%).")
        print("ACTION: Archive HYP-008. Information is already encoded in base features.")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_lift_audit()
