"""
V5.1 Regime Router - The Layer 2 Meta-Model
Trains a classifier to predict the optimal Specialist Strategy 
(None, Momentum, or Reversion) based on macro-regime features.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_regime_router():
    # 1. LOAD THE GROUND TRUTH MAP
    df = pd.read_csv("data/router_labels_v5.csv")
    
    # Features for the Router (Macro Sentiment + Physics)
    feature_cols = ['hurst', 'zscore', 'csd', 'iv_spread', 'adx', 'volatility']


    target_col = 'label'
    
    # 2. CLASS BALANCING (Oversampling the minority regimes)
    # Since 88% is 'Stay Flat', we want the model to actually learn the signal windows.
    from sklearn.utils import resample
    
    df_flat = df[df[target_col] == 0]
    df_mom = df[df[target_col] == 1]
    df_rev = df[df[target_col] == 2]
    
    # Upsample specialists to match 30% of the flat volume
    df_mom_upsampled = resample(df_mom, replace=True, n_samples=len(df_flat)//3, random_state=42)
    df_rev_upsampled = resample(df_rev, replace=True, n_samples=len(df_flat)//3, random_state=42)
    
    df_balanced = pd.concat([df_flat, df_mom_upsampled, df_rev_upsampled])
    
    X = df_balanced[feature_cols]
    y = df_balanced[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. TRAIN THE ROUTER
    logger.info("Training Layer 2 Regime Router (XGBoost)...")
    router = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    
    router.fit(X_train, y_train)
    
    # 4. EVALUATION
    y_pred = router.predict(X_test)
    
    logger.info("========================================")
    logger.info("LAYER 2 ROUTER PERFORMANCE")
    logger.info("========================================")
    print(classification_report(y_test, y_pred))
    
    # 5. ARCHIVE
    with open('training/regime_router_v5.pkl', 'wb') as f_out:
        pickle.dump({'model': router, 'features': feature_cols}, f_out)
    
    logger.info("Router archived: training/regime_router_v5.pkl")

if __name__ == "__main__":
    train_regime_router()
