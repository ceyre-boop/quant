# sovereign/ml_trainer.py
"""
Trains two XGBoost models: one per regime.
Retrains automatically after each simulation batch.
The correction loop: bad predictions → new labels → better model.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)

class MLTrainer:
    
    MOMENTUM_FEATURES  = ['zscore_20','zscore_50','roc_5','roc_10',
                          'rsi_14','rsi_divergence','volume_zscore',
                          'atr_distance','adx_14','hurst','atr_zscore']
    
    REVERSION_FEATURES = ['zscore_20','zscore_50','bb_pct_b','rsi_14',
                          'rsi_divergence','volume_zscore','atr_distance',
                          'adx_14','atr_zscore']
    
    XGB_PARAMS = {
        'n_estimators':       300,
        'max_depth':          4,      # shallow = less overfit
        'learning_rate':      0.05,
        'subsample':          0.8,
        'colsample_bytree':   0.8,
        'eval_metric':        'auc',
        'early_stopping_rounds': 30,
        'random_state':       42,
    }
    
    def build_momentum_labels(self, df: pd.DataFrame,
                               forward_bars: int = 10,
                               continuation_threshold: float = 0.6) -> pd.Series:
        """
        Winner: price continues in breakout direction by 60% of
        initial move within 10 bars WITHOUT reversing through stop.
        Entry anchor: T+1 open.
        """
        initial_move = df['close'] - df['close'].shift(5)
        direction    = np.sign(initial_move)
        atr          = df['close'].rolling(14).std() * 1.5  # simplified ATR proxy
        
        labels = []
        for i in range(len(df)):
            if i + forward_bars >= len(df) or pd.isna(initial_move.iloc[i]):
                labels.append(np.nan)
                continue
            
            entry     = df['open'].iloc[i+1] if i+1 < len(df) else np.nan
            if pd.isna(entry):
                labels.append(np.nan)
                continue
            
            d         = direction.iloc[i]
            stop_dist = atr.iloc[i]
            target    = continuation_threshold * abs(initial_move.iloc[i])
            
            future    = (df['close'].iloc[i+1:i+1+forward_bars] - entry) * d
            
            if (future < -stop_dist).any():
                labels.append(0)   # stopped out
            elif (future >= target).any():
                labels.append(1)   # hit target
            else:
                labels.append(0)   # neither
        
        return pd.Series(labels, index=df.index)
    
    def build_reversion_labels(self, df: pd.DataFrame,
                                 forward_bars: int = 15,
                                 reversion_threshold: float = 0.6) -> pd.Series:
        """
        Winner: price reverts 60% of distance to 50-bar mean
        within 15 bars WITHOUT hitting 1.5x ATR stop.
        Entry anchor: T+1 open.
        """
        mean     = df['close'].rolling(50).mean()
        distance = mean - df['close']
        atr      = df['close'].rolling(14).std() * 1.5
        
        labels = []
        for i in range(len(df)):
            if i + forward_bars >= len(df) or pd.isna(distance.iloc[i]):
                labels.append(np.nan)
                continue
            
            entry     = df['open'].iloc[i+1] if i+1 < len(df) else np.nan
            if pd.isna(entry):
                labels.append(np.nan)
                continue
            
            d         = np.sign(distance.iloc[i])
            stop_dist = atr.iloc[i]
            target    = reversion_threshold * abs(distance.iloc[i])
            
            future    = (df['close'].iloc[i+1:i+1+forward_bars] - entry) * d
            
            if (future < -stop_dist).any():
                labels.append(0)
            elif (future >= target).any():
                labels.append(1)
            else:
                labels.append(0)
        
        return pd.Series(labels, index=df.index)
    
    def train(self, features_df: pd.DataFrame,
              labels: pd.Series,
              feature_cols: list,
              regime: str) -> tuple:
        """
        Walk-forward train. Returns calibrated model + AUC score.
        Prints fold results so you know exactly where it fails.
        """
        clean = features_df[feature_cols].copy()
        clean['label'] = labels
        clean = clean.dropna()
        
        if len(clean) < 200:
            print(f"⚠️  {regime}: only {len(clean)} clean rows — need 200+")
            return None, 0.0
        
        # Walk-forward: last 20% as test
        split = int(len(clean) * 0.8)
        X_train, X_test = clean[feature_cols].iloc[:split], clean[feature_cols].iloc[split:]
        y_train, y_test = clean['label'].iloc[:split], clean['label'].iloc[split:]
        
        model = xgb.XGBClassifier(**self.XGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        
        # Calibrate
        calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
        calibrated.fit(X_train, y_train)
        
        probs = calibrated.predict_proba(X_test)[:,1]
        auc   = roc_auc_score(y_test, probs)
        
        # Win rate at probability floor
        floor = 0.60 if regime == 'momentum' else 0.50
        high_conf = y_test[probs >= floor]
        win_rate  = high_conf.mean() if len(high_conf) > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"  {regime.upper()} MODEL TRAINED")
        print(f"  AUC:          {auc:.4f}")
        print(f"  Win rate ({floor}+): {win_rate:.1%} on {len(high_conf)} signals")
        print(f"  Train rows:   {len(X_train)}")
        print(f"  Test rows:    {len(X_test)}")
        print(f"  Label balance: {y_train.mean():.1%} positive")
        print(f"{'='*50}\n")
        
        # Feature importance — tells you what actually matters
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi = fi.sort_values(ascending=False)
        print("  TOP FEATURES:")
        for feat, imp in fi.head(5).items():
            print(f"    {feat:<25} {imp:.4f}")
        
        return calibrated, auc
    
    def retrain_on_failures(self, features_df, labels, feature_cols,
                             regime, failure_indices):
        """
        Called after each simulation batch.
        Up-weights the failures so the model learns from them.
        This is the correction loop.
        """
        clean = features_df[feature_cols].copy()
        clean['label'] = labels
        clean = clean.dropna()
        
        # Build sample weights — failures get 3x weight
        weights = pd.Series(1.0, index=clean.index)
        for idx in failure_indices:
            if idx in weights.index:
                weights[idx] = 3.0
        
        X = clean[feature_cols]
        y = clean['label']
        w = weights
        
        model = xgb.XGBClassifier(**self.XGB_PARAMS)
        model.fit(X, y, sample_weight=w, verbose=False)
        
        calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
        calibrated.fit(X, y, sample_weight=w)
        
        return calibrated
