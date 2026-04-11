"""
V4.8 Institutional Engine - Integrated Reality Substrate
Pillar 15: Price Physics + Expectation Fusion

Final Production Configuration:
1. Momentum Trigger: |Z| > 2.0
2. Resilience Guard: CSD < 0.65
3. Inflection Guard: Logistic ODE (k)
4. Expectation Veto: IV Term Structure (VIX9D/VIX)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging
from training.engine_v4 import build_v4_features, build_v4_labels, AccumulationODE
from research.critical_slowing_detector import CriticalSlowingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_v4_8_production_rebuild():
    symbols = ['NVDA', 'AMZN', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'META', 'GOOGL']
    
    # 1. Harvest IV Data (Daily Spread)
    vix = yf.download("^VIX", period="2y", interval="1d")
    vix9d = yf.download("^VIX9D", period="2y", interval="1d")
    vix.columns = [c.lower() for c in (vix.columns.get_level_values(0) if isinstance(vix.columns, pd.MultiIndex) else vix.columns)]
    vix9d.columns = [c.lower() for c in (vix9d.columns.get_level_values(0) if isinstance(vix9d.columns, pd.MultiIndex) else vix9d.columns)]
    
    iv_df = pd.DataFrame(index=vix.index)
    iv_df['iv_spread'] = vix9d['close'] / vix['close']
    iv_df.index = iv_df.index.normalize()

    all_dfs = []
    detector = CriticalSlowingDetector(window=60)
    
    logger.info("Retraining V4.8 Integrated Reality Engine...")

    for s in symbols:
        df = yf.download(s, period="2y", interval="1h")
        if df.empty: continue
        df.columns = [c.lower() for c in (df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns)]
        
        # Core Features
        f = build_v4_features(df)
        
        # Physics Indicator (CSD)
        csd_res = detector.compute(df['close'])
        f['csd_score'] = csd_res['csd_score']
        
        # Expectation Indicator (IV Spread - Daily aligned to Hourly)
        df_daily_idx = df.index.normalize()
        f['iv_spread'] = df_daily_idx.map(iv_df['iv_spread'])
        f['iv_spread'] = f['iv_spread'].ffill().fillna(1.0)
        
        # Labels
        l = build_v4_labels(df)
        
        combined = pd.concat([f, l.rename('target')], axis=1).dropna()
        all_dfs.append(combined)

    full_df = pd.concat(all_dfs).sort_index()

    full_df = full_df.sort_index()
    
    feature_cols = [c for c in full_df.columns if c not in ['target']]
    
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    
    # Validation split (last 4 months)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, random_state=42)
    model.fit(train_df[feature_cols], train_df['target'])
    
    probs = model.predict_proba(test_df[feature_cols])[:, 1]
    auc = roc_auc_score(test_df['target'], probs)
    
    logger.info("========================================")
    logger.info("V4.8 INTEGRATED REALITY ENGINE")
    logger.info(f"OOS AUC (Physics + IV): {auc:.4f}")
    logger.info("========================================")
    
    with open('training/xgb_model_v4_8.pkl', 'wb') as f_out:
        pickle.dump({'model': model, 'features': feature_cols, 'type': 'integrated_reality'}, f_out)

if __name__ == "__main__":
    run_v4_8_production_rebuild()
