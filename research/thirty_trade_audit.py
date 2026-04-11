"""
Thirty Trade Audit Protocol (V2.2 - FINAL)
Pillar 9: Verification & Validation

Locked Institutional Configuration:
- Probability Floor: 0.50 (Empirically Derived)
- Stop Logic: Direction-aware (v2)
- Entry: Next Bar Open
- Regime Gate: Enabled (ADX < 28, VIX_Z < 2.0)
"""

import pandas as pd
import numpy as np
import logging
import pickle
import random
import yfinance as yf
from training.mean_reversion_engine import build_mean_reversion_features, MeanReversionRiskEngine
from orchestrator.regime_filter import is_mean_reversion_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LOCKED CONFIG
PROB_FLOOR = 0.50

def run_thirty_trade_audit():
    # 1. LOAD REBUILD MODEL
    try:
        with open('training/xgb_model.pkl', 'rb') as f:
            payload = pickle.load(f)
            model = payload['model']
            feature_cols = payload['features']
        logger.info(f"V2.2 Model Loaded. Locked Prob Floor: {PROB_FLOOR}")
    except Exception as e:
        logger.error(f"Failed to load V2.2 Rebuild: {e}")
        return

    # 2. SELECT CANDIDATE POOL (2024 Data)
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    start_date = "2023-10-01"
    end_date = "2024-12-31"
    
    all_signals = []
    
    logger.info("Downloading audit data and generating signals...")
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
    vix_df = pd.DataFrame({'close': vix.squeeze()}, index=vix.index)

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        features = build_mean_reversion_features(df, vix_df)
        df_2024 = df[df.index >= '2024-01-01']
        
        # Risk Engine
        risk_engine = MeanReversionRiskEngine()
        
        for i in range(len(df_2024) - 16):
            idx = df_2024.index[i]
            feat_row = features.loc[idx]
            
            if feat_row.isna().any(): continue
            
            # RULE 1: Regime Gate
            regime_pass, reason = is_mean_reversion_regime(feat_row.to_dict())
            if not regime_pass: continue
            
            # RULE 2: XGBoost Probability
            X = pd.DataFrame([feat_row[feature_cols]])
            prob = model.predict_proba(X)[:, 1][0]
            
            # RULE 3: LOCKED PROBABILITY FLOOR
            if prob < PROB_FLOOR: continue
            
            # Reconstruct mean/atr
            mean_50 = df_2024['close'].rolling(50).mean().loc[idx]
            high_low = df['high'] - df['low']
            atr_val = high_low.rolling(14).mean().loc[idx]
            
            # RULE 4: EV Gate
            trade = risk_engine.calculate_trade(
                entry_price=df_2024['close'].iloc[i],
                xgb_prob=prob,
                atr=atr_val,
                mean_target=mean_50
            )
            
            if trade['take_trade']:
                is_long = trade['target'] > trade['entry']
                entry_price = df_2024['open'].iloc[i+1] # Next Bar Open
                
                # RULE 5: Fixed Direction-Aware Stop
                atr_stop = 1.5 * atr_val
                if is_long:
                    stop_p = entry_price - atr_stop
                else:
                    stop_p = entry_price + atr_stop # ABOVE for short
                
                all_signals.append({
                    'symbol': s,
                    'date': idx,
                    'entry_price': entry_price,
                    'target': trade['target'],
                    'stop': stop_p,
                    'prob': prob,
                    'is_long': is_long,
                    'original_df': df
                })

    if len(all_signals) == 0:
        logger.error("Zero signals found for audit with current filters.")
        return

    # 3. SAMPLE OR EXHAUSTIVE AUDIT
    n_actual = min(len(all_signals), 30)
    audit_sample = random.sample(all_signals, n_actual)
    logger.info(f"Auditing {n_actual} signals (Population Exhaustion Audit)...")


    hits = 0
    total_pnl = 0
    
    for i, t in enumerate(audit_sample):
        df = t['original_df']
        entry_idx = df.index.get_loc(t['date']) + 1
        entry_price = t['entry_price']
        stop = t['stop']
        target = t['target']
        is_long = t['is_long']
        
        # Reversion check (15 bars forward)
        fwd = df.iloc[entry_idx + 1 : entry_idx + 16]
        
        success = False
        outcome_p = fwd['close'].iloc[-1] # default if neither hit
        
        if is_long:
            for _, bar in fwd.iterrows():
                if bar['low'] <= stop:
                    outcome_p = stop
                    break
                if bar['high'] >= target:
                    success = True
                    outcome_p = target
                    break
        else:
            for _, bar in fwd.iterrows():
                if bar['high'] >= stop:
                    outcome_p = stop
                    break
                if bar['low'] <= target:
                    success = True
                    outcome_p = target
                    break
                    
        if success: hits += 1
        
        pnl = (outcome_p - entry_price) if is_long else (entry_price - outcome_p)
        total_pnl += pnl

    actual_reversion_pct = (hits / 30) * 100
    
    logger.info("========================================")
    logger.info("FINAL 30-TRADE AUDIT RESULTS (2024 OOS)")
    logger.info(f"Actual Reversion Pct: {actual_reversion_pct:.2f}%")
    logger.info(f"Total Audit PnL: ${total_pnl:.2f}")
    logger.info(f"Audit Pass (>= 65%): {actual_reversion_pct >= 65}")
    logger.info("========================================")

if __name__ == "__main__":
    run_thirty_trade_audit()
