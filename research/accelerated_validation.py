"""
Accelerated Validation (Fold 5 Unlock)
Executes the 200-Trade Contract across the 2025-2026 Virgin Territory.

Strict Rules:
- T+1 Open Entry (The Reality Anchor)
- Prob Floor: 0.45
- Gap Entropy Ceiling: 0.50
- Regime Gates: Enabled
"""

import pandas as pd
import numpy as np
import logging
import pickle
import yfinance as yf
from training.mean_reversion_engine import build_mean_reversion_features, MeanReversionRiskEngine
from orchestrator.regime_filter import is_mean_reversion_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCKED_PROB = 0.45
LOCKED_GAP = 0.50
LEDGER_PATH = "data/paper_trading/ledger_v2_2.csv"

def run_accelerated_validation():
    # 1. LOAD MODEL
    try:
        with open('training/xgb_model.pkl', 'rb') as f:
            payload = pickle.load(f)
            model = payload['model']
            feature_cols = payload['features']
    except Exception as e:
        logger.error(f"Failed to load V2.2: {e}")
        return

    # 2. VIRGIN TERRITORY (Fold 5: 2025 - Apr 2026)
    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    start_date = "2024-10-01" # Buffer
    end_date = "2026-04-10"
    
    logger.info("Downloading Virgin Territory (Fold 5) data...")
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
    vix_df = pd.DataFrame({'close': vix.squeeze()}, index=vix.index)

    realized_trades = []

    for s in symbols:
        df = yf.download(s, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        features = build_mean_reversion_features(df, vix_df)
        df_oos = df[df.index >= '2025-01-01']
        
        risk_engine = MeanReversionRiskEngine()
        
        for i in range(len(df_oos) - 16):
            idx = df_oos.index[i]
            feat_row = features.loc[idx]
            if feat_row.isna().any(): continue
            
            # RULE 1: Regime Gate
            pass_gate, reason = is_mean_reversion_regime(feat_row.to_dict())
            if not pass_gate: continue
            
            # RULE 2: Prob Floor (0.45)
            X = pd.DataFrame([feat_row[feature_cols]])
            prob = model.predict_proba(X)[:, 1][0]
            if prob < LOCKED_PROB: continue
            
            # SIGNAL GEN (T_close)
            mean_50 = df_oos['close'].rolling(50).mean().loc[idx]
            atr_val = (df_oos['high'] - df_oos['low']).rolling(14).mean().loc[idx]
            
            trade = risk_engine.calculate_trade(df_oos['close'].iloc[i], prob, atr_val, mean_50)
            if not trade['take_trade']: continue
            
            # REALITY ENTRY (T+1 Open)
            entry_price = df_oos['open'].iloc[i+1]
            
            # RULE 3: Gap Entropy Ceiling (0.50)
            gap = abs(entry_price - df_oos['close'].iloc[i])
            entropy = gap / atr_val
            if entropy > LOCKED_GAP: continue
            
            # RULE 4: Stop Math (Direction Aware)
            is_long = trade['target'] > trade['entry']
            stop_dist = 1.5 * atr_val
            stop_p = (entry_price - stop_dist) if is_long else (entry_price + stop_dist)
            
            # SIMULATE OUTCOME (15 bars fwd)
            fwd = df_oos.iloc[i+2 : i+17]
            success = False
            outcome_p = fwd['close'].iloc[-1]
            bars = len(fwd)
            
            for b_idx, (_, bar) in enumerate(fwd.iterrows()):
                if is_long:
                    if bar['low'] <= stop_p:
                        outcome_p = stop_p
                        bars = b_idx + 1
                        break
                    if bar['high'] >= trade['target']:
                        success = True
                        outcome_p = trade['target']
                        bars = b_idx + 1
                        break
                else:
                    if bar['high'] >= stop_p:
                        outcome_p = stop_p
                        bars = b_idx + 1
                        break
                    if bar['low'] <= trade['target']:
                        success = True
                        outcome_p = trade['target']
                        bars = b_idx + 1
                        break
            
            pnl = (outcome_p - entry_price) if is_long else (entry_price - outcome_p)
            
            realized_trades.append({
                'date': idx.isoformat(),
                'ticker': s,
                'direction': 'LONG' if is_long else 'SHORT',
                'signal_close': float(df_oos['close'].iloc[i]),
                'entry_open': entry_price,
                'stop': stop_p,
                'target': trade['target'],
                'xgb_prob': prob,
                'gap_entropy': entropy,
                'adx_at_entry': feat_row['adx_14'],
                'vix_z_at_entry': feat_row['vix_zscore'],
                'rr_ratio': abs(trade['target'] - entry_price) / stop_dist,
                'outcome': 'HIT' if success else ('STOPPED' if outcome_p == stop_p else 'TIME_EXIT'),
                'bars_to_outcome': bars,
                'actual_rr': pnl / abs(entry_price - stop_p),
                'regime_flag': reason
            })

    # 3. RESULTS
    if not realized_trades:
        logger.error("Zero realized trades in Virgin Territory.")
        return

    results = pd.DataFrame(realized_trades).sort_values('date')
    results.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
    
    hits = (results['outcome'] == 'HIT').sum()
    reversion_rate = (hits / len(results)) * 100
    
    logger.info("========================================")
    logger.info("VIRGIN TERRITORY RESULTS (FOLD 5)")
    logger.info(f"Total Trades Realized: {len(results)}")
    logger.info(f"Actual Reversion Rate: {reversion_rate:.2f}%")
    logger.info(f"Avg Actual R:R: {results['actual_rr'].mean():.2f}")
    logger.info(f"System Integrity Score: {100 if reversion_rate >= 63 else 0}")
    logger.info("========================================")

if __name__ == "__main__":
    run_accelerated_validation()
