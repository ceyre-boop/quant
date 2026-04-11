"""
Pillar 10: Paper Trade Lifecycle Engine (V2.2)
Governs the 200-Trade Contract execution with total reality separation.

Flow:
1. Midnight (T): Run identify_signals() -> Append to PENDING
2. Market Open (T+1): Run execute_open() -> Move to OPEN with actual tape price
3. Market Close (T+1): Run monitor_outcomes() -> Check for HIT/STOPPED
"""

import pandas as pd
import numpy as np
import logging
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from training.mean_reversion_engine import build_mean_reversion_features, MeanReversionRiskEngine
from orchestrator.regime_filter import is_mean_reversion_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEDGER_PATH = "data/paper_trading/ledger_v2_2.csv"
PENDING_PATH = "data/paper_trading/pending_v2_2.csv"
PROB_FLOOR = 0.45
GAP_CEILING = 0.50

VETO_LEDGER_PATH = "data/paper_trading/veto_ledger_v2_2.csv"

def identify_signals():
    """Phase A: Midnight Signal Harvesting with Veto Logging."""
    try:
        with open('training/xgb_model.pkl', 'rb') as f:
            payload = pickle.load(f)
            model = payload['model']
            feature_cols = payload['features']
    except Exception as e:
        logger.error(f"Failed to load V2.2 Rebuild: {e}")
        return

    symbols = ['SPY', 'QQQ', 'NVDA', 'AMD', 'GLD', 'SLV', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    vix = yf.download('^VIX', end=end_date)['Close']
    if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
    vix_df = pd.DataFrame({'close': vix.squeeze()}, index=vix.index)

    new_pending = []
    vetoes = []

    for s in symbols:
        df = yf.download(s, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        features = build_mean_reversion_features(df, vix_df)
        last_idx = df.index[-1]
        feat_row = features.loc[last_idx]
        
        # XGBoost Score (computed before gates for Veto Logging)
        X = pd.DataFrame([feat_row[feature_cols]])
        prob = model.predict_proba(X)[:, 1][0]
        
        # 1. REGIME GATE
        regime_pass, reason = is_mean_reversion_regime(feat_row.to_dict())
        if not regime_pass:
            vetoes.append({
                'date': last_idx.isoformat(),
                'ticker': s,
                'veto_reason': reason,
                'xgb_prob': prob,
                'gap_entropy': float(feat_row['gap_entropy'])
            })
            continue
        
        # 2. PROBABILITY GATE
        if prob < PROB_FLOOR:
            vetoes.append({
                'date': last_idx.isoformat(),
                'ticker': s,
                'veto_reason': f"PROB: {prob:.3f} < {PROB_FLOOR}",
                'xgb_prob': prob,
                'gap_entropy': float(feat_row['gap_entropy'])
            })
            continue
        
        # 3. EV GATE
        mean_50 = df['close'].rolling(50).mean().loc[last_idx]
        atr_val = (df['high'] - df['low']).rolling(14).mean().loc[last_idx]
        
        risk_engine = MeanReversionRiskEngine()
        trade = risk_engine.calculate_trade(df['close'].iloc[-1], prob, atr_val, mean_50)
        
        if trade['take_trade']:
            new_pending.append({
                'date': last_idx.isoformat(),
                'ticker': s,
                'direction': 'LONG' if trade['target'] > trade['entry'] else 'SHORT',
                'signal_close': float(df['close'].iloc[-1]),
                'xgb_prob': prob,
                'target': float(trade['target']),
                'atr_at_signal': float(atr_val),
                'adx_at_entry': float(feat_row['adx_14']),
                'vix_z_at_entry': float(feat_row['vix_zscore']),
                'rr_ratio': float(trade['rr_actual']),
                'regime_flag': reason
            })
        else:
            vetoes.append({
                'date': last_idx.isoformat(),
                'ticker': s,
                'veto_reason': "EV_GATE: Negative EV or R:R < 1.5",
                'xgb_prob': prob,
                'gap_entropy': float(feat_row['gap_entropy'])
            })

    # Save Vetoes
    if vetoes:
        v_df = pd.DataFrame(vetoes)
        try:
            old_v = pd.read_csv(VETO_LEDGER_PATH)
            pd.concat([old_v, v_df]).to_csv(VETO_LEDGER_PATH, index=False)
        except:
            v_df.to_csv(VETO_LEDGER_PATH, index=False)

    if new_pending:

        pending_df = pd.DataFrame(new_pending)
        # Use simple CSV append or create
        try:
            old_pending = pd.read_csv(PENDING_PATH)
            pd.concat([old_pending, pending_df]).to_csv(PENDING_PATH, index=False)
        except:
            pending_df.to_csv(PENDING_PATH, index=False)
        logger.info(f"Signal Phase Complete: {len(new_pending)} signals pending for open.")

def execute_pending_at_open():
    """Phase B: Reality Entry at Market Open."""
    try:
        pending = pd.read_csv(PENDING_PATH)
    except:
        return

    executed = []
    still_pending = []

    for _, sig in pending.iterrows():
        # Get actual morning open
        ticker = sig['ticker']
        df = yf.download(ticker, period="1d")
        if df.empty:
            still_pending.append(sig)
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        open_price = float(df['open'].iloc[0])
        
        # 4. GAP ENTROPY CEILING
        gap = abs(open_price - sig['signal_close'])
        gap_entropy = gap / sig['atr_at_signal']
        
        if gap_entropy > GAP_CEILING:
            logger.warning(f"SKIPPED {ticker}: Gap Entropy {gap_entropy:.2f} > {GAP_CEILING}")
            continue
            
        # 5. FINAL STOP LOGIC
        is_long = sig['direction'] == 'LONG'
        stop_dist = 1.5 * sig['atr_at_signal']
        stop_price = (open_price - stop_dist) if is_long else (open_price + stop_dist)
        
        # Log to Active Ledger
        trade_log = {
            'date': sig['date'],
            'ticker': sig['ticker'],
            'direction': sig['direction'],
            'signal_close': sig['signal_close'],
            'entry_open': open_price,
            'stop': stop_price,
            'target': sig['target'],
            'xgb_prob': sig['xgb_prob'],
            'gap_entropy': gap_entropy,
            'adx_at_entry': sig['adx_at_entry'],
            'vix_z_at_entry': sig['vix_z_at_entry'],
            'rr_ratio': abs(sig['target'] - open_price) / stop_dist,
            'outcome': 'OPEN',
            'bars_to_outcome': 0,
            'actual_rr': 0.0,
            'regime_flag': sig['regime_flag']
        }
        executed.append(trade_log)

    if executed:
        ledger = pd.read_csv(LEDGER_PATH)
        pd.concat([ledger, pd.DataFrame(executed)]).to_csv(LEDGER_PATH, index=False)
        logger.info(f"Execution Phase Complete: {len(executed)} trades moved to Ledger.")
    
    # Clear pending
    pd.DataFrame(still_pending).to_csv(PENDING_PATH, index=False)

if __name__ == "__main__":
    # In a real environment, these would be called on schedule.
    # Today we initialize the first signals.
    identify_signals()
    # execute_pending_at_open() will be run tomorrow at 9:31 AM.
