"""
MONDAY KILL ZONE EXECUTION (09:50 - 10:30 AM ET)
Fuses Layer 2 Router (V5.1) and Specialist (V4.8) with V5.2 Dynamic RR.
Objective: Execute the first automated paper trade with Grade-Based Sizing.
"""

import yfinance as yf
import pandas as pd
import pickle
import logging
from datetime import datetime
from execution.paper_trading import PaperTradingEngine, EnhancedEntrySignal
from training.engine_v4 import build_v4_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# THE MONDAY CONFIG
TRINITY_ASSETS = ['META', 'PFE', 'UNH']
ROUTER_PATH = 'training/regime_router_v5.pkl'
SPECIALIST_PATH = 'training/xgb_model_v4_8.pkl'

def run_monday_killzone():
    logger.info("Initializing Monday Kill Zone Execution (09:50 AM)...")
    
    # 1. LOAD THE MODELS
    with open(ROUTER_PATH, 'rb') as f:
        router_payload = pickle.load(f)
        router_model = router_payload['model']
        router_features = router_payload['features']
        
    with open(SPECIALIST_PATH, 'rb') as f:
        mom_payload = pickle.load(f)
        mom_model = mom_payload['model']
        mom_features = mom_payload['features']

    # 2. THE PAPER ENGINE
    paper_engine = PaperTradingEngine()

    for ticker in TRINITY_ASSETS:
        logger.info(f"Scanning {ticker} for Momentum Window...")
        
        # Pull 30d hourly history for feature building
        df = yf.download(ticker, period="30d", interval="1h")
        if df.empty: continue
        df.columns = [c.lower() for c in (df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns)]
        
        # 3. BUILD FEATURES (V5.1 Integrated Reality)
        f_mom = build_v4_features(df)
        # Note: In production, IV_Spread and CSD would be pulled from Realtime feeds
        # For this Monday Phase 1 version, we use the baseline V4.0 feature set + ATR
        current_features = f_mom.iloc[-1:]
        
        # 4. LAYER 2 ROUTER (MOMENTUM WINDOW IDENTIFICATION)
        # We assume the Router inputs are correctly mapped from f_mom
        try:
            regime_pred = router_model.predict(current_features[router_features])[0]
        except:
            regime_pred = 0 # Default to Stay Flat

        if regime_pred == 1: # Momentum SPECIALIST ACTIVATED
            logger.info(f"🚀 {ticker} MOMENTUM WINDOW IDENTIFIED. ACTIVATING SPECIALIST.")
            
            # Predict Probability
            prob = mom_model.predict_proba(current_features[mom_features])[0][1]
            
            # 5. GRADE-BASED SIZING
            grade = 'C'
            risk_pct = 0.0025
            if prob > 0.92: grade, risk_pct = 'A+', 0.015
            elif prob > 0.78: grade, risk_pct = 'A', 0.010
            elif prob > 0.65: grade, risk_pct = 'B', 0.005
            
            logger.info(f"📈 Signal Confirmed: Grade {grade} ({prob:.2%}) | Risk: {risk_pct:.2%}")
            
            # 6. EXECUTE WITH V5.2 RR ENGINE
            current_price = df['close'].iloc[-1]
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            
            signal = EnhancedEntrySignal(
                symbol=ticker,
                direction=1 if current_features['zscore_20'].iloc[-1] > 0 else -1,
                position_size=risk_pct, # Grade-based risk
                stop_loss=0, # Calculated by RR Engine
                take_profit_1=0, # Calculated by RR Engine
                entry_model="V4.8-Integrated",
                regime=f"Momentum (Prob: {prob:.2%})",
                entry_model_expected_r=2.0
            )
            
            paper_engine.execute_signal(signal, current_price, atr)
            logger.info("============================================")
            logger.info(f"✅ PAPER TRADE EXECUTED: {ticker} (GRADE {grade})")
            logger.info("============================================")
            break # Objective complete: One Trade rule

if __name__ == "__main__":
    run_monday_killzone()
