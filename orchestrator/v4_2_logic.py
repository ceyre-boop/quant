"""
V4.2 Momentum + CSD Predictive Exit Engine
Operationalizes the V4.1 Momentum signals with the V4.2 CSD Exhaustion Exit.
"""

import pandas as pd
import numpy as np
import logging
from training.engine_v4 import build_v4_features
from research.critical_slowing_detector import CriticalSlowingDetector

logger = logging.getLogger(__name__)

def monitor_trade_health(symbol, recent_1h_data):
    """
    V4.2 Predictive Exit Logic.
    Returns (Should_Exit, CSD_Score, Reason)
    """
    detector = CriticalSlowingDetector(window=60)
    
    # Generate CSD scores
    csd_results = detector.compute(recent_1h_data['close'])
    current_csd = csd_results['csd_score'].iloc[-1]
    
    # REGIME EXHAUSTION GATE (0.65)
    # Based on 2025 Falsification Test results.
    if current_csd > 0.65:
        return True, current_csd, "CRITICAL_SLOWING_DETECTED (Regime Transition Imminent)"
        
    return False, current_csd, "HEALTHY_TREND"

def run_v4_2_signal_harvester(df: pd.DataFrame, ticker: str, model_data: dict):
    """
    Generates V4.1 Momentum signals with V4.2 CSD pre-validation.
    """
    model = model_data['model']
    feature_cols = model_data['features']
    
    # 1. Feature Generation
    f = build_v4_features(df)
    
    # 2. THE MOMENTUM TRIGGER (H > 0.52)
    current_h = f['hurst'].iloc[-1]
    if current_h <= 0.52:
        return None, f"VETO: Hurst={current_h:.3f} (Not Persistent)"
        
    # 3. THE CSD HEALTH GATE (CSD < 0.65)
    # Never enter a trade if the system is already destabilizing
    detector = CriticalSlowingDetector(window=60)
    csd_res = detector.compute(df['close'])
    current_csd = csd_res['csd_score'].iloc[-1]
    
    if current_csd > 0.65:
        return None, f"VETO: CSD={current_csd:.3f} (System Destabilizing)"
        
    # 4. MODEL PROBABILITY
    prob = model.predict_proba(f[feature_cols].iloc[[-1]])[0, 1]
    
    if prob >= 0.60:
        return {
            'ticker': ticker,
            'prob': prob,
            'hurst': current_h,
            'csd': current_csd,
            'direction': 'LONG' if f['zscore_20'].iloc[-1] > 0 else 'SHORT',
            'timestamp': df.index[-1]
        }, "SIGNAL_GENERATED"
        
    return None, f"VETO: Prob={prob:.3f} (Wait for Conviction)"
