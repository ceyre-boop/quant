"""
Phase 9 — Veto Rate Diagnostic (V1.2)
Objective: Audit the rejection rate of each stage over a 20-day period.
"""

import pandas as pd
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from sovereign.orchestrator import SovereignOrchestrator
from sovereign.data.feeds.alpaca_feed import AlpacaFeed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_veto_diagnostic():
    orchestrator = SovereignOrchestrator(mode='paper')
    feed = AlpacaFeed()
    
    # 10 Trinity & Growth Assets for wide testing
    universe = ["META", "AAPL", "NVDA", "TSLA", "PFE", "UNH", "SPY", "TLT", "AMD", "ARM"]
    
    logger.info(f"Running Veto Rate Diagnostic across {len(universe)} assets...")

    for symbol in universe:
        logger.info(f"Auditing {symbol}...")
        
        # 1. Pull data
        df = feed.get_bars(symbol, start=datetime.now()-timedelta(days=30), timeframe="1Day")
        if df.empty: continue
        
        # 2. Build correctly typed feature record (Aligned with contracts/types.py V5.1)
        from contracts.types import (
            SovereignFeatureRecord, RegimeFeatures, MacroFeatures, 
            MomentumFeatures
        )
        
        last_price = df['close'].iloc[-1]
        last_atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        try:
            from sovereign.features.regime.hurst import compute_hurst_features
            h_df = compute_hurst_features(df)
            
            # Match 665-line spec exactly
            mock_record = SovereignFeatureRecord(
                symbol=symbol,
                timestamp=df.index[-1].strftime('%Y-%m-%dT%H:%M:%S'),
                regime=RegimeFeatures(
                    hurst_short=h_df['hurst_short'].iloc[-1],
                    hurst_long=h_df['hurst_long'].iloc[-1],
                    hurst_signal='NEUTRAL',
                    csd_score=0.4,
                    csd_signal='NEUTRAL',
                    hmm_state=1,
                    hmm_state_label='NORMAL',
                    hmm_confidence=0.9,
                    hmm_transition_prob=0.1,
                    adx=25.0,
                    adx_signal='ESTABLISHED'
                ),
                momentum=MomentumFeatures(
                    logistic_ode_score=0.5,
                    jt_momentum_12_1=0.0,
                    volume_entropy=1.5,
                    rsi_14=50.0,
                    rsi_signal='NEUTRAL'
                ),
                macro=MacroFeatures(
                    yield_curve_slope=0.5, 
                    yield_curve_velocity=0.0,
                    erp=0.05,
                    m2_velocity=1.5,
                    cape_zscore=1.0,
                    cot_zscore=0.0,
                    hyg_spread_bps=150,
                    macro_signal='NEUTRAL'
                ),
                petroulas=None,
                bar_ohlcv=df.iloc[-1].to_dict(),
                is_valid=True,
                validation_errors=[]
            )
            
            # 3. RUN ORCHESTRATOR
            orchestrator.run_session(symbol, mock_record, last_price, last_atr, 100000.0)
            
        except Exception as e:
            logger.error(f"Error auditing {symbol}: {e}")

    # 4. PRINT HEALTH REPORT
    orchestrator.veto_ledger.print_health_report()

if __name__ == "__main__":
    run_veto_diagnostic()
