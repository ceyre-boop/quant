"""
Sovereign Phase 2 Gate Execution
Runs FactorZooScanner on SPY historical data.
"""

import pandas as pd
import logging
from sovereign.data.feeds.alpaca_feed import AlpacaFeed
from sovereign.features.factor_zoo import FactorZooScanner, run_phase2_gate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def execute_gate():
    from datetime import datetime, timezone
    from sovereign.data.feeds.macro_feed import MacroFeed
    
    feed = AlpacaFeed()
    macro_feed = MacroFeed()
    tickers = ['SPY', 'NVDA', 'TLT']
    start_dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
    
    logger.info("Loading macro data...")
    macro_df = macro_feed.get_all_series(start='2021-01-01')
    
    robust_features = set()
    
    for ticker in tickers:
        logger.info(f"--- Processing {ticker} ---")
        df = feed.get_bars(ticker, start=start_dt, use_cache=True)
        
        if df.empty:
            continue
            
        scanner = FactorZooScanner()
        
        # IS Scan
        df_is = df.loc['2022-01-01':'2024-12-31']
        feat_is = scanner.build_feature_matrix(df_is, macro_df=macro_df)
        res_is = scanner.scan(feat_is)
        
        # OOS Scan
        df_oos = df.loc['2025-01-01':]
        feat_oos = scanner.build_feature_matrix(df_oos, macro_df=macro_df)
        res_oos = scanner.scan(feat_oos)
        
        # Intersection
        is_passed = set(res_is[res_is['is_real']]['feature'])
        oos_passed = set(res_oos[res_oos['is_real']]['feature'])
        
        ticker_robust = is_passed.intersection(oos_passed)
        logger.info(f"{ticker} Robust Features: {ticker_robust}")
        robust_features.update(ticker_robust)

    logger.info("==========================================")
    logger.info(f"FINAL REGIME_ROBUST FEATURES ({len(robust_features)}):")
    for f in sorted(list(robust_features)):
        logger.info(f" [PASS] {f}")
    logger.info("==========================================")
    
    if len(robust_features) >= 8:
        logger.info("PHASE 2 GATE: CLEARED")
    else:
        logger.info(f"PHASE 2 GATE: FAILED ({len(robust_features)}/8 robust features)")

if __name__ == "__main__":
    execute_gate()
