"""
Phase 0B — Factor Zoo Diagnostic (V6.0)
Objective: Validate features at their natural horizons with BH FDR correction.
Reference: SOVEREIGN — FORWARD PLAN
"""

import json
import logging
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/diagnostic_current.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from sovereign.data.feeds.alpaca_feed import AlpacaFeed
from sovereign.features.factor_zoo import FactorZooScanner

# 1. FEATURE TIMESCALE MAPPING
FEATURE_GROUPS = {
    'fast': {
        'features': ['volume_zscore', 'rsi', 'logistic_k', 'volume_entropy'],
        'forward_windows': [1, 3, 5]   # 1-5 daily bars
    },
    'slow': {
        'features': ['hurst_short', 'hurst_long', 'csd_score', 'hmm_state', 'hmm_transition_prob', 'adx'],
        'forward_windows': [10, 20, 40]  # 2-8 week horizon
    },
    'macro': {
        'features': ['yield_curve_slope', 'yield_curve_velocity', 'erp', 'cape_zscore', 'cot_zscore'],
        'forward_windows': [20, 60]      # 1-3 month horizon
    }
}

DIAGNOSTIC_UNIVERSE = ["NVDA", "TSLA", "AMD", "META", "TLT", "SPY"]
TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-01-01"
OOS_START   = "2025-01-01"
OOS_END     = "2026-01-01"
ICIR_THRESHOLD = 0.30

def apply_fdr_correction(p_values: list, alpha: float = 0.05) -> list:
    """Benjamini-Hochberg FDR correction."""
    if not p_values: return []
    rejected, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return list(rejected)

def run_diagnostic():
    feed = AlpacaFeed()
    scanner = FactorZooScanner()
    
    # Storage for absolute best stats per feature
    feature_stats = {}
    
    # Aggregate data for all symbols
    for symbol in DIAGNOSTIC_UNIVERSE:
        logger.info(f"Scanning {symbol}...")
        df = feed.get_bars(symbol, start=datetime(2021,1,1, tzinfo=timezone.utc), timeframe="1Day")
        if df.empty: continue
        
        feat_df = scanner.build_feature_matrix(df.loc[TRAIN_START:])
        close = feat_df['close']
        
        for group_name, config in FEATURE_GROUPS.items():
            for feat in config['features']:
                if feat not in feat_df.columns: continue
                
                # Test across windows
                best_icir = -1.0
                best_p = 1.0
                best_win = 0
                
                for win in config['forward_windows']:
                    fwd_ret = close.shift(-win).pct_change(win)
                    valid = feat_df[feat].notna() & fwd_ret.notna()
                    if valid.sum() < 60: continue
                    
                    # Spearman IC
                    from scipy.stats import spearmanr
                    ic, p = spearmanr(feat_df.loc[valid, feat], fwd_ret[valid])
                    
                    # Compute ICIR (Simple Proxy for IC/std)
                    icir = abs(ic) * np.sqrt(valid.sum()) # t-stat proxy
                    
                    if icir > best_icir:
                        best_icir = icir
                        best_p = p
                        best_win = win
                
                # Update global stats for this feature (keeping best across assets)
                if feat not in feature_stats or best_icir > feature_stats[feat]['icir']:
                    feature_stats[feat] = {
                        "group": group_name,
                        "best_window": int(best_win),
                        "ic": float(best_icir / np.sqrt(252)), # scaled back
                        "icir": float(best_icir / 10.0), # normalized for display
                        "p_value": float(best_p)
                    }

    # 2. MULTIPLE TESTING CORRECTION
    p_values = [v['p_value'] for v in feature_stats.values()]
    rejected = apply_fdr_correction(p_values)
    
    for i, (feat, stats) in enumerate(feature_stats.items()):
        stats['passes'] = bool(rejected[i] and stats['icir'] >= ICIR_THRESHOLD)

    # 3. DECISION LOGIC
    fast_passing = [f for f,s in feature_stats.items() if s['group'] == 'fast' and s['passes']]
    slow_passing = [f for f,s in feature_stats.items() if s['group'] == 'slow' and s['passes']]
    macro_passing = [f for f,s in feature_stats.items() if s['group'] == 'macro' and s['passes']]
    
    decision = "STOP"
    if len(slow_passing) >= 2:
        decision = "PROCEED"
        
    # 4. JSON OUTPUT
    results = {
        "fast_features": {f:s for f,s in feature_stats.items() if s['group'] == 'fast'},
        "slow_features": {f:s for f,s in feature_stats.items() if s['group'] == 'slow'},
        "macro_features": {f:s for f,s in feature_stats.items() if s['group'] == 'macro'},
        "decision": decision,
        "fast_passing": len(fast_passing),
        "slow_passing": len(slow_passing),
        "macro_passing": len(macro_passing),
        "config_updates": {
            "factor_zoo.resolution": "1d",
            "factor_zoo.slow_passing_features": slow_passing,
            "factor_zoo.fast_passing_features": fast_passing
        }
    }
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("logs") / f"factor_zoo_diagnostic_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"FINAL DECISION: {decision}")
    logger.info(f"Passing: Fast={len(fast_passing)}, Slow={len(slow_passing)}, Macro={len(macro_passing)}")
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_diagnostic()
