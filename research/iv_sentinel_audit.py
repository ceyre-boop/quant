"""
V4.7 Options Sentinel - IV Term Structure Audit
Tests the Standalone Information Coefficient (IC) of the IV-Inversion 
(VIX9D / VIX) as a Momentum Veto.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_iv_sentinel_audit():
    logger.info("Harvesting IV Term Structure Data (^VIX, ^VIX9D)...")
    
    # 1. Pull VIX term structure (Daily as proxy for 1h session sentiment)
    # yfinance has ^VIX (30d) and ^VIX9D (9d).
    vix = yf.download("^VIX", start="2024-01-01", end="2026-04-10", interval="1d")
    vix9d = yf.download("^VIX9D", start="2024-01-01", end="2026-04-10", interval="1d")

    if vix.empty or vix9d.empty:
        logger.error("Failed to retrieve IV data.")
        return

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(vix9d.columns, pd.MultiIndex):
        vix9d.columns = vix9d.columns.get_level_values(0)
        
    vix.columns = [c.lower() for c in vix.columns]
    vix9d.columns = [c.lower() for c in vix9d.columns]


    df = pd.DataFrame(index=vix.index)
    df['vix'] = vix['close']
    df['vix9d'] = vix9d['close']
    
    # 2. THE IV SENTINEL: Term Structure Inversion
    # Above 1.0 = Short-term panic > Long-term fear (Inversion)
    df['iv_spread'] = df['vix9d'] / df['vix']
    df['iv_velocity'] = df['iv_spread'].pct_change(5) # 1-week velocity
    
    # 3. Target: SPY Forward Returns (1 week)
    spy = yf.download("SPY", start="2024-01-01", end="2026-04-11", interval="1d")
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.lower() for c in spy.columns]

    df['fwd_ret'] = np.log(spy['close'].shift(-5) / spy['close'])
    
    # 4. Standalone IC Test
    valid = df.dropna()
    ic, p_val = stats.spearmanr(valid['iv_spread'], valid['fwd_ret'])
    
    logger.info("========================================")
    logger.info("V4.7 IV SENTINEL STANDALONE AUDIT")
    logger.info("========================================")
    logger.info(f"IV-Spread (VIX9D/VIX) IC: {ic:.4f}")
    logger.info(f"P-Value: {p_val:.1e}")
    logger.info("-" * 40)
    
    # Check conditional returns
    avg_ret = valid['fwd_ret'].mean()
    inverted_ret = valid[valid['iv_spread'] > 1.0]['fwd_ret'].mean()
    normal_ret = valid[valid['iv_spread'] <= 1.0]['fwd_ret'].mean()
    
    logger.info(f"Avg Weekly Return: {avg_ret:.2%}")
    logger.info(f"Avg Ret during IV Inversion (>1.0): {inverted_ret:.2%}")
    logger.info(f"Avg Ret during Normal IV: {normal_ret:.2%}")
    
    if inverted_ret < normal_ret:
        lift = normal_ret - inverted_ret
        logger.info(f"SENTINEL LIFT: {lift:.2%} (Avoidance Edge)")
    
    logger.info("========================================")

if __name__ == "__main__":
    run_iv_sentinel_audit()
