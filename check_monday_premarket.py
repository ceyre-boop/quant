"""
MONDAY PREMARKET SAFETY SCAN (08:45 AM ET)
Checks VIX, ATR and Data Feeds for the Trinity Assets.
Objective: Confirm no-go conditions before the NY Open.
"""

import yfinance as yf
import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# THE SAFETY GATES (Phase 1 Baseline)
VIX_THRESHOLD = 25.0
TRINITY_ASSETS = ['META', 'PFE', 'UNH']
ATR_PCT_GATE = {'META': 4.0, 'PFE': 3.0, 'UNH': 3.0}

def check_safety_gates():
    logger.info("Initializing Premarket Safety Scan (V6.0)...")
    
    # 1. VIX SENTINEL
    vix = yf.download("^VIX", period="1d", interval="1m")
    current_vix = vix['Close'].iloc[-1]
    
    if current_vix > VIX_THRESHOLD:
        logger.error(f"❌ VIX AT {current_vix:.2f} - EXCEEDS STABILITY LIMIT (25). STAY FLAT.")
        return False
    logger.info(f"✅ VIX AT {current_vix:.2f} (SAFE)")

    # 2. ATR HEMORRHAGE CHECK (Using Daily ATR % for safety)
    for ticker in TRINITY_ASSETS:
        df = yf.download(ticker, period="30d", interval="1d")
        # Calc Daily ATR % 
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        atr_pct = (atr / df['Close'].iloc[-1]) * 100
        
        gate = ATR_PCT_GATE.get(ticker, 3.0)
        if atr_pct > gate:
            logger.error(f"❌ {ticker} ATR AT {atr_pct:.2f}% - EXCEEDS GATE ({gate}%). BLOCKED.")
            return False
        logger.info(f"✅ {ticker} ATR AT {atr_pct:.2f}% (PASS)")

    logger.info("============================================")
    logger.info("✅ ALL SAFETY GATES PASSED (NY OPEN GO)")
    logger.info("============================================")
    return True

if __name__ == "__main__":
    if check_safety_gates():
        sys.exit(0)
    else:
        sys.exit(1)
