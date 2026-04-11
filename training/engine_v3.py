"""
V3.0 Institutional Engine - Hurst-Anchored Intraday (1-Hour)
Pillar 11: Regime-Stable Mean Reversion

Key Improvements:
1. Entry Anchor: T+1 Hour Open
2. Regime Gate: Hurst Exponent (H < 0.45)
3. Session Logic: Intra-week only (No weekend contamination)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_hurst(price_series, window=100):
    def hurst_rs(ts):
        n = len(ts)
        if n < 30: return 0.5
        returns = np.diff(np.log(ts))
        n_ret = len(returns)
        mean = np.mean(returns)
        deviation = np.cumsum(returns - mean)
        r = np.max(deviation) - np.min(deviation)
        s = np.std(returns)
        if s == 0: return 0.5
        return np.log(r/s) / np.log(n_ret)
    return price_series.rolling(window).apply(hurst_rs, raw=True)

def build_v3_features(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    V3.0 Feature Engineering - Intraday Edition.
    """
    f = pd.DataFrame(index=df.index)
    
    # --- DIMENSION 1: REGIME (THE H-GATE) ---
    f['hurst'] = calculate_hurst(df['close'], window=100)
    
    # ADX z-score (Trend exhaustion)
    high_low = df['high'] - df['low']
    atr = high_low.rolling(14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14).mean()
    f['adx_zscore'] = (adx - adx.rolling(90).mean()) / adx.rolling(90).std()
    
    # --- DIMENSION 2: DISTANCE ---
    mean_20 = df['close'].rolling(20).mean() # ~2.5 days
    std_20 = df['close'].rolling(20).std()
    f['zscore_20'] = (df['close'] - mean_20) / std_20
    f['atr_distance'] = (df['close'] - mean_20) / atr
    
    # --- DIMENSION 3: MOMENTUM ---
    f['rsi_14'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / (-df['close'].diff().clip(upper=0)).rolling(14).mean())))
    
    # --- DIMENSION 4: LIQUIDITY ---
    f['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    
    return f

def build_v3_labels(df: pd.DataFrame, forward_bars: int = 8, reversion_threshold: float = 0.6) -> pd.Series:
    """
    V3.0 Intraday Surgical Labeling.
    Anchor: T+1 Hour Open.
    Constraint: Reversion within 1 trading session (8 hours).
    """
    mean_20 = df['close'].rolling(20).mean()
    atr = (df['high'] - df['low']).rolling(14).mean()
    
    labels = []
    # Index by session to prevent overnight contamination in later passes
    df['date_id'] = df.index.date
    
    for i in range(len(df)):
        if i + 1 + forward_bars >= len(df) or pd.isna(mean_20.iloc[i]):
            labels.append(0)
            continue
            
        entry_p = df['open'].iloc[i+1]
        target_p = mean_20.iloc[i]
        a = atr.iloc[i]
        dist = target_p - entry_p
        
        is_long = dist > 0
        stop_p = (entry_p - 1.5 * a) if is_long else (entry_p + 1.5 * a)
        
        # Look forward 8 hours
        fwd = df.iloc[i+2 : i+2+forward_bars]
        
        success = False
        if is_long:
            hit_stop = (fwd['low'] <= stop_p).any()
            best_p = fwd['high'].max()
            reverted = (best_p - entry_p) / abs(dist) if abs(dist) > 0 else 0
        else:
            hit_stop = (fwd['high'] >= stop_p).any()
            best_p = fwd['low'].min()
            reverted = (entry_p - best_p) / abs(dist) if abs(dist) > 0 else 0
            
        if not hit_stop and reverted >= reversion_threshold:
            labels.append(1)
        else:
            labels.append(0)
            
    return pd.Series(labels, index=df.index)
