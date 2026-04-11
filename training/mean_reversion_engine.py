"""
Mean Reversion Engine (V2.2 - Institutional Grade)
Implements:
1. Surgical Labeling (Eliminating Hope-Trading)
2. 4-Dimension Feature Engineering (Distance, Velocity, Regime, Exhaustion)
3. EV-Gated Kelly Risk Engine
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- CORE UTILITIES ---

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window).mean()

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Standard ADX calculation for trend strength filtering."""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(df, window)
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/window).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/window).mean()
    return adx

# --- ROOT CAUSE 1: THE REALITY ANCHOR ---

def build_mean_reversion_label(df: pd.DataFrame, forward_bars: int = 15, reversion_threshold: float = 0.6) -> pd.Series:
    """
    V2.2 - REALITY ANCHORED LABELS
    A trade is a TRUE WINNER only if execution on T+1 OPEN results in 60% reversion.
    This eliminates the 'Gap Cheat' where signal close is better than next open.
    """
    mean = df['close'].rolling(50).mean()
    atr = calculate_atr(df, 14)
    
    labels = []
    for i in range(len(df)):
        # Must have T+1 open and forward window
        if i + 1 + forward_bars >= len(df) or pd.isna(mean.iloc[i]) or pd.isna(atr.iloc[i]):
            labels.append(0)
            continue
            
        # THE ANCHOR: Entry at Next Day Open
        entry_p = df['open'].iloc[i+1]
        m = mean.iloc[i] # Target is fixed at signal time
        a = atr.iloc[i]
        dist = m - entry_p
        
        # Verify setup still exists at open
        is_long = dist > 0
        stop_dist = 1.5 * a
        stop_p = (entry_p - stop_dist) if is_long else (entry_p + stop_dist)
        
        # Forward window check starting from T+1
        fwd = df.iloc[i+1 : i+1+forward_bars+1]
        
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


# --- ROOT CAUSE 2: THE WRONG FEATURES (Reality Edition) ---

def build_mean_reversion_features(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    4-Dimension Feature Set with Gap Entropy addition.
    """
    f = pd.DataFrame(index=df.index)
    
    # Gap Entropy (Execution Risk)
    overnight_gap = df['open'] - df['close'].shift(1)
    atr_14 = calculate_atr(df, 14)
    f['gap_entropy'] = overnight_gap.abs() / atr_14
    
    # --- DIMENSION 1: DISTANCE ---
    for window in [20, 50, 100]:
        mean = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        f[f'zscore_{window}'] = (df['close'] - mean) / std
        f[f'pct_from_mean_{window}'] = (df['close'] - mean) / mean
    
    bb_mean = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    f['bb_pct_b'] = (df['close'] - (bb_mean - 2*bb_std)) / (4*bb_std)
    
    # --- DIMENSION 2: VELOCITY ---
    f['roc_5'] = df['close'].pct_change(5)
    f['roc_10'] = df['close'].pct_change(10)
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi_14'] = 100 - (100 / (1 + gain / loss))
    f['rsi_divergence'] = f['rsi_14'] - f['rsi_14'].shift(5)
    
    # --- DIMENSION 3: REGIME ---
    if vix_df is not None:
        # Correctly handle 1D or 2D VIX alignment
        vix_aligned = vix_df['close'].reindex(df.index).ffill()
        if isinstance(vix_aligned, pd.DataFrame):
            vix_aligned = vix_aligned.iloc[:, 0]
        f['vix_level'] = vix_aligned
        f['vix_zscore'] = (f['vix_level'] - f['vix_level'].rolling(60).mean()) / f['vix_level'].rolling(60).std()
    else:
        f['vix_level'] = 18.0
        f['vix_zscore'] = 0.0
        
    atr_pct = atr_14 / df['close']
    f['atr_zscore'] = (atr_pct - atr_pct.rolling(60).mean()) / atr_pct.rolling(60).std()
    f['adx_14'] = calculate_adx(df, 14)
    
    # --- DIMENSION 4: EXHAUSTION ---
    f['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    f['atr_distance'] = (df['close'] - df['close'].rolling(50).mean()) / atr_14
    
    return f


# --- THE RISK/REWARD ENGINE ---

class MeanReversionRiskEngine:
    """
    Inputs: XGBoost probability score, ATR, distance to mean
    Outputs: Position size, stop loss, target, expected value
    """
    def __init__(self, account_size: float = 100000.0, max_risk_per_trade: float = 0.01):
        self.account_size = account_size
        self.max_risk = max_risk_per_trade
        
    def calculate_trade(self, entry_price: float, xgb_prob: float, atr: float, mean_target: float, ict_structural_stop: Optional[float] = None) -> Dict[str, Any]:
        """
        Gate Decision uses conservative 1.5x ATR.
        ICT Structural only tightens sizing if Gate approved.
        """
        # 1. Gate Calculation (ATR Baseline)
        atr_stop_distance = 1.5 * atr
        target_distance = abs(mean_target - entry_price)
        
        rr_ratio_for_gate = target_distance / atr_stop_distance
        prob_loss = 1 - xgb_prob
        ev_for_gate = (xgb_prob * target_distance) - (prob_loss * atr_stop_distance)
        
        # 2. Gate Decision
        take_trade = ev_for_gate > 0 and rr_ratio_for_gate >= 2.0
        
        # 3. Precision Sizing (ICT Structural Improvement)
        if take_trade and ict_structural_stop is not None:
            actual_stop_distance = abs(entry_price - ict_structural_stop)
            # Must be tighter or we reject ICT improvement
            if actual_stop_distance < atr_stop_distance:
                # Tightest valid stop
                pass
            else:
                actual_stop_distance = atr_stop_distance
        else:
            actual_stop_distance = atr_stop_distance
            
        # 4. Final Math
        # Position sizing never exceeds Kelly * 0.5 or self.max_risk
        kelly = xgb_prob - (prob_loss / (target_distance / actual_stop_distance))
        kelly_fraction = max(0, min(kelly * 0.5, 0.02))
        
        risk_dollars = self.account_size * min(kelly_fraction, self.max_risk)
        position_size = risk_dollars / actual_stop_distance if actual_stop_distance > 0 else 0
        
        return {
            'take_trade': take_trade,
            'entry': entry_price,
            'stop': entry_price - actual_stop_distance, # Assuming LONG for logic
            'target': mean_target,
            'rr_actual': target_distance / actual_stop_distance if actual_stop_distance > 0 else 0,
            'xgb_prob': xgb_prob,
            'ev_gate': round(ev_for_gate, 4),
            'position_size': round(position_size, 2)
        }
