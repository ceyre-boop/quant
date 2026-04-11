"""
V4.0 Institutional Engine - The Momentum Pivot
Pillar 12: Trend Persistence & Continuation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

class AccumulationODE:
    """Logistic Growth ODE inflection point discovery."""
    def logistic_model(self, t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    def calculate_inflection(self, price_series, window=50):
        def _get_k(ts):
            try:
                t = np.arange(len(ts))
                p0 = [ts.max(), 1, len(ts)/2] # Initial guess
                popt, _ = curve_fit(self.logistic_model, t, ts, p0=p0, maxfev=1000)
                return popt[1] # Growth rate k
            except:
                return 0
        return price_series.rolling(window).apply(_get_k, raw=True)

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

def build_v4_features(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f['hurst'] = calculate_hurst(df['close'], window=100)
    ode = AccumulationODE()
    f['logistic_k'] = ode.calculate_inflection(df['close'], 50)
    
    atr = (df['high'] - df['low']).rolling(14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    f['adx'] = dx.ewm(alpha=1/14).mean()
    
    mean_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    f['zscore_20'] = (df['close'] - mean_20) / std_20
    f['atr_zscore'] = (atr - atr.rolling(90).mean()) / atr.rolling(90).std()

    f = f.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    
    cols_to_clip = ['zscore_20', 'atr_zscore', 'logistic_k']
    for col in cols_to_clip:
        if col in f.columns:
            f[col] = np.clip(f[col], -5, 5)
    return f

def build_v4_labels(df: pd.DataFrame, forward_bars: int = 15, continuation_threshold: float = 0.6) -> pd.Series:
    initial_move = df['close'] - df['close'].shift(5)
    atr = (df['high'] - df['low']).rolling(14).mean()
    labels = []
    
    for i in range(len(df)):
        if i + 1 + forward_bars >= len(df) or pd.isna(initial_move.iloc[i]):
            labels.append(0)
            continue
        entry_p = df['open'].iloc[i+1]
        move_mag = initial_move.iloc[i]
        direction = np.sign(move_mag)
        a = atr.iloc[i]
        target_dist = abs(move_mag) * continuation_threshold
        target_p = entry_p + (direction * target_dist)
        stop_p = entry_p - (direction * a)
        fwd = df.iloc[i+2 : i+2+forward_bars]
        success = False
        for _, bar in fwd.iterrows():
            if direction > 0:
                if bar['low'] <= stop_p: break
                if bar['high'] >= target_p:
                    success = True
                    break
            else:
                if bar['high'] >= stop_p: break
                if bar['low'] <= target_p:
                    success = True
                    break
        labels.append(1 if success else 0)
    return pd.Series(labels, index=df.index)
