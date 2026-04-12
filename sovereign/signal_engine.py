# sovereign/signal_engine.py
"""
Minimal signal engine.
Input:  ticker + date
Output: {direction, probability, regime, features}

ML model: XGBoost classifier
Trained on: whatever labeled data you have right now
Iterated on: every simulation loop that produces new outcomes
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

@dataclass
class Signal:
    ticker:      str
    date:        str
    direction:   str    # 'long' | 'short' | 'none'
    probability: float  # XGBoost confidence
    regime:      str    # 'momentum' | 'reversion' | 'dead_zone' | 'flat'
    features:    dict   # everything the model saw — for debugging
    entry_price: float  # T+1 open (filled after market open)
    stop:        float
    target:      float
    ev:          float  # expected value — must be > 0 to trade

    def to_dict(self):
        return asdict(self)

class SignalEngine:
    
    # ── REGIME THRESHOLDS ──────────────────────────────────
    HURST_MOMENTUM_FLOOR   = 0.52
    HURST_REVERSION_CEIL   = 0.45
    # 0.45-0.52 = dead zone = no trade regardless of signal
    
    # ── SIGNAL FILTERS ────────────────────────────────────
    MOMENTUM_PROB_FLOOR    = 0.60
    REVERSION_PROB_FLOOR   = 0.50
    GAP_ENTROPY_CEILING    = 0.50
    EV_FLOOR               = 0.0
    RR_FLOOR               = 1.5
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.momentum_model  = None
        self.reversion_model = None
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal feature set. Four dimensions.
        Every feature uses only data available at signal time.
        """
        df = df.copy()
        close = df['close']
        volume = df['volume']
        
        # DIMENSION 1: DISTANCE (where is price vs mean?)
        for w in [20, 50]:
            mean = close.rolling(w).mean()
            std  = close.rolling(w).std().replace(0, np.nan)
            df[f'zscore_{w}'] = (close - mean) / std
        
        bb_mean = close.rolling(20).mean()
        bb_std  = close.rolling(20).std()
        df['bb_pct_b'] = (close - (bb_mean - 2*bb_std)) / (4 * bb_std.replace(0, np.nan))
        
        # DIMENSION 2: VELOCITY (how fast did it get here?)
        df['roc_5']  = close.pct_change(5)
        df['roc_10'] = close.pct_change(10)
        
        # RSI calculation
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs   = gain / loss.replace(0, np.nan)
        df['rsi_14']        = 100 - (100 / (1 + rs))
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(5)
        
        # DIMENSION 3: REGIME (is this environment tradeable?)
        df['hurst']     = self._rolling_hurst(close, window=90)
        df['adx_14']    = self._calculate_adx(df, 14)
        df['atr_14']    = self._calculate_atr(df, 14)
        df['atr_zscore'] = (df['atr_14']/close - (df['atr_14']/close).rolling(60).mean()) \
                           / (df['atr_14']/close).rolling(60).std()
        
        # DIMENSION 4: EXHAUSTION (is the move running out?)
        vol_mean = volume.rolling(20).mean()
        vol_std  = volume.rolling(20).std().replace(0, np.nan)
        df['volume_zscore'] = (volume - vol_mean) / vol_std
        df['atr_distance']  = (close - close.rolling(50).mean()) / df['atr_14'].replace(0, np.nan)
        
        return df
    
    def classify_regime(self, features: dict) -> str:
        h = features.get('hurst', 0.5)
        if h > self.HURST_MOMENTUM_FLOOR:
            return 'momentum'
        elif h < self.HURST_REVERSION_CEIL:
            return 'reversion'
        else:
            return 'dead_zone'  # no trade in dead zone
    
    def generate_signal(self, ticker: str, date: str) -> Signal:
        # Map date str to datetime if needed. DataProvider.get_historical_data uses period/interval
        # We need a custom way to get bars up to a specific date.
        # For MVP, we'll assume we can get enough data and slice it.
        
        df = self.data_provider.get_historical_data(ticker, period="2y", interval="1d")
        if df is None or df.empty:
            return Signal(ticker, date, 'none', 0.0, 'no_data', {}, 0,0,0,0)
            
        # Slice df up to date
        if date in df.index:
            df = df.loc[:date]
        else:
            # Try to find the closest date before
            df = df[df.index <= date]
            
        if len(df) < 100:
            return Signal(ticker, date, 'none', 0.0, 'insufficient_data', {}, 0,0,0,0)
            
        df = self.compute_features(df)
        latest = df.iloc[-1].to_dict()
        
        regime = self.classify_regime(latest)
        
        if regime == 'dead_zone':
            return Signal(ticker, date, 'none', 0.0, regime, latest, 0,0,0,0)
        
        # Gap entropy check
        if len(df) < 2:
            return Signal(ticker, date, 'none', 0.0, 'insufficient_data', latest, 0,0,0,0)
            
        gap = abs(df['open'].iloc[-1] - df['close'].iloc[-2])
        gap_entropy = gap / latest['atr_14'] if latest['atr_14'] > 0 else 99
        if gap_entropy > self.GAP_ENTROPY_CEILING:
            return Signal(ticker, date, 'none', 0.0, 'gap_rejected', latest, 0,0,0,0)
        
        # Get probability from appropriate model
        model = self.momentum_model if regime == 'momentum' else self.reversion_model
        if model is None:
            return Signal(ticker, date, 'none', 0.0, 'no_model', latest, 0,0,0,0)
        
        feature_cols = self._get_feature_cols(regime)
        # Ensure all columns exist in 'latest' and handle NaNs
        X_dict = {col: latest.get(col, 0) for col in feature_cols}
        X = pd.DataFrame([X_dict])[feature_cols].fillna(0)
        
        prob = model.predict_proba(X)[0][1]
        
        # Direction
        if regime == 'momentum':
            direction = 'long' if latest.get('roc_10', 0) > 0 else 'short'
            prob_floor = self.MOMENTUM_PROB_FLOOR
        else:
            direction = 'long' if latest.get('zscore_50', 0) < -2.0 else \
                       'short' if latest.get('zscore_50', 0) > 2.0 else 'none'
            prob_floor = self.REVERSION_PROB_FLOOR
        
        if direction == 'none' or prob < prob_floor:
            return Signal(ticker, date, 'none', prob, regime, latest, 0,0,0,0)
        
        # EV calculation
        entry  = df['close'].iloc[-1]  # Entry at close of signal bar for MVP simulation if T+1 open not available
        # The plan says "entry = df['open'].iloc[-1] # will be T+1 — placeholder until open"
        # Since we are at 'date', the 'open' of 'date' is already known. 
        # But signals are usually generated at close for next day open.
        
        atr    = latest['atr_14']
        stop   = entry - 1.5*atr if direction=='long' else entry + 1.5*atr
        mean_50 = df['close'].rolling(50).mean().iloc[-1]
        target = mean_50 if regime=='reversion' else entry + (3.0*atr if direction=='long' else -3.0*atr)
        
        stop_dist   = abs(entry - stop)
        target_dist = abs(target - entry)
        rr          = target_dist / stop_dist if stop_dist > 0 else 0
        ev          = (prob * target_dist) - ((1-prob) * stop_dist)
        
        if ev <= self.EV_FLOOR or rr < self.RR_FLOOR:
            return Signal(ticker, date, 'none', prob, regime, latest, entry, stop, target, ev)
        
        return Signal(ticker, date, direction, prob, regime, latest, entry, stop, target, ev)
    
    def _rolling_hurst(self, series: pd.Series, window: int = 90) -> pd.Series:
        def hurst_rs(x):
            if len(x) < 20:
                return 0.5
            mean = np.mean(x)
            dev  = np.cumsum(x - mean)
            r    = np.max(dev) - np.min(dev)
            s    = np.std(x)
            if s == 0:
                return 0.5
            return np.log(r/s) / np.log(len(x))
        return series.rolling(window).apply(hurst_rs, raw=True)
    
    def _calculate_atr(self, df, period=14):
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low']  - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_adx(self, df, period=14):
        # Simplified ADX
        tr  = self._calculate_atr(df, period)
        dmp = (df['high'].diff().clip(lower=0))
        dmm = (-df['low'].diff().clip(upper=0))
        dip = (dmp.rolling(period).mean() / tr.replace(0,np.nan)) * 100
        dim = (dmm.rolling(period).mean() / tr.replace(0,np.nan)) * 100
        dx  = ((dip - dim).abs() / (dip + dim).replace(0,np.nan)) * 100
        return dx.rolling(period).mean()
    
    def _get_feature_cols(self, regime: str) -> list:
        base = ['zscore_20','zscore_50','bb_pct_b','roc_5','roc_10',
                'rsi_14','rsi_divergence','volume_zscore','atr_distance',
                'adx_14','atr_zscore','hurst']
        return base
