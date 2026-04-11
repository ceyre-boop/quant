"""
Dynamic RR Engine V2.2 - Institutional Grade Single-File Specification
Pillars 6 & 7: Modular Risk Infrastructure and Dynamic Coupling.

This module integrates ICT-First stops, Asset-Specific profiles (Fingerprints), 
10-Condition Priority Monitoring, and Forensic Post-Trade Analysis.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

# --- Enums & Contracts ---

class ExitReason(Enum):
    HARD_STOP = "HARD_STOP"
    SHOCK_CANDLE = "SHOCK_CANDLE"
    TAKE_PROFIT_1 = "TP1"
    TAKE_PROFIT_FULL = "TP_FULL"
    TIME_EXIT = "TIME_EXIT"
    STAGNATION_SHOCK = "STAGNATION"
    NEWS_WINDOW = "NEWS_EXIT"
    ICT_INVALIDATION = "ICT_VOID"
    MANUAL_OVERRIDE = "OVERRIDE"
    TRAILED_STOP = "TRAILING"

# --- Module 1: AssetProfile Foundation ---

@dataclass
class AssetProfile:
    """Institutional Behavioral Fingerprint per Asset."""
    symbol: str
    stop_atr_multiplier: float  # Baseline ATR stop
    tp_min_r: float             # Minimum acceptable R for TP1
    tp_target_r: float          # High-probability target R
    trail_activation_r: float   # R-level where trail activates
    trail_atr_mult: float       # Width of trailing stop
    shock_exit_atr_mult: float  # Volatility shock threshold (2.5 ATR proxy)
    max_stop_pct: float         # Absolute dollar risk cap
    min_stop_atr: float         # Noise floor (e.g. 0.75 ATR)

def get_asset_profiles() -> Dict[str, AssetProfile]:
    """Factory for asset fingerprints based on MAE/MFE Discovery."""
    return {
        'ARM': AssetProfile('ARM', 2.0, 1.5, 4.0, 1.5, 2.5, 2.0, 0.05, 0.75),
        'AAPL': AssetProfile('AAPL', 1.5, 1.5, 3.0, 1.2, 1.5, 2.5, 0.02, 0.5),
        'NVDA': AssetProfile('NVDA', 2.5, 2.0, 5.0, 2.0, 3.0, 2.5, 0.05, 1.0),
        'META': AssetProfile('META', 2.2, 1.8, 4.0, 1.5, 2.5, 2.5, 0.04, 0.75),
        'SPY':  AssetProfile('SPY',  1.5, 1.5, 3.0, 1.2, 1.5, 3.0, 0.01, 0.5),
        'QQQ':  AssetProfile('QQQ',  1.5, 1.5, 3.0, 1.2, 1.5, 3.0, 0.01, 0.5),
        '_DEFAULT': AssetProfile('DEFAULT', 1.5, 1.5, 3.0, 1.0, 2.0, 2.5, 0.02, 0.75)
    }

# --- Module 2: StopCalculator (ICT-First) ---

class StopCalculator:
    """Calculates ICT-First stops validated by ATR noise floors."""
    @staticmethod
    def calculate_stop(entry: float, direction: int, atr: float, profile: AssetProfile, ict_structure: Dict[str, float]) -> Tuple[float, str]:
        # 1. ICT Structural Source
        # direction: 1 for Long, -1 for Short
        if direction == 1:
            levels = [l for l in [ict_structure.get('sweep_low'), ict_structure.get('orderblock_low'), ict_structure.get('fvg_low')] if l and l < entry]
            structural_stop = max(levels) if levels else entry - (atr * profile.stop_atr_multiplier)
        else:
            levels = [l for l in [ict_structure.get('sweep_high'), ict_structure.get('orderblock_high'), ict_structure.get('fvg_high')] if l and l > entry]
            structural_stop = min(levels) if levels else entry + (atr * profile.stop_atr_multiplier)
            
        # 2. ATR Noise Floor Validation
        stop_dist = abs(entry - structural_stop)
        min_dist = atr * profile.min_stop_atr
        
        if stop_dist < min_dist:
            final_stop = entry - min_dist if direction == 1 else entry + min_dist
            return final_stop, "ATR_NOISE_FLOOR"
            
        # 3. Max Risk Cap
        max_dist = entry * profile.max_stop_pct
        if stop_dist > max_dist:
            return entry - max_dist if direction == 1 else entry + max_dist, "MAX_RISK_CAP"
            
        return structural_stop, "ICT_STRUCTURE"

# --- Module 3: TargetCalculator ---

class TargetCalculator:
    """Calculates compressed targets based on ICT and Regime Gates."""
    @staticmethod
    def calculate_targets(entry: float, stop: float, direction: int, profile: AssetProfile, ict: Dict[str, float], vix: float) -> Dict[str, float]:
        risk = abs(entry - stop)
        multiplier = 1.0
        
        # Target Compression (VIX Crisis)
        if vix > 30: multiplier = 0.7
        
        # 1. TP1 (ICT Liquidity Pool or R-Floor)
        tp1_limit = ict.get('tp1_target')
        if tp1_limit:
            ict_r = abs(tp1_limit - entry) / risk
            tp1 = tp1_limit if ict_r >= profile.tp_min_r else entry + (direction * risk * profile.tp_min_r)
        else:
            tp1 = entry + (direction * risk * profile.tp_min_r)
            
        # 2. TP2 (Runner)
        tp2 = entry + (direction * risk * profile.tp_target_r * multiplier)
        
        return {'tp1': tp1, 'tp2': tp2, 'risk': risk}

# --- Module 4: TradeMonitor (10-Condition Priority) ---

class TradeMonitor:
    """Institutional Trade Life-Cycle Monitor."""
    def __init__(self, entry_p: float, direction: int, stop_p: float, tp1: float, tp2: float, profile: AssetProfile, entry_time: datetime):
        self.entry_p = entry_p
        self.direction = direction
        self.stop_p = stop_p
        self.tp1 = tp1
        self.tp2 = tp2
        self.profile = profile
        self.entry_time = entry_time
        
        self.highest_mfe = 0.0
        self.lowest_mae = 0.0
        self.trail_active = False
        self.peak_p = entry_p
        
    def check_exit(self, bar: pd.Series, atr: float, ts: datetime, is_news: bool = False) -> Tuple[Optional[float], Optional[ExitReason]]:
        price = bar['close']
        high = bar['high']
        low = bar['low']
        
        # Update Peaks
        if self.direction == 1:
            self.highest_mfe = max(self.highest_mfe, (high - self.entry_p) / self.entry_p)
            self.lowest_mae = max(self.lowest_mae, (self.entry_p - low) / self.entry_p)
            self.peak_p = max(self.peak_p, high)
        else:
            self.highest_mfe = max(self.highest_mfe, (self.entry_p - low) / self.entry_p)
            self.lowest_mae = max(self.lowest_mae, (high - self.entry_p) / self.entry_p)
            self.peak_p = min(self.peak_p, low)

        # PRIORITY 1: HARD STOP
        if (self.direction == 1 and low <= self.stop_p) or (self.direction == -1 and high >= self.stop_p):
            return self.stop_p, ExitReason.HARD_STOP

        # PRIORITY 2: SHOCK CANDLE (2.5 ATR MOVE)
        bar_range = abs(high - low)
        if bar_range > (atr * self.profile.shock_exit_atr_mult):
             against = (self.entry_p - price) if self.direction == 1 else (price - self.entry_p)
             if against > atr: return price, ExitReason.SHOCK_CANDLE

        # PRIORITY 3: NEWS WINDOW
        if is_news: return price, ExitReason.NEWS_WINDOW

        # PRIORITY 4: STAGNATION (24H Efficiency Trap)
        hours = (ts - self.entry_time).total_seconds() / 3600
        if hours > 24 and self.highest_mfe < 0.01: # less than 1% gain in 24h
            return price, ExitReason.STAGNATION_SHOCK

        # PRIORITY 5: TAKE PROFIT
        if (self.direction == 1 and high >= self.tp1): 
            self.trail_active = True
            if high >= self.tp2: return self.tp2, ExitReason.TAKE_PROFIT_FULL
        elif (self.direction == -1 and low <= self.tp1):
            self.trail_active = True
            if low <= self.tp2: return self.tp2, ExitReason.TAKE_PROFIT_FULL

        # PRIORITY 6: TRAILING STOP
        if self.trail_active:
            trail_dist = atr * self.profile.trail_atr_mult
            new_stop = self.peak_p - trail_dist if self.direction == 1 else self.peak_p + trail_dist
            if self.direction == 1: self.stop_p = max(self.stop_p, new_stop)
            else: self.stop_p = min(self.stop_p, new_stop)

        return None, None

# --- Module 5: PostTradeAnalyzer ---

class PostTradeAnalyzer:
    """MAE/MFE Forensic Auditor."""
    @staticmethod
    def analyze_trade(outcome: Dict[str, Any]) -> str:
        mae = outcome['mae']
        mfe = outcome['mfe']
        reason = outcome['exit_reason']
        
        if reason == ExitReason.HARD_STOP and mfe > 0.015:
             return "INSIGHT: STOP TOO TIGHT (Reverted 1.5% before stop hit)"
        if mae < 0.002 and reason == ExitReason.TAKE_PROFIT_FULL:
             return "INSIGHT: PERFECT ENTRY (Almost zero heat)"
        if reason == ExitReason.TIME_EXIT and mfe > 0.02:
             return "INSIGHT: EXIT TOO EARLY (Wasted momentum)"
        return "INSIGHT: STANDARD_OUTCOME"
