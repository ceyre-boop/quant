"""
Dynamic RR Engine - Institutional Grade Risk and Trade Management
Implements Asset Profiles, ICT-First Stops, Target Compression, and Multi-Condition Monitoring.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from contracts.types import (
    RiskOutput, BiasOutput, RegimeState, AccountState, MarketData, Direction
)
from governance.policy_engine import GOVERNANCE

logger = logging.getLogger(__name__)

@dataclass
class AssetProfile:
    """Asset-specific behavioral fingerprint."""
    symbol: str
    stop_atr_multiplier: float
    tp_min_r: float
    tp_target_r: float
    trail_activation_r: float
    trail_atr_mult: float
    shock_exit_atr_mult: float
    max_stop_pct: float
    min_stop_atr: float

def build_profiles_from_config() -> Dict[str, AssetProfile]:
    """Factory for profiles based on Pillar 6 config."""
    cfg = GOVERNANCE.parameters['asset_profiles']
    default = cfg['_DEFAULT']
    profiles = {}
    
    for symbol, params in cfg.items():
        # Defensive Coding (Pillar 4): Merge with default
        merged = {**default, **params}
        profiles[symbol] = AssetProfile(symbol=symbol, **merged)
    
    return profiles

ASSET_PROFILES = build_profiles_from_config()

class StopCalculator:
    """ICT-First Stop Logic."""
    
    @staticmethod
    def calculate_stop(
        entry_price: float,
        direction: Direction,
        atr_14: float,
        profile: AssetProfile,
        ict_structure: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calculate stop based on ICT structure, validated by ATR.
        """
        # 1. Structural Stop (ICT Levels)
        # Use levels like 'sweep_low', 'orderblock_low', 'fvg_low'
        levels = []
        if direction == Direction.LONG:
            levels = [
                ict_structure.get('swing_low'),
                ict_structure.get('orderblock_low'),
                ict_structure.get('fvg_low')
            ]
            levels = [l for l in levels if l is not None and l < entry_price]
            structural_stop = max(levels) if levels else entry_price * (1 - profile.max_stop_pct)
        else:
            levels = [
                ict_structure.get('swing_high'),
                ict_structure.get('orderblock_high'),
                ict_structure.get('fvg_high')
            ]
            levels = [l for l in levels if l is not None and l > entry_price]
            structural_stop = min(levels) if levels else entry_price * (1 + profile.max_stop_pct)

        # 2. ATR Validation (Noise Floor)
        stop_dist = abs(entry_price - structural_stop)
        min_stop_dist = atr_14 * profile.min_stop_atr
        
        if stop_dist < min_stop_dist:
            # Widen to noise floor
            if direction == Direction.LONG:
                final_stop = entry_price - min_stop_dist
            else:
                final_stop = entry_price + min_stop_dist
            return final_stop, "ATR_NOISE_FLOOR"
        
        # 3. Max Risk Cap
        max_dist = entry_price * profile.max_stop_pct
        if stop_dist > max_dist:
            if direction == Direction.LONG:
                return entry_price - max_dist, "MAX_RISK_CAP"
            else:
                return entry_price + max_dist, "MAX_RISK_CAP"
                
        return structural_stop, "ICT_STRUCTURE"

class TargetCalculator:
    """ICT-Aware Target Compression."""
    
    @staticmethod
    def calculate_targets(
        entry_price: float,
        stop_price: float,
        direction: Direction,
        profile: AssetProfile,
        ict_structure: Dict[str, Any],
        regime: RegimeState
    ) -> Dict[str, float]:
        risk = abs(entry_price - stop_price)
        
        # Base R Floor
        tp1_r = 1.5
        tp2_r = profile.tp_target_r
        
        # Target Compression (VIX / Macro Crisis)
        multiplier = 1.0
        if regime.volatility.value == 'EXTREME':
            multiplier = 0.7  # 30% compression
            
        # 1. TP1: Look for nearest ICT Liquidity Pool or R Floor
        tp1_limit = ict_structure.get('tp1_target')
        if tp1_limit:
            # Validate if it's at least min_r
            ict_r = abs(tp1_limit - entry_price) / risk
            if ict_r < profile.tp_min_r:
                tp1_price = entry_price + (risk * tp1_r * multiplier) if direction == Direction.LONG else entry_price - (risk * tp1_r * multiplier)
            else:
                tp1_price = tp1_limit
        else:
            tp1_price = entry_price + (risk * tp1_r * multiplier) if direction == Direction.LONG else entry_price - (risk * tp1_r * multiplier)

        # 2. TP2: Runner
        tp2_price = entry_price + (risk * tp2_r * multiplier) if direction == Direction.LONG else entry_price - (risk * tp2_r * multiplier)
        
        return {'tp1': tp1_price, 'tp2': tp2_price, 'risk': risk}

class TradeMonitor:
    """The 10-Condition Priority Engine."""
    
    def __init__(self, entry_price: float, direction: Direction, stop_price: float, tp1: float, tp2: float, profile: AssetProfile):
        self.entry_price = entry_price
        self.direction = direction
        self.stop_price = stop_price
        self.tp1 = tp1
        self.tp2 = tp2
        self.profile = profile
        
        self.highest_since_entry = entry_price if direction == Direction.LONG else -np.inf
        self.lowest_since_entry = entry_price if direction == Direction.SHORT else np.inf
        self.breakeven_active = False
        self.last_atr = 0.0
        self.entry_time = None # Set by monitor when starting
        self.max_mae_usd = 0.0
        self.max_mfe_usd = 0.0
        
    def check_exits(self, current_bar: pd.Series, atr_14: float, timestamp: datetime) -> Tuple[Optional[float], Optional[str]]:
        """Evaluate 10 priority conditions for exit."""
        price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        self.last_atr = atr_14
        
        # Update peaks and forensic excursions (MAE/MFE)
        if self.direction == Direction.LONG:
            self.highest_since_entry = max(self.highest_since_entry, high)
            self.max_mfe_usd = max(self.max_mfe_usd, high - self.entry_price)
            self.max_mae_usd = max(self.max_mae_usd, self.entry_price - low)
        else:
            self.lowest_since_entry = min(self.lowest_since_entry, low)
            self.max_mfe_usd = max(self.max_mfe_usd, self.entry_price - low)
            self.max_mae_usd = max(self.max_mae_usd, high - self.entry_price)

        # 1. HARD STOP (Absolute Priority)
        if self.direction == Direction.LONG and low <= self.stop_price:
            return self.stop_price, "HARD_STOP"
        if self.direction == Direction.SHORT and high >= self.stop_price:
            return self.stop_price, "HARD_STOP"

        # 2. SHOCK CANDLE (Urgency)
        # 2.5 ATR move against us in one bar
        bar_range = abs(current_bar['high'] - current_bar['low'])
        if bar_range > (atr_14 * self.profile.shock_exit_atr_mult):
            move_against = (self.entry_price - price) if self.direction == Direction.LONG else (price - self.entry_price)
            if move_against > atr_14: # and it's moving against us
                return price, "SHOCK_EXIT"

        # 3. TAKE PROFIT (TP1)
        if self.direction == Direction.LONG and high >= self.tp1:
            # We hit TP1, activate Breakeven
            self.breakeven_active = True
            self.stop_price = max(self.stop_price, self.entry_price)
            # Check if we also hit TP2
            if high >= self.tp2:
                return self.tp2, "TAKE_PROFIT_FULL"
            # Otherwise we keep running for TP2
        
        if self.direction == Direction.SHORT and low <= self.tp1:
            self.breakeven_active = True
            self.stop_price = min(self.stop_price, self.entry_price)
            if low <= self.tp2:
                return self.tp2, "TAKE_PROFIT_FULL"

        # 4. TRAILING STOP (After TP1 activation)
        if self.breakeven_active:
            trail_dist = atr_14 * self.profile.trail_atr_mult
            if self.direction == Direction.LONG:
                new_trail = self.highest_since_entry - trail_dist
                self.stop_price = max(self.stop_price, new_trail)
            else:
                new_trail = self.lowest_since_entry + trail_dist
                self.stop_price = min(self.stop_price, new_trail)

        # 5. STAGNATION SHOCK EXIT (V2.2 Hard Constraint)
        # Rule: If t > 24h and MAE > 2.5 ATR and MFE < 1.0R, cut at market
        if self.entry_time:
            hours_held = (timestamp - self.entry_time).total_seconds() / 3600
            if hours_held > 24:
                risk_r = abs(self.entry_price - self.stop_price)
                if (self.max_mae_usd > (atr_14 * 2.5)) and (self.max_mfe_usd < risk_r):
                    return price, "STAGNATION_SHOCK"

        return None, None

class DynamicRREngine:
    """Orchestrator for the Dynamic RR System."""
    
    def __init__(self):
        self.stop_calc = StopCalculator()
        self.target_calc = TargetCalculator()
    
    def get_profile(self, symbol: str) -> AssetProfile:
        return ASSET_PROFILES.get(symbol, ASSET_PROFILES['_DEFAULT'])

    def build_risk_structure(
        self,
        symbol: str,
        entry_price: float,
        direction: Direction,
        market_data: MarketData,
        regime: RegimeState,
        ict_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        profile = self.get_profile(symbol)
        
        stop_price, stop_method = self.stop_calc.calculate_stop(
            entry_price, direction, market_data.atr_14, profile, ict_structure
        )
        
        targets = self.target_calc.calculate_targets(
            entry_price, stop_price, direction, profile, ict_structure, regime
        )
        
        return {
            'stop_price': stop_price,
            'stop_method': stop_method,
            'tp1': targets['tp1'],
            'tp2': targets['tp2'],
            'risk_dist': targets['risk'],
            'profile': profile
        }

class PostTradeAnalyzer:
    """MAE/MFE Feedback Loop."""
    
    @staticmethod
    def analyze(trade_history: pd.DataFrame, entry_price: float, direction: Direction):
        if trade_history.empty:
            return {}
            
        if direction == Direction.LONG:
            mfe = trade_history['high'].max() - entry_price
            mae = entry_price - trade_history['low'].min()
        else:
            mfe = entry_price - trade_history['low'].min()
            mae = trade_history['high'].max() - entry_price
            
        return {
            'mfe_r': mfe / (entry_price * 0.01), # Normalized to 1% move
            'mae_r': mae / (entry_price * 0.01)
        }
