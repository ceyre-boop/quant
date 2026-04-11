"""
ICT Structural Checklist - Entry Decision Tree
Implements ICT (Inner Circle Trader) concepts for high-probability entry confirmation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from contracts.types import Direction

@dataclass
class ICTChecklistResult:
    """Result of the ICT Structural Checklist."""
    passed: bool
    score: float  # 0 to 10.0
    grade: str    # A+, A, B, C
    confirmations: List[str]
    missing: List[str]
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ICTDecisionTree:
    """
    Decision tree to identify high-convince entries based on ICT theory.
    Focuses on: Liquidity -> Displacement -> MSS -> FVG Tap.
    """
    
    def __init__(self, fvg_threshold: float = 0.001):
        self.fvg_threshold = fvg_threshold

    def evaluate(self, 
                 df: pd.DataFrame, 
                 bias_direction: Direction, 
                 current_price: float,
                 timestamp: datetime) -> ICTChecklistResult:
        """
        Evaluate price action with Weighted Confluence Scoring V2.
        Max Score: 10.0.
        """
        confirmations = []
        missing = []
        raw_score = 0.0
        
        # WEIGHTS (V2)
        WEIGHTS = {
            "bias": 2.0,      # XGBoost Anchor
            "sweep": 2.0,     # Institutional Fuel
            "mss": 2.0,       # Structural Intent
            "pd_array": 1.5,  # Value Alignment
            "kill_zone": 1.5, # Timing Window
            "fvg": 1.0        # Precision Reclaim
        }

        # 1. HTF BIAS (XGBoost) - 2.0 points
        raw_score += WEIGHTS["bias"]
        confirmations.append(f"HTF Bias ({WEIGHTS['bias']})")

        # 2. KILL ZONE (Timing) - 1.5 points
        in_kill_zone = self._is_kill_zone(timestamp)
        if in_kill_zone:
            raw_score += WEIGHTS["kill_zone"]
            confirmations.append(f"Kill Zone ({WEIGHTS['kill_zone']})")
        else:
            missing.append("Kill Zone Timing")

        # 3. LIQUIDITY SWEEP (Fuel) - 2.0 points
        swept = self._check_liquidity_sweep(df, bias_direction)
        if swept:
            raw_score += WEIGHTS["sweep"]
            confirmations.append(f"Liquidity Sweep ({WEIGHTS['sweep']})")
        else:
            missing.append("Liquidity Sweep")

        # 4. MSS/CHOCH (Intent) - 2.0 points
        mss_detected, _ = self._detect_mss(df, bias_direction)
        if mss_detected:
            raw_score += WEIGHTS["mss"]
            confirmations.append(f"MSS/CHOCH ({WEIGHTS['mss']})")
        else:
            missing.append("Structural Shift (MSS)")

        # 5. PD ARRAY (Value) - 1.5 points
        in_favorable_zone, _ = self._check_pd_array(df, current_price, bias_direction)
        if in_favorable_zone:
            raw_score += WEIGHTS["pd_array"]
            zone = "Discount" if bias_direction == Direction.LONG else "Premium"
            confirmations.append(f"{zone} Zone ({WEIGHTS['pd_array']})")
        else:
            missing.append("Premium/Discount Alignment")

        # 6. FVG RECLAIM (Precision) - 1.0 point
        fvg_valid, fvg_range = self._detect_refined_fvg(df, bias_direction, current_price)
        if fvg_valid:
            raw_score += WEIGHTS["fvg"]
            confirmations.append(f"FVG Reclaim ({WEIGHTS['fvg']})")
        else:
            missing.append("FVG Imbalance/Reclaim")

        # --- CONVICTION GRADING ---
        # A+ Setup: >= 8.5
        # A Setup: 6.5 - 8.4
        # B Setup: 5.0 - 6.4 (Log Only)
        if raw_score >= 8.5:
            grade = "A+"
            passed = True
        elif raw_score >= 6.5:
            grade = "A"
            passed = True
        elif raw_score >= 5.0:
            grade = "B"
            passed = False # Log only
        else:
            grade = "C"
            passed = False

        stop_loss = self._calculate_structural_stop(df, bias_direction, fvg_range)

        return ICTChecklistResult(
            passed=passed,
            score=raw_score, # 0-10.0 scale now
            grade=grade,
            confirmations=confirmations,
            missing=missing,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=self._calculate_targets(current_price, bias_direction, df)
        )

    def _is_kill_zone(self, dt: datetime) -> bool:
        """London: 2-5 AM ET, NY: 7-10 AM ET or 1:30-4 PM ET"""
        h = dt.hour
        return (2 <= h < 5) or (7 <= h < 10) or (13.5 <= (h + dt.minute/60.0) < 16)

    def _detect_refined_fvg(self, df, direction, price) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Bullish FVG: Gap where Price > Midpoint of the gap."""
        if len(df) < 3: return False, None
        
        if direction == Direction.LONG:
            b1_high = df['high'].iloc[-3]
            b3_low = df['low'].iloc[-1]
            if b3_low > b1_high:
                midpoint = (b1_high + b3_low) / 2
                return price > midpoint, (b1_high, b3_low)
        else:
            b1_low = df['low'].iloc[-3]
            b3_high = df['high'].iloc[-1]
            if b3_high < b1_low:
                midpoint = (b1_low + b3_high) / 2
                return price < midpoint, (b1_low, b3_high)
        return False, None

    def _calculate_structural_stop(self, df, direction, fvg_range):
        if direction == Direction.LONG:
            # Low of the last 10 bars (sweep low)
            base_stop = df['low'].tail(15).min()
            return base_stop * 0.999
        else:
            base_stop = df['high'].tail(15).max()
            return base_stop * 1.001

    def _calculate_targets(self, entry, direction, df):
        # Target next major liquidity level (Phase 7)
        if direction == Direction.LONG:
            return df['high'].tail(50).max()
        else:
            return df['low'].tail(50).min()

    def _check_liquidity_sweep(self, df, direction) -> bool:
        # Simplified: check if current low swept the 10-bar low (SSL) for a buy
        if direction == Direction.LONG:
            recent_low = df['low'].iloc[-1]
            old_low = df['low'].iloc[-15:-5].min()
            return recent_low < old_low
        else:
            recent_high = df['high'].iloc[-1]
            old_high = df['high'].iloc[-15:-5].max()
            return recent_high > old_high

    def _detect_mss(self, df, direction) -> Tuple[bool, float]:
        # Market Structure Shift: Price closes above previous swing high (for long)
        if direction == Direction.LONG:
            swing_high = df['high'].iloc[-10:-2].max()
            if df['close'].iloc[-1] > swing_high:
                return True, swing_high
        else:
            swing_low = df['low'].iloc[-10:-2].min()
            if df['close'].iloc[-1] < swing_low:
                return True, swing_low
        return False, 0.0

    def _check_pd_array(self, df, current_price, direction) -> Tuple[bool, float]:
        # Calculate 50% equilibrium of the recent significant range (20 bars)
        high = df['high'].tail(20).max()
        low = df['low'].tail(20).min()
        equilibrium = (high + low) / 2
        
        range_pos = (current_price - low) / (high - low) if high > low else 0.5
        
        if direction == Direction.LONG:
            # Buy only in Discount (< 50%)
            return current_price < equilibrium, range_pos
        else:
            # Sell only in Premium (> 50%)
            return current_price > equilibrium, range_pos

    def _detect_fvg(self, df, direction) -> Tuple[bool, Tuple[float, float]]:
        # FVG is a gap between High of Bar 1 and Low of Bar 3
        # Look at last 3 bars
        if len(df) < 5: return False, (0,0)
        
        if direction == Direction.LONG:
            # Bullish FVG
            b1_high = df['high'].iloc[-3]
            b3_low = df['low'].iloc[-1]
            if b3_low > b1_high:
                return True, (b1_high, b3_low)
        else:
            # Bearish FVG
            b1_low = df['low'].iloc[-3]
            b3_high = df['high'].iloc[-1]
            if b3_high < b1_low:
                return True, (b3_high, b1_low)
        return False, (0,0)
