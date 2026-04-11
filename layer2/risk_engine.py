"""Layer 2: Quant Risk Model

Risk engine, position sizing, stops, targets, and EV calculations.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from contracts.types import RiskOutput, BiasOutput, RegimeState, AccountState, MarketData
from layer1.feature_builder import FeatureVector
from layer2.dynamic_rr_engine import DynamicRREngine

logger = logging.getLogger(__name__)


class PositionSizing:
    """Position sizing with Kelly criterion."""
    
    def __init__(self, default_risk_pct: float = 0.005):
        self.default_risk_pct = default_risk_pct
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """Calculate Kelly criterion fraction.
        
        Kelly = (p*b - q) / b
        where p = win rate, b = avg_win/avg_loss, q = 1-p
        
        Returns:
            Kelly fraction (capped at 0.25 for safety)
        """
        if avg_loss <= 0:
            return 0.0
        
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b if b > 0 else 0
        
        # Cap Kelly at 25% and floor at 0
        return max(0, min(kelly, 0.25))
    
    def calculate_position_size(
        self,
        equity: float,
        stop_distance: float,
        risk_pct: Optional[float] = None,
        kelly_fraction: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate position size.
        
        Args:
            equity: Account equity
            stop_distance: Stop distance in price units
            risk_pct: Risk percentage per trade
            kelly_fraction: Optional Kelly criterion fraction
        
        Returns:
            Dict with position_size and size_breakdown
        """
        risk_pct = risk_pct or self.default_risk_pct
        
        # Base position size
        risk_amount = equity * risk_pct
        base_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Apply Kelly if provided
        if kelly_fraction is not None:
            kelly_size = equity * kelly_fraction
            final_size = min(base_size, kelly_size)
        else:
            kelly_size = None
            final_size = base_size
        
        return {
            'position_size': final_size,
            'base_size': base_size,
            'kelly_size': kelly_size,
            'risk_amount': risk_amount,
            'multipliers': {
                'kelly_applied': kelly_fraction is not None
            }
        }


class StopCalculator:
    """Calculate stop loss levels."""
    
    def __init__(self):
        self.stop_multipliers = {
            'LOW': 1.0,
            'NORMAL': 1.25,
            'ELEVATED': 1.5,
            'EXTREME': 1.75
        }
    
    def calculate_atr_stop(
        self,
        entry_price: float,
        atr_14: float,
        direction: int,  # 1 for long, -1 for short
        regime: RegimeState,
        atr_multiple: float = 1.5
    ) -> float:
        """Calculate ATR-based stop.
        
        Args:
            entry_price: Entry price
            atr_14: 14-period ATR
            direction: 1 for long, -1 for short
            regime: Current regime
            atr_multiple: ATR multiplier
        
        Returns:
            Stop price
        """
        regime_mult = self.stop_multipliers.get(regime.volatility.value, 1.0)
        adjusted_atr = atr_14 * atr_multiple * regime_mult
        
        if direction == 1:  # Long
            return entry_price - adjusted_atr
        else:  # Short
            return entry_price + adjusted_atr
    
    def calculate_structural_stop(
        self,
        entry_price: float,
        direction: int,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
        atr_14: float = 0
    ) -> float:
        """Calculate structural stop based on swing levels.
        
        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            swing_low: Recent swing low (for longs)
            swing_high: Recent swing high (for shorts)
            atr_14: ATR for buffer calculation
        
        Returns:
            Stop price
        """
        buffer = 0.1 * atr_14
        
        if direction == 1:  # Long
            if swing_low is not None:
                return swing_low - buffer
            else:
                return entry_price * 0.98  # 2% fallback
        else:  # Short
            if swing_high is not None:
                return swing_high + buffer
            else:
                return entry_price * 1.02  # 2% fallback
    
    def calculate_final_stop(
        self,
        entry_price: float,
        direction: int,
        atr_stop: float,
        structural_stop: float
    ) -> Tuple[float, str]:
        """Calculate final stop (most conservative).
        
        Returns:
            (stop_price, stop_method)
        """
        if direction == 1:  # Long - use tighter stop
            if atr_stop > structural_stop:
                return atr_stop, 'atr'
            else:
                return structural_stop, 'structural'
        else:  # Short - use tighter stop (higher price)
            if atr_stop < structural_stop:
                return atr_stop, 'atr'
            else:
                return structural_stop, 'structural'


class TargetCalculator:
    """Calculate profit targets."""
    
    def calculate_targets(
        self,
        entry_price: float,
        stop_price: float,
        direction: int
    ) -> Dict[str, float]:
        """Calculate R-based profit targets.
        
        TP1 = 1R (1:1 risk/reward)
        TP2 = 2R (2:1 risk/reward)
        
        Returns:
            Dict with tp1 and tp2
        """
        risk = abs(entry_price - stop_price)
        
        if direction == 1:  # Long
            tp1 = entry_price + risk
            tp2 = entry_price + 2 * risk
        else:  # Short
            tp1 = entry_price - risk
            tp2 = entry_price - 2 * risk
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'risk_r': risk
        }
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_since_entry: float,
        atr_5: float,
        direction: int
    ) -> float:
        """Calculate trailing stop based on ATR.
        
        Uses the "chandelier exit" method: highest high - 3xATR
        """
        if direction == 1:  # Long
            return max(highest_since_entry - 3 * atr_5, current_price * 0.95)
        else:  # Short
            return min(highest_since_entry + 3 * atr_5, current_price * 1.05)


class ExpectedValueCalculator:
    """Calculate expected value of trades."""
    
    def calculate_ev(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float
    ) -> Dict[str, float]:
        """Calculate expected value in R multiples.
        
        EV = (p_win * avg_win) - (p_loss * avg_loss)
        
        Returns:
            Dict with EV metrics
        """
        p_loss = 1 - win_rate
        
        ev = (win_rate * avg_win_r) - (p_loss * avg_loss_r)
        
        # Edge ratio
        edge_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0
        
        # Minimum win rate for profitability
        breakeven_wr = 1 / (1 + edge_ratio) if edge_ratio > 0 else 0
        
        return {
            'expected_value_r': ev,
            'ev_positive': ev > 0,
            'edge_ratio': edge_ratio,
            'breakeven_win_rate': breakeven_wr,
            'win_rate_advantage': win_rate - breakeven_wr
        }


class RiskEngine:
    """Master risk calculator."""
    
    def __init__(
        self,
        default_risk_pct: float = 0.005,
        default_win_rate: float = 0.55,
        default_avg_win_r: float = 1.5,
        default_avg_loss_r: float = 1.0
    ):
        self.position_sizing = PositionSizing(default_risk_pct)
        self.stop_calculator = StopCalculator()
        self.target_calculator = TargetCalculator()
        self.ev_calculator = ExpectedValueCalculator()
        self.dynamic_engine = DynamicRREngine()
        
        self.default_win_rate = default_win_rate
        self.default_avg_win_r = default_avg_win_r
        self.default_avg_loss_r = default_avg_loss_r
    
    def compute_risk_structure(
        self,
        bias: BiasOutput,
        regime: RegimeState,
        market_data: MarketData,
        account_state: AccountState,
        strategy_params: Optional[Dict] = None,
        ict_result: Optional[Any] = None
    ) -> RiskOutput:
        """Compute complete risk structure.
        
        Args:
            bias: Bias output from Layer 1
            regime: Regime state
            market_data: Current market data
            account_state: Account state
            strategy_params: Optional strategy parameters
        
        Returns:
            RiskOutput with sizing, stops, targets, EV
        """
        # 1. Base Risk Percentage (Adjusted by ICT Grade)
        base_risk = self.position_sizing.default_risk_pct
        grade_multiplier = 1.0
        
        if ict_result:
            score = getattr(ict_result, 'score', 5.0)
            if score >= 8.5: # A+
                grade_multiplier = 1.5
            elif score >= 7.0: # A
                grade_multiplier = 1.0
            elif score >= 5.5: # B
                grade_multiplier = 0.5
            else: # C
                grade_multiplier = 0.25
        
        risk_pct = base_risk * grade_multiplier
        
        # 2. Dynamic RR Engine Call (ICT-First Stops, Asset Profiles)
        ict_structure = {
            'swing_low': getattr(ict_result, 'swing_low', None),
            'swing_high': getattr(ict_result, 'swing_high', None),
            'orderblock_low': getattr(ict_result, 'orderblock_low', None),
            'orderblock_high': getattr(ict_result, 'orderblock_high', None),
            'fvg_low': getattr(ict_result, 'fvg_low', None),
            'fvg_high': getattr(ict_result, 'fvg_high', None),
            'tp1_target': getattr(ict_result, 'liquidity_target', None)
        }
        
        dyn_struct = self.dynamic_engine.build_risk_structure(
            market_data.symbol,
            market_data.current_price,
            bias.direction,
            market_data,
            regime,
            ict_structure
        )
        
        stop_price = dyn_struct['stop_price']
        tp1 = dyn_struct['tp1']
        tp2 = dyn_struct['tp2']
        
        # 3. Position Sizing
        stop_distance = abs(market_data.current_price - stop_price)
        
        # Calculate Kelly fraction
        kelly = self.position_sizing.calculate_kelly_fraction(
            self.default_win_rate,
            self.default_avg_win_r,
            self.default_avg_loss_r
        )
        
        sizing = self.position_sizing.calculate_position_size(
            account_state.equity,
            stop_distance,
            risk_pct=risk_pct,
            kelly_fraction=kelly
        )
        
        # Calculate EV
        ev = self.ev_calculator.calculate_ev(
            self.default_win_rate,
            self.default_avg_win_r,
            self.default_avg_loss_r
        )
        
        return RiskOutput(
            position_size=sizing['position_size'],
            kelly_fraction=kelly,
            stop_price=stop_price,
            stop_method=dyn_struct['stop_method'],
            tp1_price=tp1,
            tp2_price=tp2,
            trail_config={'atr_multiple': dyn_struct['profile'].trail_atr_mult},
            expected_value=ev['expected_value_r'],
            ev_positive=ev['ev_positive'],
            size_breakdown=sizing
        )
