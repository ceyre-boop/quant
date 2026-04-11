"""Entry Engine - 12-gate entry logic.

EntryEngine validates all three layers and ICT patterns before generating signals.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd  # Moved from bottom - fixes ICTDetector crash

from contracts.types import (
    Direction, BiasOutput, RiskOutput, GameOutput, RegimeState,
    AccountState, EntrySignal, ThreeLayerContext, FeatureGroup
)
from governance.policy_engine import GOVERNANCE
from layer1.hard_constraints import HardConstraints, ConstraintCheck
from entry_engine.ict_decision_tree import ICTDecisionTree, ICTChecklistResult

logger = logging.getLogger(__name__)


@dataclass
class GateCheck:
    """Result of a single gate check."""
    gate_number: int
    name: str
    passed: bool
    reason: Optional[str] = None


class EntryEngine:
    """12-gate entry validation engine."""
    
    def __init__(self):
        self.constraints = HardConstraints()
        self.ict_tree = ICTDecisionTree()
        self.logger = logging.getLogger(__name__)
    
    def validate_entry(
        self,
        symbol: str,
        context: ThreeLayerContext,
        account: AccountState,
        entry_price: float,
        timestamp: Optional[datetime] = None,
        df: Optional[pd.DataFrame] = None,
        ict_setup: Optional[Dict] = None
    ) -> Optional[EntrySignal]:
        """Validate entry through hard constraints + ICT Decision Tree.
        
        Args:
            symbol: Trading symbol
            context: ThreeLayerContext with all layers
            account: Account state
            entry_price: Proposed entry price
            df: Historical price data (required for structural confirmation)
            ict_setup: Optional ICT pattern detection results
        
        Returns:
            EntrySignal if all gates pass, None otherwise
        """
        # 1. Check Hard Constraints (Safety First)
        # We use the index or a passed timestamp to check Kill Zones and Hours
        check_time = timestamp if timestamp else (df.index[-1] if df is not None and not df.empty else datetime.utcnow())
        
        hard_gates = self._run_hard_gates(symbol, context, account, entry_price, check_time)
        failed_hard = [g for g in hard_gates if not g.passed]
        if failed_hard:
            self.logger.info(f"Entry blocked by Safety Gate {failed_hard[0].gate_number}: {failed_hard[0].reason}")
            return None

        # 2. Level 1: Confidence Check (Pillar 6)
        threshold = GOVERNANCE.parameters['bias_engine']['confidence_threshold']
        if context.bias.confidence < threshold:
            self.logger.info(f"Entry blocked: Confidence {context.bias.confidence:.2f} < {threshold}")
            return None

        # 3. Level 2: Structural ICT Confirmation (The "Timing" piece)
        if df is not None:
            # We use the index or a passed timestamp to check Kill Zones
            check_time = timestamp if timestamp else (df.index[-1] if not df.empty else datetime.utcnow())
            ict_result = self.ict_tree.evaluate(df, context.bias.direction, entry_price, check_time)
            
            if not ict_result.passed:
                self.logger.info(f"Entry blocked by ICT Checklist (Grade: {ict_result.grade}, Score: {ict_result.score:.1f}/10): Missing {ict_result.missing}")
                return None
            
            self.logger.info(f"ICT Confirmation Passed (Grade: {ict_result.grade}, Score: {ict_result.score:.1f}/10): {ict_result.confirmations}")
            # Use ICT-suggested levels if provided
            entry_price = ict_result.entry_price or entry_price
        
        # 4. Final Agreement (Layer Alignment)
        if not context.all_aligned():
            self.logger.info(f"Entry blocked: Layer misalignment - {context.block_reason()}")
            return None
        # ICT Confirmation Passed - create entry signal
        return self._create_entry_signal(symbol, context, entry_price, ict_result)
    
    def _run_hard_gates(
        self,
        symbol: str,
        context: ThreeLayerContext,
        account: AccountState,
        entry_price: float,
        timestamp: datetime
    ) -> List[GateCheck]:
        """Run safety and risk hard gates."""
        gates = []
        
        # Gate 6: Daily loss limit
        gates.append(self._gate_6_daily_loss(context, account))
        
        # Gate 7: Max positions
        gates.append(self._gate_7_max_positions(account))
        
        # Gate 8: Position size limit
        gates.append(self._gate_8_position_size(context, account))
        
        # Gate 9: Trading hours
        gates.append(self._gate_9_trading_hours(timestamp))
        
        return gates
    
    def _gate_1_confidence(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 1: Layer 1 confidence >= 0.55."""
        passed = context.bias.confidence >= 0.55
        return GateCheck(
            gate_number=1,
            name='Layer1_Confidence',
            passed=passed,
            reason=None if passed else f"Confidence {context.bias.confidence:.2f} < 0.55"
        )
    
    def _gate_2_direction(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 2: Bias direction != NEUTRAL."""
        passed = context.bias.direction != Direction.NEUTRAL
        return GateCheck(
            gate_number=2,
            name='Bias_Direction',
            passed=passed,
            reason=None if passed else "Bias is NEUTRAL"
        )
    
    def _gate_3_positive_ev(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 3: Layer 2 positive EV."""
        passed = context.risk.ev_positive
        return GateCheck(
            gate_number=3,
            name='Positive_EV',
            passed=passed,
            reason=None if passed else f"EV {context.risk.expected_value:.4f} <= 0"
        )
    
    def _gate_4_risk_structure(
        self,
        context: ThreeLayerContext,
        entry_price: float
    ) -> GateCheck:
        """Gate 4: Valid risk structure."""
        stop_distance = abs(entry_price - context.risk.stop_price)
        stop_pct = stop_distance / entry_price if entry_price > 0 else 0
        
        # Stop should be within reasonable bounds (0.5% - 5%)
        passed = 0.005 <= stop_pct <= 0.05
        
        return GateCheck(
            gate_number=4,
            name='Risk_Structure',
            passed=passed,
            reason=None if passed else f"Stop distance {stop_pct:.2%} outside bounds"
        )
    
    def _gate_5_game_state(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 5: Layer 3 not EXTREME adversarial (unless aligned)."""
        if context.game.adversarial_risk == 'EXTREME' and not context.game.game_state_aligned:
            passed = False
            reason = "EXTREME adversarial risk without alignment"
        else:
            passed = True
            reason = None
        
        return GateCheck(
            gate_number=5,
            name='Game_State',
            passed=passed,
            reason=reason
        )
    
    def _gate_6_daily_loss(
        self,
        context: ThreeLayerContext,
        account: AccountState
    ) -> GateCheck:
        """Gate 6: Daily loss limit."""
        check = self.constraints.check_daily_loss_limit(account)
        return GateCheck(
            gate_number=6,
            name='Daily_Loss_Limit',
            passed=check.passed,
            reason=check.reason
        )
    
    def _gate_7_max_positions(self, account: AccountState) -> GateCheck:
        """Gate 7: Max positions."""
        check = self.constraints.check_max_positions(account)
        return GateCheck(
            gate_number=7,
            name='Max_Positions',
            passed=check.passed,
            reason=check.reason
        )
    
    def _gate_8_position_size(
        self,
        context: ThreeLayerContext,
        account: AccountState
    ) -> GateCheck:
        """Gate 8: Position size limit."""
        check = self.constraints.check_position_size(context.risk, account)
        return GateCheck(
            gate_number=8,
            name='Position_Size',
            passed=check.passed,
            reason=check.reason
        )
    
    def _gate_9_trading_hours(self, dt: datetime) -> GateCheck:
        """Gate 9: Trading hours."""
        check = self.constraints.check_trading_hours(dt)
        return GateCheck(
            gate_number=9,
            name='Trading_Hours',
            passed=check.passed,
            reason=check.reason
        )
    
    def _gate_10_regime_override(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 10: Regime override check."""
        # Allow entry even with override if confidence is very high
        if context.bias.regime_override and context.bias.confidence < 0.75:
            passed = False
            reason = "Regime override with insufficient confidence"
        else:
            passed = True
            reason = None
        
        return GateCheck(
            gate_number=10,
            name='Regime_Override',
            passed=passed,
            reason=reason
        )
    
    def _gate_11_ict_pattern(self, ict_setup: Optional[Dict]) -> GateCheck:
        """Gate 11: ICT pattern validation."""
        if ict_setup is None:
            # ICT pattern not required
            return GateCheck(
                gate_number=11,
                name='ICT_Pattern',
                passed=True,
                reason=None
            )
        
        # Check if pattern is valid
        pattern_valid = ict_setup.get('valid', False)
        
        return GateCheck(
            gate_number=11,
            name='ICT_Pattern',
            passed=pattern_valid,
            reason=None if pattern_valid else "Invalid ICT pattern"
        )
    
    def _gate_12_agreement(self, context: ThreeLayerContext) -> GateCheck:
        """Gate 12: Final three-layer agreement."""
        passed = context.all_aligned()
        return GateCheck(
            gate_number=12,
            name='Three_Layer_Agreement',
            passed=passed,
            reason=None if passed else context.block_reason()
        )
    
    def _create_entry_signal(
        self,
        symbol: str,
        context: ThreeLayerContext,
        entry_price: float,
        ict_result: Optional[ICTChecklistResult] = None
    ) -> EntrySignal:
        """Create entry signal from validated context and ICT confirmation."""
        # Standard rationale from ICT result
        rationale = ict_result.confirmations if ict_result else ["Manual/Layer Alignment"]
        grade = ict_result.grade if ict_result else "C"
        score = ict_result.score if ict_result else 0.0
        
        # Determine targets and stops
        stop_loss = (ict_result.stop_loss if ict_result and ict_result.stop_loss 
                     else context.risk.stop_price)
        tp1 = (ict_result.take_profit if ict_result and ict_result.take_profit 
               else context.risk.tp1_price)
        
        # Position Scaling based on Grade (Phase 6 Alignment)
        size_multiplier = 1.0
        if grade == "A+":
            size_multiplier = 1.5
        elif grade == "A":
            size_multiplier = 1.0
        elif grade in ["B", "C"]:
            # These were already filtered by 'passed=True/False' in ict_tree,
            # but we set size to 0 for safety if it somehow reached here.
            size_multiplier = 0.0
            
        position_size = context.risk.position_size * size_multiplier
        
        return EntrySignal(
            symbol=symbol,
            direction=context.bias.direction,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=context.risk.tp2_price,
            confidence=context.bias.confidence,
            rationale=rationale,
            timestamp=datetime.utcnow(),
            layer_context=context,
            grade=grade,
            score=score
        )


class ICTDetector:
    """ICT pattern detector for Model D (Opening Range + AMD cycle)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_amd_pattern(
        self,
        ohlcv: pd.DataFrame,
        opening_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """Detect AMD (Accumulation, Manipulation, Distribution) pattern.
        
        Returns:
            Dict with pattern details
        """
        if len(ohlcv) < 5:
            return {'valid': False}
        
        # Get opening range (first N bars)
        opening_bars = ohlcv.head(opening_range_minutes // 5)  # Assuming 5m bars
        
        or_high = opening_bars['high'].max()
        or_low = opening_bars['low'].min()
        or_mid = (or_high + or_low) / 2
        
        # Get subsequent bars
        if len(ohlcv) <= len(opening_bars):
            return {'valid': False}
        
        subsequent = ohlcv.iloc[len(opening_bars):]
        
        # Check for manipulation (break of OR in one direction, then reversal)
        manipulation_up = subsequent['high'].max() > or_high
        manipulation_down = subsequent['low'].min() < or_low
        
        # Check for distribution back into OR
        if manipulation_up:
            # Look for reversal back into range
            valid = subsequent['close'].min() < or_high
            direction = 'short'
        elif manipulation_down:
            valid = subsequent['close'].max() > or_low
            direction = 'long'
        else:
            valid = False
            direction = None
        
        return {
            'valid': valid,
            'pattern': 'AMD',
            'direction': direction,
            'opening_range': {'high': or_high, 'low': or_low, 'mid': or_mid},
            'manipulation_up': manipulation_up,
            'manipulation_down': manipulation_down
        }
    
    def detect_fvg(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 10
    ) -> List[Dict[str, Any]]:
        """Detect Fair Value Gaps (FVGs).
        
        An FVG occurs when:
        - Current bar low > previous bar high (bullish FVG)
        - Current bar high < previous bar low (bearish FVG)
        """
        if len(ohlcv) < 3:
            return []
        
        fvgs = []
        recent = ohlcv.tail(lookback)
        
        for i in range(2, len(recent)):
            prev_high = recent.iloc[i-2]['high']
            prev_low = recent.iloc[i-2]['low']
            
            curr_high = recent.iloc[i]['high']
            curr_low = recent.iloc[i]['low']
            
            # Bullish FVG
            if curr_low > prev_high:
                fvgs.append({
                    'type': 'bullish',
                    'top': curr_low,
                    'bottom': prev_high,
                    'bar_index': i
                })
            
            # Bearish FVG
            elif curr_high < prev_low:
                fvgs.append({
                    'type': 'bearish',
                    'top': prev_low,
                    'bottom': curr_high,
                    'bar_index': i
                })
        
        return fvgs
    
    def detect_order_block(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Detect Order Block (OB).
        
        An OB is the last opposing candle before a strong move.
        """
        if len(ohlcv) < 5:
            return None
        
        recent = ohlcv.tail(lookback)
        
        # Look for strong bullish move
        for i in range(3, len(recent)):
            # Check for 3+ consecutive up bars
            if all(recent.iloc[j]['close'] > recent.iloc[j]['open'] 
                   for j in range(i-3, i)):
                # The last down bar before the move is the bullish OB
                for j in range(i-4, -1, -1):
                    if recent.iloc[j]['close'] < recent.iloc[j]['open']:
                        return {
                            'type': 'bullish',
                            'high': recent.iloc[j]['high'],
                            'low': recent.iloc[j]['low'],
                            'index': j
                        }
        
        # Look for strong bearish move
        for i in range(3, len(recent)):
            if all(recent.iloc[j]['close'] < recent.iloc[j]['open']
                   for j in range(i-3, i)):
                for j in range(i-4, -1, -1):
                    if recent.iloc[j]['close'] > recent.iloc[j]['open']:
                        return {
                            'type': 'bearish',
                            'high': recent.iloc[j]['high'],
                            'low': recent.iloc[j]['low'],
                            'index': j
                        }
        
        return None
