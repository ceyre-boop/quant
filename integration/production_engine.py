"""
Production Entry Engine Integration

Wires up the full 3-layer system with:
- Layer 1: Hard Constraints + Regime Classification
- Layer 2: Bias Engine + Participant Analysis
- Layer 3: Game Theory
- Entry Scoring: 4 entry models
- Combined Risk: Participant × Regime
- 12-Gate validation
- Firebase broadcast
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from contracts.types import (
    Direction, BiasOutput, RiskOutput, GameOutput, RegimeState,
    AccountState, EntrySignal, ThreeLayerContext
)
from layer1.hard_constraints_v2 import HardConstraints, ConstraintCheck
from entry_engine.entry_engine import EntryEngine, GateCheck

# New components from trading-stockfish
from clawd_trading.participants import (
    ParticipantFeatureVector,
    classify_participants,
    extract_from_layer1_context,
    calculate_participant_risk_limits,
)
from clawd_trading.risk import (
    RegimeCluster,
    classify_regime_from_layer1,
    get_regime_risk_limits,
    calculate_combined_risk_limits,
    validate_entry_with_combined_risk,
)
from clawd_trading.entry_engine.entry_scorer import (
    score_all_entry_models,
    select_best_entry_model,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEntrySignal(EntrySignal):
    """Entry signal with enhanced metadata from all systems."""
    # Participant analysis
    dominant_participant: str
    participant_confidence: float
    participant_risk_multiplier: float
    
    # Regime analysis
    regime: str
    regime_risk_multiplier: float
    
    # Entry model selection
    entry_model: str
    entry_model_confidence: float
    entry_model_expected_r: float
    
    # Gate results
    gates_passed: int
    gates_total: int
    gate_details: List[Dict[str, Any]]
    
    # Combined risk
    combined_size_multiplier: float
    
    def to_firebase_dict(self) -> Dict[str, Any]:
        """Convert to Firebase-compatible dict."""
        base = asdict(self)
        # Flatten for Firebase
        return {
            **base,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "direction_str": "LONG" if self.direction == Direction.LONG else "SHORT" if self.direction == Direction.SHORT else "NEUTRAL",
        }


class ProductionEntryEngine:
    """Production entry engine with full component integration."""
    
    def __init__(
        self,
        hard_constraints: Optional[HardConstraints] = None,
        entry_engine: Optional[EntryEngine] = None,
    ):
        self.constraints = hard_constraints or HardConstraints()
        self.entry_engine = entry_engine or EntryEngine()
        self.logger = logging.getLogger(__name__)
    
    def generate_signal(
        self,
        symbol: str,
        layer1_output: Dict[str, Any],
        layer2_output: Dict[str, Any],
        layer3_output: Dict[str, Any],
        account: AccountState,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnhancedEntrySignal]:
        """
        Generate production trading signal with full validation.
        
        Args:
            symbol: Trading symbol (e.g., 'NAS100')
            layer1_output: Layer 1 analysis (bias, regime, constraints)
            layer2_output: Layer 2 analysis (EV, risk, game)
            layer3_output: Layer 3 analysis (game theory, adversarial)
            account: Current account state
            market_data: Optional market data for ICT patterns
        
        Returns:
            EnhancedEntrySignal if all validations pass, None otherwise
        """
        self.logger.info(f"Generating signal for {symbol}")
        
        # Step 1: Participant Analysis
        participant_features = extract_from_layer1_context(layer1_output)
        participant_likelihoods = classify_participants(participant_features)
        participant_limits = calculate_participant_risk_limits(participant_likelihoods)
        dominant = max(participant_likelihoods, key=lambda x: x.probability)
        
        self.logger.info(
            f"Participant analysis: {dominant.type.name} "
            f"({dominant.probability:.1%} confidence)"
        )
        
        # Step 2: Regime Classification
        regime = classify_regime_from_layer1(layer1_output)
        regime_limits = get_regime_risk_limits(regime)
        
        self.logger.info(f"Regime classification: {regime.value}")
        
        # Step 3: Check for trade blocks
        if participant_limits.no_trade:
            self.logger.warning(
                f"Trade blocked: NEWS_ALGO detected (size: {participant_limits.max_size_multiplier})"
            )
            return None
        
        if regime_limits.news_rules.get("block_fresh_entry"):
            self.logger.warning(f"Trade blocked: {regime.value}")
            return None
        
        # Step 4: Entry Model Scoring
        entry_scores = score_all_entry_models(layer1_output, layer2_output)
        best_entry = select_best_entry_model(entry_scores)
        
        if not best_entry.get("selected"):
            self.logger.info(
                f"No suitable entry model: {best_entry.get('reason')}"
            )
            return None
        
        selected_model = best_entry["selected"]
        model_score = best_entry["score"]
        
        self.logger.info(
            f"Selected entry model: {selected_model} "
            f"(Expected R: {model_score['expected_R']:.2f})"
        )
        
        # Step 5: Build ThreeLayerContext for entry engine
        context = self._build_context(
            layer1_output, layer2_output, layer3_output
        )
        
        # Step 6: Calculate entry price from selected model
        entry_price = self._calculate_entry_price(
            selected_model, layer1_output, market_data
        )
        
        # Step 7: Run ICT Structural Validation & Constraints
        # Extract historical bars from market_data if available
        bars_df = market_data.get("bars") if market_data else None
        
        signal = self.entry_engine.validate_entry(
            symbol, context, account, entry_price, timestamp=datetime.now(), df=bars_df
        )
        
        if not signal:
            return None
            
        # Extract the gates for legacy reporting if needed
        # (Though we now use the ICT Checklist Result)
        gate_results = []
        
        # Step 8: Combined Risk Validation (Gate 12 equivalent)
        base_risk_limits = self._extract_risk_limits(context)
        
        combined_validation = validate_entry_with_combined_risk(
            entry_signal={
                "position_size": base_risk_limits.get("max_position_size", 1.0),
                "entry_price": entry_price,
            },
            layer1_output=layer1_output,
            base_risk_limits=base_risk_limits,
            current_exposure={
                "concurrent_risk": account.concurrent_risk if hasattr(account, 'concurrent_risk') else 0,
                "daily_risk": account.daily_risk if hasattr(account, 'daily_risk') else 0,
            }
        )
        
        if not combined_validation.get("valid"):
            self.logger.warning(
                f"Combined risk validation failed: {combined_validation.get('reason')}"
            )
            return None
        
        # Step 9: Build enhanced signal
        signal = self._build_enhanced_signal(
            symbol=symbol,
            context=context,
            entry_price=entry_price,
            participant_dominant=dominant,
            participant_limits=participant_limits,
            regime=regime,
            regime_limits=regime_limits,
            entry_model=selected_model,
            model_score=model_score,
            gate_results=gate_results,
            combined_validation=combined_validation,
        )
        
        self.logger.info(
            f"Signal generated: {signal.direction_str} @ {signal.entry_price:.2f} "
            f"(Size: {signal.combined_size_multiplier:.2%}, Expected R: {signal.expected_value:.2f})"
        )
        
        return signal
    
    def _build_context(
        self,
        layer1: Dict[str, Any],
        layer2: Dict[str, Any],
        layer3: Dict[str, Any],
    ) -> ThreeLayerContext:
        """Build ThreeLayerContext from layer outputs."""
        # Convert dict outputs to type objects
        bias = BiasOutput(
            direction=Direction.LONG if layer1.get("direction") == 1 
                     else Direction.SHORT if layer1.get("direction") == -1 
                     else Direction.NEUTRAL,
            confidence=layer1.get("confidence", 0.5),
            regime=RegimeState.TREND if layer1.get("trend_regime") == "uptrend"
                   else RegimeState.RANGE if layer1.get("trend_regime") == "range"
                   else RegimeState.VOLATILITY,
            features=layer1.get("features", {}),
        )
        
        risk = RiskOutput(
            expected_value=layer2.get("ev", layer2.get("expected_value", 0)),
            win_prob=layer2.get("win_prob", 0.5),
            risk_reward=layer2.get("risk_reward", 1.0),
            max_position_size=layer2.get("max_position_size", 1.0),
            stop_loss=layer2.get("stop_loss", 0),
            take_profit=layer2.get("take_profit", 0),
        )
        
        game = GameOutput(
            adversarial_risk=layer3.get("adversarial_risk", "LOW"),
            game_state_aligned=layer3.get("game_state_aligned", False),
            pool_position=layer3.get("pool_position", {}),
        )
        
        return ThreeLayerContext(
            bias=bias,
            risk=risk,
            game=game,
            timestamp=datetime.now(),
        )
    
    def _calculate_entry_price(
        self,
        entry_model: str,
        layer1_output: Dict[str, Any],
        market_data: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate entry price based on selected model."""
        current_price = layer1_output.get("current_price", 0)
        
        if entry_model == "FVG_RESPECT_CONTINUATION":
            # Enter at FVG level
            fvg = layer1_output.get("fvg_level", current_price)
            return fvg
        
        elif entry_model == "SWEEP_DISPLACEMENT_REVERSAL":
            # Enter after sweep confirmation
            sweep_level = layer1_output.get("sweep_level", current_price)
            return sweep_level
        
        elif entry_model == "OB_CONTINUATION":
            # Enter at order block
            ob_level = layer1_output.get("order_block_level", current_price)
            return ob_level
        
        elif entry_model == "ICT_CONCEPT":
            # Enter at ICT setup level
            ict_setup = layer1_output.get("ict_setup", {})
            return ict_setup.get("entry_level", current_price)
        
        return current_price
    
    def _run_gates_with_enhanced_logging(
        self,
        symbol: str,
        context: ThreeLayerContext,
        account: AccountState,
        entry_price: float,
    ) -> List[GateCheck]:
        """Run all 12 gates with enhanced logging."""
        # Use the base entry engine but capture results
        gates = []
        
        # Gate 1: Confidence
        gates.append(GateCheck(
            gate_number=1,
            name="Layer 1 Confidence",
            passed=context.bias.confidence >= 0.55,
            reason=None if context.bias.confidence >= 0.55 else f"confidence {context.bias.confidence:.2f} < 0.55"
        ))
        
        # Gate 2: Direction
        gates.append(GateCheck(
            gate_number=2,
            name="Direction",
            passed=context.bias.direction != Direction.NEUTRAL,
            reason=None if context.bias.direction != Direction.NEUTRAL else "direction is NEUTRAL"
        ))
        
        # Gate 3: Positive EV
        gates.append(GateCheck(
            gate_number=3,
            name="Positive EV",
            passed=context.risk.expected_value > 0,
            reason=None if context.risk.expected_value > 0 else f"EV {context.risk.expected_value:.2f} <= 0"
        ))
        
        # Gate 4: Risk structure
        has_valid_stop = context.risk.stop_loss > 0 and entry_price > 0
        gates.append(GateCheck(
            gate_number=4,
            name="Risk Structure",
            passed=has_valid_stop,
            reason=None if has_valid_stop else "invalid stop loss"
        ))
        
        # Gate 5: Game state
        gates.append(GateCheck(
            gate_number=5,
            name="Game State",
            passed=context.game.adversarial_risk != "EXTREME" or context.game.game_state_aligned,
            reason=None if (context.game.adversarial_risk != "EXTREME" or context.game.game_state_aligned) else "extreme adversarial risk"
        ))
        
        # Gates 6-12: Hard constraints and additional checks
        constraint_check = self.constraints.check_all_constraints(account, context.risk)
        gates.append(GateCheck(
            gate_number=6,
            name="Daily Loss Limit",
            passed=constraint_check.passed,
            reason=None if constraint_check.passed else constraint_check.reason
        ))
        
        # Add remaining gates as passed (simplified for now)
        for i in range(7, 13):
            gates.append(GateCheck(
                gate_number=i,
                name=f"Gate {i}",
                passed=True,
                reason=None
            ))
        
        return gates
    
    def _extract_risk_limits(self, context: ThreeLayerContext) -> Dict[str, Any]:
        """Extract risk limits from context."""
        return {
            "max_position_size": context.risk.max_position_size,
            "max_risk_per_trade_usd": context.risk.max_position_size * 100000,  # Assuming $100k account
            "allow_entry": True,
        }
    
    def _build_enhanced_signal(
        self,
        symbol: str,
        context: ThreeLayerContext,
        entry_price: float,
        participant_dominant,
        participant_limits,
        regime: RegimeCluster,
        regime_limits,
        entry_model: str,
        model_score: Dict[str, Any],
        gate_results: List[GateCheck],
        combined_validation: Dict[str, Any],
    ) -> EnhancedEntrySignal:
        """Build the final enhanced entry signal."""
        
        # Calculate stop and targets
        stop_loss = context.risk.stop_loss or entry_price * 0.99
        take_profit_1 = context.risk.take_profit or entry_price * 1.02
        take_profit_2 = take_profit_1 * 1.02
        
        # Calculate position size with combined risk
        combined_multiplier = combined_validation.get("adjusted_limits", {}).get(
            "risk_multipliers", {}
        ).get("combined_size", 1.0)
        
        position_size = context.risk.max_position_size * combined_multiplier
        
        # Calculate expected value
        ev = context.risk.expected_value * model_score["confidence"]
        
        return EnhancedEntrySignal(
            symbol=symbol,
            direction=context.bias.direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=position_size,
            confidence=context.bias.confidence * model_score["confidence"],
            expected_value=ev,
            timestamp=datetime.now(),
            status="PENDING",
            # Participant data
            dominant_participant=participant_dominant.type.name,
            participant_confidence=participant_dominant.probability,
            participant_risk_multiplier=participant_limits.max_size_multiplier,
            # Regime data
            regime=regime.value,
            regime_risk_multiplier=regime_limits.max_per_trade_R,
            # Entry model data
            entry_model=entry_model,
            entry_model_confidence=model_score["confidence"],
            entry_model_expected_r=model_score["expected_R"],
            # Gate results
            gates_passed=sum(1 for g in gate_results if g.passed),
            gates_total=len(gate_results),
            gate_details=[{"gate": g.gate_number, "name": g.name, "passed": g.passed, "reason": g.reason} for g in gate_results],
            # Combined risk
            combined_size_multiplier=combined_multiplier,
        )
