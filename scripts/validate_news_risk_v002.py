
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contracts.types import (
    SovereignFeatureRecord, RegimeFeatures, MomentumFeatures, 
    MacroFeatures, PetrolausDecision, Direction, Magnitude
)
from clawd_trading.risk.combined_risk import calculate_combined_risk_limits
from clawd_trading.participants import ParticipantLikelihood, ParticipantType
from sovereign.orchestrator import SovereignOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RiskValidation")

def create_mock_record(symbol: str, event_risk: str) -> SovereignFeatureRecord:
    """Creates a mock feature record with specific news risk."""
    return SovereignFeatureRecord(
        symbol=symbol,
        timestamp=datetime.utcnow().isoformat(),
        regime=RegimeFeatures(
            hurst_short=0.55, hurst_long=0.52, hurst_signal='TRENDING',
            csd_score=0.1, csd_signal='NEUTRAL', hmm_state=1,
            hmm_state_label='NORMAL', hmm_confidence=0.9,
            hmm_transition_prob=0.05, adx=28.0, adx_signal='ESTABLISHED'
        ),
        momentum=MomentumFeatures(
            logistic_ode_score=0.7, jt_momentum_12_1=0.02,
            volume_entropy=3.1, rsi_14=58.0, rsi_signal='NEUTRAL'
        ),
        macro=MacroFeatures(
            yield_curve_slope=0.1, yield_curve_velocity=0.01,
            erp=0.05, cape_zscore=1.2, cot_zscore=0.5,
            m2_velocity=1.5, hyg_spread_bps=120, macro_signal='RISK_ON'
        ),
        petroulas=None,
        bar_ohlcv={'close': 1.1000},
        is_valid=True,
        validation_errors=[],
        event_risk=event_risk
    )

def validate_risk_logic():
    logger.info("=== RE-VALIDATING RISK MODEL CORRECTNESS (v002) ===")
    
    symbols = ['EURUSD', 'XAUUSD']
    scenarios = [
        ('CLEAR', 1.0, "expecting full size"),
        ('ELEVATED', 0.6, "expecting 40% reduction"),
        ('HIGH', 0.25, "expecting 75% reduction"),
        ('EXTREME', 0.0, "expecting HARD VETO")
    ]
    
    mock_likelihoods = [
        ParticipantLikelihood(ParticipantType.RETAIL, 0.8, {}),
        ParticipantLikelihood(ParticipantType.NEWS_ALGO, 0.05, {})
    ]
    
    from clawd_trading.risk.regime_risk import RegimeCluster
    regime_cluster = RegimeCluster.QUIET_ACCUMULATION
    
    base_limits = {
        'max_position_size': 100000,
        'max_risk_per_trade_usd': 500,
        'allow_entry': True
    }

    for risk_level, expected_mult, note in scenarios:
        logger.info(f"\nScenario: {risk_level} ({note})")
        
        # Test 1: Combined Risk Logic direct check
        limits = calculate_combined_risk_limits(
            participant_likelihoods=mock_likelihoods,
            regime=regime_cluster,
            base_limits=base_limits,
            event_risk=risk_level
        )
        
        allowed = limits.get('allow_entry', True)
        mult = limits.get('risk_multipliers', {}).get('participant_size', 1.0)
        
        if risk_level == 'EXTREME':
            if not allowed:
                logger.info(f"  ✔ CORRECT: Trade blocked for EXTREME risk. Reason: {limits.get('block_reason')}")
            else:
                logger.error(f"  ✘ FAILED: EXTREME risk did not block trade!")
        else:
            if allowed:
                logger.info(f"  ✔ CORRECT: Trade allowed for {risk_level} risk.")
                # Since participant_size multiplier is what we modified
                # Note: other factors in combined_risk might affect 'combined_size', 
                # but we specifically want to verify the propagation of 'event_risk' into 'participant_size'.
                if abs(mult - expected_mult) < 0.01:
                    logger.info(f"  ✔ CORRECT: Multiplier is {mult} (expected {expected_mult})")
                else:
                    logger.error(f"  ✘ FAILED: Multiplier is {mult} (expected {expected_mult})")
            else:
                logger.error(f"  ✘ FAILED: Trade blocked for {risk_level} risk!")

    # Test 2: Orchestrator Integration Propagation
    logger.info("\n--- Orchestrator Propagation Test ---")
    
    # We mock the calendar to return EXTREME risk
    orchestrator = SovereignOrchestrator(mode='dry_run')
    # Manually overriding calendar response for the test
    from unittest.mock import MagicMock
    orchestrator.calendar.fetch_events = MagicMock(return_value=[])
    orchestrator.calendar.calculate_event_risk = MagicMock(return_value='EXTREME')
    
    # We'll run a mini session logic
    record = create_mock_record('EURUSD', 'EXTREME')
    # _run_symbol_session returns None if vetoed
    result = orchestrator._run_symbol_session(
        symbol='EURUSD',
        feature_record=record,
        current_price=1.10,
        atr=0.01,
        equity=10000
    )
    
    if result is None:
        logger.info("  ✔ CORRECT: Orchestrator vetoed the trade at Stage 1b (Economic Calendar Gate)")
    else:
        logger.error("  ✘ FAILED: Orchestrator failed to veto EXTREME risk!")

    logger.info("\n=== RISK VALIDATION COMPLETE ===")

if __name__ == "__main__":
    try:
        validate_risk_logic()
    except Exception as e:
        logger.error(f"Validation crashed: {e}")
        import traceback
        traceback.print_exc()
