"""
Phase 8 — Sovereign Orchestrator (V1.0)
The Master Execution Loop.
Data -> Petroulas -> Router -> Specialist -> Risk -> Ledger -> Broker.
"""

import logging
from datetime import datetime
from contracts.types import SovereignFeatureRecord, VetoRecord, PetrolausDecision
from sovereign.kimi.fault_detector import PetrolausGate
from sovereign.router.regime_router import RegimeRouter
from sovereign.specialists.momentum_specialist import MomentumSpecialist
from sovereign.specialists.reversion_specialist import ReversionSpecialist
from sovereign.risk.kelly_engine import SovereignRiskEngine
from sovereign.ledger.trade_ledger import TradeLedger
from sovereign.ledger.veto_ledger import VetoLedger
from config.loader import params

logger = logging.getLogger(__name__)

class SovereignOrchestrator:
    """
    Coordinates the multi-stage Sovereign trading loop.
    """

    def __init__(self, mode='paper'):
        self.mode = mode # 'paper' or 'live'
        self.petroulas = PetrolausGate()
        self.router = RegimeRouter()
        self.risk = SovereignRiskEngine()
        self.trade_ledger = TradeLedger()
        self.veto_ledger = VetoLedger()
        
        # Specialist Storage
        self.specialists = {
            'momentum': MomentumSpecialist(),
            'reversion': ReversionSpecialist()
        }
        
    def run_session(self, symbol: str, feature_record: SovereignFeatureRecord, 
                    current_price: float, atr: float, equity: float):
        """Processes a single bar/signal through the entire Sovereign stack."""
        logger.info(f"--- SESSION START: {symbol} | {feature_record.timestamp} ---")

        # 1. PETROULAS GATE (Macro Shield)
        # Assuming macro features and transition prob are in the record
        macro_decision = self.petroulas.evaluate(
            feature_record.macro, 
            feature_record.regime.hmm_transition_prob
        )
        
        if macro_decision.fault_detected:
            self.veto_ledger.log_veto(VetoRecord(
                timestamp=feature_record.timestamp,
                symbol=symbol,
                veto_stage='PETROULAS',
                veto_reason=macro_decision.fault_reason
            ))
            logger.warning(f"HALT: Petroulas fault detected. {macro_decision.fault_reason}")
            return

        # 2. REGIME ROUTER
        router_out = self.router.classify(feature_record)
        
        if router_out.regime == 'FLAT':
            self.veto_ledger.log_veto(VetoRecord(
                timestamp=feature_record.timestamp,
                symbol=symbol,
                veto_stage='ROUTER/FLAT',
                veto_reason='Hurst Dead Zone or Noise Prediction'
            ))
            logger.info("SKIP: Router identifies Noise/Flat regime.")
            return

        # 3. SPECIALIST EXECUTION
        specialist = self.specialists.get(router_out.specialist_to_run)
        if not specialist:
            logger.error(f"Error: No specialist found for {router_out.specialist_to_run}")
            return

        bias = specialist.predict(feature_record)
        
        if bias.direction.value == 0: # NEUTRAL
            self.veto_ledger.log_veto(VetoRecord(
                timestamp=feature_record.timestamp,
                symbol=symbol,
                veto_stage='SPECIALIST',
                veto_reason=f"Neutral bias from {router_out.specialist_to_run} ({bias.rationale[0]})"
            ))
            logger.info(f"SKIP: {router_out.specialist_to_run} specialist returns neutral.")
            return

        # 4. RISK & SIZING
        risk_out = self.risk.compute(bias, router_out, equity, atr, current_price)
        
        if risk_out.position_size == 0 or not risk_out.ev_positive:
            reason = risk_out.stop_method if risk_out.stop_method else "Negative EV"
            self.veto_ledger.log_veto(VetoRecord(
                timestamp=feature_record.timestamp,
                symbol=symbol,
                veto_stage='RISK/EV',
                veto_reason=reason
            ))
            logger.warning(f"BLOCK: Risk gate fired. Reason: {reason}")
            return

        # 5. EXECUTION
        logger.info(f"✅ EXECUTION AUTHORIZED: {symbol} {bias.direction} @ {current_price}")
        logger.info(f"Size: {risk_out.position_size} | SL: {risk_out.stop_price} | TP: {risk_out.tp1_price}")

        if self.mode == 'paper':
            self.trade_ledger.log_entry(
                trade_id=f"SVRN_{datetime.utcnow().strftime('%H%M%S')}",
                symbol=symbol,
                direction=str(bias.direction),
                entry_price=current_price,
                size=risk_out.position_size,
                sl=risk_out.stop_price,
                tp=risk_out.tp1_price,
                confidence=bias.confidence
            )
        elif self.mode == 'live':
            # Live execution logic would hook here into broker.submit_order()
            logger.info("Sovereign: Live execution pending broker wiring.")

        return {
            'symbol': symbol,
            'status': 'EXECUTED',
            'p_size': risk_out.position_size,
            'confidence': bias.confidence
        }
