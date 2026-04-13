"""
Phase 6 — Risk Engine Wrapper (V1.0)
Grade-based sizing and ATR gates on top of Layer 2 risk math.
"""

from layer2.risk_engine import RiskEngine
from layer2.dynamic_rr_engine import DynamicRREngine
from contracts.types import RiskOutput, BiasOutput, RouterOutput
from config.loader import params

class SovereignRiskEngine:
    """
    Implements Sovereign Grade-based sizing and ATR protection gates.
    Wraps existing Layer 2 risk and dynamic RR logic.
    """

    def __init__(self):
        self.base_engine = RiskEngine()
        self.rr_engine   = DynamicRREngine()

    def _grade_risk_pct(self, confidence: float) -> float:
        """Determines risk-per-trade percentage based on model confidence."""
        g = params['risk']['grade_risk']
        
        if confidence >= 0.92: return g['A_plus']
        if confidence >= 0.78: return g['A']
        if confidence >= 0.65: return g['B']
        return g['C']

    def compute(self, bias: BiasOutput, router: RouterOutput, 
                account_equity: float, atr: float, entry_price: float) -> RiskOutput:
        """
        Calculates final trade parameters including size, stops, and targets.
        Includes ATR safety gate.
        """
        
        # 1. Determine Grade-Based Risk %
        risk_pct = self._grade_risk_pct(bias.confidence)

        # 2. ATR SAFETY GATE
        symbol = router.symbol # RouterOutput contains the symbol
        atr_pct = (atr / entry_price) * 100
        atr_limit = params['atr_gate'].get(symbol, 4.0)
        
        if atr_pct > atr_limit:
            # NO-TRADE: ATR exceeds safety threshold
            return RiskOutput(
                position_size=0.0,
                kelly_fraction=0.0,
                stop_price=0.0,
                stop_method='ATR_GATE_BLOCKED',
                tp1_price=0.0,
                tp2_price=0.0,
                trail_config={'reason': f'ATR {atr_pct:.2f}% > limit {atr_limit}%'},
                expected_value=-1.0,
                ev_positive=False,
                size_breakdown={'block_reason': f'Volatility Hemorrhage Threshold Exceeded'}
            )

        # 3. COMPUTE BASE RISK STRUCTURE (from Layer 2)
        # We pass the Sovereign risk_pct as an override to the internal Kelly calc
        return self.base_engine.compute_risk_structure(
            bias=bias,
            regime=router.feature_record.regime,
            risk_pct_override=risk_pct,
            account_equity=account_equity,
            atr=atr,
            entry_price=entry_price
        )
