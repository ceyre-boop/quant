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

    def _get_stop_mult(self, regime: str) -> float:
        """Return stop ATR multiplier for the current regime.

        Priority:
          1. params['regime_params'][regime]['stop_atr_mult']  — monthly re-opt result
          2. params['risk']['atr_stop_multiplier']             — global fallback
        """
        regime_params = params.get("regime_params", {})
        regime_block  = regime_params.get(regime, {}) if isinstance(regime_params, dict) else {}
        if regime_block and "stop_atr_mult" in regime_block:
            return float(regime_block["stop_atr_mult"])
        return float(params["risk"].get("atr_stop_multiplier", 1.5))

    def _get_tp_rr(self, regime: str, symbol: str = "") -> float:
        """Return TP reward:risk ratio. Priority: asset > regime > global.

        asset_params[symbol]['atr_target_multiplier'] overrides everything — used
        for high-range movers (GOOGL, GLD, QQQ, TSLA) that need 4.0 to capture
        the full move instead of the global 3.0.
        """
        if symbol:
            asset_params = params.get("asset_params", {})
            asset_block  = asset_params.get(symbol, {}) if isinstance(asset_params, dict) else {}
            if asset_block and "atr_target_multiplier" in asset_block:
                return float(asset_block["atr_target_multiplier"])

        regime_params = params.get("regime_params", {})
        regime_block  = regime_params.get(regime, {}) if isinstance(regime_params, dict) else {}
        if regime_block and "tp_rr" in regime_block:
            return float(regime_block["tp_rr"])
        return float(params["risk"].get("atr_target_multiplier", 3.0))

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
        risk_out = self.base_engine.compute_risk_structure(
            bias=bias,
            regime=router.feature_record.regime,
            risk_pct_override=risk_pct,
            account_equity=account_equity,
            atr=atr,
            entry_price=entry_price
        )

        # 4. APPLY REGIME-SPECIFIC STOP/TP MULTIPLIERS (from monthly re-opt)
        #    Overrides the base engine's generic multipliers when regime_params exist.
        regime_label = router.regime   # "MOMENTUM" | "REVERSION" | "FLAT"
        stop_mult = self._get_stop_mult(regime_label)
        tp_rr     = self._get_tp_rr(regime_label, symbol=symbol)

        if risk_out.ev_positive and risk_out.position_size > 0:
            direction = 1 if bias.direction.value == 1 else -1
            new_stop = entry_price - direction * stop_mult * atr
            new_tp1  = entry_price + direction * stop_mult * atr * tp_rr
            new_tp2  = entry_price + direction * stop_mult * atr * tp_rr * 1.5

            risk_out.stop_price  = new_stop
            risk_out.tp1_price   = new_tp1
            risk_out.tp2_price   = new_tp2
            risk_out.stop_method = f"regime_atr_{stop_mult}x"

        return risk_out
