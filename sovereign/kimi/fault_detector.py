"""
Phase 4 — Petroulas Gate (V2.0 — Demoted to Advisory)
6-framework macro scanner. Returns warning signals.
No longer blocks execution — logged for post-trade review only.
"""

from contracts.types import PetrolausDecision, MacroFeatures
from config.loader import params
import math

class PetrolausGate:
    """
    Scans for structural market faults based on macro data.
    """

    def evaluate(self, macro: MacroFeatures, hmm_transition_prob: float) -> PetrolausDecision:
        triggered = []
        p = params['petroulas']

        # 1. YIELD CURVE INVERSION (T10Y - T2Y)
        if macro.yield_curve_slope < p['yield_curve_inversion_threshold']:
            triggered.append('YIELD_CURVE_INVERSION')

        # 2. M2 VELOCITY COLLAPSE
        if macro.m2_velocity < p['m2_velocity_floor']:
            triggered.append('M2_VELOCITY_COLLAPSE')

        # 3. CAPE VALUATION EXTREME
        if not self._is_nan(macro.cape_zscore) and macro.cape_zscore > p['cape_z_extreme']:
            triggered.append('CAPE_EXTREME')

        # 4. EQUITY RISK PREMIUM COMPRESSION
        if macro.erp < p.get('erp_floor', 0.02):
            triggered.append('ERP_COMPRESSION')

        # 5. CREDIT SPREAD SPIKE (HYG)
        if macro.hyg_spread_bps > p['hyg_spread_threshold']:
            triggered.append('HYG_SPREAD_SPIKE')

        # 6. HMM REGIME TRANSITION IMMINENT
        if hmm_transition_prob > p.get('hmm_transition_fault_threshold', 0.80):
            triggered.append('HMM_TRANSITION_IMMINENT')

        # FAULT TRIGGER: 2+ frameworks must agree
        fault_detected = len(triggered) >= p.get('fault_frameworks_required', 2)

        return PetrolausDecision(
            fault_detected=fault_detected,
            fault_reason=f'{len(triggered)} frameworks triggered: {triggered}' if fault_detected else None,
            fault_frameworks=triggered,
            action='HALT' if fault_detected else 'TRADE',
            macro_features=macro
        )

    def _is_nan(self, v) -> bool:
        """Safe NaN check for various types."""
        try:
            return math.isnan(v)
        except (TypeError, ValueError):
            return v is None
