"""
Phase 5 — Reversion Specialist
"""
from sovereign.specialists.base_specialist import BaseSpecialist
from contracts.types import SovereignFeatureRecord
from config.loader import params

class ReversionSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(regime_label='REVERSION', model_version='reversion_v1')

    def _regime_matches(self, record: SovereignFeatureRecord) -> bool:
        """Reversion gate: mean-reverting regime."""
        return record.regime.hurst_long < params['regime']['hurst_mean_revert_threshold']
