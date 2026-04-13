"""
Phase 5 — Momentum Specialist
"""
from sovereign.specialists.base_specialist import BaseSpecialist
from contracts.types import SovereignFeatureRecord
from config.loader import params

class MomentumSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(regime_label='MOMENTUM', model_version='momentum_v1')

    def _regime_matches(self, record: SovereignFeatureRecord) -> bool:
        """Momentum gate: 10yr/90-bar trend integrity."""
        return record.regime.hurst_long > params['regime']['hurst_trending_threshold']
