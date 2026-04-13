"""
Phase 5 — Specialists (V1.0)
Wraps Layer 1 BiasEngine with Sovereign Regime Awareness.
"""

from layer1.bias_engine import BiasEngine
from contracts.types import BiasOutput, SovereignFeatureRecord, Direction, Magnitude
from config.loader import params

class BaseSpecialist:
    """
    Base class for Sovereign specialists. Wraps the core BiasEngine.
    """

    def __init__(self, regime_label: str, model_version: str):
        self.regime_label = regime_label
        self.model_version = model_version
        self.engine = BiasEngine()
        self.is_fitted = False
        # Use only fast features for entry logic
        self.feature_names = params['factor_zoo'].get('fast_passing_features', [])

    def _neutral_bias(self, reason: str) -> BiasOutput:
        """Returns a non-trade bias when conditions aren't met."""
        return BiasOutput(
            direction=Direction.NEUTRAL,
            magnitude=Magnitude.SMALL,
            confidence=0.0,
            regime_override=True,
            rationale=[reason],
            model_version=self.model_version,
            feature_snapshot={}
        )

    def train(self, records: list):
        """Train on filtered records matching this regime."""
        filtered = [r for r in records if self._regime_matches(r)]
        if len(filtered) < 200:
            raise ValueError(f'{self.regime_label} specialist: {len(filtered)} samples, need 200+.')
        
        # Pass to core engine for internal XGB/ML fit
        self.engine.train(filtered)
        self.is_fitted = True

    def predict(self, record: SovereignFeatureRecord) -> BiasOutput:
        """Predict bias if regime matches, else return NEUTRAL."""
        if not self.is_fitted:
            raise RuntimeError(f'{self.regime_label} specialist not fitted.')
        
        if not self._regime_matches(record):
            return self._neutral_bias('REGIME_MISMATCH')
            
        return self.engine.get_daily_bias(
            symbol=record.symbol,
            feature_vector=record,
            regime=record.regime
        )

    def _regime_matches(self, record: SovereignFeatureRecord) -> bool:
        """Regime gating logic - implemented by subclasses."""
        raise NotImplementedError

    def save(self, path: str):
        self.engine.save(path)

    def load(self, path: str):
        self.engine.load(path)
        self.is_fitted = True

# --- CONCRETE SPECIALISTS ---

class MomentumSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(regime_label='MOMENTUM', model_version='momentum_v1')

    def _regime_matches(self, record: SovereignFeatureRecord) -> bool:
        """Momentum gate: 10yr/90-bar trend integrity."""
        return record.regime.hurst_long > params['regime']['hurst_trending_threshold']

class ReversionSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(regime_label='REVERSION', model_version='reversion_v1')

    def _regime_matches(self, record: SovereignFeatureRecord) -> bool:
        """Reversion gate: mean-reverting regime."""
        return record.regime.hurst_long < params['regime']['hurst_mean_revert_threshold']
