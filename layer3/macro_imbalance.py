"""
Macro Imbalance Framework
Computes three macro-regime stress features.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MacroImbalanceFramework:
    """Computes macro-regime stress features for Layer 1 and Layer 3."""
    
    def __init__(self):
        # Initialize internal models (HMM, PCA baseline, NY Fed Probit)
        self.pca_baseline_mean = 0.0
        self.pca_baseline_std = 1.0
        
    def compute(self, data: dict) -> dict:
        """
        Compute features for live production data.
        Expected keys: vix, bond_yield_10yr, spread_3m_10yr, spy_returns_1m, credit_spread
        """
        vix = data.get('vix', 20.0)
        yield_spread = data.get('spread_3m_10yr', -0.2)
        
        # 1. HMM Stress Probability (Approximation)
        # HMM usually detects "high vol" vs "low vol" regimes
        hmm_stress = 1.0 / (1.0 + np.exp(-(vix - 25) / 5.0))
        
        # 2. PCA Mahalanobis Distance (Approximation)
        # Meaures distance from historical normalcy
        pca_dist = abs(vix - 20) / 10.0 + abs(yield_spread + 0.1) / 0.5
        
        # 3. NY Fed Recession Prob (Estrella-Mishkin 1996)
        # Probit model: prob = NormCDF(-0.533 - 0.633 * (10y-3m spread))
        from scipy.stats import norm
        recession_prob = norm.cdf(-0.533 - 0.633 * yield_spread)
        
        return {
            'hmm_regime_stress': float(hmm_stress),
            'pca_mahalanobis': float(pca_dist),
            'recession_prob_12m': float(recession_prob)
        }
        
    def simulate_macro(self, date_dt: datetime, seed: int = 42) -> dict:
        """
        Deterministic macro simulation for backtests.
        Uses date-seeded randomness to provide consistent but varied macro states.
        """
        import random
        random.seed(int(date_dt.timestamp()) + seed)
        
        # Simulate realistic macro inputs
        sim_data = {
            'vix': 15 + random.random() * 20,
            'spread_3m_10yr': -0.5 + random.random() * 1.5,
            'spy_returns_1m': -0.05 + random.random() * 0.1,
            'credit_spread': 1.0 + random.random() * 2.0
        }
        
        return self.compute(sim_data)
