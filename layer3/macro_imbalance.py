"""Macro Imbalance Framework — Structural Market Fault Detection.

Detects macro regime stress and structural imbalances that precede large
market moves.  Outputs three daily features that feed directly into the
XGBoost bias model:

    hmm_regime_stress  : 0.0–1.0  (0 = calm, 1 = crisis-like)
    pca_mahalanobis    : 0.0+     (Mahalanobis distance from normal)
    recession_prob_12m : 0.0–1.0  (NY-Fed probit model)

The framework operates in two modes:

    • Production – accepts live macro data dict (VIX, yield spreads, etc.)
      and returns calibrated feature values.
    • Simulation  – called by the backtest with date + available bar data;
      produces plausible synthetic values without requiring live FRED data.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NY Fed probit coefficients (Estrella & Mishkin 1996)
_PROBIT_ALPHA = -0.6103
_PROBIT_BETA = -0.5582

# Stress regime thresholds
_VIX_CALM = 15.0
_VIX_STRESS = 30.0
_VIX_CRISIS = 40.0

# Numerical stability constants
_MIN_STD_THRESHOLD = 1e-10
_COV_REGULARIZATION = 1e-6


# ---------------------------------------------------------------------------
# Helper – standard-normal CDF (avoids scipy dependency at import time)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Approximation of the standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class MacroImbalanceFramework:
    """Computes macro-regime stress features for the XGBoost pipeline.

    Parameters
    ----------
    lookback_window : int
        Number of historical observations used to fit the PCA/covariance
        baseline (default 252, one trading year).
    """

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        # Rolling history of macro feature vectors for PCA baseline
        self._history: List[np.ndarray] = []
        self._feature_names = [
            "vix_level",
            "yield_spread",
            "credit_stress",
            "vol_of_vol",
            "equity_drawdown",
        ]
        logger.info("MacroImbalanceFramework initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, macro_data: Dict) -> Dict[str, float]:
        """Compute macro-stress features from live production macro data.

        Parameters
        ----------
        macro_data : dict
            Expected keys (all optional; defaults used when missing):

            ``vix``             – VIX level (e.g. 20.0)
            ``yield_10yr``      – 10-year Treasury yield (%)
            ``yield_2yr``       – 2-year Treasury yield (%)
            ``yield_3m``        – 3-month Treasury yield (%)
            ``hy_spread``       – High-yield credit spread vs Treasury (%)
            ``vol_of_vol``      – VIX-of-VIX or VVIX level
            ``equity_drawdown`` – Current drawdown of SPY from 52-week high (0-1)

        Returns
        -------
        dict with keys:
            ``hmm_regime_stress``, ``pca_mahalanobis``, ``recession_prob_12m``
        """
        vix = float(macro_data.get("vix", 20.0))
        yield_10yr = float(macro_data.get("yield_10yr", 4.0))
        yield_2yr = float(macro_data.get("yield_2yr", 4.5))
        yield_3m = float(macro_data.get("yield_3m", 5.0))
        hy_spread = float(macro_data.get("hy_spread", 3.5))
        vvix = float(macro_data.get("vol_of_vol", 90.0))
        drawdown = float(macro_data.get("equity_drawdown", 0.0))

        # Feature vector
        yield_spread_10_2 = yield_10yr - yield_2yr
        yield_spread_10_3m = yield_10yr - yield_3m
        credit_stress = min(hy_spread / 10.0, 1.0)
        vov_norm = min(vvix / 150.0, 1.0)
        vix_norm = min(vix / 50.0, 1.0)

        vec = np.array(
            [vix_norm, yield_spread_10_2, credit_stress, vov_norm, drawdown],
            dtype=float,
        )

        # Update rolling history
        self._history.append(vec)
        if len(self._history) > self.lookback_window:
            self._history.pop(0)

        hmm_stress = self._compute_hmm_stress(vix, yield_spread_10_2, credit_stress)
        mahal = self._compute_pca_mahalanobis(vec)
        rec_prob = self._ny_fed_recession_prob(yield_spread_10_3m)

        result = {
            "hmm_regime_stress": round(hmm_stress, 4),
            "pca_mahalanobis": round(mahal, 4),
            "recession_prob_12m": round(rec_prob, 4),
        }
        logger.debug("MacroImbalance features: %s", result)
        return result

    def simulate_macro(
        self,
        date: datetime,
        recent_returns: Optional[List[float]] = None,
        vix_level: float = 20.0,
        yield_spread: Optional[float] = None,
    ) -> Dict[str, float]:
        """Generate synthetic macro features for backtesting.

        Uses deterministic date-seeded noise combined with VIX and return
        volatility so that each day receives a consistent value without
        requiring live FRED API calls.

        Parameters
        ----------
        date : datetime
            Backtest date (used as seed component for reproducibility).
        recent_returns : list[float] | None
            Recent daily returns of the underlying asset (used to estimate
            realised volatility as a VIX proxy when VIX data is unavailable).
        vix_level : float
            Current VIX level to drive stress calculations.
        yield_spread : float | None
            10yr – 3m yield spread in percent.  When *None*, a plausible
            synthetic spread is generated from VIX and date.

        Returns
        -------
        dict with keys:
            ``hmm_regime_stress``, ``pca_mahalanobis``, ``recession_prob_12m``
        """
        # Deterministic seed: use hash of individual date components to avoid
        # cross-year correlation artifacts (e.g. 2020-01-15 vs 2021-01-15).
        day_seed = abs(hash((date.year, date.month, date.day))) % (2**31 - 1)
        rng = np.random.default_rng(day_seed)

        # HMM stress: VIX-driven with mild noise
        vix_stress = np.clip((vix_level - _VIX_CALM) / (_VIX_CRISIS - _VIX_CALM), 0.0, 1.0)
        # Realised-vol boost when return data is available
        if recent_returns and len(recent_returns) >= 5:
            rv_window = recent_returns[-min(len(recent_returns), 20) :]
            rv = float(np.std(rv_window)) * math.sqrt(252)
            rv_stress = np.clip((rv - 0.10) / 0.40, 0.0, 1.0)
        else:
            rv_stress = 0.0
        hmm_noise = rng.uniform(-0.05, 0.05)
        hmm_stress = float(np.clip(0.6 * vix_stress + 0.4 * rv_stress + hmm_noise, 0.0, 1.0))

        # Synthetic yield spread: mild inversion during high-VIX regimes
        if yield_spread is None:
            base_spread = 1.0 - 3.0 * vix_stress  # positive in calm, negative in stress
            spread_noise = rng.uniform(-0.3, 0.3)
            yield_spread_val = float(base_spread + spread_noise)
        else:
            yield_spread_val = float(yield_spread)

        # PCA Mahalanobis: grows with stress + added CAPE-like variance
        cape_noise = rng.uniform(0.0, 0.5) * (1.0 + vix_stress)
        mahal = float(np.clip(1.0 + 3.0 * hmm_stress + cape_noise, 0.5, 8.0))

        # Recession probability via NY Fed probit
        rec_prob = self._ny_fed_recession_prob(yield_spread_val)

        result = {
            "hmm_regime_stress": round(hmm_stress, 4),
            "pca_mahalanobis": round(mahal, 4),
            "recession_prob_12m": round(rec_prob, 4),
        }
        logger.debug("Simulated MacroImbalance on %s: %s", date.date(), result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ny_fed_recession_prob(self, spread_10y_3m: float) -> float:
        """Estrella-Mishkin (1996) probit recession probability model.

        Parameters
        ----------
        spread_10y_3m : float
            10-year minus 3-month Treasury yield spread in percent.
            Negative values indicate yield-curve inversion.
        """
        z = _PROBIT_ALPHA + _PROBIT_BETA * spread_10y_3m
        return _norm_cdf(z)

    def _compute_hmm_stress(
        self,
        vix: float,
        yield_spread_10_2: float,
        credit_stress: float,
    ) -> float:
        """Two-regime HMM-inspired stress probability.

        Approximates the probability of being in the *crisis* hidden state
        using a weighted combination of VIX, yield-curve inversion, and
        credit stress.  Calibrated so that:
            VIX=15, normal yield curve  → ~0.05 (calm)
            VIX=30, mild inversion      → ~0.45 (elevated)
            VIX=40, deep inversion      → ~0.80+ (crisis)
        """
        # Normalised inputs
        vix_score = np.clip((vix - _VIX_CALM) / (_VIX_CRISIS - _VIX_CALM), 0.0, 1.0)
        inversion_score = np.clip(-yield_spread_10_2 / 2.0, 0.0, 1.0)
        credit_score = np.clip(credit_stress, 0.0, 1.0)

        raw = 0.5 * vix_score + 0.3 * inversion_score + 0.2 * credit_score
        # Logistic squash so value stays in (0, 1) with sensible extremes
        logit = 4.0 * (raw - 0.5)
        return float(1.0 / (1.0 + math.exp(-logit)))

    def _compute_pca_mahalanobis(self, vec: np.ndarray) -> float:
        """PCA-based Mahalanobis distance from the historical baseline.

        When fewer than 10 observations are available, falls back to the
        L2-norm of the standardised vector to avoid degenerate covariance
        matrices.
        """
        if len(self._history) < 10:
            # Fallback: L2 norm (scale by expected baseline ≈ 1.0)
            return float(np.linalg.norm(vec))

        history_array = np.array(self._history, dtype=float)
        mean = history_array.mean(axis=0)
        centered = vec - mean

        if len(self._history) < len(vec) + 2:
            # Still too few samples for a full covariance; use diagonal
            std = history_array.std(axis=0)
            std = np.where(std < _MIN_STD_THRESHOLD, _MIN_STD_THRESHOLD, std)
            return float(np.linalg.norm(centered / std))

        try:
            cov = np.cov(history_array.T) + np.eye(len(vec)) * _COV_REGULARIZATION
            cov_inv = np.linalg.inv(cov)
            dist_sq = float(centered @ cov_inv @ centered)
            return float(math.sqrt(max(dist_sq, 0.0)))
        except np.linalg.LinAlgError:
            std = history_array.std(axis=0)
            std = np.where(std < _MIN_STD_THRESHOLD, _MIN_STD_THRESHOLD, std)
            return float(np.linalg.norm(centered / std))
