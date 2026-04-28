"""
Ornstein-Uhlenbeck mean reversion model.

Answers: given the current fair-value z-score, how many days until
the price reverts to fair value?

The OU process: dz = -κ(z - μ) dt + σ dW
  κ (kappa)   = mean reversion speed
  half_life   = ln(2) / κ   (time for deviation to halve)
  μ           = long-run mean (assumed 0 for fair-value z-scores)

This gives a DYNAMICALLY computed hold period instead of a fixed
20- or 60-day constant. The hold period is the physics of the
specific deviation you're in right now.

Used for:
  - Forex: IRP/PPP z-score reversion speed
  - Equity: Hurst-regime pairs (REVERSION regime only)

Reference: Ornstein & Uhlenbeck (1930), Vasicek (1977),
           applied to finance by Avellaneda & Lee (2010).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_HALFLIFE_DAYS  = 3
MAX_HALFLIFE_DAYS  = 120
DEFAULT_HALFLIFE   = 20


@dataclass
class OUPrediction:
    kappa: float            # mean reversion speed (per day)
    half_life_days: float   # ln(2) / kappa
    current_z: float        # current distance from mean
    sigma: float            # residual volatility
    expected_reversion_days: float  # E[time to cross zero]
    confidence: str         # 'HIGH' / 'MEDIUM' / 'LOW'

    def __str__(self) -> str:
        return (f'OU[κ={self.kappa:.4f}  half_life={self.half_life_days:.1f}d  '
                f'z={self.current_z:+.2f}  σ={self.sigma:.4f}  '
                f'reversion_in≈{self.expected_reversion_days:.1f}d  '
                f'confidence={self.confidence}]')


class OUModel:
    """
    Fits an OU process to a z-score time series and predicts reversion time.

    Usage:
        ou = OUModel()
        pred = ou.fit_and_predict(z_score_series, current_z)
        hold_days = pred.half_life_days
    """

    MIN_SERIES_LEN = 63   # need at least 3 months

    def fit_and_predict(
        self,
        z_series: pd.Series,
        current_z: Optional[float] = None,
        min_data_points: int = 63,
    ) -> OUPrediction:
        """
        Fit OU process to z_series and return reversion prediction.

        z_series: time series of fair-value z-scores (zero = fair value)
        current_z: current z-score (defaults to last value in series)
        """
        z = z_series.dropna().values
        if len(z) < min_data_points:
            return self._default_prediction(current_z or (float(z[-1]) if len(z) else 0.0))

        current_z = current_z if current_z is not None else float(z[-1])

        # Estimate κ via OLS regression on lagged z-scores
        # OU discretised: z(t) = α + β·z(t-1) + ε
        # β = exp(-κ·Δt), so κ = -ln(β) / Δt (Δt = 1 day)
        z_lag  = z[:-1]
        z_next = z[1:]

        # OLS
        A = np.vstack([np.ones_like(z_lag), z_lag]).T
        result, _, _, _ = np.linalg.lstsq(A, z_next, rcond=None)
        alpha_hat, beta_hat = result

        # Clamp beta — if >= 1, series is not mean-reverting
        beta_hat = float(np.clip(beta_hat, 0.01, 0.999))
        kappa    = -np.log(beta_hat)   # per day (Δt = 1 business day)

        half_life = float(np.log(2) / kappa)
        half_life = float(np.clip(half_life, MIN_HALFLIFE_DAYS, MAX_HALFLIFE_DAYS))

        # Residual sigma
        z_pred = alpha_hat + beta_hat * z_lag
        resid  = z_next - z_pred
        sigma  = float(np.std(resid))

        # Expected time to reach z=0 from current_z:
        # For OU: E[T_0] ≈ half_life * log(2) / log(2) = half_life
        # More precisely: E[T_0] ≈ |current_z| / (κ · σ) (approximate)
        # Use half_life as the main signal; scale by current z magnitude
        reversion_days = half_life * min(abs(current_z), 3.0)
        reversion_days = float(np.clip(reversion_days, MIN_HALFLIFE_DAYS, MAX_HALFLIFE_DAYS))

        # Confidence: based on how stationary the series looks
        # High: beta < 0.93 (half-life < 10d), good fit
        # Medium: beta < 0.97
        # Low: nearly random walk
        if beta_hat < 0.93:
            confidence = 'HIGH'
        elif beta_hat < 0.97:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return OUPrediction(
            kappa=round(kappa, 6),
            half_life_days=round(half_life, 1),
            current_z=round(current_z, 3),
            sigma=round(sigma, 4),
            expected_reversion_days=round(reversion_days, 1),
            confidence=confidence,
        )

    def fit_from_prices(
        self,
        prices: pd.Series,
        fair_value: pd.Series,
        current_price: Optional[float] = None,
        current_fv: Optional[float] = None,
    ) -> OUPrediction:
        """
        Convenience: compute z-scores from price and fair_value series,
        then fit OU.
        """
        aligned = pd.concat(
            [prices.rename('price'), fair_value.rename('fv')], axis=1
        ).dropna()

        if len(aligned) < self.MIN_SERIES_LEN:
            return self._default_prediction(0.0)

        dev = (aligned['price'] - aligned['fv']) / aligned['fv']
        mu, sigma = dev.mean(), dev.std()
        z_series = (dev - mu) / (sigma + 1e-9)

        current_z = None
        if current_price is not None and current_fv is not None and current_fv != 0:
            raw_dev = (current_price - current_fv) / current_fv
            current_z = (raw_dev - mu) / (sigma + 1e-9)

        return self.fit_and_predict(z_series, current_z)

    @staticmethod
    def _default_prediction(current_z: float) -> OUPrediction:
        return OUPrediction(
            kappa=np.log(2) / DEFAULT_HALFLIFE,
            half_life_days=float(DEFAULT_HALFLIFE),
            current_z=current_z,
            sigma=0.0,
            expected_reversion_days=float(DEFAULT_HALFLIFE),
            confidence='LOW',
        )
