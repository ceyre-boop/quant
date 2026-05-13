"""
Kalman Filter Regime Estimator — CS229 Lecture 19

Bayesian state-space model for continuous regime estimation.
Uses a linear Gaussian state-space model (LGSSM) to track the underlying
regime state from noisy observations of market returns.

CS229 L19 theory applied:
  "The Kalman filter solves the linear-Gaussian state-space model optimally.
   State equation:  z_t = F·z_{t-1} + ε_t,  ε_t ~ N(0, Q)
   Obs. equation:   x_t = H·z_t  + δ_t,    δ_t ~ N(0, R)

   The predict/update equations give us the posterior:
     Predict: z_{t|t-1} = F·z_{t-1|t-1}
              P_{t|t-1} = F·P_{t-1|t-1}·Fᵀ + Q
     Update:  K_t = P_{t|t-1}·Hᵀ·(H·P_{t|t-1}·Hᵀ + R)⁻¹
              z_{t|t} = z_{t|t-1} + K_t·(x_t − H·z_{t|t-1})
              P_{t|t} = (I − K_t·H)·P_{t|t-1}

   The Kalman gain K_t is the optimal Bayesian update weight."

State vector (3-dim):
  z[0] = trend factor        (positive = trending, negative = mean-reverting)
  z[1] = volatility factor   (positive = high vol, negative = calm)
  z[2] = momentum factor     (positive = upward momentum, negative = down)

Observation vector (5-dim):
  x = [EURUSD_ret, GBPUSD_ret, AUDUSD_ret, USDJPY_ret, AUDNZD_ret]

Used by orchestrator at stage 4h:
  KalmanRegimeEstimator.update(observation_vector) → state estimate
  KalmanRegimeEstimator.get_regime_output() → {regime, confidence, trend_factor}
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "kalman_regime.pkl"

_N_STATE = 3   # state dimension
_N_OBS   = 5   # observation dimension


class KalmanRegimeEstimator:
    """
    Linear Gaussian state-space model for continuous regime tracking.

    The state z ∈ ℝ³ tracks trend, volatility, and momentum factors.
    Each call to update() feeds one bar of multi-pair returns and returns
    the posterior state estimate.

    Attributes:
        _z (ndarray): Current state estimate [3]
        _P (ndarray): Current error covariance [3, 3]
    """

    def __init__(
        self,
        process_noise:     float = 0.01,
        observation_noise: float = 0.10,
    ) -> None:
        # State transition: regime is persistent with small random walk
        self._F = np.eye(_N_STATE, dtype=np.float64) * 0.95

        # Observation matrix: each return is a noisy mix of all three factors
        # [trend contributes 0.6×, vol 0.3×, momentum 0.1×] — simplified
        self._H = np.array([
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.5, 0.3, 0.2],
            [-0.4, 0.4, 0.2],   # USDJPY — typically inverse trend
            [0.4, 0.3, 0.3],
        ], dtype=np.float64)

        # Process and observation noise covariances
        self._Q = np.eye(_N_STATE, dtype=np.float64) * process_noise
        self._R = np.eye(_N_OBS,   dtype=np.float64) * observation_noise

        # Prior state: neutral regime
        self._z = np.zeros(_N_STATE, dtype=np.float64)
        self._P = np.eye(_N_STATE, dtype=np.float64)

        self._n_updates = 0
        self._try_load()

    # ── Serialisation ─────────────────────────────────────────────────── #

    def _try_load(self) -> None:
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    state = pickle.load(f)
                self._z = state["z"]
                self._P = state["P"]
                self._n_updates = state.get("n_updates", 0)
                logger.debug(f"[Kalman] Loaded from {_CHECKPOINT} "
                             f"(n_updates={self._n_updates})")
            except Exception as e:
                logger.debug(f"[Kalman] Checkpoint load failed (non-fatal): {e}")

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump({
                    "z": self._z,
                    "P": self._P,
                    "n_updates": self._n_updates,
                }, f)
        except Exception as e:
            logger.debug(f"[Kalman] Checkpoint save failed (non-fatal): {e}")

    # ── Kalman predict / update ─────────────────────────────────────────── #

    def update(self, observation: np.ndarray) -> np.ndarray:
        """
        Feed one bar of observations and return the posterior state estimate.

        CS229 L19 predict-update equations:
          Predict: z̃ = F·z, P̃ = F·P·Fᵀ + Q
          Update:  K = P̃·Hᵀ·(H·P̃·Hᵀ + R)⁻¹
                   z ← z̃ + K·(x − H·z̃)
                   P ← (I − K·H)·P̃

        Args:
            observation: shape (n_obs,) = 5-dim return vector.
                         Missing pairs: pass 0.0 — won't crash.

        Returns:
            Posterior state estimate z [3]: (trend, vol, momentum)
        """
        x = np.asarray(observation, dtype=np.float64)
        if x.shape[0] != _N_OBS:
            # Pad or truncate to match observation dim
            x_full = np.zeros(_N_OBS, dtype=np.float64)
            n = min(x.shape[0], _N_OBS)
            x_full[:n] = x[:n]
            x = x_full

        # ── Predict step ────────────────────────────────────────────── #
        z_pred = self._F @ self._z
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # ── Update step ─────────────────────────────────────────────── #
        S = self._H @ P_pred @ self._H.T + self._R           # innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)            # Kalman gain
        innov = x - self._H @ z_pred                          # innovation
        self._z = z_pred + K @ innov
        self._P = (np.eye(_N_STATE) - K @ self._H) @ P_pred

        self._n_updates += 1
        if self._n_updates % 100 == 0:
            self._save()

        return self._z.copy()

    # ── Regime output ─────────────────────────────────────────────────── #

    def get_regime_output(self) -> Dict[str, Any]:
        """
        Translate the Kalman state vector into regime labels and confidence.

        Returns:
            dict with:
              regime        : 'MOMENTUM' / 'REVERSION' / 'FLAT'
              confidence    : float [0, 1] — certainty of regime classification
              trend_factor  : float — z[0], the raw trend component
              vol_factor    : float — z[1], the vol component
              momentum_factor: float — z[2]
              n_updates     : int
        """
        trend = float(self._z[0])
        vol   = float(self._z[1])

        # Regime classification from state
        if abs(trend) < 0.15:
            regime = "FLAT"
        elif trend > 0:
            regime = "MOMENTUM"
        else:
            regime = "REVERSION"

        # Confidence: how far the state vector is from the neutral boundary
        # Use the normalised magnitude of the trend component
        # Clamp to [0.3, 0.95] — Kalman is always uncertain at the boundaries
        raw_conf = min(1.0, abs(trend) / 0.5)
        confidence = float(max(0.30, min(0.95, raw_conf)))

        # High vol reduces confidence slightly (regime transitions likely)
        if vol > 0.30:
            confidence = float(max(0.30, confidence * (1.0 - 0.2 * min(vol, 1.0))))

        return {
            "regime":          regime,
            "confidence":      confidence,
            "trend_factor":    round(trend, 4),
            "vol_factor":      round(vol, 4),
            "momentum_factor": round(float(self._z[2]), 4),
            "n_updates":       self._n_updates,
        }

    def state_uncertainty(self) -> float:
        """
        Return the trace of the current error covariance P as a scalar
        uncertainty measure. Lower = more confident.
        """
        return float(np.trace(self._P))

    def describe(self) -> str:
        out = self.get_regime_output()
        return (f"KalmanRegimeEstimator regime={out['regime']} "
                f"conf={out['confidence']:.3f} "
                f"trend={out['trend_factor']:+.3f} "
                f"n_updates={self._n_updates}")
