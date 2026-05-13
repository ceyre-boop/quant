"""
Softmax Regime Classifier — CS229 Lecture 04

Multinomial logistic regression for 3-class regime detection.
Outputs a probability VECTOR, not a hard label, so downstream consumers
(Kalman, KMeans, Kelly) retain the uncertainty information.

Lecture 04 theory applied:
  "Softmax regression generalises logistic regression to K classes.
   The hypothesis is: h(x) = softmax(θᵀx) where the output is a
   probability distribution over {MOMENTUM, REVERSION, FLAT}.
   We maximise the log-likelihood of the multinomial distribution."

Online SGD update:
  CS229 L03: "Mini-batch SGD with step size α. On each new trade close
  we have a ground-truth regime label from the HMM — one labelled
  example is enough to do one gradient step."

Used by orchestrator:
  - predict_proba(features_dict) → dict[regime → probability]
  - encode(hurst, hmm_prob, adx, prev_regime, strategy) → np.ndarray
  - update_online(x, regime_label, alpha) → gradient step
  - fit(X, y) → batch fit from ledger on cold start
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "softmax_regime.pkl"

REGIMES = ["MOMENTUM", "REVERSION", "FLAT"]
_REGIME_TO_IDX = {r: i for i, r in enumerate(REGIMES)}

_STRATEGY_MAP: Dict[str, float] = {
    "momentum":           1.0,
    "momentum_sma":       1.0,
    "donchian":           1.0,
    "donchian_breakout":  1.0,
    "reversion":         -1.0,
    "bb_reversion":      -1.0,
    "mean_reversion":    -1.0,
    "atr_channel":        0.0,
    "fvg_fill":           0.0,
    "liquidity_sweep":    0.5,
}
_PREV_REGIME_MAP: Dict[str, float] = {
    "MOMENTUM":  1.0,
    "REVERSION": -1.0,
    "FLAT":       0.0,
}

N_FEATURES = 5


class SoftmaxRegimeClassifier:
    """
    Multinomial logistic regression (softmax) — 3-class regime classifier.

    Feature vector (5 dims):
        [hurst_short, hmm_transition_prob, adx_norm, prev_regime_enc, strategy_enc]

    Attributes:
        _fitted (bool): True once at least one fit() or update_online() has run
                        with ≥10 examples.
        _W (ndarray): weight matrix [K × d], K=3 classes, d=N_FEATURES
        _b (ndarray): bias vector [K]
    """

    def __init__(self, C: float = 1.0) -> None:
        self._C = C          # L2 regularisation strength (inverse of lambda)
        self._W = np.zeros((3, N_FEATURES), dtype=np.float64)
        self._b = np.zeros(3, dtype=np.float64)
        self._fitted = False
        self._n_updates = 0
        self._scaler_mean = np.zeros(N_FEATURES, dtype=np.float64)
        self._scaler_std  = np.ones(N_FEATURES, dtype=np.float64)
        self._try_load()

    # ── Serialisation ─────────────────────────────────────────────────── #

    def _try_load(self) -> None:
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    state = pickle.load(f)
                self._W = state["W"]
                self._b = state["b"]
                self._fitted = state["fitted"]
                self._n_updates = state.get("n_updates", 0)
                self._scaler_mean = state.get("scaler_mean", np.zeros(N_FEATURES))
                self._scaler_std  = state.get("scaler_std",  np.ones(N_FEATURES))
                logger.debug(f"[SoftmaxRegime] Loaded from {_CHECKPOINT} "
                             f"(n_updates={self._n_updates})")
            except Exception as e:
                logger.debug(f"[SoftmaxRegime] Checkpoint load failed (non-fatal): {e}")

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump({
                    "W": self._W,
                    "b": self._b,
                    "fitted": self._fitted,
                    "n_updates": self._n_updates,
                    "scaler_mean": self._scaler_mean,
                    "scaler_std":  self._scaler_std,
                }, f)
        except Exception as e:
            logger.debug(f"[SoftmaxRegime] Checkpoint save failed (non-fatal): {e}")

    # ── Feature engineering ───────────────────────────────────────────── #

    @staticmethod
    def encode(
        hurst: float,
        hmm_prob: float,
        adx: float,
        prev_regime: str,
        strategy: str,
    ) -> np.ndarray:
        """
        Encode scalar inputs into the 5-dim feature vector.

        Returns a float64 array of shape (5,).
        """
        return np.array([
            float(hurst),
            float(hmm_prob),
            float(adx) / 50.0,                                    # normalise to ~[0, 1]
            float(_PREV_REGIME_MAP.get(prev_regime, 0.0)),
            float(_STRATEGY_MAP.get(strategy, 0.0)),
        ], dtype=np.float64)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self._scaler_mean) / (self._scaler_std + 1e-9)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    # ── Batch fit (cold-start from ledger) ───────────────────────────── #

    def fit(self, X: np.ndarray, y) -> "SoftmaxRegimeClassifier":
        """
        Batch fit on ledger data using sklearn LogisticRegression.

        Args:
            X: shape (n, N_FEATURES) — output of encode() stacked
            y: list/array of regime strings ('MOMENTUM', 'REVERSION', 'FLAT')

        CS229 L04: we maximise the multinomial log-likelihood with L2 penalty.
        sklearn's lbfgs solver does exactly this.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        y_idx = np.array([_REGIME_TO_IDX.get(r, 2) for r in y], dtype=np.int32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler_mean = scaler.mean_.astype(np.float64)
        self._scaler_std  = scaler.scale_.astype(np.float64)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=self._C,
        )
        clf.fit(X_scaled, y_idx)

        # Extract weights so we can do online SGD later
        self._W = clf.coef_.astype(np.float64)    # (3, d)
        self._b = clf.intercept_.astype(np.float64)
        self._fitted = True
        self._n_updates = len(y)
        self._save()
        return self

    # ── Online SGD update (CS229 L03) ─────────────────────────────────── #

    def update_online(self, x: np.ndarray, regime_label: str,
                      alpha: float = 0.05) -> None:
        """
        One SGD step on a single (x, y) example.

        CS229 L03 gradient of log-loss for softmax:
            ∂ℓ/∂wₖ = (1[y=k] − p_k) · x
            w ← w + α · ∂ℓ/∂w − α/C · w   (L2 regularisation)

        Args:
            x: shape (N_FEATURES,) from encode()
            regime_label: 'MOMENTUM', 'REVERSION', or 'FLAT'
            alpha: learning rate (default 0.05)
        """
        k = _REGIME_TO_IDX.get(regime_label, 2)
        x_s = self._scale(x)

        logits = self._W @ x_s + self._b
        probs  = self._softmax(logits)

        # Gradient: one-hot minus probabilities
        for c in range(3):
            indicator = 1.0 if c == k else 0.0
            grad = (indicator - probs[c]) * x_s
            # SGD + L2
            self._W[c] += alpha * grad - (alpha / self._C) * self._W[c]
            self._b[c] += alpha * (indicator - probs[c])

        self._n_updates += 1
        self._fitted = True

        # Persist periodically
        if self._n_updates % 50 == 0:
            self._save()

    # ── Inference ─────────────────────────────────────────────────────── #

    def predict_proba(self, x: np.ndarray) -> Dict[str, float]:
        """
        Returns probability distribution over MOMENTUM / REVERSION / FLAT.

        Args:
            x: shape (N_FEATURES,) from encode()

        Returns:
            dict mapping regime name → probability [0, 1], summing to 1.
        """
        if not self._fitted:
            return {r: 1.0 / 3.0 for r in REGIMES}

        x_s = self._scale(x)
        logits = self._W @ x_s + self._b
        probs  = self._softmax(logits)
        return {REGIMES[i]: float(probs[i]) for i in range(3)}

    def dominant_regime(self, x: np.ndarray):
        """Return (regime_label, probability) for the most likely regime."""
        proba = self.predict_proba(x)
        regime = max(proba, key=proba.get)
        return regime, proba[regime]

    def describe(self) -> str:
        return (f"SoftmaxRegimeClassifier fitted={self._fitted} "
                f"n_updates={self._n_updates}")
