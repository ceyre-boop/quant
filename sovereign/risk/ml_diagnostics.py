# LOW-USE: imported by sovereign/orchestrator.py (verified 2026-07-20). NOT dead code — do not delete.
# The earlier header read "not imported by any live path"; that was false for all 11 modules
# in this group. See trial/subtraction_verdicts.md:47.
"""
ML Diagnostics — CS229 Lectures 10 / 12

KMeansRegimeClusterer: unsupervised 3-cluster regime classification
for the three-vote ensemble regime system (HMM + Softmax + KMeans).

CS229 L12 theory applied:
  "K-Means algorithm: initialise cluster centroids μ₁,...,μ_K.
   E-step: assign c⁽ⁱ⁾ = argmin_j ||x⁽ⁱ⁾ − μⱼ||²
   M-step: μⱼ = Σᵢ 1[c⁽ⁱ⁾=j]·x⁽ⁱ⁾ / Σᵢ 1[c⁽ⁱ⁾=j]
   Convergence: when assignments stop changing."

Three regime interpretation:
  Cluster 0 → MOMENTUM  (high Hurst, high ADX, low HMM transition prob)
  Cluster 1 → REVERSION (low Hurst, low ADX, moderate HMM transition prob)
  Cluster 2 → FLAT      (Hurst near 0.5, low ADX, high HMM transition prob)

The cluster→regime mapping is determined by fitting a labelled dataset and
matching cluster centroids to expected regime feature patterns.

Feature vector (3 dims):
  [hurst, adx_norm (adx/50), hmm_transition_prob]

Refit schedule (from orchestrator.on_trade_close):
  Every 10 new trades once ≥30 observations are available.
  Last 200 observations are used for each refit.

Used by orchestrator:
  KMeansRegimeClusterer.fit(X) → batch fit
  KMeansRegimeClusterer.predict(x) → regime string
  KMeansRegimeClusterer._centroids → ndarray or None (None = not fitted)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "kmeans_regime.pkl"

# Expected centroid feature profiles for regime identification:
# [hurst, adx_norm, hmm_transition_prob]
_REGIME_PROTOTYPES: Dict[str, np.ndarray] = {
    "MOMENTUM":  np.array([0.65, 0.70, 0.15]),   # trending, strong, stable
    "REVERSION": np.array([0.40, 0.35, 0.40]),   # mean-reverting, weak, moderate
    "FLAT":      np.array([0.50, 0.20, 0.60]),   # random walk, very weak, unstable
}


class KMeansRegimeClusterer:
    """
    Unsupervised K-Means regime clusterer for the three-vote ensemble.

    Provides a third independent vote on regime type, complementing
    the HMM (Hurst/ADX hard rules) and SoftmaxRegimeClassifier (logistic).

    K-Means produces assignments without regime labels, so we map each
    cluster to a regime by matching its centroid to the expected prototype.

    Attributes:
        _centroids (ndarray or None): Cluster centroids, shape (k, 3).
                                       None until first fit().
        _cluster_regime (dict): cluster_id → regime string.
    """

    def __init__(self, k: int = 3, n_init: int = 10) -> None:
        self._k = k
        self._n_init = n_init
        self._centroids: Optional[np.ndarray] = None
        self._cluster_regime: Dict[int, str] = {}
        self._try_load()

    # ── Serialisation ─────────────────────────────────────────────────── #

    def _try_load(self) -> None:
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    state = pickle.load(f)
                self._centroids     = state["centroids"]
                self._cluster_regime = state["cluster_regime"]
                logger.debug(f"[KMeans] Loaded from {_CHECKPOINT}")
            except Exception as e:
                logger.debug(f"[KMeans] Checkpoint load failed (non-fatal): {e}")

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump({
                    "centroids":      self._centroids,
                    "cluster_regime": self._cluster_regime,
                }, f)
        except Exception as e:
            logger.debug(f"[KMeans] Save failed (non-fatal): {e}")

    # ── Fit ───────────────────────────────────────────────────────────── #

    def fit(self, X: np.ndarray) -> "KMeansRegimeClusterer":
        """
        Fit K-Means on the feature matrix X.

        CS229 L12: K-Means minimises the within-cluster sum of squares.
        Multiple restarts (n_init) are used to avoid local minima.

        Args:
            X: shape (n_samples, 3). Must have ≥ k=3 distinct samples.
               Feature order: [hurst, adx_norm, hmm_transition_prob]

        Returns:
            self (for chaining)
        """
        if len(X) < self._k:
            logger.debug(f"[KMeans] Not enough samples ({len(X)}) for k={self._k}")
            return self

        try:
            from sklearn.cluster import KMeans

            km = KMeans(
                n_clusters=self._k,
                n_init=self._n_init,
                max_iter=300,
                random_state=42,
            )
            km.fit(X)
            self._centroids = km.cluster_centers_   # (k, 3)

            # Map each cluster to a regime by matching centroids to prototypes
            self._cluster_regime = self._assign_regimes(self._centroids)
            self._save()

            logger.debug(f"[KMeans] Fit on {len(X)} samples. "
                         f"Mapping: {self._cluster_regime}")
        except Exception as e:
            logger.warning(f"[KMeans] Fit failed (non-fatal): {e}")

        return self

    def _assign_regimes(self, centroids: np.ndarray) -> Dict[int, str]:
        """
        Assign each cluster centroid to the closest regime prototype.

        Greedy assignment: find the closest (centroid, prototype) pair,
        assign, then exclude both from future matching.
        """
        prototypes = {r: v for r, v in _REGIME_PROTOTYPES.items()}
        available_regimes = list(prototypes.keys())
        assignment: Dict[int, str] = {}

        for cluster_id in range(len(centroids)):
            c = centroids[cluster_id]
            best_regime, best_dist = None, float("inf")
            for r in available_regimes:
                d = float(np.linalg.norm(c - prototypes[r]))
                if d < best_dist:
                    best_dist = d
                    best_regime = r
            if best_regime:
                assignment[cluster_id] = best_regime
                available_regimes.remove(best_regime)

        # Handle unassigned clusters (shouldn't happen with k=3)
        for cluster_id in range(len(centroids)):
            if cluster_id not in assignment:
                assignment[cluster_id] = "FLAT"

        return assignment

    # ── Predict ───────────────────────────────────────────────────────── #

    def predict(self, x: np.ndarray) -> str:
        """
        Predict the regime for a single feature vector.

        Returns the regime label ('MOMENTUM', 'REVERSION', 'FLAT').
        Falls back to 'FLAT' if not fitted.

        Args:
            x: shape (3,) — [hurst, adx_norm, hmm_transition_prob]
        """
        if self._centroids is None or not self._cluster_regime:
            return "FLAT"

        # Assign to nearest centroid (E-step)
        diffs     = self._centroids - x
        dists     = np.sqrt((diffs ** 2).sum(axis=1))
        cluster   = int(dists.argmin())
        return self._cluster_regime.get(cluster, "FLAT")

    def predict_distances(self, x: np.ndarray) -> Dict[str, float]:
        """
        Return normalised distance from x to each regime centroid.

        Useful for soft regime probability estimates from K-Means.
        Returns dict mapping regime → normalised_distance (lower = closer).
        """
        if self._centroids is None:
            return {r: 1.0 for r in _REGIME_PROTOTYPES}

        diffs = self._centroids - x
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        total = dists.sum() + 1e-9
        return {
            self._cluster_regime.get(i, f"cluster_{i}"): float(dists[i] / total)
            for i in range(len(self._centroids))
        }

    def describe(self) -> str:
        fitted = self._centroids is not None
        return (f"KMeansRegimeClusterer k={self._k} fitted={fitted} "
                f"mapping={self._cluster_regime}")
