"""
PredictNow — Ernest Chan (Algorithmic Trading, Lecture 2)

Regime-conditional win-rate estimator using LOESS kernel regression.
Outputs P(trade profitable | current regime features) as a dynamic,
per-trade probability that feeds directly into Kelly as the win_rate prior.

Ernest Chan (lecture, 52:00):
  "In classical statistics you get a static probability — 5% every day.
   In machine learning you get a DIFFERENT probability each day based on
   current market conditions. That is a much more nuanced understanding."

Implementation:
  1. Feature space: [regime_enc, hmm_prob, hurst, adx_norm, strategy_enc]
     If ICA preprocessor is provided, features are projected into the ICA
     independent component space before LOESS distance computation.

  2. LOESS (Locally Weighted Scatterplot Smoothing):
     Kernel: tricube weight w_i = (1 − (d_i/d_max)³)³
     where d_i is the distance from the query point to training point i.
     The local win probability is: p = Σ(w_i · y_i) / Σ(w_i)
     where y_i ∈ {0, 1} is the trade outcome.

  3. Online Newton IRLS (CS229 L04):
     Iteratively Reweighted Least Squares for logistic regression.
     Each trade close provides one (x, y) pair for gradient updates.
     Blends with the LOESS estimate using sample complexity weighting.

  4. Library integration (Integration Point I2):
     library_informed_win_rate() blends PredictNow's own estimate with
     the Alexandrian Library's historical win rates for the current regime.
     Hoeffding ramp: Library prior dominates when n_trades < 400.

Used by orchestrator at stages 4 (Kelly input) and 4f (size multiplier):
  PredictNow.evaluate(regime, hmm_transition_prob, hurst, adx, strategy,
                      ica_preprocessor) → PredictNowOutput
  PredictNow.record_outcome(regime, ..., won, pnl) → online SGD update
  library_informed_win_rate(own_estimate, n_trades, library_insight)
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "predict_now.pkl"
_LEDGER_DIR = _ROOT / "data" / "ledger"

# Feature encoding maps (mirrors SoftmaxRegimeClassifier for consistency)
_REGIME_ENC: Dict[str, float] = {
    "MOMENTUM":  1.0,
    "REVERSION": -1.0,
    "FLAT":       0.0,
}
_STRATEGY_ENC: Dict[str, float] = {
    "momentum": 1.0, "momentum_sma": 1.0, "donchian": 1.0,
    "donchian_breakout": 1.0, "reversion": -1.0, "bb_reversion": -1.0,
    "mean_reversion": -1.0, "atr_channel": 0.0, "fvg_fill": 0.0,
    "liquidity_sweep": 0.5,
}

# Library historical win rates by regime (used in I2 blend)
_LIBRARY_WIN_RATES: Dict[str, float] = {
    "MOMENTUM":   0.57,
    "REVERSION":  0.53,
    "FLAT":       0.48,
    "TRENDING":   0.58,
    "RANGING":    0.51,
    "VOLATILE":   0.42,
}

N_FEATURES = 5
_LOESS_BANDWIDTH = 0.40    # fraction of dataset to use in local window
_MIN_LOESS_NEIGHBOURS = 5  # minimum samples for LOESS estimate


@dataclass
class PredictNowOutput:
    """Result of PredictNow.evaluate()."""
    prob_profitable:    float  # P(trade is profitable) ∈ [0.01, 0.99]
    size_multiplier:    float  # position size modifier (0.0 = skip, 1.0 = full)
    reason:             str
    n_trades_in_window: int    # number of trades used in LOESS estimate
    regime_win_rate:    float  # observed win rate for this regime


class PredictNow:
    """
    Dynamic per-trade profitability estimator.

    Cold-start behaviour:
      - If no ledger data → returns uninformative prior (0.55) with full size.
      - As trades accumulate: LOESS uses the nearest neighbours in feature space.
      - After 20+ trades: Newton IRLS online logistic regression blends in.

    Attributes:
        _X (list): Accumulated feature vectors from closed trades.
        _y (list): Corresponding binary outcomes (1 = win, 0 = loss).
        _w_irls (ndarray): Newton IRLS logistic regression weights [N_FEATURES+1].
        _n_irls (int): Number of IRLS gradient steps applied.
    """

    def __init__(self) -> None:
        self._X: List[np.ndarray] = []
        self._y: List[int]        = []
        self._w_irls = np.zeros(N_FEATURES + 1, dtype=np.float64)   # +1 for bias
        self._n_irls = 0
        self._loaded = False

    def load_or_train(self) -> None:
        """Load checkpoint or bootstrap from ledger."""
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    state = pickle.load(f)
                self._X      = state.get("X", [])
                self._y      = state.get("y", [])
                self._w_irls = state.get("w_irls", np.zeros(N_FEATURES + 1))
                self._n_irls = state.get("n_irls", 0)
                self._loaded = True
                logger.debug(f"[PredictNow] Loaded checkpoint "
                             f"({len(self._X)} trades, {self._n_irls} IRLS steps)")
                return
            except Exception as e:
                logger.debug(f"[PredictNow] Checkpoint load failed (non-fatal): {e}")

        self._bootstrap_from_ledger()

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump({
                    "X":      self._X,
                    "y":      self._y,
                    "w_irls": self._w_irls,
                    "n_irls": self._n_irls,
                }, f)
        except Exception as e:
            logger.debug(f"[PredictNow] Save failed (non-fatal): {e}")

    def _bootstrap_from_ledger(self) -> None:
        if not _LEDGER_DIR.exists():
            return
        try:
            for f in sorted(_LEDGER_DIR.glob("trade_ledger_*.jsonl")):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    if t.get("status") != "closed":
                        continue
                    x = self._encode(
                        regime=t.get("regime", "FLAT"),
                        hmm_transition_prob=float(t.get("hmm_transition_prob", 0.5)),
                        hurst=float(t.get("hurst", 0.5)),
                        adx=float(t.get("adx", 20.0)),
                        strategy=t.get("strategy", "momentum"),
                    )
                    y = 1 if float(t.get("pnl", 0.0)) > 0 else 0
                    self._X.append(x)
                    self._y.append(y)
            logger.debug(f"[PredictNow] Bootstrap: {len(self._X)} trades from ledger")
            if len(self._X) >= 20:
                self._irls_batch()
        except Exception as e:
            logger.debug(f"[PredictNow] Bootstrap failed (non-fatal): {e}")

    # ── Feature engineering ───────────────────────────────────────────── #

    @staticmethod
    def _encode(
        regime:              str,
        hmm_transition_prob: float,
        hurst:               float,
        adx:                 float,
        strategy:            str,
    ) -> np.ndarray:
        return np.array([
            float(_REGIME_ENC.get(regime, 0.0)),
            float(hmm_transition_prob),
            float(hurst),
            float(adx) / 50.0,
            float(_STRATEGY_ENC.get(strategy, 0.0)),
        ], dtype=np.float64)

    # ── LOESS kernel ─────────────────────────────────────────────────── #

    def _loess_estimate(
        self,
        x: np.ndarray,
        ica_preprocessor=None,
    ) -> Tuple[float, int]:
        """
        Locally-weighted win probability using tricube kernel.

        Returns (prob_profitable, n_neighbours_used).
        """
        if len(self._X) < _MIN_LOESS_NEIGHBOURS:
            return 0.55, 0

        X_mat = np.array(self._X, dtype=np.float64)
        y_vec = np.array(self._y, dtype=np.float64)

        # ICA projection for better distance metric (CS229 L15)
        if ica_preprocessor is not None and getattr(ica_preprocessor, "_fitted", False):
            try:
                x_proj  = ica_preprocessor.transform(x)
                X_proj  = ica_preprocessor.transform(X_mat)
            except Exception:
                x_proj, X_proj = x, X_mat
        else:
            x_proj, X_proj = x, X_mat

        # Euclidean distances in projected space
        diffs = X_proj - x_proj
        dists = np.sqrt((diffs ** 2).sum(axis=1))

        # Bandwidth: use the fraction of total dataset
        k = max(_MIN_LOESS_NEIGHBOURS, int(len(dists) * _LOESS_BANDWIDTH))
        k = min(k, len(dists))

        # Sort by distance; use the k nearest
        sorted_idx = np.argsort(dists)[:k]
        d_near     = dists[sorted_idx]
        y_near     = y_vec[sorted_idx]
        d_max      = d_near[-1] + 1e-9

        # Tricube kernel weights: w = (1 − (d/d_max)³)³
        u = d_near / d_max
        u = np.clip(u, 0.0, 1.0)
        weights = (1.0 - u ** 3) ** 3

        w_sum = weights.sum()
        if w_sum < 1e-12:
            return 0.55, k

        prob = float(np.dot(weights, y_near) / w_sum)
        return float(max(0.05, min(0.95, prob))), k

    # ── Newton IRLS logistic regression ──────────────────────────────── #

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _irls_batch(self, n_iter: int = 10) -> None:
        """
        Batch Newton IRLS fit on all accumulated (X, y) pairs.

        CS229 L04 (Newton's method for logistic regression):
          θ ← θ − H⁻¹·∇ℓ(θ)
          where H is the Hessian and ∇ℓ the gradient of log-likelihood.
        """
        if len(self._X) < 5:
            return
        X_b = np.column_stack([np.ones(len(self._X)), np.array(self._X)])
        y_b = np.array(self._y, dtype=np.float64)

        w = self._w_irls.copy()
        for _ in range(n_iter):
            p = self._sigmoid(X_b @ w)
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            # Gradient: Xᵀ(p − y)
            grad = X_b.T @ (p - y_b)
            # Hessian: Xᵀ·diag(p(1-p))·X
            W_diag = p * (1.0 - p)
            H = X_b.T @ (W_diag[:, None] * X_b) + 1e-4 * np.eye(X_b.shape[1])
            try:
                w = w - np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        self._w_irls = w
        self._n_irls += n_iter

    def _irls_online(self, x: np.ndarray, y: int, alpha: float = 0.10) -> None:
        """One SGD step on a single (x, y) pair (faster than full IRLS)."""
        x_b = np.concatenate([[1.0], x])
        p   = float(self._sigmoid(np.array([self._w_irls @ x_b]))[0])
        grad = (p - float(y)) * x_b
        self._w_irls -= alpha * grad
        self._n_irls += 1

    def _irls_predict(self, x: np.ndarray) -> float:
        """IRLS logistic regression probability estimate."""
        if self._n_irls < 5:
            return 0.55
        x_b = np.concatenate([[1.0], x])
        return float(self._sigmoid(np.array([self._w_irls @ x_b]))[0])

    # ── Evaluate ─────────────────────────────────────────────────────── #

    def evaluate(
        self,
        regime:              str,
        hmm_transition_prob: float,
        hurst:               float,
        adx:                 float,
        strategy:            str,
        ica_preprocessor=None,
    ) -> PredictNowOutput:
        """
        Return P(trade profitable) for the current conditions.

        Blends LOESS kernel estimate with Newton IRLS logistic regression,
        weighted by sample complexity (CS229 L09 Hoeffding).

        Args:
            regime:              Current regime ('MOMENTUM', 'REVERSION', 'FLAT').
            hmm_transition_prob: HMM regime transition probability.
            hurst:               Hurst exponent (short window).
            adx:                 ADX-14 value.
            strategy:            Specialist name.
            ica_preprocessor:    ICAFactorSeparator instance or None.

        Returns:
            PredictNowOutput with prob_profitable and size_multiplier.
        """
        x = self._encode(regime, hmm_transition_prob, hurst, adx, strategy)
        n = len(self._X)

        # LOESS estimate
        p_loess, n_neighbours = self._loess_estimate(x, ica_preprocessor)

        # IRLS estimate
        p_irls = self._irls_predict(x)

        # Blend: weight IRLS more as data accumulates (CS229 L09 sample complexity)
        # At n=0: full LOESS prior (0.55). At n=200: 50/50. At n=500+: 80% IRLS.
        irls_weight = min(0.80, n / 500.0) if self._n_irls >= 5 else 0.0
        loess_weight = 1.0 - irls_weight
        p = irls_weight * p_irls + loess_weight * p_loess

        # Regime-based observed win rate (simple lookup from history)
        regime_wr = self._regime_win_rate(regime)

        # Size multiplier — scale by deviation from neutral
        if p <= 0.40:
            size_mult = 0.0
            reason = f"PREDICT_NOW_SKIP — P(win)={p:.2f} < 0.40 gate"
        elif p < 0.50:
            size_mult = 0.60
            reason = f"PredictNow: P(win)={p:.2f} (borderline) → ×0.60"
        elif p >= 0.70:
            size_mult = 1.20
            reason = f"PredictNow: P(win)={p:.2f} (high conviction) → ×1.20"
        else:
            size_mult = 1.0
            reason = f"PredictNow: P(win)={p:.2f} → normal size"

        # Cap size multiplier at 1.5× (never inflate aggressively)
        size_mult = min(1.5, max(0.0, size_mult))

        return PredictNowOutput(
            prob_profitable=float(max(0.01, min(0.99, p))),
            size_multiplier=size_mult,
            reason=reason,
            n_trades_in_window=n_neighbours,
            regime_win_rate=regime_wr,
        )

    def _regime_win_rate(self, regime: str) -> float:
        if not self._y:
            return _LIBRARY_WIN_RATES.get(regime, 0.50)
        regime_enc = _REGIME_ENC.get(regime, 0.0)
        relevant = [
            y for x, y in zip(self._X, self._y)
            if abs(x[0] - regime_enc) < 0.1
        ]
        if not relevant:
            return _LIBRARY_WIN_RATES.get(regime, 0.50)
        return float(np.mean(relevant))

    # ── Online learning ───────────────────────────────────────────────── #

    def record_outcome(
        self,
        regime:              str,
        hmm_transition_prob: float,
        hurst:               float,
        adx:                 float,
        strategy:            str,
        won:                 bool,
        pnl:                 float,
    ) -> None:
        """
        Online update: append (x, y) and do one IRLS SGD step.

        Called by orchestrator.on_trade_close().
        """
        x = self._encode(regime, hmm_transition_prob, hurst, adx, strategy)
        y = 1 if won else 0
        self._X.append(x)
        self._y.append(y)

        # Online SGD step
        self._irls_online(x, y, alpha=0.05)

        # Periodic batch refit for stability
        if len(self._X) % 50 == 0 and len(self._X) >= 20:
            self._irls_batch(n_iter=5)

        if len(self._X) % 100 == 0:
            self._save()

    def describe(self) -> str:
        return (f"PredictNow n_trades={len(self._X)} "
                f"n_irls={self._n_irls}")


# ── Library integration (I2) ─────────────────────────────────────────── #

def library_informed_win_rate(
    own_estimate:    float,
    n_trades:        int,
    library_insight,
) -> Tuple[float, str]:
    """
    Blend PredictNow's own estimate with Library historical win rates.

    Integration Point I2 — Alexandrian Library → PredictNow blend.

    Hoeffding ramp: when n_trades < 400, the Library's historical win rate
    carries significant weight. As n_trades grows, our own estimate dominates.

    CS229 L09: "The Hoeffding bound decays with 1/sqrt(m). With few trades,
    the historical prior is more reliable than our noisy estimate."

    Args:
        own_estimate:   PredictNow's raw P(profitable) estimate.
        n_trades:       Number of closed trades behind the own_estimate.
        library_insight: LibraryInsight from AlexandrianLibrary.query() or None.

    Returns:
        (blended_probability, reason_string)
    """
    if library_insight is None:
        return own_estimate, "no_library"

    # Get Library's regime-based historical win rate
    regime = getattr(library_insight, "primary_regime", "UNKNOWN") or "UNKNOWN"
    lib_wr = _LIBRARY_WIN_RATES.get(regime, 0.50)

    # Hoeffding ramp weight: Library weight = 1 - min(1, n_trades / 400)
    # At n=0: 100% Library. At n=400: 0% Library.
    lib_weight = max(0.0, 1.0 - n_trades / 400.0)
    own_weight = 1.0 - lib_weight

    blended = own_weight * own_estimate + lib_weight * lib_wr

    if lib_weight > 0.01:
        reason = (f"library_blend: own={own_estimate:.3f}×{own_weight:.2f} + "
                  f"lib({regime})={lib_wr:.3f}×{lib_weight:.2f} "
                  f"→ {blended:.3f} (n_trades={n_trades})")
    else:
        reason = f"own_dominated: n_trades={n_trades} (lib_weight={lib_weight:.3f})"

    return float(max(0.01, min(0.99, blended))), reason
