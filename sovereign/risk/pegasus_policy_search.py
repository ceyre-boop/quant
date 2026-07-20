# LOW-USE: imported by sovereign/orchestrator.py (verified 2026-07-20). NOT dead code — do not delete.
# The earlier header read "not imported by any live path"; that was false for all 11 modules
# in this group. See trial/subtraction_verdicts.md:47.
"""
Pegasus Policy Search — CS229 Lecture 20

REINFORCE policy gradient for joint optimisation of 6 trading parameters.
Pegasus (Policy search via Environment Generated AUgmented Samples): simulates
multiple trading episodes to estimate the gradient of E[R] w.r.t. policy params.

CS229 L20 theory applied:
  "REINFORCE: gradient of E_{τ}[R(τ)] w.r.t. θ is:
   ∇_θ E[R] = E[Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ)]
   We estimate this with Monte Carlo rollouts (Pegasus) and update θ ← θ + α·∇̂."

  Policy π_θ(a|s): each of the 6 parameters is drawn from a Gaussian
  (for continuous params) or a bounded logistic sigmoid (for threshold params).

6 parameters jointly optimised:
  1. entry_threshold  (0.40–0.70): minimum P(win) to enter a trade
  2. size_multiplier  (0.50–1.50): scale factor on base Kelly size
  3. stop_atr_mult    (1.0–3.0):  ATR multiple for stop placement
  4. tp_rr_ratio      (1.5–5.0):  reward/risk ratio for TP placement
  5. hmm_conf_gate    (0.30–0.80): minimum HMM confidence to trade
  6. kelly_fraction_cap (0.01–0.04): ceiling on Kelly bet size

Trust ramp:
  n_updates < 10:  parameters not applied (too early to trust)
  10 ≤ n_updates < 20: gate-only (hmm_conf_gate + entry_threshold active)
  20 ≤ n_updates < 30: proportional blending (trust ramp 0 → 1)
  n_updates ≥ 30:  full trust (all 6 params applied with learned values)

Used by orchestrator:
  PegasusPolicySearch.current_params → PegasusParams (dataclass)
  PegasusPolicySearch.n_updates → int
  PegasusPolicySearch.trust_multiplier → float [0, 1]
  PegasusPolicySearch.reinforce_update(state_features, action_taken,
                                        realized_pnl, ...) → gradient step
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "pegasus_policy.pkl"

# Parameter space bounds [min, max, default]
_PARAM_BOUNDS = {
    "entry_threshold":    (0.40, 0.70, 0.50),
    "size_multiplier":    (0.50, 1.50, 1.00),
    "stop_atr_mult":      (1.00, 3.00, 1.50),
    "tp_rr_ratio":        (1.50, 5.00, 2.50),
    "hmm_conf_gate":      (0.30, 0.80, 0.55),
    "kelly_fraction_cap": (0.01, 0.04, 0.025),
}

_N_PARAMS = len(_PARAM_BOUNDS)

# Trust thresholds
_GATE_THRESHOLD  = 10   # apply gate params (hmm + entry)
_EXEC_THRESHOLD  = 20   # begin blending toward learned values
_FULL_THRESHOLD  = 30   # full trust in all params


@dataclass
class PegasusParams:
    """
    Current policy parameter values.

    These are the actual parameter values used by the orchestrator.
    Before trust is earned, they default to safe hand-crafted values.
    """
    entry_threshold:    float = 0.50
    size_multiplier:    float = 1.00
    stop_atr_mult:      float = 1.50
    tp_rr_ratio:        float = 2.50
    hmm_conf_gate:      float = 0.55
    kelly_fraction_cap: float = 0.025


# TradingPolicyParams is an alias-shaped container used by the Pegasus
# scenario-evaluation API (deterministic policy scoring). It mirrors
# PegasusParams and additionally exposes the normalised search-space bounds.
@dataclass
class TradingPolicyParams:
    """
    Policy parameters consumed by PegasusPolicySearch.evaluate_policy().

    Field values are in actual (real-world) units. bounds() returns the
    normalised [0, 1] search-space limits for the underlying θ vector, which
    is the space the REINFORCE update operates and clamps in.
    """
    entry_threshold:    float = 0.50
    size_multiplier:    float = 1.00
    stop_atr_mult:      float = 1.50
    tp_rr_ratio:        float = 2.50
    hmm_conf_gate:      float = 0.55
    kelly_fraction_cap: float = 0.025

    @classmethod
    def bounds(cls) -> list:
        """
        Normalised search-space bounds for the θ vector — one (lo, hi) per
        parameter. θ is optimised inside the unit cube [0, 1]^6, so every
        entry is (0.0, 1.0). Length matches PegasusPolicySearch._theta.
        """
        return [(0.0, 1.0)] * _N_PARAMS


@dataclass
class Scenario:
    """
    A single Pegasus rollout scenario: a fixed set of trades replayed under a
    candidate policy. Holding the trades fixed (seeded once) is what makes
    policy evaluation deterministic — the Pegasus core property (CS229 L20).

    Attributes:
        seed:   Scenario seed / index (used when the scenario is generated).
        trades: List of trade dicts, each with at least a 'pnl' key and the
                gating fields 'predicted_win_rate' and 'hmm_confidence'.
    """
    seed:   int
    trades: list


class PegasusPolicySearch:
    """
    REINFORCE policy gradient for joint 6-parameter optimisation.

    The policy is a product of independent Gaussians for continuous params
    and logistic Gaussians for bounded params. The mean vector θ (6-dim)
    is updated by the REINFORCE gradient estimate.

    Attributes:
        n_updates (int): Number of gradient steps taken so far.
        current_params (PegasusParams): Current parameter values.
        trust_multiplier (float): How much to trust learned params [0, 1].
    """

    def __init__(self, n_scenarios: int = 50) -> None:
        self._n_scenarios = n_scenarios   # Pegasus MC rollouts per update

        # Policy mean (in normalised space [0, 1] for each param)
        self._theta = np.array([0.5] * _N_PARAMS, dtype=np.float64)

        # Policy std (exploration noise) — starts wide, decays slowly
        self._sigma = np.array([0.15] * _N_PARAMS, dtype=np.float64)

        # Gradient accumulator (momentum)
        self._grad_acc = np.zeros(_N_PARAMS, dtype=np.float64)
        self._grad_momentum = 0.9

        # Running reward baseline (EMA) for REINFORCE variance reduction.
        # Starts at 0.0 so the first update produces a non-zero advantage.
        self._reward_baseline = 0.0

        # Pegasus scenarios (fixed rollouts for deterministic policy scoring)
        self._scenarios: list = []

        # Learning rate (decays with updates)
        self._alpha0 = 0.05

        # History of (feature, action, reward) for REINFORCE
        self._history: list = []
        self.n_updates: int = 0

        self.current_params: PegasusParams = PegasusParams()
        self._try_load()

    # ── Serialisation ─────────────────────────────────────────────────── #

    def _try_load(self) -> None:
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    state = pickle.load(f)
                self._theta      = state["theta"]
                self._sigma      = state["sigma"]
                self._grad_acc   = state.get("grad_acc", np.zeros(_N_PARAMS))
                self.n_updates   = state.get("n_updates", 0)
                self._history    = state.get("history", [])
                self._update_current_params()
                logger.debug(f"[Pegasus] Loaded (n_updates={self.n_updates})")
            except Exception as e:
                logger.debug(f"[Pegasus] Checkpoint load failed (non-fatal): {e}")

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump({
                    "theta":    self._theta,
                    "sigma":    self._sigma,
                    "grad_acc": self._grad_acc,
                    "n_updates": self.n_updates,
                    "history":  self._history[-200:],   # keep last 200
                }, f)
        except Exception as e:
            logger.debug(f"[Pegasus] Save failed (non-fatal): {e}")

    # ── Parameter mapping ─────────────────────────────────────────────── #

    @staticmethod
    def _to_param(theta_i: float, key: str) -> float:
        """Map normalised θ_i ∈ [0, 1] to the actual parameter range."""
        lo, hi, _ = _PARAM_BOUNDS[key]
        return float(lo + np.clip(theta_i, 0.0, 1.0) * (hi - lo))

    @staticmethod
    def _from_param(val: float, key: str) -> float:
        """Map actual parameter value to normalised θ_i ∈ [0, 1]."""
        lo, hi, _ = _PARAM_BOUNDS[key]
        return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _update_current_params(self) -> None:
        """Reconstruct PegasusParams from the current θ vector."""
        keys = list(_PARAM_BOUNDS.keys())
        params = {}
        for i, key in enumerate(keys):
            params[key] = self._to_param(self._theta[i], key)
        self.current_params = PegasusParams(**params)

    # ── Trust ramp ────────────────────────────────────────────────────── #

    @property
    def trust_multiplier(self) -> float:
        """
        Proportional trust in learned parameters: 0 → 1 over n_updates 20–30.
        Used by orchestrator to blend between defaults and learned values.
        """
        if self.n_updates < _EXEC_THRESHOLD:
            return 0.0
        elif self.n_updates >= _FULL_THRESHOLD:
            return 1.0
        else:
            return (self.n_updates - _EXEC_THRESHOLD) / (_FULL_THRESHOLD - _EXEC_THRESHOLD)

    # ── REINFORCE update ──────────────────────────────────────────────── #

    def reinforce_update(
        self,
        state_features:   np.ndarray,
        action_taken:     float,
        realized_pnl:     float,
        hmm_confidence:   float = 0.55,
        stop_atr_used:    float = 1.50,
        tp_rr_used:       float = 2.50,
        kelly_frac_used:  float = 0.025,
    ) -> None:
        """
        One REINFORCE gradient step from a single trade outcome.

        CS229 L20: ∇_θ log π_θ(a|s) · R for Gaussian policy:
          π_θ(a|s) = N(μ(s;θ), σ²)
          ∇_θ log π_θ = (a − μ) · ∇_θ μ / σ²

        For simplicity we use a state-independent policy (μ = f(θ) only)
        since we have very few data points — a common practical choice.
        The state features are stored for future batched gradient estimation.

        Args:
            state_features:  np.ndarray([confidence, 1-hmm_tp, hurst, adx/50]) — 4-dim
            action_taken:    Actual position size used (scalar, proxy for all params)
            realized_pnl:    Trade PnL (positive = reward, negative = cost)
            hmm_confidence:  HMM confidence at trade entry
            stop_atr_used:   ATR multiple used for stop
            tp_rr_used:      R:R ratio used for TP
            kelly_frac_used: Kelly fraction used for sizing
        """
        # Record the trade outcome for REINFORCE
        observed = np.array([
            self._from_param(0.50, "entry_threshold"),  # entry threshold used (default)
            self._from_param(min(1.5, max(0.5, action_taken / 1000.0 + 0.5)), "size_multiplier"),
            self._from_param(stop_atr_used, "stop_atr_mult"),
            self._from_param(tp_rr_used, "tp_rr_ratio"),
            self._from_param(hmm_confidence, "hmm_conf_gate"),
            self._from_param(kelly_frac_used, "kelly_fraction_cap"),
        ], dtype=np.float64)

        self._history.append((state_features, observed, float(realized_pnl)))

        # REINFORCE gradient estimate using current + recent history
        alpha = self._alpha0 / math.sqrt(1.0 + self.n_updates)
        grad  = self._estimate_reinforce_gradient()

        # Momentum update
        self._grad_acc = (self._grad_momentum * self._grad_acc
                          + (1.0 - self._grad_momentum) * grad)

        # Gradient ascent (maximise expected reward)
        self._theta += alpha * self._grad_acc
        self._theta  = np.clip(self._theta, 0.0, 1.0)

        # Update the running reward baseline AFTER using it for this step
        self._reward_baseline = (0.9 * self._reward_baseline
                                 + 0.1 * float(realized_pnl))

        # Decay exploration noise slowly (Pegasus: exploit more as n grows)
        self._sigma = np.maximum(0.02, self._sigma * 0.998)

        self.n_updates += 1
        self._update_current_params()

        if self.n_updates % 10 == 0:
            self._save()
            logger.debug(f"[Pegasus] n_updates={self.n_updates} "
                         f"trust={self.trust_multiplier:.2f} "
                         f"entry_gate={self.current_params.entry_threshold:.3f} "
                         f"kelly_cap={self.current_params.kelly_fraction_cap:.4f}")

    def _estimate_reinforce_gradient(self) -> np.ndarray:
        """
        REINFORCE gradient from the last N episodes in history.

        ∇_θ E[R] ≈ (1/N) Σ_i (a_i − μ) · (R_i − b) / σ²

        Uses a running EMA baseline b (self._reward_baseline) for variance
        reduction. A running baseline — rather than the within-window mean —
        keeps a single-episode update non-trivial (a lone win still yields a
        non-zero advantage, so θ actually moves).
        """
        if not self._history:
            return np.zeros(_N_PARAMS)

        # Use last min(50, len) episodes
        window = self._history[-min(self._n_scenarios, len(self._history)):]

        # Baseline variance reduction against the running EMA baseline
        baseline = float(self._reward_baseline)

        # Gradient: sum of (a - μ) * A / σ² over episodes
        mu    = self._theta                              # (n_params,)
        sigma2 = (self._sigma ** 2 + 1e-9)              # (n_params,)

        grad = np.zeros(_N_PARAMS)
        for (_, a, r) in window:
            grad += (a - mu) * (float(r) - baseline) / sigma2

        grad /= max(len(window), 1)

        # Clip gradient to prevent huge steps
        grad = np.clip(grad, -0.5, 0.5)
        return grad

    # ── Pegasus deterministic policy evaluation ───────────────────────── #

    def evaluate_policy(self, params: "TradingPolicyParams") -> float:
        """
        Score a candidate policy across the fixed scenario set.

        Pegasus core property (CS229 L20): with the scenarios held fixed, the
        payoff is a deterministic function of the policy parameters — the same
        policy on the same scenarios always yields the identical payoff, which
        is what makes the search well-conditioned.

        Each trade is admitted only if it clears the policy's entry gate
        (predicted_win_rate ≥ entry_threshold) and confidence gate
        (hmm_confidence ≥ hmm_conf_gate); admitted trades contribute
        pnl × size_multiplier.

        Returns:
            Total payoff summed over all scenarios (deterministic).
        """
        total = 0.0
        for scenario in self._scenarios:
            for trade in scenario.trades:
                if float(trade.get('predicted_win_rate', 0.0)) < params.entry_threshold:
                    continue
                if float(trade.get('hmm_confidence', 0.0)) < params.hmm_conf_gate:
                    continue
                total += float(trade.get('pnl', 0.0)) * params.size_multiplier
        return float(total)

    def build_risk_neutral_scenarios(
        self,
        sigma:   float,
        r:       float,
        T:       float,
        n_steps: int = 30,
    ) -> int:
        """
        Generate n_scenarios fixed rollouts from risk-neutral GBM paths.

        Under Q, log-returns are (r − ½σ²)·dt + σ·√dt·Z. Each simulated step
        becomes one synthetic trade whose pnl is that step's log-return. Each
        scenario is seeded by its index, so the scenario set is reproducible.

        Returns:
            Number of scenarios built (== self._n_scenarios).
        """
        dt = float(T) / max(1, n_steps)
        scenarios: list = []
        for i in range(self._n_scenarios):
            rng = np.random.default_rng(i)
            log_returns = ((r - 0.5 * sigma ** 2) * dt
                           + sigma * math.sqrt(dt) * rng.standard_normal(n_steps))
            trades = [{
                'pnl':                float(step_ret),
                'status':             'closed',
                'predicted_win_rate': 0.60,
                'hmm_confidence':     0.70,
                'stop_atr_mult':      1.5,
                'tp_rr':              2.5,
            } for step_ret in log_returns]
            scenarios.append(Scenario(seed=i, trades=trades))
        self._scenarios = scenarios
        return len(scenarios)

    def describe(self) -> str:
        p = self.current_params
        return (
            f"PegasusPolicySearch n_updates={self.n_updates} "
            f"trust={self.trust_multiplier:.2f} | "
            f"entry_gate={p.entry_threshold:.3f} "
            f"size_mult={p.size_multiplier:.3f} "
            f"stop_atr={p.stop_atr_mult:.2f} "
            f"tp_rr={p.tp_rr_ratio:.2f} "
            f"hmm_gate={p.hmm_conf_gate:.3f} "
            f"kelly_cap={p.kelly_fraction_cap:.4f}"
        )
