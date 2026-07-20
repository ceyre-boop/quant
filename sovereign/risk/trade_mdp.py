# LOW-USE: imported by sovereign/orchestrator.py (verified 2026-07-20). NOT dead code — do not delete.
# The earlier header read "not imported by any live path"; that was false for all 11 modules
# in this group. See trial/subtraction_verdicts.md:47.
"""
Trade MDP — CS229 Lecture 16 (Value Iteration)

Markov Decision Process over a 72-state trade state space.
Value iteration computes V*(s) — the optimal expected cumulative R-multiple
from state s. The resulting policy: size_multiplier = softmax(V*(s)).

CS229 L16 theory applied:
  "Value iteration: V_{i+1}(s) = R(s) + γ · max_a Σ_{s'} P(s'|s,a)·V_i(s')
   Convergence: when max_s |V_{i+1}(s) − V_i(s)| < ε.
   Optimal policy: π*(s) = argmax_a Σ_{s'} P(s'|s,a)·V_{i+1}(s')"

State space (72 states):
  regime:             3 values (MOMENTUM, REVERSION, FLAT)
  consecutive_losses: 4 buckets (0, 1, 2, 3+)
  drawdown_pct:       3 buckets (low <3%, medium 3–8%, high >8%)
  hmm_confidence:     2 buckets (low <0.60, high ≥0.60)
  Total: 3 × 4 × 3 × 2 = 72 states

Action space:
  size_multiplier ∈ {0.50, 0.75, 1.00, 1.25}

Transitions:
  Updated from real trade outcomes via record_transition().
  Initially seeded with expert-defined priors based on trading intuition.

Used by orchestrator at stage 4h:
  TradeMDP.get_size_multiplier(regime, consecutive_losses, drawdown_pct,
                               hmm_confidence) → float
  TradeMDP.record_transition(prev_state, next_state, pnl, ...) → Bellman update
  TradeMDP.state_value(regime, ...) → float
  TradeMDP._m.n_trades: int — used to gate activation (need ≥20)
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT = _ROOT / "models" / "trade_mdp.pkl"

# State space dimensions
_REGIMES     = ["MOMENTUM", "REVERSION", "FLAT"]
_CONSEC_BUCKETS = [0, 1, 2, 3]    # 0, 1, 2, ≥3 consecutive losses
_DD_BUCKETS     = [0, 1, 2]       # low (<3%), medium (3–8%), high (>8%)
_CONF_BUCKETS   = [0, 1]          # low (<0.60), high (≥0.60)

N_STATES  = len(_REGIMES) * len(_CONSEC_BUCKETS) * len(_DD_BUCKETS) * len(_CONF_BUCKETS)
N_ACTIONS = 4
_ACTIONS  = [0.50, 0.75, 1.00, 1.25]   # size multipliers

_GAMMA = 0.90   # discount factor (short trading horizon)
_VI_ITERS = 50  # value iteration sweeps per update


@dataclass
class MDPMemory:
    """Persistent memory updated by on_trade_close()."""
    n_trades:        int = 0
    # transition counts: T[s, a, s'] — empirical transition matrix
    T_counts: np.ndarray = field(
        default_factory=lambda: np.ones((N_STATES, N_ACTIONS, N_STATES)) * 0.1
    )
    # Empirical mean reward per (s, a)
    R_sum:   np.ndarray = field(
        default_factory=lambda: np.zeros((N_STATES, N_ACTIONS))
    )
    R_count: np.ndarray = field(
        default_factory=lambda: np.ones((N_STATES, N_ACTIONS)) * 1e-6
    )
    # Value function (computed by value iteration)
    V: np.ndarray = field(
        default_factory=lambda: np.zeros(N_STATES)
    )


def _regime_idx(regime: str) -> int:
    try:
        return _REGIMES.index(regime)
    except ValueError:
        return 2   # default to FLAT


def _consec_bucket(consecutive_losses: int) -> int:
    return min(consecutive_losses, 3)


def _dd_bucket(drawdown_pct: float) -> int:
    if drawdown_pct < 0.03:
        return 0
    elif drawdown_pct < 0.08:
        return 1
    return 2


def _conf_bucket(hmm_confidence: float) -> int:
    return 1 if hmm_confidence >= 0.60 else 0


def _discretize_state(
    regime:            str,
    consecutive_losses: int,
    drawdown_pct:      float,
    hmm_confidence:    float,
) -> Tuple[str, int, int, int]:
    """
    Map raw inputs to the discrete MDP state tuple.

    Returns:
        (regime, consec_bucket, dd_bucket, conf_bucket) where regime is the
        canonical regime string and the remaining elements are bucket indices.
    """
    regime_canon = _REGIMES[_regime_idx(regime)]
    return (
        regime_canon,
        _consec_bucket(consecutive_losses),
        _dd_bucket(drawdown_pct),
        _conf_bucket(hmm_confidence),
    )


def _all_states() -> list:
    """Enumerate all 72 discrete state tuples (regime, consec, dd, conf)."""
    return [
        (regime, c, d, hc)
        for regime in _REGIMES
        for c in _CONSEC_BUCKETS
        for d in _DD_BUCKETS
        for hc in _CONF_BUCKETS
    ]


def state_index(
    regime:            str,
    consecutive_losses: int,
    drawdown_pct:      float,
    hmm_confidence:    float,
) -> int:
    """Map (regime, consec_losses, drawdown_pct, hmm_conf) → state index [0, 71]."""
    r  = _regime_idx(regime)
    c  = _consec_bucket(consecutive_losses)
    d  = _dd_bucket(drawdown_pct)
    hc = _conf_bucket(hmm_confidence)
    # Encoding: r * (4*3*2) + c * (3*2) + d * 2 + hc
    return r * 24 + c * 6 + d * 2 + hc


def _value_iteration(m: MDPMemory, n_iter: int = _VI_ITERS) -> None:
    """
    In-place value iteration on MDPMemory.

    CS229 L16: V_{i+1}(s) = max_a [R(s,a) + γ · Σ_{s'} P(s'|s,a)·V_i(s')]
    """
    # Normalised transition matrix
    T_norm = m.T_counts / m.T_counts.sum(axis=2, keepdims=True)   # (S, A, S)
    R_mean = m.R_sum / m.R_count                                    # (S, A)

    V = m.V.copy()
    for _ in range(n_iter):
        Q = R_mean + _GAMMA * (T_norm * V[None, None, :]).sum(axis=2)  # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    m.V = V


class TradeMDP:
    """
    Value-iteration Markov Decision Process for trade sequence sizing.

    On each trade close, the empirical transition model is updated and
    value iteration is re-run (every 10 trades) to keep V* current.

    The size_multiplier from get_size_multiplier() is applied AFTER all
    other size modulators — it is the MDP's learned sequence-aware correction.

    Activation gate: TradeMDP._m.n_trades must be ≥ 20 before the
    orchestrator applies the multiplier (orchestrator code already gates this).
    """

    def __init__(self) -> None:
        self._m = MDPMemory()
        self._try_load()
        # Seed the value function with sensible priors if cold start
        if self._m.n_trades == 0:
            self._seed_priors()
            _value_iteration(self._m, n_iter=_VI_ITERS)

    # ── Serialisation ─────────────────────────────────────────────────── #

    def _try_load(self) -> None:
        if _CHECKPOINT.exists():
            try:
                with open(_CHECKPOINT, "rb") as f:
                    self._m = pickle.load(f)
                logger.debug(f"[TradeMDP] Loaded (n_trades={self._m.n_trades})")
            except Exception as e:
                logger.debug(f"[TradeMDP] Checkpoint load failed (non-fatal): {e}")

    def _save(self) -> None:
        try:
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "wb") as f:
                pickle.dump(self._m, f)
        except Exception as e:
            logger.debug(f"[TradeMDP] Save failed (non-fatal): {e}")

    def _seed_priors(self) -> None:
        """
        Seed the reward matrix with expert-defined priors.

        These encode common-sense risk management intuition:
         - After 3+ losses in high drawdown → reduce size (negative R expected)
         - High confidence MOMENTUM → full size permissible
         - FLAT regime → smaller size (uncertainty premium)
        """
        for r_idx, regime in enumerate(_REGIMES):
            for c in range(4):
                for d in range(3):
                    for hc in range(2):
                        s = r_idx * 24 + c * 6 + d * 2 + hc

                        # Base reward by regime
                        if regime == "MOMENTUM":
                            base_r = 0.40
                        elif regime == "REVERSION":
                            base_r = 0.20
                        else:
                            base_r = -0.05

                        # Consecutive loss penalty
                        consec_pen = -0.15 * c

                        # Drawdown penalty
                        dd_pen = [-0.0, -0.20, -0.40][d]

                        # Confidence bonus
                        conf_bonus = 0.10 if hc == 1 else 0.0

                        r_vec = base_r + consec_pen + dd_pen + conf_bonus
                        # All actions get the same prior reward (MDP learns differentiation)
                        for a in range(N_ACTIONS):
                            # Larger actions get penalised in bad states
                            action_pen = (_ACTIONS[a] - 1.0) * dd_pen * 0.5
                            self._m.R_sum[s, a]   = (r_vec + action_pen) * 10   # × count
                            self._m.R_count[s, a] = 10.0   # weak prior

    # ── Query ─────────────────────────────────────────────────────────── #

    def get_size_multiplier(
        self,
        regime:             str,
        consecutive_losses: int,
        drawdown_pct:       float,
        hmm_confidence:     float,
    ) -> float:
        """
        Return the MDP-recommended size multiplier [0.50, 1.25].

        Uses V*(s) to determine the optimal action:
          Q(s,a) = R(s,a) + γ · Σ P(s'|s,a)·V*(s')
          π*(s)  = argmax_a Q(s,a) → returns _ACTIONS[π*(s)]

        The orchestrator applies this AFTER Kelly sizing.

        Returns:
            float in {0.50, 0.75, 1.00, 1.25}
        """
        s = state_index(regime, consecutive_losses, drawdown_pct, hmm_confidence)

        T_norm = self._m.T_counts / self._m.T_counts.sum(axis=2, keepdims=True)
        R_mean = self._m.R_sum / self._m.R_count
        Q = R_mean[s] + _GAMMA * (T_norm[s] * self._m.V[None, :]).sum(axis=1)

        best_action = int(Q.argmax())
        return float(_ACTIONS[best_action])

    def state_value(
        self,
        regime:             str,
        consecutive_losses: int,
        drawdown_pct:       float,
        hmm_confidence:     float,
    ) -> float:
        """Return V*(s) for the current state (used for logging)."""
        s = state_index(regime, consecutive_losses, drawdown_pct, hmm_confidence)
        return float(self._m.V[s])

    # ── Record transition ─────────────────────────────────────────────── #

    def record_transition(
        self,
        prev_regime:         str,
        prev_consec_losses:  int,
        prev_drawdown_pct:   float,
        prev_hmm_conf:       float,
        next_regime:         str,
        next_consec_losses:  int,
        next_drawdown_pct:   float,
        next_hmm_conf:       float,
        pnl:                 float,
        size_multiplier_used: float,
    ) -> None:
        """
        Update empirical model with one observed (s, a, r, s') tuple.

        CS229 L16: "With the model Psa and rewards Rsa we can use value
        iteration to compute the value function and optimal policy."

        Called by orchestrator.on_trade_close().
        """
        s  = state_index(prev_regime, prev_consec_losses, prev_drawdown_pct, prev_hmm_conf)
        sp = state_index(next_regime, next_consec_losses, next_drawdown_pct, next_hmm_conf)

        # Map the used size_multiplier to the closest action index
        diffs  = [abs(size_multiplier_used - a) for a in _ACTIONS]
        a_idx  = int(np.argmin(diffs))

        # Update transition count and reward sum
        self._m.T_counts[s, a_idx, sp] += 1.0
        self._m.R_sum[s, a_idx]        += float(pnl)
        self._m.R_count[s, a_idx]      += 1.0
        self._m.n_trades               += 1

        # Rerun value iteration every 10 trades
        if self._m.n_trades % 10 == 0:
            _value_iteration(self._m, n_iter=_VI_ITERS)
            self._save()

    def describe(self) -> str:
        return (f"TradeMDP n_trades={self._m.n_trades} "
                f"n_states={N_STATES} n_actions={N_ACTIONS}")
