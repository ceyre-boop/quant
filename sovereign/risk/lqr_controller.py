"""
LQR Position Controller — CS229 Lecture 18 (Linear Quadratic Regulator)

Drawdown-aware position sizing via the Riccati equation.
Models trading as a linear dynamical system where the "state" is the
current P&L path, and the "control" is position size (the action we can take).

CS229 L18 theory applied:
  "The LQR solves: min_u Σ [xᵀQx + uᵀRu]
   subject to: x_{t+1} = Ax_t + Bu_t
   Solution: P = solve Riccati equation
             K = -(BᵀPB + R)⁻¹ BᵀPA  (optimal feedback gain)
             u* = Kx_t                 (optimal control)"

Applied to trading:
  State vector x = [drawdown_pct, rolling_pnl_3d, consecutive_losses, kelly_fraction]
  Control u = position size multiplier deviation from 1.0 (i.e., how much we adjust)
  A = identity (state evolves by itself)
  B = control influence matrix (position size affects drawdown path)
  Q = state cost (penalises high drawdown / consecutive losses)
  R = control cost (penalises aggressive deviations from 1.0)

The Riccati solution K maps the current state to a size adjustment, which is
then applied as: size_mult = 1.0 + lqr_delta, clamped to [0.25, 1.25].

Used by orchestrator at stage 4h:
  LQRController.compute_size_multiplier(drawdown_pct, rolling_pnl_3d,
                                        consecutive_losses, kelly_fraction,
                                        base_multiplier) → (mult, debug_dict)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_N_STATE   = 4   # state dimension: [drawdown_pct, rolling_pnl_3d, consec_losses, kelly_frac]
_N_CONTROL = 1   # control: scalar size multiplier deviation


class LQRController:
    """
    Linear Quadratic Regulator for drawdown-aware position sizing.

    The LQR feedback gain K is computed once via the Riccati equation at
    construction time (deterministic for given Q, R, A, B matrices).
    Each call to compute_size_multiplier() applies K to the current state.

    This is computationally trivial at inference time (one matrix multiply).

    Args:
        horizon (int): MPC planning horizon (not used for infinite-horizon LQR
                       but kept for API compatibility).
        q_drawdown (float): Penalty weight on drawdown_pct in the cost matrix.
        q_consec   (float): Penalty weight on consecutive_losses.
        q_pnl      (float): Penalty weight on negative rolling PnL.
        r_control  (float): Control effort penalty (higher = more conservative).
    """

    def __init__(
        self,
        horizon:    int   = 10,
        q_drawdown: float = 2.0,
        q_consec:   float = 1.5,
        q_pnl:      float = 0.5,
        r_control:  float = 1.0,
    ) -> None:
        self._horizon = horizon

        # State transition: A ≈ identity (state persists between trades)
        self._A = np.eye(_N_STATE, dtype=np.float64)

        # Control matrix B: how does position size affect each state component?
        # Larger position → more drawdown risk and more PnL variance
        self._B = np.array([[0.40],   # drawdown_pct sensitivity
                             [0.20],   # rolling_pnl sensitivity
                             [0.10],   # consecutive_losses (lower impact)
                             [0.05]],  # kelly_fraction (minimal)
                            dtype=np.float64)

        # State cost Q: penalise being in bad states
        # Diagonal: [drawdown, rolling_pnl_cost, consec_losses, kelly_frac]
        self._Q = np.diag([
            q_drawdown,
            q_pnl,
            q_consec,
            0.10,          # small penalty on high Kelly (already gated)
        ]).astype(np.float64)

        # Control cost R: penalise large deviations from neutral sizing
        self._R = np.array([[r_control]], dtype=np.float64)

        # Solve the discrete-time algebraic Riccati equation
        self._K = self._solve_riccati()

    # ── Riccati equation solver ──────────────────────────────────────── #

    def _solve_riccati(self) -> np.ndarray:
        """
        Solve discrete-time algebraic Riccati equation via iteration.

        P = Q + AᵀPA − AᵀPB(BᵀPB + R)⁻¹BᵀPA
        K = −(BᵀPB + R)⁻¹BᵀPA

        Returns:
            K: (n_control, n_state) = (1, 4) feedback gain matrix.
        """
        P = self._Q.copy()
        A, B, Q, R = self._A, self._B, self._Q, self._R

        for _ in range(1000):
            BtP   = B.T @ P
            BtPB  = BtP @ B
            M     = BtPB + R
            M_inv = np.linalg.inv(M)
            P_new = Q + A.T @ P @ A - A.T @ P @ B @ M_inv @ BtP @ A
            if np.max(np.abs(P_new - P)) < 1e-10:
                P = P_new
                break
            P = P_new

        K = -np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A
        logger.debug(f"[LQR] Riccati solved. K={K.round(4)}")
        return K

    # ── Compute size multiplier ──────────────────────────────────────── #

    def compute_size_multiplier(
        self,
        drawdown_pct:       float,
        rolling_pnl_3d:     float,
        consecutive_losses: int,
        kelly_fraction:     float,
        base_multiplier:    float = 1.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply LQR feedback control to compute the size multiplier.

        State vector: x = [drawdown_pct, -rolling_pnl_3d (cost=negative pnl),
                           consecutive_losses, kelly_fraction]

        The LQR control u* = K·x gives the DEVIATION from the base multiplier.
        Final multiplier = base_multiplier + lqr_delta, clamped to [0.25, 1.25].

        Rationale: when drawdown and consecutive losses are high, the LQR
        automatically reduces position size to preserve capital. When conditions
        are favourable, it allows full sizing.

        Args:
            drawdown_pct:       Current drawdown from high water mark [0, 1].
            rolling_pnl_3d:     3-day rolling PnL (positive = good).
            consecutive_losses: Number of consecutive losses so far.
            kelly_fraction:     Kelly sizing fraction used [0, 0.04].
            base_multiplier:    Starting multiplier before LQR adjustment.

        Returns:
            (size_multiplier, debug_dict)
        """
        # Build state vector
        # Note: rolling_pnl enters as negative cost (lower PnL = worse state)
        x = np.array([
            float(drawdown_pct),
            float(-rolling_pnl_3d),   # cost representation (positive = bad)
            float(consecutive_losses) / 3.0,   # normalise
            float(kelly_fraction) * 10.0,       # scale to similar range
        ], dtype=np.float64)

        # LQR control: u* = K·x   (scalar)
        u = float((self._K @ x)[0])

        # Map control to size delta: negative u = reduce, positive = increase
        # Scale to a [-0.50, +0.25] adjustment range (asymmetric for safety)
        lqr_delta = float(np.clip(u, -0.50, 0.25))

        final_mult = float(np.clip(base_multiplier + lqr_delta, 0.25, 1.25))

        debug = {
            "state_x":          x.round(4).tolist(),
            "lqr_u":            round(u, 4),
            "lqr_delta":        round(lqr_delta, 4),
            "base_multiplier":  round(base_multiplier, 4),
            "final_multiplier": round(final_mult, 4),
        }

        return final_mult, debug

    def describe(self) -> str:
        return (f"LQRController n_state={_N_STATE} n_control={_N_CONTROL} "
                f"K_norm={float(np.linalg.norm(self._K)):.4f}")
