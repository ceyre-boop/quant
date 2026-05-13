"""
Correlated Position Tracker — Andrew Lo (AFML, Ch. 2 + 9)

Implements two Lo-derived risk controls:

1. Sequential investment information: when recent trades in the same regime
   have a high win rate, we increase the Kelly win_rate estimate. When they've
   been losing, we reduce it. This is the within-session information value of
   correlated bets.

2. Five-level uncertainty gate: Lo's taxonomy of uncertainty levels applied to
   market regimes. Combined with Alexandrian Library convergence to enforce
   minimum uncertainty levels in systemic risk environments.

Lo theory applied (AFML Ch. 9):
  "The information content of correlated sequential bets is not additive —
   the second bet provides less new information than the first when they're
   positively correlated. The effective sample size m_eff = m / (1 + ρ(m-1))
   where ρ is the average pairwise correlation."

  Uncertainty levels:
  Level 1 (Certainty):      Prob distribution known, point estimate reliable → 1.0×
  Level 2 (Risk):           Distribution known, parameter uncertainty only   → 1.0×
  Level 3 (Reducible):      Distribution unknown but can be estimated         → 0.50×
  Level 4 (Partially Red.): Distribution partially observable                → 0.25×
  Level 5 (Knightian):      True uncertainty — no reliable distribution      → 0.0× (halt)

Library convergence integration (Integration Point I1):
  7+ volumes converging → Level 3 minimum (0.50×) regardless of HMM state
  5–6 volumes           → Level 3 minimum (0.50×, more lenient than 7+)
  3+ volumes            → Level 2 permitted (0.75×)

Used by orchestrator:
  CorrelatedPositionTracker.open_position(symbol, regime, direction)
  CorrelatedPositionTracker.record_outcome(symbol, regime, direction, won, pnl)
  CorrelatedPositionTracker.get_win_rate_update(symbol, regime) → WinRateUpdate
  lo_uncertainty_gate(hmm_transition_prob, regime_confidence) → (mult, desc)
  library_adjusted_uncertainty_level(hmm_transition_prob, regime_confidence,
                                     library_insight) → (mult, desc)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Uncertainty level thresholds ─────────────────────────────────────── #
# hmm_transition_prob: P(regime change this bar). High = unstable.
# regime_confidence:   blended three-vote ensemble confidence.
_UNCERTAINTY_LEVELS: List[Tuple[float, float, str, float]] = [
    # (max_transition_prob, min_regime_conf, label, size_mult)
    (0.10, 0.85, "LEVEL_1_CERTAINTY",    1.00),
    (0.25, 0.65, "LEVEL_2_RISK",         1.00),
    (0.45, 0.45, "LEVEL_3_REDUCIBLE",    0.50),
    (0.65, 0.30, "LEVEL_4_PARTIAL",      0.25),
    (1.00, 0.00, "LEVEL_5_KNIGHTIAN",    0.00),   # halt
]


@dataclass
class WinRateUpdate:
    """Result of get_win_rate_update()."""
    win_rate_adjustment: float   # additive δ on Kelly win_rate (e.g. +0.05)
    reason:              str
    n_corr_trades:       int     # number of recent correlated trades used
    effective_sample:    float   # m_eff after Lo correlation adjustment


class CorrelatedPositionTracker:
    """
    Session-level tracker for correlated sequential trade information.

    Tracks open positions and their outcomes within the current session.
    Resets each session so only within-session serial correlation is captured.

    Args:
        lookback (int): Number of recent same-regime trades to consider.
        max_adjustment (float): Maximum win_rate δ applied in either direction.
        corr_decay (float): Assumed serial correlation decay per trade.
    """

    def __init__(
        self,
        lookback:       int   = 5,
        max_adjustment: float = 0.12,
        corr_decay:     float = 0.70,
    ) -> None:
        self._lookback        = lookback
        self._max_adj         = max_adjustment
        self._corr_decay      = corr_decay
        # recent trades keyed by regime: deque of (symbol, direction, won, pnl)
        self._by_regime: Dict[str, deque] = {}
        # open positions: symbol → (regime, direction, open_time)
        self._open: Dict[str, dict] = {}

    # ── Position lifecycle ───────────────────────────────────────────── #

    def open_position(self, symbol: str, regime: str, direction: str) -> None:
        self._open[symbol] = {
            "regime":    regime,
            "direction": direction,
        }

    def record_outcome(
        self,
        symbol:    str,
        regime:    str,
        direction: str,
        won:       bool,
        pnl:       float,
    ) -> None:
        """Record a closed trade into the regime-keyed rolling history."""
        regime = regime or "UNKNOWN"
        if regime not in self._by_regime:
            self._by_regime[regime] = deque(maxlen=self._lookback * 2)
        self._by_regime[regime].append((symbol, direction, won, pnl))
        self._open.pop(symbol, None)

    # ── Win-rate update ─────────────────────────────────────────────── #

    def get_win_rate_update(
        self,
        current_symbol: str,
        current_regime: str,
    ) -> WinRateUpdate:
        """
        Compute win_rate adjustment from recent same-regime trades.

        Lo (AFML Ch. 9): "Each additional correlated bet contributes
        m_eff = 1 / (1 + ρ(n-1)) new information. The adjustment to the
        Kelly win_rate estimate is: δ = (observed_win_rate − prior) × m_eff_weight."

        Returns:
            WinRateUpdate with additive win_rate_adjustment.
            A δ of +0.05 raises the Kelly win_rate from e.g. 0.55 → 0.60.
            A δ of -0.05 lowers it.
        """
        regime = current_regime or "UNKNOWN"
        recent = list(self._by_regime.get(regime, []))

        if not recent:
            return WinRateUpdate(
                win_rate_adjustment=0.0,
                reason="No prior same-regime trades this session",
                n_corr_trades=0,
                effective_sample=0.0,
            )

        # Take the most recent lookback trades in this regime
        window = recent[-self._lookback:]
        n = len(window)

        # Observed win rate in the correlated window
        wins = sum(1 for _, _, won, _ in window if won)
        obs_wr = wins / n

        # Effective sample size (Lo serial correlation adjustment)
        # ρ: we estimate inter-trade correlation as corr_decay^1 for adjacent trades
        rho = self._corr_decay
        m_eff = n / (1.0 + rho * (n - 1)) if n > 1 else 1.0

        # Weight: how much do the correlated trades shift the prior?
        # Ramp from 0 (m_eff=0) to max_adjustment (m_eff=5)
        weight = min(1.0, m_eff / 5.0)

        # Adjustment: deviation from neutral (0.50) scaled by weight
        delta = (obs_wr - 0.50) * weight * self._max_adj / 0.50
        delta = max(-self._max_adj, min(self._max_adj, delta))

        direction_summary = f"{wins}/{n} wins in {regime} regime"
        reason = (f"Lo sequential info [{direction_summary}]: "
                  f"obs_wr={obs_wr:.2f} m_eff={m_eff:.1f} "
                  f"→ win_rate δ={delta:+.3f}")

        return WinRateUpdate(
            win_rate_adjustment=delta,
            reason=reason,
            n_corr_trades=n,
            effective_sample=m_eff,
        )

    def open_position_count(self) -> int:
        return len(self._open)

    def session_win_rate(self, regime: Optional[str] = None) -> Optional[float]:
        """Return session win rate for the given regime, or overall if None."""
        records = []
        if regime:
            records = list(self._by_regime.get(regime, []))
        else:
            for v in self._by_regime.values():
                records.extend(v)
        if not records:
            return None
        wins = sum(1 for _, _, w, _ in records if w)
        return wins / len(records)

    def describe(self) -> str:
        n = sum(len(q) for q in self._by_regime.values())
        return (f"CorrelatedPositionTracker open={len(self._open)} "
                f"session_trades={n} "
                f"regimes={list(self._by_regime.keys())}")


# ── Standalone gate functions (used by orchestrator directly) ─────────── #

def lo_uncertainty_gate(
    hmm_transition_prob: float,
    regime_confidence:   float,
) -> Tuple[float, str]:
    """
    Classify current market into Lo's 5-level uncertainty taxonomy.

    Args:
        hmm_transition_prob: P(regime change this bar) from HMM [0, 1].
                             Low = regime stable. High = transition likely.
        regime_confidence:   Blended three-vote ensemble confidence [0, 1].

    Returns:
        (size_multiplier, description)
        size_multiplier: 0.0 (halt), 0.25, 0.50, or 1.0
    """
    for max_tp, min_rc, label, mult in _UNCERTAINTY_LEVELS:
        if hmm_transition_prob <= max_tp and regime_confidence >= min_rc:
            desc = (f"Lo/{label}: hmm_tp={hmm_transition_prob:.3f} "
                    f"regime_conf={regime_confidence:.3f} → ×{mult:.2f}")
            return mult, desc

    # Fallback: Knightian
    desc = (f"Lo/LEVEL_5_KNIGHTIAN: hmm_tp={hmm_transition_prob:.3f} "
            f"regime_conf={regime_confidence:.3f} → halt")
    return 0.0, desc


def library_adjusted_uncertainty_level(
    hmm_transition_prob: float,
    regime_confidence:   float,
    library_insight,
) -> Tuple[float, str]:
    """
    Library-adjusted Lo uncertainty gate (Integration Point I1).

    When the Alexandrian Library has 7+ volumes converging above 0.60
    similarity, the macro regime is in systemic stress — uncertainty is
    elevated regardless of the HMM point estimate. The Library overrides
    the HMM-based level to a more conservative minimum.

    Args:
        hmm_transition_prob: HMM regime transition probability.
        regime_confidence:   Three-vote blended ensemble confidence.
        library_insight:     LibraryInsight from AlexandrianLibrary.query(),
                             or None if library not loaded.

    Returns:
        (size_multiplier, description)
    """
    # Start with the raw Lo gate
    base_mult, base_desc = lo_uncertainty_gate(hmm_transition_prob, regime_confidence)

    if library_insight is None:
        return base_mult, base_desc

    # Count converging volumes
    converging = 0
    try:
        for vm in library_insight.volume_matches:
            if vm.similarity >= 0.60:
                converging += 1
    except Exception:
        return base_mult, base_desc

    # Apply Library override: force minimum uncertainty level
    if converging >= 7:
        forced_mult = 0.50
        label = "LIBRARY_7+_VOLUMES"
    elif converging >= 5:
        forced_mult = 0.50
        label = "LIBRARY_5-6_VOLUMES"
    elif converging >= 3:
        forced_mult = 0.75
        label = "LIBRARY_3-4_VOLUMES"
    else:
        return base_mult, base_desc

    # Library can only reduce size, never increase it
    effective_mult = min(base_mult, forced_mult) if base_mult > 0.0 else 0.0
    desc = (f"Lo/{label}: converging={converging} "
            f"base_mult={base_mult:.2f} → library_forced={effective_mult:.2f} | "
            f"{base_desc}")
    return effective_mult, desc
