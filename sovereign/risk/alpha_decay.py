"""
Alpha Decay Monitor — Ernest Chan (Algorithmic Trading, Ch. 3)

Tracks whether edge is alive, degrading, or dead by fitting an exponential
decay curve to the rolling Sharpe ratio over time.

Chan: "When alpha decays, tweak risk management — not the signal. The signal
may still be valid on longer horizons even as short-term autocorrelation
decays. What you adjust is position sizing, not strategy selection."

Half-life calculation:
  Sharpe(t) = S₀ · e^{-λt} + noise
  Half-life = ln(2) / λ

Decay levels:
  STRONG     (rolling ≥ baseline)           → multiplier 1.0  (full size)
  NORMAL     (≥60% of baseline)             → multiplier 1.0
  DEGRADED   (30%–60% of baseline)          → multiplier 0.70 (size reduction)
  CRITICAL   (< 30% of baseline)            → multiplier 0.40 (emergency reduction)
  DEAD       (rolling Sharpe < 0)           → multiplier 0.0  (halt)

Used by orchestrator at stage 4g:
  AlphaDecayMonitor(strategy='momentum')  — instantiated per strategy
  monitor.record_trade(r_multiple)        — feed each closed trade
  monitor.check()                         → DecayOutput(multiplier, level, reason)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT        = Path(__file__).resolve().parent.parent.parent
_LEDGER_DIR  = _ROOT / "data" / "ledger"


@dataclass
class DecayOutput:
    """Result of AlphaDecayMonitor.check()."""
    multiplier:  float     # position size multiplier [0.0, 1.0]
    level:       str       # 'STRONG' / 'NORMAL' / 'DEGRADED' / 'CRITICAL' / 'DEAD'
    reason:      str       # human-readable explanation
    rolling_sharpe: float  # most recent rolling Sharpe
    baseline_sharpe: float # strategy baseline Sharpe
    half_life_trades: float  # estimated edge half-life in trades (nan if insufficient)


class AlphaDecayMonitor:
    """
    Rolling Sharpe decay tracker with exponential fit.

    Reads from the trade ledger so that historical trades accumulated in
    prior sessions are reflected automatically — no additional record call
    is needed beyond writing to the ledger via TradeLedger.log_close().

    The orchestrator also calls record_trade() for the current-session
    trades before the ledger write completes (for real-time check).

    Args:
        strategy (str): 'momentum' or 'reversion' — filters ledger records.
        window (int): Rolling Sharpe window (default 20 trades).
        baseline_sharpe (float): Expected Sharpe in favourable conditions.
                                 Default 0.6 (conservative for live).
        halt_threshold (float): Rolling Sharpe below this halts trading.
                                Default 0.0 (negative expectancy).
    """

    def __init__(
        self,
        strategy: str = "momentum",
        window: int = 20,
        baseline_sharpe: float = 0.6,
        halt_threshold: float = 0.0,
    ) -> None:
        self._strategy       = strategy
        self._window         = window
        self._baseline       = baseline_sharpe
        self._halt_threshold = halt_threshold
        self._r_history: List[float] = []   # R-multiples, newest last
        self._load_from_ledger()

    # ── Ledger bootstrap ─────────────────────────────────────────────── #

    def _load_from_ledger(self) -> None:
        """Load closed trade R-multiples for this strategy from ledger."""
        if not _LEDGER_DIR.exists():
            return
        try:
            records = []
            for f in sorted(_LEDGER_DIR.glob("trade_ledger_*.jsonl")):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    if t.get("status") != "closed":
                        continue
                    strat = t.get("strategy", "")
                    if self._strategy not in strat and strat not in self._strategy:
                        continue
                    pnl = float(t.get("pnl", 0.0))
                    # Convert $ PnL to approximate R-multiple
                    sl_dist = abs(float(t.get("entry_price", 1)) -
                                  float(t.get("sl", 1)))
                    size    = float(t.get("size", 1.0))
                    r_risk  = sl_dist * size
                    r_mult  = pnl / (r_risk + 1e-9)
                    r_mult  = max(-10.0, min(10.0, r_mult))  # clip outliers
                    records.append((t.get("exit_time", ""), r_mult))
            # Sort chronologically
            records.sort(key=lambda x: x[0])
            self._r_history = [r for _, r in records]
            if self._r_history:
                logger.debug(f"[AlphaDecay/{self._strategy}] "
                             f"Loaded {len(self._r_history)} trades from ledger")
        except Exception as e:
            logger.debug(f"[AlphaDecay] Ledger load failed (non-fatal): {e}")

    # ── Feed ─────────────────────────────────────────────────────────── #

    def record_trade(self, r_multiple: float) -> None:
        """
        Append a single trade's R-multiple to the rolling history.

        Args:
            r_multiple: +2.0 for a trade that hit 2R, -1.0 for a full stop.
        """
        self._r_history.append(float(r_multiple))

    # ── Rolling Sharpe ───────────────────────────────────────────────── #

    def rolling_sharpe_series(self) -> List[float]:
        """
        Compute rolling Sharpe ratio over the full history.

        Returns a list of Sharpe estimates, one per window-step.
        Annualised assuming 252 trades/year (adjust if needed).
        """
        sharpes = []
        for i in range(self._window, len(self._r_history) + 1):
            seg = self._r_history[i - self._window: i]
            mu  = float(np.mean(seg))
            sd  = float(np.std(seg))
            # Annualised Sharpe: each trade is ~1 observation
            s = (mu / (sd + 1e-9)) * math.sqrt(252)
            sharpes.append(s)
        return sharpes

    def current_rolling_sharpe(self) -> Optional[float]:
        """Return the most recent rolling Sharpe, or None if insufficient data."""
        if len(self._r_history) < self._window:
            return None
        seg = self._r_history[-self._window:]
        mu  = float(np.mean(seg))
        sd  = float(np.std(seg))
        return (mu / (sd + 1e-9)) * math.sqrt(252)

    # ── Decay fit ────────────────────────────────────────────────────── #

    def decay_fit(self) -> dict:
        """
        Fit exponential decay model to rolling Sharpe series.

        Model: Sharpe(t) = S₀ · exp(−λ · t)
        Half-life = ln(2) / λ

        Returns dict with: status, initial_sharpe, decay_rate, half_life_trades,
        current_sharpe, baseline_sharpe, alert.
        """
        sharpes = self.rolling_sharpe_series()
        if len(sharpes) < 5:
            return {"status": "insufficient_data", "n": len(sharpes)}

        t = np.arange(len(sharpes), dtype=float)
        try:
            from scipy.optimize import curve_fit

            def _decay(t, s0, lam):
                return s0 * np.exp(-lam * t)

            popt, _ = curve_fit(
                _decay, t, sharpes,
                p0=[self._baseline, 0.01],
                maxfev=5000,
                bounds=([-10, -1], [10, 2]),
            )
            s0, lam = float(popt[0]), float(popt[1])
            half_life = math.log(2) / lam if lam > 1e-9 else float("inf")

            return {
                "status":             "ok",
                "initial_sharpe":     round(s0, 3),
                "decay_rate":         round(lam, 5),
                "half_life_trades":   round(half_life, 1),
                "current_sharpe":     round(sharpes[-1], 3),
                "baseline_sharpe":    self._baseline,
                "alert":              sharpes[-1] < self._baseline * 0.6,
            }
        except Exception:
            return {
                "status":           "fit_failed",
                "current_sharpe":   round(sharpes[-1], 3) if sharpes else None,
            }

    def z_score_vs_baseline(self) -> float:
        """How many std-devs is current rolling Sharpe from baseline?"""
        sharpes = self.rolling_sharpe_series()
        if len(sharpes) < 5:
            return 0.0
        recent_mu = float(np.mean(sharpes[-10:]))
        se        = float(np.std(sharpes)) / math.sqrt(len(sharpes)) + 1e-9
        return (recent_mu - self._baseline) / se

    # ── Gate check ───────────────────────────────────────────────────── #

    def check(self) -> DecayOutput:
        """
        Return the current decay assessment and position size multiplier.

        Used by orchestrator stage 4g:
          if decay.multiplier == 0.0 → halt (veto trade)
          elif decay.multiplier < 1.0 → reduce position size

        Returns:
            DecayOutput with multiplier, level, reason, rolling_sharpe.
        """
        n = len(self._r_history)
        half_life = float("nan")
        rolling_s = self.current_rolling_sharpe()

        if rolling_s is None:
            return DecayOutput(
                multiplier=1.0,
                level="INSUFFICIENT_DATA",
                reason=f"AlphaDecay [{self._strategy}]: {n}/{self._window} trades — no gate applied",
                rolling_sharpe=0.0,
                baseline_sharpe=self._baseline,
                half_life_trades=half_life,
            )

        # Decay fit for half-life info (best-effort)
        fit = self.decay_fit()
        if fit.get("status") == "ok":
            half_life = fit.get("half_life_trades", float("nan"))

        # Classify decay level
        ratio = rolling_s / (self._baseline + 1e-9)

        if rolling_s <= self._halt_threshold:
            level = "DEAD"
            mult  = 0.0
            reason = (f"ALPHA_DECAY_HALT [{self._strategy}]: rolling_sharpe={rolling_s:.3f} "
                      f"≤ halt_threshold={self._halt_threshold} | "
                      f"baseline={self._baseline:.2f}")
        elif ratio < 0.30:
            level = "CRITICAL"
            mult  = 0.40
            reason = (f"ALPHA_DECAY_CRITICAL [{self._strategy}]: "
                      f"rolling={rolling_s:.3f} = {ratio*100:.0f}% of baseline")
        elif ratio < 0.60:
            level = "DEGRADED"
            mult  = 0.70
            reason = (f"ALPHA_DECAY_DEGRADED [{self._strategy}]: "
                      f"rolling={rolling_s:.3f} = {ratio*100:.0f}% of baseline")
        elif ratio < 1.0:
            level = "NORMAL"
            mult  = 1.0
            reason = (f"AlphaDecay [{self._strategy}]: NORMAL "
                      f"rolling={rolling_s:.3f} baseline={self._baseline:.2f}")
        else:
            level = "STRONG"
            mult  = 1.0
            reason = (f"AlphaDecay [{self._strategy}]: STRONG "
                      f"rolling={rolling_s:.3f} (+{(ratio-1)*100:.0f}% above baseline)")

        return DecayOutput(
            multiplier=mult,
            level=level,
            reason=reason,
            rolling_sharpe=rolling_s,
            baseline_sharpe=self._baseline,
            half_life_trades=half_life,
        )

    def describe(self) -> str:
        result = self.check()
        return (f"AlphaDecayMonitor strategy={self._strategy} "
                f"level={result.level} "
                f"rolling_sharpe={result.rolling_sharpe:.3f} "
                f"n_trades={len(self._r_history)}")
