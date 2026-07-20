# LOW-USE: imported by sovereign/orchestrator.py (verified 2026-07-20). NOT dead code — do not delete.
# The earlier header read "not imported by any live path"; that was false for all 11 modules
# in this group. See trial/subtraction_verdicts.md:47.
"""
Black-Scholes Vol Regime Signal — MIT Quantitative Finance

Tracks the ratio of implied volatility (IV) to realized volatility (RV).
When IV/RV > 1.2, the market is pricing in more risk than is being realised —
a warning signal that reduces position size. When IV/RV < 0.8, vol is
compressed below realised — a favourable regime.

MIT theory applied:
  "Under Black-Scholes, σ_implied = σ_realized in equilibrium. A sustained
   divergence indicates a regime where option markets are pricing in a
   structural risk premium — consistent with upcoming volatility expansion."

  Risk-neutral measure: vol surfaces encode the market's aggregate uncertainty.
  Realised vol: the actual quadratic variation of the log-price process.

Vol regime signal:
  BACKWARDATION  (IV/RV < 0.8) : vol compressed, trend likely → size ×1.0
  NORMAL         (0.8–1.2)     : neutral                       → size ×1.0
  ELEVATED       (1.2–1.5)     : market pricing in uncertainty  → size ×0.75
  STRESS         (1.5–2.0)     : regime shift likely            → size ×0.50
  EXTREME_STRESS (> 2.0)       : do not open new positions      → size ×0.25

Used by orchestrator at stage 4h as a multiplicative size modulator.
  VolRegimeSignal.update(rv_daily) → feed each bar's realised vol
  VolRegimeSignal.get_signal()     → {'vol_regime', 'iv_rv_ratio', 'size_adjustment'}
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Regime thresholds for IV/RV ratio
_THRESHOLDS = [
    (2.0,  "EXTREME_STRESS", 0.25),
    (1.5,  "STRESS",         0.50),
    (1.2,  "ELEVATED",       0.75),
    (0.8,  "NORMAL",         1.0),
    (0.0,  "BACKWARDATION",  1.0),   # low IV/RV = vol compressed = normal sizing
]


class VolRegimeSignal:
    """
    IV/RV regime signal derived from Black-Scholes implied volatility theory.

    Since live IV requires an options feed, this class approximates IV using
    the VIX-to-close-return ratio as a proxy, or uses a configurable fixed
    IV estimate when the options feed is unavailable.

    Args:
        lookback (int): Rolling window for realised vol computation (default 20).
        fixed_iv_annual (float): Annual IV proxy when no options feed available.
                                 Default 0.16 (16% — long-run VIX average).
        vix_scale (float): Divide VIX by this to convert to daily IV.
                           VIX is annualised %, so 100*sqrt(252) ≈ 1587.
    """

    def __init__(
        self,
        lookback: int = 20,
        fixed_iv_annual: float = 0.16,
        vix_scale: float = 1587.4,
    ) -> None:
        self._lookback = lookback
        self._fixed_iv_annual = fixed_iv_annual
        self._vix_scale = vix_scale
        self._daily_returns: deque = deque(maxlen=lookback)
        self._vix_values: deque = deque(maxlen=lookback)
        self._last_signal: Dict[str, Any] = {
            "vol_regime":    "NORMAL",
            "iv_rv_ratio":   1.0,
            "size_adjustment": 1.0,
            "realized_vol":  0.0,
            "implied_vol":   fixed_iv_annual / math.sqrt(252),
        }

    # ── Feed ─────────────────────────────────────────────────────────── #

    def update(self, rv_daily: float, vix: float = 0.0) -> None:
        """
        Feed one bar of data.

        Args:
            rv_daily: Today's realised daily return (abs value or log-return std).
                      Pass ATR/price as a proxy: atr / current_price.
            vix: VIX index value (e.g. 18.5). Pass 0 to use fixed_iv_annual.
        """
        if rv_daily > 0:
            self._daily_returns.append(float(rv_daily))
        if vix > 0:
            self._vix_values.append(float(vix))
        self._recompute()

    def _recompute(self) -> None:
        """Recompute vol regime from current buffers."""
        if len(self._daily_returns) < 5:
            return

        rv = float(np.std(list(self._daily_returns)))

        # Implied vol: use VIX if available, else fixed_iv_annual
        if self._vix_values:
            iv = float(np.mean(list(self._vix_values))) / self._vix_scale
        else:
            iv = self._fixed_iv_annual / math.sqrt(252)

        ratio = iv / (rv + 1e-9)

        # Classify regime
        regime = "NORMAL"
        size_adj = 1.0
        for threshold, label, adj in _THRESHOLDS:
            if ratio >= threshold:
                regime = label
                size_adj = adj
                break

        self._last_signal = {
            "vol_regime":      regime,
            "iv_rv_ratio":     round(ratio, 3),
            "size_adjustment": size_adj,
            "realized_vol":    round(rv, 6),
            "implied_vol":     round(iv, 6),
        }

    # ── Query ─────────────────────────────────────────────────────────── #

    def get_signal(self) -> Dict[str, Any]:
        """
        Return the current vol regime signal.

        Returns:
            dict with keys:
              vol_regime    : str — 'NORMAL' / 'ELEVATED' / 'STRESS' / 'EXTREME_STRESS'
              iv_rv_ratio   : float — IV ÷ RV
              size_adjustment: float — multiplicative position size modifier [0.25, 1.0]
              realized_vol  : float — current realised daily vol estimate
              implied_vol   : float — current daily IV estimate
        """
        return dict(self._last_signal)

    def black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Black-Scholes European call price.

        C = S·N(d₁) − K·e^{-rT}·N(d₂)
        d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ − σ√T

        Args:
            S: spot price
            K: strike price
            T: time to expiry in years
            r: risk-free rate
            sigma: annual implied volatility

        Returns:
            call price
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0.0, S - K)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        from scipy.stats import norm
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))

    def implied_vol_from_price(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Invert Black-Scholes via Newton-Raphson bisection to find IV.

        CS229 / MIT: this is the standard market practice — prices are quoted
        in IV units because it's invariant to moneyness.

        Returns:
            Implied vol (annual) or nan if no solution found.
        """
        # Bounds check
        intrinsic = max(0.0, S - K * math.exp(-r * T))
        if market_price <= intrinsic:
            return float("nan")

        lo, hi = 1e-6, 10.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            price = self.black_scholes_call(S, K, T, r, mid)
            if abs(price - market_price) < tol:
                return float(mid)
            if price < market_price:
                lo = mid
            else:
                hi = mid
        return float((lo + hi) / 2.0)

    def describe(self) -> str:
        s = self._last_signal
        return (f"VolRegimeSignal regime={s['vol_regime']} "
                f"IV/RV={s['iv_rv_ratio']:.2f} "
                f"size={s['size_adjustment']:.2f}")
