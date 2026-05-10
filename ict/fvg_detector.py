"""
ict/fvg_detector.py
====================
Fair Value Gap (FVG) and Order Block detector for the ICT micro-edge engine.

Fair Value Gap (3-candle imbalance):
  Bullish FVG: candle[i].High < candle[i+2].Low  → gap between c1 top and c3 bottom
  Bearish FVG: candle[i].Low  > candle[i+2].High → gap between c1 bottom and c3 top

Order Block (OB):
  Bullish OB: last bearish candle before a strong bullish impulse (≥ N × ATR)
  Bearish OB: last bullish candle before a strong bearish impulse (≥ N × ATR)

This module wraps the primitive logic from `sovereign/forex/ict_engine.py`
and returns clean typed results for use in the ICT pipeline.

ISOLATION: No imports from sovereign/risk, layer1, layer2, layer3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_FVG_MIN_ATR = 0.3
_DEFAULT_FVG_MAX_AGE = 50
_DEFAULT_FVG_TAP_PROXIMITY = 0.5
_DEFAULT_OB_LOOKBACK = 20
_DEFAULT_OB_IMPULSE_ATR = 1.5


# ── Data classes ─────────────────────────────────────────────────────────── #

@dataclass
class FVGResult:
    """A detected, unfilled Fair Value Gap."""
    kind: str           # 'BULLISH' | 'BEARISH'
    top: float
    bottom: float
    midpoint: float
    formed_at: pd.Timestamp
    size: float
    size_atr_ratio: float
    filled: bool
    age_bars: int       # bars since formation (relative to as_of)

    @property
    def is_bullish(self) -> bool:
        return self.kind == "BULLISH"

    @property
    def is_bearish(self) -> bool:
        return self.kind == "BEARISH"

    def price_tapping(self, price: float, proximity_fraction: float = 0.5) -> bool:
        """Return True if *price* is within *proximity_fraction* × size of the gap."""
        margin = proximity_fraction * self.size
        if self.is_bullish:
            return self.bottom - margin <= price <= self.top + margin
        return self.bottom - margin <= price <= self.top + margin


@dataclass
class OrderBlockResult:
    """A detected Order Block."""
    kind: str           # 'BULLISH' | 'BEARISH'
    high: float
    low: float
    midpoint: float
    formed_at: pd.Timestamp
    valid: bool
    is_breaker: bool    # True if price has since broken through (invalidated OB → breaker)
    impulse_atr_ratio: float

    @property
    def is_bullish(self) -> bool:
        return self.kind == "BULLISH"

    @property
    def is_bearish(self) -> bool:
        return self.kind == "BEARISH"


# ── Detector ─────────────────────────────────────────────────────────────── #

class FVGDetector:
    """
    Detect Fair Value Gaps and Order Blocks on an OHLCV DataFrame.

    Usage::

        det = FVGDetector()
        fvgs, obs = det.detect(df)
        bullish_fvgs = [f for f in fvgs if f.is_bullish and not f.filled]
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg = self._load_config(config_path)
        fvg_cfg = cfg.get("fvg", {})
        ob_cfg = cfg.get("order_block", {})

        self._fvg_min_atr: float = fvg_cfg.get("min_size_atr_fraction", _DEFAULT_FVG_MIN_ATR)
        self._fvg_max_age: int = fvg_cfg.get("max_age_bars", _DEFAULT_FVG_MAX_AGE)
        self._tap_proximity: float = fvg_cfg.get("tap_proximity_fraction", _DEFAULT_FVG_TAP_PROXIMITY)
        self._ob_lookback: int = ob_cfg.get("lookback_bars", _DEFAULT_OB_LOOKBACK)
        self._ob_impulse_atr: float = ob_cfg.get("impulse_atr_multiple", _DEFAULT_OB_IMPULSE_ATR)

    # ── Public API ─────────────────────────────────────────────────────── #

    def detect(
        self,
        df: pd.DataFrame,
        as_of_idx: int = -1,
    ) -> Tuple[List[FVGResult], List[OrderBlockResult]]:
        """
        Run FVG and OB detection on *df*.

        Args:
            df: OHLCV DataFrame (uppercase or lowercase column names accepted).
            as_of_idx: Bar index to treat as "now" (-1 = last bar).

        Returns:
            (fvgs, order_blocks) — both lists sorted newest first.
        """
        df = self._normalise(df)
        if len(df) < 5:
            return [], []

        as_of_idx = as_of_idx if as_of_idx >= 0 else len(df) - 1
        price_now = float(df["Close"].iloc[as_of_idx])
        atr = self._atr(df)

        fvgs = self._find_fvgs(df, atr, price_now, as_of_idx)
        obs = self._find_obs(df, atr, price_now, as_of_idx)

        fvgs.sort(key=lambda f: f.formed_at, reverse=True)
        obs.sort(key=lambda o: o.formed_at, reverse=True)
        return fvgs, obs

    def nearest_actionable(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[FVGResult], Optional[FVGResult], Optional[OrderBlockResult], Optional[OrderBlockResult]]:
        """
        Return (bullish_fvg, bearish_fvg, bullish_ob, bearish_ob) closest to current price.

        Bullish items are below price (potential support / re-entry zones).
        Bearish items are above price (potential resistance / short zones).
        """
        fvgs, obs = self.detect(df)
        price = float(df["Close"].iloc[-1])

        bull_fvgs = [f for f in fvgs if f.is_bullish and not f.filled and f.top < price]
        bear_fvgs = [f for f in fvgs if f.is_bearish and not f.filled and f.bottom > price]
        bull_obs = [o for o in obs if o.is_bullish and o.valid and not o.is_breaker and o.high < price]
        bear_obs = [o for o in obs if o.is_bearish and o.valid and not o.is_breaker and o.low > price]

        nearest_bull_fvg = max(bull_fvgs, key=lambda f: f.top) if bull_fvgs else None
        nearest_bear_fvg = min(bear_fvgs, key=lambda f: f.bottom) if bear_fvgs else None
        nearest_bull_ob = max(bull_obs, key=lambda o: o.high) if bull_obs else None
        nearest_bear_ob = min(bear_obs, key=lambda o: o.low) if bear_obs else None

        return nearest_bull_fvg, nearest_bear_fvg, nearest_bull_ob, nearest_bear_ob

    # ── FVG detection ─────────────────────────────────────────────────── #

    def _find_fvgs(
        self, df: pd.DataFrame, atr: float, price_now: float, as_of: int
    ) -> List[FVGResult]:
        results: List[FVGResult] = []
        lookback = min(self._fvg_max_age, as_of - 1)
        start = as_of - lookback

        for i in range(start, as_of - 1):
            c1 = df.iloc[i]
            c3 = df.iloc[i + 2]
            age = as_of - (i + 1)

            # Bullish FVG: gap between c1.High and c3.Low
            if c1["High"] < c3["Low"]:
                top = float(c3["Low"])
                bottom = float(c1["High"])
                size = top - bottom
                if size >= self._fvg_min_atr * atr:
                    results.append(FVGResult(
                        kind="BULLISH",
                        top=top,
                        bottom=bottom,
                        midpoint=(top + bottom) / 2,
                        formed_at=df.index[i + 1],
                        size=size,
                        size_atr_ratio=size / atr if atr > 0 else 0.0,
                        filled=price_now < bottom,
                        age_bars=age,
                    ))

            # Bearish FVG: gap between c1.Low and c3.High
            if c1["Low"] > c3["High"]:
                top = float(c1["Low"])
                bottom = float(c3["High"])
                size = top - bottom
                if size >= self._fvg_min_atr * atr:
                    results.append(FVGResult(
                        kind="BEARISH",
                        top=top,
                        bottom=bottom,
                        midpoint=(top + bottom) / 2,
                        formed_at=df.index[i + 1],
                        size=size,
                        size_atr_ratio=size / atr if atr > 0 else 0.0,
                        filled=price_now > top,
                        age_bars=age,
                    ))

        return results

    # ── Order Block detection ─────────────────────────────────────────── #

    def _find_obs(
        self, df: pd.DataFrame, atr: float, price_now: float, as_of: int
    ) -> List[OrderBlockResult]:
        results: List[OrderBlockResult] = []
        threshold = self._ob_impulse_atr * atr
        lookback = min(self._ob_lookback, as_of - 1)
        start = as_of - lookback

        for i in range(start, as_of):
            c_ob = df.iloc[i]
            c_imp = df.iloc[i + 1] if i + 1 <= as_of else None
            if c_imp is None:
                break

            impulse_size = abs(float(c_imp["Close"]) - float(c_imp["Open"]))

            # Bullish OB: last bearish candle before strong bullish impulse
            if (c_ob["Close"] < c_ob["Open"]
                    and c_imp["Close"] > c_imp["Open"]
                    and impulse_size >= threshold):
                high = float(c_ob["High"])
                low = float(c_ob["Low"])
                results.append(OrderBlockResult(
                    kind="BULLISH",
                    high=high,
                    low=low,
                    midpoint=(high + low) / 2,
                    formed_at=df.index[i],
                    valid=price_now > low,
                    is_breaker=price_now < low,
                    impulse_atr_ratio=impulse_size / atr if atr > 0 else 0.0,
                ))

            # Bearish OB: last bullish candle before strong bearish impulse
            if (c_ob["Close"] > c_ob["Open"]
                    and c_imp["Close"] < c_imp["Open"]
                    and impulse_size >= threshold):
                high = float(c_ob["High"])
                low = float(c_ob["Low"])
                results.append(OrderBlockResult(
                    kind="BEARISH",
                    high=high,
                    low=low,
                    midpoint=(high + low) / 2,
                    formed_at=df.index[i],
                    valid=price_now < high,
                    is_breaker=price_now > high,
                    impulse_atr_ratio=impulse_size / atr if atr > 0 else 0.0,
                ))

        return results

    # ── Helpers ────────────────────────────────────────────────────────── #

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rename = {c: c.capitalize()
                  for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename)
        return df[["Open", "High", "Low", "Close"]].dropna()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        val = float(tr.rolling(period).mean().iloc[-1])
        return val if val > 0 else 1.0

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using defaults", path)
            return {}


def _default_config_path() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
