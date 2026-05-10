"""
ict/sweep_detector.py
=====================
Liquidity sweep detector for the ICT micro-edge engine.

A liquidity sweep (stop hunt) occurs when price:
  • Wicks below a key swing low (Buy-Side SSL sweep → expect upward reversal)
  • Wicks above a key swing high (Sell-Side BSL sweep → expect downward reversal)

This module wraps the primitive detection logic from `sovereign/forex/ict_engine.py`
and produces typed `SweepResult` objects suitable for use in the ICT pipeline.

ISOLATION: No imports from sovereign/risk, layer1, layer2, layer3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import yaml

from ict._atr_utils import compute_atr

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK = 20
_DEFAULT_REVERSAL_BARS = 3
_DEFAULT_MIN_WICK_ATR = 0.3


@dataclass(frozen=True)
class SweepResult:
    """A detected liquidity sweep event."""
    direction: str              # 'BULLISH_SWEEP' | 'BEARISH_SWEEP'
    swept_level: float          # SSL (for bullish) or BSL (for bearish)
    wick_low: float             # lowest point of the sweep candle
    wick_high: float            # highest point of the sweep candle
    close_price: float          # close of the sweep candle
    reversal_confirmed: bool    # price confirmed reversal in next N bars
    formed_at: pd.Timestamp
    wick_size: float            # absolute wick size
    wick_atr_ratio: float       # wick_size / ATR at time of sweep

    @property
    def is_bullish(self) -> bool:
        return self.direction == "BULLISH_SWEEP"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "BEARISH_SWEEP"


class SweepDetector:
    """
    Detect liquidity sweeps on an OHLCV DataFrame.

    Usage::

        det = SweepDetector()
        sweeps = det.detect(df)
        bullish = [s for s in sweeps if s.is_bullish]
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg = self._load_config(config_path)
        self._lookback: int = cfg.get("lookback_bars", _DEFAULT_LOOKBACK)
        self._reversal_bars: int = cfg.get("reversal_bars", _DEFAULT_REVERSAL_BARS)
        self._min_wick_atr: float = cfg.get("min_wick_atr_multiple", _DEFAULT_MIN_WICK_ATR)

    # ── Public API ─────────────────────────────────────────────────────── #

    def detect(self, df: pd.DataFrame) -> List[SweepResult]:
        """
        Scan *df* (OHLCV with uppercase columns) for sweep events.

        Returns a list of `SweepResult`, newest first.
        Requires at least ``lookback_bars + reversal_bars + 1`` rows.
        """
        df = self._normalise(df)
        min_rows = self._lookback + self._reversal_bars + 2
        if len(df) < min_rows:
            logger.debug("Not enough bars (%d < %d) for sweep detection", len(df), min_rows)
            return []

        atr = compute_atr(df)
        results: List[SweepResult] = []

        start = self._lookback
        end = len(df) - self._reversal_bars

        for i in range(start, end):
            candle = df.iloc[i]
            prior = df.iloc[i - self._lookback: i]

            ssl = float(prior["Low"].min())
            bsl = float(prior["High"].max())

            close = float(candle["Close"])
            low = float(candle["Low"])
            high = float(candle["High"])

            future = df.iloc[i + 1: i + self._reversal_bars + 1]

            # ── Bullish sweep: wick below SSL, close back above ─────────
            if low < ssl and close > ssl:
                wick = ssl - low
                if wick >= self._min_wick_atr * atr:
                    reversal = bool(future["Close"].iloc[-1] > close) if len(future) else False
                    results.append(SweepResult(
                        direction="BULLISH_SWEEP",
                        swept_level=ssl,
                        wick_low=low,
                        wick_high=high,
                        close_price=close,
                        reversal_confirmed=reversal,
                        formed_at=df.index[i],
                        wick_size=wick,
                        wick_atr_ratio=wick / atr if atr > 0 else 0.0,
                    ))

            # ── Bearish sweep: wick above BSL, close back below ─────────
            if high > bsl and close < bsl:
                wick = high - bsl
                if wick >= self._min_wick_atr * atr:
                    reversal = bool(future["Close"].iloc[-1] < close) if len(future) else False
                    results.append(SweepResult(
                        direction="BEARISH_SWEEP",
                        swept_level=bsl,
                        wick_low=low,
                        wick_high=high,
                        close_price=close,
                        reversal_confirmed=reversal,
                        formed_at=df.index[i],
                        wick_size=wick,
                        wick_atr_ratio=wick / atr if atr > 0 else 0.0,
                    ))

        results.sort(key=lambda s: s.formed_at, reverse=True)
        return results

    def most_recent(self, df: pd.DataFrame) -> Optional[SweepResult]:
        """Return the most recent sweep, or None."""
        sweeps = self.detect(df)
        return sweeps[0] if sweeps else None

    # ── Private helpers ────────────────────────────────────────────────── #

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                full = yaml.safe_load(f)
            return full.get("sweep", {})
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using defaults", path)
            return {}


def _default_config_path() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
