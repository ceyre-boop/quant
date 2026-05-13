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
_DEFAULT_MIN_DISPLACEMENT_ATR = 0.5   # post-sweep displacement body must be ≥ 0.5 × local ATR
_DEFAULT_MIN_REJECTION_PCT = 0.5      # close must be ≥ 50 % back inside the prior range


@dataclass(frozen=True)
class SweepResult:
    """A detected liquidity sweep event."""
    direction: str              # 'BULLISH_SWEEP' | 'BEARISH_SWEEP'
    swept_level: float          # SSL (for bullish) or BSL (for bearish)
    wick_low: float             # lowest point of the sweep candle
    wick_high: float            # highest point of the sweep candle
    close_price: float          # close of the sweep candle
    reversal_confirmed: bool    # price confirmed reversal in next N bars
    displacement_confirmed: bool  # post-sweep bar shows impulsive displacement
    rejection_quality: float    # 0–1: how firmly the sweep candle closed back inside range
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
        self._min_displacement_atr: float = cfg.get("min_displacement_atr", _DEFAULT_MIN_DISPLACEMENT_ATR)
        self._min_rejection_pct: float = cfg.get("min_rejection_pct", _DEFAULT_MIN_REJECTION_PCT)

    # ── Public API ─────────────────────────────────────────────────────── #

    def detect(self, df: pd.DataFrame) -> List[SweepResult]:
        """
        Scan *df* (OHLCV with uppercase columns) for sweep events.

        Quality upgrades vs. baseline:
          • Local ATR: ATR is computed on the ``lookback_bars`` window ending at each
            sweep candidate, not once globally.  This avoids underestimating the wick
            threshold during high-volatility sessions.
          • Rejection quality: ``(close - swept_level) / wick_size`` for bullish sweeps,
            measuring how firmly the sweep candle closed back inside the prior range.
            Quality = 1.0 means close is at the swept_level + full wick size (maximum rejection).
          • Displacement confirmed: the first bar after the sweep must have a body
            (|close − open|) ≥ ``min_displacement_atr × local_ATR`` in the expected
            direction.  This filters low-quality "wicks" that immediately reverse back.

        Returns a list of `SweepResult`, newest first.
        Requires at least ``lookback_bars + reversal_bars + 1`` rows.
        """
        df = self._normalise(df)
        min_rows = self._lookback + self._reversal_bars + 2
        if len(df) < min_rows:
            logger.debug("Not enough bars (%d < %d) for sweep detection", len(df), min_rows)
            return []

        results: List[SweepResult] = []

        start = self._lookback
        end = len(df) - self._reversal_bars

        for i in range(start, end):
            candle = df.iloc[i]
            prior = df.iloc[i - self._lookback: i]

            # Local ATR: measure volatility at the sweep bar, not globally
            local_atr = compute_atr(prior) if len(prior) >= 14 else compute_atr(df)
            if local_atr <= 0:
                continue

            ssl = float(prior["Low"].min())
            bsl = float(prior["High"].max())

            close  = float(candle["Close"])
            low    = float(candle["Low"])
            high   = float(candle["High"])
            c_open = float(candle["Open"])

            future = df.iloc[i + 1: i + self._reversal_bars + 1]
            disp_bar = df.iloc[i + 1] if i + 1 < len(df) else None

            # ── Bullish sweep: wick below SSL, close back above ─────────
            if low < ssl and close > ssl:
                wick = ssl - low
                if wick >= self._min_wick_atr * local_atr:
                    reversal = bool(future["Close"].iloc[-1] > close) if len(future) else False

                    # Rejection quality: how far above SSL did the close land?
                    #   quality = (close - ssl) / wick_size, clamped [0, 1]
                    rejection_q = min((close - ssl) / wick, 1.0) if wick > 0 else 0.0
                    rejection_ok = rejection_q >= self._min_rejection_pct

                    # Displacement: next bar must be bullish with body ≥ threshold
                    disp_conf = False
                    if disp_bar is not None:
                        bar_body = float(disp_bar["Close"]) - float(disp_bar["Open"])
                        disp_conf = bar_body >= self._min_displacement_atr * local_atr

                    if rejection_ok:
                        results.append(SweepResult(
                            direction="BULLISH_SWEEP",
                            swept_level=ssl,
                            wick_low=low,
                            wick_high=high,
                            close_price=close,
                            reversal_confirmed=reversal,
                            displacement_confirmed=disp_conf,
                            rejection_quality=round(rejection_q, 3),
                            formed_at=df.index[i],
                            wick_size=wick,
                            wick_atr_ratio=round(wick / local_atr, 3),
                        ))

            # ── Bearish sweep: wick above BSL, close back below ─────────
            if high > bsl and close < bsl:
                wick = high - bsl
                if wick >= self._min_wick_atr * local_atr:
                    reversal = bool(future["Close"].iloc[-1] < close) if len(future) else False

                    # Rejection quality for bearish: how far below BSL did the close land?
                    rejection_q = min((bsl - close) / wick, 1.0) if wick > 0 else 0.0
                    rejection_ok = rejection_q >= self._min_rejection_pct

                    # Displacement: next bar must be bearish
                    disp_conf = False
                    if disp_bar is not None:
                        bar_body = float(disp_bar["Open"]) - float(disp_bar["Close"])
                        disp_conf = bar_body >= self._min_displacement_atr * local_atr

                    if rejection_ok:
                        results.append(SweepResult(
                            direction="BEARISH_SWEEP",
                            swept_level=bsl,
                            wick_low=low,
                            wick_high=high,
                            close_price=close,
                            reversal_confirmed=reversal,
                            displacement_confirmed=disp_conf,
                            rejection_quality=round(rejection_q, 3),
                            formed_at=df.index[i],
                            wick_size=wick,
                            wick_atr_ratio=round(wick / local_atr, 3),
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rename = {c: c.capitalize() for c in df.columns if c.lower() in ("open","high","low","close","volume")}
        return df.rename(columns=rename)[["Open","High","Low","Close"]].dropna()

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
