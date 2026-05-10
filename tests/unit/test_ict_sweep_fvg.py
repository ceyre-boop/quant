"""Unit tests for ict/sweep_detector.py and ict/fvg_detector.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ict.sweep_detector import SweepDetector, SweepResult
from ict.fvg_detector import FVGDetector, FVGResult, OrderBlockResult


# ── OHLCV fixtures ────────────────────────────────────────────────────────── #

def _flat_df(n: int = 100, base: float = 1.0, spread: float = 0.001) -> pd.DataFrame:
    """Return a flat, random-walk-like OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.uniform(-spread, spread, n))
    opens  = closes + rng.uniform(-spread / 2, spread / 2, n)
    highs  = np.maximum(opens, closes) + rng.uniform(0, spread, n)
    lows   = np.minimum(opens, closes) - rng.uniform(0, spread, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _sweep_df(direction: str = "BULLISH") -> pd.DataFrame:
    """
    Build a DataFrame that contains exactly one detectable sweep.

    Bullish sweep:
      • 60 bars of stable prices to establish SSL.
      • 1 bar whose Low pierces the SSL and Close recovers above SSL.
      • 3 bars of upward price action (reversal).
    """
    n_base = 60
    base = 1.0500
    atr_approx = 0.0010

    opens  = [base] * n_base
    highs  = [base + atr_approx * 0.6] * n_base
    lows   = [base - atr_approx * 0.4] * n_base
    closes = [base] * n_base

    ssl = base - atr_approx * 0.4         # established swing low
    sweep_low = ssl - atr_approx * 0.6    # wick below SSL

    if direction == "BULLISH":
        # Sweep candle
        opens  += [base]
        highs  += [base + atr_approx * 0.3]
        lows   += [sweep_low]
        closes += [base + atr_approx * 0.2]   # close above SSL
        # Reversal bars
        for k in range(1, 4):
            opens  += [base + atr_approx * 0.2 * k]
            highs  += [base + atr_approx * 0.5 * k]
            lows   += [base + atr_approx * 0.1 * k]
            closes += [base + atr_approx * 0.4 * k]
    else:
        bsl = base + atr_approx * 0.4
        sweep_high = bsl + atr_approx * 0.6
        opens  += [base]
        highs  += [sweep_high]
        lows   += [base - atr_approx * 0.3]
        closes += [base - atr_approx * 0.2]   # close below BSL
        for k in range(1, 4):
            opens  += [base - atr_approx * 0.2 * k]
            highs  += [base - atr_approx * 0.1 * k]
            lows   += [base - atr_approx * 0.5 * k]
            closes += [base - atr_approx * 0.4 * k]

    idx = pd.date_range("2024-01-01", periods=len(opens), freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _fvg_df(direction: str = "BULLISH") -> pd.DataFrame:
    """
    Build a DataFrame containing an obvious Fair Value Gap.

    Bullish FVG: candle[i].High < candle[i+2].Low
    """
    n = 60
    base = 1.0500
    step = 0.0002
    bars = []
    for i in range(n):
        o = base + i * step * 0.1
        c = o + step * 0.2
        h = c + step * 0.3
        l = o - step * 0.1
        bars.append((o, h, l, c))

    # Inject FVG at bar 40–42
    if direction == "BULLISH":
        bars[40] = (bars[40][0], 1.0600, bars[40][2], 1.0605)  # c1 high = 1.0600
        bars[41] = (1.0610, 1.0650, 1.0608, 1.0640)            # impulse candle
        bars[42] = (1.0645, 1.0680, 1.0620, 1.0670)            # c3 low = 1.0620 > c1 high 1.0600 → FVG
    else:
        bars[40] = (1.0700, 1.0710, 1.0650, 1.0650)  # c1 low = 1.0650
        bars[41] = (1.0640, 1.0645, 1.0600, 1.0605)  # impulse candle
        bars[42] = (1.0600, 1.0610, 1.0560, 1.0565)  # c3 high = 1.0610 < c1 low 1.0650 → FVG

    opens, highs, lows, closes = zip(*bars)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": list(opens), "High": list(highs), "Low": list(lows), "Close": list(closes)}, index=idx)


def _ob_df() -> pd.DataFrame:
    """
    Build a DataFrame with a clear bullish Order Block:
      • Last bearish candle at i followed by a strong bullish impulse at i+1.
    Uses narrow-range base bars so the ATR is small and the impulse clearly exceeds threshold.
    """
    n = 50
    base = 1.0500
    # Narrow-range base bars: ATR ≈ 0.0002
    opens  = [base] * n
    closes = [base] * n
    highs  = [base + 0.0001] * n
    lows   = [base - 0.0001] * n

    # Bearish OB at bar 45
    opens[45]  = base + 0.0003
    highs[45]  = base + 0.0004
    lows[45]   = base - 0.0001
    closes[45] = base - 0.0002   # bearish

    # Strong bullish impulse at bar 46: body >> 1.5 × ATR (0.0002)
    opens[46]  = base
    closes[46] = base + 0.0010   # body = 0.0010 >> threshold ~ 0.0003
    highs[46]  = base + 0.0011
    lows[46]   = base - 0.0001

    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


# ════════════════════════════════════════════════════════════════════════════ #
# SweepDetector tests
# ════════════════════════════════════════════════════════════════════════════ #

class TestSweepDetectorInit:
    def test_default_construction(self):
        det = SweepDetector()
        assert det._lookback > 0
        assert det._reversal_bars > 0
        assert det._min_wick_atr > 0


class TestSweepDetectorDetect:
    def setup_method(self):
        self.det = SweepDetector()

    def test_returns_list(self):
        df = _flat_df()
        result = self.det.detect(df)
        assert isinstance(result, list)

    def test_not_enough_bars_returns_empty(self):
        df = _flat_df(n=10)
        assert self.det.detect(df) == []

    def test_detects_bullish_sweep(self):
        df = _sweep_df("BULLISH")
        sweeps = self.det.detect(df)
        bullish = [s for s in sweeps if s.is_bullish]
        assert len(bullish) >= 1, "Expected at least 1 bullish sweep"

    def test_detects_bearish_sweep(self):
        df = _sweep_df("BEARISH")
        sweeps = self.det.detect(df)
        bearish = [s for s in sweeps if s.is_bearish]
        assert len(bearish) >= 1, "Expected at least 1 bearish sweep"

    def test_sweep_result_fields(self):
        df = _sweep_df("BULLISH")
        sweeps = self.det.detect(df)
        bullish = [s for s in sweeps if s.is_bullish]
        assert len(bullish) >= 1
        s = bullish[0]
        assert s.swept_level > 0
        assert s.wick_low < s.swept_level  # wick went below SSL
        assert s.close_price > s.swept_level  # recovered above SSL
        assert s.wick_size > 0
        assert s.wick_atr_ratio > 0

    def test_sorted_newest_first(self):
        df = _flat_df(200)
        sweeps = self.det.detect(df)
        if len(sweeps) >= 2:
            assert sweeps[0].formed_at >= sweeps[-1].formed_at

    def test_most_recent_returns_none_on_flat(self):
        # Completely flat data → no sweeps
        n = 80
        closes = [1.0] * n
        df = pd.DataFrame({
            "Open": closes, "High": closes, "Low": closes, "Close": closes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
        result = self.det.most_recent(df)
        assert result is None

    def test_lowercase_columns_accepted(self):
        df = _flat_df(80)
        df.columns = [c.lower() for c in df.columns]
        # Should not raise
        result = self.det.detect(df)
        assert isinstance(result, list)


# ════════════════════════════════════════════════════════════════════════════ #
# FVGDetector tests
# ════════════════════════════════════════════════════════════════════════════ #

class TestFVGDetectorInit:
    def test_default_construction(self):
        det = FVGDetector()
        assert det._fvg_min_atr > 0
        assert det._fvg_max_age > 0
        assert det._ob_lookback > 0
        assert det._ob_impulse_atr > 0


class TestFVGDetection:
    def setup_method(self):
        self.det = FVGDetector()

    def test_returns_tuple_of_lists(self):
        df = _flat_df()
        fvgs, obs = self.det.detect(df)
        assert isinstance(fvgs, list)
        assert isinstance(obs, list)

    def test_not_enough_bars_returns_empty(self):
        df = _flat_df(n=3)
        fvgs, obs = self.det.detect(df)
        assert fvgs == []
        assert obs == []

    def test_detects_bullish_fvg(self):
        df = _fvg_df("BULLISH")
        fvgs, _ = self.det.detect(df)
        bullish = [f for f in fvgs if f.is_bullish]
        assert len(bullish) >= 1

    def test_detects_bearish_fvg(self):
        df = _fvg_df("BEARISH")
        fvgs, _ = self.det.detect(df)
        bearish = [f for f in fvgs if f.is_bearish]
        assert len(bearish) >= 1

    def test_fvg_result_geometry(self):
        df = _fvg_df("BULLISH")
        fvgs, _ = self.det.detect(df)
        bullish = [f for f in fvgs if f.is_bullish]
        assert bullish, "No bullish FVG detected"
        f = bullish[0]
        assert f.top > f.bottom
        assert abs(f.midpoint - (f.top + f.bottom) / 2) < 1e-9
        assert f.size == pytest.approx(f.top - f.bottom, abs=1e-9)

    def test_fvg_price_tapping(self):
        df = _fvg_df("BULLISH")
        fvgs, _ = self.det.detect(df)
        bullish = [f for f in fvgs if f.is_bullish]
        assert bullish
        f = bullish[0]
        # Price exactly at midpoint should tap
        assert f.price_tapping(f.midpoint)
        # Price far above should not tap
        assert not f.price_tapping(f.top + f.size * 10)

    def test_fvg_sorted_newest_first(self):
        df = _fvg_df("BULLISH")
        fvgs, _ = self.det.detect(df)
        if len(fvgs) >= 2:
            assert fvgs[0].formed_at >= fvgs[1].formed_at

    def test_lowercase_columns_accepted(self):
        df = _fvg_df("BULLISH")
        df.columns = [c.lower() for c in df.columns]
        fvgs, obs = self.det.detect(df)
        assert isinstance(fvgs, list)


class TestOrderBlockDetection:
    def setup_method(self):
        self.det = FVGDetector()

    def test_detects_bullish_ob(self):
        df = _ob_df()
        _, obs = self.det.detect(df)
        bullish = [o for o in obs if o.is_bullish]
        assert len(bullish) >= 1

    def test_ob_result_fields(self):
        df = _ob_df()
        _, obs = self.det.detect(df)
        bullish = [o for o in obs if o.is_bullish]
        assert bullish
        o = bullish[0]
        assert o.high > o.low
        assert abs(o.midpoint - (o.high + o.low) / 2) < 1e-9
        assert o.impulse_atr_ratio > 0


class TestNearestActionable:
    def setup_method(self):
        self.det = FVGDetector()

    def test_returns_four_elements(self):
        df = _flat_df(100)
        result = self.det.nearest_actionable(df)
        assert len(result) == 4

    def test_bullish_fvg_below_price(self):
        df = _fvg_df("BULLISH")
        # Verify the returned bullish FVG is below current price
        bull_fvg, _, _, _ = self.det.nearest_actionable(df)
        if bull_fvg is not None:
            price = float(df["Close"].iloc[-1])
            assert bull_fvg.top < price
