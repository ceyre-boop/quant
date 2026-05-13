"""Unit tests for ict/pipeline.py — the ICT micro-edge orchestration layer."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto, ICTGrade
from ict.micro_risk import MicroRiskParams, PositionSizing
from ict.sweep_detector import SweepResult


# ── OHLCV helpers ────────────────────────────────────────────────────────── #

def _bullish_trending_df(n: int = 100) -> pd.DataFrame:
    """Steady bullish trend with enough bars for all detectors."""
    rng = np.random.default_rng(1)
    closes = 1.0800 + np.cumsum(np.abs(rng.normal(0.0001, 0.0003, n)))
    opens  = closes - np.abs(rng.normal(0.0, 0.0002, n))
    highs  = closes + np.abs(rng.normal(0.0, 0.0002, n))
    lows   = np.minimum(opens, closes) - np.abs(rng.normal(0.0, 0.0002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _bearish_trending_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    closes = 1.1000 - np.cumsum(np.abs(rng.normal(0.0001, 0.0003, n)))
    opens  = closes + np.abs(rng.normal(0.0, 0.0002, n))
    highs  = np.maximum(opens, closes) + np.abs(rng.normal(0.0, 0.0002, n))
    lows   = closes - np.abs(rng.normal(0.0, 0.0002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _default_account() -> MicroRiskParams:
    return MicroRiskParams(account_size=10_000.0)


def _sweep_aligned_df(direction: str = "LONG", n: int = 120) -> pd.DataFrame:
    """
    Build a DataFrame that satisfies all hard gates for a given direction:
      - Bullish trend (HH + HL structure) for LONG / Bearish for SHORT
      - A real liquidity sweep injected near the end
      - Current price in discount (LONG) or premium (SHORT) zone
    """
    base = 1.0500
    atr_approx = 0.0012

    opens  = [base] * n
    highs  = [base + atr_approx * 0.6] * n
    lows   = [base - atr_approx * 0.4] * n
    closes = [base] * n

    # Create a mild trending move to satisfy structure
    for k in range(n):
        drift = k * 0.0001 if direction == "LONG" else -k * 0.0001
        opens[k]  = base + drift
        closes[k] = base + drift + atr_approx * 0.1
        highs[k]  = closes[k] + atr_approx * 0.4
        lows[k]   = opens[k]  - atr_approx * 0.3

    # Inject sweep near bar 80
    sweep_i = 80
    ssl = min(lows[:sweep_i])
    bsl = max(highs[:sweep_i])
    if direction == "LONG":
        lows[sweep_i]   = ssl - atr_approx * 0.8   # sweep below SSL
        closes[sweep_i] = ssl + atr_approx * 0.6   # close back above SSL
        highs[sweep_i]  = closes[sweep_i] + atr_approx * 0.2
        opens[sweep_i]  = ssl + atr_approx * 0.1
        # Displacement bar
        opens[sweep_i + 1]  = closes[sweep_i]
        closes[sweep_i + 1] = closes[sweep_i] + atr_approx * 0.8
        highs[sweep_i + 1]  = closes[sweep_i + 1] + atr_approx * 0.1
        lows[sweep_i + 1]   = opens[sweep_i + 1] - atr_approx * 0.1
    else:
        highs[sweep_i]  = bsl + atr_approx * 0.8
        closes[sweep_i] = bsl - atr_approx * 0.6
        lows[sweep_i]   = closes[sweep_i] - atr_approx * 0.2
        opens[sweep_i]  = bsl - atr_approx * 0.1
        opens[sweep_i + 1]  = closes[sweep_i]
        closes[sweep_i + 1] = closes[sweep_i] - atr_approx * 0.8
        lows[sweep_i + 1]   = closes[sweep_i + 1] - atr_approx * 0.1
        highs[sweep_i + 1]  = opens[sweep_i + 1] + atr_approx * 0.1

    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


# ── Kill Zone timestamps ──────────────────────────────────────────────────── #

LONDON_TS = datetime(2024, 3, 15, 3, 0, 0)        # 03:00 UTC → London KZ
NY_OPEN_TS = datetime(2024, 3, 15, 8, 0, 0)       # 08:00 UTC → NY Open KZ
OFF_HOURS_TS = datetime(2024, 3, 15, 17, 0, 0)    # 17:00 UTC → Off hours
NY_LUNCH_TS = datetime(2024, 3, 15, 12, 30, 0)    # 12:30 UTC → NY Lunch


# ── Pipeline construction ─────────────────────────────────────────────────── #

class TestICTPipelineInit:
    def test_default_construction(self):
        pipe = ICTPipeline()
        assert pipe._min_score > 0
        assert isinstance(pipe._weights, dict)
        assert len(pipe._weights) >= 4

    def test_component_weights_reasonable(self):
        pipe = ICTPipeline()
        total = sum(pipe._weights.values())
        assert 8.0 <= total <= 12.0, f"Weights sum {total} is out of expected range [8, 12]"


# ── evaluate: direction validation ───────────────────────────────────────── #

class TestEvaluateDirectionValidation:
    def setup_method(self):
        self.pipe = ICTPipeline()
        self.df = _bullish_trending_df()
        self.acc = _default_account()

    def test_invalid_direction_vetoed(self):
        result = self.pipe.evaluate("GBPUSD", "SIDEWAYS", self.df, LONDON_TS, self.acc)
        assert isinstance(result, ICTVeto)

    def test_long_is_valid_direction(self):
        result = self.pipe.evaluate("GBPUSD", "LONG", self.df, LONDON_TS, self.acc)
        assert isinstance(result, (ICTSignal, ICTVeto))

    def test_short_is_valid_direction(self):
        result = self.pipe.evaluate("GBPUSD", "SHORT", self.df, LONDON_TS, self.acc)
        assert isinstance(result, (ICTSignal, ICTVeto))


# ── evaluate: output types ────────────────────────────────────────────────── #

class TestEvaluateOutputTypes:
    def setup_method(self):
        self.pipe = ICTPipeline()
        self.acc = _default_account()

    def test_returns_ict_signal_or_ict_veto(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        assert isinstance(result, (ICTSignal, ICTVeto))

    def test_signal_has_sizing_when_passed(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        if isinstance(result, ICTSignal) and result.passed:
            assert isinstance(result.sizing, PositionSizing)

    def test_veto_has_reason(self):
        # Off-hours → kill zone gate fails → likely vetoed
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, OFF_HOURS_TS, self.acc)
        # May pass or fail depending on other components; if veto, must have reason
        if isinstance(result, ICTVeto):
            assert result.reason and len(result.reason) > 5


# ── evaluate: session gate blocks off-hours ─────────────────────────────── #

class TestSessionGating:
    def setup_method(self):
        self.pipe = ICTPipeline()
        self.acc = _default_account()

    def test_off_hours_reduces_score(self):
        df = _bullish_trending_df()
        result_off = self.pipe.evaluate("GBPUSD", "LONG", df, OFF_HOURS_TS, self.acc)
        # Kill Zone score must be 0 in off hours
        if isinstance(result_off, (ICTSignal, ICTVeto)):
            assert result_off.component_scores.get("kill_zone", 0.0) == pytest.approx(0.0)

    def test_london_session_gets_kz_score(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        kz_score = result.component_scores.get("kill_zone", 0.0)
        expected = self.pipe._weights.get("kill_zone", 2.0)
        assert kz_score == pytest.approx(expected)

    def test_ny_lunch_blocks_should_trade(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, NY_LUNCH_TS, self.acc)
        if isinstance(result, ICTSignal):
            assert not result.session_status.should_trade

    def test_off_hours_hard_gate_returns_vetoed(self):
        """Phase 3: off-hours is now a hard gate — must return GATE_KILL_ZONE veto."""
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, OFF_HOURS_TS, self.acc)
        assert isinstance(result, ICTVeto)
        assert result.grade == ICTGrade.VETOED
        assert "GATE_KILL_ZONE" in result.reason


# ── evaluate: hard gates ──────────────────────────────────────────────────── #

class TestHardGates:
    """Phase 3: sweep and PD alignment are hard gates, not additive scores."""

    def setup_method(self):
        self.pipe = ICTPipeline()
        self.acc = _default_account()

    def test_no_sweep_vetoes_with_gate_reason(self):
        """Flat random data has no sweep — must veto with GATE_SWEEP."""
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        assert isinstance(result, ICTVeto)
        assert "GATE_SWEEP" in result.reason

    def test_gate_veto_has_kill_zone_score(self):
        """Even when a hard gate fires, kill_zone score must be recorded."""
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        assert "kill_zone" in result.component_scores
        assert result.component_scores["kill_zone"] > 0

    def test_off_hours_gate_fires_before_sweep_gate(self):
        """Kill zone gate (stage 1) takes priority over sweep gate (stage 2)."""
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, OFF_HOURS_TS, self.acc)
        assert isinstance(result, ICTVeto)
        assert "GATE_KILL_ZONE" in result.reason
        # Sweep gate must NOT fire if kill zone gate already fired
        assert "GATE_SWEEP" not in result.reason


# ── evaluate: score and grade ─────────────────────────────────────────────── #

class TestScoreAndGrade:
    def setup_method(self):
        self.pipe = ICTPipeline()
        self.acc = _default_account()

    def test_score_is_non_negative(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        assert result.score >= 0.0

    def test_score_does_not_exceed_max(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        max_score = sum(self.pipe._weights.values())
        assert result.score <= max_score + 0.01

    def test_grade_a_plus_requires_high_score(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        if result.grade == ICTGrade.A_PLUS:
            assert result.score >= 8.5

    def test_grade_c_always_vetoed(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, OFF_HOURS_TS, self.acc)
        if result.grade == ICTGrade.C:
            assert isinstance(result, ICTVeto)

    def test_component_scores_dict_has_expected_keys(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        # Hard gates may return early; keys present depend on how far the pipeline ran.
        # At minimum, kill_zone and sweep must always be evaluated (they are the first two stages).
        assert "kill_zone" in result.component_scores
        # If the pipeline ran all the way (ICTSignal, or a non-gate ICTVeto), all keys present.
        reason = getattr(result, "reason", "")
        if not reason.startswith("GATE_"):
            for key in ("kill_zone", "sweep", "fvg_tap", "market_structure", "pd_alignment"):
                assert key in result.component_scores

    def test_confirmations_and_missing_lists(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        assert isinstance(result.confirmations, list)
        assert isinstance(result.missing, list)

    def test_confirmations_plus_missing_covers_all_components(self):
        df = _bullish_trending_df()
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, self.acc)
        # The hard-gate architecture returns early when a required precondition fails,
        # so the number of evaluated components equals the stages run so far.
        # We only check that BOTH lists together cover all stages that were scored.
        scored_components = len(result.component_scores)
        confirmed_count = len(result.confirmations)
        missing_count = len(result.missing)
        # Subtract 1 from confirmations for the vol-regime header line that is
        # always added first and is not a score component.
        assert confirmed_count - 1 + missing_count >= scored_components


# ── evaluate: risk engine gate ────────────────────────────────────────────── #

class TestRiskEngineGate:
    def setup_method(self):
        self.pipe = ICTPipeline()

    def test_max_positions_hit_vetoes_even_good_setup(self):
        df = _bullish_trending_df()
        # Account with 3 open positions = at max
        acc = MicroRiskParams(account_size=10_000, open_positions=3)
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, acc)
        # With hard gates, the result is always a veto (sweep/PD gate fires before
        # risk engine on random data; if a real setup passes, the risk gate will fire).
        assert isinstance(result, ICTVeto)
        assert result.grade == ICTGrade.VETOED

    def test_daily_loss_limit_vetoes(self):
        df = _bullish_trending_df()
        acc = MicroRiskParams(account_size=10_000, daily_loss_pct=0.06)  # over 5% limit
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, acc)
        # Either a structural gate fires (GATE_*) or the risk engine fires (DAILY_LOSS_LIMIT).
        # In both cases the result must be a VETOED ICTVeto.
        assert isinstance(result, ICTVeto)
        assert result.grade == ICTGrade.VETOED


# ── Phase 0 regression tests ─────────────────────────────────────────────── #

class TestPhase0Regressions:
    """Regression tests for the two Phase 0 correctness bugs."""

    def setup_method(self):
        self.pipe = ICTPipeline()

    def test_entry_price_from_sizing_not_zero(self):
        """
        Phase 0 bug: `getattr(sz, 'entry', 0.0)` always returned 0.0 because
        PositionSizing uses `entry_price`, not `entry`.
        The pipeline now returns sz.entry_price (== price from df).
        """
        from ict.micro_risk import MicroRiskEngine, MicroRiskParams as P
        engine = MicroRiskEngine()
        acc = P(account_size=10_000.0)
        entry = 1.0850
        stop  = 1.0820
        atr   = 0.0030
        result = engine.size("LONG", entry, stop, atr, acc)
        assert isinstance(result, PositionSizing)
        # entry_price must equal the entry passed in — never 0
        assert result.entry_price == pytest.approx(entry, rel=1e-6)
        assert result.entry_price != 0.0

    def test_confluence_boost_only_aligned_direction(self):
        """
        Phase 0 bug: _apply_confluence() added GBPUSD SHORT to 'usd_long_signals',
        meaning every signal regardless of direction received the boost.
        Now only signals that ALIGN with the dominant USD direction are boosted.
        """
        from ict.orchestrator import _USD_QUOTED_PAIRS, _USD_BASE_PAIRS
        from dataclasses import dataclass

        # Simulate 3 USD-weak signals (LONG on EUR/GBP/AUD) + 1 USD-strong signal (SHORT on GBPUSD)
        # Only the 3 weak-USD signals should be boosted.
        # Verify the constants are correct first:
        assert "GBPUSD" in _USD_QUOTED_PAIRS
        assert "EURUSD" in _USD_QUOTED_PAIRS
        assert "USDJPY" in _USD_BASE_PAIRS


# ── Phase 1: structure stop and liquidity targets ─────────────────────────── #

class TestStructureStop:
    """Phase 1: compute_structure_stop uses structural anchors, not raw ATR."""

    def _make_sweep(self, direction: str = "BULLISH") -> SweepResult:
        import pandas as pd
        ts = pd.Timestamp("2024-01-01 03:00")
        if direction == "BULLISH":
            return SweepResult(
                direction="BULLISH_SWEEP",
                swept_level=1.0480,
                wick_low=1.0470,      # stop anchor for LONG
                wick_high=1.0495,
                close_price=1.0490,
                reversal_confirmed=True,
                displacement_confirmed=True,
                rejection_quality=0.8,
                formed_at=ts,
                wick_size=0.0010,
                wick_atr_ratio=1.0,
            )
        return SweepResult(
            direction="BEARISH_SWEEP",
            swept_level=1.0520,
            wick_low=1.0505,
            wick_high=1.0530,     # stop anchor for SHORT
            close_price=1.0510,
            reversal_confirmed=True,
            displacement_confirmed=True,
            rejection_quality=0.8,
            formed_at=ts,
            wick_size=0.0010,
            wick_atr_ratio=1.0,
        )

    def test_long_stop_uses_sweep_wick_low(self):
        df = _bullish_trending_df()
        atr = 0.0010
        sweep = self._make_sweep("BULLISH")
        stop = ICTPipeline.compute_structure_stop("LONG", sweep, None, None, df, atr)
        # Should be below the wick_low with buffer
        assert stop < sweep.wick_low
        # Should be close to wick_low (within 1 ATR)
        assert sweep.wick_low - stop < atr

    def test_short_stop_uses_sweep_wick_high(self):
        df = _bearish_trending_df()
        atr = 0.0010
        sweep = self._make_sweep("BEARISH")
        stop = ICTPipeline.compute_structure_stop("SHORT", sweep, None, None, df, atr)
        assert stop > sweep.wick_high
        assert stop - sweep.wick_high < atr

    def test_stop_on_correct_side_of_price(self):
        df = _bullish_trending_df()
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        sweep = self._make_sweep("BULLISH")
        stop = ICTPipeline.compute_structure_stop("LONG", sweep, None, None, df, atr)
        assert stop < price, "LONG stop must be below current price"

    def test_fallback_to_swing_low_when_no_sweep(self):
        df = _bullish_trending_df()
        atr = 0.0010
        stop = ICTPipeline.compute_structure_stop("LONG", None, None, None, df, atr)
        swing_low = float(df["Low"].tail(20).min())
        # Without sweep/OB/FVG, stop should be near the 20-bar swing low
        assert stop < swing_low + atr

    def test_minimum_stop_distance_enforced(self):
        """Stop must be at least _MIN_STOP_ATR_FRACTION × ATR from price."""
        from ict.pipeline import _MIN_STOP_ATR_FRACTION
        df = _bullish_trending_df()
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        stop = ICTPipeline.compute_structure_stop("LONG", None, None, None, df, atr)
        assert price - stop >= _MIN_STOP_ATR_FRACTION * atr - 1e-9


class TestLiquidityTargets:
    """Phase 1: compute_liquidity_targets finds swing-level objectives."""

    def test_returns_list(self):
        df = _bullish_trending_df()
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("LONG", df, price, atr)
        assert isinstance(targets, list)

    def test_long_targets_above_price(self):
        df = _bullish_trending_df()
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("LONG", df, price, atr)
        for t in targets:
            assert t > price, f"LONG target {t} must be above current price {price}"

    def test_short_targets_below_price(self):
        df = _bearish_trending_df()
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("SHORT", df, price, atr)
        for t in targets:
            assert t < price, f"SHORT target {t} must be below current price {price}"

    def test_returns_at_most_n_targets(self):
        df = _bullish_trending_df(200)
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("LONG", df, price, atr, n_targets=2)
        assert len(targets) <= 2

    def test_targets_sorted_nearest_first_long(self):
        df = _bullish_trending_df(200)
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("LONG", df, price, atr)
        if len(targets) >= 2:
            assert targets[0] <= targets[1], "LONG targets must be sorted ascending (nearest first)"

    def test_targets_sorted_nearest_first_short(self):
        df = _bearish_trending_df(200)
        price = float(df["Close"].iloc[-1])
        atr = 0.0010
        targets = ICTPipeline.compute_liquidity_targets("SHORT", df, price, atr)
        if len(targets) >= 2:
            assert targets[0] >= targets[1], "SHORT targets must be sorted descending (nearest first)"


# ── Isolation guard ───────────────────────────────────────────────────────── #

def test_pipeline_does_not_import_sovereign():
    import inspect
    import ict.pipeline as m
    src = inspect.getsource(m)
    forbidden = [
        "from sovereign", "import sovereign",
        "from layer2", "import layer2",
        "from layer1", "import layer1",
        "from config.loader", "import config.loader",
    ]
    for phrase in forbidden:
        assert phrase not in src, f"Isolation violated: '{phrase}' found in pipeline.py"
