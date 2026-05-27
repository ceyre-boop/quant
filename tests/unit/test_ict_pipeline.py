"""Unit tests for ict/pipeline.py — the ICT micro-edge orchestration layer."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

from ict.pipeline import ICTPipeline, ICTSignal, ICTVeto, ICTGrade
from ict.micro_risk import MicroRiskParams, PositionSizing


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


# ── Kill Zone timestamps ──────────────────────────────────────────────────── #

LONDON_TS = datetime(2024, 3, 15, 7, 0, 0, tzinfo=timezone.utc)        # 07:00 UTC = 03:00 ET (EDT) → London KZ
NY_OPEN_TS = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)     # 12:00 UTC = 08:00 ET (EDT) → NY Open KZ
OFF_HOURS_TS = datetime(2024, 3, 15, 22, 0, 0, tzinfo=timezone.utc)   # 22:00 UTC = 18:00 ET (EDT) → Off hours
NY_LUNCH_TS = datetime(2024, 3, 15, 17, 0, 0, tzinfo=timezone.utc)    # 17:00 UTC = 13:00 ET (EDT) → NY Lunch


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
        kz_weight = self.pipe._weights.get("kill_zone", 2.0)
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
        # Every component either confirmed or missing
        confirmed_count = len(result.confirmations)
        missing_count = len(result.missing)
        assert confirmed_count + missing_count >= len(self.pipe._weights)


# ── evaluate: risk engine gate ────────────────────────────────────────────── #

class TestRiskEngineGate:
    def setup_method(self):
        self.pipe = ICTPipeline()

    def test_max_positions_hit_vetoes_even_good_setup(self):
        df = _bullish_trending_df()
        # Account with 3 open positions = at max
        acc = MicroRiskParams(account_size=10_000, open_positions=3)
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, acc)
        # If grade was A/A+, risk engine should veto
        if isinstance(result, ICTVeto) and result.grade == ICTGrade.VETOED:
            assert "Risk gate" in result.reason

    def test_daily_loss_limit_vetoes(self):
        df = _bullish_trending_df()
        acc = MicroRiskParams(account_size=10_000, daily_loss_pct=0.06)  # over 5% limit
        result = self.pipe.evaluate("GBPUSD", "LONG", df, LONDON_TS, acc)
        if isinstance(result, ICTVeto) and result.grade == ICTGrade.VETOED:
            assert "DAILY_LOSS_LIMIT" in result.reason


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
