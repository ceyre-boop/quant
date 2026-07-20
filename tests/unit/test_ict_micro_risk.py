"""Unit tests for ict/micro_risk.py — isolated from Sovereign risk engine."""
from __future__ import annotations

import pytest

from ict.micro_risk import MicroRiskEngine, MicroRiskParams, PositionSizing, RiskVeto


# ── Fixtures ─────────────────────────────────────────────────────────────── #

def _fresh_account(
    size: float = 10_000.0,
    positions: int = 0,
    open_risk: float = 0.0,
    daily_loss: float = 0.0,
) -> MicroRiskParams:
    return MicroRiskParams(
        account_size=size,
        open_positions=positions,
        open_risk_pct=open_risk,
        daily_loss_pct=daily_loss,
    )


# ── Engine construction ───────────────────────────────────────────────────── #

class TestMicroRiskEngineInit:
    def test_default_construction(self):
        e = MicroRiskEngine()
        assert 0 < e.max_risk_per_trade <= 0.05
        assert e.max_positions >= 1
        assert e.min_rr >= 1.0
        assert e.max_leverage >= 1.0

    def test_defaults_are_isolated_from_sovereign(self):
        """Verify the default values come from ict_params.yml (0.75 % fallback, 30×),
        not parameters.yml. Per-trade risk is now conviction-scaled by grade; this
        value is the prop-safe fallback/ceiling used when no grade is supplied."""
        e = MicroRiskEngine()
        assert e.max_risk_per_trade == pytest.approx(0.0075, abs=0.0005)
        assert e.max_leverage == pytest.approx(30.0, abs=0.1)


# ── Happy-path sizing ────────────────────────────────────────────────────── #

class TestPositionSizing:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_returns_position_sizing(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)

    def test_risk_dollars_uses_fallback_without_grade(self):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        expected_risk = 10_000 * 0.0075  # prop-safe fallback when no grade supplied
        assert result.risk_dollars == pytest.approx(expected_risk, rel=0.01)

    def test_risk_pct_is_fallback_without_grade(self):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.risk_pct == pytest.approx(0.0075, abs=0.0005)

    def test_tp1_is_above_entry_for_long(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.tp1 > result.entry_price

    def test_tp2_is_above_tp1_for_long(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.tp2 > result.tp1

    def test_tp1_below_entry_for_short(self):
        acc = _fresh_account()
        result = self.engine.size("SHORT", 1.0850, 1.0880, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.tp1 < result.entry_price

    def test_tp2_below_tp1_for_short(self):
        acc = _fresh_account()
        result = self.engine.size("SHORT", 1.0850, 1.0880, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.tp2 < result.tp1

    def test_tp1_units_plus_tp2_units_equal_total(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.tp1_units + result.tp2_units == pytest.approx(result.position_units, rel=1e-5)

    def test_leverage_within_cap(self):
        acc = _fresh_account(size=1_000)  # small account → high leverage
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.leverage_used <= self.engine.max_leverage + 0.01


# ── Risk gate: daily loss limit ───────────────────────────────────────────── #

class TestDailyLossLimit:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_blocks_when_daily_loss_reached(self):
        acc = _fresh_account(daily_loss=0.06)  # 6 % > 2 % limit
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "DAILY_LOSS_LIMIT"

    def test_passes_below_daily_loss_limit(self):
        acc = _fresh_account(daily_loss=0.01)  # 1 % < 2 % limit (RISK_FRAMEWORK.md ratified 2026-07-20)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)


# ── Risk gate: max positions ──────────────────────────────────────────────── #

class TestMaxPositions:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_blocks_when_max_positions_reached(self):
        acc = _fresh_account(positions=3)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "MAX_POSITIONS"

    def test_passes_below_max_positions(self):
        acc = _fresh_account(positions=2)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)


# ── Risk gate: max concurrent risk ───────────────────────────────────────── #

class TestMaxConcurrentRisk:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_blocks_when_concurrent_risk_full(self):
        # 5.55% open + A+ 1.0% = 6.55% > 6% cap → veto (gate reserves actual graded risk)
        acc = _fresh_account(open_risk=0.0555)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade="A+")
        assert isinstance(result, RiskVeto)
        assert result.reason == "MAX_CONCURRENT_RISK"

    def test_passes_with_room(self):
        acc = _fresh_account(open_risk=0.03)  # 3% + 0.75% fallback = 3.75% < 6%
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)


# ── Conviction sizing by grade ───────────────────────────────────────────── #

class TestGradeScaledSizing:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    @pytest.mark.parametrize("grade,expected_risk", [
        ("A+", 0.010), ("A", 0.0075), ("B", 0.005), ("C", 0.0025),
    ])
    def test_risk_scales_with_grade(self, grade, expected_risk):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade=grade)
        assert isinstance(result, PositionSizing)
        assert result.risk_pct == pytest.approx(expected_risk, abs=1e-6)
        assert result.risk_dollars == pytest.approx(10_000 * expected_risk, rel=0.01)

    def test_higher_grade_risks_more(self):
        acc = _fresh_account(size=10_000)
        a_plus = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade="A+")
        c = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade="C")
        assert a_plus.risk_pct > c.risk_pct

    def test_never_exceeds_one_percent_ceiling(self):
        acc = _fresh_account(size=10_000)
        for grade in ("A+", "A", "B", "C", "UNKNOWN", None):
            result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade=grade)
            assert isinstance(result, PositionSizing)
            assert result.risk_pct <= 0.010 + 1e-9, f"grade {grade} exceeded 1% ceiling"

    def test_unknown_grade_falls_back(self):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc, grade="ZZZ")
        assert isinstance(result, PositionSizing)
        assert result.risk_pct == pytest.approx(0.0075, abs=0.0005)


# ── Risk gate: invalid stop ───────────────────────────────────────────────── #

class TestInvalidStop:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_blocks_zero_stop_distance(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0850, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "INVALID_STOP"

    def test_blocks_long_stop_above_entry(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0900, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "INVALID_STOP_DIRECTION"

    def test_blocks_short_stop_below_entry(self):
        acc = _fresh_account()
        result = self.engine.size("SHORT", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "INVALID_STOP_DIRECTION"


# ── Suggest stop ──────────────────────────────────────────────────────────── #

class TestSuggestStop:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_long_stop_below_entry(self):
        stop = self.engine.suggest_stop(entry=1.0850, direction="LONG", atr=0.0030)
        assert stop < 1.0850

    def test_short_stop_above_entry(self):
        stop = self.engine.suggest_stop(entry=1.0850, direction="SHORT", atr=0.0030)
        assert stop > 1.0850

    def test_stop_distance_equals_atr_mult(self):
        entry = 1.0850
        atr = 0.0030
        stop = self.engine.suggest_stop(entry, "LONG", atr)
        expected_distance = self.engine.stop_atr_mult * atr
        assert abs(entry - stop) == pytest.approx(expected_distance, rel=1e-6)


# ── Isolation guard ───────────────────────────────────────────────────────── #

def test_micro_risk_does_not_import_sovereign():
    """The micro_risk module must not depend on any Sovereign components."""
    import importlib, sys
    mod = importlib.import_module("ict.micro_risk")
    source_file = getattr(mod, "__file__", "")
    # Check none of the loaded modules that micro_risk imports are from sovereign
    import ict.micro_risk as m
    import inspect
    src = inspect.getsource(m)
    forbidden = ["from sovereign", "import sovereign", "from layer2", "import layer2",
                 "from layer1", "import layer1", "from config.loader", "from config import loader"]
    for phrase in forbidden:
        assert phrase not in src, f"Isolation violated: '{phrase}' found in micro_risk.py"
