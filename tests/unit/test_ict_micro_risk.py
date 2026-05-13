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
        """Verify the default values come from ict_params.yml (2 %, 30×), not parameters.yml."""
        e = MicroRiskEngine()
        assert e.max_risk_per_trade == pytest.approx(0.02, abs=0.001)
        assert e.max_leverage == pytest.approx(30.0, abs=0.1)


# ── Happy-path sizing ────────────────────────────────────────────────────── #

class TestPositionSizing:
    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_returns_position_sizing(self):
        acc = _fresh_account()
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)

    def test_risk_dollars_correct(self):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        expected_risk = 10_000 * 0.02  # 2 %
        assert result.risk_dollars == pytest.approx(expected_risk, rel=0.01)

    def test_risk_pct_is_max_risk(self):
        acc = _fresh_account(size=10_000)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)
        assert result.risk_pct == pytest.approx(0.02, abs=0.001)

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
        acc = _fresh_account(daily_loss=0.06)  # 6 % > 5 % limit
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "DAILY_LOSS_LIMIT"

    def test_passes_below_daily_loss_limit(self):
        acc = _fresh_account(daily_loss=0.03)  # 3 % < 5 % limit
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
        acc = _fresh_account(open_risk=0.05)  # 5% already open; adding 2% = 7% > 6%
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "MAX_CONCURRENT_RISK"

    def test_passes_with_room(self):
        acc = _fresh_account(open_risk=0.03)  # 3% + 2% = 5% < 6%
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)


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


# ── Phase 6: total drawdown gate ─────────────────────────────────────────── #

class TestTotalDrawdownGate:
    """Phase 6: max_total_dd_pct — prop-firm account preservation ceiling."""

    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_blocks_when_total_dd_at_limit(self):
        acc = _fresh_account()
        # Build a MicroRiskParams with total_dd_pct at the ceiling
        from ict.micro_risk import MicroRiskParams
        acc = MicroRiskParams(
            account_size=10_000.0,
            total_dd_pct=self.engine.max_total_dd_pct,   # exactly at limit
        )
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "MAX_TOTAL_DD"

    def test_blocks_when_total_dd_exceeds_limit(self):
        from ict.micro_risk import MicroRiskParams
        acc = MicroRiskParams(
            account_size=10_000.0,
            total_dd_pct=self.engine.max_total_dd_pct + 0.01,  # over limit
        )
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "MAX_TOTAL_DD"

    def test_passes_below_total_dd_limit(self):
        from ict.micro_risk import MicroRiskParams
        acc = MicroRiskParams(
            account_size=10_000.0,
            total_dd_pct=self.engine.max_total_dd_pct - 0.01,  # below limit
        )
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)

    def test_total_dd_gate_fires_before_daily_loss(self):
        """Total DD is checked before daily loss (gate ordering)."""
        from ict.micro_risk import MicroRiskParams
        # Both total DD and daily loss are exceeded
        acc = MicroRiskParams(
            account_size=10_000.0,
            total_dd_pct=self.engine.max_total_dd_pct + 0.05,
            daily_loss_pct=self.engine.max_daily_loss_pct + 0.01,
        )
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        # Kill switch fires first (gate 0), then total DD (gate 1), then daily loss (gate 2)
        assert result.reason in ("KILL_SWITCH", "MAX_TOTAL_DD")


# ── Phase 6: configurable kill-switch ────────────────────────────────────── #

class TestConfigurableKillSwitch:
    """Phase 6: max_consecutive_losses comes from config, not hardcoded 3."""

    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_kill_switch_respects_configured_limit(self):
        from ict.micro_risk import MicroRiskParams
        limit = self.engine.max_consecutive_losses
        # At limit → vetoed
        acc = MicroRiskParams(account_size=10_000.0, consecutive_losses=limit)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        assert result.reason == "KILL_SWITCH"

    def test_below_kill_switch_limit_passes(self):
        from ict.micro_risk import MicroRiskParams
        limit = self.engine.max_consecutive_losses
        acc = MicroRiskParams(account_size=10_000.0, consecutive_losses=limit - 1)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, PositionSizing)

    def test_kill_switch_detail_mentions_limit(self):
        from ict.micro_risk import MicroRiskParams
        limit = self.engine.max_consecutive_losses
        acc = MicroRiskParams(account_size=10_000.0, consecutive_losses=limit)
        result = self.engine.size("LONG", 1.0850, 1.0820, atr=0.0030, params=acc)
        assert isinstance(result, RiskVeto)
        # Detail should mention the configured limit
        assert str(limit) in result.detail


# ── Phase 1: TP overrides accepted by size() ─────────────────────────────── #

class TestTPOverrides:
    """Phase 1: size() accepts tp1_override / tp2_override from liquidity targets."""

    def setup_method(self):
        self.engine = MicroRiskEngine()

    def test_valid_tp1_override_used(self):
        acc = _fresh_account()
        entry = 1.0850
        stop  = 1.0820    # stop_dist = 0.0030; min TP1 for 2R = 1.0850 + 2×0.003 = 1.0910
        tp1_level = 1.0920  # clearly above 2R threshold
        result = self.engine.size(
            "LONG", entry, stop, atr=0.0030, params=acc,
            tp1_override=tp1_level,
        )
        assert isinstance(result, PositionSizing)
        assert result.tp1 == pytest.approx(tp1_level, abs=1e-6)

    def test_invalid_tp1_override_falls_back_to_r_multiple(self):
        """A TP1 on the wrong side of entry must be discarded in favour of R-multiple fallback."""
        acc = _fresh_account()
        bad_tp1 = 1.0800  # below entry 1.0850 — wrong side for LONG
        result = self.engine.size(
            "LONG", 1.0850, 1.0820, atr=0.0030, params=acc,
            tp1_override=bad_tp1,
        )
        assert isinstance(result, PositionSizing)
        # Fallback R-multiple TP1 for LONG must be above entry
        assert result.tp1 > 1.0850

    def test_rr_ratio_computed_from_actual_tp1(self):
        acc = _fresh_account()
        entry = 1.0850
        stop  = 1.0820
        tp1   = 1.0910   # 2R exactly (stop_dist = 0.0030, tp1 = entry + 2×0.0030)
        result = self.engine.size("LONG", entry, stop, atr=0.0030, params=acc, tp1_override=tp1)
        assert isinstance(result, PositionSizing)
        expected_rr = (tp1 - entry) / abs(entry - stop)
        assert result.rr_ratio == pytest.approx(expected_rr, rel=0.01)


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
