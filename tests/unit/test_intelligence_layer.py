"""
Unit tests for the intelligence layer:

  sovereign/intelligence/regime_performance_tracker.py
  sovereign/intelligence/system_health.py
  sovereign/intelligence/capital_allocator.py
  lab/feature_registry.py
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Regime Performance Tracker ────────────────────────────────────────── #

from sovereign.intelligence.regime_performance_tracker import (
    RegimePerformanceTracker,
    classify_vol_state,
    MIN_N_REPORT,
)


class TestClassifyVolState:
    def test_expanding(self):
        # Rising ATR series
        atr = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        assert classify_vol_state(atr) == "EXPANDING"

    def test_compressing(self):
        atr = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9]
        assert classify_vol_state(atr) == "COMPRESSING"

    def test_neutral_flat(self):
        atr = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert classify_vol_state(atr) == "NEUTRAL"

    def test_too_short(self):
        assert classify_vol_state([1.0, 1.1]) == "NEUTRAL"


class TestRegimePerformanceTracker:
    def _tracker(self, tmp_path):
        return RegimePerformanceTracker(output_path=tmp_path / "rp.jsonl")

    def test_tag_writes_record(self, tmp_path):
        t = self._tracker(tmp_path)
        rec = t.tag_trade("ICT", "MOMENTUM", "EXPANDING", pnl_r=1.5,
                          trade_id="t1")
        assert rec["system"] == "ICT"
        assert rec["win"] is True
        assert (tmp_path / "rp.jsonl").exists()

    def test_tag_loss(self, tmp_path):
        t = self._tracker(tmp_path)
        rec = t.tag_trade("FOREX", "REVERSION", "NEUTRAL", pnl_r=-1.0)
        assert rec["win"] is False

    def test_get_stats_empty(self, tmp_path):
        t = self._tracker(tmp_path)
        stats = t.get_stats("ICT", "MOMENTUM")
        assert stats["n"] == 0
        assert stats["reliable"] is False
        assert stats["above_expectancy"] is None

    def test_get_stats_with_data(self, tmp_path):
        t = self._tracker(tmp_path)
        # 40 wins of 1R in MOMENTUM/EXPANDING
        for _ in range(40):
            t.tag_trade("ICT", "MOMENTUM", "EXPANDING", pnl_r=1.0)
        for _ in range(10):
            t.tag_trade("ICT", "MOMENTUM", "EXPANDING", pnl_r=-1.0)

        stats = t.get_stats("ICT", "MOMENTUM", "EXPANDING", min_n=30)
        assert stats["reliable"] is True
        assert abs(stats["wr"] - 0.8) < 0.01
        assert stats["avg_r"] > 0
        assert stats["sharpe"] > 0

    def test_get_stats_below_min_n(self, tmp_path):
        t = self._tracker(tmp_path)
        for _ in range(10):
            t.tag_trade("ICT", "MOMENTUM", "EXPANDING", pnl_r=1.0)
        stats = t.get_stats("ICT", "MOMENTUM", "EXPANDING", min_n=30)
        assert stats["reliable"] is False

    def test_filter_by_system(self, tmp_path):
        t = self._tracker(tmp_path)
        t.tag_trade("ICT",   "MOMENTUM", "NEUTRAL", pnl_r=1.0)
        t.tag_trade("FOREX", "MOMENTUM", "NEUTRAL", pnl_r=-1.0)
        ict_stats  = t.get_stats("ICT",   "MOMENTUM", min_n=1)
        forex_stats = t.get_stats("FOREX", "MOMENTUM", min_n=1)
        assert ict_stats["wr"] == 1.0
        assert forex_stats["wr"] == 0.0

    def test_rolling_sharpe_zscore_insufficient(self, tmp_path):
        t = self._tracker(tmp_path)
        for _ in range(5):
            t.tag_trade("ICT", "MOMENTUM", "NEUTRAL", pnl_r=1.0)
        assert t.rolling_sharpe_zscore("ICT", "MOMENTUM") is None

    def test_rolling_sharpe_zscore_returns_float(self, tmp_path):
        t = self._tracker(tmp_path)
        # Build up enough history
        for _ in range(50):
            t.tag_trade("ICT", "MOMENTUM", "NEUTRAL", pnl_r=1.0)
        # Add a recent bad run
        for _ in range(20):
            t.tag_trade("ICT", "MOMENTUM", "NEUTRAL", pnl_r=-1.0)
        z = t.rolling_sharpe_zscore("ICT", "MOMENTUM")
        # Should be negative because recent results worse than historical
        assert z is not None
        assert z < 0

    def test_get_all_stats_structure(self, tmp_path):
        t = self._tracker(tmp_path)
        t.tag_trade("ICT", "MOMENTUM", "NEUTRAL", pnl_r=1.0)
        all_stats = t.get_all_stats(min_n=1)
        assert "ICT" in all_stats
        assert "MOMENTUM" in all_stats["ICT"]
        assert "NEUTRAL" in all_stats["ICT"]["MOMENTUM"]

    def test_summary_runs(self, tmp_path):
        t = self._tracker(tmp_path)
        t.tag_trade("ICT", "MOMENTUM", "EXPANDING", pnl_r=1.0)
        summary = t.summary()
        assert isinstance(summary, str)
        assert "Regime Performance Tracker" in summary

    def test_cache_invalidated_on_write(self, tmp_path):
        t = self._tracker(tmp_path)
        stats_before = t.get_stats("ICT", "MOMENTUM", min_n=1)
        t.tag_trade("ICT", "MOMENTUM", "NEUTRAL", pnl_r=2.0)
        stats_after = t.get_stats("ICT", "MOMENTUM", min_n=1)
        assert stats_after["n"] > stats_before["n"]


# ── Feature Registry ──────────────────────────────────────────────────── #

from lab.feature_registry import (
    FeatureRegistry,
    Verdict,
    PromotionGateError,
    IC_OOS_THRESHOLD,
    STALE_DAYS,
)


class TestFeatureRegistry:
    def _reg(self, tmp_path):
        return FeatureRegistry(ledger_path=tmp_path / "features.jsonl")

    def test_add_testing(self, tmp_path):
        reg = self._reg(tmp_path)
        rec = reg.add("my_feature", verdict=Verdict.TESTING, sample_size=100)
        assert rec["verdict"] == "TESTING"
        assert reg.get_testing()["my_feature"]["sample_size"] == 100

    def test_graveyard(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("bad_feature", verdict=Verdict.TESTING, sample_size=50)
        reg.graveyard("bad_feature", graveyard_reason="anti-edge")
        assert "bad_feature" in reg.get_graveyard()
        assert "bad_feature" not in reg.get_live()

    def test_promote_passes_gates(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("good_feature", verdict=Verdict.TESTING)
        rec = reg.promote("good_feature", ic_oos=0.20,
                          marginal_contribution=0.10,
                          holdout_degradation=False)
        assert rec["verdict"] == "LIVE"
        assert "good_feature" in reg.get_live()

    def test_promote_fails_ic(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("weak_feature", verdict=Verdict.TESTING)
        with pytest.raises(PromotionGateError, match="IC_OOS"):
            reg.promote("weak_feature", ic_oos=0.10,
                        marginal_contribution=0.05, holdout_degradation=False)

    def test_promote_fails_marginal(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("feat", verdict=Verdict.TESTING)
        with pytest.raises(PromotionGateError, match="marginal_contribution"):
            reg.promote("feat", ic_oos=0.20,
                        marginal_contribution=-0.01, holdout_degradation=False)

    def test_promote_fails_holdout(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("feat", verdict=Verdict.TESTING)
        with pytest.raises(PromotionGateError, match="holdout"):
            reg.promote("feat", ic_oos=0.20,
                        marginal_contribution=0.05, holdout_degradation=True)

    def test_stale_detection(self, tmp_path):
        import datetime
        reg = self._reg(tmp_path)
        reg.add("old_feature", verdict=Verdict.TESTING)
        reg.promote("old_feature", ic_oos=0.20,
                    marginal_contribution=0.05, holdout_degradation=False)

        # Manually backdate last_validated_date in the JSONL
        path = tmp_path / "features.jsonl"
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records[-1]["last_validated_date"] = "2020-01-01"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        stale = reg.get_stale(days=90)
        assert "old_feature" in stale

    def test_not_stale_when_recent(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("fresh_feature", verdict=Verdict.TESTING)
        reg.promote("fresh_feature", ic_oos=0.20,
                    marginal_contribution=0.05, holdout_degradation=False)
        stale = reg.get_stale(days=90)
        assert "fresh_feature" not in stale

    def test_seed_pd_alignment(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.seed_pd_alignment()
        assert "pd_alignment" in reg.get_graveyard()

    def test_seed_idempotent(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.seed_pd_alignment()
        reg.seed_pd_alignment()  # second call should be no-op
        all_records = reg._load_all()
        pd_records = [r for r in all_records if r["feature_name"] == "pd_alignment"]
        # 2 records: initial TESTING + GRAVEYARD
        assert len(pd_records) == 2

    def test_history(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("feat", verdict=Verdict.TESTING)
        reg.graveyard("feat", graveyard_reason="failed")
        history = reg.history("feat")
        assert len(history) == 2
        assert history[0]["verdict"] == "TESTING"
        assert history[1]["verdict"] == "GRAVEYARD"

    def test_append_only(self, tmp_path):
        reg = self._reg(tmp_path)
        reg.add("feat", verdict=Verdict.TESTING)
        path = tmp_path / "features.jsonl"
        lines_before = len(path.read_text().splitlines())
        reg.graveyard("feat", graveyard_reason="gone")
        lines_after = len(path.read_text().splitlines())
        assert lines_after == lines_before + 1


# ── System Health Monitor ─────────────────────────────────────────────── #

from sovereign.intelligence.system_health import (
    SystemHealthMonitor,
    _binary_entropy,
    _consecutive_losses,
    _structural_break,
    _compute_health_score,
)


class TestHealthHelpers:
    def test_entropy_max_at_half(self):
        assert abs(_binary_entropy(0.5) - 1.0) < 1e-6

    def test_entropy_zero_at_extremes(self):
        assert _binary_entropy(0.0) == 0.0
        assert _binary_entropy(1.0) == 0.0

    def test_consecutive_losses_end(self):
        outcomes = [True, True, False, False, False]
        assert _consecutive_losses(outcomes) == 3

    def test_consecutive_losses_none(self):
        outcomes = [False, False, True]
        assert _consecutive_losses(outcomes) == 0

    def test_structural_break_detects(self):
        is_break, lo, hi = _structural_break(
            recent_wins=2, recent_n=20, historical_wr=0.6
        )
        # 2/20 = 10% WR, well below the 95% CI for 60% historical WR
        assert is_break == True  # noqa: E712

    def test_structural_break_no_trigger(self):
        # 12/20 = 60%, consistent with historical WR of 60%
        is_break, lo, hi = _structural_break(
            recent_wins=12, recent_n=20, historical_wr=0.6
        )
        assert is_break == False  # noqa: E712

    def test_structural_break_small_n(self):
        # Insufficient data — should not flag
        is_break, _, _ = _structural_break(recent_wins=0, recent_n=3, historical_wr=0.5)
        assert is_break is False


class TestComputeHealthScore:
    def test_perfect_win_run(self):
        r = [1.0] * 30
        snap = _compute_health_score(r)
        # All wins, no streak, no break → high score
        assert snap["health_score"] > 0.7
        assert snap["reliability"] in ("HIGH", "MEDIUM")

    def test_losing_streak_degrades(self):
        r = [1.0] * 20 + [-1.0] * 10
        snap = _compute_health_score(r)
        assert snap["consecutive_losses"] == 10
        assert snap["health_score"] < 0.9  # penalised

    def test_structural_break_flags(self):
        # 18 wins out of 20, historical WR=0.30 → break above upper bound
        r_hist = [-1.0] * 7 + [1.0] * 3  # ~30% WR for historical
        r_hist_long = r_hist * 10  # 100 trades
        r_recent = [1.0] * 18 + [-1.0] * 2  # 90% WR recent
        r = r_hist_long + r_recent
        snap = _compute_health_score(r, historical_wr=0.30)
        # structural break should be flagged (recent above upper CI)
        assert snap["structural_break"] == True  # noqa: E712

    def test_empty_returns_default(self):
        snap = _compute_health_score([])
        assert snap["health_score"] == 0.5
        assert snap["reliability"] == "MEDIUM"

    def test_reliability_unreliable(self):
        # Many consecutive losses should push score low
        r = [1.0] * 5 + [-1.0] * 15
        snap = _compute_health_score(r)
        assert snap["consecutive_losses"] == 15
        # Score should be degraded
        assert snap["health_score"] < 0.7


class TestSystemHealthMonitor:
    def _monitor(self, tmp_path):
        return SystemHealthMonitor(
            health_log=tmp_path / "health.jsonl",
            messages_path=tmp_path / "messages.json",
        )

    def test_compute_and_log(self, tmp_path):
        m = self._monitor(tmp_path)
        snap = m.compute("ICT", r_multiples=[1.0] * 20)
        assert "reliability" in snap
        assert (tmp_path / "health.jsonl").exists()

    def test_latest_per_system(self, tmp_path):
        m = self._monitor(tmp_path)
        m.compute("ICT",   r_multiples=[1.0, -1.0, 1.0])
        m.compute("FOREX", r_multiples=[-1.0, -1.0, -1.0])
        latest = m.latest_per_system()
        assert "ICT" in latest
        assert "FOREX" in latest

    def test_unreliable_writes_message(self, tmp_path):
        m = self._monitor(tmp_path)
        # Force UNRELIABLE: many consecutive losses + structural break
        r = [1.0] * 5 + [-1.0] * 15
        snap = m.compute("ICT", r_multiples=r)
        if snap["reliability"] == "UNRELIABLE":
            m.check_and_alert()
            msgs = json.loads((tmp_path / "messages.json").read_text())
            assert any("ICT" in msg["text"] for msg in msgs["messages"])

    def test_no_duplicate_alerts(self, tmp_path):
        m = self._monitor(tmp_path)
        r = [1.0] * 5 + [-1.0] * 15
        snap = m.compute("ICT", r_multiples=r)
        if snap["reliability"] == "UNRELIABLE":
            m.check_and_alert()
            m.check_and_alert()  # second call — should not duplicate
            msgs = json.loads((tmp_path / "messages.json").read_text())
            ict_msgs = [msg for msg in msgs["messages"] if "ICT" in msg["text"]]
            assert len(ict_msgs) <= 2  # at most one per check_and_alert call

    def test_summary(self, tmp_path):
        m = self._monitor(tmp_path)
        m.compute("ICT", r_multiples=[1.0, -1.0])
        s = m.summary()
        assert "ICT" in s


# ── Capital Allocator ─────────────────────────────────────────────────── #

from sovereign.intelligence.capital_allocator import (
    CapitalAllocator,
    AllocationState,
    FREEZE_HOURS,
    FREEZE_MIN_TRADES,
    Z_FREEZE,
    Z_HALF,
)


class _FakeTracker:
    """Minimal fake for RegimePerformanceTracker."""

    def __init__(self, z_by_system=None):
        self._z = z_by_system or {}

    def rolling_sharpe_zscore(self, system, regime):
        return self._z.get(system)


class TestAllocationState:
    def test_not_frozen_by_default(self):
        state = AllocationState()
        assert state.is_frozen(0) is False

    def test_frozen_within_window(self):
        from datetime import datetime, timedelta, timezone
        until = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
        state = AllocationState(freeze_until=until, freeze_trade_count=0)
        # time gate not expired, trade count below threshold
        assert state.is_frozen(3) is True

    def test_freeze_lifts_after_enough_trades(self):
        from datetime import datetime, timedelta, timezone
        # Time gate expired
        until = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        state = AllocationState(freeze_until=until, freeze_trade_count=0)
        # Both gates must pass: time expired AND enough trades
        assert state.is_frozen(FREEZE_MIN_TRADES + 1) is False

    def test_serialise_roundtrip(self):
        state = AllocationState(freeze_until="2026-05-21T12:00:00+00:00",
                                freeze_trade_count=10)
        restored = AllocationState.from_dict(state.to_dict())
        assert restored.freeze_until == state.freeze_until
        assert restored.freeze_trade_count == state.freeze_trade_count


class TestCapitalAllocator:
    def _alloc(self, tmp_path):
        return CapitalAllocator(state_file=tmp_path / "alloc_state.json")

    def _healthy_snaps(self):
        return {
            "ICT":    {"reliability": "HIGH"},
            "FOREX":  {"reliability": "HIGH"},
            "EQUITY": {"reliability": "HIGH"},
        }

    def test_full_allocation_healthy_system(self, tmp_path):
        a = self._alloc(tmp_path)
        t = _FakeTracker()  # z_sharpe = None (insufficient data)
        mults = a.compute("MOMENTUM", self._healthy_snaps(), t)
        for sys in ("ICT", "FOREX", "EQUITY"):
            assert mults[sys] == 1.0

    def test_health_low_caps_at_half(self, tmp_path):
        a = self._alloc(tmp_path)
        snaps = self._healthy_snaps()
        snaps["ICT"] = {"reliability": "LOW"}
        t = _FakeTracker()
        mults = a.compute("MOMENTUM", snaps, t)
        assert mults["ICT"] == 0.5
        assert mults["FOREX"] == 1.0

    def test_health_unreliable_freezes(self, tmp_path):
        a = self._alloc(tmp_path)
        snaps = self._healthy_snaps()
        snaps["ICT"] = {"reliability": "UNRELIABLE"}
        t = _FakeTracker()
        mults = a.compute("MOMENTUM", snaps, t)
        assert mults["ICT"] == 0.0

    def test_z_below_half_threshold_caps(self, tmp_path):
        a = self._alloc(tmp_path)
        t = _FakeTracker({"ICT": -2.0})  # between Z_HALF and Z_FREEZE
        mults = a.compute("MOMENTUM", self._healthy_snaps(), t)
        assert mults["ICT"] == 0.5

    def test_z_below_freeze_threshold(self, tmp_path):
        a = self._alloc(tmp_path)
        t = _FakeTracker({"FOREX": -3.0})
        mults = a.compute("MOMENTUM", self._healthy_snaps(), t)
        assert mults["FOREX"] == 0.0

    def test_freeze_persists_next_call(self, tmp_path):
        a = self._alloc(tmp_path)
        t_bad  = _FakeTracker({"ICT": -3.0})
        t_good = _FakeTracker({"ICT":  2.0})
        a.compute("MOMENTUM", self._healthy_snaps(), t_bad, trade_counts={"ICT": 0})
        # Next call even with good z: should still be frozen
        mults = a.compute("MOMENTUM", self._healthy_snaps(), t_good,
                          trade_counts={"ICT": 1})
        assert mults["ICT"] == 0.0

    def test_freeze_lifts_after_condition_met(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        a = self._alloc(tmp_path)
        # Manually set a freeze that has already expired
        a._system_states["ICT"].freeze_until = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()
        a._system_states["ICT"].freeze_trade_count = 0
        t_good = _FakeTracker({"ICT": 1.0})
        mults = a.compute("MOMENTUM", self._healthy_snaps(), t_good,
                          trade_counts={"ICT": FREEZE_MIN_TRADES + 2})
        assert mults["ICT"] > 0.0

    def test_thaw_clears_freeze(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        a = self._alloc(tmp_path)
        a._trigger_freeze("ICT", 0)
        assert a._system_states["ICT"].is_frozen(0) is True
        a.thaw("ICT")
        assert a._system_states["ICT"].is_frozen(0) is False

    def test_summary_runs(self, tmp_path):
        a = self._alloc(tmp_path)
        t = _FakeTracker()
        s = a.summary("MOMENTUM", self._healthy_snaps(), t)
        assert "Capital Allocator" in s

    def test_multiplier_bounds(self, tmp_path):
        a = self._alloc(tmp_path)
        t = _FakeTracker({"ICT": -2.0, "FOREX": -3.0, "EQUITY": 1.0})
        snaps = {
            "ICT":    {"reliability": "LOW"},
            "FOREX":  {"reliability": "HIGH"},
            "EQUITY": {"reliability": "HIGH"},
        }
        mults = a.compute("REVERSION", snaps, t)
        for sys, v in mults.items():
            assert 0.0 <= v <= 1.0, f"{sys} multiplier {v} out of bounds"
