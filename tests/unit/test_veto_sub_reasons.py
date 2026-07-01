"""
tests/unit/test_veto_sub_reasons.py
===================================
Gate vetoes must record WHY they fired, not just the gate name.

`build_gate_veto_reason` maps a blocked A-grade signal to
``(veto_stage, veto_reason)``. These tests assert that each gate's reason
string carries the SPECIFIC triggering values (similarity, floor, magnet,
bias direction, time) — the sub-reason the veto ledger persists.
"""
from __future__ import annotations

from ict.orchestrator import build_gate_veto_reason


def _base(**overrides):
    """Defaults where nothing is vetoing — override one gate per test."""
    kwargs = dict(
        mem_veto=False,
        mem_similarity=0.0,
        mem_historical_wr=0.0,
        floor_ok=True,
        decision_score=7.0,
        score_floor=None,
        heatmap_conflict=False,
        heatmap_detail=None,
        bias_agrees=True,
        bias_dir="NEUTRAL",
        signal_direction="LONG",
        session_ok=True,
        session_name="London",
        pair="GBPUSD",
        blackout=False,
        grade="A",
        score=7.5,
        time_utc="14:30 UTC",
    )
    kwargs.update(overrides)
    return kwargs


def test_memory_veto_carries_similarity_and_wr():
    stage, reason = build_gate_veto_reason(
        **_base(mem_veto=True, mem_similarity=0.83, mem_historical_wr=0.31)
    )
    assert stage == "memory"
    assert "MEMORY_VETO" in reason
    assert "0.83" in reason        # similarity
    assert "0.31" in reason        # historical win rate


def test_memory_floor_carries_score_vs_floor():
    stage, reason = build_gate_veto_reason(
        **_base(floor_ok=False, decision_score=6.1, score_floor=6.8)
    )
    assert stage == "memory"
    assert "MEMORY_FLOOR" in reason
    assert "6.10" in reason        # decision score
    assert "6.80" in reason        # cluster floor


def test_memory_floor_handles_none_floor_without_crashing():
    stage, reason = build_gate_veto_reason(
        **_base(floor_ok=False, decision_score=6.1, score_floor=None)
    )
    assert stage == "memory"
    assert "MEMORY_FLOOR" in reason
    assert "n/a" in reason


def test_heatmap_conflict_carries_magnet_detail():
    detail = "magnet 1.08500 prob=0.82 closer than TP1"
    stage, reason = build_gate_veto_reason(
        **_base(heatmap_conflict=True, heatmap_detail=detail)
    )
    assert stage == "heatmap"
    assert "HEATMAP_CONFLICT" in reason
    assert detail in reason


def test_heatmap_conflict_without_detail_still_specific():
    stage, reason = build_gate_veto_reason(
        **_base(heatmap_conflict=True, heatmap_detail=None)
    )
    assert stage == "heatmap"
    assert "HEATMAP_CONFLICT" in reason


def test_bias_conflict_carries_directions():
    stage, reason = build_gate_veto_reason(
        **_base(bias_agrees=False, bias_dir="SHORT", signal_direction="LONG")
    )
    assert stage == "bias"
    assert "BIAS_CONFLICT" in reason
    assert "SHORT" in reason       # library bias
    assert "LONG" in reason        # signal direction


def test_session_block_carries_time_and_pair():
    stage, reason = build_gate_veto_reason(
        **_base(session_ok=False, session_name="NY_PM", pair="EURUSD",
                time_utc="18:05 UTC")
    )
    assert stage == "session"
    assert "SESSION_BLOCK" in reason
    assert "NY_PM" in reason
    assert "EURUSD" in reason
    assert "18:05 UTC" in reason   # the time


def test_blackout_named_explicitly():
    stage, reason = build_gate_veto_reason(**_base(blackout=True, pair="AUDUSD"))
    assert stage == "gate"
    assert "BLACKOUT" in reason
    assert "AUDUSD" in reason


def test_generic_gate_fallback_carries_grade_and_score():
    # Nothing above vetoes but the signal is non-actionable → generic gate.
    stage, reason = build_gate_veto_reason(**_base(grade="A", score=7.42))
    assert stage == "gate"
    assert "GATE" in reason
    assert "7.42" in reason


def test_precedence_memory_veto_beats_lower_gates():
    # When several gates would fire, memory veto wins (mirrors is_actionable order).
    stage, reason = build_gate_veto_reason(
        **_base(mem_veto=True, mem_similarity=0.5, mem_historical_wr=0.2,
                heatmap_conflict=True, bias_agrees=False, session_ok=False)
    )
    assert stage == "memory"
    assert "MEMORY_VETO" in reason
