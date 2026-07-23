"""Tests for sovereign.briefing.briefing_context — the AlphaZero sizing multiplier.

Invariants under test (all freeze-safe, provenance-honest):
  - the multiplier is ALWAYS strictly positive and inside [FLOOR, CEIL] → it can never veto
  - it fails to NEUTRAL (exactly 1.0) on deterministic fallback / missing confidence
  - higher confidence sizes up, lower sizes down, within the conservative band
  - ROTATION_WARN applies a quality haircut
  - direction-vs-carry alignment nudges only when a same-context carry direction is supplied
  - verified is False everywhere (the briefing is never a proven edge)
"""
from sovereign.briefing import briefing_context as bc


def test_floor_ceil_band_constants_sane():
    assert 0.0 < bc.MULT_FLOOR < 1.0 < bc.MULT_CEIL


def test_deterministic_fallback_neutralises():
    m = bc.compute_multiplier(0, "NEUTRAL", synthesis_source="deterministic_fallback",
                              regime_call="ROTATION_WARN")
    assert m["multiplier"] == 1.0
    assert m["effect_applied"] is False
    assert m["verified"] is False


def test_missing_confidence_neutralises():
    m = bc.compute_multiplier(None, "LONG", synthesis_source="claude-opus-4-8")
    assert m["multiplier"] == 1.0
    assert m["effect_applied"] is False


def test_confidence_monotonic_and_bounded():
    low = bc.compute_multiplier(20, "LONG", synthesis_source="claude-opus-4-8")["multiplier"]
    mid = bc.compute_multiplier(50, "LONG", synthesis_source="claude-opus-4-8")["multiplier"]
    high = bc.compute_multiplier(95, "LONG", synthesis_source="claude-opus-4-8")["multiplier"]
    assert low < mid < high
    assert mid == 1.0                      # confidence 50 is the neutral point
    for v in (low, mid, high):
        assert bc.MULT_FLOOR <= v <= bc.MULT_CEIL


def test_never_vetoes_even_at_zero_confidence():
    # A real (non-fallback) call at confidence 0 still returns a positive size, never 0.
    m = bc.compute_multiplier(0, "SHORT", synthesis_source="claude-opus-4-8")
    assert m["multiplier"] >= bc.MULT_FLOOR > 0.0


def test_rotation_warn_haircut():
    base = bc.compute_multiplier(80, "LONG", synthesis_source="claude-opus-4-8")["multiplier"]
    warned = bc.compute_multiplier(80, "LONG", synthesis_source="claude-opus-4-8",
                                   regime_call="ROTATION_WARN")["multiplier"]
    assert warned < base


def test_carry_alignment_nudges_only_with_direction():
    none = bc.compute_multiplier(80, "LONG", synthesis_source="claude-opus-4-8")["multiplier"]
    agree = bc.compute_multiplier(80, "LONG", synthesis_source="claude-opus-4-8",
                                  carry_direction="LONG")["multiplier"]
    oppose = bc.compute_multiplier(80, "LONG", synthesis_source="claude-opus-4-8",
                                   carry_direction="SHORT")["multiplier"]
    assert oppose < none < agree


def test_stamp_shape_when_briefing_present(tmp_path, monkeypatch):
    import json
    p = tmp_path / "daily_briefing.json"
    p.write_text(json.dumps({
        "date": "2026-07-23", "generated_at": "2026-07-23T12:00:00+00:00",
        "directional_bias": "LONG", "confidence": 72, "regime_call": "BREADTH",
        "key_level": 22400, "synthesis_source": "claude-opus-4-8",
        "provenance": {"verified": False},
    }))
    monkeypatch.setattr(bc, "BRIEFING_JSON", p)
    stamp = bc.briefing_stamp()
    assert stamp["directional_bias"] == "LONG"
    assert stamp["confidence"] == 72
    assert stamp["verified"] is False
    assert stamp["role"] == "context_only"


def test_stamp_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "BRIEFING_JSON", tmp_path / "nope.json")
    assert bc.briefing_stamp() is None
