"""Tests for sovereign/training/state_space.py — the locked S(t) schema (spec §4.1)."""
import numpy as np
import pytest

from sovereign.training.state_space import (
    NUM_STATE_DIMS,
    STATE_DIMS,
    build_state,
    carry_alignment,
    drawdown_from_peak,
)


def test_schema_locked_dims_and_order():
    """The 8-dim, fixed-order contract. Any drift here is a breaking change."""
    assert NUM_STATE_DIMS == 8
    assert STATE_DIMS == (
        "hold_frac", "excursion_pct", "atr_pct", "rsi_14", "carry_alignment",
        "rate_diff_z", "cot_z", "drawdown_from_peak",
    )


def _record(**overrides):
    base = dict(
        hold_count=3, hold_limit=6, excursion_pct=0.01, atr_pct=0.008,
        rsi_14=55.0, carry_alignment=1.0, rate_diff_z=0.5, cot_z=-0.2,
        drawdown_from_peak=0.002,
    )
    base.update(overrides)
    return base


def test_build_state_shape_and_order():
    s = build_state(_record())
    assert s.shape == (8,)
    assert s.dtype == np.float64
    assert s[0] == pytest.approx(0.5)   # hold_frac = 3/6
    assert s[4] == pytest.approx(1.0)   # carry_alignment
    assert s[5] == pytest.approx(0.5)   # rate_diff_z
    assert s[6] == pytest.approx(-0.2)  # cot_z


def test_build_state_missing_required_field_raises():
    rec = _record()
    del rec["atr_pct"]
    with pytest.raises(KeyError):
        build_state(rec)


def test_build_state_optional_fields_default_to_zero():
    rec = _record(rate_diff_z=None, cot_z=None)
    s = build_state(rec)
    assert s[5] == 0.0
    assert s[6] == 0.0


def test_hold_frac_clipped_to_unity():
    rec = _record(hold_count=99, hold_limit=6)
    s = build_state(rec)
    assert s[0] == 1.0


def test_hold_frac_zero_limit_defensive():
    rec = _record(hold_count=0, hold_limit=0)
    s = build_state(rec)
    assert s[0] == 1.0


def test_carry_alignment_helper():
    assert carry_alignment(1, 1) == 1.0
    assert carry_alignment(1, -1) == -1.0
    assert carry_alignment(-1, -1) == 1.0
    assert carry_alignment(1, 0) == 0.0


def test_drawdown_from_peak_helper():
    assert drawdown_from_peak(1, best_price=110.0, close_now=110.0) == 0.0
    assert drawdown_from_peak(1, best_price=110.0, close_now=105.0) == pytest.approx(5.0 / 110.0)
    assert drawdown_from_peak(-1, best_price=90.0, close_now=95.0) == pytest.approx(5.0 / 90.0)
    assert drawdown_from_peak(1, best_price=0.0, close_now=1.0) == 0.0
