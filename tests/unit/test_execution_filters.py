"""Pure frozen-filter behaviour, straddling each threshold on both sides."""
import math
from datetime import date

import pytest

from execution.config import frozen
from execution.scan import Candidate, passes_hyp093, passes_hyp107, symbol_shape_ok

DAY = date(2026, 7, 16)
C107 = frozen("hyp107")
C093 = frozen("hyp093")


def cand107(gap, log_vol, prev_close=2.0, excluded=None):
    c = Candidate(symbol="TEST", day=DAY, prev_close=prev_close)
    c.overnight_gap = gap
    c.log_vol = log_vol
    c.open_0930 = prev_close * (1 + gap)
    c.vol_0930 = int(10 ** log_vol)
    c.excluded = excluded
    return c


def cand093(price, prev_close, vol, n_bars=10, last_et="10:25", excluded=None):
    from types import SimpleNamespace
    c = Candidate(symbol="TEST", day=DAY, prev_close=prev_close)
    c.price_1025 = price
    c.cum_vol_1025 = vol
    c.gain_1025 = price / prev_close - 1
    c.excluded = excluded
    hh, mm = last_et.split(":")
    c._window_bars = [{"t": f"2026-07-16T{int(hh) + 4:02d}:{mm}:00Z"}] * n_bars
    return c


# ── HYP-107 ───────────────────────────────────────────────────────────────────

def test_hyp107_passes_inside_band():
    ok, why = passes_hyp107(cand107(0.40, 5.0))
    assert ok and why == "pass"


@pytest.mark.parametrize("gap,expect_ok", [
    (0.2999, False),   # below gap_floor 0.30
    (0.3000, True),    # exactly at floor
    (0.5770, True),    # exactly at og_max
    (0.5771, False),   # above og_max
])
def test_hyp107_gap_boundaries(gap, expect_ok):
    ok, _ = passes_hyp107(cand107(gap, 5.0))
    assert ok is expect_ok


@pytest.mark.parametrize("lv,expect_ok", [
    (5.8540, True),    # exactly at logvol_max
    (5.8541, False),   # above
])
def test_hyp107_volume_boundary(lv, expect_ok):
    ok, _ = passes_hyp107(cand107(0.40, lv))
    assert ok is expect_ok


def test_hyp107_rejects_missing_bar_with_reason():
    c = Candidate(symbol="TEST", day=DAY, prev_close=2.0)
    ok, why = passes_hyp107(c)
    assert not ok and why == "missing_0930_bar"


def test_hyp107_respects_exclusion():
    ok, why = passes_hyp107(cand107(0.40, 5.0, excluded="mna_headline"))
    assert not ok and why == "mna_headline"


def test_hyp107_reasons_are_specific():
    _, why = passes_hyp107(cand107(0.80, 5.0))
    assert "og_max" in why


# ── HYP-093 ───────────────────────────────────────────────────────────────────

def test_hyp093_passes_valid_fade_setup():
    ok, why = passes_hyp093(cand093(price=3.0, prev_close=2.0, vol=600_000))
    assert ok and why == "pass"


def test_hyp093_gain_floor_is_50_percent_not_100():
    """Guards against the brief's ">= 100% above prior close" misstatement.

    gain 0.60 with price >= 1.30x prev_close is a PASS under the sealed prereg.
    A 100% floor would reject it and change the event set entirely.
    """
    c = cand093(price=3.20, prev_close=2.0, vol=600_000)   # gain = 0.60
    assert c.gain_1025 == pytest.approx(0.60)
    ok, _ = passes_hyp093(c)
    assert ok is True


@pytest.mark.parametrize("price,prev,expect_ok", [
    (3.00, 2.00, True),    # gain 0.50 exactly at gain_min
    (2.99, 2.00, False),   # gain 0.495 below gain_min
])
def test_hyp093_gain_boundary(price, prev, expect_ok):
    ok, _ = passes_hyp093(cand093(price=price, prev_close=prev, vol=600_000))
    assert ok is expect_ok


def test_hyp093_price_min_binds():
    # gain fine, but price under $2.00
    ok, why = passes_hyp093(cand093(price=1.95, prev_close=1.0, vol=600_000))
    assert not ok and "price_min" in why


@pytest.mark.parametrize("vol,expect_ok", [(500_000, True), (499_999, False)])
def test_hyp093_volume_boundary(vol, expect_ok):
    ok, _ = passes_hyp093(cand093(price=3.0, prev_close=2.0, vol=vol))
    assert ok is expect_ok


def test_hyp093_requires_enough_bars():
    ok, why = passes_hyp093(cand093(price=3.0, prev_close=2.0,
                                    vol=600_000, n_bars=7))
    assert not ok and "too_few_bars" in why


def test_hyp093_rejects_missing_window():
    c = Candidate(symbol="TEST", day=DAY, prev_close=2.0)
    ok, why = passes_hyp093(c)
    assert not ok and why == "missing_measure_window"


# ── Symbol shape ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sym,ok", [
    ("AAPL", True), ("TGHL", True), ("ABC", True),
    ("ABCDW", False),    # warrant
    ("ABCDR", False),    # right
    ("ABCDU", False),    # unit
    ("ABCDE", True),     # 5 letters, benign last char
    ("ABCDEF", False),   # too long
    ("BRK.B", False),    # non-alpha
])
def test_symbol_shape(sym, ok):
    assert symbol_shape_ok(sym) is ok
