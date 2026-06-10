"""Structure gate unit tests — sweep/reclaim boundaries, confirmation, trade plan."""
import numpy as np
import pandas as pd
import pytest

from sovereign.es_nq.structure_gate import (
    Levels, detect_confirmation, detect_sweep, plan_trade, session_levels,
    session_vwap,
)


def make_session(closes, lows=None, highs=None, volumes=None, opens=None,
                 date="2022-03-08"):
    """5-min RTH session from close prices; OHLC derived unless given."""
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else \
        np.r_[closes[0], closes[:-1]]
    highs = np.asarray(highs, dtype=float) if highs is not None else \
        np.maximum(opens, closes) + 0.5
    lows = np.asarray(lows, dtype=float) if lows is not None else \
        np.minimum(opens, closes) - 0.5
    volumes = np.asarray(volumes, dtype=float) if volumes is not None else \
        np.full(n, 1000.0)
    idx = pd.date_range(f"{date} 09:30", periods=n, freq="5min",
                        tz="America/New_York").tz_convert("UTC")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": volumes}, index=idx)


LEVELS = Levels(pdh=110.0, pdl=100.0, onh=108.0, onl=102.0)


def test_sweep_detected_at_exact_threshold():
    # PDL=100; sweep needs Low < 100*(1-0.001) = 99.9
    closes = [105, 104, 101, 100.5, 100.6, 101.5, 102, 103]
    lows = [104, 103, 100.5, 99.85, 100.1, 101, 101.5, 102]  # bar3 Low 99.85 < 99.9
    bars = make_session(closes, lows=lows)
    sw = detect_sweep(bars, LEVELS, "UP")
    assert sw is not None
    assert sw.level_name in ("PDL", "ONL")
    assert sw.sweep_bar_idx == 3
    assert sw.reclaim_bar_idx <= 6


def test_no_sweep_when_excess_too_small():
    closes = [105, 104, 101, 100.5, 101.5, 102, 103, 104]
    lows = [104, 103, 100.5, 99.95, 101, 101.5, 102, 103]   # 99.95 > 99.9 → no sweep
    bars = make_session(closes, lows=lows)
    assert detect_sweep(bars, LEVELS, "UP") is None


def test_no_reclaim_within_three_bars_kills_sweep():
    # Sweep at bar 2, but closes stay below PDL for 4+ bars
    closes = [105, 101, 99.5, 99.6, 99.4, 99.7, 99.8, 100.5]
    lows = [c - 0.5 for c in closes]
    bars = make_session(closes, lows=lows)
    sw = detect_sweep(bars, LEVELS, "UP")
    assert sw is None    # first sweep failed to reclaim in 3 → dead (no second chance on that level)


def test_sweep_high_for_down_bias():
    closes = [106, 107, 109.5, 110.5, 109.0, 108.0, 107.0, 106.0]
    highs = [107, 108, 110.0, 110.2, 109.5, 108.5, 107.5, 106.5]
    # need High > 110*1.001=110.11 → bar3 High 110.2
    bars = make_session(closes, highs=highs)
    sw = detect_sweep(bars, LEVELS, "DOWN")
    assert sw is not None and sw.side == "high" and sw.level_name == "PDH"
    assert sw.extreme >= 110.2


def test_neutral_bias_no_sweep():
    bars = make_session([105, 99, 101, 102])
    assert detect_sweep(bars, LEVELS, "NEUTRAL") is None


def test_confirmation_requires_vwap_touch_direction_and_volume():
    # Construct: sweep of PDL at bar2, reclaim bar3, then price returns to VWAP
    # and bar5 closes up on 2x volume.
    closes = [105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0, 104.5]
    opens  = [105.5, 105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0]
    lows   = [104.5, 102.5, 99.3, 99.4, 100.8, 102.5, 103.4, 103.8]
    highs  = [106.0, 105.5, 103.2, 101.2, 103.0, 104.2, 104.2, 104.8]
    vols   = [1000, 1000, 1500, 1200, 1000, 2500, 1000, 1000]
    bars = make_session(closes, lows=lows, highs=highs, volumes=vols, opens=opens)
    sw = detect_sweep(bars, LEVELS, "UP")
    assert sw is not None and sw.reclaim_bar_idx == 3
    ci = detect_confirmation(bars, sw, "UP")
    assert ci is not None
    vwap = session_vwap(bars)
    # the confirm bar (or an earlier post-reclaim bar) touched vwap
    assert ci > sw.reclaim_bar_idx
    bar = bars.iloc[ci]
    assert bar["Close"] > bar["Open"]


def test_no_confirmation_without_volume():
    closes = [105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0, 104.5]
    opens  = [105.5, 105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0]
    lows   = [104.5, 102.5, 99.3, 99.4, 100.8, 102.5, 103.4, 103.8]
    highs  = [106.0, 105.5, 103.2, 101.2, 103.0, 104.2, 104.2, 104.8]
    bars = make_session(closes, lows=lows, highs=highs,
                        volumes=[1000] * 8, opens=opens)   # flat volume — never >1.2×
    sw = detect_sweep(bars, LEVELS, "UP")
    assert detect_confirmation(bars, sw, "UP") is None


def test_no_entry_after_deadline():
    """Confirmation completing after 12:00 ET must be rejected."""
    n = 35   # bar 30 = 12:00 ET
    closes = np.full(n, 105.0)
    closes[2] = 99.5            # sweep
    closes[3] = 101.0           # reclaim
    closes[32] = 106.0          # would-be confirm bar at 12:10
    lows = closes - 0.6
    highs = closes + 0.6
    vols = np.full(n, 1000.0)
    vols[32] = 3000.0
    opens = np.r_[closes[0], closes[:-1]]
    opens[32] = 104.0           # up close
    bars = make_session(closes, lows=lows, highs=highs, volumes=vols, opens=opens)
    sw = detect_sweep(bars, LEVELS, "UP")
    assert sw is not None
    assert detect_confirmation(bars, sw, "UP") is None


def test_plan_trade_geometry():
    closes = [105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0, 104.5]
    opens  = [105.5, 105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0]
    lows   = [104.5, 102.5, 99.3, 99.4, 100.8, 102.5, 103.4, 103.8]
    highs  = [106.0, 105.5, 103.2, 101.2, 103.0, 104.2, 104.2, 104.8]
    vols   = [1000, 1000, 1500, 1200, 1000, 2500, 1000, 1000]
    bars = make_session(closes, lows=lows, highs=highs, volumes=vols, opens=opens)
    sw = detect_sweep(bars, LEVELS, "UP")
    ci = detect_confirmation(bars, sw, "UP")
    plan = plan_trade(bars, ci, sw, "UP", "MNQ")
    assert plan is not None and plan.direction == "LONG"
    # entry = next bar open + 0.25 tick (0.0625 pts on MNQ)
    assert plan.entry == pytest.approx(float(bars["Open"].iloc[ci + 1]) + 0.0625)
    # stop = sweep extreme (99.3) − 2 ticks (0.5)
    assert plan.stop == pytest.approx(sw.extreme - 0.5)
    d = plan.entry - plan.stop
    assert plan.t1 == pytest.approx(plan.entry + 1.5 * d)
    assert plan.t2 == pytest.approx(plan.entry + 2.5 * d)


def test_session_levels_mapping():
    prior = pd.Series({"rth_high": 110.0, "rth_low": 100.0})
    cur = pd.Series({"onh": 108.0, "onl": 102.0})
    lv = session_levels(prior, cur)
    assert (lv.pdh, lv.pdl, lv.onh, lv.onl) == (110.0, 100.0, 108.0, 102.0)
