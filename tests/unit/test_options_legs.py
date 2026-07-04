"""Offline unit tests for scripts/research/positioning_options_legs.py (synthetic data only)."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research import positioning_options_legs as ol  # noqa: E402
from sovereign.research.positioning import event_study as es  # noqa: E402

IDX = pd.bdate_range("2021-01-04", periods=300)


def series(vals, idx=None):
    idx = idx if idx is not None else IDX[:len(vals)]
    return pd.Series(list(vals), index=idx, dtype=float)


def four(pair, s, fill=None):
    """Dict over OPT_PAIRS with `s` at `pair` and empty/filled elsewhere."""
    out = {}
    for p in ol.OPT_PAIRS:
        out[p] = s if p == pair else (fill if fill is not None else series([]))
    return out


# ── z construction ───────────────────────────────────────────────────────────

def test_z_trailing_strict_warmup_and_invariance():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=400), index=pd.bdate_range("2020-01-01", periods=400))
    z = ol.z_trailing(s)
    assert z.iloc[:251].isna().all() and np.isfinite(z.iloc[251])
    ol.assert_truncation_invariant(s)  # must not raise


def test_truncation_invariance_catches_centered_z():
    with pytest.raises(SystemExit):
        real = ol.z_trailing
        try:
            ol.z_trailing = lambda s, window=252: (s - s.mean()) / s.std(ddof=1)  # centered = look-ahead
            rng = np.random.default_rng(1)
            ol.assert_truncation_invariant(pd.Series(rng.normal(size=400),
                                                     index=pd.bdate_range("2020-01-01", periods=400)))
        finally:
            ol.z_trailing = real


# ── HYP-074 ──────────────────────────────────────────────────────────────────

def test_074_fade_direction_and_flip():
    z = series([0.0] * 10 + [2.5] + [1.5] * 5 + [0.5] + [2.6] + [0.0] * 10)
    for pair, want in (("EURUSD", -1), ("USDJPY", +1)):  # fade -> −1; USDJPY flips
        evs = ol.events_074(four(pair, z))[pair]
        assert len(evs) == 2                              # hysteresis: re-arm at 0.5 allows 2nd
        assert all(e.side == want for e in evs)
        assert evs[0].publish_date == ol.shift_pub(z.index[10].date())


def test_074_hysteresis_blocks_repeat_without_rearm():
    z = series([0.0] * 5 + [2.5, 2.7, 2.9] + [1.2] * 5 + [2.4] + [0.0] * 5)
    evs = ol.events_074(four("EURUSD", z))["EURUSD"]
    assert len(evs) == 1  # never re-entered |z|<=1 band


# ── HYP-075 ──────────────────────────────────────────────────────────────────

def _spot_with_high(n=100, hi_at=80):
    vals = [1.0 + 0.0001 * i for i in range(n)]
    vals[hi_at] = max(vals) + 0.05
    return series(vals)


def test_075_unconfirmed_high_fires_reversal_short():
    c = _spot_with_high()
    rng = np.random.default_rng(7)
    rr = series(list(0.01 * rng.normal(size=100)))
    rr.iloc[70] = 1.0                       # rr's 60d max sits INSIDE the window, before the high
    rr.iloc[80] = 0.0                       # ...and rr is well below it on the spot-high day
    evs = ol.events_075(four("EURUSD", c, fill=c), four("EURUSD", rr, fill=rr))["EURUSD"]
    assert any(e.side == -1 and e.publish_date == ol.shift_pub(c.index[80].date()) for e in evs)


def test_075_confirmed_high_does_not_fire():
    c = _spot_with_high()
    rr = series(list(np.linspace(0, 2.0, 100)))   # rr at its own 60d max on the high day
    evs = ol.events_075(four("EURUSD", c, fill=c), four("EURUSD", rr, fill=rr))["EURUSD"]
    assert not any(e.publish_date == ol.shift_pub(c.index[80].date()) and e.side == -1 for e in evs)


# ── HYP-078 / 079 ────────────────────────────────────────────────────────────

def test_078_onset_and_slope_rearm():
    slope = series([-0.1] * 5 + [0.2] * 10 + [-0.2] * 3 + [0.3] * 5)
    pct = series([0.95] * 23)
    evs = ol.events_078(four("EURUSD", slope, fill=series([])), four("EURUSD", pct, fill=series([])))["EURUSD"]
    assert len(evs) == 2                       # second onset only after slope<0 re-arm
    assert evs[0].crowd_pair_side == 1
    evs_j = ol.events_078(four("USDJPY", slope, fill=series([])), four("USDJPY", pct, fill=series([])))["USDJPY"]
    assert evs_j[0].crowd_pair_side == -1      # JPY-space crowd-long -> pair-space short


def test_079_spike_gate_and_rearm():
    z = series([0.0] * 5 + [2.4] + [1.5] * 4 + [0.5] + [2.2] + [0.0] * 5)
    pct = series([0.05] * 17)
    evs = ol.events_079(four("EURUSD", z, fill=series([])), four("EURUSD", pct, fill=series([])))["EURUSD"]
    assert len(evs) == 2 and evs[0].crowd_pair_side == -1


# ── HYP-076 ──────────────────────────────────────────────────────────────────

def test_076_cell_classification_and_deoverlap():
    pubs = [date(2023, 1, 10), date(2023, 1, 10), date(2023, 2, 15), date(2023, 3, 20)]
    rels = pd.DataFrame({"publish_date": pubs,
                         "surprise_z": [1.6, -2.5, 1.8, 1.7],
                         "usd_sign": [1.0, 1.0, 1.0, 1.0]})
    idx = pd.to_datetime(["2023-01-02"])
    crowded_long_base = {p: pd.Series([0.95], index=idx) for p in ol.OPT_PAIRS}
    evs = ol.events_076(rels, crowded_long_base)
    jan = [e for e in evs if e.publish_date == date(2023, 1, 10)]
    # de-overlap: only the max-|z| release (−2.5 → USD-negative) survives Jan 10
    assert all(e.side == +1 for e in jan if e.pair != "USDJPY")      # USD down -> EURUSD up
    eur = {e.publish_date: e for e in evs if e.pair == "EURUSD"}
    # Feb 15: USD-positive surprise -> EURUSD side −1; crowd long EUR (pair +1) == −side -> CROWDED
    assert eur[date(2023, 2, 15)].crowded is True
    # Jan 10 (max-|z| is USD-negative): side +1; crowd +1 == side -> aligned -> excluded
    assert date(2023, 1, 10) not in eur
    jpy = {e.publish_date: e for e in evs if e.pair == "USDJPY"}
    # USDJPY: crowd long JPY (0.95, flip -1 -> pair-space −1); Feb 15 USD-positive side +1 -> CROWDED
    assert jpy[date(2023, 2, 15)].crowded is True


def test_076_uncrowded_is_control():
    rels = pd.DataFrame({"publish_date": [date(2023, 5, 5)], "surprise_z": [2.0], "usd_sign": [1.0]})
    neutral = {p: pd.Series([0.50], index=pd.to_datetime(["2023-01-02"])) for p in ol.OPT_PAIRS}
    evs = ol.events_076(rels, neutral)
    assert evs and all(e.crowded is False for e in evs)


# ── HYP-080 ──────────────────────────────────────────────────────────────────

def test_080_aligned_fades_opposed_follows():
    z = series([0.0] * 5 + [1.8] + [0.2] * 5 + [-1.7] + [0.0] * 5)
    pct = series([0.85] * 17)  # crowd long base (>=0.80)
    evs = ol.events_080(four("EURUSD", z, fill=series([])), four("EURUSD", pct, fill=series([])))["EURUSD"]
    assert len(evs) == 2
    assert evs[0].side == -1   # tone +, crowd + -> ALIGNED -> fade -> −1
    assert evs[1].side == -1   # tone −, crowd + -> OPPOSED -> follow tone -> −1


# ── HYP-077 full composite ───────────────────────────────────────────────────

def test_077_full_composite_phi_alignment():
    cot = pd.DataFrame({"pair": ["EURUSD", "EURUSD"],
                        "publish_date": [date(2023, 1, 6), date(2023, 1, 13)],
                        "net_pct_1y": [0.95, 0.95], "flush_1w": [0.0, 0.0],
                        "measurement_date": [date(2023, 1, 3), date(2023, 1, 10)]})
    rr = {p: pd.Series([2.0, 2.0], index=pd.to_datetime(["2023-01-05", "2023-01-12"]))
          for p in ol.OPT_PAIRS}
    trades = "SENTINEL"
    dts, vals, mapping = ol.crowding_composite_full(
        cot, rr, trades, ["EURUSD"], lambda t, d: {"EURUSD": 1})
    from scipy.stats import norm
    want = (0.95 + float(norm.cdf(2.0))) / 2.0
    assert dts and abs(vals[0] - want) < 1e-9
    assert mapping[0]["pairs"]["EURUSD"]["aligned_rr"] == pytest.approx(float(norm.cdf(2.0)))


# ── stats helpers ────────────────────────────────────────────────────────────

def test_range_ratio_and_abs_move():
    idx = pd.bdate_range("2022-01-03", periods=100)
    hi = pd.Series(1.01, index=idx); lo = pd.Series(1.00, index=idx)
    hi.iloc[70:80] = 1.03                       # doubled range in the fwd window
    r = ol.range_ratio(hi, lo, idx[69].date(), h=10, base_lb=60)
    assert r == pytest.approx(3.0, abs=1e-6)    # (10*0.03)/(10*0.01)
    c = pd.Series(np.linspace(1.0, 1.2, 100), index=idx)
    assert ol.abs_move(c, idx[10].date(), h=10) > 0


def test_perm_p_stat_uniform_null():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=200)
    stat = {p: pd.Series(rng.normal(size=200), index=idx) for p in ol.OPT_PAIRS}
    dates_ = {p: [idx[50].date(), idx[120].date()] for p in ol.OPT_PAIRS}
    res = ol.perm_p_stat(stat, dates_, rng, n_perm=200)
    assert res.n_events == 8 and 0.0 < res.p <= 1.0
