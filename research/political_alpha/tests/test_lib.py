"""Unit tests for the pure pieces of _lib.py: T0 mapping, smile math on a
synthetic chain, sigma60 trailing-only-ness, and the 5-day de-dup helper in
build_event_catalog. No network."""

import math

import numpy as np
import pandas as pd

import _lib
from build_event_catalog import dedup_min_separation


def _weekday_index(start: str, days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=days)


# ── map_t0 ───────────────────────────────────────────────────────────────────────────

def test_map_t0_fx_weekday_same_day():
    idx = _weekday_index("2025-03-03", 10)
    t0 = _lib.map_t0("2025-03-04T14:30:00Z", idx, "fx")
    assert t0 == pd.Timestamp("2025-03-04")


def test_map_t0_fx_weekend_rolls_to_monday():
    idx = _weekday_index("2025-03-03", 10)
    t0 = _lib.map_t0("2025-03-08T18:00:00Z", idx, "fx")   # Saturday
    assert t0 == pd.Timestamp("2025-03-10")               # Monday


def test_map_t0_etf_before_close_same_session():
    idx = _weekday_index("2025-03-03", 10)
    # 15:00 ET on Tue 2025-03-04 == 20:00 UTC (EST)
    t0 = _lib.map_t0("2025-03-04T20:00:00+00:00", idx, "us_etf")
    assert t0 == pd.Timestamp("2025-03-04")


def test_map_t0_etf_after_close_next_session():
    idx = _weekday_index("2025-03-03", 10)
    # 21:30 UTC = 16:30 ET (EST) -> after the close -> next session
    t0 = _lib.map_t0("2025-03-04T21:30:00+00:00", idx, "us_etf")
    assert t0 == pd.Timestamp("2025-03-05")


def test_map_t0_beyond_data_returns_none():
    idx = _weekday_index("2025-03-03", 5)
    assert _lib.map_t0("2026-01-01T00:00:00Z", idx, "fx") is None


# ── sigma60 trailing-only ────────────────────────────────────────────────────────────

def test_trailing_sigma60_excludes_current_day():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 200))
    sig = _lib.trailing_sigma60(r)
    # value at position i must equal std of r[i-60:i] — never including r[i]
    i = 150
    expected = r.iloc[i - 60:i].std(ddof=1)
    assert math.isclose(sig.iloc[i], expected, rel_tol=1e-12)


# ── smile math on a synthetic Black-76 chain ─────────────────────────────────────────

def test_smile_read_recovers_flat_vol():
    """Price a flat-vol chain with the module's own _bs76_call, then check
    smile_read recovers that vol and rr25 ~ 0 (symmetric smile)."""
    F = spot = 100.0
    r, dte, sigma = _lib.R_FLAT, 30, 0.20
    T = dte / 365.0
    rows = []
    for K in range(80, 121, 2):
        call = _lib._bs76_call(F, K, T, r, sigma)
        # put via parity: P = C - e^{-rT}(F-K)
        put = call - math.exp(-r * T) * (F - K)
        rows.append({"strike": float(K), "call_mid": call, "put_mid": put,
                     "call_volume": 10.0, "put_volume": 10.0})
    chain = pd.DataFrame(rows)
    read = _lib.smile_read(chain, spot, dte, r, _lib.MIN_STRIKES)
    assert read is not None
    assert abs(read["atm_iv"] - sigma) < 0.005
    assert read["rr25"] is not None and abs(read["rr25"]) < 0.005


def test_smile_read_thin_chain_returns_none():
    chain = pd.DataFrame([{"strike": 100.0, "call_mid": 1.0, "put_mid": 1.0,
                           "call_volume": 0.0, "put_volume": 0.0}])
    assert _lib.smile_read(chain, 100.0, 30, _lib.R_FLAT, _lib.MIN_STRIKES) is None


# ── 5-trading-day per-instrument de-dup ──────────────────────────────────────────────

def test_dedup_keeps_first_drops_within_5_days():
    idx = _weekday_index("2025-02-03", 30)
    rows = [
        {"event_id": "PA-0001", "instrument_tagged": "SLX", "t0": pd.Timestamp("2025-02-05")},
        {"event_id": "PA-0002", "instrument_tagged": "SLX", "t0": pd.Timestamp("2025-02-10")},  # 3 td later -> drop
        {"event_id": "PA-0003", "instrument_tagged": "SLX", "t0": pd.Timestamp("2025-02-14")},  # 7 td after 0001 -> keep
        {"event_id": "PA-0002", "instrument_tagged": "KWEB", "t0": pd.Timestamp("2025-02-10")}, # other instrument -> keep
    ]
    kept, dropped = dedup_min_separation(rows, {"SLX": idx, "KWEB": idx}, min_sep=5)
    kept_ids = {(r["event_id"], r["instrument_tagged"]) for r in kept}
    assert ("PA-0001", "SLX") in kept_ids
    assert ("PA-0002", "SLX") not in kept_ids
    assert ("PA-0003", "SLX") in kept_ids
    assert ("PA-0002", "KWEB") in kept_ids
    assert len(dropped) == 1 and dropped[0]["kept_event_id"] == "PA-0001"
