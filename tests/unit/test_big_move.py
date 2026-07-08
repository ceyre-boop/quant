"""Big-Move-of-the-Day classifier tests.

Two jobs: (1) confirm the feature LOGIC behaves as designed (compression raises
P(big), direction tracks momentum, context raises confidence, events raise
magnitude), and (2) LOCK the discipline — insufficient data degrades to a
low-confidence NEUTRAL estimate rather than crashing or inventing a signal.

These test SHAPE/MONOTONICITY, not magnitude — the actual edge (if any) is the
job of scripts/validate_big_move.py, never of unit asserts.
"""
import numpy as np
import pandas as pd

from sovereign.intelligence import big_move as bm


def make_daily(n=80, base=1.30, daily_range=0.010, drift=0.0,
               recent_range=None, recent_k=5):
    """Build a synthetic daily OHLCV frame with controllable volatility/drift."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = base + drift * np.arange(n)
    rows = []
    for i in range(n):
        c = float(closes[i])
        o = float(closes[i - 1]) if i > 0 else c
        rng = recent_range if (recent_range is not None and i >= n - recent_k) else daily_range
        hi = max(o, c) + rng / 2
        lo = min(o, c) - rng / 2
        rows.append((o, hi, lo, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


FULL_CTX = {"rate_diff_z": 1.0, "vix": 18.0, "cot_percentile": 0.5,
            "high_impact_event_today": False}


# ── (2) discipline: graceful degradation ──────────────────────────────────────
def test_insufficient_data_returns_neutral_zero_confidence():
    df = make_daily(n=5)
    est = bm.estimate_big_move("GBPUSD", df)
    assert est.direction == "NEUTRAL"
    assert est.p_big == 0.0
    assert est.confidence == 0.0
    assert any("insufficient" in note for note in est.notes)


def test_none_frame_does_not_crash():
    est = bm.estimate_big_move("GBPUSD", None)
    assert est.confidence == 0.0 and est.direction == "NEUTRAL"


def test_deterministic_same_input_same_output():
    from datetime import datetime, timezone
    df = make_daily(n=80, drift=0.004)
    fixed = datetime(2026, 6, 13, tzinfo=timezone.utc)
    a = bm.estimate_big_move("EURUSD", df, FULL_CTX, now=fixed).to_dict()
    b = bm.estimate_big_move("EURUSD", df, FULL_CTX, now=fixed).to_dict()
    assert a == b


# ── (1) feature logic ─────────────────────────────────────────────────────────
def test_compression_raises_p_big():
    # Compressed: recent ranges far below baseline -> coiled spring -> higher P(big).
    compressed = make_daily(n=80, daily_range=0.020, recent_range=0.004)
    normal = make_daily(n=80, daily_range=0.020, recent_range=0.020)
    p_comp = bm.estimate_big_move("GBPUSD", compressed).p_big
    p_norm = bm.estimate_big_move("GBPUSD", normal).p_big
    assert p_comp > p_norm


def test_direction_follows_momentum():
    up = bm.estimate_big_move("GBPUSD", make_daily(n=80, drift=0.005))
    down = bm.estimate_big_move("GBPUSD", make_daily(n=80, drift=-0.005))
    assert up.direction == "LONG"
    assert down.direction == "SHORT"


def test_flat_market_is_neutral():
    est = bm.estimate_big_move("GBPUSD", make_daily(n=80, drift=0.0))
    assert est.direction == "NEUTRAL"


def test_context_raises_confidence():
    df = make_daily(n=80, drift=0.004)
    with_ctx = bm.estimate_big_move("GBPUSD", df, FULL_CTX).confidence
    no_ctx = bm.estimate_big_move("GBPUSD", df, None).confidence
    assert with_ctx > no_ctx


def test_event_raises_magnitude_and_p_big():
    df = make_daily(n=80, drift=0.004)
    ev = bm.estimate_big_move("GBPUSD", df, {**FULL_CTX, "high_impact_event_today": True})
    no_ev = bm.estimate_big_move("GBPUSD", df, {**FULL_CTX, "high_impact_event_today": False})
    assert ev.p_big > no_ev.p_big
    assert ev.expected_vs_adr >= no_ev.expected_vs_adr


def test_p_big_in_unit_interval_and_drivers_sorted():
    est = bm.estimate_big_move("GBPUSD", make_daily(n=80, drift=0.004), FULL_CTX)
    assert 0.0 <= est.p_big <= 1.0
    mags = [abs(v) for _, v in est.drivers]
    assert mags == sorted(mags, reverse=True)


def test_no_context_note_present():
    est = bm.estimate_big_move("GBPUSD", make_daily(n=80), None)
    assert any("no macro context" in note for note in est.notes)


# ── label helper (validation ground truth) ────────────────────────────────────
def test_big_move_label_flags_outsized_day():
    df = make_daily(n=80, daily_range=0.008)
    # Inflate one day's range far beyond the trailing distribution.
    big_idx = 75
    df.iloc[big_idx, df.columns.get_loc("High")] = df.iloc[big_idx]["Close"] + 0.05
    df.iloc[big_idx, df.columns.get_loc("Low")] = df.iloc[big_idx]["Open"] - 0.05
    label = bm.big_move_label(df, big_idx)
    assert label is not None
    is_big, _ = label
    assert is_big is True


def test_big_move_label_none_when_insufficient_trailing():
    df = make_daily(n=80)
    assert bm.big_move_label(df, 10) is None  # idx < percentile lookback
