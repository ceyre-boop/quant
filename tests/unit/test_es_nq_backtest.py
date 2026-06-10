"""Backtester unit tests — exact cost math, conservative fills, end-to-end synthetic."""
import numpy as np
import pandas as pd
import pytest

from sovereign.es_nq.backtest import (
    daily_atr, run_backtest, simulate_bracket, simulate_entry_at,
    structure_trades_for_session,
)
from sovereign.es_nq.structure_gate import Levels


def bars(rows, date="2022-03-08", start="09:30"):
    """rows = list of (open, high, low, close, volume)."""
    idx = pd.date_range(f"{date} {start}", periods=len(rows), freq="5min",
                        tz="America/New_York").tz_convert("UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=idx)


def test_full_winner_exact_r():
    """LONG 100, stop 99 (1pt), T1 101.5, T2 102.5; clean run-up, MNQ."""
    b = bars([
        (100.0, 100.5, 99.8, 100.4, 1000),
        (100.4, 101.6, 100.3, 101.5, 1000),   # T1 touched
        (101.5, 102.6, 101.4, 102.5, 1000),   # T2 touched
        (102.5, 102.8, 102.3, 102.6, 1000),
    ])
    r = simulate_bracket(b, 0, "LONG", 100.0, 99.0, 101.5, 102.5, "MNQ")
    # halves exit 101.4375 / 102.4375; pts=1.9375; $3.875−$0.70=$3.175; risk $2
    assert r["r_net"] == pytest.approx(1.5875)
    assert r["usd_net_per_contract"] == pytest.approx(3.175)
    assert r["exit_reasons"] == ["T1", "T2"]


def test_full_loser_exact_r():
    b = bars([
        (100.0, 100.2, 98.5, 98.8, 1000),     # straight to stop
        (98.8, 99.0, 98.5, 98.9, 1000),
    ])
    r = simulate_bracket(b, 0, "LONG", 100.0, 99.0, 101.5, 102.5, "MNQ")
    # both halves exit 99 − 0.125 = 98.875; pts −1.125 → $−2.25 − 0.70 = −2.95
    assert r["r_net"] == pytest.approx(-1.475)
    assert r["exit_reasons"] == ["STOP", "STOP"]


def test_stop_and_target_same_bar_is_stop_first():
    b = bars([
        (100.0, 102.0, 98.5, 101.0, 1000),    # touches stop AND T1 — conservative: stop
        (101.0, 101.5, 100.5, 101.2, 1000),
    ])
    r = simulate_bracket(b, 0, "LONG", 100.0, 99.0, 101.5, 102.5, "MNQ")
    assert r["exit_reasons"] == ["STOP", "STOP"]


def test_t1_then_breakeven():
    b = bars([
        (100.0, 101.6, 99.8, 101.4, 1000),    # T1 hit; runner stop → 100
        (101.4, 101.5, 99.9, 100.0, 1000),    # runner stopped at breakeven
    ])
    r = simulate_bracket(b, 0, "LONG", 100.0, 99.0, 101.5, 102.5, "MNQ")
    assert r["exit_reasons"] == ["T1", "BREAKEVEN"]
    # half A: 101.4375−100 = +1.4375; half B: 99.875−100 = −0.125 → pts 0.65625
    assert r["usd_net_per_contract"] == pytest.approx(0.65625 * 2 - 0.70)


def test_forced_flat_at_1555():
    rows = [(100.0, 100.4, 99.8, 100.2, 1000)] * 76   # 09:30 + 75×5min = 15:45 last
    b = bars(rows)                                     # bars 09:30..15:45
    rows_late = rows + [(100.2, 100.4, 100.0, 100.3, 1000),   # 15:50
                        (100.3, 100.5, 100.1, 100.4, 1000)]   # 15:55
    b = bars(rows_late)
    r = simulate_bracket(b, 0, "LONG", 100.0, 99.0, 105.0, 106.0, "MNQ")
    assert r["exit_reasons"] == ["FLAT", "FLAT"]
    et = b.index.tz_convert("America/New_York")
    assert (et[r["exit_bar_idx"]].hour, et[r["exit_bar_idx"]].minute) == (15, 55)


def test_short_direction_mirrors():
    b = bars([
        (100.0, 100.2, 98.4, 98.5, 1000),     # T1 (98.5) touched for short
        (98.5, 98.6, 97.4, 97.5, 1000),       # T2 (97.5) touched
    ])
    r = simulate_bracket(b, 0, "SHORT", 100.0, 101.0, 98.5, 97.5, "MNQ")
    assert r["exit_reasons"] == ["T1", "T2"]
    assert r["r_net"] > 1.0


def test_simulate_entry_at_builds_same_machinery():
    b = bars([
        (100.0, 100.5, 99.8, 100.4, 1000),
        (100.4, 102.1, 100.3, 102.0, 1000),
        (102.0, 103.0, 101.9, 102.9, 1000),
    ])
    r = simulate_entry_at(b, 0, "UP", 1.0, "MNQ")
    assert r["entry"] == pytest.approx(100.0 + 0.0625)   # bar0 Open + 0.25 tick
    assert r["stop_points"] == pytest.approx(1.0)


def test_daily_atr_positive_and_lagged():
    n = 30
    rng = np.random.RandomState(5)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    daily = pd.DataFrame({
        "rth_high": close + 1, "rth_low": close - 1, "rth_close": close,
    }, index=[f"2022-01-{i+1:02d}" for i in range(n)])
    atr = daily_atr(daily)
    assert (atr.iloc[1:] > 0).all()


def _synthetic_universe():
    """Two sessions: day1 (prior), day2 with a clean PDL sweep+confirm for an UP bias."""
    day1 = bars([(105, 110, 100, 106, 1000)] * 78, date="2022-03-07")
    closes = [105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0, 104.5] + [104.5] * 70
    opens = [105.5, 105.0, 103.0, 99.5, 101.0, 102.8, 103.6, 104.0] + [104.5] * 70
    lows = [104.5, 102.5, 99.3, 99.4, 100.8, 102.5, 103.4, 103.8] + [104.0] * 70
    highs = [106.0, 105.5, 103.2, 101.2, 103.0, 104.2, 104.2, 104.8] + [105.0] * 70
    vols = [1000, 1000, 1500, 1200, 1000, 2500, 1000, 1000] + [1000] * 70
    day2 = bars(list(zip(opens, highs, lows, closes, vols)), date="2022-03-08")
    bars5_all = pd.concat([day1, day2])
    daily = pd.DataFrame({
        "rth_open": [105.0, 105.5], "rth_close": [106.0, 104.5],
        "rth_high": [110.0, 106.0], "rth_low": [100.0, 99.3],
        "onh": [107.0, 108.0], "onl": [103.0, 102.0],
        "px_0925": [105.0, 106.0], "prior_rth_close": [np.nan, 106.0],
        "overnight_ret": [0.0, 0.004], "roll_day": [False, False],
        "rth_bars": [78, 78], "symbol": ["NQH2", "NQH2"],
    }, index=["2022-03-07", "2022-03-08"])
    ft = pd.DataFrame({
        "direction": ["UP"], "confidence": [0.6], "raw_score": [0.45],
        "event_day": [False], "roll_day": [False],
        "direction_real": ["DOWN"], "move_pct": [-0.0095], "flat_secondary": [False],
        "s_overnight": [1.0], "s_vix": [0.0], "s_hurst": [0.0],
        "s_international": [0.0], "s_calendar": [0.0],
    }, index=["2022-03-08"])
    return daily, bars5_all, ft


def test_run_backtest_bias_gate_end_to_end():
    daily, bars5_all, ft = _synthetic_universe()
    sessions = run_backtest("2022-03-08", "2022-03-08", "bias_gate",
                            daily=daily, bars5_all=bars5_all, feature_table=ft,
                            instrument="MNQ")
    assert len(sessions) == 1
    s = sessions[0]
    assert s["skipped"] is None
    assert len(s["trades"]) >= 1
    t = s["trades"][0]
    assert t["direction"] == "LONG" and t["contracts"] >= 1
    assert s["bias_was_correct"] is False     # biased UP, session closed DOWN


def test_run_backtest_skips_neutral_and_roll():
    daily, bars5_all, ft = _synthetic_universe()
    ft_neutral = ft.copy()
    ft_neutral["direction"] = ["NEUTRAL"]
    s = run_backtest("2022-03-08", "2022-03-08", "bias_gate", daily=daily,
                     bars5_all=bars5_all, feature_table=ft_neutral, instrument="MNQ")
    assert s[0]["skipped"] == "NEUTRAL_BIAS"
    ft_roll = ft.copy()
    ft_roll["roll_day"] = [True]
    s = run_backtest("2022-03-08", "2022-03-08", "full", daily=daily,
                     bars5_all=bars5_all, feature_table=ft_roll, instrument="MNQ")
    assert s[0]["skipped"] == "ROLL_DAY"


def test_structure_trades_capped_at_max():
    daily, bars5_all, ft = _synthetic_universe()
    et_dates = bars5_all.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    day2 = bars5_all[et_dates == "2022-03-08"]
    levels = Levels(pdh=110.0, pdl=100.0, onh=108.0, onl=102.0)
    trades = structure_trades_for_session(day2, levels, "UP", "MNQ")
    assert len(trades) <= 3
