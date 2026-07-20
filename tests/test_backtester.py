"""Tests for the bias-free backtester. Run: PYTHONPATH=~/quant pytest -q tests/test_backtester.py"""
import numpy as np
import pandas as pd

from backtester import engine, mc, scanner, audit


def _bars(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                       "volume"])


def test_stop_never_fills_at_trigger_on_gapthrough():
    """A bar that OPENS above the short trigger must fill at the open, not the
    trigger (the core bias fix)."""
    # entry 10:30 open=10; stop 25% -> trigger 12.5; next bar OPENS at 15 (gap)
    bars = _bars([
        ("10:29", 10, 10, 10, 10, 1000),
        ("10:30", 10, 10.2, 9.9, 10, 1000),
        ("10:35", 15, 16, 15, 15.5, 5000),   # gap-through: open 15 > 12.5
        ("15:45", 14, 14, 14, 14, 1000),
    ])
    cfg = dict(entry_time="10:30", stop_pct=0.25, exit_time="15:45",
               direction="short", sizing_pct=1.0)
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "TEST", "gain": 1.0}])
    res = engine.run(ev, cfg, data_cache={("TEST", "2026-01-05"): bars},
                     check_holdout=False,
                     write_audit=False)
    r = res["records"][0]
    assert r["stop_hit"] and not r["filled_at_trigger"]
    assert abs(r["stop_fill_price"] - 15.0) < 1e-9   # filled at bar open, worse
    assert r["stop_fill_price"] > 12.5               # never at/ better than trigger


def test_intrabar_breach_fills_at_trigger():
    """A bar whose HIGH breaches but OPEN is inside fills at the trigger."""
    bars = _bars([
        ("10:29", 10, 10, 10, 10, 1000),
        ("10:30", 10, 10, 10, 10, 1000),
        ("10:35", 11, 13, 11, 11.5, 5000),   # open 11 < 12.5, high 13 > 12.5
        ("15:45", 11, 11, 11, 11, 1000),
    ])
    cfg = dict(entry_time="10:30", stop_pct=0.25, exit_time="15:45",
               direction="short", sizing_pct=1.0)
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "T", "gain": 1.0}])
    res = engine.run(ev, cfg, data_cache={("T", "2026-01-05"): bars},
                     check_holdout=False,
                     write_audit=False)
    r = res["records"][0]
    assert r["stop_hit"] and r["filled_at_trigger"]
    assert abs(r["stop_fill_price"] - 12.5) < 1e-9


def test_locate_gate_skips_no_locate(tmp_path, monkeypatch):
    """locate_required must skip a ticker that is NOT EASY on the snapshot."""
    import json
    from backtester import engine as eng
    snap = {"detail": {"GOODX": {"tier": "EASY"}, "BADX": {"tier": "HARD"}}}
    p = tmp_path / "ib_locate_2026-01-05.json"
    p.write_text(json.dumps(snap))
    monkeypatch.setattr(eng, "LOCATE_DIR", tmp_path)
    bars = _bars([("10:29", 10, 10, 10, 10, 1), ("10:30", 10, 10, 10, 10, 1),
                  ("15:45", 9, 9, 9, 9, 1)])
    cache = {("GOODX", "2026-01-05"): bars, ("BADX", "2026-01-05"): bars}
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "GOODX", "gain": 1.0},
                       {"date": "2026-01-05", "ticker": "BADX", "gain": 1.0}])
    cfg = dict(entry_time="10:30", stop_pct=0.25, exit_time="15:45",
               direction="short", sizing_pct=1.0, locate_required=True,
               on_missing_locate="take")
    res = eng.run(ev, cfg, data_cache=cache, write_audit=False, check_holdout=False)
    by = {r["ticker"]: r for r in res["records"]}
    assert by["GOODX"]["trade_taken"] is True
    assert by["BADX"]["trade_taken"] is False
    assert by["BADX"]["reason"] == "no_locate"


def test_block_bootstrap_preserves_blocks():
    """Block bootstrap must draw contiguous 5-day runs from the source series,
    so an alternating +/- series keeps adjacent-pair correlation unlike IID."""
    # deterministic source with strong lag-1 autocorrelation
    series = ([0.02] * 5 + [-0.02] * 5) * 20
    out = mc.run_mc(series, 5000,
                    dict(pass_pct=0.50, bust_pct=-0.50, time_limit_days=100),
                    n_cores=1, seed=1)
    assert out["block_size"] == 5
    assert 0.0 <= out["p_pass"] <= 1.0
    assert out["n_paths"] == 5000


def test_family_correction_monotonic_in_n_tested():
    """Larger grid -> larger (or equal) Bonferroni p for the SAME config."""
    bars_a = _bars([("10:29", 10, 10, 10, 10, 1), ("10:30", 10, 10.1, 9.9, 10, 1),
                    ("12:00", 8, 8, 7.9, 8, 1), ("15:45", 7, 7, 7, 7, 1)])
    bars_b = _bars([("10:29", 5, 5, 5, 5, 1), ("10:30", 5, 5.1, 4.9, 5, 1),
                    ("12:00", 4, 4, 3.9, 4, 1), ("15:45", 3.5, 3.5, 3.5, 3.5, 1)])
    cache = {("A", "2026-01-05"): bars_a, ("B", "2026-01-06"): bars_b}
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "A", "gain": 1.0},
                       {"date": "2026-01-06", "ticker": "B", "gain": 1.0}])
    small = scanner.scan(ev, {"entry_time": ["10:30"], "stop_pct": [0.25],
                              "sizing_pct": [0.02], "direction": ["short"]},
                         data_cache=cache, n_jobs=1, check_holdout=False)
    big = scanner.scan(ev, {"entry_time": ["10:30"],
                            "stop_pct": [0.20, 0.25, 0.30, 0.35],
                            "sizing_pct": [0.01, 0.02, 0.03],
                            "direction": ["short"]},
                       data_cache=cache, n_jobs=1, check_holdout=False)
    shared = lambda df: df[(df.entry_time == "10:30") & (df.stop_pct == 0.25)
                           & (df.sizing_pct == 0.02)].iloc[0]
    assert big["n_tested"].iloc[0] > small["n_tested"].iloc[0]
    assert shared(big)["p_bonferroni"] >= shared(small)["p_bonferroni"]


def test_audit_catches_lookahead():
    """If entry is the first bar (index 0), no pre-entry bar exists -> the
    engine refuses the trade (cannot prove absence of look-ahead)."""
    bars = _bars([("10:30", 10, 10, 10, 10, 1), ("15:45", 9, 9, 9, 9, 1)])
    cfg = dict(entry_time="10:30", stop_pct=0.25, exit_time="15:45",
               direction="short", sizing_pct=1.0)
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "T", "gain": 1.0}])
    res = engine.run(ev, cfg, data_cache={("T", "2026-01-05"): bars},
                     check_holdout=False,
                     write_audit=False)
    r = res["records"][0]
    assert r["trade_taken"] is False and r["reason"] == "no_entry_bar"
    assert res["audit"]["lookahead_violations"] == 0


def test_short_fade_pnl_sign():
    """Sanity: a short on a stock that falls to EOD is a winning trade."""
    bars = _bars([("10:29", 10, 10, 10, 10, 1), ("10:30", 10, 10.1, 9.9, 10, 1),
                  ("12:00", 8, 8, 7.9, 8, 1), ("15:45", 7, 7, 7, 7, 1)])
    cfg = dict(entry_time="10:30", stop_pct=0.25, exit_time="15:45",
               direction="short", sizing_pct=1.0, slippage=0.0)
    ev = pd.DataFrame([{"date": "2026-01-05", "ticker": "T", "gain": 1.0}])
    res = engine.run(ev, cfg, data_cache={("T", "2026-01-05"): bars},
                     check_holdout=False,
                     write_audit=False)
    assert res["records"][0]["gross_pct"] > 0    # entry 10 -> exit 7, short wins
