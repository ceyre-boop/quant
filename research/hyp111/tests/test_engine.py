import numpy as np
import pandas as pd
import pytest

from research.hyp111.engine import simulate, naive, confluence, straddle, pick_expiration
from research.hyp111.date_bootstrap import date_block_bootstrap, date_groups


def bars(closes, lo=None, hi=None, start="09:30"):
    n = len(closes)
    t = pd.date_range(f"2024-01-02 {start}", periods=n, freq="1min").strftime("%H:%M")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"time": t, "open": o, "high": hi if hi is not None else np.maximum(o, c) + 0.01,
                         "low": lo if lo is not None else np.minimum(o, c) - 0.01, "close": c, "volume": 100.0})


def test_retrace_reclaim_target():
    # up-shock: C=100, L=99, T=101. Path: dip to 99, reclaim 100.2, next open, run to 101.
    b = bars([99.8, 99.0, 99.5, 100.2, 100.4, 101.2, 101.5])
    tr = simulate(b, 1.0, 100.0, 99.0, 101.0)
    assert tr.triggered and tr.tau1 == "09:31" and tr.tau2 == "09:33"
    assert tr.entry == pytest.approx(100.2) and tr.exit == 101.0 and tr.exit_kind == "target"


def test_no_retrace_no_trade():
    b = bars([100.5, 100.8, 101.0, 101.4])
    assert not simulate(b, 1.0, 100.0, 99.0, 101.0).triggered


def test_stop_beats_target_same_bar():
    b = bars([99.0, 100.3, 100.5, 100.5], lo=[98.9, 100.2, 98.5, 100.4], hi=[99.1, 100.4, 101.5, 100.6])
    tr = simulate(b, 1.0, 100.0, 99.0, 101.0)
    assert tr.triggered and tr.exit_kind == "stop" and tr.exit == 99.0


def test_reclaim_after_1430_is_no_trade():
    n = 5 * 60 + 5   # 09:30 .. 14:34
    c = np.full(n, 99.5); c[-2:] = 100.5
    b = bars(c)
    assert not simulate(b, 1.0, 100.0, 99.0, 101.0).triggered


def test_lookahead_canary():
    """Bars after tau2+1 must not change tau1/tau2/entry."""
    b = bars([99.0, 100.2, 100.3, 100.3, 100.3, 100.3])
    full = simulate(b, 1.0, 100.0, 99.0, 101.0)
    mutated = b.copy(); mutated.loc[3:, ["open", "high", "low", "close"]] = 50.0
    part = simulate(mutated, 1.0, 100.0, 99.0, 101.0)
    assert (full.tau1, full.tau2, full.entry) == (part.tau1, part.tau2, part.entry)


def test_down_shock_symmetric():
    b = bars([101.0, 99.7, 99.5, 98.9])
    tr = simulate(b, -1.0, 100.0, 101.0, 99.0)
    assert tr.triggered and tr.exit_kind == "target" and tr.ret_gross > 0


def test_naive_and_confluence():
    b = bars([100.5, 100.2, 100.6, 100.7])
    assert naive(b, 1.0) == pytest.approx((100.7 - 100.5) / 100.5)
    tr = simulate(bars([99.0, 100.2, 100.3, 100.3]), 1.0, 100.0, 99.0, 101.0)
    d = confluence(bars([99.0, 100.2, 100.3, 100.3]), bars([100.0, 100.1, 100.2, 100.3]), tr, 1.0, 100.0, True, False)
    assert set(d) == {"c1", "c2", "c3", "c4", "c5", "count"} and d["count"] == sum(d[k] for k in ("c1","c2","c3","c4","c5"))


def test_straddle_and_expiration():
    ch = pd.DataFrame({"date": ["20240102"] * 4 + ["20240109"] * 4, "strike": [99, 99, 100, 100] * 2,
                       "right": ["C", "P"] * 4, "bid": [2.0, 1.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.9],
                       "ask": [2.1, 1.1, 1.6, 1.6, 1.1, 1.1, 0.9, 1.0], "close": 0.0, "volume": 0.0})
    r = straddle(ch, "20240102", "20240109", spot=100.2)
    assert r["strike"] == 100 and r["premium_in"] == pytest.approx(3.2) and r["premium_out"] == pytest.approx(1.7)
    assert pick_expiration(["20240105", "20240112", "20240119"], "2024-01-02") == "20240112"


def test_date_bootstrap_keeps_same_date_rows_together():
    dates = np.array(["d1", "d1", "d2", "d3", "d3", "d3"])
    _, groups = date_groups(dates)
    seen = date_block_bootstrap(dates, lambda rows: float(np.all(np.isin(rows, groups[0])) or True), draws=5)
    # every draw must contain complete date groups: check via a stat that fails otherwise
    def complete(rows):
        for g in groups:
            inter = np.intersect1d(rows, g)
            if len(inter) not in (0, len(g)) and not (len(inter) > 0 and len(np.unique(rows[np.isin(rows, g)])) == len(g)):
                return 0.0
        return 1.0
    out = date_block_bootstrap(dates, complete, draws=200, seed=1)
    assert out.min() == 1.0
