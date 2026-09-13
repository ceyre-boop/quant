import numpy as np
import pandas as pd
import pytest
from datetime import time as dtime

from research.formal.dsl import Rule, lag, win, at, cs, fn, now
from research.formal.causality import prove
from research.formal.mechanics import max_loss_before_halt, ssr_constraint
from research.formal.costs import break_even


def test_hyp105_universe_leak_is_caught():
    """HYP-105: universe selected on gain AT 10:30, trade entered at 09:31 — the refuted look-ahead."""
    rule = Rule("HYP-105-reconstructed",
                features={"gap": fn("div", at("open", 0, "09:30"), lag("close", 1))},
                universe=fn("ge", at("gain", 0, clock="10:30"), fn("const")),
                decision_clock="09:31", entry_offset=0)
    r = prove(rule)
    assert not r.causal and r.leaks and r.leaks[0]["series"] == "gain" and r.leaks[0]["clock"] == "10:30"


def test_hyp107_honest_universe_passes():
    rule = Rule("HYP-107-honest",
                features={"gap": fn("div", at("open", 0, "09:30"), lag("close", 1)), "v1": at("vol_0931", 0, "09:30")},
                universe=fn("ge", at("gap_0931", 0, clock="09:30"), fn("const")),
                decision_clock="09:31", entry_offset=0)
    assert prove(rule).causal


def test_fade_rule_passes():
    rule = Rule("post-shock fade", features={"shock": fn("ge", fn("abs", lag("r", 1)), win("absr", -253, -2)), "dir": fn("sign", lag("r", 1))})
    assert prove(rule).causal


def test_decision_bar_use_is_a_leak():
    rule = Rule("uses own bar", features={"x": now("close")})
    r = prove(rule); assert not r.causal and r.leaks[0]["end"] == 0


def test_window_including_decision_bar_is_a_leak():
    rule = Rule("rvol incl t", features={"rvol": win("volume", -5, 0)})
    assert not prove(rule).causal


def test_carry_v1_passes():
    rule = Rule("carry v1", features={"carry": lag("rate_diff", 1), "mom12": win("ret", -12, -1), "value": lag("real_fx_5y_chg", 1), "rvol": win("ret", -3, -1)})
    assert prove(rule).causal


def test_luld_bound_matches_band_when_reference_is_entry():
    b = max_loss_before_halt(10.0, dtime(10, 0), tier=2)
    assert b.band == pytest.approx(0.10) and b.max_adverse_frac == pytest.approx(0.10, abs=1e-6)
    b2 = max_loss_before_halt(10.0, dtime(9, 35), tier=2)
    assert b2.doubled and b2.max_adverse_frac == pytest.approx(0.20, abs=1e-6)


def test_luld_bound_widens_with_lagging_reference():
    # price ran from 8 to 10 in the last 5 minutes: the reference lags, so a halt triggers later
    b = max_loss_before_halt(10.0, dtime(10, 0), tier=2, prior_closes=[8.0, 8.5, 9.0, 9.5, 10.0])
    assert b.max_adverse_frac > 0.10 and b.max_adverse_frac <= 0.30


def test_ssr():
    assert ssr_constraint(-0.12)["ssr_active"] and not ssr_constraint(-0.05)["ssr_active"]


def test_break_even_monotone_in_liquidity():
    a = break_even(5.0, 0.03, 5e4); b = break_even(5.0, 0.03, 5e6)
    assert a.min_gross_per_trade > b.min_gross_per_trade > 0
