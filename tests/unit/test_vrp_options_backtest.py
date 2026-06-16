"""VRP iron-condor backtest — architecture/math tests against a MockThetaDataLoader.

Validates the simulator's condor math + control flow WITHOUT any API or network. The mock
builds chains so that EVERY symmetric 25-wide iron condor is worth a controllable net value
`V(date)` (call vertical = put vertical = V/2), independent of which strikes the backtest
picks — so credit, sizing, costs, and each exit branch are deterministic and assertable.
"""
import math

import numpy as np
import pandas as pd
import pytest

from sovereign.research.vrp.data_loader import OPTION_CHAIN_COLUMNS
from sovereign.research.vrp import strategy_simulator as ss

WING = 25
HALF_SPREAD = 0.05          # per-leg bid/ask = mid ± 0.05  => width 0.10
STRIKES = np.arange(200.0, 601.0, 1.0)


def _build_chain(value: float, expiration, dte) -> pd.DataFrame:
    """Chain where every (K, K±25) vertical closes for value/2, so the condor = `value`."""
    h = value / 2.0
    slope = h / WING
    rows = []
    for k in STRIKES:
        call_mid = 1000.0 - slope * k          # call vertical (short K, long K+25) = slope*25 = h
        put_mid = 50.0 + slope * k             # put vertical  (short K, long K-25) = slope*25 = h
        rows.append({
            "strike": float(k),
            "call_bid": call_mid - HALF_SPREAD, "call_ask": call_mid + HALF_SPREAD,
            "call_mid": call_mid, "call_iv": 0.20, "call_delta": 0.30,
            "put_bid": put_mid - HALF_SPREAD, "put_ask": put_mid + HALF_SPREAD,
            "put_mid": put_mid, "put_iv": 0.20, "put_delta": -0.30,
            "volume": 100, "open_interest": 1000,
            "expiration": pd.Timestamp(expiration), "dte": int(dte),
        })
    return pd.DataFrame(rows)


class MockThetaDataLoader:
    """Implements the ThetaDataLoader contract with deterministic synthetic chains."""

    def __init__(self, value_fn, dte: int = 35):
        self.value_fn = value_fn        # date -> condor net value
        self.dte = dte

    def get_underlying_close(self, symbol, date):
        return 400.0

    def get_chain_for_dte_range(self, symbol, date, dte_min, dte_max):
        exp = pd.Timestamp(date) + pd.Timedelta(days=self.dte)
        return _build_chain(self.value_fn(pd.Timestamp(date)), exp, self.dte)

    def get_option_chain(self, symbol, date, expiration):
        dte = (pd.Timestamp(expiration) - pd.Timestamp(date)).days
        return _build_chain(self.value_fn(pd.Timestamp(date)), expiration, dte)


def _spy_daily():
    """Deterministic SPY daily closes ~400 with small vol (so strikes land inside STRIKES)."""
    idx = pd.bdate_range("2022-02-01", "2022-05-06")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.002, len(idx))
    return pd.Series(400.0 * np.exp(np.cumsum(rets)), index=idx, name="SPY")


SPLIT = ("2022-03-07", "2022-03-11")        # one Monday => exactly one trade
PARAMS = {
    "short_legs_sd": 1.0, "wing_points": WING, "dte_entry": [30, 45], "manage_dte": 21,
    "profit_take_pct": 0.50, "stop_x_credit": 2.0, "account_risk_pct": 0.01,
    "commission_per_contract_leg_side": 0.65, "slippage_pct_of_bidask": 0.50,
    "entry_cadence": "monday_weekly", "rv_window_days": 20, "dte_scaling": "trading_days",
}


# ── guard: inert without a full loader call ──
def test_inert_without_loader():
    assert ss.iron_condor_simulate()["status"] == "DATA_INSUFFICIENT"
    # partial args (no spy_daily/params/split) also stay inert
    assert ss.iron_condor_simulate(loader=object())["status"] == "DATA_INSUFFICIENT"


# ── credit / sizing / costs math (entry value 8, profit exit) ──
def test_credit_sizing_and_costs():
    loader = MockThetaDataLoader(lambda d: 8.0 if d == pd.Timestamp("2022-03-07") else 3.0)
    res = ss.iron_condor_simulate(loader, spy_daily=_spy_daily(), params=PARAMS,
                                  split=SPLIT, account=1_000_000.0)
    assert res["status"] == "OK" and res["n_trades"] == 1
    t = res["trades"][0]
    assert t["credit"] == pytest.approx(8.0, abs=1e-3)
    assert t["max_loss"] == pytest.approx(WING - 8.0, abs=1e-3)
    # contracts = floor(account*risk / (max_loss*100)) = floor(10000/1700) = 5
    assert t["contracts"] == 5
    # costs = commission(0.65*4legs*2sides*5) + slippage(0.5*(0.10*4legs)*(entry+exit)*5*100)
    expected_costs = 0.65 * 4 * 2 * 5 + 0.5 * (0.10 * 4) * 2 * 5 * 100
    assert t["costs"] == pytest.approx(expected_costs, abs=1e-2)     # 26 + 200 = 226
    # gross = (credit - exit_value) * contracts * 100 = (8-3)*5*100 = 2500 ; net = 2500 - 226
    assert t["net"] == pytest.approx(2500 - expected_costs, abs=1e-1)


# ── exit-branch coverage ──
def test_profit_take_exit():
    loader = MockThetaDataLoader(lambda d: 10.0 if d == pd.Timestamp("2022-03-07") else 4.0)
    res = ss.iron_condor_simulate(loader, spy_daily=_spy_daily(), params=PARAMS,
                                  split=SPLIT, account=1_000_000.0)
    assert res["trades"][0]["exit_reason"] == "profit_take"     # 4 <= 0.5*10
    assert res["trades"][0]["net"] > 0


def test_stop_exit():
    # credit 5 => 2x = 10 < wing 25, so the stop is reachable before max loss
    loader = MockThetaDataLoader(lambda d: 5.0 if d == pd.Timestamp("2022-03-07") else 12.0)
    res = ss.iron_condor_simulate(loader, spy_daily=_spy_daily(), params=PARAMS,
                                  split=SPLIT, account=1_000_000.0)
    assert res["trades"][0]["exit_reason"] == "stop"            # 12 >= 2*5
    assert res["trades"][0]["net"] < 0


def test_manage_21dte_exit():
    # value stays between take (0.5*credit) and stop (2*credit) => runs to 21 DTE
    loader = MockThetaDataLoader(lambda d: 10.0)
    res = ss.iron_condor_simulate(loader, spy_daily=_spy_daily(), params=PARAMS,
                                  split=SPLIT, account=1_000_000.0)
    t = res["trades"][0]
    assert t["exit_reason"] == "manage_21dte"
    assert t["dte_at_exit"] <= PARAMS["manage_dte"]


# ── P&L bounded by the structure; both-sides + regime present ──
def test_pnl_bounded_and_reports_present():
    loader = MockThetaDataLoader(lambda d: 8.0 if d == pd.Timestamp("2022-03-07") else 3.0)
    res = ss.iron_condor_simulate(loader, spy_daily=_spy_daily(), params=PARAMS,
                                  split=SPLIT, account=1_000_000.0)
    t = res["trades"][0]
    upper = t["credit"] * t["contracts"] * 100
    lower = -WING * t["contracts"] * 100 - t["costs"]
    assert lower <= t["net"] <= upper
    assert "both_sides_by_realized_vol" in res
    assert "regime_conditional" in res
    assert set(["strike", "call_mid", "put_mid"]).issubset(OPTION_CHAIN_COLUMNS)
