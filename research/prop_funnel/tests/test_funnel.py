"""Vectorized-vs-scalar exactness + analytic sanity anchors (TICK-022 P4)."""

import numpy as np
import pytest

from research.prop_funnel import feeds, simulate
from research.prop_funnel.funnel import run_funnel
from research.prop_funnel.rulesets import (
    ChallengeEngine, DrawdownModel, FirmSpec, Outcome, PhaseSpec, TrailingLock,
)
from research.prop_funnel.simulate import (
    OUTCOME_BUST, OUTCOME_INCOMPLETE, OUTCOME_PASS, OUTCOME_TIMEOUT,
)

CODE_TO_OUTCOME = {OUTCOME_PASS: Outcome.PASS, OUTCOME_BUST: Outcome.BUST,
                   OUTCOME_TIMEOUT: Outcome.TIMEOUT, OUTCOME_INCOMPLETE: Outcome.INCOMPLETE}


def _spec(**kw) -> FirmSpec:
    base = dict(
        name="TEST", account_size=100_000.0, drawdown_model=DrawdownModel.STATIC,
        max_dd_usd=10_000.0, daily_loss_usd=None, consistency_pct=None,
        time_limit_days=None, sim_cap_trading_days=120,
        trailing_lock=TrailingLock.NONE, trailing_lock_plus_usd=0.0, kappa=1.25,
        trading_day_basis="with_trades", phases=[], fees={}, funded={},
    )
    base.update(kw)
    return FirmSpec(**base)


SPEC_VARIANTS = [
    _spec(),
    _spec(daily_loss_usd=2_000.0),
    _spec(drawdown_model=DrawdownModel.EOD_TRAILING, trailing_lock=TrailingLock.AT_INITIAL),
    _spec(drawdown_model=DrawdownModel.EOD_TRAILING,
          trailing_lock=TrailingLock.AT_INITIAL_MINUS_DD, trading_day_basis="all",
          time_limit_days=90),
    _spec(drawdown_model=DrawdownModel.INTRADAY_TRAILING,
          trailing_lock=TrailingLock.AT_INITIAL_PLUS_USD, trailing_lock_plus_usd=100.0,
          max_dd_usd=2_500.0, daily_loss_usd=1_100.0),
    _spec(consistency_pct=0.30),
]


@pytest.mark.parametrize("spec_i", range(len(SPEC_VARIANTS)))
def test_vectorized_matches_scalar_engine(spec_i):
    """Same draws through simulate.evaluate and ChallengeEngine → identical results."""
    spec = SPEC_VARIANTS[spec_i]
    phase = PhaseSpec("P1", 8_000.0, 3)
    kappa_stress = spec.drawdown_model is DrawdownModel.INTRADAY_TRAILING
    pool = feeds.synthetic_pool(sharpe_ann=1.0, trades_per_day=1.2)
    risk_pct = 0.02

    rng = np.random.default_rng(1000 + spec_i)
    ds = simulate.sample_trades(rng, pool, n_attempts=60, days=spec.sim_cap_trading_days)
    stats = simulate.evaluate(ds, spec, phase, risk_pct, kappa_stress=kappa_stress)

    for i in range(ds.n):
        eng = ChallengeEngine(spec, risk_pct=risk_pct, kappa_stress=kappa_stress)
        res = eng.run_phase(phase, ds.day_r_lists(i))
        assert res.outcome is CODE_TO_OUTCOME[int(stats.outcome[i])], (
            f"spec {spec_i} row {i}: scalar {res.outcome} vs vector {stats.outcome[i]}")
        if res.outcome in (Outcome.PASS, Outcome.TIMEOUT):
            assert res.trading_days == int(stats.trading_days_at_event[i]), f"spec {spec_i} row {i}"
        assert res.equity_end == pytest.approx(float(stats.equity_end[i]), rel=1e-9), (
            f"spec {spec_i} row {i}")


def test_sharpe0_two_phase_anchor():
    """Zero-edge strategy through FTMO-style 2-phase: crude-MC anchor band ~25-35%."""
    spec = FirmSpec.load("FTMO_100K_SWING")
    pool = feeds.synthetic_pool(sharpe_ann=0.0, trades_per_day=1.0)
    risk = feeds.synthetic_risk_pct(0.01, 1.0)          # 1%/day vol
    rng = np.random.default_rng(7)
    row = run_funnel(rng, pool, spec, challenge_risk_mult=risk / pool.base_risk_pct,
                     n_attempts=4_000, n_funded_sims=2_000)
    assert 0.18 <= row["p_funded"] <= 0.40, row["p_funded"]
    # zero-edge funded months can't clear $10k/mo on 100k reliably
    assert row["funded"]["p_target_every_month_12"] < 0.001


def test_sharpe15_lowvol_anchor():
    """Sharpe 1.5 at 0.25%/day vol: near-certain pass, glacial, no $10k months."""
    spec = FirmSpec.load("FTMO_100K_SWING")
    pool = feeds.synthetic_pool(sharpe_ann=1.5, trades_per_day=1.0)
    risk = feeds.synthetic_risk_pct(0.0025, 1.0)
    rng = np.random.default_rng(7)
    row = run_funnel(rng, pool, spec, challenge_risk_mult=risk / pool.base_risk_pct,
                     funded_risk_mult=risk / pool.base_risk_pct,
                     n_attempts=3_000, n_funded_sims=3_000)
    p1 = row["phases"][0]
    assert p1["p_bust"] <= 0.02, p1                          # low vol almost never busts...
    assert p1["p_pass"] + p1["p_incomplete"] >= 0.95, p1     # ...it passes or runs out the 500d cap
    assert p1["tdays_to_pass_med"] > 150                     # the grind (crude-MC saw 381d at cap 1500)
    assert row["funded"]["p_month_ge_target"] < 0.01         # $10k/mo impossible at this vol


def test_sharpe15_hotvol_anchor():
    """Same edge at 2%/day vol: income possible, pass rate collapses — the tension."""
    spec = FirmSpec.load("FTMO_100K_SWING")
    pool = feeds.synthetic_pool(sharpe_ann=1.5, trades_per_day=1.0)
    risk = feeds.synthetic_risk_pct(0.02, 1.0)
    rng = np.random.default_rng(7)
    row = run_funnel(rng, pool, spec, challenge_risk_mult=risk / pool.base_risk_pct,
                     funded_risk_mult=risk / pool.base_risk_pct,
                     n_attempts=3_000, n_funded_sims=3_000)
    assert 0.30 <= row["p_funded"] <= 0.55, row["p_funded"]
    # $10k/mo = 10%/mo on the 100k account — possible but far from reliable
    assert 0.08 <= row["funded"]["p_month_ge_target"] <= 0.30, row["funded"]


def test_insufficient_pool_gets_honest_row():
    spec = FirmSpec.load("FTMO_100K_SWING")
    pool = feeds.load_futures_orb()                        # n=2
    row = run_funnel(np.random.default_rng(0), pool, spec)
    assert row["verdict"] == "INSUFFICIENT_DATA"
    assert "n=2" in row["note"] or "n=" in row["note"]
