"""Hand-computed ruleset scenarios + PropFirmRules equivalence (TICK-022 P2)."""

import numpy as np
import pytest

from research.prop_funnel.rulesets import (
    ChallengeEngine, DrawdownModel, FirmSpec, Outcome, PhaseSpec, TrailingLock,
)


def _spec(**kw) -> FirmSpec:
    base = dict(
        name="TEST", account_size=100_000.0, drawdown_model=DrawdownModel.STATIC,
        max_dd_usd=10_000.0, daily_loss_usd=None, consistency_pct=None,
        time_limit_days=None, sim_cap_trading_days=1_000,
        trailing_lock=TrailingLock.NONE, trailing_lock_plus_usd=0.0, kappa=1.25,
        trading_day_basis="with_trades", phases=[], fees={}, funded={},
    )
    base.update(kw)
    return FirmSpec(**base)


PHASE_10PCT = PhaseSpec("P1", 10_000.0, 1)


# (a) static vs REAL trailing divergence: +5k day, then -11k day
def test_static_survives_where_real_trailing_busts():
    days = [[+5.0], [-11.0]]                      # r-multiples at fixed $1k risk
    static = ChallengeEngine(_spec(), risk_pct=0.0, risk_usd=1_000.0)
    res = static.run_phase(PHASE_10PCT, days)
    assert res.outcome is Outcome.INCOMPLETE      # 94k, alive, stream exhausted
    assert res.equity_end == pytest.approx(94_000.0)

    trailing = ChallengeEngine(
        _spec(drawdown_model=DrawdownModel.EOD_TRAILING, trailing_lock=TrailingLock.AT_INITIAL),
        risk_pct=0.0, risk_usd=1_000.0)
    res = trailing.run_phase(PHASE_10PCT, days)
    assert res.outcome is Outcome.BUST            # floor rose to min(105k-10k, 100k)=95k
    assert res.fail_reason == "max_drawdown"


# (b) PropFirmRules-compat lock == static behavior
def test_compat_lock_never_trails():
    days = [[+5.0], [-11.0]]
    compat = ChallengeEngine(
        _spec(drawdown_model=DrawdownModel.EOD_TRAILING,
              trailing_lock=TrailingLock.AT_INITIAL_MINUS_DD),
        risk_pct=0.0, risk_usd=1_000.0)
    res = compat.run_phase(PHASE_10PCT, days)
    assert res.outcome is Outcome.INCOMPLETE      # floor stays 90k — same as static
    assert res.equity_end == pytest.approx(94_000.0)


# (c) intraday spike-and-giveback busts INTRADAY_TRAILING but not EOD_TRAILING
def test_intraday_trailing_catches_giveback():
    spec_kw = dict(account_size=50_000.0, max_dd_usd=2_500.0,
                   trailing_lock=TrailingLock.AT_INITIAL_PLUS_USD, trailing_lock_plus_usd=100.0)
    day = [[+3.0, -3.0]]                          # +3k then -3k, same day, $1k risk

    intraday = ChallengeEngine(_spec(drawdown_model=DrawdownModel.INTRADAY_TRAILING, **spec_kw),
                               risk_pct=0.0, risk_usd=1_000.0)
    res = intraday.run_phase(PhaseSpec("EVAL", 3_000.0, 1), day)
    assert res.outcome is Outcome.BUST            # hwm 53k -> floor locked at 50.1k; equity 50k
    assert res.fail_reason == "max_drawdown"

    eod = ChallengeEngine(_spec(drawdown_model=DrawdownModel.EOD_TRAILING, **spec_kw),
                          risk_pct=0.0, risk_usd=1_000.0)
    res = eod.run_phase(PhaseSpec("EVAL", 3_000.0, 1), day)
    assert res.outcome is Outcome.INCOMPLETE      # floor never saw the intraday high


# (d) kappa stress tightens the intraday floor
def test_kappa_stress_bracket():
    spec = _spec(drawdown_model=DrawdownModel.INTRADAY_TRAILING, account_size=50_000.0,
                 max_dd_usd=2_500.0, kappa=1.25)
    days = [[+2.0, -2.0]]                         # +2k (touch 2.5k stressed) then -2k

    stressed = ChallengeEngine(spec, risk_pct=0.0, risk_usd=1_000.0, kappa_stress=True)
    res = stressed.run_phase(PhaseSpec("EVAL", 3_000.0, 1), days)
    assert res.outcome is Outcome.BUST            # hwm 52.5k -> floor 50k; equity 50k <= 50k

    optimistic = ChallengeEngine(spec, risk_pct=0.0, risk_usd=1_000.0, kappa_stress=False)
    res = optimistic.run_phase(PhaseSpec("EVAL", 3_000.0, 1), days)
    assert res.outcome is Outcome.INCOMPLETE      # hwm 52k -> floor 49.5k; equity 50k survives


# (e) daily-loss breach measured from day-open, checked after each trade
def test_daily_loss_breach():
    spec = _spec(account_size=50_000.0, max_dd_usd=5_000.0, daily_loss_usd=1_100.0)
    eng = ChallengeEngine(spec, risk_pct=0.012)   # equity-relative sizing
    res = eng.run_phase(PhaseSpec("EVAL", 3_000.0, 1), [[-1.0, -1.0]])
    # trade1: -600 (50k*1.2%); trade2: -592.8 (49.4k*1.2%) -> day loss 1,192.8 >= 1,100
    assert res.outcome is Outcome.BUST
    assert res.fail_reason == "daily_loss"
    assert res.equity_end == pytest.approx(50_000.0 - 600.0 - 592.8)


# (f) consistency rule blocks pass until best day <= pct of total
def test_consistency_blocks_then_releases():
    spec = _spec(account_size=50_000.0, max_dd_usd=5_000.0, consistency_pct=0.30)
    days = [[+2.5], [+0.9]] + [[+0.5]] * 10       # $1k risk => dollars = r * 1000
    eng = ChallengeEngine(spec, risk_pct=0.0, risk_usd=1_000.0)
    res = eng.run_phase(PhaseSpec("EVAL", 3_000.0, 1), days)
    # target hit on day 2 (3.4k) but best day 2.5k > 30% of total until total >= 8,333.33
    assert res.outcome is Outcome.PASS
    assert res.trading_days == 12
    assert res.consistency_blocked_pass is True


# (g) min-trading-days delays pass; day basis matters
def test_min_days_and_day_basis():
    days = [[+11.0], [], [], []]                  # target hit day 1, then 3 empty days
    phase = PhaseSpec("P1", 10_000.0, 4)

    with_trades = ChallengeEngine(_spec(), risk_pct=0.0, risk_usd=1_000.0)
    res = with_trades.run_phase(phase, days)
    assert res.outcome is Outcome.INCOMPLETE      # only 1 day WITH trades
    assert res.trading_days == 1

    all_days = ChallengeEngine(_spec(trading_day_basis="all"), risk_pct=0.0, risk_usd=1_000.0)
    res = all_days.run_phase(phase, days)
    assert res.outcome is Outcome.PASS            # 4 counted days
    assert res.trading_days == 4


# (i) property equivalence vs the untouched PropFirmRules over 50 seeded sequences
def test_equivalence_vs_propfirmrules():
    from sovereign.propfirm.rules_engine import PropFirmRules

    outcomes_seen = set()
    for seed in range(50):
        rng = np.random.default_rng(seed)
        n_trades = int(rng.integers(30, 120))
        # near-zero EV mixture (~+0.04R) so PASS, INCOMPLETE and heavy-blocked paths all occur
        rs = rng.choice([-1.0, -0.5, 1.0, 1.5, 2.5], size=n_trades,
                        p=[0.40, 0.20, 0.22, 0.13, 0.05])
        risk_pct = float(rng.uniform(0.005, 0.05))
        # group into days of 0-3 trades
        days, i = [], 0
        while i < n_trades:
            k = int(rng.integers(0, 4))
            days.append(list(rs[i:i + k]))
            i += max(k, 1) if k else 0
            if k == 0:
                days[-1] = []
        # drive PropFirmRules (mff: 8% target, 10% dd, min 2 days, daily loss UNENFORCED)
        rules = PropFirmRules.mff(account_size=100_000)
        rules.risk_per_trade_pct = risk_pct
        rules.open_challenge()
        ref_outcome = None
        for day in days:
            for r in day:
                if not rules.is_active:
                    break
                rules.apply_trade_pnl(r_multiple=float(r))
            if not rules.is_active:
                ref_outcome = rules.outcome
                break
            rules.update_eod()
            if rules.is_passed():
                ref_outcome = "PASS"
                break
        if ref_outcome is None:
            ref_outcome = "INCOMPLETE"

        spec = _spec(drawdown_model=DrawdownModel.EOD_TRAILING, max_dd_usd=10_000.0,
                     trailing_lock=TrailingLock.AT_INITIAL_MINUS_DD,
                     trading_day_basis="all", sim_cap_trading_days=10_000)
        eng = ChallengeEngine(spec, risk_pct=risk_pct, buffer_cap_fraction=0.25)
        res = eng.run_phase(PhaseSpec("EVAL", 8_000.0, 2), days)

        ours = {"PASS": "PASS", "BUST": "BUST", "INCOMPLETE": "INCOMPLETE"}[res.outcome.value]
        assert ours == {"PASSED": "PASS", "PASS": "PASS", "BUST": "BUST",
                        "INCOMPLETE": "INCOMPLETE"}[ref_outcome], f"seed {seed}: {ours} != {ref_outcome}"
        assert res.equity_end == pytest.approx(rules.balance, abs=1e-6), f"seed {seed}"
        outcomes_seen.add(ours)

    # With buffer-capped sizing BUST is unreachable by construction (risk shrinks toward
    # the floor asymptotically) — the recorded artifact agrees (pass .7444 + timeout .2556 = 1.0).
    assert {"PASS", "INCOMPLETE"} <= outcomes_seen, f"property test never exercised: {outcomes_seen}"


def test_equivalence_vs_propfirmrules_uncapped_busts():
    """Disable the buffer cap on both engines so BUST paths are exercised too."""
    from sovereign.propfirm.rules_engine import PropFirmRules

    outcomes_seen = set()
    for seed in range(100, 125):
        rng = np.random.default_rng(seed)
        n_trades = int(rng.integers(60, 200))
        rs = rng.choice([-1.0, -0.5, 1.0, 1.5], size=n_trades, p=[0.40, 0.20, 0.25, 0.15])
        risk_pct = float(rng.uniform(0.02, 0.08))
        days = [list(rs[i:i + 2]) for i in range(0, n_trades, 2)]

        rules = PropFirmRules.mff(account_size=100_000)
        rules.risk_per_trade_pct = risk_pct
        rules.max_risk_buffer_fraction = 1e9        # cap never binds
        rules.open_challenge()
        ref_outcome = None
        for day in days:
            for r in day:
                if not rules.is_active:
                    break
                rules.apply_trade_pnl(r_multiple=float(r))
            if not rules.is_active:
                ref_outcome = rules.outcome
                break
            rules.update_eod()
            if rules.is_passed():
                ref_outcome = "PASS"
                break
        if ref_outcome is None:
            ref_outcome = "INCOMPLETE"

        spec = _spec(drawdown_model=DrawdownModel.EOD_TRAILING, max_dd_usd=10_000.0,
                     trailing_lock=TrailingLock.AT_INITIAL_MINUS_DD,
                     trading_day_basis="all", sim_cap_trading_days=10_000)
        eng = ChallengeEngine(spec, risk_pct=risk_pct, buffer_cap_fraction=None)
        res = eng.run_phase(PhaseSpec("EVAL", 8_000.0, 2), days)

        ours = res.outcome.value
        assert ours == {"PASSED": "PASS", "PASS": "PASS", "BUST": "BUST",
                        "INCOMPLETE": "INCOMPLETE"}[ref_outcome], f"seed {seed}: {ours} != {ref_outcome}"
        assert res.equity_end == pytest.approx(rules.balance, abs=1e-6), f"seed {seed}"
        outcomes_seen.add(ours)

    assert "BUST" in outcomes_seen, f"uncapped property test never busted: {outcomes_seen}"


# firms.yaml loads and normalizes
def test_firms_yaml_loads():
    specs = FirmSpec.load_all()
    assert {"FTMO_100K_SWING", "APEX_50K", "TOPSTEP_50K", "LUCID_100K", "MFF_100K",
            "FUNDERPRO_10K"} <= set(specs)
    ftmo = specs["FTMO_100K_SWING"]
    assert ftmo.drawdown_model is DrawdownModel.STATIC
    assert ftmo.max_dd_usd == 10_000.0 and ftmo.daily_loss_usd == 5_000.0
    assert [p.profit_target_usd for p in ftmo.phases] == [10_000.0, 5_000.0]
    apex = specs["APEX_50K"]
    assert apex.drawdown_model is DrawdownModel.INTRADAY_TRAILING
    assert apex.trailing_lock is TrailingLock.AT_INITIAL_PLUS_USD
    assert apex.max_dd_usd == 2_500.0 and apex.consistency_pct == 0.30
    mff = specs["MFF_100K"]
    assert mff.trailing_lock is TrailingLock.AT_INITIAL_MINUS_DD
    assert mff.daily_loss_usd == 4_000.0
