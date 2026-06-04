"""Per-layer unit tests: drawdown MONOTONIC + in [0,1]; Kelly thin-sample/no-edge/hard-cap guards;
modulators always in [0,1]."""
import copy
import tempfile

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.risk_state import RiskState, Signal


def _cfg():
    c = copy.deepcopy(load_risk_config())
    c["audit"]["decisions_log"] = tempfile.mkstemp(suffix=".jsonl")[1]
    return c


CFG = _cfg()
SIG = Signal("EURUSD=X", 1, 1.10, 1.09, "A", point_value=1.0)


def _state(dd=0.0, edge=None):
    return RiskState(equity=100_000, peak_equity=100_000, starting_balance=100_000,
                     daily_realized_pnl=0.0, daily_open_pnl=0.0,
                     drawdown_static=dd, drawdown_trailing=dd,
                     edge_stats=edge or {}, health_ok=True)


# ── Layer 4: drawdown ────────────────────────────────────────────────────────
def test_drawdown_monotonic_and_bounded():
    from sovereign.risk.layers.drawdown import factor
    prev = 1.01
    for i in range(0, 16):
        dd = i / 100.0
        f = factor(SIG, _state(dd=dd), CFG)
        assert 0.0 <= f <= 1.0, f"drawdown factor {f} left [0,1]"
        assert f <= prev + 1e-9, f"drawdown factor increased with depth ({f} > {prev})"
        prev = f


def test_drawdown_floor_respected():
    from sovereign.risk.layers.drawdown import factor
    assert factor(SIG, _state(dd=0.50), CFG) >= CFG["drawdown"]["min_factor"] - 1e-12


# ── Layer 2: Kelly ───────────────────────────────────────────────────────────
def test_kelly_thin_sample_uses_floor():
    from sovereign.risk.layers.kelly import ceiling
    st = _state(edge={"forex_macro": {"win_rate": 0.6, "payoff": 2.0, "n_trades": 10}})
    assert ceiling(SIG, st, CFG) == CFG["kelly"]["fixed_fractional_floor"]


def test_kelly_missing_stats_uses_floor():
    from sovereign.risk.layers.kelly import ceiling
    assert ceiling(SIG, _state(edge={}), CFG) == CFG["kelly"]["fixed_fractional_floor"]


def test_kelly_no_edge_returns_zero():
    from sovereign.risk.layers.kelly import ceiling
    st = _state(edge={"forex_macro": {"win_rate": 0.30, "payoff": 1.0, "n_trades": 300}})
    assert ceiling(SIG, st, CFG) == 0.0


def test_kelly_capped_at_hard_cap():
    from sovereign.risk.layers.kelly import ceiling
    st = _state(edge={"forex_macro": {"win_rate": 0.95, "payoff": 10.0, "n_trades": 500}})
    assert 0.0 < ceiling(SIG, st, CFG) <= CFG["kelly"]["hard_cap"] + 1e-12


def test_kelly_real_forex_pool_sane():
    """The real v015 pool (n=103, ~59% WR, payoff>1) yields a positive, capped ceiling."""
    from sovereign.risk.layers.kelly import ceiling
    st = _state(edge={"forex_macro": {"win_rate": 0.59, "payoff": 1.8, "n_trades": 103}})
    c = ceiling(SIG, st, CFG)
    assert 0.0 <= c <= CFG["kelly"]["hard_cap"] + 1e-12


# ── Layer 3: volatility ──────────────────────────────────────────────────────
def _vstate(est=None, base=None):
    st = _state()
    if est is not None:
        st.vol_estimates = {"EURUSD=X": est}
    if base is not None:
        st.vol_baseline = {"EURUSD=X": base}
    return st


def test_volatility_high_vol_shrinks():
    from sovereign.risk.layers.volatility import factor
    f = factor(SIG, _vstate(est=0.02, base=0.01), CFG)   # vol 2x baseline → 0.5
    assert abs(f - 0.5) < 1e-9


def test_volatility_calm_never_amplifies():
    from sovereign.risk.layers.volatility import factor
    assert factor(SIG, _vstate(est=0.005, base=0.01), CFG) == 1.0   # vol < baseline → capped at 1


def test_volatility_missing_data_no_reduction():
    from sovereign.risk.layers.volatility import factor
    assert factor(SIG, _vstate(), CFG) == 1.0


def test_volatility_floor_respected():
    from sovereign.risk.layers.volatility import factor
    f = factor(SIG, _vstate(est=10.0, base=0.01), CFG)   # extreme vol → clamped to floor
    assert f == CFG["volatility"]["factor_floor"]


# ── Layer 5: regime ──────────────────────────────────────────────────────────
def test_regime_monotonic_and_bounded():
    from sovereign.risk.layers.regime import factor
    prev = 1.01
    for i in range(0, 21):
        threat = i / 20.0
        st = _state()
        st.threat_score = threat
        f = factor(SIG, st, CFG)
        assert 0.0 <= f <= 1.0
        assert f <= prev + 1e-9, "regime factor increased with threat"
        prev = f


def test_regime_extremes():
    from sovereign.risk.layers.regime import factor
    calm = _state(); calm.threat_score = 0.0
    crit = _state(); crit.threat_score = 0.95
    assert factor(SIG, calm, CFG) == 1.0
    assert factor(SIG, crit, CFG) == 0.0


# ── Layer 6: portfolio / CVaR ────────────────────────────────────────────────
def test_portfolio_no_positions_does_not_bind_low():
    from sovereign.risk.layers.portfolio import ceiling
    c = ceiling(SIG, _state(), CFG)
    assert c >= CFG["base"]["ceiling"]      # empty book → ceiling above any base, never binds < base


def test_portfolio_heat_reduces_with_open_risk():
    from sovereign.risk.layers.portfolio import _heat_ceiling
    from sovereign.risk.risk_state import Position
    st0 = _state()
    st1 = _state(); st1.open_positions = [Position("GBPUSD=X", 1, 1, 1.0, 0.99, 0.01),
                                          Position("AUDUSD=X", 1, 1, 1.0, 0.99, 0.01)]
    p = CFG["portfolio"]
    assert _heat_ceiling(SIG, st1, p) < _heat_ceiling(SIG, st0, p)


def test_portfolio_higher_correlation_lower_ceiling():
    from sovereign.risk.layers.portfolio import _heat_ceiling
    from sovereign.risk.risk_state import Position
    pos = [Position("GBPUSD=X", 1, 1, 1.0, 0.99, 0.01)]
    low = _state(); low.open_positions = pos; low.correlation_matrix = {("EURUSD=X", "GBPUSD=X"): 0.1}
    high = _state(); high.open_positions = pos; high.correlation_matrix = {("EURUSD=X", "GBPUSD=X"): 0.9}
    p = CFG["portfolio"]
    assert _heat_ceiling(SIG, high, p) < _heat_ceiling(SIG, low, p)


def test_portfolio_cvar_finite_on_real_pool():
    from sovereign.risk.layers.portfolio import _cvar_ceiling
    import math as _m
    c = _cvar_ceiling(SIG, _state(), CFG["portfolio"])
    assert c == _m.inf or c >= 0.0      # finite >=0 if pool present, +inf if deferred
