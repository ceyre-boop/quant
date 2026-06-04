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
