"""Layer 0 gates: each fires independently → final_size == 0 with a specific halt_reason."""
import copy
import tempfile

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.risk_engine import decide
from sovereign.risk.risk_state import RiskState, Signal


def _cfg():
    c = copy.deepcopy(load_risk_config())
    c["audit"]["decisions_log"] = tempfile.mkstemp(suffix=".jsonl")[1]
    return c


CFG = _cfg()
SIG = Signal("EURUSD=X", 1, 1.10, 1.09, "A", point_value=1.0)


def _state(**over):
    st = RiskState(equity=100_000, peak_equity=100_000, starting_balance=100_000,
                   daily_realized_pnl=0.0, daily_open_pnl=0.0, health_ok=True,
                   threat_score=0.0, mc_breach_prob=0.0)
    for k, v in over.items():
        setattr(st, k, v)
    return st


def test_clean_state_passes():
    d = decide(SIG, _state(), CFG)
    assert not d.halt and d.final_risk_pct > 0 and d.final_size >= 0


def test_daily_loss_gate():
    d = decide(SIG, _state(daily_realized_pnl=-5_100), CFG)   # -5.1% > 5% limit
    assert d.halt and d.final_size == 0 and "daily_loss_limit" in d.halt_reason


def test_internal_guard_daily_gate():
    d = decide(SIG, _state(daily_realized_pnl=-2_100), CFG)   # -2.1% > 2% internal guard
    assert d.halt and "internal_guard_daily" in d.halt_reason


def test_internal_guard_trailing_dd_gate():
    d = decide(SIG, _state(equity=94_000, peak_equity=100_000, drawdown_trailing=0.06), CFG)
    assert d.halt and "internal_guard_trailing_dd" in d.halt_reason


def test_health_gate():
    d = decide(SIG, _state(health_ok=False), CFG)
    assert d.halt and "health" in d.halt_reason


def test_threat_gate():
    d = decide(SIG, _state(threat_score=0.9), CFG)
    assert d.halt and "threat_critical" in d.halt_reason


def test_mc_breach_gate():
    d = decide(SIG, _state(mc_breach_prob=0.40), CFG)
    assert d.halt and "mc_breach_prob" in d.halt_reason


def test_deep_drawdown_halts():
    st = _state(equity=92_600, peak_equity=100_000, drawdown_trailing=0.074, drawdown_static=0.074)
    d = decide(SIG, st, CFG)
    assert d.halt and d.final_size == 0
