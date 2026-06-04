"""Layer 7 prop survival: for ANY decision the engine PERMITS, the correlated worst case
(this stop + every open stop hitting simultaneously) must NEVER breach the daily or max-DD floor."""
import copy
import random
import tempfile

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.risk_engine import decide
from sovereign.risk.risk_state import Position, RiskState, Signal


def _cfg():
    c = copy.deepcopy(load_risk_config())
    c["audit"]["decisions_log"] = tempfile.mkstemp(suffix=".jsonl")[1]
    return c


CFG = _cfg()
P = CFG["prop"]
ACCT = P["account_size"]


def _rand_state(rng):
    start = 100_000.0
    equity = rng.uniform(93_000, 109_000)
    peak = max(equity, rng.uniform(equity, 112_000))
    dds, ddt = RiskState.derive_drawdowns(equity, peak, start)
    positions = [Position("X", 1, 1, 1.0, 0.99, rng.uniform(0, 0.012))
                 for _ in range(rng.randint(0, 4))]
    return RiskState(equity=equity, peak_equity=peak, starting_balance=start,
                     daily_realized_pnl=rng.uniform(-3000, 2000), daily_open_pnl=rng.uniform(-1000, 1000),
                     open_positions=positions, drawdown_static=dds, drawdown_trailing=ddt,
                     health_ok=True, threat_score=0.0, mc_breach_prob=0.0)


def _rand_signal(rng):
    entry = rng.uniform(0.8, 1.6)
    stop = entry * (1 - rng.uniform(0.002, 0.02))
    return Signal("EURUSD=X", 1, entry, stop, rng.choice(["A+", "A", "B", "C"]), point_value=1.0)


def _floors(st):
    today_pnl = st.daily_realized_pnl + st.daily_open_pnl
    day_start = st.equity - today_pnl
    daily_floor = day_start - P["daily_loss_limit_pct"] * ACCT
    if P["drawdown_type"] == "trailing":
        dd_floor = st.peak_equity * (1 - P["max_drawdown_pct"])
    else:
        dd_floor = st.starting_balance * (1 - P["max_drawdown_pct"])
    return max(daily_floor, dd_floor)


def test_permitted_decisions_never_breach_floors():
    rng = random.Random(3)
    permitted = 0
    for _ in range(6000):
        st, sig = _rand_state(rng), _rand_signal(rng)
        d = decide(sig, st, CFG)
        if d.halt or d.final_risk_pct == 0:
            continue
        permitted += 1
        this_loss = d.final_risk_pct * st.equity
        open_loss = sum(pos.risk_pct_at_entry for pos in st.open_positions) * st.equity
        worst_equity = st.equity - this_loss - open_loss
        assert worst_equity >= _floors(st) - 1e-6, (
            f"worst-case {worst_equity:.2f} breaches floor {_floors(st):.2f}")
    assert permitted > 0, "test exercised no permitted decisions — broaden the scenarios"


def test_already_at_edge_returns_zero():
    """Open positions consuming the whole budget → prop ceiling 0 (no new risk)."""
    st = RiskState(equity=93_000, peak_equity=100_000, starting_balance=100_000,
                   daily_realized_pnl=0.0, daily_open_pnl=0.0, health_ok=True,
                   threat_score=0.0, mc_breach_prob=0.0,
                   drawdown_trailing=0.0,  # avoid the Layer-0 dd gate so we isolate Layer 7
                   open_positions=[Position("X", 1, 1, 1.0, 0.99, 0.01) for _ in range(3)])
    sig = Signal("EURUSD=X", 1, 1.10, 1.09, "A+", point_value=1.0)
    d = decide(sig, st, CFG)
    # peak 100k, 8% trailing floor = 92k; equity 93k → only ~1k budget, 3 open @1% = 3k → prop 0.
    assert d.layer_budgets["prop_ceiling"] == 0.0
