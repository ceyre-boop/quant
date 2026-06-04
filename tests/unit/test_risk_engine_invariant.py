"""The INVARIANT: every decision satisfies final_risk <= base AND final_risk <= every ceiling.
A bug/bad estimate in any single layer may only ever make the system SAFER."""
import copy
import math
import random
import tempfile

from sovereign.risk.config.loader import load_risk_config
from sovereign.risk.risk_engine import decide
from sovereign.risk.risk_state import Position, RiskState, Signal

GRADES = ["A+", "A", "B", "C"]


def _cfg():
    c = copy.deepcopy(load_risk_config())
    c["audit"]["decisions_log"] = tempfile.mkstemp(suffix=".jsonl")[1]
    return c


CFG = _cfg()


def _rand_signal(rng):
    entry = rng.uniform(0.5, 2.0)
    stop = entry * (1 - rng.uniform(0.002, 0.02))
    return Signal("EURUSD=X", 1, entry, stop, rng.choice(GRADES), point_value=1.0)


def _rand_state(rng):
    start = 100_000.0
    equity = rng.uniform(93_000, 108_000)
    peak = max(equity, rng.uniform(equity, 110_000))
    dds, ddt = RiskState.derive_drawdowns(equity, peak, start)
    positions = [Position("X", 1, 1, 1.0, 0.99, rng.uniform(0, 0.01))
                 for _ in range(rng.randint(0, 3))]
    return RiskState(equity=equity, peak_equity=peak, starting_balance=start,
                     daily_realized_pnl=rng.uniform(-1500, 1500), daily_open_pnl=rng.uniform(-500, 500),
                     open_positions=positions, drawdown_static=dds, drawdown_trailing=ddt,
                     regime="NORMAL", threat_score=rng.uniform(0, 0.5),
                     health_ok=True, mc_breach_prob=rng.uniform(0, 0.1))


def test_final_never_exceeds_base_or_any_ceiling():
    rng = random.Random(7)
    for _ in range(3000):
        d = decide(_rand_signal(rng), _rand_state(rng), CFG)
        assert d.final_risk_pct >= 0.0
        assert d.final_risk_pct <= d.base_risk_pct + 1e-12, "final exceeded base"
        for key in ("kelly_ceiling", "portfolio_ceiling", "prop_ceiling"):
            c = d.layer_budgets[key]
            if isinstance(c, (int, float)) and math.isfinite(c):
                assert d.final_risk_pct <= c + 1e-12, f"final exceeded {key}"


def test_modulators_only_reduce():
    rng = random.Random(11)
    for _ in range(1000):
        d = decide(_rand_signal(rng), _rand_state(rng), CFG)
        for m in d.modulators.values():
            assert 0.0 <= m <= 1.0, "a modulator left [0,1] (would amplify)"


def test_adversarial_kelly_cannot_blow_account():
    """Inject edge_stats implying a huge full-Kelly. Final must STILL be capped by base/hard cap."""
    st = _rand_state(random.Random(1))
    st.edge_stats = {"forex_macro": {"win_rate": 0.9, "payoff": 10.0, "n_trades": 500}}
    sig = Signal("EURUSD=X", 1, 1.10, 1.09, "A+", point_value=1.0)
    d = decide(sig, st, CFG)
    assert d.final_risk_pct <= d.base_risk_pct + 1e-12     # never above the 1% A+ base
    assert d.final_risk_pct <= CFG["base"]["ceiling"] + 1e-12


def test_grade_parity_ict_vs_sovereign():
    """risk_config.yaml grades must match ict/micro_risk._GRADE_RISK — no silent drift."""
    from ict.micro_risk import _GRADE_RISK as ict_grades
    sov_grades = load_risk_config()["base"]["grade_risk"]
    for grade, ict_val in ict_grades.items():
        assert grade in sov_grades, f"grade {grade!r} in ICT but missing in sovereign config"
        assert abs(sov_grades[grade] - ict_val) < 1e-6, (
            f"grade {grade!r} drift: ICT={ict_val} sovereign={sov_grades[grade]}")
