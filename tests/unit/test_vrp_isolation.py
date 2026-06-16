"""VRP isolation + pre-registration tripwire tests (NN#1).

Isolation: sovereign/research/vrp/ may READ forex/ICT OUTPUT files (e.g.
logs/forex_backtest_trades.json) but must import NO forex/ICT/intelligence/oracle
modules (TRADING_PHILOSOPHY time-horizon doctrine; project CLAUDE.md non-negotiable #1).
Mirrors tests/unit/test_es_nq_isolation.py.

Tripwire: the frozen gates in data/research/vrp_preregistration.json are the pre-registered
values. If a tripwire test fails, someone retuned a gate — that invalidates the verdict. Do
not "fix" the test; revert the constant in validator.py.
"""
import hashlib
import inspect
import json
from pathlib import Path

from sovereign.research.vrp import data_loader, strategy_simulator, validator, vrp_calculator

FORBIDDEN = [
    "from sovereign.forex", "import sovereign.forex",
    "from sovereign.ict", "import sovereign.ict",
    "from sovereign.intelligence", "import sovereign.intelligence",
    "from sovereign.oracle", "import sovereign.oracle",
    "from ict", "import ict",
    "from ict_engine", "import ict_engine",
]

MODULES = [data_loader, strategy_simulator, validator, vrp_calculator]
PREREG = Path(__file__).resolve().parents[2] / "data" / "research" / "vrp_preregistration.json"


def _assert_clean(module):
    src = inspect.getsource(module)
    for phrase in FORBIDDEN:
        assert phrase not in src, f"Isolation violated: '{phrase}' in {module.__name__}"


def test_all_vrp_modules_isolated():
    for m in MODULES:
        _assert_clean(m)


def test_preregistration_exists_and_matches_code():
    assert PREREG.exists(), "pre-registration artifact missing — freeze it before validating"
    p = json.loads(PREREG.read_text())
    assert p["stress_threshold_vix"] == validator.STRESS_VIX, (
        "PRE-REGISTERED stress threshold changed — invalidates the verdict. Revert validator.STRESS_VIX.")
    assert p["corr_gates"]["correlated"] == validator.GATE_CORRELATED
    assert p["corr_gates"]["crisis_max"] == validator.GATE_CRISIS
    assert p["corr_gates"]["diversifier_max_full"] == validator.GATE_DIVERSIFIER_FULL


def test_crisis_windows_match_prereg():
    p = json.loads(PREREG.read_text())
    prereg_crises = [tuple(c) for c in p["crisis_windows"]]
    assert prereg_crises == validator.CRISES, "crisis windows drifted from pre-registration"


def test_strategy_sim_is_inert():
    """The iron-condor sim must never fabricate a backtest WITHOUT a loader — DATA_INSUFFICIENT."""
    out = strategy_simulator.iron_condor_simulate()
    assert out["status"] == "DATA_INSUFFICIENT"
    assert "required_data" in out and "cost_model_spec" in out


def test_options_prereg_signature_intact():
    """Tripwire: recomputed SHA-256 of options_backtest split+params must match the stored
    signature. A mismatch means a frozen parameter was changed after signing — revert it or
    log a param_change; do NOT re-sign to make this pass."""
    ob = json.loads(PREREG.read_text())["options_backtest"]
    payload = {"split": ob["split"], "params": ob["params"]}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode()).hexdigest()
    assert ob["content_sha256"] == digest, (
        "OPTIONS PRE-REGISTRATION ALTERED AFTER SIGNING — invalidates the backtest verdict.")


def test_options_backtest_constants_frozen():
    """Structural constants the backtest depends on must not silently drift."""
    assert strategy_simulator.CONTRACT_MULTIPLIER == 100
    assert strategy_simulator.STRIKE_INCREMENT == 1.0
