"""ES/NQ isolation + pre-registration tripwire tests.

Isolation: sovereign/es_nq/ may import sovereign.futures plumbing and
sovereign.utils.kill_switch — NEVER forex/ICT/intelligence/oracle (brief rule #8,
TRADING_PHILOSOPHY time-horizon doctrine).

Tripwire: the pre-registered bias weights in config/es_nq_params.yml are the
brief's values. If this test fails, someone fit the weights — that invalidates
every validation verdict. Do not "fix" the test; revert the weights.
"""
import inspect

from sovereign.es_nq import (backtest, config, daily_bias_engine, data_store,
                             live_scanner, session_logger, session_sizing,
                             structure_gate)

FORBIDDEN = [
    "from sovereign.forex", "import sovereign.forex",
    "from sovereign.intelligence", "import sovereign.intelligence",
    "from sovereign.oracle", "import sovereign.oracle",
    "from ict", "import ict",
    "from layer1", "import layer1",
    "from layer2", "import layer2",
]

MODULES = [backtest, config, daily_bias_engine, data_store, live_scanner,
           session_logger, session_sizing, structure_gate]


def _assert_clean(module):
    src = inspect.getsource(module)
    for phrase in FORBIDDEN:
        assert phrase not in src, f"Isolation violated: '{phrase}' in {module.__name__}"


def test_all_es_nq_modules_isolated():
    for m in MODULES:
        _assert_clean(m)


def test_preregistered_weights_unchanged():
    w = config.es_nq_params()["bias"]["weights"]
    assert w == {"overnight": 0.30, "calendar": 0.25, "vix": 0.20,
                 "hurst": 0.15, "international": 0.10}, (
        "PRE-REGISTERED WEIGHTS CHANGED — this invalidates all validation verdicts. "
        "Revert config/es_nq_params.yml (see data/research/es_nq_preregistration.json).")


def test_preregistered_neutral_threshold():
    assert config.es_nq_params()["bias"]["neutral_below_confidence"] == 0.40


def test_contract_specs():
    assert config.tick_value_usd("MES") == 1.25
    assert config.tick_value_usd("MNQ") == 0.50
    assert config.contract_spec("NQ")["dollars_per_point"] == 2.0   # NQ → MNQ alias


def test_cost_model_exact():
    # MNQ target exit: slippage (0.25+0.25 ticks)*0.25pt*$2 = $0.25; commission $0.70
    assert config.round_turn_cost_usd("MNQ") == 0.95
    # MNQ stop exit: (0.25+0.5)*0.25*$2 = $0.375 + $0.70
    assert config.round_turn_cost_usd("MNQ", stop_fill=True) == 1.075
