"""The L1/L2 wall, enforced as a test rather than a convention.

`docs/ARCHITECTURE.md:284-286`:

    "Only the FACT of a setup crosses L1 -> L2: direction, the entry trigger, and
     the risk geometry the setup implies. NO REASONING, NO PROBABILITY, NO
     NARRATIVE CROSSES."

`ARCHITECTURE.md:362-367` §6.3 records what happens without enforcement:
`futures/decision_engine.evaluate_entry()` returns setup_type+direction (L1)
fused with stop+target+contracts+confidence (L2) in one object — "prediction and
evaluation fused at birth, so neither can be tested or replaced in isolation."
That is the only component in the repo classified as violating, and it got there
because nothing stopped it.

DO NOT "FIX" A FAILURE HERE BY RELAXING THE TEST. If a signal module needs the
bias, that is a design change requiring a preregistration — the predictive layer
has 23+ clean nulls and has not earned gating authority.

Pattern borrowed from tests/unit/test_es_nq_isolation.py:14-36 (source-inspection
import guard) and tests/test_pipeline_does_not_import_sovereign.
"""
import inspect

import pytest


#: Modules that decide WHETHER TO TRADE. None may consult a predictor.
SIGNAL_LAYER_MODULES = ["execution.scan", "execution.signals"]

#: Import strings that would indicate the wall has been breached.
FORBIDDEN = ["execution.bias", "from execution import bias", "import bias",
             "derive_bias", "load_bias", "directional_bias"]


@pytest.mark.parametrize("modname", SIGNAL_LAYER_MODULES)
def test_signal_modules_do_not_import_bias(modname):
    """A signal module must not be able to see the bias at all."""
    try:
        mod = __import__(modname, fromlist=["*"])
    except ImportError:
        pytest.skip(f"{modname} not present yet")
    src = inspect.getsource(mod)
    for token in FORBIDDEN:
        assert token not in src, (
            f"{modname} references {token!r}. This breaches the L1/L2 wall "
            f"(ARCHITECTURE.md:284-286). Do not relax this test — routing a "
            f"predictor with 23+ clean nulls into the signal path is a design "
            f"change that requires a prereg, not an import."
        )


def test_bias_module_does_not_import_signal_modules():
    """Nothing crosses L2 -> L1 either: 'the evaluator never tells the predictor
    what to think' (ARCHITECTURE.md:286)."""
    from execution import bias
    src = inspect.getsource(bias)
    for token in ["execution.signals", "passes_hyp107", "passes_hyp093"]:
        assert token not in src, f"bias.py references {token!r} — reverse leak"


def test_harness_does_not_gate_on_bias():
    """The execution layer may record a bias but must never filter on it."""
    from execution import harness
    src = inspect.getsource(harness)
    assert "execution.bias" not in src and "derive_bias" not in src, (
        "harness must not consult the bias when deciding fills")


def test_bias_is_scored_not_just_recorded():
    """A bias that is never checked against reality is an opinion, not a record."""
    from execution import bias
    assert hasattr(bias, "score_bias")
    assert hasattr(bias, "track_record")


def test_track_record_reports_sample_size():
    """A hit rate without its n is the failure mode this repo has been burned by."""
    from execution.bias import track_record
    rec = track_record()
    assert "n_scored" in rec
    assert "hit_rate" in rec
