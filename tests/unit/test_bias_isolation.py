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
import ast
import inspect

import pytest


#: Modules that decide WHETHER TO TRADE. None may consult a predictor.
SIGNAL_LAYER_MODULES = ["execution.scan", "execution.signals"]


def _imported_modules(mod) -> set[str]:
    """Every module name this module actually imports, via AST.

    Deliberately NOT a substring scan of the source. The first version of this
    test substring-matched 'execution.bias' and failed on signals.py's own
    docstring saying it must not import execution.bias — a mention is not a
    reference. Parsing imports makes the guard both correct and stricter:
    documentation can name the wall freely, while a real import cannot hide
    behind an alias or an odd import form.
    """
    tree = ast.parse(inspect.getsource(mod))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def _called_names(mod) -> set[str]:
    """Function/attribute names actually invoked (catches a lazy import)."""
    tree = ast.parse(inspect.getsource(mod))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


@pytest.mark.parametrize("modname", SIGNAL_LAYER_MODULES)
def test_signal_modules_do_not_import_bias(modname):
    """A signal module must not be able to see the bias at all."""
    try:
        mod = __import__(modname, fromlist=["*"])
    except ImportError:
        pytest.skip(f"{modname} not present yet")

    imports = _imported_modules(mod)
    offending = {n for n in imports if n.split(".")[-1] == "bias" or "bias" in n.split(".")}
    assert not offending, (
        f"{modname} imports {offending}. This breaches the L1/L2 wall "
        f"(ARCHITECTURE.md:284-286). Do not relax this test — routing a predictor "
        f"with 23+ clean nulls into the signal path is a design change requiring a "
        f"prereg, not an import."
    )

    calls = _called_names(mod)
    for banned in ("derive_bias", "load_bias", "score_bias"):
        assert banned not in calls, f"{modname} calls {banned}() — wall breached"


def test_bias_module_does_not_import_signal_modules():
    """Nothing crosses L2 -> L1 either: 'the evaluator never tells the predictor
    what to think' (ARCHITECTURE.md:286)."""
    from execution import bias
    imports = _imported_modules(bias)
    assert "execution.signals" not in imports, "bias.py imports signals — reverse leak"
    calls = _called_names(bias)
    for banned in ("passes_hyp107", "passes_hyp093", "build_signals"):
        assert banned not in calls, f"bias.py calls {banned}() — reverse leak"


def test_harness_does_not_gate_on_bias():
    """The execution layer may record a bias but must never filter on it."""
    from execution import harness
    imports = _imported_modules(harness)
    assert not any(n.split(".")[-1] == "bias" for n in imports), (
        "harness imports bias — it must not consult a predictor when deciding fills")


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
