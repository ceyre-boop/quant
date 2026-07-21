"""Regression tests for audit/claim_check.py.

The decisive test is `test_regression_corpus_2026_07_20`: the six false claims from
that day's two audit passes must come back REFUTED. If that test ever goes green
for the wrong reason, the tool would not have prevented what it was built for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit import claim_check as cc

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "audit" / "claims" / "regression_2026_07_20.json"


@pytest.fixture(scope="module")
def spec():
    s, _sha, _v = cc.load_spec()
    return s


def _run(kind: str, subject: str, spec, **kw) -> cc.Result:
    claim = cc.Claim(kind=cc.ClaimKind(kind), subject=subject, **kw)
    return cc._CHECKERS[claim.kind](claim, spec)


# ── isolation ────────────────────────────────────────────────────────────────

def test_imports_nothing_from_execution_path():
    """A claim about the execution path must be checked by something the
    execution path cannot influence."""
    assert cc.self_test() == 0


def test_spec_is_hashed_and_versioned():
    _spec, sha, version = cc.load_spec()
    assert len(sha) == 64
    assert version >= 1


# ── the six false claims of 2026-07-20 ───────────────────────────────────────

@pytest.mark.parametrize("module", [
    "sovereign/risk/black_scholes.py",
    "sovereign/risk/kalman_regime.py",
    "sovereign/risk/lqr_controller.py",
    "sovereign/risk/pegasus_policy_search.py",
])
def test_cs229_deadness_claim_is_refuted(module, spec):
    """'CS229 stack inert, safe to deprecate' — all are imported by the
    orchestrator. Acting on this would have broken it."""
    r = _run("DEAD", module, spec)
    assert r.verdict is cc.Verdict.REFUTED
    assert any("orchestrator.py" in e for e in r.evidence)


def test_ny_scanner_silent_crash_claim_is_refuted(spec):
    """'Silent crash since May, 0-byte logs' — the 0-byte file is the launchd
    log; the script writes its own, and it is large and current."""
    r = _run("LOGPATH", "com.clawd.ny_am_scanner", spec)
    assert r.verdict is cc.Verdict.REFUTED
    assert any("ny_scanner.log" in e for e in r.evidence)
    assert any("TIME GUARD" in e for e in r.evidence), \
        "must flag the time guard — removing it fires the scanner at 3am"


def test_absent_but_deliberately_removed_is_distinguished(spec):
    """pca_compressor really is gone, but it was removed on purpose. The claim
    is CONFIRMED, and the evidence must say so — otherwise someone rebuilds it."""
    r = _run("ABSENT", "sovereign/risk/pca_compressor.py", spec)
    assert r.verdict is cc.Verdict.CONFIRMED
    assert any("remov" in e.lower() for e in r.evidence)


def test_citation_of_missing_document_is_refuted(spec):
    """The RISK_FRAMEWORK.md failure mode: a constant justified by a document
    nobody wrote."""
    r = _run("CITED", "docs/NOT_A_REAL_DOCUMENT.md", spec, cites="ANY")
    assert r.verdict is cc.Verdict.REFUTED
    assert any("DOES NOT EXIST" in e for e in r.evidence)


def test_citation_of_absent_string_is_refuted(spec):
    """Document exists but does not say what it was cited for."""
    r = _run("CITED", "RISK_FRAMEWORK.md", spec, cites="NO_SUCH_CONSTANT_XYZ")
    assert r.verdict is cc.Verdict.REFUTED


# ── behaviour that keeps the tool honest ─────────────────────────────────────

def test_existing_path_refutes_absence_claim(spec):
    r = _run("ABSENT", "audit/claim_check.py", spec)
    assert r.verdict is cc.Verdict.REFUTED


def test_test_only_importers_do_not_prove_liveness(spec):
    """A module imported only by its own test is dead to the running system.
    Counting test imports as liveness would let dead code look live."""
    r = _run("DEAD", "sovereign/risk/black_scholes.py", spec)
    live_lines = [e for e in r.evidence if "LIVE importer" in e]
    assert live_lines, "orchestrator import must be reported as live"
    assert not any("tests/" in e and "LIVE" in e for e in r.evidence)


def test_ast_not_substring_matching():
    """Substring search matched module names inside their own docstrings twice
    this month. The import extractor must be AST-based."""
    src = '"""This docstring mentions sovereign.risk.black_scholes."""\nimport os\n'
    p = ROOT / "audit" / "_tmp_ast_probe.py"
    p.write_text(src)
    try:
        names = [n for n, _ln in cc._imports_in(p)]
        assert "os" in names
        assert not any("black_scholes" in n for n in names)
    finally:
        p.unlink(missing_ok=True)


def test_refuted_claim_sets_nonzero_exit():
    """The checker must be able to gate a routine."""
    rc = cc.main(["--claims", str(CORPUS)])
    assert rc == 1


def test_corpus_verdict_counts():
    """The corpus is the contract: every false claim refuted, every control
    confirmed, nothing silently unverifiable."""
    spec, _sha, _v = cc.load_spec()
    claims = [cc.Claim.from_dict(d) for d in json.loads(CORPUS.read_text())]
    results = cc.run(claims, spec)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
    assert counts.get("REFUTED", 0) == 4
    assert counts.get("CONFIRMED", 0) == 6
    assert counts.get("UNVERIFIABLE", 0) == 0
