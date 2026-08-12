"""Gate-zero: the prereg + ledger must be intact BEFORE any data work.
This is the regression test for the V2 ledger-gap failure class."""

import json

import pytest

from research.modern import _lib


def test_gate_zero_green_on_current_state():
    doc = _lib.gate_zero()
    assert doc["id"] == "HYP-090"
    assert doc["prior_expectation"] == "NOT_ROBUST"
    assert doc["grid"]["n_variants"] == 5775
    assert doc["hash_lock"] == _lib.canonical_hash(doc)


def test_gate_zero_fails_on_tampered_doc(tmp_path, monkeypatch):
    doc = json.loads(_lib.PREREG_PATH.read_text())
    doc["prior_expectation"] = "CONFIRMED"          # the tamper
    tampered = tmp_path / "prereg.json"
    tampered.write_text(json.dumps(doc))
    monkeypatch.setattr(_lib, "PREREG_PATH", tampered)
    with pytest.raises(SystemExit, match="hash mismatch"):
        _lib.gate_zero()


def test_gate_zero_fails_on_missing_prereg(tmp_path, monkeypatch):
    monkeypatch.setattr(_lib, "PREREG_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="prereg missing"):
        _lib.gate_zero()


def test_gate_zero_fails_on_missing_ledger_entry(tmp_path, monkeypatch):
    ledger = [e for e in json.loads(_lib.LEDGER_PATH.read_text()) if e.get("id") != "HYP-090"]
    fake = tmp_path / "ledger.json"
    fake.write_text(json.dumps(ledger))
    monkeypatch.setattr(_lib, "LEDGER_PATH", fake)
    with pytest.raises(SystemExit, match="missing from hypothesis ledger"):
        _lib.gate_zero()


def test_block_bootstrap_basics():
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(0.0002, 0.01, 2000)
    better = base + 0.0012                           # clearly higher Sharpe, same noise
    res = _lib.block_bootstrap_sharpe_diff_p(better, base, n_boot=500, seed=1)
    assert res["p_one_sided"] < 0.05
    same = _lib.block_bootstrap_sharpe_diff_p(base, base.copy(), n_boot=200, seed=1)
    assert same["p_one_sided"] > 0.4                 # no self-difference
