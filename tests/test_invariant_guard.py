"""Tests for the adversarial invariant guard (Layer-4 integrity check).

Covers each invariant's fire/quiet behavior, the gate, the independence cross-check
against the audited code's own probe heuristic, and the read-only guarantee.
"""
from __future__ import annotations

import ast
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audit import invariant_guard as ig

ROOT = Path(__file__).resolve().parents[1]
SPEC, _SHA, _V = ig.load_spec()


def _recent() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()


def _rec(pair, outcome="WIN", ts=None, source="live", test_fill=False,
         entry=1.10, stop=1.09, r=1.0):
    return {"pair": pair, "outcome": outcome, "entry_timestamp": ts or _recent(),
            "source": source, "test_fill": test_fill, "entry_level": entry,
            "stop_loss": stop, "r_realized": r}


def _fill(pair, units=100000, entry=1.10, stop=1.09, ts=None):
    return {"pair": pair, "units": units, "fill_price": entry, "stop_price": stop,
            "timestamp": ts or _recent()}


def _eval(records=None, fills=None, present=("data/ledger/oanda_fills.jsonl",), stale=()):
    return ig.evaluate(records or [], fills or [], list(present), list(stale), SPEC)


# ── I1 — Oracle reflection purity ─────────────────────────────────────────────

def test_i1_flags_forbidden_pair_with_outcome():
    f = _eval(records=[_rec("USD_CAD", outcome="LOSS", r=-1.0)])
    assert len(f.i1) == 1
    assert "forbidden" in f.i1[0]["reasons"][0]


def test_i1_flags_probe_record():
    f = _eval(records=[_rec("EUR_USD", test_fill=True)])
    assert len(f.i1) == 1


def test_i1_flags_insane_risk_levels():
    # stop >50% of entry away = proof-of-life sentinel (a real FX stop is a fraction of a %)
    f = _eval(records=[_rec("EUR_USD", entry=1.10, stop=0.40)])
    assert len(f.i1) == 1


def test_i1_ignores_genuine_allowed_trade():
    f = _eval(records=[_rec("USD_JPY", outcome="WIN", entry=150.0, stop=149.5)])
    assert f.i1 == []


def test_i1_ignores_open_records():
    # forbidden pair but never filled → not in the reflection window → not I1
    f = _eval(records=[_rec("USD_CAD", outcome="OPEN")])
    assert f.i1 == []


def test_i1_ignores_old_records():
    f = _eval(records=[_rec("USD_CAD", outcome="LOSS", ts=_old())])
    assert f.i1 == []


# ── I2 — rogue / unlogged OANDA writes ────────────────────────────────────────

def test_i2_flags_forbidden_fill():
    f = _eval(fills=[_fill("USD_CAD")])
    assert len(f.i2) == 1


def test_i2_flags_one_unit_sentinel():
    f = _eval(fills=[_fill("EUR_USD", units=1, entry=1.10, stop=1.0)])
    assert len(f.i2) == 1


def test_i2_ignores_normal_fill():
    f = _eval(fills=[_fill("EUR_USD", units=100000, entry=1.10, stop=1.099)])
    assert f.i2 == []


def test_i2_stale_ledger_escalates_soft():
    f = _eval(fills=[], present=("data/ledger/oanda_fills.jsonl",),
              stale=("data/ledger/oanda_fills.jsonl",))
    events = ig.build_events(f, SPEC)
    assert any(t == "FILLS_STALE" for _, t, _ in events)


def test_no_fills_ledger_escalates_soft():
    f = _eval(fills=[], present=(), stale=())
    events = ig.build_events(f, SPEC)
    assert any(t == "NO_FILLS_LEDGER" for _, t, _ in events)


# ── I3 — forbidden-pair guard (broad) ─────────────────────────────────────────

def test_i3_flags_forbidden_even_when_open():
    # OPEN forbidden record is invisible to I1 but I3 must still catch it
    f = _eval(records=[_rec("AUD_NZD", outcome="OPEN")])
    assert f.i1 == []
    assert len(f.i3) == 1


def test_i3_flags_forbidden_fill():
    f = _eval(fills=[_fill("USD_CAD")])
    assert len(f.i3) == 1


# ── gate + soft signals ───────────────────────────────────────────────────────

def test_gate_pass_on_clean_data():
    f = _eval(records=[_rec("EUR_USD"), _rec("GBP_USD")],
              fills=[_fill("USD_JPY", entry=150.0, stop=149.5)])
    assert ig.overall(f, SPEC) == "PASS"
    assert not ig.build_events(f, SPEC)  # no URGENT/IMPORTANT


def test_gate_fail_on_contamination():
    f = _eval(records=[_rec("USD_CAD", outcome="LOSS")])
    assert ig.overall(f, SPEC) == "FAIL"
    events = ig.build_events(f, SPEC)
    assert any(p == "URGENT" and t == "I1_ORACLE_CONTAMINATION" for p, t, _ in events)


def test_unknown_pair_is_soft_not_hard():
    # a plausible-but-not-allowed pair: reviewed, not a hard fence failure
    f = _eval(records=[_rec("NZD_USD")])
    assert f.i1 == []
    assert ig.overall(f, SPEC) == "PASS"
    assert any(t == "UNKNOWN_PAIR" for _, t, _ in ig.build_events(f, SPEC))


# ── independence cross-check (spec promise) ───────────────────────────────────

def test_insane_risk_matches_canonical():
    """The guard reimplements the probe heuristic independently; verify it agrees
    with scripts/backfill_decision_records.py::_is_test_fill on the known cases."""
    spec = importlib.util.spec_from_file_location(
        "_bf", ROOT / "scripts" / "backfill_decision_records.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    frac = float(SPEC["insane_risk_fraction"])
    cases = [
        {"fill_price": 1.10, "stop_price": 0.40},   # >50% away → test
        {"fill_price": 1.10, "stop_price": 1.099},  # normal → not test
        {"fill_price": 0.0, "stop_price": 0.0},     # non-positive → test
        {"fill_price": 150.0, "stop_price": 149.5}, # normal JPY → not test
    ]
    for c in cases:
        assert bf._is_test_fill(c) == ig._insane_risk(c["fill_price"], c["stop_price"], frac), c


# ── read-only guarantee ───────────────────────────────────────────────────────

def test_invariant_guard_does_not_import_execution():
    """Mirror of test_pipeline_does_not_import_sovereign: the adversarial guard must
    not import the execution path it audits (would give it write reach + shared blind spots)."""
    src = (ROOT / "audit" / "invariant_guard.py").read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    banned = ("sovereign.execution", "oanda_bridge", "forex_exit_manager", "tradovate_bridge")
    offenders = [m for m in imported for b in banned if b in m]
    assert not offenders, f"guard imports execution path: {offenders}"


def test_spec_single_fence_and_hashed():
    spec, sha, ver = ig.load_spec()
    assert ver >= 1 and len(sha) == 64
    assert spec["i1_contaminated_allowed"] == 0
