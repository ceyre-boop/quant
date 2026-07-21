"""Regression tests for audit/system_inventory.py.

The inventory's whole value is that its labels are earned, not asserted. These pin
the facts today's audits got wrong (CS229 'inert', pca_compressor 'depended-on') and
the count reconciliation that guarantees no file is silently dropped.
"""
from __future__ import annotations

import pytest

from audit import system_inventory as si


@pytest.fixture(scope="module")
def inv():
    rows, meta = si.build()
    return {r.path: r for r in rows}, meta


def test_isolation():
    assert si.self_test() == 0


def test_counts_reconcile(inv):
    rows, meta = inv
    assert meta["n_files"] == len(rows)
    assert sum(meta["counts"].values()) == len(rows), "every file classified exactly once"


def test_harness_is_live_firing(inv):
    rows, _ = inv
    r = rows["execution/harness.py"]
    assert r.status == "LIVE"
    assert r.sub_status == "FIRING", "harness plist writes a recent log"


@pytest.mark.parametrize("mod", [
    "sovereign/risk/black_scholes.py",
    "sovereign/risk/kalman_regime.py",
])
def test_cs229_modules_are_not_retired(inv, mod):
    """The 2026-07-20 false-death regression: LOW-USE header must NOT retire a
    module the orchestrator imports."""
    rows, _ = inv
    assert rows[mod].status != "RETIRED"
    assert rows[mod].status == "LIVE"


def test_pca_compressor_is_absent(inv):
    """Deliberately removed — must not appear as a row at all."""
    rows, _ = inv
    assert "sovereign/risk/pca_compressor.py" not in rows


def test_module_run_via_dash_m_is_a_live_root(inv):
    """The harness runs `python3 -m execution.harness`, not a .py path. A plist
    parser that only matched .py suffixes missed it and mislabelled it TEST-ONLY."""
    rows, _ = inv
    assert rows["execution/harness.py"].status == "LIVE"


def test_ondemand_tool_not_mislabelled_testonly(inv):
    """A backtester run by hand is ON-DEMAND, not TEST-ONLY."""
    rows, _ = inv
    assert rows["backtester/engine.py"].status == "ON-DEMAND"


def test_retired_by_location(inv):
    rows, _ = inv
    attic = [r for p, r in rows.items() if p.startswith("attic/")]
    assert attic and all(r.status == "RETIRED" for r in attic)


def test_low_use_marker_does_not_retire():
    """LOW-USE means 'live but minor', the opposite of retired. Guards the exact
    bug that mislabelled all 11 CS229 modules on the first run."""
    assert "# LOW-USE" not in si.RETIRED_MARKERS


def test_undocumented_flagged_not_invented(inv):
    """Files without a docstring must be marked, never silently described."""
    rows, _ = inv
    undoc = [r for r in rows.values() if r.provenance == "undocumented"]
    assert all(r.description.startswith("(UNDOCUMENTED") for r in undoc)
