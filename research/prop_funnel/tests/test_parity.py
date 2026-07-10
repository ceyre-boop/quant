"""Parity gate tests (TICK-022 P1). These are the Phase-1 GATE: engine code is
not trusted until all three recorded artifacts reproduce.

Marked slow — full recorded trial counts (5k/10k/10k). Run explicitly:
    python3 -m pytest research/prop_funnel/tests/test_parity.py -q
"""

import pytest

from research.prop_funnel import parity


@pytest.fixture(scope="module")
def report():
    return parity.run_all_parity()


def _result(report, name):
    return next(r for r in report["results"] if r["name"] == name)


def test_parity_1_ict_mff_exact(report):
    r = _result(report, "parity_1_ict_mff")
    assert r["ok"], r
    assert r["exact"], f"parity 1 is random.Random-seeded and must be EXACT: {r}"


def test_parity_2_ftmo(report):
    r = _result(report, "parity_2_ftmo")
    assert r["ok"], r


def test_parity_3_carry_bootstrap(report):
    r = _result(report, "parity_3_carry_bootstrap")
    assert r["ok"], r


def test_parity_3_did_not_touch_live_output():
    from research.prop_funnel._lib import ROOT
    live = ROOT / "data" / "risk" / "prop_monte_carlo.json"
    import json
    recorded = json.loads(live.read_text())
    # The live artifact must still be the recorded one (generated before this ticket).
    assert recorded["generated_at"] < "2026-07-10", (
        "live data/risk/prop_monte_carlo.json was regenerated during research — write-safety violated")
