"""Layer 5 — the ratified five, and nothing else.

DO NOT ADD GATES HERE. Daily-loss, consecutive-loss and VIX thresholds are not
legislated; Article 5 forbids override. See docs/proposed_amendment_art7-9.md.
"""
import pytest
import yaml
from pathlib import Path

from execution.risk import (Action, AccountState, CARRY_HEAT_CAP, DD_FLATTEN,
                            DD_HALT, DD_HALVE, PER_TRADE_CAP, check, constants,
                            is_carry, ladder_action)

ROOT = Path(__file__).resolve().parents[2]
_REVERT = ("RATIFIED CONSTITUTIONAL VALUE CHANGED. Article 5: no override; "
           "amendments require the MD and the YAML twin in the same commit. "
           "Revert — do not fix the test.")


def test_ratified_values():
    assert PER_TRADE_CAP == 0.0075, _REVERT
    assert CARRY_HEAT_CAP == 0.025, _REVERT
    assert (DD_HALVE, DD_HALT, DD_FLATTEN) == (0.035, 0.050, 0.065), _REVERT


def test_constitution_drift_against_yaml_twin():
    """Compared from the TEST side only. test_risk_constitution.py:167-174
    forbids live code from importing the twin."""
    y = yaml.safe_load((ROOT / "config" / "risk_constitution.yaml").read_text())
    flat = {}
    def walk(d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                walk(v)
            else:
                flat[k] = v
    walk(y)
    assert flat.get("hard_cap_frac") == PER_TRADE_CAP, _REVERT
    assert flat.get("carry_heat_cap_frac") == CARRY_HEAT_CAP, _REVERT


def test_ladder_ordering_invariant():
    """halve < halt < flatten < prop line. The draft's 8.5% flatten sat ABOVE the
    8% prop halt — 'a decorative emergency brake' (RISK_CONSTITUTION.md:36-41)."""
    assert DD_HALVE < DD_HALT < DD_FLATTEN < 0.08


@pytest.mark.parametrize("dd,expected,mult", [
    (0.000, Action.ALLOW, 1.0),
    (0.034, Action.ALLOW, 1.0),
    (0.035, Action.HALVE, 0.5),     # inclusive boundary
    (0.049, Action.HALVE, 0.5),
    (0.050, Action.BLOCK, 0.0),
    (0.064, Action.BLOCK, 0.0),
    (0.065, Action.FLATTEN, 0.0),
    (0.200, Action.FLATTEN, 0.0),
])
def test_ladder_table(dd, expected, mult):
    assert ladder_action(dd) == (expected, mult)


def test_drawdown_is_peak_to_trough():
    s = AccountState(equity=95_000, peak_equity=100_000)
    assert s.drawdown == pytest.approx(0.05)
    assert AccountState(equity=110_000, peak_equity=100_000).drawdown == 0.0


def test_per_trade_cap_blocks_oversize():
    s = AccountState(equity=100_000, peak_equity=100_000)
    d = check(symbol="AAPL", risk_fraction=0.01, state=s)
    assert not d.allowed
    assert any("ART1_PER_TRADE" in b for b in d.breached)


def test_per_trade_cap_boundary_allows_exact():
    s = AccountState(equity=100_000, peak_equity=100_000)
    assert check(symbol="AAPL", risk_fraction=PER_TRADE_CAP, state=s).allowed


def test_halve_can_bring_an_oversize_trade_under_the_cap():
    """At 3.5% DD sizing halves; 1.0% requested becomes 0.5% effective, which is
    inside Art. 1. The interaction must be evaluated on the EFFECTIVE size."""
    s = AccountState(equity=96_500, peak_equity=100_000)   # 3.5% DD
    d = check(symbol="AAPL", risk_fraction=0.01, state=s)
    assert d.action is Action.HALVE
    assert d.allowed
    assert d.detail["effective_risk"] == pytest.approx(0.005)


def test_halt_blocks_even_a_tiny_trade():
    s = AccountState(equity=95_000, peak_equity=100_000)   # 5% DD
    d = check(symbol="AAPL", risk_fraction=0.0001, state=s)
    assert not d.allowed and d.action is Action.BLOCK


def test_flatten_blocks_and_zeroes_size():
    s = AccountState(equity=93_500, peak_equity=100_000)   # 6.5% DD
    d = check(symbol="AAPL", risk_fraction=0.001, state=s)
    assert not d.allowed and d.action is Action.FLATTEN and d.size_mult == 0.0


def test_carry_heat_aggregate():
    s = AccountState(equity=100_000, peak_equity=100_000, open_carry_risk=0.024)
    d = check(symbol="EURUSD", risk_fraction=0.005, state=s)
    assert not d.allowed
    assert any("ART2_CARRY_HEAT" in b for b in d.breached)


def test_carry_cap_does_not_bind_non_carry():
    """Art. 2 legislates the carry complex only; nothing binds non-carry aggregate."""
    s = AccountState(equity=100_000, peak_equity=100_000, open_carry_risk=0.024)
    assert check(symbol="TSLA", risk_fraction=0.005, state=s).allowed


@pytest.mark.parametrize("sym,expected", [
    ("EURUSD", True), ("EUR_USD", True), ("USD_JPY", True), ("AUDNZD", True),
    ("TSLA", False), ("USDCAD", False), ("TGHL", False),
])
def test_carry_membership(sym, expected):
    assert is_carry(sym) is expected


def test_unlegislated_gates_are_absent():
    """The three gates without constitutional authority must not exist here."""
    import execution.risk as r
    src = Path(r.__file__).read_text()
    for banned in ("DAILY_LOSS_LIMIT", "CONSECUTIVE_LOSS", "VIX_THRESHOLD",
                   "VIX_SPIKE"):
        assert banned not in src, (
            f"{banned} has no constitutional authority. Article 5 forbids "
            f"override; propose an amendment instead.")


def test_decision_serialises_with_breaches():
    s = AccountState(equity=100_000, peak_equity=100_000)
    j = check(symbol="AAPL", risk_fraction=0.02, state=s).to_json()
    assert j["allowed"] is False and j["breached"] and j["reason"]


def test_constants_exposed_for_drift_check():
    assert constants()["hard_cap_frac"] == PER_TRADE_CAP
