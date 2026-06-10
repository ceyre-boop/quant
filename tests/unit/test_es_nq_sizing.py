"""Adaptive session ladder unit tests."""
import pytest

from sovereign.es_nq.session_sizing import SessionLadder


def make_ladder(flat=None):
    return SessionLadder(account_usd=50000.0, flat_risk_pct=flat)


def test_probe_press_runner_sequence():
    lad = make_ladder()
    assert lad.next_role() == "probe"
    lad.record(1.0, 250.0)                      # probe wins
    assert lad.next_role() == "press"
    lad.record(1.5, 750.0)                      # press wins
    assert lad.next_role() == "runner"
    lad.record(2.5, 625.0)
    assert lad.next_role() is None              # max 3 trades


def test_probe_loss_leads_to_pullback_then_done():
    lad = make_ladder()
    lad.record(-1.0, -250.0)                    # probe stopped
    assert lad.next_role() == "pullback"
    lad.record(1.0, 125.0)                      # pullback wins — but trade1 lost
    assert lad.next_role() is None              # runner needs trades 1 AND 2 won


def test_risk_percentages():
    lad = make_ladder()
    assert lad.risk_pct("probe") == 0.005
    assert lad.risk_pct("press") == 0.010
    assert lad.risk_pct("pullback") == 0.0025
    assert lad.risk_pct("runner") == 0.005


def test_flat_sizing_overrides_ladder():
    lad = make_ladder(flat=0.005)
    for role in ("probe", "press", "pullback", "runner"):
        assert lad.risk_pct(role) == 0.005


def test_contracts_floor_math():
    lad = make_ladder()
    # probe: 0.5% of 50k = $250 risk; stop 25 pts × $2/pt = $50/contract → 5 contracts
    assert lad.contracts("probe", 25.0, "MNQ") == 5
    # huge stop → 0 contracts = skip
    assert lad.contracts("probe", 200.0, "MNQ") == 0
    with pytest.raises(ValueError):
        lad.contracts("probe", 0.0, "MNQ")


def test_daily_loss_cap_halts():
    lad = make_ladder()
    lad.record(-1.0, -400.0)
    assert not lad.halted()
    lad.record(-1.0, -400.0)                    # −800 > −750 cap → halted
    assert lad.halted()
    assert lad.next_role() is None
