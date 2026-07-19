"""Drift tripwire: alert-only, with honest power reporting."""
import math
import pytest

from execution import drift
from execution.drift import (BASELINE, assess, detectable_below, min_n_for_drop,
                             sigma_at)


def test_never_modifies_thresholds():
    """The whole point. HYP-090: adaptive lost to random selection AND to doing nothing."""
    import inspect
    src = inspect.getsource(drift)
    for banned in ("og_max =", "logvol_max =", "FROZEN[", "write_frozen", "update_threshold"):
        assert banned not in src, f"drift.py must never write a threshold ({banned})"


@pytest.mark.parametrize("n,expected_floor", [
    (20, 0.495), (50, 0.570), (100, 0.608), (200, 0.635),
])
def test_documented_power_table_is_accurate(n, expected_floor):
    """The numbers in the module docstring must match the arithmetic."""
    assert detectable_below(n) == pytest.approx(expected_floor, abs=0.001)


def test_alert_at_n20_requires_collapse_not_drift():
    """At n=20 a 60% win rate — a 10-point degradation — does NOT alert."""
    returns = [1.0] * 12 + [-1.0] * 8        # 60%
    rep = assess(returns)
    assert rep.n == 20
    assert rep.live_win_rate == pytest.approx(0.60)
    assert rep.alert is False, "10-point drop is invisible at n=20"
    assert "only fires below" in rep.note


def test_alert_fires_on_genuine_collapse():
    returns = [1.0] * 8 + [-1.0] * 12        # 40% at n=20
    rep = assess(returns)
    assert rep.alert is True
    assert rep.z_score < -2.0


def test_both_sample_size_questions_reported():
    """Two different questions, easy to conflate, so both must be surfaced.

    (a) observed 60% first trips the alert at n=84 (ignores sampling noise)
    (b) a TRUE 60% rate is reliably caught at n~177 (80% power)

    Quoting only (a) overstates the instrument; an earlier draft did exactly that.
    """
    from execution.drift import n_for_power
    rep = assess([1.0] * 10)
    assert rep.min_n_for_10pt_drop == min_n_for_drop(0.10) == 84
    assert rep.n_for_80pct_power == n_for_power(0.60) == 177
    assert rep.n_for_80pct_power > rep.min_n_for_10pt_drop, (
        "power requirement must exceed the naive threshold-crossing n")
    assert str(rep.min_n_for_10pt_drop) in rep.note
    assert str(rep.n_for_80pct_power) in rep.note


def test_power_requires_more_than_threshold_crossing():
    from execution.drift import n_for_power
    assert n_for_power(0.60) > min_n_for_drop(0.10)
    assert n_for_power(0.60, power=0.90) > n_for_power(0.60, power=0.80)
    assert n_for_power(0.75) == -1, "no sample detects an IMPROVEMENT as drift"


def test_zero_data_does_not_imply_health():
    rep = assess([])
    assert rep.alert is False
    assert rep.n == 0
    assert "not evidence of health" in rep.note


def test_within_2sigma_is_stated_as_weak():
    """Silence must never read as an all-clear."""
    rep = assess([1.0] * 14 + [-1.0] * 6)    # exactly baseline
    assert rep.alert is False
    assert "weak statement" in rep.note


def test_sigma_matches_binomial_formula():
    p0 = BASELINE["win_rate"]
    assert sigma_at(50) == pytest.approx(math.sqrt(p0 * (1 - p0) / 50))
    assert sigma_at(0) is None


def test_skips_and_missing_returns_excluded(tmp_path):
    import json
    p = tmp_path / "fill_log.jsonl"
    rows = [
        {"hypothesis": "HYP-107", "signal_type": "LONG", "net_return": 0.05},
        {"hypothesis": "HYP-107", "signal_type": "SKIP_HALT", "net_return": None},
        {"hypothesis": "HYP-093", "signal_type": "SHORT", "net_return": 0.02},
        {"hypothesis": "HYP-107", "signal_type": "LONG", "net_return": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert drift.load_outcomes(tmp_path, "HYP-107") == [0.05]
