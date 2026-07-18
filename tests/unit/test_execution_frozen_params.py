"""Pre-registration lock for the execution harness frozen parameters.

Pattern copied from tests/unit/test_es_nq_isolation.py:40-47.

DO NOT "FIX" THIS TEST. If it fails, a pre-registered threshold was edited.
Revert the edit. If the change is genuinely intended, it requires a
data/agent/param_change_log.jsonl entry BEFORE the value moves, per CLAUDE.md
NON-NEGOTIABLE #4 — and a new prereg, since these values seal published verdicts.
"""
import pytest

from execution.config import (FROZEN, FROZEN_HASH, FrozenConfigError,
                              compute_hash, frozen, verify_frozen_hash)

_REVERT = ("PRE-REGISTERED THRESHOLD CHANGED — this invalidates the sealed "
           "verdict for this hypothesis. Revert the config; do not fix the test.")


def test_hyp107_thresholds_frozen():
    """HYP-107, frozen at commit 48303cd (research/gapper/hyp107_shadow.py:38-39)."""
    c = frozen("hyp107")
    assert c["og_max"] == 0.577, _REVERT
    assert c["logvol_max"] == 5.854, _REVERT
    assert c["gap_floor"] == 0.30, _REVERT
    assert c["stop_pct"] == 0.25, _REVERT


def test_hyp093_thresholds_frozen():
    """HYP-093, prereg c5b10616 (research/yield_frontier/live_shadow.py:34-37).

    Note gain_min is 0.50, NOT 1.00. Task briefs have described this leg as
    ">= 100% above prior close"; that is not the sealed spec and would produce a
    different event set entirely.
    """
    c = frozen("hyp093")
    assert c["gain_min"] == 0.50, _REVERT
    assert c["qual_gain"] == 1.30, _REVERT
    assert c["price_min"] == 2.00, _REVERT
    assert c["vol_min"] == 500_000, _REVERT
    assert c["stop_mult"] == 1.30, _REVERT


def test_movers_thresholds_differ_by_leg_deliberately():
    """The two legs screen at different thresholds. This fork is intentional."""
    assert frozen("hyp107")["movers_pct_change_min"] == 30.0, _REVERT
    assert frozen("hyp093")["movers_pct_change_min"] == 40.0, _REVERT


def test_sip_lag_beyond_measured_boundary():
    """Deferred capture must sit outside the measured 15-minute SIP window."""
    assert frozen("capture")["sip_lag_minutes"] > 15


def test_frozen_hash_matches():
    assert compute_hash() == FROZEN_HASH, _REVERT
    verify_frozen_hash()


def test_drift_is_detected():
    """A mutated block must not silently pass the hash gate."""
    mutated = {k: dict(v) if isinstance(v, dict) else v for k, v in FROZEN.items()}
    mutated["hyp107"] = dict(mutated["hyp107"])
    mutated["hyp107"]["og_max"] = 0.6
    assert compute_hash(mutated) != FROZEN_HASH


def test_frozen_returns_copy():
    """Callers must not be able to mutate the frozen block through frozen()."""
    c = frozen("hyp107")
    c["og_max"] = 999.0
    assert frozen("hyp107")["og_max"] == 0.577
    verify_frozen_hash()


def test_unknown_leg_raises():
    with pytest.raises(KeyError):
        frozen("hyp999")
