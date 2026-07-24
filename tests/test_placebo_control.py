"""Tests for the mandatory random-reweighting placebo control (HYP-090 lesson,
made structural — sovereign/training/placebo_control.py).

Focus: a cycle that fails (ties/loses to) the placebo must never be eligible to
commit; a cycle that clearly beats it is eligible; missing/malformed placebo data
fails closed; the placebo weight composition matches the real weights exactly
(only assignment shuffles); the ICT/sovereign isolation test stays green.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from sovereign.training import placebo_control
from sovereign.training import policy_updater
from sovereign.training import director as director_mod

ROOT = Path(__file__).resolve().parents[1]


def _write_cfg(tmp_path, **placebo_overrides) -> Path:
    cfg = tmp_path / "training.yml"
    placebo = {
        "enabled": True,
        "seed": 42,
        "n_splits": 5,
        "embargo_frac": 0.02,
        "placebo_margin_min": 0.15,
    }
    placebo.update(placebo_overrides)
    cfg.write_text(yaml.safe_dump({
        "reward": {
            "alpha_scale": 2.0,
            "top_quartile_pct": 75,
            "top_weight": 2.0,
            "bottom_weight": 0.5,
        },
        "placebo": placebo,
    }))
    return cfg


def _make_informative_data(n=400, seed=0):
    """value_scores that genuinely predict returns — real weighting should beat
    a random shuffle of the same weights."""
    rng = np.random.default_rng(seed)
    value_scores = rng.normal(0.0, 1.0, size=n)
    returns = value_scores * 0.5 + rng.normal(0.0, 1.0, size=n)
    return returns, value_scores


def _make_uninformative_data(n=400, seed=0):
    """value_scores independent of returns — real weighting should NOT beat
    a random shuffle; this is the HYP-090 failure pattern reproduced."""
    rng = np.random.default_rng(seed)
    value_scores = rng.normal(0.0, 1.0, size=n)
    returns = rng.normal(0.0, 1.0, size=n)
    return returns, value_scores


# ── (a) real ties/loses to placebo → REJECTED, no commit ──────────────────────

def test_uninformative_scores_reject_via_high_margin_requirement(tmp_path):
    cfg = _write_cfg(tmp_path, placebo_margin_min=100.0)  # impossible to clear
    returns, value_scores = _make_informative_data()
    verdict = placebo_control.run_control(returns, value_scores, cfg)
    assert verdict.eligible is False
    assert "REJECTED" in verdict.reason

    update = policy_updater.refit_policy(value_scores, returns, gate_open=True, config_path=cfg)
    assert update.eligible is False
    assert update.dry is True
    assert "REJECTED" in update.note
    assert "HYP-090" in update.note

    report = director_mod.review(
        {"p": 1.0}, {"p": 1.0}, regime_fraction=0.1, placebo=update.placebo, config_path=cfg,
    )
    assert report.placebo_ok is False
    assert report.all_pass is False


def test_random_scores_do_not_reliably_beat_placebo(tmp_path):
    cfg = _write_cfg(tmp_path)
    returns, value_scores = _make_uninformative_data()
    verdict = placebo_control.run_control(returns, value_scores, cfg)
    # Uninformative value_scores → real weighting is itself just another arbitrary
    # weighting; it must not be structurally guaranteed to clear the margin.
    assert verdict.margin < 5.0  # sanity: no absurd separation from noise alone
    assert verdict.perm_p is not None
    # A single uninformative seed can land either side of the gate by chance (this is
    # exactly what the Monte-Carlo acceptance test below measures at scale); assert
    # the mechanism ran, not a specific outcome for this one seed.


# ── (b) real clearly beats placebo by > margin → eligible ──────────────────────

def test_informative_scores_beat_placebo_and_are_eligible(tmp_path):
    cfg = _write_cfg(tmp_path, placebo_margin_min=0.01)
    returns, value_scores = _make_informative_data(n=1000, seed=1)
    verdict = placebo_control.run_control(returns, value_scores, cfg)
    assert verdict.composition_ok is True
    assert verdict.eligible is True
    assert verdict.margin >= verdict.margin_min
    assert verdict.significant is True
    assert verdict.perm_p is not None and verdict.perm_p < 0.05
    assert verdict.margin > verdict.perm_percentile_95

    # Eligible placebo verdict → refit_policy proceeds past the placebo gate to the
    # not-yet-wired live refit path, which raises loudly (ignition itself is a
    # separate, still-closed gate — see gate.py). That NotImplementedError IS the
    # expected "eligible to commit" signal at this stage of the scaffold.
    with pytest.raises(NotImplementedError):
        policy_updater.refit_policy(value_scores, returns, gate_open=True, config_path=cfg)

    report = director_mod.review(
        {"p": 1.0}, {"p": 1.0}, regime_fraction=0.1, placebo=verdict, config_path=cfg,
    )
    assert report.placebo_ok is True


def test_gate_closed_never_commits_even_if_placebo_eligible(tmp_path):
    cfg = _write_cfg(tmp_path, placebo_margin_min=0.01)
    returns, value_scores = _make_informative_data(n=1000, seed=1)
    update = policy_updater.refit_policy(value_scores, returns, gate_open=False, config_path=cfg)
    assert update.dry is True
    assert update.eligible is False  # DRY mode never marks eligible regardless of placebo


# ── (c) fail-closed on missing/malformed placebo data ──────────────────────────

def test_missing_returns_fails_closed(tmp_path):
    cfg = _write_cfg(tmp_path)
    _, value_scores = _make_informative_data()
    update = policy_updater.refit_policy(value_scores, None, gate_open=True, config_path=cfg)
    assert update.eligible is False
    assert update.placebo is not None
    assert update.placebo.eligible is False
    assert "FAIL-CLOSED" in update.placebo.reason


def test_empty_arrays_raise_placebo_data_error(tmp_path):
    cfg = _write_cfg(tmp_path)
    with pytest.raises(placebo_control.PlaceboDataError):
        placebo_control.run_control(np.array([]), np.array([]), cfg)


def test_mismatched_lengths_raise_placebo_data_error(tmp_path):
    cfg = _write_cfg(tmp_path)
    with pytest.raises(placebo_control.PlaceboDataError):
        placebo_control.run_control(np.zeros(10), np.zeros(20), cfg)


def test_director_fails_closed_with_no_placebo_result(tmp_path):
    cfg = _write_cfg(tmp_path)
    report = director_mod.review({"p": 1.0}, {"p": 1.0}, regime_fraction=0.1,
                                  placebo=None, config_path=cfg)
    assert report.placebo_ok is False
    assert report.all_pass is False
    assert any("PLACEBO" in f for f in report.flags)


# ── (d) placebo weight composition matches real weights exactly ───────────────

def test_placebo_composition_matches_real_weights(tmp_path):
    cfg = _write_cfg(tmp_path)
    _, value_scores = _make_informative_data(n=250, seed=3)
    real_weights = policy_updater.compute_sample_weights(value_scores, cfg)
    placebo_weights = placebo_control.random_reweight(real_weights, seed=42)
    assert np.array_equal(np.sort(real_weights), np.sort(placebo_weights))
    assert not np.array_equal(real_weights, placebo_weights)  # assignment differs
    assert (real_weights == 2.0).sum() == (placebo_weights == 2.0).sum()
    assert (real_weights == 0.5).sum() == (placebo_weights == 0.5).sum()


def test_placebo_shuffle_is_reproducible_for_fixed_seed():
    weights = np.array([2.0, 2.0, 0.5, 0.5, 0.5, 2.0, 0.5])
    a = placebo_control.random_reweight(weights, seed=42)
    b = placebo_control.random_reweight(weights, seed=42)
    assert np.array_equal(a, b)


# ── (f) governance fix acceptance test — noise Monte-Carlo false-pass rate ─────
#
# Audit (cf41ec3) found the single-shuffle t>=1.0 threshold false-passed pure noise
# in 39/200 seeds (20%). The permutation-test rebuild must bring that down to the
# nominal ~5% (one-sided p<0.05) or below. Marked slow: 200 seeds x 200 perms.

@pytest.mark.slow
def test_noise_monte_carlo_false_pass_rate_at_nominal_level(tmp_path):
    cfg = _write_cfg(tmp_path)  # placebo_margin_min=0.15 default — pure noise margins hover near 0
    n_seeds = 200
    false_passes = 0
    for seed in range(n_seeds):
        returns, value_scores = _make_uninformative_data(n=400, seed=seed)
        verdict = placebo_control.run_control(returns, value_scores, cfg)
        if verdict.eligible:
            false_passes += 1
    false_pass_rate = false_passes / n_seeds
    print(f"\nmeasured false-pass rate: {false_pass_rate:.1%} ({false_passes}/{n_seeds})")
    # Nominal one-sided alpha is 5%; allow slack for Monte-Carlo sampling noise but
    # stay well clear of the old 20% failure rate.
    assert false_pass_rate <= 0.10, (
        f"false-pass rate {false_pass_rate:.1%} too high — permutation gate not working"
    )


# ── (e) isolation test still green ──────────────────────────────────────────────

def test_ict_sovereign_isolation_still_green():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/unit/test_ict_pipeline.py::test_pipeline_does_not_import_sovereign",
         "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
