"""Validation harness tests — permutation sanity on planted signal vs noise,
ladder permutation helper, holdout sentinel refusal."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "validate_es_nq_system", ROOT / "scripts" / "validate_es_nq_system.py")
vh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vh)


def _make_calls_outcomes(n, accuracy, seed=11):
    rng = np.random.RandomState(seed)
    outcomes = rng.choice(["UP", "DOWN"], size=n)
    calls = outcomes.copy()
    flip = rng.rand(n) > accuracy
    calls[flip] = np.where(calls[flip] == "UP", "DOWN", "UP")
    return calls, outcomes


def test_planted_signal_yields_small_p():
    calls, outcomes = _make_calls_outcomes(400, accuracy=0.62)
    p, null = vh.label_permutation_pvalue(calls, outcomes, 2000, 7)
    assert p < 0.01
    assert abs(null.mean() - 0.5) < 0.02      # null centers at chance


def test_pure_noise_yields_large_p():
    rng = np.random.RandomState(3)
    ps = []
    for seed in range(5):
        calls = rng.choice(["UP", "DOWN"], size=400)
        outcomes = rng.choice(["UP", "DOWN"], size=400)
        p, _ = vh.label_permutation_pvalue(calls, outcomes, 1000, seed)
        ps.append(p)
    assert np.mean(ps) > 0.2                  # no systematic false positives


def test_circular_shift_pvalue_runs():
    calls, outcomes = _make_calls_outcomes(300, accuracy=0.60)
    p = vh.circular_shift_pvalue(calls, outcomes, 500, 7)
    assert 0.0 <= p <= 1.0
    assert p < 0.10                            # planted signal still detected


def test_ann_sharpe_basics():
    assert vh._ann_sharpe(np.array([])) == 0.0
    assert vh._ann_sharpe(np.array([0.01, 0.01])) == 0.0   # zero std
    up = np.array([0.01, 0.012, 0.008, 0.011, 0.009] * 20)
    assert vh._ann_sharpe(up) > 5.0


def test_ladder_usd_orders_matter():
    """probe-win-first unlocks press; loss-first shrinks trade 2 — order changes P&L."""
    p = vh.es_nq_params()
    win = (1.5, 20.0, 60.0)     # r, stop_pts, usd/contract
    loss = (-1.0, 20.0, -40.0)
    a = vh._ladder_usd([win, loss], p, "MNQ", flat=None)
    b = vh._ladder_usd([loss, win], p, "MNQ", flat=None)
    assert a != b
    # flat sizing is order-independent
    fa = vh._ladder_usd([win, loss], p, "MNQ", flat=0.005)
    fb = vh._ladder_usd([loss, win], p, "MNQ", flat=0.005)
    assert fa == pytest.approx(fb)


def test_holdout_sentinel_refusal(tmp_path, monkeypatch):
    sentinel = tmp_path / ".es_nq_holdout_done"
    sentinel.write_text(json.dumps({"run_at": "2026-01-01"}))
    monkeypatch.setattr(vh, "HOLDOUT_SENTINEL", sentinel)
    monkeypatch.setattr(vh, "load_validation", lambda: {
        "stage1": {"verdict": "VALID_EDGE"},
        "stage2": {"verdict": "VALID_EDGE", "gate_kept": True},
        "stage3": {"verdict": "VALID_EDGE", "adaptive_adopted": True},
    })
    with pytest.raises(SystemExit, match="HOLDOUT ALREADY RUN"):
        vh.stage4(10, 7)


def test_stage_requires_prior_verdict():
    with pytest.raises(SystemExit, match="no recorded verdict"):
        vh.require_prior_stage(2, {})


def test_ledger_duplicate_refusal(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"id": "ESNQ-BIAS-01"}]))
    monkeypatch.setattr(vh, "LEDGER_PATH", ledger)
    with pytest.raises(SystemExit, match="already has"):
        vh.append_ledger({"id": "ESNQ-BIAS-01"})
