"""Tests for the self-play training scaffold (spec: research/SELF_PLAY_TRAINING_ARCHITECTURE.md).

Focus: the safety machinery — ignition gate refusal, the net-return hard guard, and
the director's ±20% magnitude check. These prove the board is physically incapable
of igniting a real cycle before TICK-024 + HYP-071-net-CONFIRMED.
"""
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from sovereign.training import gate as gate_mod
from sovereign.training import director as director_mod
from sovereign.training import policy_updater
from sovereign.training.value_scorer import (
    ValueScorer, trade_score, GrossReturnError,
)

ROOT = Path(__file__).resolve().parents[1]

# Mirrors the real killed-verdict ledger entry (data/agent/hypothesis_ledger.json,
# id=HYP-071, status=METRIC_ARTIFACT) plus the HYP-071-GOVFLAG record — the "current
# real state" fixture the revival guard must stay CLOSED against.
_KILLED_LEDGER = [
    {
        "id": "HYP-071",
        "status": "METRIC_ARTIFACT",
        "date_tested": "2026-06-30",
        "result": {
            "prereg_hash": "3d500bda3249c4615698ce311a7cbad41a35600a23abd2a4ea4526416eac06a4",
            "addendum_hash": "c1fab80730f1ebf3af7c35e4bbd8fc80e2bafd86419fc0125acc140b414d806f",
        },
    },
    {
        "id": "HYP-071-GOVFLAG",
        "status": "governance_flag",
        "date": "2026-07-24",
    },
]


def _write_ledger(tmp_path, entries) -> Path:
    ledger = tmp_path / "hypothesis_ledger.json"
    ledger.write_text(json.dumps(entries))
    return ledger


def _write_cfg(tmp_path, *, tick=False, hyp=False, board_gross=True,
                ledger_entries=None) -> Path:
    """Build a self-contained training config + a stub value board in tmp_path.

    `ledger_entries` defaults to the real killed-verdict fixture (_KILLED_LEDGER)
    so tests exercise the HYP-071 REVIVAL GUARD against the actual adjudicated
    state unless a test explicitly overrides it.
    """
    board = tmp_path / "board.json"
    summary = {"gross_R_caveat": "gross"} if board_gross else {"net_confirmed": True}
    board.write_text('{"cells": {}, "summary": %s}' % (
        '{"gross_R_caveat": "gross"}' if board_gross else '{}'))
    ledger_path = _write_ledger(
        tmp_path, _KILLED_LEDGER if ledger_entries is None else ledger_entries)
    cfg = tmp_path / "training.yml"
    cfg.write_text(textwrap.dedent(f"""
        ignition:
          tick_024_carry_fix_landed: {str(tick).lower()}
          hyp_071_net_confirmed: {str(hyp).lower()}
          hypothesis_ledger_path: {ledger_path}
        value_function:
          board_path: {board.relative_to(ROOT) if board.is_relative_to(ROOT) else board}
          gross_marker_key: gross_R_caveat
        reward:
          alpha_scale: 2.0
          top_quartile_pct: 75
          top_weight: 2.0
          bottom_weight: 0.5
        rollout:
          pairs: [GBPUSD, EURUSD, AUDUSD, GBPJPY]
        director:
          max_param_change_pct: 20.0
          auto_approve: false
        policy_params:
          conviction_entry_min: 0.62
        paths:
          checkpoint_dir: {tmp_path}/ckpt
          training_log: {tmp_path}/log.jsonl
    """))
    # value_function.board_path is resolved relative to ROOT by the modules; point
    # it at an absolute path by rewriting after templating.
    data = yaml.safe_load(cfg.read_text())
    data["value_function"]["board_path"] = str(board.relative_to(ROOT)) if board.is_relative_to(ROOT) else str(board)
    cfg.write_text(yaml.safe_dump(data))
    return cfg, board


# ── Ignition gate ─────────────────────────────────────────────────────────────

def test_gate_closed_when_both_blockers_unmet(tmp_path):
    cfg, _ = _write_cfg(tmp_path, tick=False, hyp=False, board_gross=True)
    # board_path is relative-to-ROOT logic won't find tmp board; patch to absolute.
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is False
    assert status.mode == "SCAFFOLD/DRY"
    assert any("8.1" in r for r in status.reasons)
    assert any("8.2" in r for r in status.reasons)


def test_gate_stays_closed_on_gross_board_even_if_flags_set(tmp_path):
    """Even with both ignition flags TRUE, a gross board keeps the gate closed."""
    cfg, _ = _write_cfg(tmp_path, tick=True, hyp=True, board_gross=True)
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is False
    assert any("NET-RETURN GUARD" in r for r in status.reasons)


def test_gate_stays_closed_with_real_flags_but_no_ledger_revival(tmp_path):
    """Even with both ignition flags TRUE and a net board, the killed-verdict
    ledger fixture (METRIC_ARTIFACT + GOVFLAG, no fresh prereg) holds the gate
    CLOSED — this is the exact scenario HYP-071-GOVFLAG warns about."""
    cfg, _ = _write_cfg(tmp_path, tick=True, hyp=True, board_gross=False)
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is False
    assert any("HYP-071 REVIVAL GUARD" in r for r in status.reasons)
    assert "fresh prereg + adjudication required" in status.reasons[0] or \
        any("fresh prereg + adjudication required" in r for r in status.reasons)


def test_gate_opens_only_when_all_conditions_met(tmp_path):
    fresh_entries = _KILLED_LEDGER + [{
        "id": "HYP-071",
        "status": "CONFIRMED",
        "date_tested": "2026-08-01",
        "result": {"prereg_hash": "f" * 64},
    }]
    cfg, _ = _write_cfg(tmp_path, tick=True, hyp=True, board_gross=False,
                         ledger_entries=fresh_entries)
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is True
    assert status.mode == "LIVE"


# ── HYP-071 revival guard ────────────────────────────────────────────────────────

def test_revival_guard_rejects_reused_locked_prereg_hash(tmp_path):
    """A recompute that cites one of the original hash-locked prereg files (even
    if it were marked CONFIRMED) must NOT open the gate — reuse is not revival."""
    entries = [{
        "id": "HYP-071",
        "status": "CONFIRMED",
        "date_tested": "2026-08-01",
        "result": {"prereg_hash": gate_mod.HYP071_LOCKED_PREREG_HASHES.__iter__().__next__()},
    }]
    cfg, _ = _write_cfg(tmp_path, tick=True, hyp=True, board_gross=False,
                         ledger_entries=entries)
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is False
    assert any("reuses the" in r and "locked" in r for r in status.reasons)


def test_revival_guard_closed_on_current_real_ledger_state():
    """Against the ACTUAL repo config + hypothesis ledger (no overrides), the
    revival guard must independently hold the gate CLOSED right now."""
    ok, detail = gate_mod._hyp071_revival_confirmed({"ignition": {}})
    assert ok is False
    assert "METRIC_ARTIFACT verdict stands" in detail


def test_revival_guard_opens_on_hypothetical_fresh_confirmed_adjudication(tmp_path):
    """Fixture-only: a hypothetical fresh prereg + new CONFIRMED adjudication,
    dated after the 2026-06-30 METRIC_ARTIFACT verdict, WOULD satisfy the guard.
    Does not touch the real ledger or create any prereg."""
    entries = _KILLED_LEDGER + [{
        "id": "HYP-071",
        "status": "CONFIRMED",
        "date_tested": "2026-08-15",
        "result": {"prereg_hash": "a" * 64},
    }]
    ledger_path = _write_ledger(tmp_path, entries)
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(ledger_path)}})
    assert ok is True
    assert "fresh CONFIRMED adjudication found" in detail


def test_revival_guard_fails_closed_on_missing_ledger(tmp_path):
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(tmp_path / "nope.json")}})
    assert ok is False
    assert "missing" in detail


def test_revival_guard_fails_closed_on_malformed_ledger(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(bad)}})
    assert ok is False
    assert "unreadable" in detail


def test_revival_guard_fails_closed_on_non_list_ledger(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"id": "HYP-071"}')
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(bad)}})
    assert ok is False
    assert "malformed" in detail


def test_revival_guard_rejects_confirmed_entry_missing_prereg_hash(tmp_path):
    entries = [{"id": "HYP-071", "status": "CONFIRMED", "date_tested": "2026-08-01"}]
    ledger_path = _write_ledger(tmp_path, entries)
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(ledger_path)}})
    assert ok is False


def test_revival_guard_rejects_fresh_prereg_dated_before_original_verdict(tmp_path):
    """A CONFIRMED entry with a novel prereg hash but dated BEFORE (or on) the
    2026-06-30 METRIC_ARTIFACT verdict cannot be a revival of that verdict."""
    entries = [{
        "id": "HYP-071",
        "status": "CONFIRMED",
        "date_tested": "2026-06-30",
        "result": {"prereg_hash": "b" * 64},
    }]
    ledger_path = _write_ledger(tmp_path, entries)
    ok, detail = gate_mod._hyp071_revival_confirmed(
        {"ignition": {"hypothesis_ledger_path": str(ledger_path)}})
    assert ok is False


def _patch_board_abs(cfg_path, tmp_path):
    data = yaml.safe_load(Path(cfg_path).read_text())
    data["value_function"]["board_path"] = str(tmp_path / "board.json")
    Path(cfg_path).write_text(yaml.safe_dump(data))
    # gate/scorer resolve board_path against ROOT; make it absolute-safe by using
    # a path that ROOT / abs == abs on POSIX.


# ── Net-return hard guard ───────────────────────────────────────────────────────

def test_trade_score_refuses_gross():
    with pytest.raises(GrossReturnError):
        trade_score(1.5, 0.0, net_confirmed=False)


def test_trade_score_accepts_net():
    s = trade_score(1.0, 0.0, alpha_scale=2.0, net_confirmed=True)
    assert -1.0 < s < 1.0
    assert s == pytest.approx(np.tanh(2.0))


def test_value_scorer_refuses_gross_board(tmp_path):
    cfg, _ = _write_cfg(tmp_path, board_gross=True)
    _patch_board_abs(cfg, tmp_path)
    scorer = ValueScorer(cfg)
    scorer.load_board()
    assert scorer.net_confirmed is False
    with pytest.raises(GrossReturnError):
        scorer.score_trade(1.0, 0.0)


def test_value_scorer_accepts_net_board(tmp_path):
    cfg, _ = _write_cfg(tmp_path, board_gross=False)
    _patch_board_abs(cfg, tmp_path)
    scorer = ValueScorer(cfg)
    scorer.load_board()
    assert scorer.net_confirmed is True
    assert -1.0 < scorer.score_trade(1.0, 0.0) < 1.0


# ── Director magnitude check ────────────────────────────────────────────────────

def test_magnitude_flags_oversized_diff():
    old = {"conviction_exit_trail": 0.58}
    new = {"conviction_exit_trail": 0.80}   # +37.9% > 20% cap
    report = director_mod.review(old, new, regime_fraction=0.3,
                                 config_path=ROOT / "config" / "training.yml")
    assert report.magnitude_ok is False
    assert any("MAGNITUDE" in f for f in report.flags)
    assert report.all_pass is False


def test_magnitude_passes_within_band():
    old = {"conviction_exit_trail": 0.58}
    new = {"conviction_exit_trail": 0.61}   # +5.2% < 20%
    report = director_mod.review(old, new, regime_fraction=0.3,
                                 config_path=ROOT / "config" / "training.yml")
    assert report.magnitude_ok is True


def test_director_never_auto_approves():
    report = director_mod.review({"conviction_exit_trail": 0.58},
                                 {"conviction_exit_trail": 0.58},
                                 regime_fraction=0.3,
                                 config_path=ROOT / "config" / "training.yml")
    assert report.human_gated is True


def test_regime_check_flags_single_regime_dominance():
    report = director_mod.review({"conviction_exit_trail": 0.58},
                                 {"conviction_exit_trail": 0.58},
                                 regime_fraction=0.60,
                                 config_path=ROOT / "config" / "training.yml")
    assert report.regime_ok is False


# ── Policy updater sample weights ───────────────────────────────────────────────

def test_sample_weights_split_on_quartile():
    scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    w = policy_updater.compute_sample_weights(
        scores, config_path=ROOT / "config" / "training.yml")
    assert set(np.unique(w)).issubset({0.5, 2.0})
    assert (w == 2.0).sum() >= 1   # at least the top scores up-weighted
