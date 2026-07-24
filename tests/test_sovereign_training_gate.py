"""Tests for the self-play training scaffold (spec: research/SELF_PLAY_TRAINING_ARCHITECTURE.md).

Focus: the safety machinery — ignition gate refusal, the net-return hard guard, and
the director's ±20% magnitude check. These prove the board is physically incapable
of igniting a real cycle before TICK-024 + HYP-071-net-CONFIRMED.
"""
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


def _write_cfg(tmp_path, *, tick=False, hyp=False, board_gross=True) -> Path:
    """Build a self-contained training config + a stub value board in tmp_path."""
    board = tmp_path / "board.json"
    summary = {"gross_R_caveat": "gross"} if board_gross else {"net_confirmed": True}
    board.write_text('{"cells": {}, "summary": %s}' % (
        '{"gross_R_caveat": "gross"}' if board_gross else '{}'))
    cfg = tmp_path / "training.yml"
    cfg.write_text(textwrap.dedent(f"""
        ignition:
          tick_024_carry_fix_landed: {str(tick).lower()}
          hyp_071_net_confirmed: {str(hyp).lower()}
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


def test_gate_opens_only_when_all_conditions_met(tmp_path):
    cfg, _ = _write_cfg(tmp_path, tick=True, hyp=True, board_gross=False)
    _patch_board_abs(cfg, tmp_path)
    status = gate_mod.evaluate_gate(cfg)
    assert status.open is True
    assert status.mode == "LIVE"


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
