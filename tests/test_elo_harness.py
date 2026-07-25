"""Tests for sovereign/training/elo_harness.py — head-to-head Elo (spec §5)."""
import pytest

from sovereign.training.elo_harness import (
    STATIC_ELO,
    Game,
    GameWinner,
    Outcome,
    classify_outcome,
    play_match,
    run_provisional_baseline,
)


def test_classify_outcome_pure_net_r_thresholds():
    assert classify_outcome(1.5) == Outcome.WIN
    assert classify_outcome(1.0) == Outcome.WIN
    assert classify_outcome(-0.75) == Outcome.LOSS
    assert classify_outcome(-1.2) == Outcome.LOSS
    assert classify_outcome(0.2) == Outcome.DRAW
    assert classify_outcome(-0.5) == Outcome.DRAW


def test_game_winner_higher_net_r_wins():
    g = Game(pair="EURUSD", date="d1", static_net_R=0.5, agent_net_R=1.2)
    assert g.winner() == GameWinner.AGENT
    g2 = Game(pair="EURUSD", date="d2", static_net_R=1.2, agent_net_R=0.5)
    assert g2.winner() == GameWinner.STATIC
    g3 = Game(pair="EURUSD", date="d3", static_net_R=0.5, agent_net_R=0.5)
    assert g3.winner() == GameWinner.DRAW


def test_play_match_all_agent_wins_yields_elo_above_static():
    games = [Game("EURUSD", f"d{i}", static_net_R=0.0, agent_net_R=1.0) for i in range(20)]
    result = play_match(games)
    assert result.agent_wins == 20
    assert result.win_rate == 1.0
    assert result.elo_agent > STATIC_ELO
    assert result.elo_static == STATIC_ELO


def test_play_match_all_static_wins_yields_elo_below_static():
    games = [Game("EURUSD", f"d{i}", static_net_R=1.0, agent_net_R=0.0) for i in range(20)]
    result = play_match(games)
    assert result.static_wins == 20
    assert result.win_rate == 0.0
    assert result.elo_agent < STATIC_ELO


def test_play_match_even_split_near_static_elo():
    games = (
        [Game("EURUSD", f"w{i}", 0.0, 1.0) for i in range(10)]
        + [Game("EURUSD", f"l{i}", 1.0, 0.0) for i in range(10)]
    )
    result = play_match(games)
    assert result.win_rate == pytest.approx(0.5)
    assert result.elo_agent == pytest.approx(STATIC_ELO)


def test_play_match_empty_raises():
    with pytest.raises(ValueError):
        play_match([])


def test_run_provisional_baseline_is_labeled():
    result = run_provisional_baseline(n=50, seed=1)
    assert result.provisional is True
    assert result.n_games == 50
    assert any("PROVISIONAL" in n for n in result.notes)
    assert any("HYP-071" in n for n in result.notes)
    assert any("ignition gate" in n for n in result.notes)


def test_run_provisional_baseline_deterministic_for_fixed_seed():
    r1 = run_provisional_baseline(n=50, seed=7)
    r2 = run_provisional_baseline(n=50, seed=7)
    assert r1.win_rate == r2.win_rate
    assert r1.elo_agent == r2.elo_agent
