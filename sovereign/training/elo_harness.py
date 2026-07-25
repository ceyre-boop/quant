"""sovereign/training/elo_harness.py — head-to-head agent-vs-static-v015 Elo (spec §5).

Each game: same entry/pair/date, static v015 exits per its own rules vs. agent
exits per its policy, higher net R wins. Win rate converts to Elo with v015
pinned at Elo 1000 (spec §5's "Elo analogy").

GATED — read before touching this file:
  HYP-071 is METRIC_ARTIFACT (killed 2026-06-30); its board score ("value > 0.5")
  must NOT be used as a win/loss metric until a fresh prereg + CONFIRMED
  adjudication revives it (see sovereign/training/gate.py's HYP-071 REVIVAL GUARD).
  The win/loss classification below is therefore PURE NET-R THRESHOLDS ONLY:
      WIN  : net_R >= +1.0
      LOSS : net_R <= -0.75
      DRAW : otherwise
  Board-score-based classification is NOT implemented here. Do not add a
  `board_score > 0.5` branch without a qualifying HYP-071 revival — see gate.py.

PROVISIONAL: no trained self-play agent exists yet (ignition gate is CLOSED —
see gate.py) and TICK-024's net carry-cost fix has not landed, so every number
this module produces today is provisional twice over: (1) it substitutes the
SCAFFOLD/DRY placeholder rollout (policy_rollout._dry_rollout) for a real agent
because there is nothing else to compare, and (2) returns are gross, not net.
`run_provisional_baseline()` labels its MatchResult accordingly. This module
does not open the gate and does not train anything — it only scores games.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from sovereign.training.gate import evaluate_gate
from sovereign.training.policy_rollout import _dry_rollout

STATIC_ELO = 1000.0
WIN_THRESHOLD_R = 1.0
LOSS_THRESHOLD_R = -0.75


class Outcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"


class GameWinner(str, Enum):
    AGENT = "AGENT"
    STATIC = "STATIC"
    DRAW = "DRAW"


def classify_outcome(net_R: float) -> Outcome:
    """Pure net-R classification of a single side's trade result. Deliberately does
    NOT consult the HYP-071 board score — see module docstring GATED section."""
    if net_R >= WIN_THRESHOLD_R:
        return Outcome.WIN
    if net_R <= LOSS_THRESHOLD_R:
        return Outcome.LOSS
    return Outcome.DRAW


@dataclass
class Game:
    pair: str
    date: str
    static_net_R: float
    agent_net_R: float

    def winner(self, epsilon: float = 1e-9) -> GameWinner:
        if abs(self.agent_net_R - self.static_net_R) <= epsilon:
            return GameWinner.DRAW
        return GameWinner.AGENT if self.agent_net_R > self.static_net_R else GameWinner.STATIC


@dataclass
class MatchResult:
    n_games: int
    agent_wins: int
    static_wins: int
    draws: int
    win_rate: float          # agent win rate, draws counted as 0.5
    elo_agent: float
    elo_static: float = STATIC_ELO
    provisional: bool = True
    notes: list[str] = field(default_factory=list)


def _win_rate_to_elo(win_rate: float, opponent_elo: float = STATIC_ELO) -> float:
    """Standard Elo inversion: win_rate = 1 / (1 + 10^((opponent_elo - elo)/400))."""
    p = min(max(win_rate, 0.01), 0.99)  # clip away from 0/1 to avoid +/-inf
    return opponent_elo + 400.0 * math.log10(p / (1.0 - p))


def play_match(games: list[Game]) -> MatchResult:
    """Score a list of head-to-head games and produce the Elo baseline. Contains
    no I/O and no gate logic — pass it whatever games list you have."""
    if not games:
        raise ValueError("play_match: empty games list")
    agent_wins = sum(1 for g in games if g.winner() == GameWinner.AGENT)
    static_wins = sum(1 for g in games if g.winner() == GameWinner.STATIC)
    draws = len(games) - agent_wins - static_wins
    win_rate = (agent_wins + 0.5 * draws) / len(games)
    return MatchResult(
        n_games=len(games),
        agent_wins=agent_wins,
        static_wins=static_wins,
        draws=draws,
        win_rate=win_rate,
        elo_agent=_win_rate_to_elo(win_rate),
    )


def run_provisional_baseline(config_path: Path | None = None, *, n: int = 200,
                              seed: int = 0) -> MatchResult:
    """Wiring-test baseline ONLY. No trained self-play agent exists (ignition
    CLOSED) — this pairs two independent draws of the SAME scaffold DRY rollout
    (policy_rollout._dry_rollout) as "static" and "agent" stand-ins, purely to
    exercise play_match() end-to-end and produce a reportable number. It carries
    NO information about a real agent's skill: rerun it and the number moves.
    """
    gate = evaluate_gate(config_path)
    pairs = ["GBPUSD", "EURUSD", "AUDUSD", "GBPJPY"]
    static_rollout = _dry_rollout(pairs, n=n, seed=seed)
    agent_rollout = _dry_rollout(pairs, n=n, seed=seed + 1)

    games = [
        Game(
            pair=s["pair"], date=f"dry-{i}",
            static_net_R=s["gross_return_r"], agent_net_R=a["gross_return_r"],
        )
        for i, (s, a) in enumerate(zip(static_rollout.trades, agent_rollout.trades))
    ]
    result = play_match(games)
    result.notes = [
        "PROVISIONAL: gross returns, pre-TICK-024 (net carry-cost fix not landed).",
        "PROVISIONAL: 'agent' side is the SCAFFOLD/DRY placeholder rollout, NOT a "
        "trained self-play policy — ignition gate is CLOSED (see below).",
        "HYP-071 board score NOT used for win/loss (see module GATED section).",
        f"ignition gate: {'OPEN' if gate.open else 'CLOSED'} — {'; '.join(gate.reasons) or 'all checks pass'}",
    ]
    return result
