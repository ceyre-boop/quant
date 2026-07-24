"""Value scorer — wraps the HYP-071 exit-value board READ-ONLY and maps a trade's
net return to the [-1, +1] Stockfish scale (spec §4.2, §9).

Two things live here:

  trade_score(net_return_r, expected_r)  — the reward mapping, tanh(alpha * scale).
  ValueScorer                             — loads the HYP-071 board (read-only),
                                            resolves a trade's position to its cell,
                                            and scores it, refusing on gross returns.

THE NET-RETURN HARD GUARD: trade_score and the board loader both refuse if the cost
model in effect is still the known-bad gross one (detected via the board's gross
marker). Training on gross returns is the exact failure mode the spec warns about
(§4.2, §8.2) — it is made impossible here, not merely discouraged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"


class GrossReturnError(RuntimeError):
    """Raised when scoring is attempted while the cost model is still gross."""


def trade_score(net_return_r: float, expected_r: float,
                alpha_scale: float = 2.0, *, net_confirmed: bool) -> float:
    """Map a trade's NET return (R-multiples) to the [-1, +1] scale.

    +1.0 = perfect trade; 0.0 = break-even vs carry-and-hold; -1.0 = max loss.
    The mapping is nonlinear (tanh) — small alpha compresses near 0, extremes
    saturate toward ±1 without ever reaching it.

    HARD GUARD: `net_confirmed` MUST be True. It asserts the caller has verified
    (via the ignition gate / board net-check) that `net_return_r` is a NET figure.
    If False, we refuse — training on gross returns is forbidden (spec §4.2).
    """
    if not net_confirmed:
        raise GrossReturnError(
            "trade_score refused: net_confirmed=False. The reward signal must "
            "consume NET returns (post TICK-024). Refusing to score gross returns."
        )
    alpha = net_return_r - expected_r          # excess over carry-and-hold
    return float(np.tanh(alpha * alpha_scale))


def _load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


class ValueScorer:
    """Read-only wrapper over the HYP-071 board. Never mutates it, never retrains it."""

    def __init__(self, config_path: Path | None = None):
        self.cfg = _load_config(config_path)
        self.reward_cfg = self.cfg.get("reward", {})
        vf = self.cfg.get("value_function", {})
        self.board_path = ROOT / vf.get("board_path", "")
        self.gross_marker = vf.get("gross_marker_key", "gross_R_caveat")
        self._board: dict | None = None
        self._cells: dict | None = None
        self.net_confirmed: bool = False

    def load_board(self) -> None:
        """Load the board read-only and evaluate the net-return guard."""
        if not self.board_path.exists():
            # No silent mocking: missing board is a hard stop.
            raise FileNotFoundError(
                f"HYP-071 value board not found: {self.board_path}. Cannot score."
            )
        self._board = json.loads(self.board_path.read_text())
        self._cells = self._board.get("cells", {})
        summary = self._board.get("summary", {})
        # NET-RETURN HARD GUARD: gross marker present → board is gross → not net.
        self.net_confirmed = not bool(summary.get(self.gross_marker))

    def _require_net(self) -> None:
        if self._board is None:
            self.load_board()
        if not self.net_confirmed:
            raise GrossReturnError(
                f"ValueScorer refused: board '{self.board_path.name}' carries the "
                f"gross marker '{self.gross_marker}'. Scoring gross returns is "
                "forbidden until TICK-024 lands and HYP-071 is recomputed on net."
            )

    @staticmethod
    def _cell_key(coords: dict) -> str:
        return "|".join(
            f"{k}={coords[k]}" for k in
            ("atr_tercile", "excursion", "hold_frac", "rsi_extreme", "carry")
        )

    def lookup_cell(self, coords: dict) -> dict | None:
        """Resolve a trade position (coords) to its board cell, by matching coords.
        READ-ONLY. Returns the cell dict or None if no evaluated cell matches."""
        self._require_net()
        target = self._cell_key(coords)
        for cell in self._cells.values():
            c = cell.get("coords", {})
            if cell.get("evaluated") and self._cell_key(c) == target:
                return cell
        return None

    def score_trade(self, net_return_r: float, expected_r: float) -> float:
        """Score one trade's NET return on the [-1, +1] scale. Guarded."""
        self._require_net()
        return trade_score(
            net_return_r, expected_r,
            alpha_scale=float(self.reward_cfg.get("alpha_scale", 2.0)),
            net_confirmed=self.net_confirmed,
        )

    def score_batch(self, net_returns_r, expected_returns_r) -> np.ndarray:
        """Vectorised scoring of a batch of NET returns. Guarded once up front."""
        self._require_net()
        net = np.asarray(net_returns_r, dtype=float)
        exp = np.asarray(expected_returns_r, dtype=float)
        alpha_scale = float(self.reward_cfg.get("alpha_scale", 2.0))
        return np.tanh((net - exp) * alpha_scale)
