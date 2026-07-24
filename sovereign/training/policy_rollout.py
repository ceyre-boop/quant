"""Policy rollout (Phase 1) — generate simulated trades from the current XGBoost policy.

In LIVE mode this loads the current conviction policy and rolls it forward over the
last `lookback_days` to emit N simulated trades across the 4 v015 pairs. In
SCAFFOLD/DRY mode (gate closed) it exercises the SAME code path shape but emits a
small deterministic placeholder rollout and marks it dry — it does NOT stand in for
real trades and is never fed to a production fit. This is a simulation loop only:
it NEVER places live trades and never touches the MT5 / OANDA execution bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "training.yml"

# HYP-071 board coordinate axes — rollout positions are expressed in these so the
# value scorer can resolve each simulated trade to a board cell (read-only).
_ATR = ("low", "mid", "high")
_EXCURSION = ("underwater", "flat", "profit")
_HOLD = ("early", "mid", "late")


@dataclass
class Rollout:
    trades: list[dict]
    dry: bool
    pair_distribution: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trades)


def _load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def rollout_policy(config_path: Path | None = None, *, gate_open: bool,
                   dry_n: int = 512, seed: int = 0) -> Rollout:
    """Generate simulated trades.

    gate_open=False → DRY: deterministic placeholder rollout of `dry_n` positions,
        enough to exercise Phases 2-5 end-to-end. NOT real trades.
    gate_open=True  → LIVE rollout of the configured n_trades from the real policy.
    """
    cfg = _load_config(config_path)
    rc = cfg.get("rollout", {})
    pairs = rc.get("pairs", ["GBPUSD", "EURUSD", "AUDUSD", "GBPJPY"])

    if not gate_open:
        return _dry_rollout(pairs, n=dry_n, seed=seed)

    # LIVE path. Wiring the real XGBoost policy rollout is deferred to ignition —
    # building it now, ungated, would be dead code exercised only past the gate.
    # No silent mocking: stop loudly rather than fabricate a "live" rollout.
    raise NotImplementedError(
        "LIVE policy rollout is not wired yet. It must be implemented against the "
        "net-return board once the ignition gate opens (TICK-024 + HYP-071-net). "
        "Until then the loop runs DRY. See spec §9."
    )


def _dry_rollout(pairs: list[str], n: int, seed: int) -> Rollout:
    """Deterministic placeholder rollout. Positions span the board coordinate space
    so downstream scoring exercises real cell lookups. Returns are labelled GROSS
    and are NOT usable for training — the gate + net guard block that."""
    rng = np.random.default_rng(seed)
    trades = []
    dist: dict[str, int] = {}
    for i in range(n):
        pair = pairs[i % len(pairs)]
        dist[pair] = dist.get(pair, 0) + 1
        coords = {
            "atr_tercile": _ATR[rng.integers(len(_ATR))],
            "excursion": _EXCURSION[rng.integers(len(_EXCURSION))],
            "hold_frac": _HOLD[rng.integers(len(_HOLD))],
            "rsi_extreme": bool(rng.integers(2)),
            "carry": "aligned" if rng.integers(2) else "misaligned",
        }
        trades.append({
            "pair": pair,
            "coords": coords,
            # Placeholder gross R — NEVER consumed for a production fit while dry.
            "gross_return_r": float(rng.normal(0.1, 1.0)),
            "expected_r": 0.0,
            "dry": True,
        })
    return Rollout(trades=trades, dry=True, pair_distribution=dist)
