"""
Strategy Selector — regime-conditional strategy dispatch

Loads logs/strategy_regime_table.json to decide which strategy and assets
to run given the current regime classification.

Table is rebuilt by scripts/build_regime_table.py whenever new backtest
data is available.

Usage:
    from sovereign.strategies.strategy_selector import StrategySelector

    selector = StrategySelector()
    rec = selector.select("MOMENTUM")
    print(rec["strategy"])     # "momentum_sma"
    print(rec["best_assets"])  # ["BLK", "QQQ", "ASML"]
    print(rec["avoid_clusters"])  # [1, 4]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_TABLE_PATH = Path("logs/strategy_regime_table.json")

_FALLBACK_TABLE: Dict[str, Any] = {
    "MOMENTUM": {
        "strategy":       "momentum_sma",
        "best_assets":    [],
        "avg_sharpe":     0.0,
        "avoid_clusters": [1, 4],
    },
    "REVERSION": {
        "strategy":       "bb_reversion",
        "best_assets":    [],
        "avg_sharpe":     0.0,
        "avoid_clusters": [2],
    },
    "FLAT": {
        "strategy":       None,
        "best_assets":    [],
        "avg_sharpe":     0.0,
        "avoid_clusters": [0, 3],
    },
    "NEUTRAL": {
        "strategy":       None,
        "best_assets":    [],
        "avg_sharpe":     0.0,
        "avoid_clusters": [0, 3],
    },
}


class StrategySelector:
    """Regime-conditional strategy selector backed by the universe backtest table.

    Args:
        table_path: Path to strategy_regime_table.json. Defaults to
                    logs/strategy_regime_table.json.
    """

    def __init__(self, table_path: str | Path = _TABLE_PATH):
        self._path  = Path(table_path)
        self._table: Dict[str, Any] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, regime: str) -> Dict[str, Any]:
        """Return the recommended strategy record for a given regime.

        Args:
            regime: "MOMENTUM", "REVERSION", "FLAT", or "NEUTRAL".

        Returns:
            Dict with keys:
                strategy (str|None)    — canonical strategy name
                best_assets (list)     — top assets for this regime
                avg_sharpe (float)     — expected Sharpe from backtest
                avoid_clusters (list)  — cluster IDs to keep the veto aware of
        """
        # FLAT and NEUTRAL are equivalent for trading purposes
        key = "FLAT" if regime in ("FLAT", "NEUTRAL") else regime

        record = self._table.get(key)
        if record is None:
            logger.warning(f"[StrategySelector] Unknown regime '{regime}' — returning FLAT")
            record = self._table.get("FLAT", _FALLBACK_TABLE["FLAT"])

        return dict(record)

    def all_regimes(self) -> Dict[str, Any]:
        """Return the full table."""
        return dict(self._table)

    def reload(self) -> None:
        """Hot-reload the table from disk (e.g. after monthly re-opt)."""
        self._load()
        logger.info("[StrategySelector] Table reloaded from disk")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._table = json.loads(self._path.read_text())
                logger.info(
                    f"[StrategySelector] Loaded {len(self._table)} regime entries "
                    f"from {self._path}"
                )
                return
            except Exception as e:
                logger.warning(f"[StrategySelector] Could not parse table: {e} — using fallback")

        logger.info(
            "[StrategySelector] Table file not found — using built-in fallback. "
            "Run scripts/build_regime_table.py to generate from backtest data."
        )
        self._table = dict(_FALLBACK_TABLE)
