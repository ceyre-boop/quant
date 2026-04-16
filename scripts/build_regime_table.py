"""
Task 4 — Build strategy × regime performance table

Reads logs/universe_backtest_results.json and logs/failure_clusters.json,
computes the best strategy per regime, and saves the result to
logs/strategy_regime_table.json.

Also updates sovereign/strategies/strategy_selector.py's table on disk
(the selector auto-reloads the JSON on next instantiation).

Run: python3 scripts/build_regime_table.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

LOGS = Path("logs")
RESULTS_JSON  = LOGS / "universe_backtest_results.json"
CLUSTERS_JSON = LOGS / "failure_clusters.json"
TABLE_JSON    = LOGS / "strategy_regime_table.json"


def build_table() -> Dict[str, Any]:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"{RESULTS_JSON} not found — run universe_sweep.py first")

    results: List[dict] = json.loads(RESULTS_JSON.read_text()).get("results", [])
    clusters: List[dict] = []
    if CLUSTERS_JSON.exists():
        clusters = json.loads(CLUSTERS_JSON.read_text())

    # Map cluster_id → cluster info
    cluster_map = {c["cluster"]: c for c in clusters}

    # ── Canonical strategy → regime mapping ──────────────────────────────────
    # Each strategy has a home regime where it is designed to operate.
    # Only count a combo as contributing to a regime if it is the right
    # strategy for that regime AND has positive sharpe.
    STRATEGY_HOME_REGIME = {
        "momentum_sma":      "MOMENTUM",
        "atr_channel":       "MOMENTUM",
        "bb_reversion":      "REVERSION",
        "donchian_breakout": "FLAT",   # cross-regime; assign to FLAT (neutral)
    }

    regime_sharpes: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    regime_assets:  Dict[str, Dict[str, List[dict]]]  = defaultdict(lambda: defaultdict(list))

    for r in results:
        strategy = r.get("strategy", "")
        sharpe   = r.get("sharpe", 0.0)
        if not strategy:
            continue
        home = STRATEGY_HOME_REGIME.get(strategy, "FLAT")
        # Only include positive-sharpe combos in the regime bucket
        if sharpe > 0:
            regime_sharpes[home][strategy].append(sharpe)
            regime_assets[home][strategy].append({
                "asset":  r["asset"],
                "sharpe": sharpe,
            })

    # Rename for use below
    regime_strategy_sharpes = regime_sharpes
    regime_strategy_assets  = regime_assets

    all_regimes = {"MOMENTUM", "REVERSION", "FLAT"}

    # ── Find avoid_clusters per regime ───────────────────────────────────────
    regime_avoid: Dict[str, List[int]] = defaultdict(list)
    for c in clusters:
        dom_regime = c.get("dominant_regime", "NEUTRAL")
        if dom_regime in ("NEUTRAL", "FLAT"):
            dom_regime = "FLAT"
        if c.get("mean_loss_r", 0) < -2.5:
            regime_avoid[dom_regime].append(c["cluster"])
        # Also map the dominant strategy's regime failures
        dom_strategy = c.get("dominant_strategy", "")
        if dom_strategy in ("momentum_sma",):
            regime_avoid["MOMENTUM"].append(c["cluster"])
        elif dom_strategy in ("bb_reversion",):
            regime_avoid["REVERSION"].append(c["cluster"])

    # Deduplicate
    for k in regime_avoid:
        regime_avoid[k] = sorted(set(regime_avoid[k]))

    # ── Build table ───────────────────────────────────────────────────────────
    table: Dict[str, Any] = {}

    for regime in all_regimes:
        strat_sharpes = regime_strategy_sharpes.get(regime, {})

        if not strat_sharpes or regime == "FLAT":
            table[regime] = {
                "strategy":       None,
                "best_assets":    [],
                "avg_sharpe":     0.0,
                "avoid_clusters": regime_avoid.get("FLAT", [0, 3]),
            }
            continue

        # Best strategy = highest avg sharpe in this regime
        best_strat = max(strat_sharpes, key=lambda s: sum(strat_sharpes[s]) / len(strat_sharpes[s]))
        avg_sharpe = sum(strat_sharpes[best_strat]) / len(strat_sharpes[best_strat])

        # Top 3 assets for this regime × strategy, by individual sharpe
        assets_list = regime_strategy_assets[regime][best_strat]
        top_assets = [
            a["asset"] for a in sorted(assets_list, key=lambda x: x["sharpe"], reverse=True)[:3]
        ]

        table[regime] = {
            "strategy":       best_strat,
            "best_assets":    top_assets,
            "avg_sharpe":     round(avg_sharpe, 4),
            "avoid_clusters": regime_avoid.get(regime, []),
        }

    # Ensure NEUTRAL is an alias for FLAT
    table["NEUTRAL"] = table["FLAT"]

    return table


def main():
    print("[Build] Reading backtest results...")
    table = build_table()

    TABLE_JSON.write_text(json.dumps(table, indent=2))
    print(f"[Build] Saved → {TABLE_JSON}")

    print("\nStrategy × Regime Table:")
    print("─" * 72)
    for regime, rec in sorted(table.items()):
        if regime == "NEUTRAL":
            continue  # alias — skip printing
        strat    = rec["strategy"] or "STAY OUT"
        assets   = ", ".join(rec["best_assets"]) or "—"
        sharpe   = rec["avg_sharpe"]
        avoid    = rec["avoid_clusters"]
        print(f"  {regime:<12} → {strat:<22} sharpe={sharpe:>5.2f}  "
              f"top=[{assets}]  avoid_clusters={avoid}")
    print("─" * 72)
    print("\nSTRATEGY TABLE: saved, selector will auto-load on next import")


if __name__ == "__main__":
    main()
