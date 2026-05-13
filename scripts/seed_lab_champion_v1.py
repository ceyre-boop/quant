#!/usr/bin/env python3
"""Seed lab baseline registry with champion v1 from validated metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from lab.baseline_registry import BaselineRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed champion v1 baseline")
    parser.add_argument("--version", default="v1", help="Champion version label")
    parser.add_argument("--config", default="config/ict_params.yml", help="Champion config path")
    parser.add_argument(
        "--metrics",
        default="data/reports/replay_validation_latest.json",
        help="Validated metrics source JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config
    metrics_path = repo_root / args.metrics

    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    metrics_src = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    backtest_metrics = metrics_src.get("backtest_metrics", {})
    metrics = {
        "ev_per_trade": float(backtest_metrics.get("total_return", 0.0)),
        "max_dd": float(backtest_metrics.get("max_drawdown", 0.0)),
        "wf_pass_rate": 1.0 if bool(metrics_src.get("replay_passed", False)) else 0.0,
        "trade_count": int(backtest_metrics.get("total_trades", 0)),
        "source_report": str(metrics_path),
    }

    registry = BaselineRegistry()
    registry.set_champion(args.version, config=config, metrics=metrics)

    print(json.dumps({"version": args.version, "metrics": metrics, "registry_root": str(registry.root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

