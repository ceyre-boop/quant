from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class BaselineRegistry:
    root: Path = Path("data/lab/baseline_registry")

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def champion_config_path(self, version: str) -> Path:
        return self.root / f"champion_{version}.yaml"

    def champion_metrics_path(self, version: str) -> Path:
        return self.root / f"champion_{version}_metrics.json"

    @property
    def experiment_log_path(self) -> Path:
        return self.root / "experiment_log.jsonl"

    def set_champion(self, version: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.champion_config_path(version).write_text(yaml.safe_dump(config, sort_keys=False))
        self.champion_metrics_path(version).write_text(json.dumps(metrics, indent=2, default=str))

    def get_champion_metrics(self, version: str) -> Dict[str, Any]:
        path = self.champion_metrics_path(version)
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def append_experiment(self, record: Dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.experiment_log_path.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def get_trade_count(self) -> int:
        """Return the total number of closed trades recorded in the experiment log.

        Each ``append_experiment`` call may include a ``trade_count`` field inside
        the ``results`` sub-dict from the backtest runner.  This method sums those
        counts across all log entries.  If *no* entry carries ``trade_count``, the
        raw entry count is returned as a conservative lower bound.

        The mode is determined by the *first* entry that has an explicit
        ``trade_count``.  All subsequent entries without the field contribute 0
        rather than 1, avoiding mixed-mode double-counting.

        This value feeds the minimum-N gates in ``lab/run_experiment.py``.
        """
        if not self.experiment_log_path.exists():
            return 0
        total = 0
        has_explicit_count = False
        entry_count = 0
        with self.experiment_log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry_count += 1
                try:
                    entry = json.loads(line)
                    results = entry.get("results", {})
                    if "trade_count" in results:
                        total += int(results["trade_count"])
                        has_explicit_count = True
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        return total if has_explicit_count else entry_count

