from __future__ import annotations

import json

from lab.baseline_registry import BaselineRegistry


def test_baseline_registry_roundtrip(tmp_path):
    reg = BaselineRegistry(root=tmp_path / "baseline_registry")
    cfg = {"pipeline": {"adr_exhaustion_threshold": 0.85}}
    metrics = {"ev_per_trade": 0.25, "sharpe": 1.2, "max_dd": 8.5}

    reg.set_champion("v1", cfg, metrics)

    loaded = reg.get_champion_metrics("v1")
    assert loaded["ev_per_trade"] == 0.25

    reg.append_experiment({"name": "exp1", "verdict": {"go": True}})
    lines = reg.experiment_log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["name"] == "exp1"

