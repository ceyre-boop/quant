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


def test_get_trade_count_empty_registry(tmp_path):
    reg = BaselineRegistry(root=tmp_path / "baseline_registry")
    assert reg.get_trade_count() == 0


def test_get_trade_count_with_explicit_trade_count(tmp_path):
    reg = BaselineRegistry(root=tmp_path / "baseline_registry")
    reg.append_experiment({"results": {"trade_count": 80, "ev_per_trade": 0.2}})
    reg.append_experiment({"results": {"trade_count": 40, "ev_per_trade": 0.3}})
    assert reg.get_trade_count() == 120


def test_get_trade_count_fallback_entry_count(tmp_path):
    reg = BaselineRegistry(root=tmp_path / "baseline_registry")
    # No trade_count in results → falls back to counting entries
    reg.append_experiment({"results": {"ev_per_trade": 0.2}})
    reg.append_experiment({"results": {"ev_per_trade": 0.3}})
    assert reg.get_trade_count() == 2


def test_get_trade_count_mixed_entries(tmp_path):
    """Entries with explicit trade_count win; fallback entries are not mixed in."""
    reg = BaselineRegistry(root=tmp_path / "baseline_registry")
    reg.append_experiment({"results": {"trade_count": 60}})
    reg.append_experiment({"results": {"trade_count": 50}})
    assert reg.get_trade_count() == 110

