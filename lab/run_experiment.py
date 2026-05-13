from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml

from lab.baseline_registry import BaselineRegistry

import logging

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "min_ev_improvement": 0.05,
    "max_dd_degradation": 0.0,
    "required_wf_pass_rate": 0.70,
}

# Mutation key -> runtime config path
RUNTIME_TARGETS = {
    "pipeline.adr_exhaustion_threshold": "pipeline.adr_exhaustion_threshold",
    "pipeline.displacement_atr_multiplier": "pipeline.displacement_atr_multiplier",
    "risk.tp1_r": "micro_risk.tp1_r",
    "risk.tp2_r": "micro_risk.tp2_r",
    "memory.cluster_k": "memory.cluster_k",
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _spec_for(config: Dict[str, Any], mutation_key: str) -> Dict[str, Any]:
    spec = config.get("ml_lab", {})
    node: Any = spec
    for part in mutation_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def validate_mutation_scope(config: Dict[str, Any], mutations: Dict[str, Any]) -> None:
    for key, value in mutations.items():
        spec = _spec_for(config, key)
        if not spec:
            raise ValueError(
                f"Unknown mutation key: {key}. Valid keys must be defined in ml_lab config."
            )
        if not bool(spec.get("mutable_by_ml", False)):
            raise ValueError(f"Mutation not allowed for invariant key: {key}")
        mn = spec.get("min")
        mx = spec.get("max")
        if mn is not None and value < mn:
            raise ValueError(f"{key}={value} below min {mn}")
        if mx is not None and value > mx:
            raise ValueError(f"{key}={value} above max {mx}")


def _set_path(data: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = data
    for p in parts[:-1]:
        if p not in cursor or not isinstance(cursor[p], dict):
            cursor[p] = {}
        cursor = cursor[p]
    cursor[parts[-1]] = value


def apply_mutations(config: Dict[str, Any], mutations: Dict[str, Any]) -> Dict[str, Any]:
    candidate = copy.deepcopy(config)
    for key, value in mutations.items():
        runtime_path = RUNTIME_TARGETS.get(key, key)
        # Runtime path used by current ICT engine
        _set_path(candidate, runtime_path, value)
        # Keep mirrored high-level key in sync when runtime path differs.
        if runtime_path != key:
            _set_path(candidate, key, value)
        # Keep ml_lab value updated for registry/audit
        _set_path(candidate, f"ml_lab.{key}.value", value)
    return candidate


def _extract_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = metrics.get(key, default)
    try:
        return float(val)
    except Exception:
        logger.warning(
            "Failed to convert metric %s=%r to float; using default %s",
            key, val, default,
        )
        return default


def evaluate_vs_baseline(
    results: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    min_ev_improvement: float = DEFAULT_THRESHOLDS["min_ev_improvement"],
    max_dd_degradation: float = DEFAULT_THRESHOLDS["max_dd_degradation"],
    required_wf_pass_rate: float = DEFAULT_THRESHOLDS["required_wf_pass_rate"],
) -> Dict[str, Any]:
    cand_ev = _extract_metric(results, "ev_per_trade")
    base_ev = _extract_metric(baseline_metrics, "ev_per_trade")
    cand_dd = _extract_metric(results, "max_dd")
    base_dd = _extract_metric(baseline_metrics, "max_dd")
    pass_rate = _extract_metric(results, "wf_pass_rate")

    ev_ok = cand_ev >= (base_ev + min_ev_improvement)
    dd_ok = cand_dd <= (base_dd + max_dd_degradation)
    wf_ok = pass_rate >= required_wf_pass_rate

    return {
        "go": bool(ev_ok and dd_ok and wf_ok),
        "ev_ok": ev_ok,
        "dd_ok": dd_ok,
        "wf_ok": wf_ok,
        "candidate_ev": cand_ev,
        "baseline_ev": base_ev,
        "candidate_max_dd": cand_dd,
        "baseline_max_dd": base_dd,
        "wf_pass_rate": pass_rate,
    }


def _default_runner(candidate_config: Dict[str, Any], windows: List[str]) -> Dict[str, Any]:
    # Import lazily (network/data heavy path)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ict_backtest.py"
    spec = importlib.util.spec_from_file_location("run_ict_backtest", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load backtest script from {script_path}: file not found or invalid module"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backtest_pair = getattr(module, "backtest_pair")
    compute_stats = getattr(module, "compute_stats")

    pairs = candidate_config.get("sessions", {}).get(
        "ny_pm_pairs", ["GBPUSD", "EURUSD", "AUDUSD", "AUDNZD"]
    )
    pairs = [f"{p}=X" if "=X" not in p else p for p in pairs]

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(candidate_config, tmp, sort_keys=False)
        temp_config = tmp.name

    old_cfg = os.environ.get("ICT_CONFIG_PATH")
    os.environ["ICT_CONFIG_PATH"] = temp_config
    try:
        all_trades = []
        window_stats = []
        targets = windows if windows else [None]
        for w in targets:
            if w is None:
                start, end = None, None
            else:
                start, end = (w.split(":") + [None, None])[:2]
            window_trades = []
            for pair in pairs:
                window_trades.extend(backtest_pair(pair, start=start, end=end))
            stats = compute_stats(window_trades)
            window_stats.append(stats)
            all_trades.extend(window_trades)
        total = compute_stats(all_trades)
        pos_windows = sum(1 for s in window_stats if float(s.get("avg_r", 0)) > 0)
        total["ev_per_trade"] = float(total.get("avg_r", 0.0))
        total["max_dd"] = abs(float(total.get("max_dd_pct", 0.0)))
        total["wf_pass_rate"] = (pos_windows / len(window_stats)) if window_stats else 0.0
        total["windows"] = window_stats
        return total
    finally:
        if old_cfg is None:
            os.environ.pop("ICT_CONFIG_PATH", None)
        else:
            os.environ["ICT_CONFIG_PATH"] = old_cfg
        Path(temp_config).unlink(missing_ok=True)


def run_experiment(
    base_config_path: str | Path,
    mutation_path: str | Path,
    baseline_metrics: Dict[str, Any],
    runner: Callable[[Dict[str, Any], List[str]], Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = load_yaml(base_config_path)
    payload = load_json(mutation_path)
    mutations = payload.get("mutations", {})
    windows = payload.get("backtest_windows", [])

    validate_mutation_scope(config, mutations)
    candidate_config = apply_mutations(config, mutations)

    runner_fn = runner or _default_runner
    results = runner_fn(candidate_config, windows)
    verdict = evaluate_vs_baseline(results, baseline_metrics)
    return results, verdict, candidate_config


def run_experiment_with_registry(
    base_config_path: str | Path,
    mutation_path: str | Path,
    baseline_version: str = "v1",
    runner: Callable[[Dict[str, Any], List[str]], Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    registry = BaselineRegistry()
    baseline_metrics = registry.get_champion_metrics(baseline_version)
    if not baseline_metrics:
        raise ValueError(f"No champion metrics found for version '{baseline_version}'")

    results, verdict, candidate_config = run_experiment(
        base_config_path=base_config_path,
        mutation_path=mutation_path,
        baseline_metrics=baseline_metrics,
        runner=runner,
    )

    mutation_payload = load_json(mutation_path)
    registry.append_experiment(
        {
            "baseline_version": baseline_version,
            "mutation": mutation_payload,
            "results": results,
            "verdict": verdict,
        }
    )
    return results, verdict, candidate_config
