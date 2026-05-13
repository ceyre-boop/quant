from __future__ import annotations

import json

import yaml

from lab.run_experiment import (
    apply_mutations,
    evaluate_vs_baseline,
    run_experiment,
    validate_mutation_scope,
)


def _base_config() -> dict:
    return {
        "micro_risk": {"tp1_r": 2.0, "tp2_r": 4.0},
        "pipeline": {"adr_exhaustion_threshold": 0.85},
        "ml_lab": {
            "pipeline": {
                "adr_exhaustion_threshold": {
                    "value": 0.85,
                    "mutable_by_ml": True,
                    "min": 0.75,
                    "max": 0.95,
                }
            },
            "risk": {
                "tp1_r": {"value": 2.0, "mutable_by_ml": True, "min": 1.5, "max": 3.0},
                "tp2_r": {"value": 4.0, "mutable_by_ml": True, "min": 3.0, "max": 6.0},
                "risk_per_trade_pct": {"value": 0.02, "mutable_by_ml": False},
            },
        },
    }


def test_validate_mutation_scope_rejects_invariant():
    config = _base_config()
    try:
        validate_mutation_scope(config, {"risk.risk_per_trade_pct": 0.03})
        assert False, "Expected ValueError for invariant key"
    except ValueError as e:
        assert "invariant" in str(e)


def test_apply_mutations_updates_runtime_targets():
    config = _base_config()
    candidate = apply_mutations(config, {"risk.tp1_r": 2.25})
    assert candidate["micro_risk"]["tp1_r"] == 2.25
    assert candidate["ml_lab"]["risk"]["tp1_r"]["value"] == 2.25


def test_evaluate_vs_baseline_go_true():
    baseline = {"ev_per_trade": 0.20, "max_dd": 8.0}
    result = {"ev_per_trade": 0.30, "max_dd": 8.0, "wf_pass_rate": 0.75}
    verdict = evaluate_vs_baseline(result, baseline)
    assert verdict["go"] is True


def test_run_experiment_with_stub_runner(tmp_path):
    cfg = _base_config()
    cfg_path = tmp_path / "ict_params.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    mutation = {
        "hypothesis": "test",
        "mutations": {"pipeline.adr_exhaustion_threshold": 0.9, "risk.tp1_r": 2.25},
        "backtest_windows": ["2023-01-01:2023-06-30"],
    }
    mutation_path = tmp_path / "mutation.json"
    mutation_path.write_text(json.dumps(mutation))

    baseline = {"ev_per_trade": 0.10, "max_dd": 10.0}

    def _runner(candidate_config, windows):
        assert candidate_config["pipeline"]["adr_exhaustion_threshold"] == 0.9
        assert candidate_config["micro_risk"]["tp1_r"] == 2.25
        assert windows == ["2023-01-01:2023-06-30"]
        return {"ev_per_trade": 0.20, "max_dd": 10.0, "wf_pass_rate": 1.0}

    results, verdict, candidate = run_experiment(
        base_config_path=cfg_path,
        mutation_path=mutation_path,
        baseline_metrics=baseline,
        runner=_runner,
    )
    assert results["ev_per_trade"] == 0.20
    assert verdict["go"] is True
    assert candidate["ml_lab"]["pipeline"]["adr_exhaustion_threshold"]["value"] == 0.9

