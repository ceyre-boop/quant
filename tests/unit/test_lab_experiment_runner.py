from __future__ import annotations

import json

import pytest
import yaml

from lab.run_experiment import (
    apply_mutations,
    can_run_experiment,
    evaluate_vs_baseline,
    run_experiment,
    validate_mutation_scope,
    MINIMUM_TRADES_FOR_PARAMETER_TUNING,
    MINIMUM_TRADES_FOR_META_LABEL_TRAINING,
    MINIMUM_TRADES_FOR_REGIME_ATTRIBUTION,
    MINIMUM_TRADES_FOR_CONFIG_PROMOTION,
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


# ── Minimum N gate tests ──────────────────────────────────────────────────── #

def test_can_run_experiment_below_threshold_blocked():
    ok, msg = can_run_experiment(50, "parameter_nudge")
    assert ok is False
    assert "Insufficient data" in msg
    shortfall = MINIMUM_TRADES_FOR_PARAMETER_TUNING - 50
    assert str(shortfall) in msg


def test_can_run_experiment_at_threshold_allowed():
    ok, msg = can_run_experiment(MINIMUM_TRADES_FOR_PARAMETER_TUNING, "parameter_nudge")
    assert ok is True
    assert "Gate cleared" in msg


def test_can_run_experiment_above_threshold_allowed():
    ok, msg = can_run_experiment(500, "config_promotion")
    assert ok is True


def test_can_run_experiment_blocked_for_config_promotion():
    ok, msg = can_run_experiment(150, "config_promotion")
    assert ok is False
    assert str(MINIMUM_TRADES_FOR_CONFIG_PROMOTION - 150) in msg


def test_can_run_experiment_meta_label_blocked():
    ok, _msg = can_run_experiment(MINIMUM_TRADES_FOR_META_LABEL_TRAINING - 1, "meta_label_retrain")
    assert ok is False


def test_can_run_experiment_regime_specific_allowed():
    ok, _msg = can_run_experiment(MINIMUM_TRADES_FOR_REGIME_ATTRIBUTION, "regime_specific")
    assert ok is True


def test_can_run_experiment_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown experiment_type"):
        can_run_experiment(500, "random_mutation")  # type: ignore[arg-type]


def test_minimum_gate_constants_are_ordered():
    """Sanity check: promotion gate must be the strictest."""
    assert MINIMUM_TRADES_FOR_REGIME_ATTRIBUTION < MINIMUM_TRADES_FOR_PARAMETER_TUNING
    assert MINIMUM_TRADES_FOR_PARAMETER_TUNING < MINIMUM_TRADES_FOR_META_LABEL_TRAINING
    assert MINIMUM_TRADES_FOR_META_LABEL_TRAINING < MINIMUM_TRADES_FOR_CONFIG_PROMOTION

