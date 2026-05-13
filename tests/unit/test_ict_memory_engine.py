from __future__ import annotations

import yaml

from ict.memory_engine import ICTMemoryEngine, MemoryMatch


def test_memory_assessment_soft_and_hard_veto(tmp_path):
    config_path = tmp_path / "ict_params.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {
                    "cluster_soft_veto_wr": 0.40,
                    "cluster_hard_veto_wr": 0.30,
                    "cluster_similarity_floor": 0.50,
                    "cluster_soft_veto_penalty": 0.75,
                    "cluster_soft_veto_score_floor": 7.5,
                }
            },
            sort_keys=False,
        )
    )

    engine = ICTMemoryEngine(
        memory_path=tmp_path / "ict_memory.json",
        config_path=str(config_path),
    )

    hard_match = MemoryMatch(
        pair="GBPUSD",
        cluster=0,
        similarity=0.40,
        expected_outcome="REVERSAL",
        historical_wr=0.50,
        n_samples=30,
        analog_date="2026-01-01",
        analog_outcome="LOSS",
        available=True,
    )
    hard = engine.assess_match(hard_match)
    assert hard["hard_veto"] is True
    assert hard["soft_veto"] is False

    soft_match = MemoryMatch(
        pair="GBPUSD",
        cluster=1,
        similarity=0.80,
        expected_outcome="FLAT",
        historical_wr=0.35,
        n_samples=30,
        analog_date="2026-01-02",
        analog_outcome="LOSS",
        available=True,
    )
    soft = engine.assess_match(soft_match)
    assert soft["hard_veto"] is False
    assert soft["soft_veto"] is True
    assert soft["penalty"] == 0.75

