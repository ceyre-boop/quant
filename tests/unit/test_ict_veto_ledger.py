from __future__ import annotations

import json

import pytest

from ict.ict_veto_ledger import ICTVetoLedger


def _ledger(tmp_path) -> ICTVetoLedger:
    return ICTVetoLedger(ledger_root=tmp_path)


def test_record_veto_creates_shard(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.record_veto(
        pair="GBPUSD",
        session="NY_PM",
        signal="LONG",
        grade="B",
        score=6.1,
        veto_reason="grade",
        veto_stage="grade",
        timestamp="2026-05-01T14:00:00+00:00",
    )
    shards = list(tmp_path.glob("ict_veto_ledger_*.jsonl"))
    assert len(shards) == 1
    records = json.loads(shards[0].read_text().strip())
    assert records["pair"] == "GBPUSD"
    assert records["grade"] == "B"
    assert records["outcome"] is None


def test_record_veto_multiple_records(tmp_path):
    ledger = _ledger(tmp_path)
    for i in range(3):
        ledger.record_veto(
            pair="EURUSD",
            session="London",
            signal="SHORT",
            grade="C",
            score=4.0 + i,
            veto_reason="grade",
            veto_stage="grade",
            timestamp=f"2026-05-02T0{i}:00:00+00:00",
        )
    shard = list(tmp_path.glob("ict_veto_ledger_*.jsonl"))[0]
    lines = [l for l in shard.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_label_outcome_updates_record(tmp_path):
    ts = "2026-05-03T15:30:00+00:00"
    ledger = _ledger(tmp_path)
    ledger.record_veto(
        pair="AUDUSD",
        session="NY_PM",
        signal="LONG",
        grade="B",
        score=6.5,
        veto_reason="session",
        veto_stage="session",
        entry_level=0.6500,
        tp1=0.6520,
        tp2=0.6550,
        stop=0.6480,
        timestamp=ts,
    )
    found = ledger.label_outcome("AUDUSD", ts, "TP2", outcome_r=4.0)
    assert found is True

    labeled = ledger.get_labeled()
    assert len(labeled) == 1
    assert labeled[0]["outcome"] == "TP2"
    assert labeled[0]["outcome_r"] == 4.0
    assert labeled[0]["outcome_labeled_at"] is not None


def test_label_outcome_missing_record_returns_false(tmp_path):
    ledger = _ledger(tmp_path)
    found = ledger.label_outcome("GBPUSD", "2026-05-01T14:00:00+00:00", "SL")
    assert found is False


def test_get_unlabeled_returns_only_unlabeled(tmp_path):
    ts1 = "2026-05-04T13:00:00+00:00"
    ts2 = "2026-05-04T14:00:00+00:00"
    ledger = _ledger(tmp_path)
    ledger.record_veto(pair="GBPUSD", session="NY_PM", signal="LONG",
                       grade="B", score=6.0, veto_reason="bias", veto_stage="bias",
                       timestamp=ts1)
    ledger.record_veto(pair="GBPUSD", session="NY_PM", signal="SHORT",
                       grade="B", score=6.2, veto_reason="memory", veto_stage="memory",
                       timestamp=ts2)
    ledger.label_outcome("GBPUSD", ts1, "TP1", outcome_r=2.0)

    unlabeled = ledger.get_unlabeled()
    assert len(unlabeled) == 1
    assert unlabeled[0]["timestamp"] == ts2


def test_false_negative_rate_no_data(tmp_path):
    ledger = _ledger(tmp_path)
    report = ledger.false_negative_rate()
    assert report["total"] == 0
    assert report["false_negative_rate"] == 0.0


def test_false_negative_rate_calculation(tmp_path):
    ledger = _ledger(tmp_path)
    timestamps = [f"2026-05-05T{10 + i}:00:00+00:00" for i in range(4)]
    pairs = ["GBPUSD", "EURUSD", "AUDUSD", "AUDNZD"]
    outcomes = ["TP2", "SL", "TP1", "SL"]
    outcome_rs = [4.0, -1.0, 2.0, -1.0]

    for ts, pair, outcome, outcome_r in zip(timestamps, pairs, outcomes, outcome_rs):
        ledger.record_veto(pair=pair, session="NY_PM", signal="LONG",
                           grade="B", score=6.0, veto_reason="grade", veto_stage="grade",
                           timestamp=ts)
        ledger.label_outcome(pair, ts, outcome, outcome_r)

    report = ledger.false_negative_rate()
    assert report["total"] == 4
    assert report["profitable"] == 2           # TP2 + TP1
    assert report["false_negative_rate"] == 0.5


def test_false_negative_rate_filtered_by_stage(tmp_path):
    ledger = _ledger(tmp_path)
    ts_grade = "2026-05-06T10:00:00+00:00"
    ts_memory = "2026-05-06T11:00:00+00:00"
    ledger.record_veto(pair="GBPUSD", session="NY_PM", signal="LONG",
                       grade="B", score=6.0, veto_reason="grade", veto_stage="grade",
                       timestamp=ts_grade)
    ledger.record_veto(pair="EURUSD", session="NY_PM", signal="LONG",
                       grade="A", score=7.5, veto_reason="memory", veto_stage="memory",
                       timestamp=ts_memory)
    ledger.label_outcome("GBPUSD", ts_grade, "TP2", 4.0)
    ledger.label_outcome("EURUSD", ts_memory, "SL", -1.0)

    grade_report = ledger.false_negative_rate(veto_stage="grade")
    assert grade_report["total"] == 1
    assert grade_report["profitable"] == 1

    memory_report = ledger.false_negative_rate(veto_stage="memory")
    assert memory_report["total"] == 1
    assert memory_report["profitable"] == 0


def test_record_veto_with_all_fields(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.record_veto(
        pair="AUDNZD",
        session="London",
        signal="SHORT",
        grade="A",
        score=7.8,
        veto_reason="heatmap",
        veto_stage="heatmap",
        entry_level=1.0850,
        stop=1.0880,
        tp1=1.0820,
        tp2=1.0790,
        adr_pct=0.72,
        risk_pct=0.02,
        confirmations=["sweep", "fvg_tap"],
        missing=["pd_alignment"],
        component_scores={"sweep": 2.5, "fvg_tap": 2.0},
        timestamp="2026-05-07T03:00:00+00:00",
    )
    unlabeled = ledger.get_unlabeled()
    assert len(unlabeled) == 1
    rec = unlabeled[0]
    assert rec["adr_pct"] == 0.72
    assert rec["confirmations"] == ["sweep", "fvg_tap"]
    assert rec["component_scores"]["sweep"] == 2.5
