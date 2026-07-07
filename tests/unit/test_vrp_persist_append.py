"""E0 memory-integrity tests: a second validator run can never destroy a first."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("validate_vrp", ROOT / "scripts" / "validate_vrp.py")
vv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vv)


def _run(tmp_path, result, name, run_ts, account):
    vv.persist_options_result(result, name, ("2022-01-01", "2022-12-31"), run_ts, account,
                              options_path=tmp_path / "opt.json", ledger_path=tmp_path / "ledger.json")


def test_two_runs_same_stage_both_preserved(tmp_path):
    r1 = {"status": "OK", "n_trades": 50, "sharpe_weekly_ann": 1.248}
    r2 = {"status": "NO_TRADES", "n_trades": 0, "sharpe_weekly_ann": 0.0}
    _run(tmp_path, r1, "IS", "2026-06-29T18:00:00+00:00", 1_000_000.0)
    doc_after_1 = json.loads((tmp_path / "opt.json").read_text())
    rec1_snapshot = copy.deepcopy(doc_after_1["stages"]["IS"][0])

    _run(tmp_path, r2, "IS", "2026-07-06T00:00:00+00:00", 100_000.0)
    doc = json.loads((tmp_path / "opt.json").read_text())
    assert len(doc["stages"]["IS"]) == 2
    assert doc["stages"]["IS"][0] == rec1_snapshot          # run-1 content-identical
    assert doc["stages"]["IS"][1]["account"] == 100_000.0

    led = json.loads((tmp_path / "ledger.json").read_text())
    entries = [e for e in led if e["id"] == "VRP-001-OPTIONS"]
    assert len(entries) == 1                                # one entry: consumers count entries
    e = entries[0]
    assert e["status"] == "NO_TRADES" and e["account"] == 100_000.0
    assert len(e["runs"]) == 1
    assert e["runs"][0]["status"] == "OK" and e["runs"][0]["account"] == 1_000_000.0
    assert "runs" not in e["runs"][0]                       # snapshots never nest histories


def test_legacy_dict_stage_wrapped_once_idempotently(tmp_path):
    legacy = {"id": "VRP-001-OPTIONS",
              "stages": {"IS": {"status": "OK", "n_trades": 1, "sharpe_weekly_ann": 0.0}}}
    (tmp_path / "opt.json").write_text(json.dumps(legacy))
    _run(tmp_path, {"status": "NO_TRADES", "n_trades": 0}, "OOS", "2026-07-06T01:00:00+00:00", 100_000.0)
    doc = json.loads((tmp_path / "opt.json").read_text())
    assert isinstance(doc["stages"]["IS"], list) and len(doc["stages"]["IS"]) == 1
    assert doc["stages"]["IS"][0]["n_trades"] == 1          # legacy content intact
    assert len(doc["stages"]["OOS"]) == 1
    # second pass over the already-wrapped doc must not double-wrap
    _run(tmp_path, {"status": "NO_TRADES", "n_trades": 0}, "OOS", "2026-07-06T02:00:00+00:00", 100_000.0)
    doc = json.loads((tmp_path / "opt.json").read_text())
    assert len(doc["stages"]["IS"]) == 1 and len(doc["stages"]["OOS"]) == 2


def test_three_runs_accumulate_full_history(tmp_path):
    for i, ts in enumerate(["2026-07-06T03:00:00+00:00", "2026-07-06T04:00:00+00:00",
                            "2026-07-06T05:00:00+00:00"]):
        _run(tmp_path, {"status": "OK", "n_trades": i}, "IS", ts, 100_000.0)
    led = json.loads((tmp_path / "ledger.json").read_text())
    e = next(x for x in led if x["id"] == "VRP-001-OPTIONS")
    assert len(e["runs"]) == 2                              # two priors snapshotted
    assert [r["result"][:3] for r in e["runs"]] == ["[IS", "[IS"]
    doc = json.loads((tmp_path / "opt.json").read_text())
    assert [r["n_trades"] for r in doc["stages"]["IS"]] == [0, 1, 2]


def test_other_ledger_entries_untouched(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps([{"id": "HYP-001", "status": "PREREGISTERED"}]))
    _run(tmp_path, {"status": "OK", "n_trades": 3}, "IS", "2026-07-06T06:00:00+00:00", 100_000.0)
    led = json.loads((tmp_path / "ledger.json").read_text())
    assert {e["id"] for e in led} == {"HYP-001", "VRP-001-OPTIONS"}
    assert next(e for e in led if e["id"] == "HYP-001") == {"id": "HYP-001", "status": "PREREGISTERED"}
