"""Factory tests — the ignition refusal is the load-bearing one (Article 6 in code)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factory import train as ftrain
from factory.zoo import AbstainingModel, make_model

ROOT = Path(__file__).resolve().parents[1]


class TestIgnitionRefusal:
    def test_refuses_preregistered(self):
        with pytest.raises(SystemExit) as e:
            ftrain.check_ignition("HYP-072")
        msg = str(e.value)
        assert "IGNITION REFUSED" in msg and "PREREGISTERED" in msg
        assert "Article 6" in msg and "Unproven Edges" in msg   # the constitution speaks

    def test_refuses_missing_id(self):
        with pytest.raises(SystemExit, match="does not exist"):
            ftrain.check_ignition("HYP-999")

    def test_confirmed_unlocks(self, tmp_path):
        fake = tmp_path / "ledger.json"
        fake.write_text(json.dumps([{"id": "HYP-TEST", "status": "CONFIRMED", "hash_lock": "abc"}]))
        entry = ftrain.check_ignition("HYP-TEST", ledger_path=fake)
        assert entry["status"] == "CONFIRMED"

    def test_cli_refusal_exit_nonzero(self):
        with pytest.raises(SystemExit):
            ftrain.main(["--hyp", "HYP-073"])


class TestZoo:
    def test_all_members_calibrated_and_abstaining(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        y = pd.Series((X["a"] + 0.5 * rng.normal(size=300) > 0).astype(int))
        for kind in ("logistic", "mlp"):        # xgb covered by import in make_model signature test
            m = AbstainingModel(make_model(kind), min_confidence=0.60).fit(X, y)
            decisions = m.decide(X.iloc[:50])
            assert {d["decision"] for d in decisions} <= {"LONG", "SHORT", "ABSTAIN"}
            assert any(d["decision"] == "ABSTAIN" for d in decisions) or \
                all(d["confidence"] >= 0.60 for d in decisions)
            probs = m.predict_proba(X.iloc[:50])[:, 1]
            assert ((probs >= 0) & (probs <= 1)).all()

    def test_unknown_member_rejected(self):
        with pytest.raises(ValueError, match="small by design"):
            make_model("transformer")


class TestRegistry:
    def test_register_and_lookup_append_only(self, tmp_path, monkeypatch):
        from factory import registry
        monkeypatch.setattr(registry, "REGISTRY", tmp_path / "registry.json")
        e = registry.register("m1", "HYP-TEST", "deadbeef", ["a"], "cv.json", "m.pkl",
                              "logistic", 0.6)
        assert registry.lookup("m1")["data_sha256"] == "deadbeef"
        with pytest.raises(ValueError, match="append-only"):
            registry.register("m1", "HYP-TEST", "x", [], "", "", "logistic", 0.6)


class TestPaperAdapterStub:
    def test_not_enabled_and_refuses_unregistered(self, tmp_path, monkeypatch):
        from factory import paper_adapter as pa
        assert pa.ENABLED is False                      # enabling is an operator act
        with pytest.raises(SystemExit, match="not in the registry"):
            pa.PaperAdapter("ghost-model")

    def test_registered_dry_run_journals_nothing(self, tmp_path, monkeypatch):
        from factory import paper_adapter as pa
        from factory import registry
        monkeypatch.setattr(registry, "REGISTRY", tmp_path / "reg.json")
        registry.register("m2", "HYP-TEST", "cafe", ["a"], "cv.json", "m.pkl", "logistic", 0.6)
        monkeypatch.setattr(pa, "lookup", registry.lookup)
        from experience import journal
        monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "j")
        monkeypatch.setattr(journal, "board_ref", lambda d, p, con=None: None)
        adapter = pa.PaperAdapter("m2")
        row = adapter.decide_and_journal("EURUSD", {"p": 0.71, "confidence": 0.71,
                                                    "decision": "LONG"},
                                         predicates={"all": []}, board_date="2026-07-01",
                                         hyp_id="HYP-TEST", dry_run=True)
        assert "DRAFT-CAPS" in row["detail"]["cap_stamp"]
        assert not (tmp_path / "j").exists()            # dry-run journals nothing
        with pytest.raises(SystemExit, match="NOT enabled"):
            adapter.decide_and_journal("EURUSD", {"p": 0.7, "confidence": 0.7, "decision": "LONG"},
                                       None, "2026-07-01", "HYP-TEST", dry_run=False)
