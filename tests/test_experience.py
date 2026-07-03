"""Memory-organ tests: rubric pin, classifier tree, journal idempotency, review proposals."""
import json
from datetime import date
from pathlib import Path

import pytest

from experience import attribution as att
from experience import journal, weekly_review

ROOT = Path(__file__).resolve().parents[1]


def _cd(**kw):
    base = dict(decision_id="t1", engine="carry", thesis_kind="structural_carry",
                rate_diff_sign_entry=1, rate_diff_sign_exit=1,
                vix_gate_entry="below_thr", vix_gate_exit="below_thr",
                exit_reason="TIME", realized_r=0.4, fill_slippage_r=None)
    base.update(kw)
    return att.ClosedDecision(**base)


class TestRubricLaw:
    def test_rubric_hash_is_pinned(self):
        assert att.assert_rubric_pinned() in att.PINNED_RUBRIC_HASHES

    def test_tampered_rubric_refuses(self, monkeypatch, tmp_path):
        fake = tmp_path / "R.md"
        fake.write_text("changed law")
        monkeypatch.setattr(att, "RUBRIC", fake)
        with pytest.raises(SystemExit, match="not pinned"):
            att.classify(_cd())


class TestClassifierTree:
    def test_win_thesis_intact_confirmed(self):
        assert att.classify(_cd()).cls == "thesis_confirmed"

    def test_win_thesis_dead_luck_good(self):
        a = att.classify(_cd(rate_diff_sign_exit=-1))
        assert a.cls == "luck_good"

    def test_loss_thesis_dead_invalidated(self):
        a = att.classify(_cd(realized_r=-1.0, rate_diff_sign_exit=-1, exit_reason="STOP"))
        assert a.cls == "thesis_invalidated"

    def test_loss_thesis_intact_luck_bad_stop_and_time(self):
        assert att.classify(_cd(realized_r=-0.5, exit_reason="STOP")).cls == "luck_bad"
        assert att.classify(_cd(realized_r=-0.2, exit_reason="TIME")).cls == "luck_bad"

    def test_unknown_reason_or_missing_r_ambiguous(self):
        assert att.classify(_cd(exit_reason="UNKNOWN")).cls == "AMBIGUOUS"
        assert att.classify(_cd(realized_r=None)).cls == "AMBIGUOUS"

    def test_unevaluable_proxy_ambiguous(self):
        assert att.classify(_cd(vix_gate_exit=None)).cls == "AMBIGUOUS"

    def test_slippage_overlay_and_policy_exit(self):
        a = att.classify(_cd(fill_slippage_r=0.3, exit_reason="CB_REFRESH"))
        assert "execution_variance" in a.overlays and "policy_exit" in a.overlays
        assert a.cls == "thesis_confirmed"

    def test_hypothesis_predicates_drive_alive(self):
        d = _cd(thesis_kind="hypothesis", predicate_eval_at_exit={"a": True, "b": False},
                realized_r=-0.4, exit_reason="STOP")
        assert att.classify(d).cls == "thesis_invalidated"


class TestJournal:
    def test_upsert_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path)
        row = {"decision_ts": "2026-07-01T00:00:00+00:00", "decision_id": "x:1",
               "engine": "carry", "pair": "EURUSD", "action": "ABSTAIN",
               "thesis": {"kind": "abstention"}, "board_ref": None, "size": None}
        assert journal.upsert([row]) == 1
        assert journal.upsert([row]) == 0                       # dedupe
        rows = journal.read_all.__wrapped__ if hasattr(journal.read_all, "__wrapped__") else None
        data = [json.loads(l) for l in (tmp_path / "journal_2026_07.jsonl").read_text().splitlines()]
        assert len(data) == 1

    def test_board_row_hash_stable_and_nan_safe(self):
        h1 = journal.board_row_hash({"a": 1.0, "b": float("nan"), "built_at": "x"})
        h2 = journal.board_row_hash({"b": float("nan"), "a": 1.0, "built_at": "y"})
        assert h1 == h2 and len(h1) == 64


class TestWeeklyReview:
    def test_review_generates_and_proposes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "j")
        monkeypatch.setattr(att, "OUT_DIR", tmp_path / "j")
        monkeypatch.setattr(weekly_review, "REVIEW_DIR", tmp_path / "review")
        ledger = tmp_path / "ledger.json"
        ledger.write_text("[]")
        monkeypatch.setattr(weekly_review, "LEDGER", ledger)
        journal.upsert([
            {"decision_ts": "2026-06-29T00:00:00+00:00", "decision_id": "carry:E:1",
             "engine": "carry", "pair": "EURUSD", "action": "ENTER",
             "thesis": {"kind": "structural_carry"}, "board_ref": None, "size": None},
            {"decision_ts": "2026-06-30T00:00:00+00:00", "decision_id": "abstain:G:1",
             "engine": "carry", "pair": "GBPUSD", "action": "ABSTAIN",
             "thesis": {"kind": "abstention"}, "board_ref": None, "size": None,
             "inferred": True}])
        att.write_attributions([att.Attribution("carry:E:1", "AMBIGUOUS", [], {}, "no reason", "x")],
                               "2026-06")
        path = weekly_review.build_review(date(2026, 6, 29))
        text = path.read_text()
        assert "2026-W27" in path.name and "AMBIGUOUS" in text
        entries = json.loads(ledger.read_text())
        assert entries and entries[0]["status"] == "PROPOSED"
        # idempotent: second run adds no duplicate proposal
        weekly_review.build_review(date(2026, 6, 29))
        assert len(json.loads(ledger.read_text())) == len(entries)
