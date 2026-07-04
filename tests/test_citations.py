"""Tests for experience/citations.py — L3 citation records."""
from datetime import date, timedelta

from experience import citations as cit

PRECEDENT = {
    "entry_id": "v1_2008_gfc", "source": "canonical", "label": "SYSTEMIC_CREDIT_FAILURE",
    "event_date": "2008-09-15", "why": "Lehman collapse", "what_followed": "57% drawdown",
    "outcome_days": 540, "severity": 2, "matched_tags": ["credit", "systemic"], "score": 2,
}
WEEK = {"tag": "2026-W27", "end": date(2026, 7, 5), "review_path": "review/2026-W27.md"}


class TestMakeCitation:
    def test_schema_and_ids(self):
        c = cit.make_citation(WEEK, PRECEDENT, ["carry:E:1"], basis={"pairs": ["EURUSD"]})
        assert c["citation_id"] == "cite:2026-W27:v1_2008_gfc"
        assert c["week"] == "2026-W27"
        assert c["review_path"] == "review/2026-W27.md"
        assert c["entry_id"] == "v1_2008_gfc"
        assert c["entry_source"] == "canonical"
        assert c["label"] == "SYSTEMIC_CREDIT_FAILURE"
        assert c["decision_ids"] == ["carry:E:1"]
        assert c["pairs"] == ["EURUSD"]
        assert c["scored"] is None
        assert c["similarity_basis"]["method"] == "structured_v1"
        assert c["similarity_basis"]["matched_tags"] == ["credit", "systemic"]
        assert c["similarity_basis"]["score"] == 2
        assert "ts" in c and "rubric_sha" in c

    def test_scoring_due_capped_at_90_days(self):
        c = cit.make_citation(WEEK, PRECEDENT, [], basis={})    # outcome_days=540, capped
        assert c["scoring_due"] == str(date(2026, 7, 5) + timedelta(days=90))

    def test_scoring_due_uses_outcome_days_when_under_cap(self):
        short = dict(PRECEDENT, outcome_days=14)
        c = cit.make_citation(WEEK, short, [], basis={})
        assert c["scoring_due"] == str(date(2026, 7, 5) + timedelta(days=14))

    def test_defaults_when_basis_omitted(self):
        c = cit.make_citation(WEEK, PRECEDENT, [], basis=None)
        assert c["pairs"] == []
        assert c["similarity_basis"]["board_extremes"] == []
        assert c["analogy_prediction"]                          # non-empty default narrative
        assert c["rubric_sha"] == ""

    def test_week_end_accepts_iso_string(self):
        week = {"tag": "2026-W27", "end": "2026-07-05", "review_path": "x"}
        c = cit.make_citation(week, PRECEDENT, [], basis={})
        assert c["scoring_due"] == str(date(2026, 7, 5) + timedelta(days=90))


class TestAppendCitations:
    def test_idempotent_dedupe_by_citation_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cit, "CITATIONS_PATH", tmp_path / "citations.jsonl")
        c = cit.make_citation(WEEK, PRECEDENT, [], basis={})
        assert cit.append_citations([c]) == 1
        assert cit.append_citations([c]) == 0
        assert len(cit.read_citations()) == 1

    def test_read_citations_absent_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cit, "CITATIONS_PATH", tmp_path / "nope.jsonl")
        assert cit.read_citations() == []

    def test_append_citations_empty_list_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cit, "CITATIONS_PATH", tmp_path / "citations.jsonl")
        assert cit.append_citations([]) == 0
        assert not (tmp_path / "citations.jsonl").exists()
