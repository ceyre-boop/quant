"""Tests for the guarded Precedents section wired into experience/weekly_review.py (TICK-005).

Mirrors tests/test_experience.py::TestWeeklyReview's fixture shape exactly (same engine="carry"
journal row, same monkeypatch targets) — this ticket is additive, so the baseline scenario it
already covers must keep behaving identically; these tests add the flag/dry_run/failure matrix
on top of it.
"""
import json
from datetime import date

from experience import attribution as att
from experience import citations as cit
from experience import journal
from experience import precedents as prec
from experience import weekly_review


def _seed_week(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "j")
    monkeypatch.setattr(att, "OUT_DIR", tmp_path / "j")
    monkeypatch.setattr(weekly_review, "REVIEW_DIR", tmp_path / "review")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("[]")
    monkeypatch.setattr(weekly_review, "LEDGER", ledger)
    journal.upsert([{
        "decision_ts": "2026-06-29T00:00:00+00:00", "decision_id": "carry:E:1",
        "engine": "carry", "pair": "EURUSD", "action": "ENTER",
        "thesis": {"kind": "structural_carry"}, "board_ref": None, "size": None,
    }])
    return ledger


class TestFlagOff:
    def test_no_section_and_no_citations_when_disabled(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        citations_path = tmp_path / "citations.jsonl"
        monkeypatch.setattr(cit, "CITATIONS_PATH", citations_path)
        monkeypatch.setitem(weekly_review.params["experience"]["precedents"], "review_enabled", False)

        path = weekly_review.build_review(date(2026, 6, 29))
        text = path.read_text()
        assert "## Precedents" not in text
        assert not citations_path.exists()


class TestFlagOnDryRun:
    def test_section_present_but_no_citation_file_written(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        citations_path = tmp_path / "citations.jsonl"
        monkeypatch.setattr(cit, "CITATIONS_PATH", citations_path)
        monkeypatch.setitem(weekly_review.params["experience"]["precedents"], "review_enabled", True)

        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        text = path.read_text()
        assert "## Precedents (Alexandrian Library)" in text
        assert not citations_path.exists()


class TestFlagOnLive:
    def test_section_present_and_citations_written_with_scoring_due(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        citations_path = tmp_path / "citations.jsonl"
        monkeypatch.setattr(cit, "CITATIONS_PATH", citations_path)
        monkeypatch.setitem(weekly_review.params["experience"]["precedents"], "review_enabled", True)

        path = weekly_review.build_review(date(2026, 6, 29))
        text = path.read_text()
        assert "## Precedents (Alexandrian Library)" in text
        assert citations_path.exists()

        rows = [json.loads(l) for l in citations_path.read_text().splitlines() if l.strip()]
        assert rows
        assert all("scoring_due" in r and r["scored"] is None for r in rows)
        assert all(r["week"] == "2026-W27" for r in rows)
        assert all(r["similarity_basis"]["method"] == "structured_v1" for r in rows)

    def test_second_run_same_week_does_not_duplicate_citations(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        citations_path = tmp_path / "citations.jsonl"
        monkeypatch.setattr(cit, "CITATIONS_PATH", citations_path)
        monkeypatch.setitem(weekly_review.params["experience"]["precedents"], "review_enabled", True)

        weekly_review.build_review(date(2026, 6, 29))
        n_after_first = len(cit.read_citations())
        weekly_review.build_review(date(2026, 6, 29))
        assert len(cit.read_citations()) == n_after_first


class TestPrecedentsRaisingDoesNotBreakReview:
    def test_review_still_writes_md_and_ledger_on_precedents_failure(self, tmp_path, monkeypatch):
        ledger = _seed_week(tmp_path, monkeypatch)
        monkeypatch.setattr(cit, "CITATIONS_PATH", tmp_path / "citations.jsonl")
        monkeypatch.setitem(weekly_review.params["experience"]["precedents"], "review_enabled", True)

        def _boom(*a, **kw):
            raise RuntimeError("library on fire")
        monkeypatch.setattr(prec, "week_board_extremes", _boom)

        path = weekly_review.build_review(date(2026, 6, 29))
        assert path.exists()
        text = path.read_text()
        assert "## Precedents (Alexandrian Library)" in text     # error-noted fallback, not a crash
        assert "library on fire" in text
        assert "Acted vs abstained" in text                       # rest of the review intact
        assert json.loads(ledger.read_text()) == []                # no proposals this fixture, no crash either
