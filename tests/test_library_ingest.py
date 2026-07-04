"""Tests for experience/library_ingest.py — L1 converters + idempotent backfill.

Includes the MANDATORY (TICK-005 plan) byte-equality check on models/alexandrian_library.json
before/after ingest: that file is LIVE-read by sovereign/orchestrator.py:122 and must never be
written by this work. Every path this ingest WRITES to is monkeypatched to tmp_path; the
canonical library path is only ever read (never opened for writing anywhere in this module).
"""
import json

from experience import attribution as att
from experience import journal
from experience import library_annex as la
from experience import library_ingest as ing
from sovereign.risk.alexandrian_library import LIBRARY_PATH


class TestEntriesFromReview:
    def test_skips_content_free_review(self, tmp_path):
        p = tmp_path / "2026-W27.md"
        p.write_text("# Weekly Self-Review — 2026-W27 (2026-06-29 → 2026-07-05)\n\n"
                     "## Surprises\n- none\n")
        assert ing.entries_from_review(p) == []

    def test_extracts_entry_from_real_shaped_review(self, tmp_path):
        p = tmp_path / "2026-W27.md"
        p.write_text(
            "# Weekly Self-Review — 2026-W27 (2026-06-29 → 2026-07-05)\n\n"
            "## Attribution\n"
            "- Classes: {'AMBIGUOUS': 3, 'luck_good': 1}\n\n"
            "## Surprises\n"
            "- 3/4 attributions AMBIGUOUS — missing exit-mechanism inputs.\n"
            "- 1 win(s) with a dead thesis (luck_good).\n"
        )
        entries = ing.entries_from_review(p)
        assert len(entries) == 1
        e = entries[0]
        assert e.entry_id == "annex:review:2026-W27"
        assert e.date == "2026-07-05"
        assert e.source_kind == "review"
        assert e.volume == la.VOLUME_XI_LIVED
        assert "weekly_review" in e.tags
        assert "AMBIGUOUS" in e.outcome or "luck_good" in e.outcome

    def test_missing_file_returns_empty(self, tmp_path):
        assert ing.entries_from_review(tmp_path / "does_not_exist.md") == []


class TestEntriesFromAttributions:
    def test_converts_row_with_journal_context(self):
        atts = [{"decision_id": "carry:E:1", "cls": "thesis_confirmed", "overlays": [],
                "evidence": {"realized_r": 0.4}, "rationale": "win with thesis intact",
                "rubric_sha": "x", "ts": "2026-07-01T00:00:00+00:00"}]
        journal_by_id = {"carry:E:1": {"decision_ts": "2026-06-30T00:00:00+00:00",
                                       "pair": "EURUSD", "engine": "carry"}}
        entries = ing.entries_from_attributions(atts, journal_by_id)
        assert len(entries) == 1
        e = entries[0]
        assert e.entry_id == "annex:attr:carry:E:1"
        assert e.date == "2026-06-30"
        assert e.severity == -1                     # thesis_confirmed -> positive
        assert "carry" in e.tags and "thesis_confirmed" in e.tags
        assert e.source_kind == "attribution"

    def test_missing_journal_context_degrades_safely(self):
        atts = [{"decision_id": "orphan:1", "cls": "AMBIGUOUS", "rationale": "?",
                "ts": "2026-07-01T00:00:00+00:00"}]
        entries = ing.entries_from_attributions(atts, {})
        assert len(entries) == 1
        assert entries[0].severity == 0
        assert entries[0].entry_id == "annex:attr:orphan:1"

    def test_empty_inputs_yield_empty_list(self):
        assert ing.entries_from_attributions([], {}) == []


class TestEntriesFromLedgerSeals:
    def test_extracts_interim_seal_only(self, tmp_path):
        p = tmp_path / "ledger.json"
        p.write_text(json.dumps([
            {"id": "HYP-072", "name": "COT fade", "family": "POSITIONING-BOARD-2026-07",
             "mechanism": "late-cycle chase", "prior_expectation": "NOT_SIGNIFICANT",
             "annotations": [{"date": "2026-07-02T19:21:41+00:00",
                              "note": "INTERIM SEAL — NO VERDICT (family BH pending)."}]},
            {"id": "HYP-001", "name": "unrelated", "status": "CONFIRMED",
             "annotations": [{"date": "2026-01-01T00:00:00+00:00", "note": "final verdict"}]},
            {"id": "HYP-002", "name": "no annotations at all"},
        ]))
        entries = ing.entries_from_ledger_seals(p)
        assert len(entries) == 1
        e = entries[0]
        assert e.entry_id == "annex:seal:HYP-072"
        assert e.source_kind == "ledger_seal"
        assert e.date == "2026-07-02"
        assert e.volume == la.VOLUME_XI_LIVED

    def test_missing_or_malformed_ledger_returns_empty(self, tmp_path):
        assert ing.entries_from_ledger_seals(tmp_path / "nope.json") == []
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert ing.entries_from_ledger_seals(bad) == []


class TestBackfillIdempotentAndSafe:
    def _patch_all(self, tmp_path, monkeypatch, review_dir=None, ledger_path=None):
        monkeypatch.setattr(ing, "REVIEW_DIR", review_dir or tmp_path / "review_absent")
        monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "journal")
        monkeypatch.setattr(att, "OUT_DIR", tmp_path / "journal")
        monkeypatch.setattr(ing, "LEDGER_PATH", ledger_path or tmp_path / "no_ledger.json")
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")

    def test_idempotent_second_run_appends_nothing_new(self, tmp_path, monkeypatch):
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "2026-W27.md").write_text(
            "# Weekly Self-Review — 2026-W27 (2026-06-29 → 2026-07-05)\n\n"
            "## Surprises\n- something happened worth remembering\n")
        self._patch_all(tmp_path, monkeypatch, review_dir=review_dir)

        r1 = ing.backfill()
        assert r1["review"] == 1
        assert r1["appended"] == 1
        r2 = ing.backfill()
        assert r2["appended"] == 0                    # idempotent re-run

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (review_dir / "2026-W27.md").write_text(
            "# Weekly Self-Review — 2026-W27 (2026-06-29 → 2026-07-05)\n\n"
            "## Surprises\n- something happened worth remembering\n")
        self._patch_all(tmp_path, monkeypatch, review_dir=review_dir)

        result = ing.backfill(dry_run=True)
        assert result["dry_run"] is True
        assert result["appended"] == 0
        assert not (tmp_path / "annex.jsonl").exists()

    def test_no_sources_present_returns_zero_counts(self, tmp_path, monkeypatch):
        self._patch_all(tmp_path, monkeypatch)
        result = ing.backfill()
        assert result == {"review": 0, "attribution": 0, "ledger_seal": 0,
                          "candidates": 0, "appended": 0, "dry_run": False}


class TestLibraryJsonByteEquality:
    def test_canonical_library_json_untouched_by_backfill(self, tmp_path, monkeypatch):
        """MANDATORY (TICK-005 plan): models/alexandrian_library.json is LIVE-read by
        sovereign/orchestrator.py:122 (feeds live size_modifier) — ingest must never write
        it. Byte-for-byte comparison of the REAL file around real backfill() + dry_run
        invocations; every write path is monkeypatched to tmp_path.
        """
        before = LIBRARY_PATH.read_bytes()

        monkeypatch.setattr(ing, "REVIEW_DIR", tmp_path / "review_absent")
        monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "journal")
        monkeypatch.setattr(att, "OUT_DIR", tmp_path / "journal")
        monkeypatch.setattr(ing, "LEDGER_PATH", tmp_path / "no_ledger.json")
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")

        ing.backfill()
        ing.backfill(dry_run=True)

        after = LIBRARY_PATH.read_bytes()
        assert before == after
