"""Tests for experience/library_annex.py — the L1 sibling annex (VOLUME_XI_LIVED).

Mirrors tests/test_experience.py's monkeypatch convention (journal.JOURNAL_DIR) applied to
library_annex.ANNEX_PATH — never touches the real data/experience/library_annex.jsonl.
"""
from experience import library_annex as la


def _entry(**kw):
    base = dict(entry_id="annex:review:2026-W27", volume=la.VOLUME_XI_LIVED,
                label="TEST_EVENT", date="2026-07-02", description="desc", outcome="outcome",
                outcome_days=7, severity=0, tags=["carry"], source_kind="review",
                source_ref="review/2026-W27.md")
    base.update(kw)
    return la.LivedEntry(**base)


class TestLivedEntry:
    def test_roundtrip_read_after_append(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        e = _entry()
        assert la.append_entries([e]) == 1
        got = la.read_entries()
        assert len(got) == 1
        assert got[0].entry_id == e.entry_id
        assert got[0].tags == ["carry"]
        assert got[0].volume == la.VOLUME_XI_LIVED

    def test_idempotent_reappend_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        e = _entry()
        assert la.append_entries([e]) == 1
        assert la.append_entries([e]) == 0                  # dedupe by entry_id
        lines = (tmp_path / "annex.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_distinct_entry_ids_both_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        e1 = _entry(entry_id="annex:review:2026-W27")
        e2 = _entry(entry_id="annex:review:2026-W28")
        assert la.append_entries([e1, e2]) == 2
        assert {e.entry_id for e in la.read_entries()} == {e1.entry_id, e2.entry_id}

    def test_mixed_new_and_existing_only_appends_new(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        e1 = _entry(entry_id="annex:attr:carry:E:1")
        e2 = _entry(entry_id="annex:attr:carry:E:2")
        assert la.append_entries([e1]) == 1
        assert la.append_entries([e1, e2]) == 1              # only e2 is new
        assert len(la.read_entries()) == 2

    def test_read_entries_absent_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "does_not_exist.jsonl")
        assert la.read_entries() == []

    def test_append_entries_empty_list_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "ANNEX_PATH", tmp_path / "annex.jsonl")
        assert la.append_entries([]) == 0
        assert not (tmp_path / "annex.jsonl").exists()
