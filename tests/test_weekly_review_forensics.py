"""Tests for the guarded Forensics feeds wired into experience/weekly_review.py (TICK-006).

Mirrors tests/test_weekly_review_precedents.py's fixture shape (same engine="carry" journal
row, same monkeypatch pattern) and extends it with the six new forensics path constants
(ORACLE_REFLECTIONS_DIR, LEDGER, VETO_LEDGER_DIR, AUDIT_REPORTS_DIR, LESSON_VELOCITY_PATH,
MARKET_BRIEFING_LATEST) — all monkeypatched to tmp_path locations, so nothing here ever
touches a real data/ or audit/ path. Precedents (TICK-005) stays at its config default
(review_enabled: false), keeping these tests focused on Forensics alone.

For each of the six feeds: present / absent / corrupt, in every case asserting the review
still generates (the .md file is written, "## Forensics" appears) — a feed must never raise.
"""
import json
from datetime import date

from experience import attribution as att
from experience import journal
from experience import weekly_review


def _seed_week(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path / "j")
    monkeypatch.setattr(att, "OUT_DIR", tmp_path / "j")
    monkeypatch.setattr(weekly_review, "REVIEW_DIR", tmp_path / "review")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("[]")
    monkeypatch.setattr(weekly_review, "LEDGER", ledger)
    # All six forensics sources default to nonexistent tmp_path locations == "absent";
    # individual tests seed one of them as present/corrupt as needed.
    monkeypatch.setattr(weekly_review, "ORACLE_REFLECTIONS_DIR", tmp_path / "reflections")
    monkeypatch.setattr(weekly_review, "VETO_LEDGER_DIR", tmp_path / "veto")
    monkeypatch.setattr(weekly_review, "AUDIT_REPORTS_DIR", tmp_path / "audit_reports")
    monkeypatch.setattr(weekly_review, "LESSON_VELOCITY_PATH", tmp_path / "lesson_velocity.json")
    monkeypatch.setattr(weekly_review, "MARKET_BRIEFING_LATEST", tmp_path / "briefing.json")
    journal.upsert([{
        "decision_ts": "2026-06-29T00:00:00+00:00", "decision_id": "carry:E:1",
        "engine": "carry", "pair": "EURUSD", "action": "ENTER",
        "thesis": {"kind": "structural_carry"}, "board_ref": None, "size": None,
    }])
    return ledger


class TestAllSourcesAbsent:
    """Baseline: none of the six forensics sources exist (except the empty '[]' ledger
    _seed_week always writes). The review must still generate, one graceful line per feed."""

    def test_review_still_generates_with_all_sources_absent(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "## Forensics" in text
        assert "### This week's research (hypothesis ledger)" in text
        assert "Verdicts this week: none" in text
        assert "### System health (Oracle)" in text
        assert "No reflection on or before week end." in text
        assert "### Acted : abstained : vetoed" in text
        assert "Ratio (acted:abstained:vetoed) = 1:0:0" in text
        assert "### Audit parity (shadow window)" in text
        assert "No audit report on or before week end." in text
        assert "### Lesson velocity (Oracle)" in text
        assert "feed unavailable: lesson_velocity" in text
        assert "### Macro (FRED, via market briefing)" in text
        assert "feed unavailable: macro_briefing" in text
        # rest of the review is intact — forensics is purely additive
        assert "Acted vs abstained" in text


class TestFeedOracleHealth:
    def test_present_shows_note_with_red1_quarantine_marker(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        refl_dir = tmp_path / "reflections"
        refl_dir.mkdir()
        (refl_dir / "2026-06-30.json").write_text(json.dumps({
            "reflection": {"system_health_note": "Zero trades in last 7 days; system conservative."}
        }))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Zero trades in last 7 days" in text
        assert "[source quarantined: RED-1 open — context only]" in text

    def test_ignores_reflection_after_week_end(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        refl_dir = tmp_path / "reflections"
        refl_dir.mkdir()
        (refl_dir / "2026-06-30.json").write_text(json.dumps(
            {"reflection": {"system_health_note": "the correct in-window note"}}))
        (refl_dir / "2026-07-10.json").write_text(json.dumps(
            {"reflection": {"system_health_note": "future note must not appear"}}))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "the correct in-window note" in text
        assert "future note must not appear" not in text

    def test_corrupt_json_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        refl_dir = tmp_path / "reflections"
        refl_dir.mkdir()
        (refl_dir / "2026-06-30.json").write_text("{not valid json")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "feed unavailable: oracle_health" in text
        assert "## Forensics" in text


class TestFeedHypothesisResearch:
    def test_present_counts_verdicts_interim_seal_and_blocked_in_window(self, tmp_path, monkeypatch):
        ledger = _seed_week(tmp_path, monkeypatch)
        ledger.write_text(json.dumps([
            {"id": "HYP-900", "status": "CONFIRMED", "date_confirmed": "2026-06-30"},
            {"id": "HYP-901", "status": "RETEST_BLOCKED", "date_decided": "2026-07-01"},
            {"id": "HYP-902", "status": "PREREGISTERED", "date_registered": "2020-01-01",
             "annotations": [{"date": "2026-06-29T00:00:00Z",
                               "note": "INTERIM SEAL — NO VERDICT (family pending)."}]},
            {"id": "HYP-903", "status": "NOT_SIGNIFICANT", "date_tested": "2020-01-01"},  # out of window
        ]))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "This week's research (hypothesis ledger)" in text
        assert "'CONFIRMED': 1" in text
        assert "'RETEST_BLOCKED': 1" in text
        assert "NOT_SIGNIFICANT" not in text.split("This week's research")[1].split("###")[0]
        assert "INTERIM SEAL annotations: 1" in text
        assert "BLOCKED: 1" in text

    def test_absent_ledger_fails_open_and_still_generates(self, tmp_path, monkeypatch):
        ledger = _seed_week(tmp_path, monkeypatch)
        ledger.unlink()
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "feed unavailable: hypothesis_research" in text
        assert "## Forensics" in text

    def test_corrupt_ledger_fails_open_and_still_generates(self, tmp_path, monkeypatch):
        ledger = _seed_week(tmp_path, monkeypatch)
        ledger.write_text("{not valid json")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "feed unavailable: hypothesis_research" in text

    def test_terminal_verdict_suppresses_reproposal_and_ledger_write(self, tmp_path, monkeypatch):
        """The stable proposal key already carries a terminal verdict (REJECTED) under a
        past week's tag -> this week must not re-propose it, in prose OR in the ledger."""
        ledger = _seed_week(tmp_path, monkeypatch)
        ledger.write_text(json.dumps([
            {"id": "PROP-2026-W20-exit-reason-capture", "status": "REJECTED",
             "date_decided": "2026-06-20"},
        ]))
        att.write_attributions(
            [att.Attribution("carry:E:1", "AMBIGUOUS", [], {}, "no reason", "x")], "2026-06")
        path = weekly_review.build_review(date(2026, 6, 29))  # dry_run=False: exercises the real write path
        text = path.read_text()
        assert "AMBIGUOUS" in text                        # the underlying surprise still surfaces
        assert "Record the exit MECHANISM" not in text     # candidate proposal NOT surfaced
        assert "- none this week" in text
        entries = json.loads(ledger.read_text())
        assert len(entries) == 1                           # only the pre-seeded REJECTED entry
        assert entries[0]["id"] == "PROP-2026-W20-exit-reason-capture"

    def test_non_terminal_entry_does_not_suppress_reproposal(self, tmp_path, monkeypatch):
        """A PROPOSED (non-terminal) entry under the same stable slug from a prior week must
        NOT block this week's candidate — only a resolved (terminal) verdict does."""
        ledger = _seed_week(tmp_path, monkeypatch)
        ledger.write_text(json.dumps([
            {"id": "PROP-2026-W20-exit-reason-capture", "status": "PROPOSED",
             "date_proposed": "2026-05-18"},
        ]))
        att.write_attributions(
            [att.Attribution("carry:E:1", "AMBIGUOUS", [], {}, "no reason", "x")], "2026-06")
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Record the exit MECHANISM" in text          # still proposed this week


class TestFeedVetoRatio:
    def test_present_counts_forex_and_ict_vetoes_with_top_stages(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        veto_dir = tmp_path / "veto"
        veto_dir.mkdir()
        (veto_dir / "veto_ledger_2026_06.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [
                {"timestamp": "2026-06-30T10:00:00", "symbol": "EURUSD", "stage": "RISK/EV"},
                {"timestamp": "2026-06-30T11:00:00", "symbol": "GBPUSD", "stage": "RISK/EV"},
            ]) + "\n")
        (veto_dir / "ict_veto_ledger_2026_07.jsonl").write_text(
            json.dumps({"timestamp": "2026-07-01T09:00:00", "pair": "USDJPY",
                        "veto_stage": "session"}) + "\n")
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Acted : abstained : vetoed" in text
        assert "Ratio (acted:abstained:vetoed) = 1:0:3" in text
        assert "RISK/EV" in text

    def test_absent_dir_yields_zero_counts_without_raising(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)  # VETO_LEDGER_DIR left nonexistent
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "Ratio (acted:abstained:vetoed) = 1:0:0" in text

    def test_corrupt_jsonl_line_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        veto_dir = tmp_path / "veto"
        veto_dir.mkdir()
        (veto_dir / "veto_ledger_2026_06.jsonl").write_text("{not valid json\n")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        text = path.read_text()
        assert "feed unavailable: veto_ratio" in text


class TestFeedAuditParity:
    def test_present_shows_l1_l2_and_continuity_violations(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        audit_dir = tmp_path / "audit_reports"
        audit_dir.mkdir()
        (audit_dir / "2026-06-30.json").write_text(json.dumps({
            "l1": {"pass_rate": 1.0, "continuity_violations": [{"trade_id": "1"}, {"trade_id": "2"}]},
            "l2": {"match_rate": 0.875, "matched": 7, "scored": 8},
        }))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Audit parity (shadow window)" in text
        assert "pass_rate=1.0" in text
        assert "match_rate=0.875 (7/8)" in text
        assert "continuity_violations=2" in text

    def test_picks_latest_report_on_or_before_week_end_not_after(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        audit_dir = tmp_path / "audit_reports"
        audit_dir.mkdir()
        (audit_dir / "2026-06-30.json").write_text(json.dumps(
            {"l1": {"pass_rate": 1.0, "continuity_violations": []},
             "l2": {"match_rate": 0.9, "matched": 9, "scored": 10}}))
        (audit_dir / "2026-07-10.json").write_text(json.dumps(  # after week end -> must be ignored
            {"l1": {"pass_rate": 0.0, "continuity_violations": []},
             "l2": {"match_rate": 0.0, "matched": 0, "scored": 1}}))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "2026-06-30: L1" in text
        assert "match_rate=0.9" in text

    def test_absent_dir_shows_graceful_message(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)  # AUDIT_REPORTS_DIR left nonexistent
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "No audit report on or before week end." in path.read_text()

    def test_corrupt_json_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        audit_dir = tmp_path / "audit_reports"
        audit_dir.mkdir()
        (audit_dir / "2026-06-30.json").write_text("{not valid json")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "feed unavailable: audit_parity" in path.read_text()


class TestFeedLessonVelocity:
    def test_present_shows_current_lesson_and_stage(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        (tmp_path / "lesson_velocity.json").write_text(json.dumps({
            "current_lesson": {"component_label": "bars_since_signal", "status": "forming",
                               "days_forming": 4,
                               "latest_lesson_text": "Stale sweep signals are anti-edge."}
        }))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Lesson velocity (Oracle)" in text
        assert "bars_since_signal" in text
        assert "forming, 4d forming" in text
        assert "Stale sweep signals are anti-edge." in text

    def test_absent_file_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)  # LESSON_VELOCITY_PATH left nonexistent
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "feed unavailable: lesson_velocity" in path.read_text()

    def test_corrupt_json_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        (tmp_path / "lesson_velocity.json").write_text("{not valid json")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "feed unavailable: lesson_velocity" in path.read_text()


class TestFeedMacroBriefing:
    def test_present_shows_fred_summary_and_discards_regime_read(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        (tmp_path / "briefing.json").write_text(json.dumps({
            "date": "2026-07-03",
            "regime_read": "NQ leading / pulling away from ES -- should never appear",
            "macro_economic": {"summary": {"cpi_yoy_pct": 3.67, "unemployment_pct": 4.2}},
            "provenance": {"verified": False},
        }))
        text = weekly_review.build_review(date(2026, 6, 29), dry_run=True).read_text()
        assert "Macro (FRED, via market briefing)" in text
        assert "cpi_yoy_pct" in text
        assert "provenance.verified=false" in text
        assert "NQ leading" not in text  # regime_read is discarded, never surfaced

    def test_absent_file_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)  # MARKET_BRIEFING_LATEST left nonexistent
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "feed unavailable: macro_briefing" in path.read_text()

    def test_corrupt_json_degrades_to_feed_unavailable(self, tmp_path, monkeypatch):
        _seed_week(tmp_path, monkeypatch)
        (tmp_path / "briefing.json").write_text("{not valid json")
        path = weekly_review.build_review(date(2026, 6, 29), dry_run=True)
        assert path.exists()
        assert "feed unavailable: macro_briefing" in path.read_text()
