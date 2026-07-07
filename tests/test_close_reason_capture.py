"""Tests for the closed-carry exit_reason JOIN against the shadow exit log
(TICK-015 slice 1: experience/backfill.py only — decision_logger/oanda_bridge are
FROZEN and untouched by this change).

Fixture-only: SHADOW_LOG is monkeypatched to a tmp_path file in every test. Never
reads or writes data/exec/exit_manager_shadow.jsonl or any other real data/ path
(house idiom — matches tests/test_experience.py's monkeypatch-the-module-constant
pattern, e.g. `monkeypatch.setattr(journal, "JOURNAL_DIR", tmp_path)`).
"""
import json

from experience import backfill


def _carry_row(*, decision_id="carry:EURUSD:1", pair="EURUSD", outcome="WIN",
                entry_ts="2026-06-25T00:00:00+00:00",
                exit_timestamp="2026-07-01T00:00:00+00:00", trade_id=None):
    """Build a minimal 'closed carry decision' journal row — the shape
    closed_decisions() consumes for engine == 'carry' / action == 'ENTER'."""
    det = {"outcome": outcome, "r_realized": 0.4, "exit_timestamp": exit_timestamp}
    if trade_id is not None:
        det["trade_id"] = trade_id
    return {"decision_id": decision_id, "engine": "carry", "action": "ENTER",
            "pair": pair, "decision_ts": entry_ts, "detail": det}


def _write_shadow(tmp_path, lines):
    """Write raw shadow-log JSONL lines (dicts are serialized; strings written
    verbatim so malformed/blank lines can be injected)."""
    p = tmp_path / "shadow.jsonl"
    with open(p, "w") as f:
        for line in lines:
            f.write(line if isinstance(line, str) else json.dumps(line))
            f.write("\n")
    return p


def _stub_board_helpers(monkeypatch):
    """closed_decisions() also calls rate_diff_sign()/vix_gate_state() — neither is
    under test here and both touch real data/ paths (macro parquet cache, sentiment
    DuckDB). Stub them so these tests never touch anything outside tmp_path."""
    monkeypatch.setattr(backfill, "rate_diff_sign", lambda pair, d: None)
    monkeypatch.setattr(backfill, "vix_gate_state", lambda pair, d: None)


class TestShadowExitReasonJoin:
    def test_join_hit_by_trade_id(self, tmp_path, monkeypatch):
        _stub_board_helpers(monkeypatch)
        shadow = _write_shadow(tmp_path, [
            {"trade_id": "130", "pair": "AUD_NZD", "bar_date": "2026-07-01",
             "decision": "REVERSAL", "action": "CLOSE"},
            # Decoy: same pair+date, different trade_id. If the join fell back to
            # pair+date instead of honoring trade_id first, this would still match
            # (same reason here) so a second decoy with a DIFFERENT reason proves
            # trade_id took priority and pair+date candidates were never consulted.
            {"trade_id": "999", "pair": "AUD_NZD", "bar_date": "2026-07-01",
             "decision": "TIME", "action": "CLOSE"},
        ])
        monkeypatch.setattr(backfill, "SHADOW_LOG", shadow)

        rows = [_carry_row(pair="AUDNZD", trade_id="130",
                            exit_timestamp="2026-07-01T00:00:00+00:00")]
        closed, conflicts = backfill.closed_decisions(rows)

        assert len(closed) == 1
        assert closed[0].exit_reason == "REVERSAL"
        assert conflicts == 0

    def test_join_hit_by_pair_and_date_fallback(self, tmp_path, monkeypatch):
        _stub_board_helpers(monkeypatch)
        shadow = _write_shadow(tmp_path, [
            {"trade_id": "105", "pair": "EUR_USD", "bar_date": "2026-06-28",
             "decision": "TRAILING_ATR", "action": "CLOSE"},
        ])
        monkeypatch.setattr(backfill, "SHADOW_LOG", shadow)

        # No trade_id on the carry side -> must fall back to normalized pair + close-date.
        rows = [_carry_row(pair="EURUSD", trade_id=None,
                            exit_timestamp="2026-06-28T18:00:00+00:00")]
        closed, conflicts = backfill.closed_decisions(rows)

        assert len(closed) == 1
        assert closed[0].exit_reason == "TRAILING"
        assert conflicts == 0

    def test_conflict_keeps_unknown_and_is_counted(self, tmp_path, monkeypatch):
        _stub_board_helpers(monkeypatch)
        # Two CLOSE rows for the same trade_id disagree on why it closed.
        shadow = _write_shadow(tmp_path, [
            {"trade_id": "227", "pair": "EUR_USD", "bar_date": "2026-07-02",
             "decision": "INITIAL_STOP", "action": "CLOSE"},
            {"trade_id": "227", "pair": "EUR_USD", "bar_date": "2026-07-02",
             "decision": "TIME", "action": "CLOSE"},
        ])
        monkeypatch.setattr(backfill, "SHADOW_LOG", shadow)

        rows = [_carry_row(pair="EURUSD", trade_id="227",
                            exit_timestamp="2026-07-02T00:00:00+00:00")]
        closed, conflicts = backfill.closed_decisions(rows)

        assert len(closed) == 1
        assert closed[0].exit_reason == "UNKNOWN"
        assert conflicts == 1

    def test_no_shadow_row_keeps_unknown_without_conflict(self, tmp_path, monkeypatch):
        _stub_board_helpers(monkeypatch)
        shadow = _write_shadow(tmp_path, [])  # empty shadow log
        monkeypatch.setattr(backfill, "SHADOW_LOG", shadow)

        rows = [_carry_row(pair="USDJPY", trade_id="999",
                            exit_timestamp="2026-07-01T00:00:00+00:00")]
        closed, conflicts = backfill.closed_decisions(rows)

        assert len(closed) == 1
        assert closed[0].exit_reason == "UNKNOWN"
        assert conflicts == 0

    def test_malformed_shadow_line_is_skipped_not_raised(self, tmp_path, monkeypatch):
        _stub_board_helpers(monkeypatch)
        good = {"trade_id": "105", "pair": "EUR_USD", "bar_date": "2026-06-28",
                "decision": "TRAILING_ATR", "action": "CLOSE"}
        shadow = _write_shadow(tmp_path, [
            "{not valid json",
            good,
            "",  # blank lines must also be tolerated
        ])
        monkeypatch.setattr(backfill, "SHADOW_LOG", shadow)

        rows = [_carry_row(pair="EURUSD", trade_id=None,
                            exit_timestamp="2026-06-28T18:00:00+00:00")]
        closed, conflicts = backfill.closed_decisions(rows)

        assert len(closed) == 1
        assert closed[0].exit_reason == "TRAILING"
        assert conflicts == 0
