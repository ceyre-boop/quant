"""Outcome loop: alarm accounting + day-boundary matching.

Two defects, both found 2026-07-18 behind a permanently-firing alarm:

1. pulse_check counted every closed OANDA trade as "attempted" on every 2h pulse,
   including ones matched weeks earlier. update_outcome correctly refuses to
   re-close them, so n_backfilled was legitimately 0 and the alarm screamed
   forever. It grew 9 -> 23 while the loop was HEALTHY.
2. Matching fell back to a [:10] date-string compare, so a signal and its fill
   straddling midnight never matched (logs/pulse.err:5971-5976).
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from sovereign.intelligence import decision_logger as dl

UTC = timezone.utc


# ── day-boundary matching ────────────────────────────────────────────────────

@pytest.mark.parametrize("stored,incoming,expected,why", [
    ("2026-06-15T12:00:15", "2026-06-16T12:01:37", True,
     "THE REAL MISS: AUDUSD signal 06-15, fill 06-16 (pulse.err:5971)"),
    ("2026-06-15T23:59:00", "2026-06-16T00:01:00", True, "straddles midnight by 2 min"),
    ("2026-06-16T12:00:00", "2026-06-16T13:30:00", True, "same day, later hour"),
    ("2026-06-16T13:00:00", "2026-06-16T13:00:00", True, "exact"),
    ("2026-06-10T12:00:00", "2026-06-16T12:00:00", False, "6 days apart"),
    ("2026-06-13T12:00:00", "2026-06-16T12:00:00", False, "3 days apart, beyond 36h"),
])
def test_match_window(stored, incoming, expected, why):
    assert dl._outcome_entry_match(stored, incoming) is expected, why


def test_window_is_asymmetric_no_lookahead():
    """A fill must never close a signal logged AFTER it by more than clock skew.

    A symmetric window would let a Monday fill close a Tuesday signal — the same
    class of look-ahead error that refuted HYP-105/106.
    """
    assert dl._outcome_entry_match("2026-06-16T12:00:00", "2026-06-15T12:00:00") is False
    # small skew after the fill is tolerated
    assert dl._outcome_entry_match("2026-06-16T14:00:00", "2026-06-16T13:00:00") is True


def test_clock_skew_boundary():
    base = datetime(2026, 6, 16, 13, 0, tzinfo=UTC)
    inside = (base + dl.MAX_CLOCK_SKEW - timedelta(minutes=1)).isoformat()
    outside = (base + dl.MAX_CLOCK_SKEW + timedelta(minutes=1)).isoformat()
    assert dl._outcome_entry_match(inside, base.isoformat()) is True
    assert dl._outcome_entry_match(outside, base.isoformat()) is False


def test_lead_boundary():
    base = datetime(2026, 6, 16, 13, 0, tzinfo=UTC)
    inside = (base - dl.MAX_SIGNAL_LEAD + timedelta(minutes=1)).isoformat()
    outside = (base - dl.MAX_SIGNAL_LEAD - timedelta(minutes=1)).isoformat()
    assert dl._outcome_entry_match(inside, base.isoformat()) is True
    assert dl._outcome_entry_match(outside, base.isoformat()) is False


def test_unparseable_falls_back_to_date_compare():
    assert dl._outcome_entry_match("garbage", "also garbage") is False


# ── end-to-end across a day boundary ─────────────────────────────────────────

def _write_record(tmp_path, month, rec):
    p = tmp_path / f"decisions_{month}.jsonl"
    with open(p, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return p


def test_update_outcome_matches_across_midnight(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path)
    _write_record(tmp_path, "2026_06", {
        "pair": "AUDUSD", "system": "FOREX", "outcome": None,
        "entry_timestamp": "2026-06-15T12:00:15+00:00"})
    assert dl.update_outcome(pair="AUDUSD", entry_timestamp="2026-06-16T12:01:37+00:00",
                             outcome="WIN", r_realized=1.2, system="FOREX") is True
    rec = json.loads((tmp_path / "decisions_2026_06.jsonl").read_text().strip())
    assert rec["outcome"] == "WIN"


def test_update_outcome_checks_previous_month(tmp_path, monkeypatch):
    """A fill on the 1st can close a signal from the 30th."""
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path)
    _write_record(tmp_path, "2026_06", {
        "pair": "EURUSD", "system": "FOREX", "outcome": None,
        "entry_timestamp": "2026-06-30T20:00:00+00:00"})
    assert dl.update_outcome(pair="EURUSD", entry_timestamp="2026-07-01T14:00:00+00:00",
                             outcome="LOSS", r_realized=-1.0, system="FOREX") is True
    rec = json.loads((tmp_path / "decisions_2026_06.jsonl").read_text().strip())
    assert rec["outcome"] == "LOSS"


def test_already_closed_record_is_not_reclosed(tmp_path, monkeypatch):
    """The guard that made the old alarm fire forever — and it is CORRECT."""
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path)
    _write_record(tmp_path, "2026_06", {
        "pair": "GBPUSD", "system": "FOREX", "outcome": "WIN",
        "entry_timestamp": "2026-06-15T12:00:00+00:00"})
    assert dl.update_outcome(pair="GBPUSD", entry_timestamp="2026-06-15T13:00:00+00:00",
                             outcome="LOSS", r_realized=-1.0, system="FOREX") is False


def test_fifo_oldest_open_closes_first(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "LOG_DIR", tmp_path)
    for ts in ("2026-06-16T09:00:00+00:00", "2026-06-16T11:00:00+00:00"):
        _write_record(tmp_path, "2026_06", {"pair": "EURUSD", "system": "FOREX",
                                            "outcome": None, "entry_timestamp": ts})
    dl.update_outcome(pair="EURUSD", entry_timestamp="2026-06-16T12:00:00+00:00",
                      outcome="WIN", r_realized=1.0, system="FOREX")
    recs = [json.loads(l) for l in
            (tmp_path / "decisions_2026_06.jsonl").read_text().splitlines()]
    assert recs[0]["outcome"] == "WIN", "oldest open must close first"
    assert recs[1]["outcome"] is None
