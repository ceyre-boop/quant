"""Behaviour of `sovereign.pit.reader.AsOfReader` against a deterministic,
temporary DuckDB — never against `data/fundamentals.db`.

Every fixture here is a KNOWN leak trap taken from the real fundamentals
schema (see sovereign/pit/spec.py docstrings): the 13F filing-lag trap, the
earnings-AMC same-day trap, the NULL-published-ts trap, and the restatement
/ vintage trap. If any of these regress, the layer has stopped doing the one
job it exists to do.

DO NOT "FIX" A FAILURE HERE BY RELAXING THE TEST. A failure means the reader
leaked, not that the fixture is wrong.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sovereign.pit.clock import as_of
from sovereign.pit.errors import NotPointInTime, PitSchemaMismatch, UnknownFact
from sovereign.pit.reader import view
from sovereign.pit.spec import FACTS


# ── fixture: a temp DuckDB with real fundamentals-shaped tables ────────────

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "pit_test.db"


def _connect_factory(path: Path):
    """Mirrors sovereign.pit.store.ro_connect's signature: a zero-arg
    callable returning a connection (or None)."""
    def _connect():
        return duckdb.connect(str(path))
    return _connect


def _build_schema(con) -> None:
    con.execute("""
        CREATE TABLE fund_institution_holding (
            period_end    DATE,
            filer_cik     INTEGER,
            cusip         VARCHAR,
            ticker        VARCHAR,
            filer_name    VARCHAR,
            filing_date   DATE,
            shares        DOUBLE,
            value_usd     DOUBLE,
            is_amendment  BOOLEAN,
            source        VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE fund_earnings_event (
            ticker            VARCHAR,
            fiscal_end        DATE,
            report_date       DATE,
            report_time       VARCHAR,
            eps_estimate      DOUBLE,
            eps_actual        DOUBLE,
            source            VARCHAR,
            published_ts      TIMESTAMP
        )
    """)


@pytest.fixture()
def seeded_db(db_path: Path) -> Path:
    con = duckdb.connect(str(db_path))
    try:
        _build_schema(con)

        # ── 13F trap: period_end in the past, filing_date in the future ────
        # At as_of 2026-04-15 (between period end and filing), this row must
        # be invisible — reading on period_end would leak ~6 weeks.
        con.execute(
            "INSERT INTO fund_institution_holding VALUES "
            "(DATE '2026-03-31', 1234, 'CUSIP1', 'AAPL', 'Berkshire', "
            "DATE '2026-05-15', 1000.0, 500000.0, false, 'sec_edgar')"
        )

        # ── earnings AMC trap: published mid-evening, not at midnight ──────
        con.execute(
            "INSERT INTO fund_earnings_event VALUES "
            "('AAPL', DATE '2026-06-30', DATE '2026-07-30', 'amc', "
            "1.50, 1.55, 'yahoo', TIMESTAMP '2026-07-30 20:30:00')"
        )

        # ── NULL published_ts: must never be returned, at any as_of ────────
        con.execute(
            "INSERT INTO fund_earnings_event VALUES "
            "('MSFT', DATE '2026-06-30', DATE '2026-07-28', 'bmo', "
            "2.00, 2.10, 'yahoo', NULL)"
        )
    finally:
        con.close()
    return db_path


# ── 13F filing-lag trap ──────────────────────────────────────────────────

def test_13f_not_visible_between_period_end_and_filing(seeded_db):
    v = view(as_of("2026-04-15"), connect=_connect_factory(seeded_db))
    rows = v.facts("institutions", "AAPL")
    assert rows == []


def test_13f_visible_after_filing(seeded_db):
    v = view(as_of("2026-06-01"), connect=_connect_factory(seeded_db))
    rows = v.facts("institutions", "AAPL")
    assert len(rows) == 1
    assert rows[0].data["filer_name"] == "Berkshire"


# ── earnings AMC same-day trap ──────────────────────────────────────────

def test_earnings_amc_not_visible_at_midnight_same_day(seeded_db):
    v = view(as_of("2026-07-30"), connect=_connect_factory(seeded_db))
    rows = v.facts("earnings", "AAPL")
    assert rows == []


def test_earnings_amc_visible_next_day(seeded_db):
    v = view(as_of("2026-07-31"), connect=_connect_factory(seeded_db))
    rows = v.facts("earnings", "AAPL")
    assert len(rows) == 1


# ── NULL published_ts is never returned ──────────────────────────────────

def test_null_published_ts_never_returned(seeded_db):
    for when in ("2026-01-01", "2026-08-01", "2099-01-01"):
        v = view(as_of(when), connect=_connect_factory(seeded_db))
        rows = v.facts("earnings", "MSFT")
        assert rows == [], f"NULL published_ts row leaked at as_of={when}"


# ── blocked / unregistered facts ─────────────────────────────────────────

@pytest.mark.parametrize(
    "fact", ["short_volume", "borrow", "institutions_agg", "price_reaction"]
)
def test_blocked_fact_raises_not_point_in_time(seeded_db, fact):
    v = view(as_of("2026-06-01"), connect=_connect_factory(seeded_db))
    with pytest.raises(NotPointInTime):
        v.facts(fact, "AAPL")


def test_unregistered_fact_raises_unknown_fact(seeded_db):
    v = view(as_of("2026-06-01"), connect=_connect_factory(seeded_db))
    with pytest.raises(UnknownFact):
        v.facts("not_a_real_fact", "AAPL")


# ── spec/schema mismatch raises loudly, never returns [] ────────────────

def test_schema_mismatch_raises_pit_schema_mismatch(tmp_path):
    """A table missing its declared published column must fail loud, not
    return an empty (and therefore indistinguishable-from-true-negative)
    result."""
    path = tmp_path / "mismatch.db"
    con = duckdb.connect(str(path))
    try:
        # earnings spec expects a published_ts column; this table has none.
        con.execute("""
            CREATE TABLE fund_earnings_event (
                ticker      VARCHAR,
                fiscal_end  DATE,
                report_date DATE
            )
        """)
        con.execute("INSERT INTO fund_earnings_event VALUES ('AAPL', DATE '2026-06-30', DATE '2026-07-30')")
    finally:
        con.close()

    v = view(as_of("2026-08-01"), connect=_connect_factory(path))
    with pytest.raises(PitSchemaMismatch):
        v.facts("earnings", "AAPL")


def test_missing_table_returns_empty_not_mismatch(tmp_path):
    """A table that simply doesn't exist yet (nothing harvested) is benign
    and must return [] rather than raise — distinct from a schema that
    exists but disagrees with the spec."""
    path = tmp_path / "empty.db"
    con = duckdb.connect(str(path))
    con.close()
    v = view(as_of("2026-08-01"), connect=_connect_factory(path))
    assert v.facts("earnings", "AAPL") == []


# ── restatement / vintage trap — the important one ──────────────────────

@pytest.fixture()
def vintage_db(db_path: Path) -> Path:
    """Two vintages of the SAME (ticker, fiscal_end) identity: a pre-print
    consensus estimate row, then a restated/actual row published later."""
    con = duckdb.connect(str(db_path))
    try:
        _build_schema(con)
        con.execute(
            "INSERT INTO fund_earnings_event VALUES "
            "('AAPL', DATE '2026-06-30', DATE '2026-07-30', 'amc', "
            "1.40, NULL, 'estimate', TIMESTAMP '2026-07-01 00:00:00')"
        )
        con.execute(
            "INSERT INTO fund_earnings_event VALUES "
            "('AAPL', DATE '2026-06-30', DATE '2026-07-30', 'amc', "
            "1.40, 1.55, 'actual', TIMESTAMP '2026-07-30 20:30:00')"
        )
    finally:
        con.close()
    return db_path


def test_vintage_between_publications_sees_old_value(vintage_db):
    v = view(as_of("2026-07-15"), connect=_connect_factory(vintage_db))
    rows = v.facts("earnings", "AAPL")
    assert len(rows) == 1
    assert rows[0].data["source"] == "estimate"
    assert rows[0].data["eps_actual"] is None


def test_vintage_after_both_sees_new_value(vintage_db):
    v = view(as_of("2026-08-01"), connect=_connect_factory(vintage_db))
    rows = v.facts("earnings", "AAPL")
    assert len(rows) == 1
    assert rows[0].data["source"] == "actual"
    assert rows[0].data["eps_actual"] == pytest.approx(1.55)


def test_vintages_returns_both(vintage_db):
    v = view(as_of("2026-08-01"), connect=_connect_factory(vintage_db))
    rows = v.vintages("earnings", "AAPL")
    assert len(rows) == 2
    sources = {r.data["source"] for r in rows}
    assert sources == {"estimate", "actual"}


def test_latest_only_collapses_to_one(vintage_db):
    v = view(as_of("2026-08-01"), connect=_connect_factory(vintage_db))
    rows = v.facts("earnings", "AAPL", latest_only=True)
    assert len(rows) == 1


# ── monotonicity property ────────────────────────────────────────────────

def test_row_count_is_monotonic_and_cut_always_holds(vintage_db):
    dates = ["2026-06-01", "2026-07-02", "2026-07-15", "2026-07-31", "2026-09-01"]
    prev_count = 0
    for d in dates:
        at = as_of(d)
        v = view(at, connect=_connect_factory(vintage_db))
        rows = v.vintages("earnings", "AAPL")
        assert len(rows) >= prev_count, f"row count decreased at as_of={d}"
        for r in rows:
            assert r.published_ts < at.ts
        prev_count = len(rows)
    assert prev_count == 2  # sanity: both vintages eventually surface


# ── truncation invariance ────────────────────────────────────────────────

def test_truncation_invariance(vintage_db, tmp_path):
    """Reading as-of T from the full table must equal reading as-of T from a
    table physically truncated to rows published < T. If these disagree,
    the SQL cut and the data itself are telling two different stories."""
    cutoff = datetime(2026, 7, 31, tzinfo=timezone.utc)
    at = as_of(cutoff)

    full = view(at, connect=_connect_factory(vintage_db)).facts(
        "earnings", "AAPL", latest_only=False
    )

    truncated_path = tmp_path / "truncated.db"
    src = duckdb.connect(str(vintage_db))
    try:
        # Bind NAIVE UTC, exactly as the reader does. Binding the tz-aware
        # `cutoff` against a naive TIMESTAMP column makes DuckDB resolve the
        # comparison via the session's local timezone, so this truncation would
        # itself become machine-dependent — reproducing the very bug this suite
        # caught rather than testing against it. (Failed under TZ=America/Detroit.)
        rows = src.execute(
            "SELECT * FROM fund_earnings_event WHERE published_ts < ?",
            [cutoff.astimezone(timezone.utc).replace(tzinfo=None)],
        ).fetchall()
        cols = [d[0] for d in src.description]
    finally:
        src.close()

    tcon = duckdb.connect(str(truncated_path))
    try:
        _build_schema(tcon)
        placeholders = ", ".join("?" * len(cols))
        for r in rows:
            tcon.execute(f"INSERT INTO fund_earnings_event VALUES ({placeholders})", list(r))
    finally:
        tcon.close()

    truncated = view(at, connect=_connect_factory(truncated_path)).facts(
        "earnings", "AAPL", latest_only=False
    )

    full_keys = sorted((o.published_ts, o.data["source"]) for o in full)
    trunc_keys = sorted((o.published_ts, o.data["source"]) for o in truncated)
    assert full_keys == trunc_keys


# ── every registered point-in-time fact is reachable through this reader ──

def test_all_point_in_time_facts_are_declared_with_a_table():
    """Sanity check that the spec this file exercises is the same spec the
    reader consults — guards against a stale duplicate spec creeping in."""
    pit_facts = [f for f in FACTS.values() if f.is_point_in_time]
    assert pit_facts, "expected at least one point-in-time fact in the spec"
    for f in pit_facts:
        assert f.published_col is not None
        assert f.table


# ── regression: the as-of cut must not depend on the session timezone ──────
#
# DO NOT "FIX" A FAILURE HERE BY RELAXING THE TEST.
#
# The reader once bound a tz-AWARE datetime against DuckDB's naive TIMESTAMP
# columns. DuckDB resolves such a comparison using the SESSION'S LOCAL TIMEZONE,
# so the same as-of read returned different rows on different machines:
#
#   column TIMESTAMP '2026-07-30 20:30:00', param 2026-07-31T00:00Z
#     TimeZone=UTC             -> True   (correct)
#     TimeZone=America/Detroit -> False  (wrong — silently drops the row)
#
# A point-in-time answer that depends on where it was computed is not a
# point-in-time answer.
@pytest.mark.parametrize(
    "session_tz",
    ["UTC", "America/Detroit", "Asia/Tokyo", "Pacific/Auckland", "America/Los_Angeles"],
)
def test_as_of_cut_is_timezone_independent(seeded_db, session_tz):
    def connect_in_tz():
        con = duckdb.connect(str(seeded_db), read_only=True)
        con.execute(f"SET TimeZone='{session_tz}'")
        return con

    # AMC print published 2026-07-30 20:30 UTC.
    before = view(as_of("2026-07-30"), connect=connect_in_tz).facts("earnings", "AAPL")
    after = view(as_of("2026-07-31"), connect=connect_in_tz).facts("earnings", "AAPL")

    assert before == [], f"leaked under TimeZone={session_tz}"
    assert len(after) == 1, f"silently dropped a knowable row under TimeZone={session_tz}"
