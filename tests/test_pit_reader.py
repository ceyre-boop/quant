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

from datetime import date, datetime, timedelta, timezone
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
            -- published_ts gates knowability as of the sovereign/pit/spec.py
            -- repoint (institutions moved off the DATE-only filing_date onto
            -- the EDGAR acceptance instant, same reasoning as insider below).
            -- filing_date is kept alongside it for display/audit, matching
            -- the real fund_institution_holding schema.
            published_ts  TIMESTAMP,
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
    # fund_insider_txn — the largest point-in-time fact (860 rows live) and,
    # before this file, entirely uncovered by reader tests. Columns mirror
    # sovereign/fundamentals/store.py exactly, including published_ts (the
    # EDGAR acceptance instant the pit/spec.py repoint now gates knowability
    # on — filing_date, a bare DATE, is kept only for display/audit).
    con.execute("""
        CREATE TABLE fund_insider_txn (
            accession      VARCHAR,
            line_no        INTEGER,
            ticker         VARCHAR,
            issuer_cik     INTEGER,
            owner_name     VARCHAR,
            owner_title    VARCHAR,
            is_director    BOOLEAN,
            is_officer     BOOLEAN,
            is_ten_pct     BOOLEAN,
            txn_date       DATE,
            filing_date    DATE,
            published_ts   TIMESTAMP,
            code           VARCHAR,
            shares         DOUBLE,
            price          DOUBLE,
            value_usd      DOUBLE,
            shares_after   DOUBLE,
            is_open_market BOOLEAN,
            source         VARCHAR,
            fetched_at     TIMESTAMP
        )
    """)
    # Needed only for the leakage sweep (test_leakage_sweep_*), which
    # parametrises over every point-in-time fact the fixtures can build.
    con.execute("""
        CREATE TABLE fund_short_interest (
            settlement_date  DATE,
            ticker           VARCHAR,
            shares_short     DOUBLE,
            avg_daily_volume DOUBLE,
            days_to_cover    DOUBLE,
            pct_float        DOUBLE,
            source           VARCHAR,
            published_ts     TIMESTAMP,
            fetched_at       TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE fund_estimate_snapshot (
            snapshot_date DATE,
            ticker        VARCHAR,
            period        VARCHAR,
            period_end    DATE,
            eps_avg       DOUBLE,
            eps_low       DOUBLE,
            eps_high      DOUBLE,
            n_analysts    INTEGER,
            up_30d        INTEGER,
            down_30d      INTEGER,
            source        VARCHAR,
            fetched_at    TIMESTAMP
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
        # published_ts is what actually gates knowability (see spec.py); it
        # is set to the same instant as filing_date here since this fixture
        # only needs day-level resolution, not the acceptance-time trap that
        # the insider fixture below covers.
        con.execute(
            "INSERT INTO fund_institution_holding "
            "(period_end, filer_cik, cusip, ticker, filer_name, filing_date, "
            " published_ts, shares, value_usd, is_amendment, source) VALUES "
            "(DATE '2026-03-31', 1234, 'CUSIP1', 'AAPL', 'Berkshire', "
            "DATE '2026-05-15', TIMESTAMP '2026-05-15 00:00:00', "
            "1000.0, 500000.0, false, 'sec_edgar')"
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


# ── Form 4 lag trap: fund_insider_txn (860 rows, the LARGEST point-in-time
# fact, entirely uncovered by reader tests before this) ────────────────────

_INSIDER_SPEC = FACTS["insider"]


@pytest.fixture()
def insider_db(db_path: Path) -> Path:
    """Two Form 4 cases, both columns populated (filing_date DATE AND
    published_ts TIMESTAMP) so this test is correct regardless of which one
    sovereign.pit.spec.FACTS['insider'].published_col currently names — the
    reader is driven by the spec, never a column literal in this file, and
    every assertion below reads Observation.published_ts (the reader's own
    generic abstraction over that column), never a raw dict key.
    """
    con = duckdb.connect(str(db_path))
    try:
        _build_schema(con)

        # Case A — the spec's verbatim example: a transaction on 2024-02-01,
        # filed/accepted 2024-02-05. Invisible on the 3rd, visible on the 6th.
        con.execute(
            "INSERT INTO fund_insider_txn "
            "(accession, line_no, ticker, issuer_cik, owner_name, owner_title, "
            " is_director, is_officer, is_ten_pct, txn_date, filing_date, "
            " published_ts, code, shares, price, value_usd, shares_after, "
            " is_open_market, source, fetched_at) VALUES "
            "('0001-A', 1, 'AAPL', 320193, 'Insider A', 'CEO', true, true, false, "
            " DATE '2024-02-01', DATE '2024-02-05', "
            " TIMESTAMP '2024-02-05 00:00:00', 'S', 100.0, 50.0, 5000.0, 900.0, "
            " true, 'sec_edgar', TIMESTAMP '2024-02-05 00:00:00')"
        )

        # Case B — the real-world ~22h leak: accepted 2024-02-05T23:30:00Z
        # (18:30 ET, AFTER the close), but filing_date — a bare DATE — is only
        # 2024-02-05. A date-only knowable_at would already consider this
        # knowable at midnight that day, and therefore visible at noon; the
        # acceptance instant must not.
        con.execute(
            "INSERT INTO fund_insider_txn "
            "(accession, line_no, ticker, issuer_cik, owner_name, owner_title, "
            " is_director, is_officer, is_ten_pct, txn_date, filing_date, "
            " published_ts, code, shares, price, value_usd, shares_after, "
            " is_open_market, source, fetched_at) VALUES "
            "('0001-B', 1, 'MSFT', 789019, 'Insider B', 'CFO', false, true, false, "
            " DATE '2024-02-01', DATE '2024-02-05', "
            " TIMESTAMP '2024-02-05 23:30:00', 'S', 200.0, 300.0, 60000.0, "
            " 1800.0, true, 'sec_edgar', TIMESTAMP '2024-02-05 23:30:00')"
        )
    finally:
        con.close()
    return db_path


def test_insider_form4_invisible_before_filing(insider_db):
    v = view(as_of("2024-02-03"), connect=_connect_factory(insider_db))
    assert v.facts("insider", "AAPL") == []


def test_insider_form4_visible_after_filing(insider_db):
    v = view(as_of("2024-02-06"), connect=_connect_factory(insider_db))
    rows = v.facts("insider", "AAPL")
    assert len(rows) == 1
    assert rows[0].data["accession"] == "0001-A"
    assert rows[0].published_ts is not None


def test_insider_form4_late_accept_invisible_midday_same_day(insider_db):
    """The real ~22h leak a date-only knowable_at caused: a Form 4 accepted
    at 23:30 UTC (18:30 ET, after the close) is not knowable at noon UTC the
    same day, even though its filing_date (date-only) is that same day."""
    v = view(as_of("2024-02-05T12:00:00Z"), connect=_connect_factory(insider_db))
    rows = v.facts("insider", "MSFT")
    assert rows == [], "Form 4 accepted after the close leaked ~22h early"


def test_insider_form4_late_accept_visible_next_day(insider_db):
    v = view(as_of("2024-02-06"), connect=_connect_factory(insider_db))
    rows = v.facts("insider", "MSFT")
    assert len(rows) == 1
    assert rows[0].data["accession"] == "0001-B"


# ── leakage sweep: every point-in-time fact, a uniform one-year gap ────────
#
# Not just earnings: every fact whose fixture we can build gets the same
# trap — knowable_at is exactly one year after event_date, several tickers,
# several as-of dates strictly inside the gap. The second half (visible AFTER
# the gap) is essential: a reader that always returned [] would pass the
# empty-inside-gap half vacuously.

_LEAK_TICKERS = ("AAPL", "MSFT", "GOOG")
_LEAK_EVENT = date(2020, 1, 15)
_LEAK_PUBLISHED_DATE = _LEAK_EVENT + timedelta(days=365)  # 2021-01-14
_LEAK_PUBLISHED_TS = datetime(
    _LEAK_PUBLISHED_DATE.year, _LEAK_PUBLISHED_DATE.month, _LEAK_PUBLISHED_DATE.day,
    tzinfo=timezone.utc,
)

_LEAK_FACTS = ("earnings", "insider", "institutions", "short_interest", "estimates")


def _seed_leak_row(con, fact_name: str, ticker: str) -> None:
    if fact_name == "earnings":
        con.execute(
            "INSERT INTO fund_earnings_event "
            "(ticker, fiscal_end, report_date, report_time, eps_estimate, "
            " eps_actual, source, published_ts) VALUES "
            "(?, ?, ?, 'amc', 1.0, 1.0, 'sweep', ?)",
            [ticker, _LEAK_EVENT, _LEAK_EVENT, _LEAK_PUBLISHED_TS],
        )
    elif fact_name == "insider":
        con.execute(
            "INSERT INTO fund_insider_txn "
            "(accession, line_no, ticker, issuer_cik, owner_name, owner_title, "
            " is_director, is_officer, is_ten_pct, txn_date, filing_date, "
            " published_ts, code, shares, price, value_usd, shares_after, "
            " is_open_market, source, fetched_at) VALUES "
            "(?, 1, ?, 1, 'Sweep Insider', 'CEO', true, true, false, "
            " ?, ?, ?, 'S', 1.0, 1.0, 1.0, 1.0, true, 'sweep', ?)",
            [
                f"ACC-{ticker}", ticker, _LEAK_EVENT, _LEAK_PUBLISHED_DATE,
                _LEAK_PUBLISHED_TS, _LEAK_PUBLISHED_TS,
            ],
        )
    elif fact_name == "institutions":
        con.execute(
            "INSERT INTO fund_institution_holding "
            "(period_end, filer_cik, cusip, ticker, filer_name, filing_date, "
            " published_ts, shares, value_usd, is_amendment, source) VALUES "
            "(?, 1, ?, ?, 'Sweep Filer', ?, ?, 1.0, 1.0, false, 'sweep')",
            [
                _LEAK_EVENT, f"CUSIP-{ticker}", ticker, _LEAK_PUBLISHED_DATE,
                _LEAK_PUBLISHED_TS,
            ],
        )
    elif fact_name == "short_interest":
        con.execute(
            "INSERT INTO fund_short_interest "
            "(settlement_date, ticker, shares_short, avg_daily_volume, "
            " days_to_cover, pct_float, source, published_ts, fetched_at) "
            "VALUES (?, ?, 1.0, 1.0, 1.0, 1.0, 'sweep', ?, ?)",
            [_LEAK_EVENT, ticker, _LEAK_PUBLISHED_TS, _LEAK_PUBLISHED_TS],
        )
    elif fact_name == "estimates":
        con.execute(
            "INSERT INTO fund_estimate_snapshot "
            "(snapshot_date, ticker, period, period_end, eps_avg, eps_low, "
            " eps_high, n_analysts, up_30d, down_30d, source, fetched_at) "
            "VALUES (?, ?, '0q', ?, 1.0, 1.0, 1.0, 1, 0, 0, 'sweep', ?)",
            [_LEAK_PUBLISHED_DATE, ticker, _LEAK_EVENT, _LEAK_PUBLISHED_TS],
        )
    else:  # pragma: no cover - guard against silently skipping a new fact
        raise ValueError(f"no leak-sweep row builder for fact {fact_name!r}")


@pytest.fixture()
def leak_db(db_path: Path) -> Path:
    con = duckdb.connect(str(db_path))
    try:
        _build_schema(con)
        for fact_name in _LEAK_FACTS:
            for ticker in _LEAK_TICKERS:
                _seed_leak_row(con, fact_name, ticker)
    finally:
        con.close()
    return db_path


@pytest.mark.parametrize("fact_name", _LEAK_FACTS)
def test_leakage_sweep_empty_inside_one_year_gap(leak_db, fact_name):
    for when in ("2020-02-01", "2020-06-15", "2020-12-31"):
        v = view(as_of(when), connect=_connect_factory(leak_db))
        for ticker in _LEAK_TICKERS:
            rows = v.facts(fact_name, ticker)
            assert rows == [], f"{fact_name}/{ticker} leaked at as_of={when}"


@pytest.mark.parametrize("fact_name", _LEAK_FACTS)
def test_leakage_sweep_visible_after_the_gap(leak_db, fact_name):
    """Essential complement to the empty-inside-gap test above: without this,
    a reader that always returns [] would pass that test vacuously."""
    v = view(as_of("2021-06-01"), connect=_connect_factory(leak_db))
    for ticker in _LEAK_TICKERS:
        rows = v.facts(fact_name, ticker)
        assert len(rows) == 1, f"{fact_name}/{ticker} never surfaced after its gap"


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
