"""Behaviour of `sovereign.pit.store.append` — the append-only write path —
against a deterministic, temporary DuckDB. Never against `data/fundamentals.db`.

The property under test is that history cannot be silently destroyed: the
same observation seen twice is a no-op, a genuinely new vintage (a different
published instant for the same identity) is additive, an un-timestamped row
is dropped rather than stored as a silent knowability hole, and a blocked
fact refuses to accept writes at all.

DO NOT "FIX" A FAILURE HERE BY RELAXING THE TEST. A failure means append()
regressed, not that the fixture is wrong.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sovereign.pit import store
from sovereign.pit.errors import NotPointInTime

# "earnings" is used throughout: it is the simplest registered point-in-time
# fact (spec: table=fund_earnings_event, identity=(ticker, fiscal_end),
# published_col=published_ts) and is not touched by the concurrent
# published_ts repoint happening in sovereign/pit/ and sovereign/fundamentals/
# elsewhere in this repo right now.


@pytest.fixture()
def con(tmp_path: Path):
    """A temp DuckDB with a minimal earnings-shaped table, built here rather
    than via the real schema in sovereign/fundamentals/store.py — this file
    must never import or run that module's DDL, only exercise
    sovereign.pit.store.append() against something it owns outright."""
    path = tmp_path / "pit_ingest_test.db"
    c = duckdb.connect(str(path))
    c.execute("""
        CREATE TABLE fund_earnings_event (
            ticker       VARCHAR,
            fiscal_end   DATE,
            eps_actual   DOUBLE,
            published_ts TIMESTAMP
        )
    """)
    yield c
    c.close()


def _row(
    ticker: str = "AAPL",
    fiscal_end: date = date(2024, 3, 31),
    eps_actual: float = 1.0,
    published_ts: datetime | None = datetime(2024, 4, 25, tzinfo=timezone.utc),
) -> dict:
    return {
        "ticker": ticker,
        "fiscal_end": fiscal_end,
        "eps_actual": eps_actual,
        "published_ts": published_ts,
    }


def _row_count(con) -> int:
    return con.execute("SELECT count(*) FROM fund_earnings_event").fetchone()[0]


# ── re-appending the same rows is a no-op ───────────────────────────────────

def test_append_same_rows_twice_yields_identical_row_count(con):
    rows = [_row()]
    first = store.append("earnings", rows, con=con)
    second = store.append("earnings", list(rows), con=con)  # fresh list, same content
    assert first == 1
    assert second == 0, "re-observing an identical vintage must be a no-op"
    assert _row_count(con) == 1


# ── a different published instant for the same identity is a NEW vintage ───

def test_append_restatement_with_new_published_instant_adds_a_row(con):
    original = _row(published_ts=datetime(2024, 4, 25, tzinfo=timezone.utc))
    restated = _row(eps_actual=1.05, published_ts=datetime(2024, 5, 10, tzinfo=timezone.utc))

    store.append("earnings", [original], con=con)
    store.append("earnings", [restated], con=con)

    assert _row_count(con) == 2, "a restatement must be a new vintage, not a replacement"
    published = {
        r[0] for r in con.execute("SELECT published_ts FROM fund_earnings_event").fetchall()
    }
    assert len(published) == 2, "both vintages must survive with distinct published_ts"
    eps_values = {
        r[0] for r in con.execute("SELECT eps_actual FROM fund_earnings_event").fetchall()
    }
    assert eps_values == {1.0, 1.05}, "the original value must not be overwritten"


# ── a row with no published instant is dropped, never stored ───────────────

def test_append_drops_row_with_null_published_without_raising(con):
    row = _row(published_ts=None)
    written = store.append("earnings", [row], con=con)
    assert written == 0
    assert _row_count(con) == 0, "an un-timestamped row must not be stored — it is a silent knowability hole"


def test_append_drops_only_the_null_row_and_keeps_the_rest(con):
    good = _row(ticker="AAPL")
    bad = _row(ticker="MSFT", published_ts=None)
    written = store.append("earnings", [good, bad], con=con)
    assert written == 1
    assert _row_count(con) == 1
    assert con.execute("SELECT ticker FROM fund_earnings_event").fetchone()[0] == "AAPL"


# ── append() on a blocked fact refuses to write at all ──────────────────────

def test_append_on_blocked_fact_raises_not_point_in_time(con):
    # "short_volume" is registered blocked (published_col=None) in
    # sovereign/pit/spec.py — it must never accept a write through this path
    # regardless of what table/columns the caller thinks it has.
    with pytest.raises(NotPointInTime):
        store.append("short_volume", [{"date": date(2024, 1, 1), "ticker": "AAPL"}], con=con)
