"""The knowability rule itself: `sovereign.pit.clock`.

Everything downstream (reader, store, the enforcement wall) trusts that
`AsOf.knows()` is STRICT and that `as_of()` cannot be called with nothing.
If either of those drifts, every "no lookahead" claim in the rest of the
suite is unearned. This file is the one place that checks the rule in
isolation, with no DB involved.
"""
from __future__ import annotations

import inspect
from datetime import date, datetime, timezone

import pytest

from sovereign.pit.clock import AsOf, as_of, as_of_now
from sovereign.pit.errors import AsOfRequired, LookaheadError


# ── strict `<`, not `<=` ────────────────────────────────────────────────────

def test_knows_is_strict_before():
    at = as_of("2026-05-15")
    assert at.knows(datetime(2026, 5, 14, tzinfo=timezone.utc)) is True


def test_knows_exactly_equal_is_false():
    """Tie goes to 'not knowable' — a filing timestamped at the decision
    instant was not readable before the decision. This is the single most
    important boundary in the layer."""
    at = as_of(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
    assert at.knows(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)) is False


def test_knows_after_is_false():
    at = as_of("2026-05-15")
    assert at.knows(datetime(2026, 5, 16, tzinfo=timezone.utc)) is False


def test_knows_none_published_is_never_knowable():
    at = as_of("2099-01-01")  # arbitrarily far in the future — still False
    assert at.knows(None) is False


# ── as_of(None) / empty / wrong type raises, never defaults ────────────────

def test_as_of_none_raises():
    with pytest.raises(AsOfRequired):
        as_of(None)


def test_as_of_empty_string_raises():
    with pytest.raises(AsOfRequired):
        as_of("")


def test_as_of_whitespace_string_raises():
    with pytest.raises(AsOfRequired):
        as_of("   ")


def test_as_of_wrong_type_raises():
    with pytest.raises(AsOfRequired):
        as_of(12345)  # type: ignore[arg-type]


def test_as_of_unparseable_string_raises():
    with pytest.raises(AsOfRequired):
        as_of("not-a-date")


# ── naive datetimes are UTC, never local ────────────────────────────────────

def test_naive_datetime_is_treated_as_utc():
    naive = datetime(2026, 6, 1, 9, 30)  # no tzinfo
    at = as_of(naive)
    assert at.ts.tzinfo is not None
    assert at.ts.utcoffset() == timezone.utc.utcoffset(None)
    assert at.ts == datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc)


def test_naive_date_is_midnight_utc():
    at = as_of(date(2026, 6, 1))
    assert at.ts == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


# ── bare date string means midnight UTC START of day ───────────────────────

def test_bare_date_string_is_start_of_day_utc():
    at = as_of("2026-03-03")
    assert at.ts == datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def test_bare_date_string_excludes_same_day_publication():
    """This is the conservative reading a daily-bar backtest needs:
    as_of('2026-03-03') means 'as the day opened', so anything published
    later that same day was not yet knowable."""
    at = as_of("2026-03-03")
    published_same_day_morning = datetime(2026, 3, 3, 9, 0, tzinfo=timezone.utc)
    assert at.knows(published_same_day_morning) is False


def test_bare_date_string_includes_prior_day_publication():
    at = as_of("2026-03-03")
    published_day_before = datetime(2026, 3, 2, 23, 59, tzinfo=timezone.utc)
    assert at.knows(published_day_before) is True


# ── as_of(AsOf) is idempotent ───────────────────────────────────────────────

def test_as_of_of_asof_is_idempotent():
    first = as_of("2026-04-01")
    second = as_of(first)
    assert first == second
    assert second.ts == first.ts


# ── as_of_now() is real, separate, and not silently substitutable ─────────

def test_as_of_now_returns_utc_aware_asof():
    now = as_of_now()
    assert isinstance(now, AsOf)
    assert now.ts.tzinfo is not None


def test_as_of_and_as_of_now_are_distinct_callables():
    """There is no default 'now' path through as_of(); a caller who wants the
    present must type the more awkward, more visible name."""
    assert as_of is not as_of_now
    assert inspect.signature(as_of_now).parameters == {}


def test_as_of_has_no_default_argument():
    """Nobody can call as_of() with no argument and get something plausible."""
    sig = inspect.signature(as_of)
    (param,) = sig.parameters.values()
    assert param.default is inspect.Parameter.empty


# ── AsOf construction is guarded even when built directly ──────────────────

def test_asof_rejects_naive_datetime_directly():
    with pytest.raises(AsOfRequired):
        AsOf(datetime(2026, 1, 1))  # no tzinfo, bypasses as_of()


def test_asof_rejects_non_datetime():
    with pytest.raises(AsOfRequired):
        AsOf("2026-01-01")  # type: ignore[arg-type]


# ── assert_knows raises LookaheadError with the offending timestamps ───────

def test_assert_knows_raises_with_timestamps_in_message():
    at = as_of("2026-05-15")
    published = datetime(2026, 5, 16, tzinfo=timezone.utc)
    with pytest.raises(LookaheadError) as exc:
        at.assert_knows("earnings", published, source="test_source")
    msg = str(exc.value)
    assert "2026-05-16" in msg
    assert "2026-05-15" in msg
    assert "earnings" in msg
    assert "test_source" in msg


def test_assert_knows_passes_silently_when_knowable():
    at = as_of("2026-05-16")
    published = datetime(2026, 5, 15, tzinfo=timezone.utc)
    at.assert_knows("earnings", published)  # must not raise


def test_assert_knows_raises_on_none_published():
    at = as_of("2026-05-16")
    with pytest.raises(LookaheadError):
        at.assert_knows("earnings", None)
