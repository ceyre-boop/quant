"""Short-locate gating policy.

The two rules under test both exist to protect the measurement rather than to
maximise fills: no stale fallback, and HARD tiers skip.
"""
import json
from datetime import date

import pytest

from execution import borrow

DAY = date(2026, 7, 16)


@pytest.fixture
def locate_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(borrow, "LOCATE_DIR", tmp_path)
    return tmp_path


def _write(dirpath, day, payload):
    (dirpath / f"ib_locate_{day}.json").write_text(json.dumps(payload))


def test_missing_snapshot_returns_none_not_empty(locate_dir):
    """None means UNKNOWN. An empty dict would read as 'nothing is borrowable',
    which is a different and equally wrong claim."""
    assert borrow.load_locate(DAY) is None


def test_no_snapshot_skips_rather_than_allows(locate_dir):
    ok, why = borrow.borrow_ok("AAPL", None)
    assert ok is False
    assert why == "no_locate_snapshot"


def test_never_falls_back_to_stale_snapshot(locate_dir):
    """Yesterday's EASY is not evidence about today's borrow."""
    _write(locate_dir, date(2026, 7, 15), {"TGHL": "EASY"})
    assert borrow.load_locate(DAY) is None       # today has no file -> unknown


def test_easy_tier_is_fillable(locate_dir):
    _write(locate_dir, DAY, {"TGHL": "EASY"})
    ok, why = borrow.borrow_ok("TGHL", borrow.load_locate(DAY))
    assert ok is True and why == "tier_EASY"


@pytest.mark.parametrize("tier", ["HARD", "UNAVAILABLE", "NOT_LISTED"])
def test_non_easy_tiers_skip(locate_dir, tier):
    """HARD skips deliberately: a borrow you might not obtain would flatter the
    short leg, biasing the exact number this harness measures."""
    _write(locate_dir, DAY, {"TGHL": tier})
    ok, why = borrow.borrow_ok("TGHL", borrow.load_locate(DAY))
    assert ok is False and why == f"tier_{tier}"


def test_symbol_absent_from_snapshot_skips(locate_dir):
    _write(locate_dir, DAY, {"OTHER": "EASY"})
    ok, why = borrow.borrow_ok("TGHL", borrow.load_locate(DAY))
    assert ok is False and why == "tier_NOT_LISTED"


def test_case_insensitive(locate_dir):
    _write(locate_dir, DAY, {"tghl": "easy"})
    ok, _ = borrow.borrow_ok("TGHL", borrow.load_locate(DAY))
    assert ok is True


def test_nested_schema_supported(locate_dir):
    _write(locate_dir, DAY, {"symbols": {"TGHL": {"tier": "EASY"}}})
    ok, _ = borrow.borrow_ok("TGHL", borrow.load_locate(DAY))
    assert ok is True


def test_corrupt_snapshot_raises_loudly(locate_dir):
    (locate_dir / f"ib_locate_{DAY}.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="corrupt locate snapshot"):
        borrow.load_locate(DAY)


def test_available_days_discovery(locate_dir):
    _write(locate_dir, date(2026, 7, 15), {})
    _write(locate_dir, date(2026, 7, 16), {})
    (locate_dir / "ib_locate_garbage.json").write_text("{}")
    assert borrow.available_locate_days() == [date(2026, 7, 15), date(2026, 7, 16)]
