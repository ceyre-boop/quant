"""Unit tests for ict/session_classifier.py"""
from __future__ import annotations

from datetime import datetime, timezone, time

import pytest

from ict.session_classifier import SessionClassifier, KillZoneStatus, SessionWindow


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _utc(hour: int, minute: int = 0) -> datetime:
    """Build a naive UTC datetime (today) at the given hour:minute."""
    return datetime(2024, 3, 15, hour, minute, 0)  # March = EDT season


def _utc_aware(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 3, 15, hour, minute, 0, tzinfo=timezone.utc)


# ── SessionClassifier construction ────────────────────────────────────────── #

class TestSessionClassifierInit:
    def test_loads_default_windows(self):
        clf = SessionClassifier()
        assert len(clf.windows) >= 4

    def test_window_names(self):
        clf = SessionClassifier()
        names = {w.name for w in clf.windows}
        assert "London" in names
        assert "NY_Open" in names
        assert "NY_PM" in names
        assert "Asia" in names

    def test_hp_zones_flagged(self):
        clf = SessionClassifier()
        hp = {w.name for w in clf.windows if w.is_high_probability}
        assert "London" in hp
        assert "NY_Open" in hp
        assert "NY_PM" in hp
        # Asia is NOT HP by default
        assert "Asia" not in hp


# ── Kill Zone detection ───────────────────────────────────────────────────── #

class TestKillZoneDetection:
    def setup_method(self):
        self.clf = SessionClassifier()

    def test_london_kill_zone_start(self):
        status = self.clf.classify(_utc(2, 30))
        assert status.in_kill_zone
        assert status.kill_zone_name == "London"
        assert status.is_high_probability

    def test_london_kill_zone_boundary(self):
        # 05:00 UTC = end of London window
        status = self.clf.classify(_utc(5, 0))
        assert status.in_kill_zone
        assert status.kill_zone_name == "London"

    def test_ny_open_kill_zone(self):
        status = self.clf.classify(_utc(8, 0))
        assert status.in_kill_zone
        assert status.kill_zone_name == "NY_Open"
        assert status.is_high_probability

    def test_ny_pm_kill_zone(self):
        status = self.clf.classify(_utc(14, 0))
        assert status.in_kill_zone
        assert status.kill_zone_name == "NY_PM"
        assert status.is_high_probability

    def test_asia_session_not_hp(self):
        status = self.clf.classify(_utc(21, 0))
        assert status.in_kill_zone
        assert status.kill_zone_name == "Asia"
        assert not status.is_high_probability

    def test_off_hours_not_in_kz(self):
        # 06:00 UTC = gap between London and NY Open
        status = self.clf.classify(_utc(6, 0))
        assert not status.in_kill_zone
        assert status.kill_zone_name is None

    def test_tz_aware_input(self):
        # Same as 08:00 UTC but passed as tz-aware
        status = self.clf.classify(_utc_aware(8, 0))
        assert status.in_kill_zone
        assert status.kill_zone_name == "NY_Open"


# ── NY Lunch detection ────────────────────────────────────────────────────── #

class TestNYLunch:
    def setup_method(self):
        self.clf = SessionClassifier()

    def test_in_ny_lunch(self):
        # 12:30 UTC = NY Lunch
        status = self.clf.classify(_utc(12, 30))
        assert status.in_ny_lunch

    def test_before_ny_lunch(self):
        status = self.clf.classify(_utc(11, 59))
        assert not status.in_ny_lunch

    def test_after_ny_lunch(self):
        status = self.clf.classify(_utc(13, 31))
        assert not status.in_ny_lunch


# ── should_trade logic ────────────────────────────────────────────────────── #

class TestShouldTrade:
    def setup_method(self):
        self.clf = SessionClassifier()

    def test_should_trade_london(self):
        status = self.clf.classify(_utc(3, 0))
        assert status.should_trade

    def test_should_trade_ny_open(self):
        status = self.clf.classify(_utc(9, 0))
        assert status.should_trade

    def test_should_not_trade_ny_lunch(self):
        # 12:30 UTC falls in NY Lunch
        status = self.clf.classify(_utc(12, 30))
        assert not status.should_trade

    def test_should_not_trade_off_hours(self):
        # 17:00 UTC — no kill zone
        status = self.clf.classify(_utc(17, 0))
        assert not status.should_trade

    def test_should_not_trade_asia_only(self):
        status = self.clf.classify(_utc(21, 0))
        assert not status.should_trade


# ── SessionWindow helper ──────────────────────────────────────────────────── #

class TestSessionWindow:
    def test_contains_simple(self):
        w = SessionWindow("Test", time(8, 0), time(10, 0), True)
        assert w.contains(time(9, 0))
        assert w.contains(time(8, 0))
        assert w.contains(time(10, 0))
        assert not w.contains(time(7, 59))
        assert not w.contains(time(10, 1))
