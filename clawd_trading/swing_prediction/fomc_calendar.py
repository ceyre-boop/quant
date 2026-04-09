"""
FOMC Calendar - Federal Reserve Meeting Dates

Provides real FOMC meeting dates and calculates timing scores.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FOMCCalendar:
    """Federal Reserve FOMC meeting calendar."""

    # 2026 FOMC Meeting Dates (scheduled)
    MEETINGS_2026 = [
        datetime(2026, 1, 28),  # Jan 27-28
        datetime(2026, 3, 18),  # Mar 17-18
        datetime(2026, 4, 29),  # Apr 28-29
        datetime(2026, 6, 17),  # Jun 16-17
        datetime(2026, 7, 29),  # Jul 28-29
        datetime(2026, 9, 16),  # Sep 15-16
        datetime(2026, 10, 28),  # Oct 27-28
        datetime(2026, 12, 16),  # Dec 15-16
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_meetings_for_year(self, year: int) -> List[datetime]:
        """Get all FOMC meetings for a given year."""
        if year == 2026:
            return self.MEETINGS_2026
        # For other years, would need to fetch from external source
        return []

    def get_next_meeting(self, from_date: datetime = None) -> Optional[datetime]:
        """Get the next FOMC meeting from a given date."""
        if from_date is None:
            from_date = datetime.now()

        for meeting in self.MEETINGS_2026:
            if meeting > from_date:
                return meeting

        return None

    def get_timing_score(self, date: datetime = None) -> float:
        """
        Calculate FOMC timing score for a given date.

        Returns:
            -1.0: Pre-meeting week (avoid)
            -0.5: Pre-meeting 2 weeks (caution)
            0.0: Neutral
            +1.0: Post-meeting week (opportunity)
        """
        if date is None:
            date = datetime.now()

        next_meeting = self.get_next_meeting(date)

        if next_meeting is None:
            return 0.0  # No upcoming meetings

        days_to_meeting = (next_meeting - date).days

        # Pre-meeting caution
        if days_to_meeting <= 7:
            return -1.0  # Last week before - extreme caution
        elif days_to_meeting <= 14:
            return -0.5  # Two weeks before - caution
        elif days_to_meeting <= 21:
            return -0.3  # Three weeks before - mild caution

        # Post-meeting opportunity (check previous meeting)
        prev_meeting = None
        for meeting in reversed(self.MEETINGS_2026):
            if meeting < date:
                prev_meeting = meeting
                break

        if prev_meeting:
            days_since_meeting = (date - prev_meeting).days
            if days_since_meeting <= 5:
                return 1.0  # First week after - opportunity
            elif days_since_meeting <= 10:
                return 0.5  # Second week after - mild opportunity

        return 0.0  # Neutral zone

    def get_meeting_context(self, date: datetime = None) -> Dict:
        """Get full FOMC context for a date."""
        if date is None:
            date = datetime.now()

        next_meeting = self.get_next_meeting(date)
        prev_meeting = None

        for meeting in reversed(self.MEETINGS_2026):
            if meeting < date:
                prev_meeting = meeting
                break

        return {
            "next_meeting": next_meeting.isoformat() if next_meeting else None,
            "previous_meeting": prev_meeting.isoformat() if prev_meeting else None,
            "days_to_next": (next_meeting - date).days if next_meeting else None,
            "days_since_previous": (date - prev_meeting).days if prev_meeting else None,
            "timing_score": self.get_timing_score(date),
            "zone": self._get_zone(date),
        }

    def _get_zone(self, date: datetime) -> str:
        """Get FOMC timing zone."""
        score = self.get_timing_score(date)

        if score <= -1.0:
            return "pre_meeting_week"
        elif score <= -0.5:
            return "pre_meeting_2weeks"
        elif score < 0:
            return "pre_meeting_3weeks"
        elif score >= 1.0:
            return "post_meeting_week"
        elif score > 0:
            return "post_meeting_decay"
        else:
            return "neutral"


# Singleton instance
_fomc_calendar = None


def get_fomc_calendar() -> FOMCCalendar:
    """Get or create FOMC calendar instance."""
    global _fomc_calendar
    if _fomc_calendar is None:
        _fomc_calendar = FOMCCalendar()
    return _fomc_calendar


def get_fomc_timing_score(date: datetime = None) -> float:
    """Convenience function for FOMC timing score."""
    return get_fomc_calendar().get_timing_score(date)
