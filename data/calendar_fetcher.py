"""Calendar Fetcher - Economic calendar events."""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


class CalendarFetcher:
    """Fetches economic calendar events."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.tradingeconomics.com",
    ):
        self.api_key = api_key or os.getenv("TRADING_ECON_API_KEY")
        self.base_url = base_url

    def fetch_events(self, country: str = "united states", days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Fetch economic calendar events.

        Args:
            country: Country filter
            days_ahead: Number of days to look ahead

        Returns:
            List of calendar events
        """
        if not self.api_key:
            logger.warning("TRADING_ECON_API_KEY not set, returning mock data")
            return self._get_mock_events()

        try:
            endpoint = f"/calendar/country/{country}"
            url = urljoin(self.base_url, endpoint)

            params = {"c": self.api_key, "format": "json"}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            events = response.json()

            # Filter to upcoming events
            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)

            filtered_events = []
            for event in events:
                try:
                    event_date = datetime.strptime(event.get("Date", ""), "%Y-%m-%dT%H:%M:%S").date()
                    if today <= event_date <= end_date:
                        filtered_events.append(event)
                except (ValueError, TypeError):
                    continue

            return filtered_events

        except Exception as e:
            logger.error(f"Failed to fetch calendar events: {e}")
            return self._get_mock_events()

    def get_high_impact_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for high impact events.

        High impact events include:
        - FOMC announcements
        - Non-farm payrolls
        - CPI releases
        - GDP reports
        """
        high_impact_keywords = [
            "fed",
            "fomc",
            "interest rate",
            "nfp",
            "non-farm payrolls",
            "cpi",
            "inflation",
            "gdp",
            "employment",
            "unemployment",
        ]

        high_impact = []
        for event in events:
            event_name = event.get("Event", "").lower()
            importance = event.get("Importance", 1)

            if importance >= 3 or any(kw in event_name for kw in high_impact_keywords):
                high_impact.append(event)

        return high_impact

    def _get_mock_events(self) -> List[Dict[str, Any]]:
        """Return mock events for testing."""
        return [
            {
                "Date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "Country": "United States",
                "Event": "FOMC Statement",
                "Importance": 3,
                "Actual": "",
                "Forecast": "",
                "Previous": "",
            }
        ]

    def calculate_event_risk(self, events: List[Dict[str, Any]]) -> str:
        """Calculate event risk level for the day.

        Returns:
            'CLEAR', 'ELEVATED', 'HIGH', or 'EXTREME'
        """
        high_impact = self.get_high_impact_events(events)

        if len(high_impact) == 0:
            return "CLEAR"
        elif len(high_impact) == 1:
            return "ELEVATED"
        elif len(high_impact) <= 3:
            return "HIGH"
        else:
            return "EXTREME"
