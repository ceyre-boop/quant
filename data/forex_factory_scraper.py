"""Forex Factory economic calendar scraper.

Returns today's events with keys: currency, event, impact
Impact values: 'High', 'Medium', 'Low', 'Non-Economic'
Both call sites (calendar_fetcher, ict/daily_bias) wrap in try/except.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent.parent / 'data' / 'cache' / 'ff_calendar.json'
_CACHE_TTL_HOURS = 4
_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.forexfactory.com/',
}


class ForexFactoryScraper:
    """Scrapes the Forex Factory economic calendar.

    Caches results for _CACHE_TTL_HOURS hours so launchd runs don't
    hammer FF on every 5-minute ICT scan. Returns [] on any failure —
    both CalendarFetcher and daily_bias already handle the empty case.
    """

    BASE_URL = 'https://www.forexfactory.com/calendar'

    def fetch_today_events(self) -> List[Dict[str, Any]]:
        """Return today's high/medium/low impact events.

        Each event dict has: currency, event, impact
        """
        cached = self._load_cache()
        if cached is not None:
            logger.debug("ForexFactory: returning %d cached events", len(cached))
            return cached

        try:
            resp = requests.get(self.BASE_URL, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            events = self._parse(resp.text)
            self._save_cache(events)
            logger.info("ForexFactory: fetched %d events", len(events))
            return events
        except Exception as exc:
            logger.warning("ForexFactory scrape failed: %s", exc)
            return []

    # ── Parsing ──────────────────────────────────────────────────────────── #

    def _parse(self, html: str) -> List[Dict[str, Any]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', class_='calendar__table')
        if not table:
            return []

        events: List[Dict[str, Any]] = []
        current_currency = ''

        for row in table.find_all('tr', class_='calendar__row'):
            try:
                # Currency column spans multiple rows — carry it forward
                currency_td = row.find('td', class_='calendar__currency')
                if currency_td:
                    text = currency_td.get_text(strip=True)
                    if text:
                        current_currency = text

                impact = self._parse_impact(row)
                event_name = self._parse_event(row)

                if event_name and current_currency:
                    events.append({
                        'currency': current_currency,
                        'event': event_name,
                        'impact': impact,
                    })
            except Exception:
                continue

        return events

    def _parse_impact(self, row) -> str:
        td = row.find('td', class_='calendar__impact')
        if not td:
            return 'Low'
        span = td.find('span')
        if not span:
            return 'Low'
        cls = ' '.join(span.get('class', []))
        if 'red' in cls:
            return 'High'
        if 'ora' in cls:
            return 'Medium'
        if 'yel' in cls:
            return 'Low'
        return 'Non-Economic'

    def _parse_event(self, row) -> Optional[str]:
        td = row.find('td', class_='calendar__event')
        if not td:
            return None
        # FF wraps the title in a nested span
        title_span = td.find('span', class_='calendar__event-title')
        if title_span:
            return title_span.get_text(strip=True)
        return td.get_text(strip=True) or None

    # ── Cache ─────────────────────────────────────────────────────────────── #

    def _load_cache(self) -> Optional[List[Dict[str, Any]]]:
        try:
            if not _CACHE_FILE.exists():
                return None
            data = json.loads(_CACHE_FILE.read_text())
            if data.get('date') != datetime.now().strftime('%Y-%m-%d'):
                return None
            cached_at = datetime.fromisoformat(data['cached_at'])
            if datetime.now() - cached_at > timedelta(hours=_CACHE_TTL_HOURS):
                return None
            return data['events']
        except Exception:
            return None

    def _save_cache(self, events: List[Dict[str, Any]]) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps({
                'cached_at': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'events': events,
            }))
        except Exception:
            pass
