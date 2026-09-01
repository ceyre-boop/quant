"""Nasdaq short-interest transport.

api.nasdaq.com requires a browser-like User-Agent (the SEC UA gets a bot
block) but no key. Verified shape (curl, 2026-08-31):

  {"data": {"symbol": "aapl",
            "shortInterestTable": {"rows": [
                {"settlementDate": "08/14/2026", "interest": "116,327,753",
                 "avgDailyShareVolume": "46,065,396", "daysToCover": 2.525274},
                ...]}}}

There is no separate publication/disclosure date in the payload — only the
regulatory settlement_date — so published_ts is left None rather than
fabricated from settlement_date (settlement != publication, and per errors.py
None must mean "we don't actually know this", not a guess).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Optional

from sovereign.fundamentals.errors import SectionUnavailable
from sovereign.fundamentals.types import ShortInterestPoint

try:
    from sovereign.fundamentals.httpcache import TTLClass, get_json as _cache_get_json
except ImportError:
    TTLClass = None
    _cache_get_json = None

_SOURCE = "nasdaq"

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _fetch_status(url: str, timeout: int = 30) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url, headers=_BROWSER_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError) as e:
        raise SectionUnavailable("short_interest", _SOURCE, f"{url}: {e}") from e


def _get(url: str, timeout: int = 30) -> dict:
    status, text = _fetch_status(url, timeout)
    if not (200 <= status < 300):
        raise SectionUnavailable("short_interest", _SOURCE, f"{url}: HTTP {status}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SectionUnavailable("short_interest", _SOURCE, f"{url}: {e}") from e


def _num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if not s or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_settlement_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%m/%d/%Y").date()
    except ValueError:
        return None


def short_interest(ticker: str) -> list[ShortInterestPoint]:
    url = f"https://api.nasdaq.com/api/quote/{ticker.upper()}/short-interest?assetClass=stocks"
    if _cache_get_json is not None:
        try:
            payload = _cache_get_json(
                _SOURCE, f"short_interest_{ticker.upper()}", TTLClass.DAILY,
                lambda: _fetch_status(url), ticker=ticker,
            )
        except json.JSONDecodeError as e:
            raise SectionUnavailable("short_interest", _SOURCE, f"{url}: {e}") from e
    else:
        payload = _get(url)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not data:
        # Nasdaq returns HTTP 200 with data:null and an explanatory `message`
        # for tickers it simply does not cover -- most importantly NYSE-listed
        # names: "Short interest is only supported for Nasdaq Listed stocks".
        # Returning [] here would render an empty short-interest panel for every
        # NYSE ticker with no reason given, which is exactly the failure mode
        # SectionUnavailable exists to prevent. CRM surfaced this.
        message = ""
        if isinstance(payload, dict):
            message = str(payload.get("message") or "").strip()
        raise SectionUnavailable(
            "short_interest", "nasdaq",
            message or f"Nasdaq returned no short-interest data for {ticker}",
        )
    rows = (data.get("shortInterestTable") or {}).get("rows") or []

    out: list[ShortInterestPoint] = []
    for row in rows:
        out.append(ShortInterestPoint(
            source=_SOURCE,
            published_ts=None,
            ticker=ticker.upper(),
            settlement_date=_parse_settlement_date(row.get("settlementDate")),
            shares_short=_num(row.get("interest")),
            avg_daily_volume=_num(row.get("avgDailyShareVolume")),
            days_to_cover=_num(row.get("daysToCover")),
        ))
    return out
