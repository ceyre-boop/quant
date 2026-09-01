"""FINRA Reg SHO daily short-volume transport.

One file per trading day covers the entire market (pipe-delimited,
Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market), so this fetches
by DAY and filters to the requested tickers in a single pass rather than
per-ticker — there is no per-symbol endpoint to hit anyway.

Short VOLUME is not short INTEREST: it's every day's shares sold short
(including routine market-maker facilitation short sales), not a snapshot of
open short positions. Never merge ShortVolumePoint into ShortInterestPoint —
see ShortVolumePoint's docstring in types.py.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional

from sovereign.fundamentals.errors import SectionUnavailable
from sovereign.fundamentals.types import ShortVolumePoint

try:
    from sovereign.fundamentals.httpcache import TTLClass, get_text as _cache_get_text
except ImportError:
    TTLClass = None
    _cache_get_text = None

_SOURCE = "finra"

UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}


def _fetch_day(day: date) -> Optional[str]:
    """Fetch one day's file. Returns None on 404 (weekend/holiday/no data yet)
    — that's a genuinely absent day, not a fetch failure worth raising over.
    Any other network failure raises SectionUnavailable."""
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day.strftime('%Y%m%d')}.txt"

    def _fetch_status() -> tuple[int, str]:
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""

    try:
        if _cache_get_text is not None:
            status_holder: list[int] = []

            def _fetch_status_capture() -> tuple[int, str]:
                status, text = _fetch_status()
                status_holder.append(status)
                return status, text

            text = _cache_get_text(_SOURCE, f"shvol_{day.isoformat()}", TTLClass.IMMUTABLE, _fetch_status_capture)
            # httpcache doesn't surface the status on a cache hit (status_holder
            # stays empty); a cache hit only ever happens for a day we already
            # cached, which can only be a prior 2xx — so an empty holder is fine.
            if status_holder and not (200 <= status_holder[-1] < 300):
                return None
            return text
        status, text = _fetch_status()
        if status == 404:
            return None
        if not (200 <= status < 300):
            raise SectionUnavailable("short_volume", _SOURCE, f"{url}: HTTP {status}")
        return text
    except (urllib.error.URLError, OSError) as e:
        raise SectionUnavailable("short_volume", _SOURCE, f"{url}: {e}") from e


def _parse_day(text: str, wanted: set[str]) -> dict[str, ShortVolumePoint]:
    out: dict[str, ShortVolumePoint] = {}
    lines = text.splitlines()
    if not lines:
        return out
    header = lines[0].split("|")
    try:
        i_date = header.index("Date")
        i_sym = header.index("Symbol")
        i_short = header.index("ShortVolume")
        i_exempt = header.index("ShortExemptVolume")
        i_total = header.index("TotalVolume")
    except ValueError:
        return out

    for line in lines[1:]:
        if not line or "|" not in line:
            continue
        fields = line.split("|")
        if len(fields) <= max(i_date, i_sym, i_short, i_exempt, i_total):
            continue
        symbol = fields[i_sym]
        if symbol not in wanted:
            continue
        try:
            day = datetime.strptime(fields[i_date], "%Y%m%d").date()
            short_vol = float(fields[i_short])
            exempt_vol = float(fields[i_exempt])
            total_vol = float(fields[i_total])
        except ValueError:
            continue
        short_pct = (short_vol / total_vol) if total_vol else None
        out[symbol] = ShortVolumePoint(
            source=_SOURCE,
            published_ts=datetime.combine(day, datetime.min.time()),
            ticker=symbol,
            date=day,
            short_volume=short_vol,
            short_exempt_volume=exempt_vol,
            total_volume=total_vol,
            short_pct=short_pct,
        )
    return out


def short_volume(tickers: set[str], days: int = 30) -> dict[str, list[ShortVolumePoint]]:
    wanted = {t.upper() for t in tickers}
    out: dict[str, list[ShortVolumePoint]] = {t: [] for t in wanted}

    cursor = date.today()
    fetched_days = 0
    lookback_limit = days * 3 + 10  # generous ceiling so weekends/holidays don't starve `days`
    scanned = 0
    while fetched_days < days and scanned < lookback_limit:
        scanned += 1
        cursor -= timedelta(days=1)
        if cursor.weekday() >= 5:  # Sat/Sun — FINRA publishes trading days only
            continue
        text = _fetch_day(cursor)
        fetched_days += 1
        if text is None:
            continue
        day_points = _parse_day(text, wanted)
        for symbol, point in day_points.items():
            out[symbol].append(point)

    for symbol in out:
        out[symbol].sort(key=lambda p: p.date)
    return out
