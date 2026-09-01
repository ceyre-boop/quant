"""sovereign/fundamentals/httpcache.py — raw-payload HTTP cache for the fundamentals layer.

Deliberately NOT ``sovereign.data.cache.DataCache``: that cache's invalidation rule is
bar-specific — "a past trading day is immutable, today is provisional until close" — which
has no meaning for a filing. An SEC accession never changes once filed (IMMUTABLE); a
submissions index or an estimate snapshot is a daily read that should refresh once a day
(DAILY); a 13F bulk file is a quarterly artifact (QUARTERLY). Baking bar-day logic into a
filing cache would either over-cache (serve a stale snapshot for a week) or under-cache
(refetch an immutable accession on every run, burning EDGAR's rate limit for nothing) — so
this module owns its own three-class TTL instead of stretching DataCache's one rule to fit.

``CacheStats`` IS reused from sovereign/data/cache.py — its hits/misses/stale/corrupt/writes
counters are exactly the shape fundamentals telemetry needs, and importing it keeps cache-hit
reporting readable the same way across the repo instead of inventing a second counter shape.

Layout: data/cache/fundamentals/{source}/{ticker_or_key}/{key}.json (or .txt for get_text).
One file per (source, ticker, key) so a single corrupt payload never takes out a whole source.

Never imports from ict/ (NN#1 isolation) or the execution path.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from sovereign.data.cache import CacheStats

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "fundamentals"


class TTLClass(Enum):
    """How long a cached payload for this key stays valid."""
    IMMUTABLE = "immutable"   # SEC accession, Form 4 body — the filed text never changes
    DAILY = "daily"           # submissions index, estimate consensus snapshot — expire at next UTC midnight
    QUARTERLY = "quarterly"   # 13F bulk form — one artifact per reporting quarter


def _path_for(source: str, key: str, ticker: str | None, ext: str) -> Path:
    sub = (ticker or key).replace("/", "_").upper()
    safe_key = key.replace("/", "_")
    return CACHE_DIR / source / sub / f"{safe_key}.{ext}"


def _is_stale(path: Path, ttl_class: TTLClass) -> bool:
    """IMMUTABLE never goes stale. DAILY expires at the next UTC midnight after it was
    written (not a rolling 24h window — a file written at 23:59 should still refresh at
    00:00, matching "today's snapshot" semantics). QUARTERLY uses a 90-day rolling window,
    since 13F bulk files don't have a clean calendar-quarter write instant to anchor on."""
    if ttl_class is TTLClass.IMMUTABLE:
        return False
    written = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    if ttl_class is TTLClass.DAILY:
        next_midnight = (written + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return datetime.now(timezone.utc) >= next_midnight
    # QUARTERLY
    return datetime.now(timezone.utc) - written >= timedelta(days=90)


STATS = CacheStats()


def _fetch_raw(source: str, key: str, ticker: str | None, ttl_class: TTLClass,
                fetcher: Callable[[], tuple[int, str]], ext: str,
                _skip_cache_check: bool = False) -> str:
    """Shared get_json/get_text body: return the cached text, or call fetcher() and cache
    the result. fetcher() returns (http_status, text); a non-2xx status is not cached, so a
    transient failure never becomes a permanent one.

    ``_skip_cache_check`` lets get_json reuse this for the fetch+write half only, after it
    has already done its own JSON-aware read of the cache — avoids checking the file twice
    and double-counting hits/corrupt.
    """
    path = _path_for(source, key, ticker, ext)

    if not _skip_cache_check and path.exists():
        if _is_stale(path, ttl_class):
            STATS.stale += 1
        else:
            try:
                text = path.read_text()
                STATS.hits += 1
                return text
            except OSError:
                STATS.corrupt += 1
                path.unlink(missing_ok=True)

    STATS.misses += 1
    status, text = fetcher()
    if 200 <= status < 300:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        STATS.writes += 1
    return text


def get_json(source: str, key: str, ttl_class: TTLClass,
             fetcher: Callable[[], tuple[int, str]], ticker: str | None = None):
    """Return parsed JSON for (source, key), fetching+caching via ``fetcher`` on a miss.

    On a corrupt cache file (valid file on disk, but not parseable JSON) the file is deleted
    and the fetch is retried once, rather than raising — a corrupt cache entry should never
    be a harder failure than a cold cache. ``_fetch_raw`` already handles the unreadable-file
    case (OSError); this layer adds the JSON-specific "readable but not valid JSON" case.
    """
    path = _path_for(source, key, ticker, "json")
    if path.exists():
        if _is_stale(path, ttl_class):
            STATS.stale += 1
        else:
            try:
                text = path.read_text()
                parsed = json.loads(text)  # validate before counting a hit
            except (OSError, json.JSONDecodeError):
                STATS.corrupt += 1
                path.unlink(missing_ok=True)
            else:
                STATS.hits += 1
                return parsed

    text = _fetch_raw(source, key, ticker, ttl_class, fetcher, "json", _skip_cache_check=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The freshly-fetched payload itself is bad JSON — don't leave it cached.
        path.unlink(missing_ok=True)
        raise


def get_text(source: str, key: str, ttl_class: TTLClass,
             fetcher: Callable[[], tuple[int, str]], ticker: str | None = None) -> str:
    """Return raw text for (source, key) — FINRA .txt drops, Form 4 XML, 13F bulk XML."""
    return _fetch_raw(source, key, ticker, ttl_class, fetcher, "txt")


def summary() -> str:
    return STATS.summary()
