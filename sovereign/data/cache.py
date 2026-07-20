"""Local parquet cache for historical bar pulls.

Layout: ``data/cache/{symbol}/{date}.parquet`` — one file per symbol-day, so a
partial failure corrupts one day rather than a whole series (the same rationale
that drove the per-asset cache in ``sovereign/data/feeds/alpaca_feed.py``, whose
logic this module generalises).

Invalidation:
  - a past trading day is immutable — once written it is never refetched
  - *today* is provisional until the session closes; a cached today-file written
    before 16:00 ET is treated as stale and refetched
  - a corrupt parquet is deleted and refetched rather than raised

Isolation: stdlib + pandas only. Imports nothing from ``ict/`` (NN#1) and nothing
from the execution path.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Repo root -> data/cache. Never hardcode an absolute path.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "bars"

# US market close, in UTC. 16:00 ET is 20:00 UTC (EDT) / 21:00 UTC (EST); we take
# the later of the two so a cached today-file is never promoted to immutable
# early. Being conservative costs one refetch; being early caches a partial day
# forever.
_CLOSE_UTC_HOUR = 21


@dataclass
class CacheStats:
    """Hit/miss counters so rate-limit savings are visible, not assumed."""

    hits: int = 0
    misses: int = 0
    stale: int = 0
    corrupt: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def summary(self) -> str:
        return (f"cache hits={self.hits} misses={self.misses} "
                f"stale={self.stale} corrupt={self.corrupt} "
                f"writes={self.writes} hit_rate={self.hit_rate:.1%}")


class DataCache:
    """Symbol-day parquet cache in front of any bar fetcher."""

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    # ── paths ────────────────────────────────────────────────────────────────

    def path_for(self, symbol: str, date: str) -> Path:
        return self.cache_dir / symbol.upper().strip() / f"{date}.parquet"

    # ── invalidation ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_today(date: str) -> bool:
        return date == datetime.now(timezone.utc).date().isoformat()

    def _is_stale(self, path: Path, date: str) -> bool:
        """Historical days never go stale. Today goes stale until after close."""
        if not self._is_today(date):
            return False
        written = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        close = written.replace(hour=_CLOSE_UTC_HOUR, minute=0, second=0,
                                microsecond=0)
        return written < close

    # ── main entry point ─────────────────────────────────────────────────────

    def get_or_fetch(self, symbol: str, date: str,
                     fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """Return cached bars for symbol/date, else call ``fetcher`` and cache it.

        ``fetcher`` takes no arguments and returns a DataFrame; bind the symbol
        and date into it at the call site. An empty result is *not* cached — an
        empty frame is usually a transient API failure, and caching it would
        make the failure permanent.
        """
        path = self.path_for(symbol, date)

        if path.exists():
            if self._is_stale(path, date):
                self.stats.stale += 1
                logger.debug("cache stale (intraday today): %s %s", symbol, date)
            else:
                try:
                    df = pd.read_parquet(path)
                    self.stats.hits += 1
                    logger.debug("cache hit: %s %s (%d rows)", symbol, date, len(df))
                    return df
                except Exception as e:  # noqa: BLE001 - corrupt file is recoverable
                    self.stats.corrupt += 1
                    logger.warning("cache corrupt, refetching: %s %s: %s",
                                   symbol, date, e)
                    path.unlink(missing_ok=True)

        self.stats.misses += 1
        logger.debug("cache miss -> API: %s %s", symbol, date)
        df = fetcher()

        if df is not None and not df.empty:
            self.put(symbol, date, df)
        return df if df is not None else pd.DataFrame()

    def put(self, symbol: str, date: str, df: pd.DataFrame) -> None:
        path = self.path_for(symbol, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path)
            self.stats.writes += 1
        except Exception as e:  # noqa: BLE001 - a cache write must never break a fetch
            logger.warning("cache write failed for %s %s: %s", symbol, date, e)

    # ── maintenance ──────────────────────────────────────────────────────────

    def invalidate(self, symbol: str, date: str | None = None) -> int:
        """Drop one symbol-day, or the whole symbol if ``date`` is None."""
        if date is not None:
            path = self.path_for(symbol, date)
            if path.exists():
                path.unlink()
                return 1
            return 0
        sym_dir = self.cache_dir / symbol.upper().strip()
        if not sym_dir.exists():
            return 0
        n = len(list(sym_dir.glob("*.parquet")))
        shutil.rmtree(sym_dir)
        return n

    def log_stats(self) -> None:
        logger.info("DataCache: %s", self.stats.summary())


def date_range(start: str, end: str) -> list[str]:
    """Inclusive YYYY-MM-DD day list — the cache's natural iteration unit."""
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
