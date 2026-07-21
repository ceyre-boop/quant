#!/usr/bin/env python3
"""
Refresh the data caches that go stale without a scheduler.

agent_scheduler (which used to refresh these) was deprecated/unloaded, orphaning its cache-refresh
duties — so the Reddit sentiment cache and the macro snapshot just went stale (reddit ~13d, macro
~25h), tripping the health check. This is the one entry point that refreshes both; the
com.alta.cache.refresh launchd job runs it hourly so they stay GREEN.

Fail-soft: a failure in one refresh never blocks the other.

Usage:  python3 scripts/refresh_caches.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REDDIT_CACHE = ROOT / "data" / "cache" / "reddit_sentiment.json"
MACRO_SNAP = ROOT / "data" / "macro" / "macro_snapshot.json"


def _age_min(p: Path) -> str:
    if not p.exists():
        return "missing"
    return f"{(time.time() - p.stat().st_mtime) / 60:.0f}m old"


def _refresh_reddit() -> None:
    """RETIRED 2026-07-20 — see execution/context.py::REDDIT_RETIREMENT.

    Reddit 403s unauthenticated JSON on both old.reddit.com and www.reddit.com
    (tested live with a browser User-Agent) and no OAuth app is registered. Calling
    the scraper burns three retries per subreddit to fail, then rewrites the cache
    with posts_scanned=0 — refreshing the mtime so every staleness check reads
    green over an empty payload. Skipping is the honest behaviour: the cache stays
    visibly old instead of freshly empty.

    `sovereign/data/reddit_scraper.py` is deliberately left on disk and correct. It
    is the starting point if an OAuth app is ever registered.
    """
    print("  reddit: SKIPPED — source retired (403, no OAuth app); "
          f"cache left at {_age_min(REDDIT_CACHE)}")


def _refresh_macro() -> None:
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_macro_cache.py")],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0 and MACRO_SNAP.exists()
        print(f"  macro:  {'refreshed' if ok else 'ran (check)'} -> {_age_min(MACRO_SNAP)}"
              + ("" if ok else f"  rc={r.returncode} {r.stderr.strip()[:120]}"))
    except Exception as e:  # noqa: BLE001 — fail-soft
        print(f"  macro:  FAILED ({type(e).__name__}: {e})")


def main() -> int:
    print(f"[refresh_caches] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  before: reddit {_age_min(REDDIT_CACHE)} | macro {_age_min(MACRO_SNAP)}")
    _refresh_reddit()
    _refresh_macro()
    return 0


if __name__ == "__main__":
    sys.exit(main())
