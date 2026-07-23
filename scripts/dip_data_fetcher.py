"""
DIP Data Fetcher — centralized parent-process price pull
========================================================
The one genuinely new file in the Daily Intelligence Pipeline revival.

WHY THIS EXISTS
---------------
`continuous_harvester.py` fans out across a fork-based multiprocessing Pool and
calls `yf.download` *inside the forked workers*. On macOS, forking after network
/ thread state exists (curl_cffi sessions, yfinance worker threads) deadlocks the
child — the classic fork-after-network hang. Symptom: the harvester hangs
indefinitely and `data/harvest.db` never grows (frozen since 2026-06-29).

THE FIX
-------
Do all network I/O in the PARENT process, up front, warming the on-disk parquet
cache. Once every symbol's parquet exists and is fresh (<12h), the harvester's
`load_ohlcv` short-circuits the download branch entirely, so the forked workers
only ever READ cached files — no network in a fork, no deadlock.

This module deliberately REUSES `continuous_harvester.load_ohlcv` and its cache
constants as the single source of truth for cache path + schema. It does not
re-implement the download. It only changes WHERE (which process) the fetch runs.

ISOLATION: imports `config.universe` and `continuous_harvester` only. Never
imports `sovereign/`. Pure data-plane; never calls order_send or any bridge.

Usage:
    python scripts/dip_data_fetcher.py                 # full universe
    python scripts/dip_data_fetcher.py --symbols AAPL NVDA
    python scripts/dip_data_fetcher.py --force         # ignore 12h cache TTL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the harvester's cache logic verbatim — single source of truth.
from scripts.continuous_harvester import (  # noqa: E402
    CACHE_DIR,
    HISTORY_END,
    HISTORY_START,
    load_ohlcv,
    log,
)
from config.universe import UNIVERSE  # noqa: E402

CHECKPOINT = ROOT / "data" / "_dip_fetch_checkpoint.json"


def prefetch(symbols: List[str], force: bool = False) -> dict:
    """Warm the parquet cache for every symbol IN THE PARENT PROCESS.

    Returns a summary dict and writes a checkpoint/error file so no failure is
    silent (CLAUDE.md: every phase writes a checkpoint or an error file).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok: List[str] = []
    failed: List[str] = []

    for i, sym in enumerate(symbols):
        cache_file = CACHE_DIR / f"{sym.replace('=', '_')}.parquet"
        if force and cache_file.exists():
            cache_file.unlink()  # force a fresh pull by removing the cache
        df = load_ohlcv(sym)  # parent-process fetch; writes parquet as a side effect
        if df is None or df.empty:
            failed.append(sym)
            log.warning(f"  [{i+1}/{len(symbols)}] {sym:8s}  FETCH FAILED (no data)")
        else:
            ok.append(sym)
            log.info(f"  [{i+1}/{len(symbols)}] {sym:8s}  {len(df):>5} bars cached")

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        "requested": len(symbols),
        "cached_ok": len(ok),
        "failed": failed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(summary, indent=2))
    log.info(
        f"DIP prefetch done: {len(ok)}/{len(symbols)} cached, "
        f"{len(failed)} failed, {summary['elapsed_sec']}s → {CHECKPOINT}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="DIP centralized parent-process price fetcher")
    ap.add_argument("--symbols", nargs="+", help="Subset of symbols (default: full universe)")
    ap.add_argument("--force", action="store_true", help="Ignore the 12h cache TTL, re-pull all")
    args = ap.parse_args()

    symbols = args.symbols if args.symbols else list(UNIVERSE)
    log.info(f"DIP prefetch starting — {len(symbols)} symbols (force={args.force})")
    summary = prefetch(symbols, force=args.force)

    # Non-zero exit if EVERYTHING failed — a real failure the caller must see.
    return 1 if summary["cached_ok"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
