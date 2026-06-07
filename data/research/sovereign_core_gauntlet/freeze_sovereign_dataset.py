"""
Stage 0 — Frozen equity dataset for the Sovereign Core validation gauntlet.

WHY THIS EXISTS
---------------
The Sovereign Core trades on Alpaca's free IEX feed, which was empirically shown to
return gappy, truncated history (SPY only from 2018-11 with ~28% of trading days
missing; QQQ only from mid-2020). That cannot support a credible 2015-2024 walk-forward
and would mechanically distort hold-period alignment in the permutation test.

This script instead freezes CLEAN, COMPLETE daily OHLCV from yfinance (the same source
the forex gauntlet uses) so every downstream stage reads from a single immutable store
with ZERO live API calls. The IEX gap problem is recorded as a separate live-readiness
blocker in the final verdict — it is not allowed to silently corrupt the edge test.

OUTPUT
------
  data/cache/equity/{SYMBOL}.parquet   raw daily OHLCV (open/high/low/close/volume), UTC index
  data/cache/equity/manifest.json      per-symbol rows + date range + sha256, plus a global hash

Usage:
  python3 scripts/freeze_sovereign_dataset.py
  python3 scripts/freeze_sovereign_dataset.py --symbols SPY QQQ --start 2015-01-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Quiet yfinance / urllib noise — we want a clean machine-readable run.
logging.basicConfig(level=logging.ERROR)
for _lib in ("yfinance", "urllib3", "requests", "peewee"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "cache" / "equity"

# Liquid, low-idiosyncratic universe (mirrors the forex gauntlet's discipline).
# SPY is mandatory: the router bear-filter and PTJ 200SMA gates read SPY, so the
# permutation harness needs it frozen to neutralize the live fetch.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"  # yfinance end is exclusive → captures through 2024-12-31


def _sha256_df(df: pd.DataFrame) -> str:
    """Stable content hash of a frozen OHLCV frame (index + values)."""
    h = hashlib.sha256()
    h.update(df.index.astype("int64").values.tobytes())
    for col in ("open", "high", "low", "close", "volume"):
        h.update(df[col].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol, start=start, end=end, interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"{symbol}: yfinance returned no data")

    # yfinance can return a MultiIndex column frame for a single ticker — flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns={c: c.lower() for c in raw.columns})[
        ["open", "high", "low", "close", "volume"]
    ].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Normalize index to tz-naive UTC dates for reproducible hashing.
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df.index.name = "timestamp"
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze clean equity OHLCV for the gauntlet")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "interval": "1d",
        "auto_adjust": True,
        "requested_start": args.start,
        "requested_end": args.end,
        "symbols": {},
    }

    global_hash = hashlib.sha256()
    failures = []

    for sym in args.symbols:
        try:
            df = _download(sym, args.start, args.end)
        except Exception as e:  # noqa: BLE001 — report, don't crash the whole freeze
            print(f"  {sym}: FAILED — {type(e).__name__}: {e}")
            failures.append(sym)
            continue

        path = OUT_DIR / f"{sym}.parquet"
        df.to_parquet(path)
        digest = _sha256_df(df)
        global_hash.update(digest.encode())

        manifest["symbols"][sym] = {
            "rows": int(len(df)),
            "start": df.index.min().date().isoformat(),
            "end": df.index.max().date().isoformat(),
            "sha256": digest,
            "path": str(path.relative_to(ROOT)),
        }
        print(f"  {sym}: {len(df)} bars | {df.index.min().date()} -> {df.index.max().date()} | {digest[:12]}")

    manifest["dataset_hash"] = global_hash.hexdigest()
    manifest["failures"] = failures
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDataset hash: {manifest['dataset_hash'][:16]}")
    print(f"Manifest: {OUT_DIR / 'manifest.json'}")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
