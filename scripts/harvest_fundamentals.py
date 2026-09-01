#!/usr/bin/env python3
"""scripts/harvest_fundamentals.py — batch fundamentals harvest CLI.

Fetches earnings/estimates/insider/short-interest/short-volume/borrow for the
watchlist (or an explicit ticker list), writes everything into
sovereign/fundamentals/store.py's DuckDB, logs every attempt (success or
failure) to fund_fetch_log, and optionally emits the static JSON artifacts
app/src/lib/fundamentals.ts's ``loadPanel`` reads first (the "warm" tier).

Institutional (13F) data is NOT fetched here — that's a separate bulk-ingest
job, scripts/harvest_13f_bulk.py, because 13F arrives as a multi-GB quarterly
dataset rather than a per-ticker pull.

Usage:
    python scripts/harvest_fundamentals.py                          # watchlist, all sections
    python scripts/harvest_fundamentals.py --ticker AAPL MSFT
    python scripts/harvest_fundamentals.py --sections earnings,insider
    python scripts/harvest_fundamentals.py --emit-json --limit 5
    python scripts/harvest_fundamentals.py --dry-run
    python scripts/harvest_fundamentals.py --report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # WATCHLIST_SYMBOLS
from sovereign.fundamentals import httpcache, store
from sovereign.fundamentals.errors import BudgetExhausted, SectionUnavailable, TickerUnresolved
from sovereign.fundamentals.panel import build_panel
from sovereign.fundamentals.providers.free import FreeProvider
from sovereign.fundamentals.transports import finra, sec
from sovereign.fundamentals.transports.alphavantage import CallBudget

log = logging.getLogger("harvest_fundamentals")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALL_SECTIONS = ["earnings", "insider", "short_interest", "short_volume", "borrow"]
AV_DAILY_BUDGET = 20  # hard-capped below AV's real 25/day free ceiling, leaving headroom for other tools

DATA_DIR = ROOT / "data" / "fundamentals"
WATCHLIST_FILE = ROOT / "config" / "fundamentals_watchlist.txt"

# Heuristic ETF/index-proxy filter: these tickers have no earnings/insider/13F
# sections (an ETF doesn't report EPS or file Form 4s), but DO have short
# interest/volume, so they stay in the universe for the "short" section only.
# This mirrors config.WATCHLIST_SYMBOLS's own mix of index/sector ETFs and
# single names — documented here rather than inferred, since "looks like a
# ticker with no earnings" is not reliably detectable any other way without a
# paid reference-data call.
KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "SLV", "USO", "UNG", "TLT",
    "LQD", "HYG", "VIXY", "UVXY", "SQQQ", "SPXL", "XLE", "XLF", "XLK", "XLV",
    "XLY", "XLI", "XLU", "XLP", "XLB", "XLRE", "XLC", "SMH", "SOXX", "ARKK",
    "EEM", "EFA", "IEF", "SHY", "TQQQ",
}


def is_etf(ticker: str) -> bool:
    return ticker.upper() in KNOWN_ETFS


# Mirrors the default in config.py, deliberately duplicated rather than imported.
# `import config` resolves to the config/ PACKAGE, not config.py -- the repo has
# both, and the directory shadows the module. Reading the env var directly is the
# only unambiguous way to get this without renaming one of them.
_WATCHLIST_DEFAULT = (
    "SPY,QQQ,MSFT,AAPL,TSLA,NVDA,AMZN,GOOGL,META,NFLX,AMD,CRM,"
    "ES=F,NQ=F,YM=F,CL=F,GC=F"
)

# Futures contracts carry no earnings, Form 4 or 13F -- drop them the same way
# ETFs are dropped, so the harvester does not spend SEC calls proving a negative.
_FUTURES_SUFFIXES = ("=F", "=X")


def load_watchlist() -> list[str]:
    raw = os.getenv("WATCHLIST_SYMBOLS", _WATCHLIST_DEFAULT)
    symbols = list(dict.fromkeys(
        s.strip().upper() for s in raw.split(",") if s.strip()
    ))
    symbols = [s for s in symbols if not s.endswith(_FUTURES_SUFFIXES)]
    if WATCHLIST_FILE.exists():
        extra = [
            line.strip().upper()
            for line in WATCHLIST_FILE.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        symbols = list(dict.fromkeys(symbols + extra))
    return symbols


# ── rate limiting (Nasdaq: 1 req/s; SEC/AV/FINRA already throttle themselves
#    inside their transports) ───────────────────────────────────────────────

_last_nasdaq_call = 0.0
_NASDAQ_MIN_INTERVAL = 1.0


def _throttle_nasdaq() -> None:
    global _last_nasdaq_call
    now = time.monotonic()
    wait = _NASDAQ_MIN_INTERVAL - (now - _last_nasdaq_call)
    if wait > 0:
        time.sleep(wait)
    _last_nasdaq_call = time.monotonic()


class Harvest:
    def __init__(self, run_id: str, dry_run: bool, max_age_hours: Optional[float]):
        self.run_id = run_id
        self.dry_run = dry_run
        self.max_age_hours = max_age_hours
        self.av_budget = CallBudget(AV_DAILY_BUDGET)
        self.provider = FreeProvider(av_budget=self.av_budget)
        self.attempts = 0
        self.successes = 0

    def _log(self, source: str, section: str, ticker: Optional[str], endpoint: str,
              ok: bool, rows: int, latency_ms: int, error: Optional[str] = None) -> None:
        self.attempts += 1
        if ok:
            self.successes += 1
        if self.dry_run:
            return
        store.log_fetch(self.run_id, source, section, ticker, endpoint,
                        200 if ok else None, ok, rows, latency_ms, error)

    def _stale(self, ticker: str, table: str, date_col: str) -> bool:
        """True if the newest row for this ticker is older than max_age_hours,
        or there is no row at all. Skips re-fetching a section that was just
        refreshed — --max-age-hours is the caller's freshness contract."""
        if self.max_age_hours is None:
            return True
        with store.connect(read_only=False) as con:
            row = con.execute(
                f"SELECT max(fetched_at) FROM {table} WHERE ticker = ?", [ticker]
            ).fetchone()
        if not row or not row[0]:
            return True
        age_hours = (datetime.now(timezone.utc) - row[0]).total_seconds() / 3600
        return age_hours >= self.max_age_hours

    # ── per-section fetchers ────────────────────────────────────────────

    def earnings(self, ticker: str) -> None:
        t0 = time.monotonic()
        try:
            rows = self.provider.earnings_history(ticker, limit=20)
        except (SectionUnavailable, BudgetExhausted) as e:
            self._log("free", "earnings", ticker, "earnings_history", False, 0,
                      int((time.monotonic() - t0) * 1000), str(e))
            log.warning("earnings(%s): %s", ticker, e)
            return
        # fund_earnings_event's PK is (ticker, fiscal_end) NOT NULL, but yahoo
        # (the primary free earnings transport) never populates fiscal_end —
        # only report_date. free.py's own in-memory merge already treats
        # "fiscal_end or report_date" as the row identity for the same reason;
        # this backfill makes that same fallback hold for the persisted PK
        # rather than crashing the upsert on a NOT NULL violation. Rows that
        # genuinely have a fiscal_end (e.g. merged in from Alpha Vantage) keep it.
        storable = [
            (r if r.fiscal_end is not None else replace(r, fiscal_end=r.report_date))
            for r in rows
        ]
        storable = [r for r in storable if r.fiscal_end is not None]
        n = 0 if self.dry_run else store.upsert_earnings(storable)
        self._log("free", "earnings", ticker, "earnings_history", True, len(rows),
                  int((time.monotonic() - t0) * 1000))
        log.info("earnings(%s): %d rows", ticker, len(rows))

        # Reaction computation is cheap once bars are pulled once — piggyback
        # here so the store is warm for panel.py's store-first read.
        if rows and not self.dry_run:
            try:
                from sovereign.fundamentals.reaction import compute_reactions

                reactions = compute_reactions(ticker, rows)
                if reactions:
                    store.upsert_reactions(reactions)
            except Exception as e:  # noqa: BLE001 - reaction is a bonus, never fails the harvest
                log.info("earnings(%s): reaction computation skipped: %s", ticker, e)

    def insider(self, ticker: str) -> None:
        t0 = time.monotonic()
        try:
            cik = sec.resolve_cik(ticker)
        except TickerUnresolved as e:
            self._log("sec_edgar", "insider", ticker, "resolve_cik", False, 0,
                      int((time.monotonic() - t0) * 1000), str(e))
            log.warning("insider(%s): %s", ticker, e)
            return
        since = date.today() - timedelta(days=180)
        try:
            rows = self.provider.insider_transactions(ticker, cik, since, max_filings=40)
        except SectionUnavailable as e:
            self._log("sec_edgar", "insider", ticker, "form4_transactions", False, 0,
                      int((time.monotonic() - t0) * 1000), str(e))
            log.warning("insider(%s): %s", ticker, e)
            return
        n = 0 if self.dry_run else store.upsert_insider(rows)
        self._log("sec_edgar", "insider", ticker, "form4_transactions", True, len(rows),
                  int((time.monotonic() - t0) * 1000))
        log.info("insider(%s): %d rows", ticker, len(rows))

    def short_interest(self, ticker: str) -> None:
        t0 = time.monotonic()
        _throttle_nasdaq()
        since = date.today() - timedelta(days=400)
        try:
            rows = self.provider.short_interest(ticker, since)
        except SectionUnavailable as e:
            self._log("nasdaq", "short_interest", ticker, "short_interest", False, 0,
                      int((time.monotonic() - t0) * 1000), str(e))
            log.warning("short_interest(%s): %s", ticker, e)
            return
        n = 0 if self.dry_run else store.upsert_short_interest(rows)
        self._log("nasdaq", "short_interest", ticker, "short_interest", True, len(rows),
                  int((time.monotonic() - t0) * 1000))
        log.info("short_interest(%s): %d rows", ticker, len(rows))

    def short_volume_batch(self, tickers: list[str], days: int = 90) -> None:
        """FINRA serves one day-file for the WHOLE market — fetched exactly
        once here for every ticker in the batch, per the transport's own
        docstring. Never call this per ticker."""
        t0 = time.monotonic()
        try:
            by_ticker = finra.short_volume(set(tickers), days=days)
        except SectionUnavailable as e:
            for t in tickers:
                self._log("finra", "short_volume", t, "short_volume_batch", False, 0,
                          int((time.monotonic() - t0) * 1000), str(e))
            log.warning("short_volume_batch: %s", e)
            return
        for t in tickers:
            rows = by_ticker.get(t, [])
            n = 0 if self.dry_run else store.upsert_short_volume(rows)
            self._log("finra", "short_volume", t, "short_volume_batch", True, len(rows),
                      int((time.monotonic() - t0) * 1000))
        log.info("short_volume_batch: %d tickers, %d total rows",
                 len(tickers), sum(len(v) for v in by_ticker.values()))

    def borrow(self, ticker: str) -> None:
        t0 = time.monotonic()
        try:
            rows = self.provider.borrow(ticker)
        except SectionUnavailable as e:
            self._log("ib_shortable_snapshot", "borrow", ticker, "borrow", False, 0,
                      int((time.monotonic() - t0) * 1000), str(e))
            log.warning("borrow(%s): %s", ticker, e)
            return
        n = 0 if self.dry_run else store.upsert_borrow(rows)
        self._log("ib_shortable_snapshot", "borrow", ticker, "borrow", True, len(rows),
                  int((time.monotonic() - t0) * 1000))
        log.info("borrow(%s): %d rows", ticker, len(rows))

    def run(self, tickers: list[str], sections: list[str]) -> None:
        # Symbol table upsert so panel.py's _store_name has something to read.
        if not self.dry_run:
            store.upsert_symbols([{"ticker": t, "in_watchlist": True} for t in tickers])

        per_ticker_sections = [s for s in sections if s != "short_volume"]
        equity_tickers = [t for t in tickers if not is_etf(t)]

        for t in tickers:
            for section in per_ticker_sections:
                if section in ("earnings", "insider") and is_etf(t):
                    continue  # ETFs have no earnings/insider data — not a fetch failure, just N/A
                if not self._stale(t, {
                    "earnings": "fund_earnings_event", "insider": "fund_insider_txn",
                    "short_interest": "fund_short_interest", "borrow": "fund_borrow",
                }.get(section, "fund_fetch_log"), "fetched_at"):
                    log.info("%s(%s): fresh within max_age_hours, skipping", section, t)
                    continue
                getattr(self, section)(t)

        if "short_volume" in sections and tickers:
            self.short_volume_batch(tickers, days=90)


def emit_json(tickers: list[str]) -> None:
    tickers_dir = DATA_DIR / "tickers"
    tickers_dir.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "tickers": []}
    coverage = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "tickers": {}}

    for t in tickers:
        try:
            panel = build_panel(t, warm_only=True)
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the artifact batch
            log.warning("emit_json(%s): build_panel failed: %s", t, e)
            continue
        (tickers_dir / f"{t}.json").write_text(json.dumps(panel, indent=1))
        index["tickers"].append(t)
        coverage["tickers"][t] = {
            "partial": panel["partial"],
            "sections": {k: {"rows": len(v.get("rows", [])), "gaps": v.get("gaps", [])}
                        for k, v in panel["sections"].items()},
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=1))
    (DATA_DIR / "coverage.json").write_text(json.dumps(coverage, indent=1))
    log.info("emit_json: wrote %d ticker artifacts + index.json + coverage.json", len(index["tickers"]))


def print_report() -> None:
    with store.connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT source, section, count(*) AS attempts,
                   sum(CASE WHEN ok THEN 1 ELSE 0 END) AS successes,
                   count(DISTINCT ticker) AS tickers,
                   min(fetched_at) AS oldest
            FROM fund_fetch_log
            GROUP BY source, section
            ORDER BY source, section
            """
        ).fetchall()
    if not rows:
        print("no fund_fetch_log entries yet")
        return
    print(f"{'source':<20}{'section':<16}{'attempts':>9}{'successes':>11}{'tickers':>9}  oldest")
    for source, section, attempts, successes, tickers, oldest in rows:
        print(f"{source:<20}{section:<16}{attempts:>9}{successes:>11}{tickers:>9}  {oldest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watchlist", action="store_true", help="use config.WATCHLIST_SYMBOLS (default if no --ticker)")
    ap.add_argument("--ticker", nargs="+", metavar="SYM", help="explicit ticker list")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS),
                    help=f"comma-separated subset of {ALL_SECTIONS}")
    ap.add_argument("--emit-json", action="store_true", help="write data/fundamentals/{tickers,index,coverage}")
    ap.add_argument("--dry-run", action="store_true", help="fetch but write nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of tickers processed")
    ap.add_argument("--max-age-hours", type=float, default=None,
                    help="skip a ticker/section already fetched more recently than this")
    ap.add_argument("--report", action="store_true", help="print fund_fetch_log coverage and exit")
    args = ap.parse_args()

    if args.report:
        print_report()
        return

    tickers = [t.upper() for t in args.ticker] if args.ticker else load_watchlist()
    if args.limit:
        tickers = tickers[: args.limit]
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = set(sections) - set(ALL_SECTIONS)
    if unknown:
        ap.error(f"unknown sections {sorted(unknown)}; have {ALL_SECTIONS}")

    log.info("harvest_fundamentals: %d tickers, sections=%s, dry_run=%s",
             len(tickers), sections, args.dry_run)

    run_id = uuid.uuid4().hex[:12]
    harvest = Harvest(run_id, args.dry_run, args.max_age_hours)
    harvest.run(tickers, sections)

    log.info("run %s: %d/%d fetches ok", run_id, harvest.successes, harvest.attempts)

    if args.emit_json and not args.dry_run:
        emit_json(tickers)

    print(httpcache.summary())


if __name__ == "__main__":
    main()
