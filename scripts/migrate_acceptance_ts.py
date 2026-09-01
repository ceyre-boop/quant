#!/usr/bin/env python3
"""Backfill EDGAR acceptance instants as knowable_at. Phase 0.5, step 1.

THE DEFECT THIS FIXES

`recent_filings` read EDGAR's `filingDate` — a DATE — and discarded
`acceptanceDateTime`, which SEC supplies in the same payload. The point-in-time
layer then used that date as `knowable_at`, which resolves to midnight. Measured
on live data:

  AAPL earnings 8-K   accepted 2026-07-30T20:30:28Z = 16:30 ET, one minute AFTER
                      the close. A date-only knowable_at claims it was public at
                      00:00 that morning — a full trading day early, and it
                      inverts the entry: today's close instead of tomorrow's open.

  AAPL Form 4s        accepted ~22:30Z (18:30 ET). Confirmed leak: an as-of read
                      at noon on the filing day returned a filing that was not
                      public for another 10.5 hours.

For daily swing trading that is a rounding error. For catalyst momentum it is
the whole trade.

WHAT THIS DOES

  1. adds `published_ts TIMESTAMP` to fund_insider_txn and fund_institution_holding
  2. backfills it from EDGAR `acceptanceDateTime`, per accession
  3. for any accession EDGAR will not give us, falls back to END of the filing
     day (23:59:59Z) — NEVER the start. A filing we cannot time precisely must be
     assumed LATE. Assuming early is the leak.

Rows keep their `filing_date` for display and audit; only knowability moves.

  --dry-run   report what would change, touch nothing
  --yes       apply without prompting
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from sovereign.fundamentals.transports import sec  # noqa: E402
from sovereign.pit.store import DB_PATH  # noqa: E402

TABLES = {
    "fund_insider_txn": "accession",
    "fund_institution_holding": None,   # no accession column; date fallback only
}


def ensure_column(con, table: str) -> bool:
    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    if not cols:
        return False
    if "published_ts" not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN published_ts TIMESTAMP")
        print(f"  {table}: added published_ts")
    return True


def eod(d) -> datetime:
    """End of the filing day, UTC. The conservative direction."""
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def build_acceptance_map(con) -> dict[str, datetime]:
    """accession -> acceptance instant, fetched from EDGAR per issuer CIK."""
    rows = con.execute(
        "SELECT DISTINCT issuer_cik, ticker FROM fund_insider_txn "
        "WHERE issuer_cik IS NOT NULL OR ticker IS NOT NULL"
    ).fetchall()

    ciks: dict[int, str] = {}
    for cik, ticker in rows:
        if cik:
            ciks[int(cik)] = ticker or ""
        elif ticker:
            try:
                ciks[sec.resolve_cik(ticker)] = ticker
            except Exception:
                pass

    accepted: dict[str, datetime] = {}
    for cik, ticker in sorted(ciks.items()):
        try:
            for f in sec.recent_filings(cik, forms={"4"}):
                acc, at = f.get("accession"), f.get("accepted_at")
                if acc and at:
                    accepted[acc] = at
        except Exception as e:  # noqa: BLE001 — a dead CIK must not abort the backfill
            print(f"  warn: CIK {cik} ({ticker}): {e}")
    return accepted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no store at {DB_PATH}")
        return 0

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        n_ins = con.execute("SELECT count(*) FROM fund_insider_txn").fetchone()[0]
        n_inst = con.execute("SELECT count(*) FROM fund_institution_holding").fetchone()[0]
    finally:
        con.close()

    print(f"store: {DB_PATH}")
    print(f"  fund_insider_txn         {n_ins:>6} rows")
    print(f"  fund_institution_holding {n_inst:>6} rows\n")
    print("knowable_at moves from midnight-of-filing-date to the EDGAR acceptance")
    print("instant. Form 4s are typically accepted ~18:30 ET, so this REMOVES about")
    print("22 hours of overstated knowledge per row.\n")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0
    if not args.yes:
        try:
            if input("apply? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        except EOFError:
            print("non-interactive; re-run with --yes")
            return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = DB_PATH.with_suffix(f".pre-acceptance-{stamp}.db")
    shutil.copy2(DB_PATH, backup)
    print(f"backup: {backup}")

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("SET TimeZone='UTC'")
        for table in TABLES:
            ensure_column(con, table)

        print("fetching acceptance timestamps from EDGAR…")
        accepted = build_acceptance_map(con)
        print(f"  resolved {len(accepted)} accessions")

        rows = con.execute(
            "SELECT accession, line_no, filing_date FROM fund_insider_txn"
        ).fetchall()
        exact = fallback = 0
        for acc, line_no, fdate in rows:
            at = accepted.get(acc)
            if at is not None:
                exact += 1
            elif fdate is not None:
                at = eod(fdate)
                fallback += 1
            else:
                continue
            con.execute(
                "UPDATE fund_insider_txn SET published_ts = ? "
                "WHERE accession = ? AND line_no = ?",
                [at.replace(tzinfo=None), acc, line_no],
            )

        # Institutions has no accession column; end-of-filing-day is the only
        # honest option, and it is the conservative one.
        con.execute(
            "UPDATE fund_institution_holding "
            "SET published_ts = filing_date + INTERVAL 23 HOUR + INTERVAL 59 MINUTE "
            "WHERE filing_date IS NOT NULL AND published_ts IS NULL"
        )

        still_null = con.execute(
            "SELECT count(*) FROM fund_insider_txn WHERE published_ts IS NULL"
        ).fetchone()[0]
        print(f"  insider: {exact} exact acceptance, {fallback} end-of-day fallback, "
              f"{still_null} still NULL")
    finally:
        con.close()

    print("\ndone — knowable_at is now an instant, not a date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
