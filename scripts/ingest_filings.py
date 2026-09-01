#!/usr/bin/env python3
"""Phase 1 — EDGAR filings ingest. Custody, not interpretation.

Writes one row per accepted filing into the `filings` table, with:

  knowable_at   the EDGAR ACCEPTANCE INSTANT. Never the filing day, never the
                period of report, never ingest time. An 8-K accepted 16:30 ET
                was not knowable that morning.
  event_date    when the thing happened: reportDate where EDGAR gives one,
                otherwise the acceptance instant (a filing with no separate
                report date IS the event).
  item_numbers  8-K items. This is the filter that removes most of the noise:
                across AAPL/NVDA/TSLA, 2.02 (results) fires 140 times, 5.02
                (executive departure) 50, 1.01 (material agreement) 21 — while
                9.01 (exhibits) fires 216 and means almost nothing on its own.

Append-only via sovereign.pit.store.append, so re-running is idempotent and an
amended filing lands as a new vintage rather than overwriting the original.

  --tickers AAPL NVDA      explicit list (default: the fundamentals watchlist)
  --forms 8-K 4 13F-HR     which forms (default: 8-K, 4, 13F-HR, 10-Q, 10-K, S-1)
  --since 2024-01-01       only filings accepted on/after this date
  --dry-run                fetch and report, write nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.fundamentals.instruments import files_with_sec  # noqa: E402
from sovereign.fundamentals.transports import sec  # noqa: E402
from sovereign.pit.store import append, rw_connect  # noqa: E402

DEFAULT_FORMS = ["8-K", "4", "13F-HR", "10-Q", "10-K", "S-1"]

WATCHLIST_DEFAULT = (
    "SPY,QQQ,MSFT,AAPL,TSLA,NVDA,AMZN,GOOGL,META,NFLX,AMD,CRM,"
    "ES=F,NQ=F,YM=F,CL=F,GC=F"
)


def watchlist() -> list[str]:
    raw = os.getenv("WATCHLIST_SYMBOLS", WATCHLIST_DEFAULT)
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # ETFs and futures file nothing under their own ticker; skip rather than
    # spend SEC calls proving a negative.
    return [s for s in syms if files_with_sec(s)]


def filing_url(cik: int, accession: str, primary_doc: str | None) -> str:
    acc = (accession or "").replace("-", "")
    if primary_doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_doc}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"


def to_row(ticker: str, cik: int, f: dict) -> dict | None:
    accepted = f.get("accepted_at")
    if accepted is None:
        # No acceptance instant and no filing date to fall back on. The layer
        # refuses NULL knowable_at rows, so writing it would create a silent
        # hole; skip and let the fetch log record the gap instead.
        return None

    report = f.get("report_date")
    event = (datetime(report.year, report.month, report.day, tzinfo=timezone.utc)
             if isinstance(report, date) else accepted)

    return {
        "accession_no": f.get("accession"),
        "cik": str(cik),
        "ticker": ticker.upper(),
        "form_type": f.get("form"),
        "item_numbers": f.get("items") or [],
        "period_of_report": report,
        "event_date": event.replace(tzinfo=None),
        "knowable_at": accepted.replace(tzinfo=None),
        "raw_text_path": None,          # full-text fetch is a separate pass
        "source_url": filing_url(cik, f.get("accession") or "", f.get("primary_document")),
        "source": "edgar",
        "ingested_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", metavar="SYM")
    ap.add_argument("--forms", nargs="+", default=DEFAULT_FORMS)
    ap.add_argument("--since", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or watchlist())]
    forms = set(args.forms)
    print(f"ingest_filings: {len(tickers)} tickers, forms={sorted(forms)}, "
          f"since={args.since or 'all'}, dry_run={args.dry_run}\n")

    con = None if args.dry_run else rw_connect()
    total = written = skipped = 0
    try:
        for t in tickers:
            try:
                cik = sec.resolve_cik(t)
            except Exception as e:  # noqa: BLE001
                print(f"  {t:6s} unresolved: {e}")
                continue
            try:
                filings = sec.recent_filings(cik, forms=forms, since=args.since)
            except Exception as e:  # noqa: BLE001
                print(f"  {t:6s} fetch failed: {e}")
                continue

            rows = []
            for f in filings:
                r = to_row(t, cik, f)
                if r is None:
                    skipped += 1
                else:
                    rows.append(r)
            total += len(rows)

            n = 0 if args.dry_run else append("filings", rows, con=con)
            written += n
            by_form: dict[str, int] = {}
            for r in rows:
                by_form[r["form_type"]] = by_form.get(r["form_type"], 0) + 1
            detail = " ".join(f"{k}:{v}" for k, v in sorted(by_form.items()))
            print(f"  {t:6s} {len(rows):4d} filings  (+{n} new)  {detail}")
    finally:
        if con is not None:
            con.close()

    print(f"\n{total} filings seen, {written} newly appended, {skipped} skipped "
          f"(no acceptance instant)")
    if args.dry_run:
        print("--dry-run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
