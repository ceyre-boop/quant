#!/usr/bin/env python3
"""scripts/harvest_13f_bulk.py — SEC 13F quarterly bulk ingest.

Downloads a quarterly SEC "Form 13F structured dataset" ZIP (SUBMISSION.tsv +
COVERPAGE.tsv + INFOTABLE.tsv, ~2-4M holding rows/quarter), resolves CUSIPs to
tickers via OpenFIGI (batched, keyless, 25 req/min, cached permanently in
fund_symbol), and loads institutional holdings into
sovereign.fundamentals.store's fund_institution_holding table, then recomputes
fund_institution_agg (QoQ deltas, new/closed positions, top buyers/sellers).

The index page (not a guessed YYYYQN URL — verified 404s for anything after
2023, the dataset moved to date-range filenames in 2024) is scraped for real
hrefs:

    https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets

Everything CSV-shaped is processed via DuckDB's own read_csv + SQL (never
pandas) — INFOTABLE.tsv alone is ~400MB/~3.8M rows per quarter, multiple
times too large to be a sane pandas frame, and DuckDB's CSV engine is the
right tool already used elsewhere in this repo (store.py itself is DuckDB).

Usage:
    python scripts/harvest_13f_bulk.py --latest                 # most recent quarter (default)
    python scripts/harvest_13f_bulk.py --quarter 2026Q1
    python scripts/harvest_13f_bulk.py --latest --max-figi-batches 20  # bound OpenFIGI calls
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.fundamentals import store

log = logging.getLogger("harvest_13f_bulk")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
FILE_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}

CACHE_DIR = ROOT / "data" / "cache" / "fundamentals" / "sec_13f_bulk"
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
# Verified live 2026-09-01: the KEYLESS tier caps a single request at 10 mapping
# jobs ("Request may only contain 10 mapping jobs", HTTP 413 above that) — 100
# per request is the WITH-API-KEY limit. Using 100 here silently 413s every
# batch, which looks like "nothing resolves" rather than a batch-size bug.
OPENFIGI_BATCH = 10
OPENFIGI_RATE_PER_MIN = 25  # keyless tier

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_QTR_RE = re.compile(r"(\d{4})q([1-4])_form13f\.zip$", re.IGNORECASE)
_RANGE_RE = re.compile(r"\d{2}([a-z]{3})(\d{4})-(\d{2})([a-z]{3})(\d{4})_form13f\.zip$", re.IGNORECASE)


@dataclass
class Dataset:
    href: str          # relative path under FILE_BASE
    filename: str
    period_end: date   # sort anchor ("latest" = greatest window/quarter end)
    quarter_key: tuple[int, int]  # (year, quarter) this file's window is built to catch,
                                   # for --quarter lookup — see _parse_dataset for the mapping


def _quarter_of(d: date) -> int:
    return (d.month - 1) // 3 + 1


def _parse_dataset(href: str) -> Optional[Dataset]:
    filename = href.rsplit("/", 1)[-1]
    m = _QTR_RE.search(filename)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        end_month = q * 3
        end_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
        return Dataset(href, filename, date(year, end_month, end_day), (year, q))
    m = _RANGE_RE.search(filename)
    if m:
        # group(1)=start mon, group(2)=start year, group(3)=end day,
        # group(4)=end mon, group(5)=end year.
        start_month = _MONTHS.get(m.group(1).lower())
        end_month = _MONTHS.get(m.group(4).lower())
        if start_month is None or end_month is None:
            return None
        window_start = date(int(m.group(2)), start_month, 1)
        window_end = date(int(m.group(5)), end_month, int(m.group(3)))
        # 2024+ filenames are FILING-WINDOW ranges, not report periods (verified
        # against the live index: "01mar2026-31may2026" spans the ~45-day 13F-HR
        # deadline window for the quarter ENDING in the window's START month,
        # e.g. Q1 2026 ends 2026-03-31, deadline mid-May, window Mar1-May31).
        # So the report quarter this file targets is the quarter containing
        # window_start's month, in window_start's year.
        quarter_key = (window_start.year, _quarter_of(window_start))
        return Dataset(href, filename, window_end, quarter_key)
    return None


def list_datasets() -> list[Dataset]:
    """Scrape the SEC 13F data-sets index page for real ZIP hrefs — filenames
    are NOT guessable (pre-2024 is YYYYqN, 2024+ moved to date-range names;
    a guessed YYYYqN URL for a recent quarter 404s, verified by hand)."""
    req = urllib.request.Request(INDEX_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    hrefs = set(re.findall(r'href="(/files/structureddata/data/form-13f-data-sets/[^"]+\.zip)"', html))
    out = []
    for href in hrefs:
        ds = _parse_dataset(href)
        if ds:
            out.append(ds)
    out.sort(key=lambda d: d.period_end)
    return out


def pick_dataset(datasets: list[Dataset], quarter: Optional[str], latest: bool) -> Dataset:
    if quarter:
        m = re.match(r"(\d{4})[qQ]([1-4])$", quarter)
        if not m:
            raise ValueError(f"--quarter must look like 2026Q1, got {quarter!r}")
        target = (int(m.group(1)), int(m.group(2)))
        for ds in datasets:
            if ds.quarter_key == target:
                return ds
        raise ValueError(f"no dataset found targeting report quarter {quarter}")
    if not datasets:
        raise ValueError("no datasets found on the SEC index page")
    return datasets[-1]  # sorted ascending by period_end


# ── download + extract ──────────────────────────────────────────────────

def download(ds: Dataset) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / ds.filename
    if dest.exists() and dest.stat().st_size > 0:
        log.info("download: %s already cached (%d bytes)", ds.filename, dest.stat().st_size)
        return dest
    url = FILE_BASE + ds.filename
    log.info("download: fetching %s", url)
    req = urllib.request.Request(url, headers=UA)
    tmp = dest.with_suffix(".zip.part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    log.info("download: wrote %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def extract(zip_path: Path) -> Path:
    extract_dir = CACHE_DIR / zip_path.stem
    needed = ["INFOTABLE.tsv", "SUBMISSION.tsv", "COVERPAGE.tsv"]
    if all((extract_dir / n).exists() for n in needed):
        log.info("extract: %s already extracted", extract_dir)
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for n in needed:
            zf.extract(n, extract_dir)
    return extract_dir


# ── OpenFIGI CUSIP -> ticker resolution ─────────────────────────────────

def resolve_cusips(cusips: list[str]) -> dict[str, str]:
    """Batch-resolve CUSIPs -> ticker via OpenFIGI's keyless mapping endpoint
    (25 req/min, up to OPENFIGI_BATCH idValues per request — 10 keyless, verified
    live; 100 needs an API key). Returns only CUSIPs that
    resolved; callers persist the result into fund_symbol so it's never
    looked up twice."""
    out: dict[str, str] = {}
    calls = 0
    for i in range(0, len(cusips), OPENFIGI_BATCH):
        batch = cusips[i:i + OPENFIGI_BATCH]
        payload = json.dumps([{"idType": "ID_CUSIP", "idValue": c} for c in batch]).encode()
        req = urllib.request.Request(
            OPENFIGI_URL, data=payload, method="POST",
            headers={"Content-Type": "application/json", **UA},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                results = json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            log.warning("resolve_cusips: batch %d failed: %s", i // OPENFIGI_BATCH, e)
            continue
        for cusip, result in zip(batch, results):
            data = result.get("data") if isinstance(result, dict) else None
            if not data:
                continue
            # Prefer a US-listed composite entry (matches the verified AAPL
            # example: the first US-exchange row carries the canonical ticker).
            chosen = next((d for d in data if d.get("exchCode") == "US"), data[0])
            ticker = chosen.get("ticker")
            if ticker:
                out[cusip] = ticker.upper()
        calls += 1
        if calls % OPENFIGI_RATE_PER_MIN == 0:
            log.info("resolve_cusips: %d batches done, pausing for the 25 req/min ceiling", calls)
            time.sleep(61)
    return out


# ── ingest ───────────────────────────────────────────────────────────────

def quarter_end_date(quarter_key: tuple[int, int]) -> date:
    year, q = quarter_key
    end_month = q * 3
    end_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
    return date(year, end_month, end_day)


def already_ingested(period_end: date) -> bool:
    with store.connect() as con:
        row = con.execute(
            "SELECT count(*) FROM fund_institution_holding WHERE period_end = ?", [period_end]
        ).fetchone()
    return bool(row and row[0])


def ingest(ds: Dataset, extract_dir: Path, max_figi_batches: Optional[int]) -> int:
    infotable = extract_dir / "INFOTABLE.tsv"
    submission = extract_dir / "SUBMISSION.tsv"
    coverpage = extract_dir / "COVERPAGE.tsv"

    with store.connect() as con:
        # Everything below runs as SQL against DuckDB's own CSV reader —
        # never pandas — per the constraint: these TSVs run into the hundreds
        # of MB / millions of rows.
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE stg_infotable AS
            SELECT ACCESSION_NUMBER AS accession, CUSIP AS cusip,
                   NAMEOFISSUER AS issuer_name,
                   sum(TRY_CAST(SSHPRNAMT AS DOUBLE)) AS shares,
                   sum(TRY_CAST(VALUE AS DOUBLE)) * 1000 AS value_usd
            FROM read_csv('{infotable.as_posix()}', delim='\t', header=true, quote='',
                          columns={{'ACCESSION_NUMBER':'VARCHAR','INFOTABLE_SK':'VARCHAR',
                          'NAMEOFISSUER':'VARCHAR','TITLEOFCLASS':'VARCHAR','CUSIP':'VARCHAR',
                          'FIGI':'VARCHAR','VALUE':'VARCHAR','SSHPRNAMT':'VARCHAR',
                          'SSHPRNAMTTYPE':'VARCHAR','PUTCALL':'VARCHAR','INVESTMENTDISCRETION':'VARCHAR',
                          'OTHERMANAGER':'VARCHAR','VOTING_AUTH_SOLE':'VARCHAR',
                          'VOTING_AUTH_SHARED':'VARCHAR','VOTING_AUTH_NONE':'VARCHAR'}})
            GROUP BY ACCESSION_NUMBER, CUSIP, NAMEOFISSUER
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE stg_submission AS
            SELECT ACCESSION_NUMBER AS accession, CIK AS filer_cik,
                   strptime(FILING_DATE, '%d-%b-%Y')::DATE AS filing_date,
                   strptime(PERIODOFREPORT, '%d-%b-%Y')::DATE AS period_end,
                   SUBMISSIONTYPE AS submission_type
            FROM read_csv('{submission.as_posix()}', delim='\t', header=true, quote='')
            WHERE SUBMISSIONTYPE IN ('13F-HR', '13F-HR/A')
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE stg_coverpage AS
            SELECT ACCESSION_NUMBER AS accession, FILINGMANAGER_NAME AS filer_name,
                   (ISAMENDMENT = 'Y') AS is_amendment
            FROM read_csv('{coverpage.as_posix()}', delim='\t', header=true, quote='')
        """)

        n_holding_rows = con.execute("SELECT count(*) FROM stg_infotable").fetchone()[0]
        log.info("ingest: %d aggregated (accession, cusip) rows in INFOTABLE", n_holding_rows)

        # CUSIP -> ticker: known ones from fund_symbol first.
        distinct_cusips = [r[0] for r in con.execute(
            "SELECT DISTINCT cusip FROM stg_infotable WHERE cusip != ''"
        ).fetchall()]
        known = {r[0]: r[1] for r in con.execute(
            "SELECT cusip, ticker FROM fund_symbol WHERE cusip IS NOT NULL AND ticker IS NOT NULL"
        ).fetchall()}
        unresolved = [c for c in distinct_cusips if c not in known]
        log.info("ingest: %d distinct CUSIPs, %d already cached, %d to resolve via OpenFIGI",
                 len(distinct_cusips), len(known), len(unresolved))

        if unresolved and max_figi_batches:
            cap = max_figi_batches * OPENFIGI_BATCH
            to_resolve = unresolved[:cap]
            if len(unresolved) > cap:
                log.info("ingest: capping OpenFIGI resolution to %d/%d CUSIPs "
                         "(--max-figi-batches=%d) — re-run to pick up more",
                         cap, len(unresolved), max_figi_batches)
            resolved = resolve_cusips(to_resolve)
            log.info("ingest: OpenFIGI resolved %d/%d requested CUSIPs", len(resolved), len(to_resolve))
            now = datetime.utcnow()
            for cusip, ticker in resolved.items():
                con.execute(
                    "INSERT OR REPLACE INTO fund_symbol (ticker, cusip, updated_at) VALUES (?, ?, ?)",
                    [ticker, cusip, now],
                )

        con.execute("""
            CREATE OR REPLACE TEMP TABLE stg_final AS
            SELECT s.period_end AS period_end, s.filer_cik AS filer_cik, i.cusip AS cusip,
                   sym.ticker AS ticker, cp.filer_name AS filer_name, s.filing_date AS filing_date,
                   i.shares AS shares, i.value_usd AS value_usd, cp.is_amendment AS is_amendment
            FROM stg_infotable i
            JOIN stg_submission s ON s.accession = i.accession
            LEFT JOIN stg_coverpage cp ON cp.accession = i.accession
            LEFT JOIN fund_symbol sym ON sym.cusip = i.cusip
            WHERE i.cusip != ''
        """)

        n_ticker_resolved = con.execute(
            "SELECT count(*) FROM stg_final WHERE ticker IS NOT NULL"
        ).fetchone()[0]
        n_total = con.execute("SELECT count(*) FROM stg_final").fetchone()[0]
        log.info("ingest: %d/%d final rows carry a resolved ticker", n_ticker_resolved, n_total)

        con.execute("""
            INSERT OR REPLACE INTO fund_institution_holding
                (period_end, filer_cik, cusip, ticker, filer_name, filing_date,
                 shares, value_usd, is_amendment, source, fetched_at)
            SELECT period_end, filer_cik, cusip, ticker, filer_name, filing_date,
                   shares, value_usd, is_amendment, 'sec_13f_bulk', now()
            FROM stg_final
        """)
        target_period_end = quarter_end_date(ds.quarter_key)
        inserted = con.execute(
            "SELECT count(*) FROM fund_institution_holding WHERE period_end = ?",
            [target_period_end],
        ).fetchone()[0]

    return inserted


def recompute_agg(period_end: date) -> None:
    """Recompute fund_institution_agg for period_end and (if present) the
    immediately preceding period on file, so QoQ deltas/new/closed/top
    buyers-sellers are available for whichever period a caller asks about."""
    with store.connect() as con:
        periods = [r[0] for r in con.execute(
            "SELECT DISTINCT period_end FROM fund_institution_holding "
            "WHERE ticker IS NOT NULL ORDER BY period_end DESC"
        ).fetchall()]
        if period_end not in periods:
            log.warning("recompute_agg: %s has no ticker-resolved holdings; skipping", period_end)
            return
        idx = periods.index(period_end)
        prior_period = periods[idx + 1] if idx + 1 < len(periods) else None

        tickers = [r[0] for r in con.execute(
            "SELECT DISTINCT ticker FROM fund_institution_holding WHERE period_end = ? AND ticker IS NOT NULL",
            [period_end],
        ).fetchall()]
        log.info("recompute_agg: %s vs prior=%s, %d tickers", period_end, prior_period, len(tickers))

        for ticker in tickers:
            current = con.execute(
                "SELECT filer_cik, filer_name, shares FROM fund_institution_holding "
                "WHERE period_end = ? AND ticker = ?", [period_end, ticker],
            ).fetchall()
            prior = con.execute(
                "SELECT filer_cik, shares FROM fund_institution_holding "
                "WHERE period_end = ? AND ticker = ?", [prior_period, ticker],
            ).fetchall() if prior_period else []

            cur_by_cik = {c: (n, s or 0) for c, n, s in current}
            prior_by_cik = {c: (s or 0) for c, s in prior}

            n_holders = len(cur_by_cik)
            total_shares = sum(s for _n, s in cur_by_cik.values())

            # A quarter-over-quarter delta is only meaningful when we actually
            # OBSERVED the prior quarter for THIS ticker. The holdings query
            # filters on `ticker`, so a prior quarter whose CUSIPs were never
            # resolved to tickers comes back empty -- and then
            # `total_shares - 0` reports the entire position as this quarter's
            # accumulation. On real data that produced BAC d_shares_qoq = +5.26B
            # against total_shares = 5.48B: every institution appearing to buy in
            # at once, which is not a thing that happens.
            #
            # "We have no prior observation" and "the position was near zero" are
            # different facts and must not render as the same number.
            comparable = bool(prior_period) and bool(prior_by_cik)

            d_holders_qoq = (n_holders - len(prior_by_cik)) if comparable else None
            d_shares_qoq = (total_shares - sum(prior_by_cik.values())) if comparable else None
            new_positions = sum(1 for c in cur_by_cik if c not in prior_by_cik) if comparable else None
            closed_positions = sum(1 for c in prior_by_cik if c not in cur_by_cik) if comparable else None

            deltas = []
            if comparable:
                for c, (name, shares) in cur_by_cik.items():
                    deltas.append((name, shares, shares - prior_by_cik.get(c, 0)))
                for c, shares in prior_by_cik.items():
                    if c not in cur_by_cik:
                        deltas.append(("(closed)", 0, -shares))
            top_buyers = sorted([d for d in deltas if d[2] > 0], key=lambda d: d[2], reverse=True)[:25]
            top_sellers = sorted([d for d in deltas if d[2] < 0], key=lambda d: d[2])[:25]

            con.execute(
                """
                INSERT OR REPLACE INTO fund_institution_agg
                    (period_end, ticker, n_holders, total_shares, total_value_usd,
                     d_shares_qoq, d_holders_qoq, new_positions, closed_positions,
                     top_buyers, top_sellers, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [period_end, ticker, n_holders, total_shares, None,
                 d_shares_qoq, d_holders_qoq, new_positions, closed_positions,
                 json.dumps([{"filer_name": n, "shares": s, "d_shares": d} for n, s, d in top_buyers]),
                 json.dumps([{"filer_name": n, "shares": s, "d_shares": d} for n, s, d in top_sellers]),
                 datetime.utcnow()],
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quarter", metavar="YYYYQN", help="e.g. 2026Q1")
    ap.add_argument("--latest", action="store_true", default=True, help="most recent quarter on file (default)")
    ap.add_argument("--max-figi-batches", type=int, default=50,
                    help="cap OpenFIGI resolution batches (OPENFIGI_BATCH CUSIPs/batch, 25/min) per run; re-run to continue")
    ap.add_argument("--force", action="store_true", help="re-ingest even if this period_end already has rows")
    args = ap.parse_args()

    datasets = list_datasets()
    log.info("found %d candidate datasets on the SEC index page", len(datasets))
    ds = pick_dataset(datasets, args.quarter, args.latest)
    target_period_end = quarter_end_date(ds.quarter_key)
    log.info("selected dataset: %s (targets report quarter %s, period_end=%s)",
             ds.filename, ds.quarter_key, target_period_end)

    if not args.force and already_ingested(target_period_end):
        log.info("period_end %s already has fund_institution_holding rows — no-op (use --force to re-ingest)",
                 target_period_end)
        return

    zip_path = download(ds)
    extract_dir = extract(zip_path)
    inserted = ingest(ds, extract_dir, args.max_figi_batches)
    log.info("ingest complete: %d rows in fund_institution_holding for period_end=%s", inserted, target_period_end)

    recompute_agg(target_period_end)
    log.info("done")


if __name__ == "__main__":
    main()
