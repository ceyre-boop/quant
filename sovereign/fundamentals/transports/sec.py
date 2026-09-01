"""SEC EDGAR transport: ticker->CIK resolution, submissions, and Form 4 parsing.

Ports the fetch idiom from research/petrules/sources.py (UA string, best-effort
urllib fetch) rather than importing it — sovereign/ must not depend on research/.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from sovereign.fundamentals.errors import SectionUnavailable, TickerUnresolved
from sovereign.fundamentals.types import InsiderTxn

try:
    from sovereign.fundamentals.httpcache import (
        TTLClass,
        get_json as _cache_get_json,
        get_text as _cache_get_text,
    )
except ImportError:  # httpcache lands in parallel; degrade to a direct fetch until it does.
    TTLClass = None
    _cache_get_json = None
    _cache_get_text = None

UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}

REPO_ROOT = Path(__file__).resolve().parents[3]
CIK_MAP_PATH = REPO_ROOT / "data" / "fundamentals" / "cik_map.json"

_SOURCE = "sec_edgar"

# SEC fair-access policy: throttle to <=5 req/s and always send a UA. A single
# module-level lock + timestamp is enough since all callers in-process share it.
_RATE_LOCK = threading.Lock()
_MIN_INTERVAL = 1.0 / 5.0
_last_call_ts = 0.0


def _throttle() -> None:
    global _last_call_ts
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _fetch_status(url: str, headers: Optional[dict] = None, timeout: int = 30) -> tuple[int, str]:
    """Rate-limited GET returning (http_status, text) — the shape httpcache's
    fetcher callback expects, so a non-2xx response is never cached."""
    _throttle()
    try:
        req = urllib.request.Request(url, headers=headers or UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError) as e:
        raise SectionUnavailable("sec_fetch", _SOURCE, f"{url}: {e}") from e


def _get(url: str, headers: Optional[dict] = None, timeout: int = 30) -> str:
    """Rate-limited GET. Raises SectionUnavailable on any network failure —
    unlike research/petrules/sources.py's best-effort `_fetch`, this transport
    must distinguish "fetched, empty" from "could not fetch" per errors.py."""
    status, text = _fetch_status(url, headers, timeout)
    if not (200 <= status < 300):
        raise SectionUnavailable("sec_fetch", _SOURCE, f"{url}: HTTP {status}")
    return text


def _get_json_cached(key: str, url: str, ttl_class: str, ticker: Optional[str] = None) -> dict:
    """Route through httpcache when available, else fetch directly (uncached)."""
    if _cache_get_json is not None:
        cls = getattr(TTLClass, ttl_class)
        try:
            return _cache_get_json(_SOURCE, key, cls, lambda: _fetch_status(url), ticker=ticker)
        except json.JSONDecodeError as e:
            raise SectionUnavailable("sec_fetch", _SOURCE, f"{url}: {e}") from e
    return json.loads(_get(url))


def _get_text_cached(key: str, url: str, ttl_class: str, ticker: Optional[str] = None) -> str:
    if _cache_get_text is not None:
        cls = getattr(TTLClass, ttl_class)
        return _cache_get_text(_SOURCE, key, cls, lambda: _fetch_status(url), ticker=ticker)
    return _get(url)


# ---------------------------------------------------------------- CIK map

def load_cik_map(refresh: bool = False) -> dict[str, int]:
    """Fetch (or load cached) the full SEC ticker->CIK map.

    Caches to data/fundamentals/cik_map.json as a flat {"AAPL": 320193, ...}
    object — this same file doubles as the browser-consumable resolver so the
    front end can map tickers to CIKs with zero network calls of its own.
    """
    if not refresh and CIK_MAP_PATH.exists():
        return json.loads(CIK_MAP_PATH.read_text())

    raw = _get("https://www.sec.gov/files/company_tickers.json")
    payload: dict[str, Any] = json.loads(raw)
    flat: dict[str, int] = {}
    for row in payload.values():
        ticker = row.get("ticker")
        cik = row.get("cik_str")
        if ticker and cik is not None:
            flat[ticker.upper()] = int(cik)

    CIK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CIK_MAP_PATH.write_text(json.dumps(flat, sort_keys=True))
    return flat


def resolve_cik(ticker: str) -> int:
    m = load_cik_map()
    cik = m.get(ticker.upper())
    if cik is None:
        raise TickerUnresolved(f"{ticker} not found in SEC company_tickers.json")
    return cik


# ---------------------------------------------------------------- submissions

def submissions(cik: int) -> dict:
    """Full submissions JSON for a CIK. Cached DAILY — filings post intraday."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    return _get_json_cached(f"submissions_{cik}", url, "DAILY")


def recent_filings(
    cik: int,
    forms: Optional[set[str]] = None,
    since: Optional[date] = None,
) -> list[dict]:
    """Flatten filings.recent's parallel arrays into per-filing dicts.

    Keys: form, filing_date (date), accession, primary_document, report_date.
    """
    sub = submissions(cik)
    recent = sub.get("filings", {}).get("recent", {})
    forms_arr = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    out: list[dict] = []
    for i, form in enumerate(forms_arr):
        if forms is not None and form not in forms:
            continue
        fdate = _parse_date(filing_dates[i]) if i < len(filing_dates) else None
        if since is not None and fdate is not None and fdate < since:
            continue
        out.append({
            "form": form,
            "filing_date": fdate,
            "accession": accessions[i] if i < len(accessions) else None,
            "primary_document": primary_docs[i] if i < len(primary_docs) else None,
            "report_date": _parse_date(report_dates[i]) if i < len(report_dates) else None,
        })
    return out


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------- Form 4

def _form4_xml_url(cik: int, accession: str, primary_document: str) -> str:
    """The submissions feed's primaryDocument for XML forms is the XSLT-rendered
    display path (e.g. "xslF345X06/form4.xml"), served as HTML. The raw
    ownership XML lives at the same accession folder under just the basename
    (e.g. ".../{accession_nodash}/form4.xml"), verified against a live filing."""
    accession_nodash = accession.replace("-", "")
    basename = primary_document.rsplit("/", 1)[-1]
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{basename}"


def _xml_value(elem: Optional[ET.Element], tag: str) -> Optional[str]:
    """SEC ownership XML inconsistently wraps scalars: sometimes
    <tag><value>X</value></tag>, sometimes <tag>X</tag> directly. Read either."""
    if elem is None:
        return None
    child = elem.find(tag)
    if child is None:
        return None
    value_child = child.find("value")
    if value_child is not None:
        return (value_child.text or "").strip() or None
    return (child.text or "").strip() or None


def _xml_float(elem: Optional[ET.Element], tag: str) -> Optional[float]:
    s = _xml_value(elem, tag)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _xml_bool(elem: Optional[ET.Element], tag: str) -> bool:
    s = _xml_value(elem, tag)
    if s is None:
        return False
    return s.strip().lower() in ("1", "true")


def _parse_form4_xml(xml_text: str, ticker: str, accession: str, filing_date: Optional[date]) -> list[InsiderTxn]:
    root = ET.fromstring(xml_text)

    owner_el = root.find("reportingOwner")
    owner_name = ""
    if owner_el is not None:
        owner_id = owner_el.find("reportingOwnerId")
        owner_name = _xml_value(owner_id, "rptOwnerName") or ""
    rel_el = owner_el.find("reportingOwnerRelationship") if owner_el is not None else None
    is_director = _xml_bool(rel_el, "isDirector")
    is_officer = _xml_bool(rel_el, "isOfficer")
    is_ten_pct = _xml_bool(rel_el, "isTenPercentOwner")
    owner_title = _xml_value(rel_el, "officerTitle") or ""

    txns: list[InsiderTxn] = []
    table = root.find("nonDerivativeTable")
    if table is None:
        return txns

    for i, txn_el in enumerate(table.findall("nonDerivativeTransaction")):
        coding = txn_el.find("transactionCoding")
        code = _xml_value(coding, "transactionCode") or ""

        amounts = txn_el.find("transactionAmounts")
        shares = _xml_float(amounts, "transactionShares")
        price = _xml_float(amounts, "transactionPricePerShare")
        acq_disp = _xml_value(amounts, "transactionAcquiredDisposedCode") or ""
        # A/D code, not the transaction code, tells us direction. Code F (tax
        # withholding) and A (grant) are administrative/compensatory events, not
        # a decision to buy or sell — see InsiderTxn.is_open_market, which only
        # trusts P/S. We still sign shares here so magnitude nets correctly.
        if shares is not None and acq_disp == "D":
            shares = -shares

        post = txn_el.find("postTransactionAmounts")
        shares_after = _xml_float(post, "sharesOwnedFollowingTransaction")

        txn_date_s = _xml_value(txn_el, "transactionDate")
        txn_date = _parse_date(txn_date_s)

        value_usd = shares * price if (shares is not None and price is not None) else None

        txns.append(InsiderTxn(
            source=_SOURCE,
            # published_ts is the FILING date, never the transaction date: a
            # trade on the 3rd disclosed on the 5th was not knowable until the
            # 5th, and keying on the 3rd poisons any backtest that consumes this.
            published_ts=datetime.combine(filing_date, datetime.min.time()) if filing_date else None,
            ticker=ticker.upper(),
            accession=accession,
            line_no=i,
            owner_name=owner_name,
            owner_title=owner_title,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_pct=is_ten_pct,
            txn_date=txn_date,
            filing_date=filing_date,
            code=code,
            shares=shares,
            price=price,
            value_usd=value_usd,
            shares_after=shares_after,
        ))
    return txns


def form4_transactions(
    ticker: str,
    cik: int,
    since: date,
    max_filings: int = 40,
) -> list[InsiderTxn]:
    """Fetch and parse up to max_filings Form 4s filed since `since`."""
    try:
        filings = recent_filings(cik, forms={"4"}, since=since)
    except SectionUnavailable:
        raise
    filings = filings[:max_filings]

    out: list[InsiderTxn] = []
    for f in filings:
        accession = f["accession"]
        primary_document = f["primary_document"]
        if not accession or not primary_document:
            continue
        url = _form4_xml_url(cik, accession, primary_document)
        try:
            xml_text = _get_text_cached(f"form4_{accession}", url, "IMMUTABLE", ticker=ticker)
        except SectionUnavailable:
            # One bad filing shouldn't kill the whole batch; skip and continue.
            continue
        try:
            out.extend(_parse_form4_xml(xml_text, ticker, accession, f["filing_date"]))
        except ET.ParseError:
            continue
    return out
