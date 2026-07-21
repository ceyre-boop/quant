"""Data-source layer — free public sources only, with a real local fixture cache.

Every source function returns either live-fetched public data (when the network is
reachable and a key is on file) or a cached real response from research/petrules/fixtures/.
Fixtures are verbatim copies of real probe responses (research/petrules_audit/probes/) — they
are NOT synthetic. If neither live nor fixture is available, the source returns None and the
caller records an ABSENT provenanced value. Nothing here fabricates a value.

Public sources used at Phase 1 (all free):
  - Alpha Vantage EARNINGS  : historical reported-vs-estimate EPS, back to 1996 (final
    pre-print consensus snapshot only — NOT the revision path; see the paid stub).
  - SEC EDGAR submissions   : Form 4 / 13D-G / 13F filing dates (free, no key).

Paid sources are never touched here — see paid_stubs.py.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
REPO_ROOT = HERE.parent.parent

UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}


def _load_env_key(name: str) -> Optional[str]:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    return None


def _fetch(url: str, headers=None, timeout=30) -> Optional[str]:
    """Best-effort live fetch. Returns None on any network failure (offline/firewalled)."""
    try:
        req = urllib.request.Request(url, headers=headers or UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


# ---------------------------------------------------------------- Alpha Vantage EARNINGS

# CIK map mirrors the Phase-0 probe universe (12 large/mid caps across sectors).
TICKERS_CIK = {
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "JPM": 19617,
    "XOM": 34088, "UNH": 731766, "CAT": 18230, "DE": 315189,
    "DKS": 1089063, "ETSY": 1370637, "CROX": 1334036, "WSM": 719955,
}

# Tickers whose real AV EARNINGS response is cached as a committed fixture (offline path).
_AV_FIXTURE = FIXTURES / "av_earnings_aapl_dks.json"
_AV_FIXTURE_TICKERS = {"AAPL", "DKS"}


def av_earnings(ticker: str) -> Optional[dict]:
    """Return the AV EARNINGS payload for `ticker` (dict with quarterlyEarnings), or None.

    Offline: served from the committed real fixture for AAPL/DKS.
    Live:    fetched from Alpha Vantage if a key is on file (25 req/day free tier).
    """
    if ticker in _AV_FIXTURE_TICKERS and _AV_FIXTURE.exists():
        blob = json.loads(_AV_FIXTURE.read_text())
        return blob.get(f"EARNINGS_{ticker}")
    key = _load_env_key("ALPHA_VANTAGE_API_KEY")
    if not key:
        return None
    raw = _fetch(f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={key}")
    time.sleep(1)  # be polite to the free tier
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if payload.get("quarterlyEarnings") else None


# ---------------------------------------------------------------- SEC EDGAR submissions

def edgar_submissions(cik: int) -> Optional[dict]:
    """EDGAR submissions JSON for a CIK (recent filings with form/filingDate/reportDate).

    Live-only (no committed fixture): returns None when the network is unavailable, and the
    disclosed-flow builders then emit ABSENT provenanced values rather than fabricated ones.
    """
    time.sleep(0.15)
    raw = _fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def edgar_recent_filings(sub: dict):
    """Yield (form, filingDate, reportDate, accession) tuples from an EDGAR submissions blob."""
    r = sub["filings"]["recent"]
    return list(zip(r["form"], r["filingDate"], r["reportDate"], r["accessionNumber"]))


# ---------------------------------------------------------------- 13F sample fixture

def edgar_13f_sample() -> Optional[dict]:
    """Real (filingDate, period-end) 13F pairs for Berkshire/Bridgewater/Renaissance.

    Used to demonstrate the filing-date-vs-period-of-report gate with real dates offline.
    """
    p = FIXTURES / "edgar_13f_samples.json"
    return json.loads(p.read_text()) if p.exists() else None
