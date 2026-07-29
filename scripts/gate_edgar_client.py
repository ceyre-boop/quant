#!/usr/bin/env python3
"""
gate_edgar_client.py — SEC EDGAR fetcher for the Petrules Gate.

Pulls Form 4 (insider transactions) and SC 13D/13G (activist/large-holder)
filings from the free SEC EDGAR full-text search API and parses them into the
per-ticker feature dicts the scorer consumes.

SEC fair-access compliance (ticket non-negotiable):
  - A real User-Agent with contact info ("Alta Research colineyre222@gmail.com")
  - >= 1 second sleep between requests
  - Respect HTTP 429 with exponential backoff

DISCIPLINE:
  - Free data only. No paid API.
  - No silent failures. If EDGAR is unreachable, raise EdgarUnavailable so the
    scanner writes the honest error JSON. NEVER fabricate a filing or a score.
  - Standalone research module — no sovereign/ or ict/ imports.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index"


class EdgarUnavailable(RuntimeError):
    """Raised when EDGAR cannot be reached or repeatedly rate-limits us."""


class EdgarClient:
    def __init__(self, cfg: dict):
        e = cfg["edgar"]
        self.user_agent = e["user_agent"]
        self.sleep = float(e["request_sleep_sec"])
        self.max_retries = int(e["max_retries"])
        self.backoff_base = float(e["backoff_base_sec"])
        self.form4_lookback = int(e["form4_lookback_days"])
        self.activist_lookback = int(e["activist_lookback_days"])
        self.base = e.get("full_text_search", EDGAR_FTS)
        self._session = requests.Session() if requests else None

    # ── low-level fetch with rate-limit discipline ────────────────────────────
    def _get(self, url: str, params: dict) -> dict:
        if self._session is None:
            raise EdgarUnavailable("requests library not available")
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        attempt = 0
        while True:
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=30
                )
            except Exception as exc:
                raise EdgarUnavailable(f"EDGAR network error: {exc}") from exc
            if resp.status_code == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise EdgarUnavailable("EDGAR rate-limited (429) past max_retries")
                time.sleep(self.backoff_base ** attempt)
                continue
            if resp.status_code != 200:
                raise EdgarUnavailable(
                    f"EDGAR HTTP {resp.status_code} for {url}"
                )
            time.sleep(self.sleep)  # fair-access spacing between requests
            try:
                return resp.json()
            except Exception as exc:
                raise EdgarUnavailable(f"EDGAR non-JSON response: {exc}") from exc

    @staticmethod
    def _daterange(lookback_days: int) -> tuple[str, str]:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=lookback_days)
        return start.isoformat(), today.isoformat()

    # ── Form 4 insider clusters ───────────────────────────────────────────────
    def fetch_form4(self) -> dict[str, dict]:
        """Return {ticker: {n_buyers, n_sellers, total_buy_usd, includes_csuite,
        cluster_within_days}} aggregated over the lookback window.

        The EDGAR FTS API returns filing hits with metadata. We aggregate by
        display ticker. This is deliberately conservative: only open-market
        purchase/sale transaction codes are counted; option exercises excluded.
        """
        startdt, enddt = self._daterange(self.form4_lookback)
        data = self._get(
            self.base,
            {"q": '"form 4"', "forms": "4", "startdt": startdt, "enddt": enddt},
        )
        return self._aggregate_form4(data)

    @staticmethod
    def _aggregate_form4(data: dict) -> dict[str, dict]:
        agg: dict[str, dict] = defaultdict(
            lambda: {
                "buyers": set(),
                "sellers": set(),
                "total_buy_usd": 0.0,
                "includes_csuite": False,
                "filing_dates": [],
            }
        )
        for hit in data.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            tickers = src.get("tickers") or []
            if not tickers:
                continue
            ticker = tickers[0].replace(".", "-")
            # EDGAR FTS exposes display names; transaction detail requires the
            # filing document. We approximate direction/amount from the indexed
            # fields when present, and record filers for cluster counting.
            filer = (src.get("display_names") or ["?"])[0]
            txn = (src.get("transaction_code") or "").upper()
            amount = float(src.get("value") or 0.0)
            fdate = src.get("file_date") or src.get("filing_date") or ""
            entry = agg[ticker]
            entry["filing_dates"].append(fdate)
            title = filer.lower()
            if any(k in title for k in ("chief", "ceo", "cfo", "coo", "president")):
                entry["includes_csuite"] = True
            if txn == "P":  # open-market purchase
                entry["buyers"].add(filer)
                entry["total_buy_usd"] += amount
            elif txn == "S":  # open-market sale
                entry["sellers"].add(filer)
        out: dict[str, dict] = {}
        for ticker, e in agg.items():
            dates = sorted(d for d in e["filing_dates"] if d)
            cluster_days = _span_days(dates)
            out[ticker] = {
                "n_buyers": len(e["buyers"]),
                "n_sellers": len(e["sellers"]),
                "total_buy_usd": e["total_buy_usd"],
                "includes_csuite": e["includes_csuite"],
                "cluster_within_days": cluster_days,
            }
        return out

    # ── 13D/13G activist / large-holder ───────────────────────────────────────
    def fetch_activist(self) -> dict[str, dict]:
        """Return {ticker: {filing_type}} where filing_type is one of
        new_13d / amend_13d_increase / new_13g / amend_13g_increase.
        Strongest signal per ticker wins (13D > 13G, new > amend)."""
        startdt, enddt = self._daterange(self.activist_lookback)
        data = self._get(
            self.base,
            {"forms": "SC 13D,SC 13G", "startdt": startdt, "enddt": enddt},
        )
        return self._aggregate_activist(data)

    @staticmethod
    def _aggregate_activist(data: dict) -> dict[str, dict]:
        rank = {
            "new_13d": 4,
            "amend_13d_increase": 3,
            "new_13g": 2,
            "amend_13g_increase": 1,
            "none": 0,
        }
        out: dict[str, dict] = {}
        for hit in data.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            tickers = src.get("tickers") or []
            if not tickers:
                continue
            ticker = tickers[0].replace(".", "-")
            form = (src.get("form") or src.get("root_form") or "").upper()
            is_amend = "/A" in form
            if "13D" in form:
                ftype = "amend_13d_increase" if is_amend else "new_13d"
            elif "13G" in form:
                ftype = "amend_13g_increase" if is_amend else "new_13g"
            else:
                continue
            prev = out.get(ticker, {}).get("filing_type", "none")
            if rank[ftype] > rank[prev]:
                out[ticker] = {"filing_type": ftype}
        return out


def _span_days(sorted_dates: list[str]) -> int:
    if len(sorted_dates) < 2:
        return 0
    try:
        first = datetime.fromisoformat(sorted_dates[0][:10])
        last = datetime.fromisoformat(sorted_dates[-1][:10])
        return (last - first).days
    except Exception:
        return 0
