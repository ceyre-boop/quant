"""Alpha Vantage EARNINGS transport.

Free tier is 25 requests/day; CallBudget is a hard in-process ceiling we
enforce ourselves rather than discovering the limit from AV's throttle
response, which is a 200 with an error-shaped body ("Note"/"Information")
instead of a 429 — silently parsing that as data would fabricate rows.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sovereign.fundamentals.errors import BudgetExhausted, SectionUnavailable
from sovereign.fundamentals.types import EarningsEvent

_SOURCE = "alphavantage"

REPO_ROOT = Path(__file__).resolve().parents[3]

UA = {"User-Agent": "Alta Research colineyre222@gmail.com"}


class CallBudget:
    """Hard per-run call ceiling. `.take()` raises BudgetExhausted past the cap
    instead of letting the caller find out from a throttled/garbage response."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def take(self) -> None:
        if self.used >= self.max_calls:
            raise BudgetExhausted(
                f"alphavantage call budget exhausted ({self.used}/{self.max_calls})"
            )
        self.used += 1


def _load_env_key(name: str) -> Optional[str]:
    """Self-contained .env reader, ported from research/petrules/sources.py's
    _load_env_key (not imported — sovereign/ must not depend on research/)."""
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


def _fetch(url: str, timeout: int = 30) -> str:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        raise SectionUnavailable("earnings", _SOURCE, f"{url}: {e}") from e


def _to_float(s) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def earnings(ticker: str, budget: CallBudget) -> list[EarningsEvent]:
    key = _load_env_key("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise SectionUnavailable("earnings", _SOURCE, "ALPHA_VANTAGE_API_KEY not set in .env")

    budget.take()
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={key}"
    raw = _fetch(url)
    time.sleep(1)  # be polite to the free tier, per research/petrules/sources.py's idiom

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SectionUnavailable("earnings", _SOURCE, f"non-JSON response: {e}") from e

    # AV's throttle/error responses are HTTP 200 with an error-shaped body
    # instead of a real payload — must be detected explicitly or a throttled
    # call silently looks like "ticker has no earnings".
    if "Note" in payload or "Information" in payload or "Error Message" in payload:
        reason = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        raise SectionUnavailable("earnings", _SOURCE, f"throttled/error response: {reason}")

    quarterly = payload.get("quarterlyEarnings")
    if not quarterly:
        return []

    out: list[EarningsEvent] = []
    for row in quarterly:
        report_date = _parse_date(row.get("reportedDate"))
        fiscal_end = _parse_date(row.get("fiscalDateEnding"))
        eps_est = _to_float(row.get("estimatedEPS"))
        eps_act = _to_float(row.get("reportedEPS"))
        eps_surprise = _to_float(row.get("surprise"))
        eps_surprise_pct = _to_float(row.get("surprisePercentage"))

        out.append(EarningsEvent(
            source=_SOURCE,
            # published_ts = the report date itself (AV gives no separate filing
            # timestamp); this is the same "as of the print" instant used by
            # yahoo.earnings_history for consistency across providers.
            published_ts=datetime.combine(report_date, datetime.min.time()) if report_date else None,
            ticker=ticker.upper(),
            fiscal_end=fiscal_end,
            report_date=report_date,
            report_time=(row.get("reportTime") or "unknown").lower() if row.get("reportTime") else "unknown",
            eps_estimate=eps_est,
            eps_actual=eps_act,
            eps_surprise=eps_surprise,
            eps_surprise_pct=eps_surprise_pct,
        ))
    return out
