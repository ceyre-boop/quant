"""FreeProvider — the zero-cost provider, composed of the verified transports.

capabilities() excludes "guidance" and "revisions": no free source in this
repo exposes forward guidance or a point-in-time analyst-revision path (see
transports/yahoo.py's estimate_snapshot docstring and research/petrules's
Phase-0 audit, which killed the same gap for a different product). That
absence is documented here rather than silently producing empty rows for
those two sections — the registry/panel layer uses capabilities() to render
the gap as "not supported free" instead of "fetch failed".
"""
from __future__ import annotations

import glob
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sovereign.fundamentals.errors import BudgetExhausted, SectionUnavailable
from sovereign.fundamentals.providers.base import FundamentalsProvider
from sovereign.fundamentals.store import DB_PATH, connect
from sovereign.fundamentals.transports import alphavantage, finra, nasdaq, sec, yahoo
from sovereign.fundamentals.types import (
    SECTIONS,
    BorrowPoint,
    EarningsEvent,
    EstimateSnapshot,
    InsiderTxn,
    InstitutionalPosition,
    ShortInterestPoint,
    ShortVolumePoint,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
IB_LOCATE_GLOB = str(REPO_ROOT / "data" / "research" / "gapper" / "ib_locate_*.json")

_CAPABILITIES = SECTIONS - {"guidance", "revisions"}


class FreeProvider(FundamentalsProvider):
    name = "free"

    def __init__(self, av_budget: Optional[alphavantage.CallBudget] = None):
        # av_budget is optional: earnings_history only reaches for Alpha
        # Vantage as a depth top-up when yahoo's history is thin, and only if
        # a budget was actually handed to this instance — see earnings_history.
        self._av_budget = av_budget

    def capabilities(self) -> frozenset[str]:
        return _CAPABILITIES

    # ── earnings ─────────────────────────────────────────────────────────

    def earnings_history(
        self, ticker: str, cik: int | None = None, limit: int = 20
    ) -> list[EarningsEvent]:
        try:
            yahoo_rows = yahoo.earnings_history(ticker, limit=max(limit, 24))
        except Exception as e:  # yfinance raises assorted exceptions, not a typed hierarchy
            yahoo_rows = None
            yahoo_err = e
        else:
            yahoo_err = None

        if yahoo_rows is None and self._av_budget is None:
            raise SectionUnavailable("earnings", self.name, f"yahoo failed: {yahoo_err}")

        rows_by_fiscal: dict[Optional[date], EarningsEvent] = {}
        for r in (yahoo_rows or []):
            rows_by_fiscal[r.fiscal_end or r.report_date] = r

        # Fall back to Alpha Vantage ONLY when a budget was configured and
        # yahoo's history is thin (<12 rows) — AV's free tier is 25 calls/day
        # total across the whole harvest run, so it is never the default path.
        thin = len(rows_by_fiscal) < 12
        if self._av_budget is not None and (yahoo_rows is None or thin):
            try:
                av_rows = alphavantage.earnings(ticker, self._av_budget)
            except (SectionUnavailable, BudgetExhausted) as e:
                if yahoo_rows is None:
                    raise SectionUnavailable("earnings", self.name, f"yahoo and AV both failed: {e}") from e
                av_rows = []
            for r in av_rows:
                key = r.fiscal_end or r.report_date
                # Prefer yahoo for recency (it carries future/estimate-only prints
                # AV doesn't) but AV fills fiscal quarters yahoo's shorter history
                # dropped — merge by fiscal_end, yahoo wins on conflict.
                if key not in rows_by_fiscal:
                    rows_by_fiscal[key] = r

        out = sorted(
            rows_by_fiscal.values(),
            key=lambda r: r.report_date or date.min,
            reverse=True,
        )
        return out[:limit] if limit else out

    def estimate_snapshot(self, ticker: str) -> list[EstimateSnapshot]:
        # yahoo.estimate_snapshot already degrades to [] internally on its own
        # (documented) API-shape misses rather than raising — that is the
        # correct "genuinely nothing forward-facing available" case for this
        # provider, not a fetch failure, since Yahoo's estimate surface being
        # absent for a ticker is common and not itself an error.
        return yahoo.estimate_snapshot(ticker)

    # ── insider ──────────────────────────────────────────────────────────

    def insider_transactions(
        self, ticker: str, cik: int, since: date, max_filings: int = 40
    ) -> list[InsiderTxn]:
        return sec.form4_transactions(ticker, cik, since, max_filings=max_filings)

    # ── institutions (13F, read from the store — populated by the separate
    #    bulk-ingest script, scripts/harvest_13f_bulk.py) ────────────────

    def institutional_positions(
        self, ticker: str, cusip: str | None = None, quarters: int = 4
    ) -> list[InstitutionalPosition]:
        # DuckDB refuses to open a nonexistent file read-only — a fresh
        # checkout before the first 13F bulk ingest has no DB file at all,
        # which is exactly the "no dataset ingested yet" case, not a fetch bug.
        if not DB_PATH.exists():
            raise SectionUnavailable("institutions", self.name, "no 13F dataset ingested yet")

        with connect(read_only=True) as con:
            try:
                any_rows = con.execute(
                    "SELECT count(*) FROM fund_institution_holding"
                ).fetchone()
            except Exception as e:  # table exists (init_schema always runs) but a
                # read-only connect against a DB that was never written can still
                # surface duckdb errors on some versions — treat as "no dataset".
                raise SectionUnavailable("institutions", self.name, str(e)) from e

            if not any_rows or not any_rows[0]:
                raise SectionUnavailable(
                    "institutions", self.name, "no 13F dataset ingested yet"
                )

            where = "ticker = ?"
            params: list = [ticker.upper()]
            if cusip:
                where = "(ticker = ? OR cusip = ?)"
                params = [ticker.upper(), cusip]

            rows = con.execute(
                f"""
                SELECT period_end, filer_cik, cusip, ticker, filer_name, filing_date,
                       shares, value_usd, is_amendment, source
                FROM fund_institution_holding
                WHERE {where}
                ORDER BY period_end DESC
                """,
                params,
            ).fetchall()

        if not rows:
            return []

        # quarters is enforced on the distinct set of period_end values present,
        # not a row-count limit (a period can have many filers).
        periods = sorted({r[0] for r in rows}, reverse=True)[:quarters]
        period_set = set(periods)

        out: list[InstitutionalPosition] = []
        for period_end, filer_cik, cusip_v, tick, filer_name, filing_date, shares, value_usd, is_amendment, source in rows:
            if period_end not in period_set:
                continue
            out.append(InstitutionalPosition(
                source=source or "sec_13f_bulk",
                published_ts=datetime.combine(filing_date, datetime.min.time()) if filing_date else None,
                ticker=tick,
                period_end=period_end,
                filer_cik=filer_cik or 0,
                filer_name=filer_name or "",
                filing_date=filing_date,
                cusip=cusip_v or "",
                shares=shares,
                value_usd=value_usd,
                is_amendment=bool(is_amendment),
            ))
        return out

    # ── short interest / short volume ───────────────────────────────────

    def short_interest(self, ticker: str, since: date) -> list[ShortInterestPoint]:
        rows = nasdaq.short_interest(ticker)
        return [r for r in rows if r.settlement_date is None or r.settlement_date >= since]

    def short_volume(self, ticker: str, since: date) -> list[ShortVolumePoint]:
        # NOTE: finra.short_volume fetches ONE day-file for the ENTIRE market
        # per call — calling it per-ticker here (as this single-ticker method
        # must, to satisfy the ABC) re-downloads that same day-file once per
        # ticker in a naive loop. The harvester (scripts/harvest_fundamentals.py)
        # batches by calling finra.short_volume(all_tickers, days) directly and
        # writing per-ticker results itself; this method exists for the
        # single-ticker panel/live path and pays the redundant-fetch cost
        # (mitigated by httpcache's IMMUTABLE day-file caching after the first
        # ticker warms each day's cache entry).
        days = max((date.today() - since).days, 1)
        result = finra.short_volume({ticker.upper()}, days=days)
        return result.get(ticker.upper(), [])

    # ── borrow (read-only: the IB snapshot file, never refetched from IB) ──

    def borrow(self, ticker: str) -> list[BorrowPoint]:
        files = sorted(glob.glob(IB_LOCATE_GLOB))
        if not files:
            raise SectionUnavailable(
                "borrow", self.name,
                "no data/research/gapper/ib_locate_*.json snapshot found — "
                "run scripts/ib_shortable_snapshot.py",
            )
        latest = files[-1]  # filenames are ib_locate_YYYY-MM-DD.json, so lexical == chronological
        m = re.search(r"ib_locate_(\d{4}-\d{2}-\d{2})\.json$", latest)
        snap_date = datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None

        try:
            payload = json.loads(Path(latest).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SectionUnavailable("borrow", self.name, f"{latest}: {e}") from e

        detail = payload.get("detail") or {}
        rec = detail.get(ticker.upper())
        if rec is None:
            # NOT_LISTED tier — the ticker is genuinely absent from IB's file,
            # not a fetch failure. See ib_shortable_snapshot.py's docstring.
            return []

        return [BorrowPoint(
            source="ib_shortable_snapshot",
            published_ts=datetime.combine(snap_date, datetime.min.time()) if snap_date else None,
            ticker=ticker.upper(),
            date=snap_date,
            tier=rec.get("tier", ""),
            available_shares=rec.get("available"),
            fee_rate=rec.get("fee_pct"),
        )]
