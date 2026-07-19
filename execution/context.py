"""Layer 1 — the consolidated morning context.

WHY THIS EXISTS
---------------
Before this module there was no pre-open orchestrator. Two chains ran independently
and never exchanged data structurally:

  Chain A  02:30 ET  `com.alta.oracle.reflect` -> oracle_cycle.py (FRED, daily panel)
  Chain B  08:15 ET  `com.alta.oracle.briefing` -> oracle_session_open.py (briefing)

Downstream components each reached into whatever files they happened to know about.
This builds the single object every consumer can read, with honest provenance.

THE LOAD-BEARING RULE: DEGRADE, NEVER FABRICATE
------------------------------------------------
The worst live data bug in this repo is not a crash — it is a source that reports
success while returning nothing. `data/cache/reddit_sentiment.json` carries a
timestamp minutes old and `"posts_scanned": 0`, and `harvest_daily_panel.py`
records that emptiness downstream *as data*. A freshness check alone calls it
healthy. A consumer cannot distinguish "no signal today" from "this source has
been dead for weeks".

So every field carries an explicit status:

  FRESH        real data, within its staleness budget
  STALE        real data, but older than its budget (value kept, flagged)
  SILENT_NULL  the source ran and succeeded but returned no content  <-- the trap
  UNAVAILABLE  the source could not be reached at all (403, missing file, 0 rows)
  ERROR        the source raised

Nothing is ever defaulted to zero. A missing macro reading is `None` with a
status, never `0.0`, because `0.0` is a number a model will happily trade on.

Known degraded sources at time of writing (2026-07-18), all correctly surfaced
rather than hidden: sentiment board 12 days stale (its plist was authored but
never installed), GDELT 0 rows ever ingested, Reddit silent-null, ForexFactory
403, VRP feed a month old.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sovereign.utils.timestamps import canonical_timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "context"
UTC = timezone.utc


class Status(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    SILENT_NULL = "SILENT_NULL"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


#: Per-source staleness budget in hours. Beyond this a value is kept but flagged
#: STALE. Chosen from each source's own cadence, not a blanket number.
BUDGET_HOURS = {
    "fred_macro": 30,          # daily 02:30 cycle
    "sentiment_board": 30,     # daily 07:45 (currently uninstalled)
    "briefing": 30,            # daily 08:15
    "daily_panel": 30,         # daily 02:30
    "reddit": 6,               # 2-hourly cache refresh
    "vix": 30,
    "calendar": 30,
}


@dataclass
class Field_:
    """One context field with full provenance."""
    name: str
    value: Any
    status: Status
    source: str
    age_seconds: float | None = None
    detail: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @property
    def usable(self) -> bool:
        """True only for FRESH. STALE is readable but must be an explicit choice."""
        return self.status is Status.FRESH


def _age(path: Path) -> float:
    return (datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)).total_seconds()


def _status_for(name: str, age_s: float | None) -> Status:
    if age_s is None:
        return Status.FRESH
    budget = BUDGET_HOURS.get(name, 30) * 3600
    return Status.FRESH if age_s <= budget else Status.STALE


def _missing(name: str, source: str, detail: str) -> Field_:
    return Field_(name=name, value=None, status=Status.UNAVAILABLE,
                  source=source, detail=detail)


def _error(name: str, source: str, exc: BaseException) -> Field_:
    return Field_(name=name, value=None, status=Status.ERROR, source=source,
                  detail=f"{type(exc).__name__}: {exc}")


# ── Sources ───────────────────────────────────────────────────────────────────

def load_fred() -> Field_:
    p = ROOT / "data" / "macro" / "fred_economic_latest.json"
    if not p.exists():
        return _missing("fred_macro", str(p), "file absent")
    try:
        d = json.loads(p.read_text())
        metrics = d.get("metrics") or {}
        if not metrics:
            return Field_("fred_macro", None, Status.SILENT_NULL, str(p),
                          _age(p), "file present but metrics empty")
        return Field_("fred_macro", {"metrics": metrics, "summary": d.get("summary"),
                                     "date": d.get("date")},
                      _status_for("fred_macro", _age(p)), str(p), _age(p),
                      f"{len(metrics)} series")
    except Exception as e:                                   # noqa: BLE001
        return _error("fred_macro", str(p), e)


def load_sentiment_board() -> Field_:
    """Fused sentiment board from DuckDB.

    Currently 12 days stale because `com.alta.sentiment_update.plist` was authored
    for 07:45 "so the board is fresh for the 08:00 scan" and never installed. That
    shows up here as STALE with a real age, not as silence.
    """
    p = ROOT / "data" / "sentiment.db"
    if not p.exists():
        return _missing("sentiment_board", str(p), "sentiment.db absent")
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        n, maxd = con.execute(
            "select count(*), max(date) from sentiment_board_state").fetchone()
        con.close()
        if not n:
            return Field_("sentiment_board", None, Status.SILENT_NULL, str(p),
                          _age(p), "board table empty")

        # FRESHNESS IS REBUILD TIME, NOT MAX DATA DATE.
        # The board's latest row is bounded by the MARKET CALENDAR, not by pipeline
        # health: a Tuesday rebuild yields Monday's date (~32h old at 08:00) and a
        # Monday rebuild yields Friday's (~56h). Judging staleness by data date
        # would therefore report STALE on every healthy run — a permanent false
        # alarm. What matters is whether the pipeline ran, so age comes from the
        # DB mtime; data coverage is reported alongside as a separate fact.
        rebuild_age = _age(p)
        detail = f"rebuilt {rebuild_age/3600:.1f}h ago, data through {maxd}"

        # Data coverage IS still checked — a board that rebuilds daily but stops
        # advancing its data is a real failure, just a different one.
        stale_data = False
        if maxd:
            data_age_days = (datetime.now(UTC).date() - maxd).days
            if data_age_days > 5:          # beyond a long weekend + a holiday
                stale_data = True
                detail += f" — DATA STALLED {data_age_days}d despite rebuilds"

        status = _status_for("sentiment_board", rebuild_age)
        if stale_data:
            status = Status.STALE
        return Field_("sentiment_board",
                      {"rows": n, "latest_date": str(maxd),
                       "rebuild_age_hours": round(rebuild_age / 3600, 2)},
                      status, str(p), rebuild_age, detail)
    except Exception as e:                                   # noqa: BLE001
        return _error("sentiment_board", str(p), e)


def load_gdelt() -> Field_:
    """GDELT tone. Has never ingested a row; its retry plist is also uninstalled."""
    p = ROOT / "data" / "sentiment.db"
    if not p.exists():
        return _missing("gdelt", str(p), "sentiment.db absent")
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        (n,) = con.execute("select count(*) from sentiment_gdelt_daily").fetchone()
        con.close()
        if not n:
            return Field_("gdelt", None, Status.UNAVAILABLE, str(p), None,
                          "0 rows — never ingested (burst-throttled free tier)")
        return Field_("gdelt", {"rows": n}, Status.FRESH, str(p), None, f"{n} rows")
    except Exception as e:                                   # noqa: BLE001
        return _error("gdelt", str(p), e)


def load_reddit() -> Field_:
    """Reddit sentiment cache — the canonical SILENT_NULL.

    Exits clean, stamps a current `last_updated`, and returns
    `posts_scanned: 0, equity: {}`. Freshness looks perfect; content is empty.
    """
    p = ROOT / "data" / "cache" / "reddit_sentiment.json"
    if not p.exists():
        return _missing("reddit", str(p), "cache absent")
    try:
        d = json.loads(p.read_text())
        scanned = int(d.get("posts_scanned", 0) or 0)
        if scanned == 0:
            return Field_("reddit", None, Status.SILENT_NULL, str(p), _age(p),
                          "job succeeded but posts_scanned=0 (credentials likely absent)")
        return Field_("reddit", {"posts_scanned": scanned,
                                 "equity": d.get("equity"), "forex": d.get("forex")},
                      _status_for("reddit", _age(p)), str(p), _age(p),
                      f"{scanned} posts")
    except Exception as e:                                   # noqa: BLE001
        return _error("reddit", str(p), e)


def load_briefing() -> Field_:
    p = ROOT / "data" / "oracle" / "market_briefings" / "latest.json"
    if not p.exists():
        return _missing("briefing", str(p), "no briefing written")
    try:
        d = json.loads(p.read_text())
        return Field_("briefing", {
            "date": d.get("date"), "regime_read": d.get("regime_read"),
            "narrative": d.get("narrative"),
            "directional_bias": d.get("directional_bias"),
            "confidence": d.get("confidence"),
            "news_count": d.get("news_count"),
            "event_calendar": d.get("event_calendar"),
        }, _status_for("briefing", _age(p)), str(p), _age(p),
            f"briefing for {d.get('date')}")
    except Exception as e:                                   # noqa: BLE001
        return _error("briefing", str(p), e)


def load_daily_panel(day: date) -> Field_:
    p = ROOT / "data" / "research" / "panel" / f"{day}.json"
    if not p.exists():
        return _missing("daily_panel", str(p), f"no panel for {day}")
    try:
        d = json.loads(p.read_text())
        return Field_("daily_panel", {"keys": sorted(d)[:20]},
                      _status_for("daily_panel", _age(p)), str(p), _age(p),
                      f"{len(d)} sections")
    except Exception as e:                                   # noqa: BLE001
        return _error("daily_panel", str(p), e)


def load_calendar() -> Field_:
    """ForexFactory economic calendar — currently 403 Forbidden."""
    p = ROOT / "data" / "cache" / "forex_factory_calendar.json"
    if not p.exists():
        return _missing("calendar", str(p),
                        "ForexFactory 403 Forbidden — no cached calendar")
    try:
        d = json.loads(p.read_text())
        events = d.get("events") or []
        if not events:
            return Field_("calendar", None, Status.SILENT_NULL, str(p), _age(p),
                          "cache present but 0 events")
        return Field_("calendar", {"events": len(events)},
                      _status_for("calendar", _age(p)), str(p), _age(p),
                      f"{len(events)} events")
    except Exception as e:                                   # noqa: BLE001
        return _error("calendar", str(p), e)


SOURCES = ["fred_macro", "sentiment_board", "gdelt", "reddit", "briefing",
           "daily_panel", "calendar"]


def build_morning_context(day: date | None = None) -> dict:
    """Assemble the single morning context object."""
    day = day or datetime.now(UTC).date()
    fields = [
        load_fred(), load_sentiment_board(), load_gdelt(), load_reddit(),
        load_briefing(), load_daily_panel(day), load_calendar(),
    ]
    by_status: dict[str, list[str]] = {}
    for f in fields:
        by_status.setdefault(f.status.value, []).append(f.name)

    healthy = sum(1 for f in fields if f.status is Status.FRESH)
    return {
        "date": str(day),
        "generated_at": canonical_timestamp(),
        "fields": {f.name: f.to_json() for f in fields},
        "health": {
            "n_sources": len(fields),
            "n_fresh": healthy,
            "fraction_fresh": round(healthy / len(fields), 3),
            "by_status": by_status,
        },
        "note": ("Every field carries an explicit status. SILENT_NULL means the "
                 "source succeeded but returned nothing — treat it as absent data, "
                 "never as a zero reading."),
    }


def write_context(ctx: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"morning_context_{ctx['date']}.json"
    p.write_text(json.dumps(ctx, indent=2, default=str))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the consolidated morning context")
    ap.add_argument("--day", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print", action="store_true", help="print the health table")
    args = ap.parse_args(argv)

    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()
    ctx = build_morning_context(day)
    p = write_context(ctx, Path(args.out) if args.out else None)

    h = ctx["health"]
    print(f"morning context {ctx['date']}: {h['n_fresh']}/{h['n_sources']} sources FRESH "
          f"({h['fraction_fresh']:.0%})")
    for name, f in ctx["fields"].items():
        age = f.get("age_seconds")
        age_s = f"{age/3600:6.1f}h" if age else "     —"
        print(f"  {f['status']:<12} {name:<18} {age_s}  {f['detail']}")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
