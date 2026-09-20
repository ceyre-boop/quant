"""
sovereign/terminal/sources.py
=============================
Every data source the terminal can show, behind one shape.

The failure mode this module exists to prevent: a dashboard that renders a
nine-day-old number in the same font as a live one. Every reader returns a
`Source` carrying its own age and status, and the renderers are required to
print that status next to the value. A source that cannot be read says so; it
never returns a plausible-looking zero.

Status ladder:
    LIVE    — fetched during this call (broker, quotes)
    FRESH   — file on disk, under 24h old
    STALE   — 24h to 7d old
    DEAD    — older than 7d; the loop that writes it has stopped
    MISSING — the file does not exist
    ERROR   — the read or the fetch raised; `note` says what
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]

PATHS = {
    "briefing":   REPO / "data" / "oracle" / "market_briefings" / "latest.json",
    "proximity":  REPO / "data" / "agent" / "forex_proximity.json",
    "loop":       REPO / "data" / "oracle" / "loop_health_status.json",
    "proof":      REPO / "data" / "agent" / "proof_of_life.json",
    "fred":       REPO / "data" / "macro" / "fred_economic_latest.json",
    "hyp_ledger": REPO / "data" / "agent" / "hypothesis_ledger.json",
    "kill":       REPO / "data" / "system" / "KILL_SWITCH",
    "env":        REPO / ".env",
    "risk_const": REPO / "RISK_CONSTITUTION.md",
}

FRESH_H, STALE_H = 24.0, 24.0 * 7


@dataclass
class Source:
    """One reading, with the truth about how old it is attached."""
    name: str
    status: str                       # LIVE | FRESH | STALE | DEAD | MISSING | ERROR
    data: Any = None
    asof: Optional[datetime] = None
    note: str = ""

    @property
    def age_hours(self) -> Optional[float]:
        if self.asof is None:
            return None
        return (datetime.now(timezone.utc) - self.asof).total_seconds() / 3600.0

    @property
    def age_label(self) -> str:
        h = self.age_hours
        if self.status == "LIVE":
            return "live"
        if h is None:
            return "—"
        if h < 1:
            return f"{int(h * 60)}m"
        if h < 48:
            return f"{h:.0f}h"
        return f"{h / 24:.0f}d"

    @property
    def ok(self) -> bool:
        return self.status in ("LIVE", "FRESH", "STALE")

    @property
    def trustworthy(self) -> bool:
        """STALE is readable; DEAD is not something to trade on."""
        return self.status in ("LIVE", "FRESH")


def _rel(path: Path) -> str:
    """Repo-relative path for messages, safe for paths outside the repo."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _classify(path: Path) -> tuple[str, Optional[datetime]]:
    if not path.exists():
        return "MISSING", None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if age < FRESH_H:
        return "FRESH", ts
    if age < STALE_H:
        return "STALE", ts
    return "DEAD", ts


def read_json(name: str) -> Source:
    """A JSON file on disk, with its mtime as the age."""
    path = PATHS[name]
    status, ts = _classify(path)
    if status == "MISSING":
        return Source(name, "MISSING", note=f"{_rel(path)} does not exist")
    try:
        return Source(name, status, json.loads(path.read_text(encoding="utf-8")), ts)
    except Exception as exc:
        return Source(name, "ERROR", asof=ts, note=f"{type(exc).__name__}: {exc}")


# ── credentials ───────────────────────────────────────────────────────────────

def _env() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in PATHS["env"].read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#") and "=" in t:
                k, v = t.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


# ── live broker ───────────────────────────────────────────────────────────────

def account() -> Source:
    """Live OANDA account + open positions. Read-only GET; cannot trade."""
    env = _env()
    key, acct = env.get("OANDA_API_KEY"), env.get("OANDA_ACCOUNT_ID")
    if not key or not acct:
        return Source("account", "ERROR", note="OANDA_API_KEY / OANDA_ACCOUNT_ID missing from .env")
    base = env.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com").rstrip("/")
    live = env.get("OANDA_LIVE") == "1"

    def _get(suffix: str) -> Any:
        req = urllib.request.Request(f"{base}/v3/accounts/{acct}{suffix}",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    try:
        summary = _get("/summary")["account"]
        positions = _get("/openPositions").get("positions", [])
        trades = _get("/openTrades").get("trades", [])
    except urllib.error.HTTPError as exc:
        return Source("account", "ERROR", note=f"OANDA HTTP {exc.code} {exc.reason}")
    except Exception as exc:
        return Source("account", "ERROR", note=f"{type(exc).__name__}: {exc}")

    def _f(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    return Source("account", "LIVE", {
        "mode": "LIVE" if live else "practice",
        "currency": summary.get("currency", "USD"),
        "nav": _f(summary.get("NAV")),
        "balance": _f(summary.get("balance")),
        "unrealized": _f(summary.get("unrealizedPL")),
        "realized_lifetime": _f(summary.get("pl")),
        "margin_used": _f(summary.get("marginUsed")),
        "margin_available": _f(summary.get("marginAvailable")),
        "open_trades": int(_f(summary.get("openTradeCount"))),
        "positions": [
            {
                "instrument": p.get("instrument"),
                "long_units": _f(p.get("long", {}).get("units")),
                "short_units": _f(p.get("short", {}).get("units")),
                "unrealized": _f(p.get("unrealizedPL")),
            }
            for p in positions
        ],
        "trades": [
            {
                "id": t.get("id"),
                "instrument": t.get("instrument"),
                "units": _f(t.get("currentUnits")),
                "price": _f(t.get("price")),
                "unrealized": _f(t.get("unrealizedPL")),
                "opened": (t.get("openTime") or "")[:19],
            }
            for t in trades
        ],
    }, datetime.now(timezone.utc))


def quote(instrument: str) -> Source:
    """Live bid/ask for one instrument from the broker's pricing endpoint."""
    env = _env()
    key, acct = env.get("OANDA_API_KEY"), env.get("OANDA_ACCOUNT_ID")
    if not key or not acct:
        return Source("quote", "ERROR", note="OANDA credentials missing")
    base = env.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com").rstrip("/")
    oanda_sym = _oanda_symbol(instrument)
    url = f"{base}/v3/accounts/{acct}/pricing?instruments={oanda_sym}"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            prices = json.load(r).get("prices", [])
        if not prices:
            return Source("quote", "ERROR", note=f"{oanda_sym}: no price returned")
        p = prices[0]
        bid = float(p["bids"][0]["price"]) if p.get("bids") else None
        ask = float(p["asks"][0]["price"]) if p.get("asks") else None
        mid = (bid + ask) / 2 if bid and ask else None
        pip = 0.01 if "JPY" in oanda_sym else 0.0001
        return Source("quote", "LIVE", {
            "instrument": oanda_sym,
            "bid": bid, "ask": ask, "mid": mid,
            "spread_pips": round((ask - bid) / pip, 1) if bid and ask else None,
            "tradeable": p.get("tradeable", False),
            "time": (p.get("time") or "")[:19],
        }, datetime.now(timezone.utc))
    except urllib.error.HTTPError as exc:
        return Source("quote", "ERROR", note=f"OANDA HTTP {exc.code} for {oanda_sym}")
    except Exception as exc:
        return Source("quote", "ERROR", note=f"{type(exc).__name__}: {exc}")


def _oanda_symbol(instrument: str) -> str:
    s = "".join(ch for ch in instrument.upper() if ch.isalpha())
    return f"{s[:3]}_{s[3:]}" if len(s) == 6 else s


# ── the written record (delegates to the context assembler) ──────────────────

def trade_context(instrument: str, offline: bool = True,
                  with_library: bool = True) -> Source:
    """The full research packet for one instrument. See sovereign.context."""
    try:
        from sovereign.context.trade_context import build_packet
        packet = build_packet(instrument, offline=offline, with_library=with_library)
        return Source("context", "LIVE", packet.to_dict(), datetime.now(timezone.utc),
                      note=packet.completeness)
    except Exception as exc:
        return Source("context", "ERROR", note=f"{type(exc).__name__}: {exc}")


# ── risk budget ──────────────────────────────────────────────────────────────

@dataclass
class RiskBudget:
    """What the constitution permits right now, in this account's currency.

    Caps are parsed from RISK_CONSTITUTION.md rather than hardcoded — the file is
    the ratified source and the numbers have already been re-anchored once.
    """
    nav: float
    currency: str
    per_trade_pct: float
    carry_heat_pct: float
    ladder_pct: tuple[float, float, float]
    parsed_ok: bool
    note: str = ""

    @property
    def per_trade(self) -> float:
        return self.nav * self.per_trade_pct / 100.0

    @property
    def carry_heat(self) -> float:
        return self.nav * self.carry_heat_pct / 100.0

    def ladder_levels(self) -> list[tuple[str, float, float]]:
        names = ("halve size", "halt new entries", "flatten predictive")
        return [(n, p, self.nav * (1 - p / 100.0))
                for n, p in zip(names, self.ladder_pct)]


def risk_budget(nav: float, currency: str = "USD") -> RiskBudget:
    import re
    defaults = (0.75, 2.5, (3.5, 5.0, 6.5))
    try:
        text = PATHS["risk_const"].read_text(encoding="utf-8")
    except Exception as exc:
        return RiskBudget(nav, currency, *defaults[:2], defaults[2], False,
                          f"RISK_CONSTITUTION.md unreadable ({type(exc).__name__}) — showing draft defaults")

    def _pct(pattern: str, fallback: float) -> tuple[float, bool]:
        m = re.search(pattern, text, re.I | re.S)
        if not m:
            return fallback, False
        try:
            return float(m.group(1)), True
        except ValueError:
            return fallback, False

    per_trade, ok1 = _pct(r"No single trade may risk more than \*\*([\d.]+)%\*\*", defaults[0])
    heat, ok2 = _pct(r"carry positions may not exceed\s*\*\*([\d.]+)%\*\*", defaults[1])
    m = re.search(r"drawdown of \*\*([\d.]+)%\*\*.*?At \*\*([\d.]+)%\*\*.*?At \*\*([\d.]+)%\*\*",
                  text, re.I | re.S)
    ladder = tuple(float(g) for g in m.groups()) if m else defaults[2]  # type: ignore[assignment]
    parsed = ok1 and ok2 and m is not None
    note = "" if parsed else "some caps fell back to draft values — check RISK_CONSTITUTION.md wording"
    return RiskBudget(nav, currency, per_trade, heat, ladder, parsed, note)  # type: ignore[arg-type]


# ── kill switch ──────────────────────────────────────────────────────────────

def kill_switch() -> Source:
    p = PATHS["kill"]
    if p.exists():
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        try:
            body = p.read_text(encoding="utf-8").strip()[:200]
        except Exception:
            body = ""
        return Source("kill", "LIVE", {"frozen": True, "reason": body}, ts)
    return Source("kill", "LIVE", {"frozen": False, "reason": ""}, datetime.now(timezone.utc))


# ── loop health, computed honestly from file mtimes ──────────────────────────

def loop_health() -> Source:
    """Which writers are actually still writing.

    The recorded loop_health_status.json is itself one of the loops, so it can be
    stale about its own staleness. This recomputes from the mtimes of the files
    each loop produces — a loop is alive iff its output moved recently.
    """
    watched = {
        "market briefing":   PATHS["briefing"],
        "forex proximity":   PATHS["proximity"],
        "proof of life":     PATHS["proof"],
        "FRED macro":        PATHS["fred"],
        "hypothesis ledger": PATHS["hyp_ledger"],
        "loop health file":  PATHS["loop"],
    }
    rows = []
    for label, path in watched.items():
        status, ts = _classify(path)
        rows.append({"loop": label, "status": status,
                     "asof": ts.isoformat(timespec="seconds") if ts else None,
                     "age_h": None if ts is None else
                              (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0,
                     "path": _rel(path)})
    dead = sum(r["status"] in ("DEAD", "MISSING") for r in rows)
    return Source("loop_health", "LIVE",
                  {"rows": rows, "n_dead": dead, "n_total": len(rows)},
                  datetime.now(timezone.utc))
