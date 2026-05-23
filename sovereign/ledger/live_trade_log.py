"""
Unified live trade log.
Every system (ICT, forex, TradingView webhooks) calls LiveTradeLog.log(event).
Events land in data/ledger/live_trade_log.jsonl — one JSON object per line.

Event schema:
  ts          ISO-8601 UTC
  type        ENTRY | TP1 | TP2 | STOP | SESSION_CLOSE | TV_ALERT | EXIT
  source      ICT | FOREX | TRADINGVIEW
  ticker      e.g. GBPUSD, EURUSD, SPY
  direction   LONG | SHORT | FLAT
  price       float
  r_value     float | None  (realized R for close events)
  grade       str | None    (ICT grade if applicable)
  session     str | None    (LONDON | NY_AM | NY_PM)
  meta        dict          (anything extra — stop, tp1, tp2, strategy name, etc.)
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/ledger/live_trade_log.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveTradeLog:
    def log(
        self,
        event_type: str,
        source: str,
        ticker: str,
        direction: str,
        price: float,
        r_value: float | None = None,
        grade: str | None = None,
        session: str | None = None,
        meta: dict | None = None,
        ts: str | None = None,
    ) -> dict:
        entry = {
            "ts": ts or _now_iso(),
            "type": event_type,
            "source": source,
            "ticker": ticker,
            "direction": direction,
            "price": float(price),
            "r_value": r_value,
            "grade": grade,
            "session": session,
            "meta": meta or {},
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    # ------------------------------------------------------------------
    # convenience wrappers
    # ------------------------------------------------------------------

    def entry(self, source: str, ticker: str, direction: str, price: float, **kw) -> dict:
        return self.log("ENTRY", source, ticker, direction, price, **kw)

    def tp1(self, source: str, ticker: str, direction: str, price: float, r_value: float, **kw) -> dict:
        return self.log("TP1", source, ticker, direction, price, r_value=r_value, **kw)

    def tp2(self, source: str, ticker: str, direction: str, price: float, r_value: float, **kw) -> dict:
        return self.log("TP2", source, ticker, direction, price, r_value=r_value, **kw)

    def stop(self, source: str, ticker: str, direction: str, price: float, r_value: float = -1.0, **kw) -> dict:
        return self.log("STOP", source, ticker, direction, price, r_value=r_value, **kw)

    def session_close(self, source: str, ticker: str, direction: str, price: float, r_value: float, **kw) -> dict:
        return self.log("SESSION_CLOSE", source, ticker, direction, price, r_value=r_value, **kw)

    def tv_alert(self, ticker: str, direction: str, price: float, strategy: str = "", **kw) -> dict:
        return self.log("TV_ALERT", "TRADINGVIEW", ticker, direction, price,
                        meta={"strategy": strategy, **kw.pop("meta", {})}, **kw)

    # ------------------------------------------------------------------
    # read helpers
    # ------------------------------------------------------------------

    @staticmethod
    def read(n: int = 200) -> list[dict]:
        if not LOG_PATH.exists():
            return []
        lines = LOG_PATH.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    @staticmethod
    def open_positions() -> list[dict]:
        """Return tickers that have an ENTRY without a matching close event."""
        events = LiveTradeLog.read(500)
        open_map: dict[str, dict] = {}
        close_types = {"TP1", "TP2", "STOP", "SESSION_CLOSE", "EXIT"}
        for e in events:
            key = e["ticker"]
            if e["type"] == "ENTRY":
                open_map[key] = e
            elif e["type"] in close_types and key in open_map:
                del open_map[key]
        return list(open_map.values())
