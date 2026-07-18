"""Real SIP bid/ask capture and fill accounting.

WHAT THIS REPLACES
------------------
`research/gapper/hyp107_shadow.py:148` computed::

    realized_spread = (b0931["h"] - b0931["l"]) / entry

and logged it as `realized_spread_pct`. That is the first-minute bar RANGE, not
a quoted spread. On a gapper's opening minute the range routinely runs 5-20%
while the actual quoted spread is well under 1%. The proxy was wrong by an order
of magnitude, and wrong in the PESSIMISTIC direction — which is a large part of
why HYP-107 was left at "plausibly unviable after costs".

Measured counter-example (2026-07-16, TGHL, 09:30:59 ET, SIP):
    bid 1.37 (size 6,700)   ask 1.38 (size 9,300)   ->  spread 0.73%
against a documented assumption of 1-15%.

FILL CONVENTION (applied everywhere, no exceptions)
---------------------------------------------------
    LONG  entry at ASK, exit at BID
    SHORT entry at BID, exit at ASK

`spread_cost` is the round trip — entry half-spread plus exit half-spread — as a
fraction of the entry mid, so it is directly comparable to
`realistic_fills._half_spread()` semantics (which also charges round-trip).

The identity `gross_return - net_return ~= spread_cost` holds by construction and
is asserted to 1bp in tests. If it ever fails, the accounting is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from execution import alpaca
from execution.config import frozen

UTC = timezone.utc


@dataclass(frozen=True)
class Quote:
    """One NBBO snapshot."""
    symbol: str
    ts_utc: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    bid_exchange: str = ""
    ask_exchange: str = ""
    conditions: list[str] = field(default_factory=list)
    tape: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_abs(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m if m > 0 else 0.0

    def is_crossed(self) -> bool:
        """Bid above ask — a broken/stale book. Never fill against it."""
        return self.bid > self.ask

    def is_locked(self) -> bool:
        """Bid equals ask — momentarily locked market."""
        return self.bid == self.ask

    def is_usable(self) -> bool:
        return (self.bid > 0 and self.ask > 0
                and not self.is_crossed() and not self.is_locked())

    def age_seconds(self, at: datetime) -> float:
        ts = datetime.fromisoformat(self.ts_utc.replace("Z", "+00:00"))
        return (at - ts).total_seconds()

    def to_record(self) -> dict[str, Any]:
        return {
            "ts": self.ts_utc, "bid": self.bid, "ask": self.ask,
            "bid_size": self.bid_size, "ask_size": self.ask_size,
            "bid_exchange": self.bid_exchange, "ask_exchange": self.ask_exchange,
            "conditions": list(self.conditions), "tape": self.tape,
        }


def _from_api(symbol: str, q: dict) -> Quote:
    return Quote(
        symbol=symbol,
        ts_utc=q["t"],
        bid=float(q.get("bp", 0.0)),
        ask=float(q.get("ap", 0.0)),
        bid_size=int(q.get("bs", 0)),
        ask_size=int(q.get("as", 0)),
        bid_exchange=str(q.get("bx", "")),
        ask_exchange=str(q.get("ax", "")),
        conditions=list(q.get("c", []) or []),
        tape=str(q.get("z", "")),
    )


def quote_at(symbol: str, ts_utc: datetime, *,
             window_seconds: int | None = None) -> Quote | None:
    """Last usable quote at or before `ts_utc`.

    Searches back `window_seconds` and takes the newest usable quote in that
    window. Returns None if no usable quote exists — the caller records
    SKIP_NO_QUOTE rather than inventing a price.

    Requires `ts_utc` to be outside the SIP recency window; otherwise Alpaca
    raises AlpacaEntitlementError, which is allowed to propagate because a
    silently-missing quote would corrupt the measurement.
    """
    cap = frozen("capture")
    window = window_seconds or cap["quote_window_seconds"]
    start = ts_utc - timedelta(seconds=window)
    rows = alpaca.raw_quotes(symbol, start, ts_utc)
    if not rows:
        return None

    for raw in reversed(rows):          # newest first
        q = _from_api(symbol, raw)
        if q.is_usable():
            return q
    return None


def fill_price(q: Quote, side: str, leg: str) -> tuple[float, float]:
    """Return (fill_price, half_spread_fraction) for one leg of one side.

    side: 'LONG' | 'SHORT'      leg: 'ENTRY' | 'EXIT'

        LONG  ENTRY -> ask      LONG  EXIT -> bid
        SHORT ENTRY -> bid      SHORT EXIT -> ask

    The half-spread is expressed against this quote's own mid; the caller
    normalises both legs to the ENTRY mid when computing spread_cost.
    """
    side = side.upper()
    leg = leg.upper()
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side must be LONG or SHORT, got {side!r}")
    if leg not in ("ENTRY", "EXIT"):
        raise ValueError(f"leg must be ENTRY or EXIT, got {leg!r}")

    take_ask = (side == "LONG") == (leg == "ENTRY")
    px = q.ask if take_ask else q.bid
    half = abs(px - q.mid) / q.mid if q.mid > 0 else 0.0
    return float(px), float(half)


def round_trip(entry_q: Quote, exit_q: Quote, side: str) -> dict[str, float]:
    """Full fill accounting for one simulated trade.

    Returns gross_return (mid-to-mid), net_return (fill-to-fill), and
    spread_cost (round-trip, as a fraction of the entry mid).

    For a SHORT, returns are sign-flipped so a profitable fade is positive.
    """
    side = side.upper()
    entry_fill, _ = fill_price(entry_q, side, "ENTRY")
    exit_fill, _ = fill_price(exit_q, side, "EXIT")

    em, xm = entry_q.mid, exit_q.mid
    if em <= 0:
        raise ValueError(f"non-positive entry mid for {entry_q.symbol}")

    if side == "LONG":
        gross = xm / em - 1.0
        net = exit_fill / entry_fill - 1.0
    else:
        gross = 1.0 - xm / em
        net = 1.0 - exit_fill / entry_fill

    return {
        "entry_fill": round(entry_fill, 6),
        "exit_fill": round(exit_fill, 6),
        "entry_mid": round(em, 6),
        "exit_mid": round(xm, 6),
        "gross_return": round(gross, 6),
        "net_return": round(net, 6),
        "spread_cost": round(gross - net, 6),
        "entry_spread_pct": round(entry_q.spread_pct, 6),
        "exit_spread_pct": round(exit_q.spread_pct, 6),
    }


def is_wide(q: Quote) -> bool:
    """True if this quote's spread exceeds the wide-quote flag threshold.

    Wide quotes are FLAGGED, never dropped. Dropping them would silently bias
    the measured spread downward — precisely the failure this harness exists to
    detect.
    """
    return q.spread_pct > frozen("capture")["wide_quote_pct"]
