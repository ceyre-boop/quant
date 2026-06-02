"""
Venue router — maps a trading symbol to its execution venue.

Forex (EUR_USD / EURUSD=X / …) → OANDA.  Index-futures (MNQ/MES/ES/NQ) → Tradovate.

There was no venue abstraction before (bridges were instantiated directly at
decision_chain.py:108 and forex_live_scan.py:173) — this is the thin seam a second venue
plugs into. Forex continues to use OandaBridge directly and unchanged; the futures path uses
this router. **ES/NQ stays DRY-RUN by default** (`is_dry_run_default` → True) until P3
produces a validated edge AND there's an explicit go.
"""
from __future__ import annotations

FUTURES_SYMBOLS = {"MNQ", "MES", "MYM", "M2K", "NQ", "ES", "YM", "RTY",
                   "NQ=F", "ES=F", "YM=F", "RTY=F"}


def venue_for(symbol: str) -> str:
    """Return 'TRADOVATE' for index futures, else 'OANDA'."""
    s = (symbol or "").upper()
    if s in FUTURES_SYMBOLS or s.startswith(("MNQ", "MES", "MYM", "M2K")):
        return "TRADOVATE"
    return "OANDA"


def is_dry_run_default(symbol: str) -> bool:
    """Futures are DRY-RUN by default — no live ES/NQ until a validated edge + explicit go."""
    return venue_for(symbol) == "TRADOVATE"


def get_bridge(symbol: str):
    """Lazily construct the bridge for `symbol`. Returns (bridge, venue_name).

    Lazy import so a missing Tradovate/OANDA credential only errors when that venue is
    actually requested — not at module import.
    """
    venue = venue_for(symbol)
    if venue == "TRADOVATE":
        from sovereign.execution.tradovate_bridge import TradovateBridge
        return TradovateBridge(), venue
    from sovereign.execution.oanda_bridge import OandaBridge
    return OandaBridge(), venue
