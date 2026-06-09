#!/usr/bin/env python3
"""
IB Gateway bridge for MES/MNQ paper trading.

Wraps ib-async (the maintained ib_insync fork) with a clean synchronous
interface for the futures sandbox. Defaults to IB Gateway PAPER port 4002.

Prerequisites:
  1. IB Gateway running locally (or TWS) with API enabled
  2. IB Gateway > Edit > Global Configuration > API > Enable ActiveX and Socket Clients = ON
  3. Socket port 4002 (paper) — change IB_PORT in .env to 4001 for live

Usage:
    bridge = IBBridge()
    bridge.connect()
    print(bridge.account_summary())
    bridge.disconnect()
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

# ── Config (from .env, with safe defaults) ──────────────────────────────────
def _env(key: str, default: str) -> str:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.environ.get(key, default)

IB_HOST      = lambda: _env("IB_HOST",      "127.0.0.1")
IB_PORT      = lambda: int(_env("IB_PORT",  "4002"))   # 4002=Gateway paper, 4001=Gateway live
IB_CLIENT_ID = lambda: int(_env("IB_CLIENT_ID", "10")) # must be unique per connection
IB_ACCOUNT   = lambda: _env("IB_ACCOUNT",   "")        # leave blank to auto-detect


@dataclass
class OrderResult:
    order_id: int
    status: str
    filled: float
    avg_fill_price: float
    remaining: float
    error: Optional[str] = None


class IBBridge:
    """
    Synchronous wrapper around ib-async for the futures sandbox.
    Paper trading only until the 150-trade analysis validates the approach.
    """

    def __init__(self):
        from ib_async import IB
        self._ib = IB()
        self._connected = False

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to IB Gateway. Raises if Gateway isn't running."""
        host, port, cid = IB_HOST(), IB_PORT(), IB_CLIENT_ID()
        logger.info(f"[IBBridge] Connecting to {host}:{port} (clientId={cid})")
        self._ib.connect(host, port, clientId=cid, readonly=False)
        self._connected = True
        acct = self._ib.managedAccounts()
        account = IB_ACCOUNT() or (acct[0] if acct else "UNKNOWN")
        self._account = account
        mode = "PAPER" if port in (4002, 7497) else "LIVE"
        print(f"[IBBridge] Connected | account={account} | mode={mode} | port={port}")

    def disconnect(self) -> None:
        if self._connected:
            self._ib.disconnect()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Contracts ────────────────────────────────────────────────────────────

    def mes_contract(self):
        """Front-month Micro E-mini S&P 500."""
        from ib_async import Future
        c = Future("MES", exchange="CME", currency="USD")
        details = self._ib.reqContractDetails(c)
        if not details:
            raise RuntimeError("No MES contract details returned — check Gateway connection")
        # Sort by expiry, take front month
        front = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)[0]
        return front.contract

    def mnq_contract(self):
        """Front-month Micro E-mini Nasdaq 100."""
        from ib_async import Future
        c = Future("MNQ", exchange="CME", currency="USD")
        details = self._ib.reqContractDetails(c)
        if not details:
            raise RuntimeError("No MNQ contract details returned")
        front = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)[0]
        return front.contract

    # ── Account ──────────────────────────────────────────────────────────────

    def account_summary(self) -> dict:
        """Net liquidation, cash, unrealized P&L."""
        from ib_async import AccountValue
        vals = self._ib.accountValues(self._account)
        wanted = {"NetLiquidation", "TotalCashValue", "UnrealizedPnL", "RealizedPnL"}
        out = {}
        for v in vals:
            if v.tag in wanted and v.currency == "USD":
                out[v.tag] = float(v.value)
        return out

    def positions(self) -> list[dict]:
        """Current open positions."""
        return [
            {
                "symbol":    p.contract.symbol,
                "expiry":    p.contract.lastTradeDateOrContractMonth,
                "position":  p.position,
                "avg_cost":  p.avgCost,
            }
            for p in self._ib.positions()
        ]

    def fills(self) -> list[dict]:
        """Today's executions — the authoritative paper-fill record (use to close the loop
        honestly instead of replaying yfinance). Each: symbol, side, shares, price, time,
        order_id, perm_id, commission."""
        out = []
        for f in self._ib.fills():
            e, c = f.execution, f.contract
            cr = getattr(f, "commissionReport", None)
            out.append({
                "symbol":     c.symbol,
                "side":       e.side,                       # BOT / SLD
                "shares":     float(e.shares),
                "price":      float(e.price),
                "time":       str(e.time),
                "order_id":   int(e.orderId),
                "perm_id":    int(e.permId),
                "commission": (float(cr.commission) if cr and cr.commission is not None else None),
            })
        return out

    # ── Orders ───────────────────────────────────────────────────────────────

    def market_order(self, contract, direction: str, quantity: int) -> OrderResult:
        """Place a market order. direction = 'BUY' | 'SELL'."""
        from ib_async import MarketOrder
        order = MarketOrder(direction.upper(), quantity)
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(1)  # give IB a moment to acknowledge
        return OrderResult(
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
            filled=trade.orderStatus.filled,
            avg_fill_price=trade.orderStatus.avgFillPrice,
            remaining=trade.orderStatus.remaining,
        )

    def bracket_order(
        self,
        contract,
        direction: str,
        quantity: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> list[OrderResult]:
        """
        Limit entry with OCO stop + target.
        Returns [entry_result, stop_result, target_result].
        direction = 'BUY' | 'SELL'
        """
        from ib_async import LimitOrder, StopOrder
        parent = LimitOrder(direction.upper(), quantity, entry_price)
        parent.transmit = False

        reverse = "SELL" if direction.upper() == "BUY" else "BUY"
        stop = StopOrder(reverse, quantity, stop_price)
        stop.parentId = parent.orderId
        stop.transmit = False

        target = LimitOrder(reverse, quantity, target_price)
        target.parentId = parent.orderId
        target.transmit = True   # transmit all three together

        results = []
        for order in (parent, stop, target):
            trade = self._ib.placeOrder(contract, order)
            results.append(OrderResult(
                order_id=trade.order.orderId,
                status=trade.orderStatus.status,
                filled=trade.orderStatus.filled,
                avg_fill_price=trade.orderStatus.avgFillPrice,
                remaining=trade.orderStatus.remaining,
            ))
        self._ib.sleep(1)
        return results

    def cancel_all_orders(self) -> None:
        """Cancel all open orders. Use before closing the session."""
        for trade in self._ib.openTrades():
            self._ib.cancelOrder(trade.order)
        self._ib.sleep(0.5)

    # ── Historical bars ────────────────────────────────────────────────────────

    def historical_bars(self, contract, duration: str = "1 D",
                        bar_size: str = "1 min", what: str = "TRADES",
                        rth: bool = True, end: str = ""):
        """Historical OHLCV as a DataFrame indexed by UTC datetime.

        Columns: Open, High, Low, Close, Volume. `end` is "" for now, or an IB
        endDateTime string ("YYYYMMDD HH:MM:SS") to pull a specific past session.
        Used by bar_feed.IBBarFeed for both live session bars and replay history.
        """
        from ib_async import util
        bars = self._ib.reqHistoricalData(
            contract, endDateTime=end, durationStr=duration,
            barSizeSetting=bar_size, whatToShow=what, useRTH=rth, formatDate=2,
        )
        df = util.df(bars)
        if df is None or df.empty:
            return df
        df = df.rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        }).set_index("Date")
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ── Live quote ───────────────────────────────────────────────────────────

    def last_price(self, contract) -> Optional[float]:
        """Snapshot last price for a contract."""
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(1)
        price = ticker.last or ticker.close
        self._ib.cancelMktData(contract)
        return float(price) if price and price == price else None  # NaN guard
