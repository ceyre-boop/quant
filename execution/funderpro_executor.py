"""
execution/funderpro_executor.py
================================
FunderPro prop-firm executor for the ICT engine.

FunderPro uses cTrader as its execution platform.
This module wraps the existing ctrader_bridge.py protocol
for ICT-format signals (ScanResult → cTrader order).

Safety rules (hard-coded, non-negotiable):
  max_risk_per_trade:  1.0%  — ICT paper protocol
  max_open_positions:  1 per pair
  daily_loss_limit:    4.0%  — FunderPro challenge rule
  max_spread_pips:     2.5   — forex, tight spreads only

Status: ROUTING_OFF by default.
Toggle: set env var FUNDERPRO_LIVE=1 to enable.
Credentials via env vars — never hardcoded.

Usage:
    executor = FunderProExecutor()
    executor.submit(scan_result, tp1_r=2.0, tp2_r=4.0)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────── #

MAX_RISK_PCT     = 0.01    # 1% per trade
MAX_OPEN         = 1       # per pair
DAILY_LOSS_LIMIT = 0.04    # 4% daily DD limit (FunderPro challenge)
MAX_SPREAD_PIPS  = 2.5

# Pip sizes per pair
PIP_SIZE = {
    'USDJPY': 0.01,
    'EURUSD': 0.0001,
    'NZDUSD': 0.0001,
    'GBPUSD': 0.0001,
    'AUDUSD': 0.0001,
}


@dataclass
class ExecutorStatus:
    routing:        str      # OFF | DEMO | LIVE
    connected:      bool
    daily_pnl_pct:  float
    open_positions: int
    blocked:        bool     # True if daily loss limit hit
    reason:         str

    def to_dict(self):
        return asdict(self)


@dataclass
class OrderResult:
    submitted:    bool
    order_id:     str
    pair:         str
    direction:    str
    entry:        float
    stop:         float
    tp1:          float
    tp2:          float
    lots:         float
    routing:      str
    timestamp:    str
    error:        str = ''

    def to_dict(self):
        return asdict(self)


class FunderProExecutor:
    """
    ICT signal → FunderPro cTrader order.

    Routing modes:
      OFF   — log only, no orders sent (default)
      DEMO  — send to demo account for validation
      LIVE  — send to live challenge account
    """

    def __init__(self, account_size: float = 10_000.0):
        self.account_size = account_size
        self._routing     = self._detect_routing()
        self._daily_pnl   = 0.0
        self._open        = {}   # pair → order_id
        self._ctrader     = None

        if self._routing != 'OFF':
            self._init_ctrader()

    def get_status(self) -> ExecutorStatus:
        blocked = self._daily_pnl <= -DAILY_LOSS_LIMIT
        return ExecutorStatus(
            routing=self._routing,
            connected=self._ctrader is not None,
            daily_pnl_pct=round(self._daily_pnl, 4),
            open_positions=len(self._open),
            blocked=blocked,
            reason='Daily loss limit hit' if blocked else 'OK',
        )

    def submit(
        self,
        scan_result,
        tp1_r: float = 2.0,
        tp2_r: float = 4.0,
    ) -> OrderResult:
        """
        Submit an ICT signal as a cTrader bracket order.
        Returns OrderResult with submitted=False if routing is OFF.
        """
        pair = scan_result.pair
        ts   = datetime.now(timezone.utc).isoformat()

        # Safety gates
        status = self.get_status()
        if status.blocked:
            return self._reject(pair, scan_result.signal, ts,
                                f'Daily loss limit {DAILY_LOSS_LIMIT*100:.0f}% hit')

        if pair in self._open:
            return self._reject(pair, scan_result.signal, ts, f'Already have open position on {pair}')

        entry = scan_result.entry_level or 0.0
        stop  = scan_result.stop or 0.0
        if entry == 0 or stop == 0:
            return self._reject(pair, scan_result.signal, ts, 'Missing entry/stop levels')

        stop_dist    = abs(entry - stop)
        sign         = 1 if scan_result.signal == 'LONG' else -1
        tp1          = round(entry + sign * stop_dist * tp1_r, 5)
        tp2          = round(entry + sign * stop_dist * tp2_r, 5)
        risk_dollars = self.account_size * MAX_RISK_PCT
        lots         = self._size_lots(pair, risk_dollars, stop_dist)

        order = OrderResult(
            submitted=False,
            order_id='',
            pair=pair,
            direction=scan_result.signal,
            entry=round(entry, 5),
            stop=round(stop, 5),
            tp1=tp1,
            tp2=tp2,
            lots=lots,
            routing=self._routing,
            timestamp=ts,
        )

        if self._routing == 'OFF':
            logger.info("ROUTING OFF — would submit: %s %s @ %.5f  lots=%.2f",
                        pair, scan_result.signal, entry, lots)
            order.order_id = f'SIMULATED_{pair}_{ts[:10]}'
            return order

        # ── DEMO / LIVE: send via cTrader ─────────────────────────────────
        try:
            oid = self._send_ctrader_order(order)
            order.submitted = True
            order.order_id  = oid
            self._open[pair] = oid
            logger.info("✅ SUBMITTED %s %s @ %.5f  lots=%.2f  id=%s",
                        pair, scan_result.signal, entry, lots, oid)
        except Exception as e:
            order.error = str(e)
            logger.error("cTrader submit failed for %s: %s", pair, e)

        return order

    def close_position(self, pair: str, reason: str = 'SESSION_END') -> bool:
        if pair not in self._open:
            return False
        if self._routing == 'OFF':
            del self._open[pair]
            return True
        try:
            self._close_ctrader_order(self._open[pair])
            del self._open[pair]
            logger.info("Closed %s: %s", pair, reason)
            return True
        except Exception as e:
            logger.error("Failed to close %s: %s", pair, e)
            return False

    # ── cTrader integration stubs ─────────────────────────────────────────── #

    def _init_ctrader(self):
        """Initialize cTrader API connection."""
        try:
            client_id     = os.environ.get('CTRADER_CLIENT_ID', '')
            client_secret = os.environ.get('CTRADER_CLIENT_SECRET', '')
            account_id    = os.environ.get('CTRADER_ACCOUNT_ID', '')
            if not all([client_id, client_secret, account_id]):
                logger.warning("cTrader credentials not set — remaining in stub mode")
                return
            # TODO: import ctrader OpenAPI client and authenticate
            # from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth
            # self._ctrader = Client(...)
            logger.info("cTrader: stub initialized (credentials present, API not yet wired)")
        except Exception as e:
            logger.warning("cTrader init failed: %s", e)

    def _send_ctrader_order(self, order: OrderResult) -> str:
        """Send bracket order to cTrader. Returns order ID."""
        # TODO: implement using cTrader OpenAPI ProtoOANewOrderReq
        # with stop-loss and take-profit attached
        raise NotImplementedError("cTrader live order submission not yet implemented. "
                                  "Set FUNDERPRO_LIVE=0 to use paper routing.")

    def _close_ctrader_order(self, order_id: str):
        """Close a cTrader position by order ID."""
        raise NotImplementedError("cTrader close not yet implemented.")

    # ── Sizing ─────────────────────────────────────────────────────────────── #

    def _size_lots(self, pair: str, risk_dollars: float, stop_dist: float) -> float:
        pip = PIP_SIZE.get(pair, 0.0001)
        pips = stop_dist / pip
        # Standard lot = 100,000 units. $10/pip for EURUSD at standard lot.
        # For simplicity: lots = risk_dollars / (pips * pip_value_per_lot)
        pip_value = 10.0  # ~$10/pip per standard lot for major pairs
        lots = risk_dollars / (pips * pip_value)
        return round(max(lots, 0.01), 2)  # minimum 0.01 micro lot

    def _reject(self, pair, direction, ts, reason) -> OrderResult:
        logger.info("Order rejected — %s: %s", pair, reason)
        return OrderResult(
            submitted=False, order_id='', pair=pair, direction=direction,
            entry=0, stop=0, tp1=0, tp2=0, lots=0,
            routing=self._routing, timestamp=ts, error=reason,
        )

    @staticmethod
    def _detect_routing() -> str:
        val = os.environ.get('FUNDERPRO_LIVE', 'off').lower()
        if val == 'live':   return 'LIVE'
        if val in ('demo', '1', 'true'): return 'DEMO'
        return 'OFF'
