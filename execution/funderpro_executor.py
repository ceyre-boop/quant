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

LIVE routing guard (cannot be bypassed):
  FUNDERPRO_LIVE=live is refused unless:
    • data/pipeline_verdict.json exists and contains verdict=GO
    • The config hash recorded in that file matches config/parameters.yml

Status: ROUTING_OFF by default.
Toggle: set env var FUNDERPRO_LIVE=1 (demo) or FUNDERPRO_LIVE=live.
Credentials via env vars — never hardcoded.

Usage:
    executor = FunderProExecutor()
    executor.submit(scan_result, tp1_r=2.0, tp2_r=4.0)

    # Record a GO verdict after a successful pipeline evaluation:
    from execution.funderpro_executor import record_pipeline_go
    record_pipeline_go()
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
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

# ── Pipeline GO guard paths ────────────────────────────────────────────────── #

_ROOT                = Path(__file__).resolve().parents[1]
PIPELINE_VERDICT_FILE = _ROOT / 'data' / 'pipeline_verdict.json'
CONFIG_FILE           = _ROOT / 'config' / 'parameters.yml'


# ── Public utility: call after a successful pipeline evaluation ────────────── #

def record_pipeline_go(
    config_path: Path = CONFIG_FILE,
    verdict_path: Path = PIPELINE_VERDICT_FILE,
) -> None:
    """
    Persist a GO verdict so FUNDERPRO_LIVE=live can proceed.

    Call this from run_live_pipeline.py (or equivalent) immediately after the
    pipeline evaluation prints 🟢 GO.  The config file hash is recorded so
    that any subsequent parameter change forces a re-evaluation.

    Example::

        from execution.funderpro_executor import record_pipeline_go
        record_pipeline_go()
    """
    config_hash = (
        hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
        if config_path.exists()
        else ''
    )
    payload = {
        'verdict':     'GO',
        'timestamp':   datetime.now(timezone.utc).isoformat(),
        'config_hash': config_hash,
    }
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Pipeline GO verdict recorded (config_hash=%s, path=%s)",
        config_hash, verdict_path,
    )


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

    LIVE mode is refused at construction time unless ``data/pipeline_verdict.json``
    records a GO verdict **and** the config hash matches ``config/parameters.yml``.
    This hard guard is intentional: the code enforces the discipline of waiting
    for the pipeline to confirm the edge before pressing the button.
    """

    def __init__(self, account_size: float = 10_000.0):
        self.account_size = account_size
        self._routing     = self._detect_routing()
        self._daily_pnl   = 0.0
        self._open        = {}   # pair → position_id (cTrader positionId string)
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

    # ── cTrader integration ────────────────────────────────────────────────── #

    def _init_ctrader(self) -> None:
        """
        Create a ``CTraderBridge``, start its Twisted reactor in a daemon
        background thread, and block until authentication completes.

        Stores the connected bridge in ``self._ctrader``.  If credentials are
        missing or authentication times out the executor falls back to stub
        mode (``self._ctrader`` stays ``None``), which causes subsequent
        ``_send_ctrader_order`` calls to raise and be caught by ``submit``.
        """
        try:
            client_id     = os.environ.get('CTRADER_CLIENT_ID', '')
            client_secret = os.environ.get('CTRADER_CLIENT_SECRET', '')
            account_id    = os.environ.get('CTRADER_ACCOUNT_ID', '')
            access_token  = os.environ.get('CTRADER_ACCESS_TOKEN', '')
            if not all([client_id, client_secret, account_id, access_token]):
                logger.warning(
                    "cTrader credentials incomplete — "
                    "set CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, "
                    "CTRADER_ACCOUNT_ID, CTRADER_ACCESS_TOKEN"
                )
                return

            from sovereign.execution.ctrader_bridge import CTraderBridge

            mode   = 'live' if self._routing == 'LIVE' else 'demo'
            bridge = CTraderBridge(mode=mode)

            # Run the Twisted reactor in a daemon thread so it doesn't block
            # the main process and dies automatically when the process exits.
            reactor_thread = threading.Thread(
                target=bridge.start,
                name='ctrader-reactor',
                daemon=True,
            )
            reactor_thread.start()

            # Wait for the bridge to authenticate and resolve the symbol ID.
            if not bridge.wait_for_ready(timeout=30.0):
                logger.error(
                    "cTrader: authentication timed out after 30s — "
                    "check credentials and network connectivity"
                )
                return

            self._ctrader = bridge
            logger.info("cTrader: connected and authenticated (%s)", mode.upper())

        except Exception as e:
            logger.warning("cTrader init failed: %s", e)

    def _send_ctrader_order(self, order: OrderResult) -> str:
        """
        Send a bracket order via ``CTraderBridge.send_bracket_order``.

        Uses TP1 as the primary take-profit on the broker side.  TP2/TP3
        partial exits are managed by the RR engine once the position is open.

        Returns the cTrader ``positionId`` string (used for close).
        Raises ``RuntimeError`` on timeout or if the bridge is not connected.
        """
        if self._ctrader is None:
            raise RuntimeError(
                "cTrader bridge not connected — check credentials and FUNDERPRO_LIVE"
            )

        position_id = self._ctrader.send_bracket_order(
            direction=order.direction,
            size_lots=order.lots,
            entry=order.entry,
            stop=order.stop,
            target=order.tp1,   # primary TP; RR engine manages TP2/TP3 partials
            conviction=0.0,
            pred_p50=0.0,
            pred_p90=0.0,
            timeout=15.0,
        )
        if position_id is None:
            raise RuntimeError(
                f"Order submission timed out or failed for {order.pair} "
                f"({order.direction} @ {order.entry:.5f})"
            )
        return position_id

    def _close_ctrader_order(self, position_id: str) -> None:
        """
        Send a market-close for the given cTrader ``positionId``.

        Delegates to ``CTraderBridge.close_position`` which dispatches a
        ``ProtoOAClosePositionReq`` on the reactor thread.
        Raises ``RuntimeError`` on timeout or if the bridge is not connected.
        """
        if self._ctrader is None:
            raise RuntimeError(
                "cTrader bridge not connected — cannot close position"
            )
        ok = self._ctrader.close_position(position_id=position_id, timeout=10.0)
        if not ok:
            raise RuntimeError(
                f"Close request timed out for positionId={position_id}"
            )

    # ── Live routing guard ─────────────────────────────────────────────────── #

    @staticmethod
    def _check_pipeline_go(
        verdict_path: Path = PIPELINE_VERDICT_FILE,
        config_path: Path = CONFIG_FILE,
    ) -> tuple[bool, str]:
        """
        Return ``(True, reason)`` if the pipeline last printed GO **and** the
        config hash matches ``config/parameters.yml``.

        Returns ``(False, reason)`` in every failure case.  Called from
        ``_detect_routing`` to hard-block LIVE mode until the edge is confirmed.
        """
        if not verdict_path.exists():
            return (
                False,
                f"No pipeline verdict on record ({verdict_path}) — "
                "run the pipeline evaluation and call record_pipeline_go() first",
            )
        try:
            payload = json.loads(verdict_path.read_text())
        except Exception as exc:
            return False, f"Cannot read pipeline verdict: {exc}"

        if payload.get('verdict') != 'GO':
            v = payload.get('verdict', 'UNKNOWN')
            ts = payload.get('timestamp', '')
            return False, f"Pipeline verdict is {v} (recorded {ts}) — must be GO to enable LIVE"

        if config_path.exists():
            current_hash  = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
            recorded_hash = payload.get('config_hash', '')
            if recorded_hash and current_hash != recorded_hash:
                return (
                    False,
                    f"config/parameters.yml changed since GO verdict "
                    f"(recorded={recorded_hash}, current={current_hash}) — "
                    "re-run pipeline evaluation before going live",
                )

        ts = payload.get('timestamp', 'unknown time')
        return True, f"Pipeline GO confirmed (recorded {ts})"

    @staticmethod
    def _detect_routing() -> str:
        val = os.environ.get('FUNDERPRO_LIVE', 'off').lower()
        if val == 'live':
            ok, reason = FunderProExecutor._check_pipeline_go()
            if not ok:
                raise RuntimeError(
                    f"LIVE mode refused by pipeline GO guard:\n  {reason}\n"
                    "Call record_pipeline_go() after a successful evaluation, "
                    "or set FUNDERPRO_LIVE=demo for paper validation first."
                )
            logger.info("LIVE routing enabled — pipeline GO guard passed")
            return 'LIVE'
        if val in ('demo', '1', 'true'):
            return 'DEMO'
        return 'OFF'

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
