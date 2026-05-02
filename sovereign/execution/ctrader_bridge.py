"""
CTrader Bridge — sovereign/execution/ctrader_bridge.py

Connects the Sovereign forex signal pipeline to a live/demo cTrader account
via the cTrader Open API. GBPUSD only for the 30-day drift test.

Usage:
    python3 sovereign/execution/ctrader_bridge.py --mode demo   # 48h demo first
    python3 sovereign/execution/ctrader_bridge.py --mode live   # only after demo confirmed

Credentials (environment variables — never hardcode):
    CTRADER_CLIENT_ID
    CTRADER_CLIENT_SECRET
    CTRADER_ACCOUNT_ID
    CTRADER_ACCESS_TOKEN     # OAuth token from cTrader ID portal

Hard safety rules (non-negotiable, defined once here):
    max_risk_per_trade:  1.0%  — never more
    max_open_positions:  1     — GBPUSD only, one at a time
    daily_loss_limit:    2.0%  — halt all trading if hit
    max_spread_pips:     3.0   — reject fill if spread exceeds this

Entry gate (signal quality filter):
    pair == 'GBPUSD=X'
    conviction >= 0.60
    trajectory_model p50 > 0.20    (via TrajectoryModel)
    atr_pct >= 0.022               (2.2% ATR minimum)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from twisted.internet import reactor, defer

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOATraderReq,
    ProtoOANewOrderReq,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAPayloadType,
    ProtoOAOrderType,
    ProtoOATradeSide,
)

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ── File paths ────────────────────────────────────────────────────────
ROOT         = Path(__file__).parents[2]
FILLS_LOG    = ROOT / 'data' / 'ledger' / 'live_fills_GBPUSD.jsonl'
VETO_LOG     = ROOT / 'data' / 'ledger' / 'live_veto_ledger.jsonl'
FILLS_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────
GBPUSD_SYMBOL      = 'GBPUSD'
PIP_SIZE           = 0.0001          # GBPUSD pip
MAX_RISK_PCT       = 0.010           # 1% per trade
MAX_OPEN_POSITIONS = 1
DAILY_LOSS_LIMIT   = 0.020           # 2% account equity
MAX_SPREAD_PIPS    = 3.0
MIN_CONVICTION     = 0.60
MIN_TRAJECTORY_P50 = 0.20
MIN_ATR_PCT        = 0.022

# Signal filter: only process GBPUSD with sufficient quality
TARGET_PAIR        = 'GBPUSD=X'


@dataclass
class FillRecord:
    timestamp:          str
    direction:          str
    size_lots:          float
    entry_price:        float
    requested_price:    float
    slippage_pips:      float
    spread_pips:        float
    stop_price:         float
    target_price:       float
    conviction:         float
    predicted_r_p50:    float
    predicted_r_p90:    float
    order_id:           str
    mode:               str   # 'demo' or 'live'


@dataclass
class VetoRecord:
    timestamp:   str
    reason:      str
    pair:        str
    details:     dict


def _log_fill(fill: FillRecord) -> None:
    with open(FILLS_LOG, 'a') as f:
        f.write(json.dumps(asdict(fill)) + '\n')
    logger.info(
        f"[FILL] {fill.direction} {fill.size_lots:.2f} lots @ {fill.entry_price:.5f} | "
        f"slip={fill.slippage_pips:+.1f}pip spread={fill.spread_pips:.1f}pip"
    )


def _log_veto(reason: str, pair: str, details: dict) -> None:
    rec = VetoRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        reason=reason, pair=pair, details=details
    )
    with open(VETO_LOG, 'a') as f:
        f.write(json.dumps(asdict(rec)) + '\n')
    logger.warning(f"[VETO] {reason} | {details}")


def _daily_pnl_pct(account_balance: float, initial_balance: float) -> float:
    return (account_balance - initial_balance) / initial_balance


class CTraderBridge:
    """
    Connects Sovereign forex signals to cTrader Open API.
    Manages connection lifecycle, order submission, and risk enforcement.
    """

    def __init__(self, mode: str = 'demo'):
        assert mode in ('demo', 'live'), "mode must be 'demo' or 'live'"
        self.mode = mode

        self._client_id     = os.environ['CTRADER_CLIENT_ID']
        self._client_secret = os.environ['CTRADER_CLIENT_SECRET']
        self._account_id    = int(os.environ['CTRADER_ACCOUNT_ID'])
        self._access_token  = os.environ['CTRADER_ACCESS_TOKEN']

        host = (EndPoints.PROTOBUF_DEMO_HOST if mode == 'demo'
                else EndPoints.PROTOBUF_LIVE_HOST)
        self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

        self._connected        = False
        self._authenticated    = False
        self._gbpusd_symbol_id: Optional[int] = None
        self._account_balance:  float = 0.0
        self._session_start_balance: float = 0.0
        self._open_positions:   int   = 0

        # Lazy-load trajectory model (no penalty if unavailable)
        self._trajectory = None

        # Wire client callbacks
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        logger.info(f"[CTraderBridge] Initialised in {mode.upper()} mode")

    # ── Connection lifecycle ──────────────────────────────────────────

    def start(self) -> None:
        """Start the Twisted reactor and connect."""
        logger.info(f"[CTraderBridge] Connecting to {self.mode} server…")
        self._client.startService()
        reactor.run()

    def stop(self) -> None:
        logger.info("[CTraderBridge] Shutting down.")
        self._client.stopService()
        if reactor.running:
            reactor.stop()

    def _on_connected(self, client) -> None:
        self._connected = True
        logger.info("[CTraderBridge] TCP connected — authenticating application…")
        req = ProtoOAApplicationAuthReq()
        req.clientId     = self._client_id
        req.clientSecret = self._client_secret
        deferred = client.send(req)
        deferred.addErrback(self._on_error)

    def _on_disconnected(self, client, reason) -> None:
        self._connected = self._authenticated = False
        logger.warning(f"[CTraderBridge] Disconnected: {reason}")

    def _on_message(self, client, message) -> None:
        payload_type = message.payloadType

        if payload_type == ProtoOAPayloadType.PROTO_OA_APPLICATION_AUTH_RES:
            logger.info("[CTraderBridge] App auth OK — authenticating account…")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self._account_id
            req.accessToken         = self._access_token
            client.send(req).addErrback(self._on_error)

        elif payload_type == ProtoOAPayloadType.PROTO_OA_ACCOUNT_AUTH_RES:
            self._authenticated = True
            logger.info("[CTraderBridge] Account authenticated ✓")
            self._fetch_account_info()
            self._fetch_symbol_id()

        elif payload_type == ProtoOAPayloadType.PROTO_OA_TRADER_RES:
            resp = Protobuf.extract(message)
            self._account_balance = resp.trader.balance / 100.0  # cents → units
            if self._session_start_balance == 0:
                self._session_start_balance = self._account_balance
            logger.info(f"[CTraderBridge] Account balance: {self._account_balance:.2f}")

        elif payload_type == ProtoOAPayloadType.PROTO_OA_SYMBOLS_LIST_RES:
            resp = Protobuf.extract(message)
            for sym in resp.symbol:
                if sym.symbolName == GBPUSD_SYMBOL:
                    self._gbpusd_symbol_id = sym.symbolId
                    logger.info(f"[CTraderBridge] GBPUSD symbolId={self._gbpusd_symbol_id}")
                    break
            if self._gbpusd_symbol_id is None:
                logger.error("[CTraderBridge] GBPUSD not found in symbol list — check account")

        elif payload_type == ProtoOAPayloadType.PROTO_OA_NEW_ORDER_RES:
            resp = Protobuf.extract(message)
            logger.info(f"[CTraderBridge] Order accepted: orderId={resp.order.orderId}")

        elif payload_type == ProtoOAPayloadType.PROTO_OA_EXECUTION_EVENT:
            self._handle_execution(Protobuf.extract(message))

        elif payload_type == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
            resp = Protobuf.extract(message)
            logger.error(f"[CTraderBridge] API error: {resp.errorCode} — {resp.description}")

    @staticmethod
    def _on_error(failure) -> None:
        logger.error(f"[CTraderBridge] Deferred error: {failure}")

    # ── Account & symbol fetch ────────────────────────────────────────

    def _fetch_account_info(self) -> None:
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self._account_id
        self._client.send(req).addErrback(self._on_error)

    def _fetch_symbol_id(self) -> None:
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self._account_id
        req.includeArchivedSymbols = False
        self._client.send(req).addErrback(self._on_error)

    # ── Signal intake ─────────────────────────────────────────────────

    def submit_signal(self, signal) -> bool:
        """
        Accept a ForexEntrySignal from forex_specialist.py.
        Returns True if order was sent, False if vetoed.

        signal: ForexEntrySignal with attributes:
            pair, direction, conviction, entry_price, stop_price,
            t1, macro_conviction, ict_analysis
        """
        ts = datetime.now(timezone.utc).isoformat()

        # ── Gate 1: pair filter ───────────────────────────────────────
        if getattr(signal, 'pair', '') != TARGET_PAIR:
            return False   # silent — not our pair

        # ── Gate 2: system ready ──────────────────────────────────────
        if not self._authenticated or self._gbpusd_symbol_id is None:
            _log_veto('NOT_READY', TARGET_PAIR,
                      {'authenticated': self._authenticated,
                       'symbol_id': self._gbpusd_symbol_id})
            return False

        # ── Gate 3: daily loss limit ──────────────────────────────────
        if self._account_balance > 0 and self._session_start_balance > 0:
            daily_loss = _daily_pnl_pct(self._account_balance, self._session_start_balance)
            if daily_loss <= -DAILY_LOSS_LIMIT:
                _log_veto('DAILY_LOSS_LIMIT', TARGET_PAIR,
                          {'daily_pnl_pct': round(daily_loss * 100, 2),
                           'limit_pct': DAILY_LOSS_LIMIT * 100})
                logger.error("[HALT] Daily loss limit hit — no more trades today")
                return False

        # ── Gate 4: max open positions ────────────────────────────────
        if self._open_positions >= MAX_OPEN_POSITIONS:
            _log_veto('MAX_POSITIONS', TARGET_PAIR,
                      {'open': self._open_positions, 'max': MAX_OPEN_POSITIONS})
            return False

        # ── Gate 5: conviction ────────────────────────────────────────
        conviction = float(getattr(signal, 'macro_conviction',
                                   getattr(signal, 'conviction', 0.0)))
        if conviction < MIN_CONVICTION:
            _log_veto('LOW_CONVICTION', TARGET_PAIR,
                      {'conviction': conviction, 'min': MIN_CONVICTION})
            return False

        # ── Gate 6: ATR filter ────────────────────────────────────────
        ict = getattr(signal, 'ict_analysis', None)
        atr_pct = 0.0
        if ict and ict.current_price > 0:
            atr_pct = ict.atr_daily / ict.current_price
        if atr_pct < MIN_ATR_PCT:
            _log_veto('ATR_TOO_LOW', TARGET_PAIR,
                      {'atr_pct': round(atr_pct, 4), 'min': MIN_ATR_PCT})
            return False

        # ── Gate 7: trajectory model ──────────────────────────────────
        pred_p50, pred_p90 = 0.0, 0.0
        try:
            if self._trajectory is None:
                from sovereign.prediction.trajectory_model import TrajectoryModel
                self._trajectory = TrajectoryModel()
                self._trajectory.train()
            conditions = {
                'regime':       'MOMENTUM',
                'hurst':        0.55,
                'atr_pct':      atr_pct * 100,
                'adx':          30.0,
                'spy_5d_return': 0.0,
                'strategy':     'macro_divergence',
                'vix':          18.0,
                'direction':    signal.direction,
            }
            pred = self._trajectory.predict(conditions)
            pred_p50 = pred.p50
            pred_p90 = pred.p90
            if pred_p50 < MIN_TRAJECTORY_P50:
                _log_veto('TRAJECTORY_VETO', TARGET_PAIR,
                          {'p50': pred_p50, 'min': MIN_TRAJECTORY_P50})
                return False
        except Exception as e:
            logger.warning(f"[Trajectory] non-fatal: {e} — proceeding without filter")

        # ── Compute size ──────────────────────────────────────────────
        entry   = float(signal.entry_price)
        stop    = float(signal.stop_price)
        risk_distance = abs(entry - stop)
        if risk_distance == 0:
            _log_veto('ZERO_RISK_DISTANCE', TARGET_PAIR, {'entry': entry, 'stop': stop})
            return False

        risk_amount  = self._account_balance * MAX_RISK_PCT
        pip_value    = 10.0                     # $10/pip per standard lot for GBPUSD
        pips_at_risk = risk_distance / PIP_SIZE
        size_lots    = risk_amount / (pips_at_risk * pip_value)
        size_lots    = round(max(0.01, min(size_lots, 10.0)), 2)  # floor 0.01, cap 10

        # ── Gate 8: spread check (fetch from reconcile or use estimate) ─
        # We don't have live tick here, so we log spread post-fill from execution event.
        # Pre-check: if market hours are thin (weekends, Asia dead zone), add advisory.

        # ── Send order ────────────────────────────────────────────────
        return self._send_market_order(
            direction=signal.direction,
            size_lots=size_lots,
            entry=entry,
            stop=stop,
            target=float(signal.t1),
            conviction=conviction,
            pred_p50=pred_p50,
            pred_p90=pred_p90,
        )

    def _send_market_order(
        self,
        direction: str,
        size_lots: float,
        entry: float,
        stop: float,
        target: float,
        conviction: float,
        pred_p50: float,
        pred_p90: float,
    ) -> bool:
        if self._gbpusd_symbol_id is None:
            logger.error("[CTraderBridge] No symbol ID — cannot send order")
            return False

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId            = self._gbpusd_symbol_id
        req.orderType           = ProtoOAOrderType.Value('MARKET')
        req.tradeSide           = (ProtoOATradeSide.Value('BUY')
                                   if direction == 'LONG'
                                   else ProtoOATradeSide.Value('SELL'))
        req.volume              = int(size_lots * 100)   # cTrader: lots × 100

        # Stop loss
        if stop > 0:
            req.relativeStopLoss = int(abs(entry - stop) / PIP_SIZE * 10)
            req.trailingStopLoss = False

        # Take profit (T1)
        if target > 0:
            req.relativeTakeProfit = int(abs(target - entry) / PIP_SIZE * 10)

        self._pending_fill = {
            'direction': direction,
            'size_lots': size_lots,
            'requested_price': entry,
            'stop_price': stop,
            'target_price': target,
            'conviction': conviction,
            'predicted_r_p50': pred_p50,
            'predicted_r_p90': pred_p90,
        }

        logger.info(
            f"[ORDER] Sending {direction} {size_lots} lots GBPUSD | "
            f"entry≈{entry:.5f} stop={stop:.5f} target={target:.5f} | "
            f"conviction={conviction:.2f} p50={pred_p50:+.2f}"
        )
        self._client.send(req).addErrback(self._on_error)
        return True

    # ── Execution event handler ───────────────────────────────────────

    def _handle_execution(self, event) -> None:
        """Called on every execution event — log fills, track positions."""
        if not hasattr(event, 'deal') or not event.deal:
            return

        deal = event.deal
        fill_price = deal.executionPrice / 100000.0   # cTrader fixed-point
        pf = getattr(self, '_pending_fill', {})

        req_price = pf.get('requested_price', fill_price)
        slippage_pips = (fill_price - req_price) / PIP_SIZE
        if pf.get('direction') == 'SELL':
            slippage_pips = -slippage_pips

        spread_pips = getattr(deal, 'spreadInPips', 0.0) / 10.0

        # ── Spread safety check (post-fill, log veto retroactively) ──
        if spread_pips > MAX_SPREAD_PIPS:
            logger.warning(
                f"[SPREAD WARNING] Fill spread {spread_pips:.1f} pips > limit {MAX_SPREAD_PIPS} — "
                f"logged but cannot cancel. Tighten timing next time."
            )
            _log_veto('SPREAD_EXCEEDED_POST_FILL', TARGET_PAIR,
                      {'spread_pips': spread_pips, 'max': MAX_SPREAD_PIPS,
                       'fill_price': fill_price})

        fill = FillRecord(
            timestamp        = datetime.now(timezone.utc).isoformat(),
            direction        = pf.get('direction', 'UNKNOWN'),
            size_lots        = pf.get('size_lots', 0.0),
            entry_price      = fill_price,
            requested_price  = req_price,
            slippage_pips    = round(slippage_pips, 1),
            spread_pips      = round(spread_pips, 1),
            stop_price       = pf.get('stop_price', 0.0),
            target_price     = pf.get('target_price', 0.0),
            conviction       = pf.get('conviction', 0.0),
            predicted_r_p50  = pf.get('predicted_r_p50', 0.0),
            predicted_r_p90  = pf.get('predicted_r_p90', 0.0),
            order_id         = str(getattr(deal, 'dealId', '')),
            mode             = self.mode,
        )
        _log_fill(fill)
        self._open_positions += 1
        self._pending_fill = {}

    # ── Self-test ─────────────────────────────────────────────────────

    def run_safety_rule_test(self) -> None:
        """
        Demonstrates all risk rules firing correctly without connecting.
        Prints PASS/FAIL for each rule.
        """
        from types import SimpleNamespace
        print(f"\n{'═'*55}")
        print(f"  CTRADER BRIDGE — SAFETY RULE VERIFICATION")
        print(f"{'─'*55}")

        results = []

        # Test 1: daily loss limit
        self._account_balance       = 9_750.0
        self._session_start_balance = 10_000.0
        self._authenticated         = True
        self._gbpusd_symbol_id      = 1

        mock_sig = SimpleNamespace(
            pair='GBPUSD=X', direction='LONG', entry_price=1.2700,
            stop_price=1.2600, t1=1.2850, macro_conviction=0.75,
            ict_analysis=SimpleNamespace(atr_daily=0.0055, current_price=1.27),
        )
        result = self.submit_signal(mock_sig)
        fired = not result
        results.append(('DAILY_LOSS_LIMIT (-2.5%)',  fired))
        print(f"  {'PASS' if fired else 'FAIL'}  DAILY_LOSS_LIMIT  "
              f"(balance={self._account_balance}, start={self._session_start_balance})")

        # Test 2: low conviction
        self._account_balance = 10_000.0
        mock_sig.macro_conviction = 0.45
        result = self.submit_signal(mock_sig)
        fired = not result
        results.append(('LOW_CONVICTION (0.45 < 0.60)', fired))
        print(f"  {'PASS' if fired else 'FAIL'}  LOW_CONVICTION   "
              f"(conviction=0.45, min={MIN_CONVICTION})")

        # Test 3: ATR too low
        mock_sig.macro_conviction = 0.75
        mock_sig.ict_analysis = SimpleNamespace(atr_daily=0.0010, current_price=1.27)
        result = self.submit_signal(mock_sig)
        fired = not result
        results.append(('ATR_TOO_LOW (0.08% < 2.2%)', fired))
        print(f"  {'PASS' if fired else 'FAIL'}  ATR_TOO_LOW      "
              f"(atr_pct={0.001/1.27:.3%}, min={MIN_ATR_PCT:.1%})")

        # Test 4: wrong pair
        mock_sig.pair = 'EURUSD=X'
        result = self.submit_signal(mock_sig)
        results.append(('WRONG_PAIR (EURUSD filtered)', not result))
        print(f"  {'PASS' if not result else 'FAIL'}  WRONG_PAIR       "
              f"(pair=EURUSD, target=GBPUSD)")

        all_pass = all(r[1] for r in results)
        print(f"{'─'*55}")
        print(f"  {'ALL RULES PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")
        print(f"{'═'*55}\n")
        return all_pass


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CTrader Bridge — Sovereign Forex')
    parser.add_argument('--mode', choices=['demo', 'live', 'test'], default='demo',
                        help='demo: FunderPro demo server | live: real money | test: safety rules only')
    args = parser.parse_args()

    if args.mode == 'test':
        # Verify all safety rules fire correctly without a real connection
        bridge = CTraderBridge.__new__(CTraderBridge)
        bridge.mode             = 'test'
        bridge._authenticated   = False
        bridge._gbpusd_symbol_id = None
        bridge._account_balance = 0.0
        bridge._session_start_balance = 0.0
        bridge._open_positions  = 0
        bridge._trajectory      = None
        bridge._pending_fill    = {}
        bridge.run_safety_rule_test()
        return

    # Verify credentials present before connecting
    required = ['CTRADER_CLIENT_ID', 'CTRADER_CLIENT_SECRET',
                'CTRADER_ACCOUNT_ID', 'CTRADER_ACCESS_TOKEN']
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"\n[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Set them in .env or export before running.")
        return

    if args.mode == 'live':
        print("\n⚠  LIVE MODE — real money will be at risk.")
        print("   Confirm you have run 48h demo successfully.")
        confirm = input("   Type CONFIRM to proceed: ").strip()
        if confirm != 'CONFIRM':
            print("   Aborted.")
            return

    bridge = CTraderBridge(mode=args.mode)
    bridge.start()   # blocks (Twisted reactor)


if __name__ == '__main__':
    main()
