"""
run_oanda_paper.py — ICT signals → OANDA practice account

Consumes ICT Grade A signals from ICTOrchestrator and places them on the
OANDA practice account. Trades appear live on TradingView when the chart
is connected to OANDA.

Decision logging is already handled inside ICTOrchestrator._log_decisions().
This script handles: execution, trade tracking, and outcome closure so that
Oracle can learn from the results.

Usage:
    python3 scripts/run_oanda_paper.py              # scan every 5 min
    python3 scripts/run_oanda_paper.py --dry-run    # signals only, no trades placed
    python3 scripts/run_oanda_paper.py --interval 60  # faster scan for testing

Architecture (ICT isolation preserved):
    ICTOrchestrator  →  [signals]  →  OandaBridge  →  OANDA API  →  TradingView
         ↑                                                 ↓
    ict-engine/                               update_outcome() on close
    (never imports sovereign/execution/)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import pandas as pd

from ict.pipeline import ICTSignal

# ict-engine/ has a hyphen — not a valid Python package name.
# Add it directly to sys.path so we can import orchestrator.py from it.
sys.path.insert(0, str(ROOT / 'ict-engine'))
from orchestrator import ICTOrchestrator, ScanCycle  # type: ignore[import]

from sovereign.execution.oanda_bridge import OandaBridge, to_oanda_pair
from sovereign.intelligence.decision_logger import update_outcome

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ── Data provider (yfinance, 5-minute bars) ───────────────────────────────────

def make_yf_provider(period: str = "5d", interval: str = "5m"):
    """Returns a data provider callable for ICTOrchestrator."""
    def provider(pair: str) -> Optional[pd.DataFrame]:
        ticker = pair if pair.endswith("=X") else pair + "=X"
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                logger.warning("yfinance returned empty data for %s", ticker)
                return None
            return df
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None
    return provider


# ── Trade tracker ─────────────────────────────────────────────────────────────

class OpenTradeTracker:
    """
    Tracks open OANDA trades so we can close the Oracle feedback loop.

    When a trade closes (TP hit, SL hit, or manual), we call update_outcome()
    so Oracle can learn from the result.
    """

    def __init__(self) -> None:
        # trade_id → {pair, entry_timestamp, entry_price, stop_price, direction}
        self._open: dict[str, dict] = {}

    def record(
        self,
        trade_id: str,
        pair_ict: str,
        entry_timestamp: str,
        entry_price: float,
        stop_price: float,
        direction: str,
    ) -> None:
        self._open[trade_id] = {
            'pair_ict':        pair_ict,
            'entry_timestamp': entry_timestamp,
            'entry_price':     entry_price,
            'stop_price':      stop_price,
            'direction':       direction,
        }
        logger.info("[Tracker] Tracking trade_id=%s %s %s", trade_id, pair_ict, direction)

    def check_closures(self, bridge: OandaBridge) -> None:
        """Poll open trades; for each closure call update_outcome()."""
        if not self._open:
            return

        closed_ids = []
        for trade_id, ctx in self._open.items():
            trade = bridge.get_trade(trade_id)
            if trade is None:
                logger.warning("[Tracker] trade_id=%s not found — assuming closed", trade_id)
                closed_ids.append((trade_id, ctx, None, None))
                continue

            if trade.get('state') == 'CLOSED':
                # Parse exit price from closing transaction
                exit_price = float(trade.get('averageClosePrice', 0) or 0)
                close_time = trade.get('closeTime', datetime.now(timezone.utc).isoformat())
                closed_ids.append((trade_id, ctx, exit_price, close_time))

        for trade_id, ctx, exit_price, close_time in closed_ids:
            self._handle_closure(trade_id, ctx, exit_price, close_time)

    def _handle_closure(
        self,
        trade_id: str,
        ctx: dict,
        exit_price: Optional[float],
        close_time: Optional[str],
    ) -> None:
        pair_ict    = ctx['pair_ict']
        entry_ts    = ctx['entry_timestamp']
        entry_price = ctx['entry_price']
        stop_price  = ctx['stop_price']
        direction   = ctx['direction']

        # R-multiple: how many stop distances did we capture?
        r_realized = 0.0
        if exit_price and entry_price and stop_price and stop_price != entry_price:
            risk = abs(entry_price - stop_price)
            gain = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
            r_realized = round(gain / risk, 3)

        outcome = 'WIN' if r_realized > 0 else 'LOSS'

        logger.info(
            "[Tracker] %s closed | tradeId=%s | r=%.3f | outcome=%s",
            pair_ict, trade_id, r_realized, outcome,
        )

        updated = update_outcome(
            pair=pair_ict,
            entry_timestamp=entry_ts,
            outcome=outcome,
            r_realized=r_realized,
            exit_timestamp=close_time,
            system='ICT',
        )
        if not updated:
            logger.warning(
                "[Tracker] update_outcome() found no matching open record "
                "(pair=%s ts=%s) — Oracle will not learn from this trade",
                pair_ict, entry_ts,
            )

        del self._open[trade_id]


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(scan_interval: int = 300, dry_run: bool = False) -> None:
    bridge = OandaBridge()
    orchestrator = ICTOrchestrator(firebase_enabled=False)
    tracker = OpenTradeTracker()
    data_provider = make_yf_provider()

    logger.info(
        "OANDA paper runner started | interval=%ds | dry_run=%s",
        scan_interval, dry_run,
    )
    if dry_run:
        logger.info("[DRY RUN] Signals will be logged but no trades placed")

    check_interval = 60  # check open trade closures every 60s
    last_check = time.monotonic()

    while True:
        try:
            # ── 1. Check for trade closures ───────────────────────────────
            now_mono = time.monotonic()
            if now_mono - last_check >= check_interval:
                tracker.check_closures(bridge)
                last_check = now_mono

            # ── 2. Scan for new signals ───────────────────────────────────
            cycle: ScanCycle = orchestrator.scan_once(data_provider=data_provider)

            for signal in cycle.signals:
                _handle_signal(signal, bridge, tracker, dry_run)

        except KeyboardInterrupt:
            logger.info("OANDA paper runner stopped by user")
            break
        except Exception as exc:
            logger.exception("Scan loop error: %s", exc)

        time.sleep(scan_interval)


def _handle_signal(
    signal: ICTSignal,
    bridge: OandaBridge,
    tracker: OpenTradeTracker,
    dry_run: bool,
) -> None:
    pair_ict  = signal.symbol             # e.g. "GBPUSD"
    pair_oanda = to_oanda_pair(pair_ict)  # e.g. "GBP_USD"
    direction  = signal.direction
    entry      = float(signal.entry_level or 0)
    stop       = float(getattr(signal.sizing, 'stop_loss', 0) or 0)
    tp1        = float(getattr(signal.sizing, 'tp1', 0) or 0)
    grade      = signal.grade.value if hasattr(signal.grade, 'value') else str(signal.grade)

    logger.info(
        "[SIGNAL] %s %s | grade=%s | score=%.1f | entry=%.5f stop=%.5f tp1=%.5f",
        pair_ict, direction, grade, signal.score, entry, stop, tp1,
    )

    if not entry or not stop or not tp1:
        logger.warning("[SIGNAL] Incomplete price levels — skipping %s %s", pair_ict, direction)
        return

    if dry_run:
        logger.info("[DRY RUN] Would place: %s %s OANDA:%s", direction, pair_ict, pair_oanda)
        return

    units = bridge.compute_units(pair_oanda, entry, stop)
    if units == 0:
        logger.warning("[SIGNAL] compute_units returned 0 — skipping %s", pair_ict)
        return

    result = bridge.place_trade(
        pair=pair_oanda,
        direction=direction,
        units=units,
        stop_price=stop,
        tp1_price=tp1,
    )

    if result['status'] == 'FILLED':
        # decision_logger entry was already written by orchestrator._log_decisions()
        # We just need to track the trade for outcome closure
        entry_ts = (signal.timestamp.isoformat()
                    if hasattr(signal.timestamp, 'isoformat')
                    else str(signal.timestamp))
        tracker.record(
            trade_id       =result['trade_id'],
            pair_ict       =pair_ict,
            entry_timestamp=entry_ts,
            entry_price    =result['fill_price'],
            stop_price     =stop,
            direction      =direction,
        )
        logger.info(
            "[PLACED] %s %s %d units @ %.5f | tradeId=%s | TradingView chart updated",
            pair_oanda, direction, units, result['fill_price'], result['trade_id'],
        )
    else:
        logger.warning("[NOT PLACED] %s %s | status=%s", pair_ict, direction, result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ICT → OANDA paper trading runner',
    )
    parser.add_argument(
        '--interval', type=int, default=300,
        help='Scan interval in seconds (default: 300)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Log signals without placing trades',
    )
    args = parser.parse_args()
    run(scan_interval=args.interval, dry_run=args.dry_run)
