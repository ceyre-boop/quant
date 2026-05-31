"""
ExecutionTracker — sovereign/execution/execution_tracker.py

Measures live execution quality: slippage (signal price vs fill price).
Called after every OANDA fill. Logs to data/execution/fills.jsonl.
After REPORT_AFTER fills, prints execution cost summary.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
FILLS_LOG = ROOT / "data" / "execution" / "fills.jsonl"
REPORT_AFTER = 20
_BACKTEST_EV = 0.40  # R per trade expected from backtest

_JPY_PAIRS = {"USDJPY", "USD_JPY", "USDJPY=X", "GBPJPY", "GBP_JPY"}


def _pip_mult(pair: str) -> float:
    return 100.0 if any(j in pair for j in ("JPY", "_JPY")) else 10000.0


def record_fill(
    pair: str,
    direction: str,
    signal_price: float,
    fill_price: float,
    stop_price: float,
    trade_id: str,
    session: str = "UNKNOWN",
) -> None:
    FILLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    mult = _pip_mult(pair)
    slippage_pips = abs(fill_price - signal_price) * mult
    stop_dist_pips = abs(fill_price - stop_price) * mult
    slippage_in_r = (slippage_pips / stop_dist_pips) if stop_dist_pips > 0 else 0.0

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "direction": direction,
        "signal_price": signal_price,
        "fill_price": fill_price,
        "slippage_pips": round(slippage_pips, 2),
        "slippage_in_r": round(slippage_in_r, 4),
        "stop_dist_pips": round(stop_dist_pips, 2),
        "session": session,
        "trade_id": trade_id,
    }
    with FILLS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("[ExecTracker] %s %s slip=%.2f pips (%.3fR)",
                pair, direction, slippage_pips, slippage_in_r)

    _maybe_report()


def _maybe_report() -> None:
    if not FILLS_LOG.exists():
        return
    fills = [json.loads(line) for line in FILLS_LOG.read_text().splitlines() if line.strip()]
    if len(fills) < REPORT_AFTER:
        return
    avg_slip_pips = sum(f["slippage_pips"] for f in fills) / len(fills)
    avg_slip_r    = sum(f["slippage_in_r"] for f in fills) / len(fills)
    live_ev = _BACKTEST_EV - avg_slip_r

    logger.info("=" * 54)
    logger.info("EXECUTION QUALITY REPORT (%d fills)", len(fills))
    logger.info("  Avg slippage:      %.2f pips", avg_slip_pips)
    logger.info("  Avg slippage R:    %.4fR", avg_slip_r)
    logger.info("  Backtest EV:       +%.2fR", _BACKTEST_EV)
    logger.info("  Estimated live EV: +%.2fR", live_ev)
    if live_ev < 0.20:
        logger.warning("EXECUTION_DEGRADATION: slippage eating >50%% of edge")
    logger.info("=" * 54)
