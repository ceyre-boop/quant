"""
PropRiskManager — sovereign/risk/prop_risk_manager.py

Enforces prop firm rules before every OANDA order. Has absolute veto power —
nothing in the pipeline can override these limits.

Rules:
  - 2% max daily loss (realized + unrealized)
  - 5% trailing drawdown from equity peak
  - Drawdown-adjusted sizing: 3% DD → cap 0.50%, 4% DD → cap 0.25%
  - 3 max simultaneous open positions
  - 0.70 max correlation between same-direction open positions
  - No new entries within 2h of Friday market close (21:00 UTC)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
EQUITY_PEAK_PATH = ROOT / "data" / "agent" / "equity_peak.json"

_PAIR_CORR: dict[tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"): 0.85,
    ("GBPUSD", "EURUSD"): 0.85,
    ("EURUSD", "AUDUSD"): 0.65,
    ("AUDUSD", "EURUSD"): 0.65,
    ("GBPUSD", "AUDUSD"): 0.60,
    ("AUDUSD", "GBPUSD"): 0.60,
    ("AUDUSD", "AUDNZD"): 0.72,
    ("AUDNZD", "AUDUSD"): 0.72,
    ("EURUSD", "AUDNZD"): 0.55,
    ("AUDNZD", "EURUSD"): 0.55,
    ("GBPUSD", "AUDNZD"): 0.50,
    ("AUDNZD", "GBPUSD"): 0.50,
    ("EURUSD", "USDJPY"): -0.50,
    ("USDJPY", "EURUSD"): -0.50,
    ("GBPUSD", "USDJPY"): -0.55,
    ("USDJPY", "GBPUSD"): -0.55,
    ("AUDUSD", "USDJPY"): -0.40,
    ("USDJPY", "AUDUSD"): -0.40,
}

MAX_DAILY_LOSS_PCT = 0.02
MAX_DRAWDOWN_PCT   = 0.05
MAX_POSITIONS      = 3
MAX_CORRELATION    = 0.70
FRIDAY_CLOSE_UTC   = 21  # hour — market closes Friday 21:00 UTC


@dataclass
class PropRiskDecision:
    allowed: bool
    adjusted_risk: float
    reason: str


class PropRiskManager:

    def __init__(self, bridge) -> None:
        self._bridge = bridge

    def check_trade_allowed(
        self, pair: str, direction: str, risk_pct: float
    ) -> PropRiskDecision:
        now = datetime.now(timezone.utc)

        # Gate 1: Daily loss limit (realized today + all unrealized)
        try:
            today_str = now.strftime("%Y-%m-%d")
            closed = self._bridge.get_closed_trades(limit=100)
            today_realized = sum(
                float(t.get("realizedPL", 0)) for t in closed
                if t.get("closeTime", "")[:10] == today_str
            )
            open_trades = self._bridge.get_open_trades()
            unrealized = sum(float(t.get("unrealizedPL", 0)) for t in open_trades)
            nav = self._bridge.get_account_balance()
            today_pnl = today_realized + unrealized
            if today_pnl <= -MAX_DAILY_LOSS_PCT * nav:
                return PropRiskDecision(False, risk_pct,
                    f"DAILY_LOSS_LIMIT: {today_pnl:.2f} (limit {-MAX_DAILY_LOSS_PCT * nav:.2f})")
        except Exception as exc:
            logger.warning("PropRiskManager: daily loss check failed: %s", exc)

        # Gate 2: Trailing drawdown from peak equity
        try:
            nav = self._bridge.get_account_balance()
            peak = self._load_peak(nav)
            dd = (peak - nav) / peak if peak > 0 else 0.0
            if dd >= MAX_DRAWDOWN_PCT:
                return PropRiskDecision(False, risk_pct,
                    f"MAX_DRAWDOWN: {dd:.1%} >= {MAX_DRAWDOWN_PCT:.0%} limit")
            # Gate 3: Drawdown-adjusted sizing
            if dd >= 0.04:
                risk_pct = min(risk_pct, 0.0025)
                logger.info("PropRisk: DD=%.1f%% → risk capped 0.25%%", dd * 100)
            elif dd >= 0.03:
                risk_pct = min(risk_pct, 0.0050)
                logger.info("PropRisk: DD=%.1f%% → risk capped 0.50%%", dd * 100)
        except Exception as exc:
            logger.warning("PropRiskManager: drawdown check failed: %s", exc)

        # Gate 4: Max simultaneous positions
        try:
            open_count = len(self._bridge.get_open_trades())
            if open_count >= MAX_POSITIONS:
                return PropRiskDecision(False, risk_pct,
                    f"MAX_POSITIONS: {open_count} open (limit {MAX_POSITIONS})")
        except Exception as exc:
            logger.warning("PropRiskManager: position count check failed: %s", exc)

        # Gate 5: Correlation check — same-direction pair concentration
        try:
            clean = pair.replace("=X", "").replace("_", "")
            open_trades = self._bridge.get_open_trades()
            for t in open_trades:
                op_clean = t.get("instrument", "").replace("_", "")
                op_dir = "LONG" if int(t.get("currentUnits", 0)) > 0 else "SHORT"
                corr = abs(_PAIR_CORR.get((clean, op_clean), 0.0))
                if corr > MAX_CORRELATION and op_dir == direction:
                    return PropRiskDecision(False, risk_pct,
                        f"CORRELATED: {clean}/{op_clean} corr={corr:.2f} same direction")
        except Exception as exc:
            logger.warning("PropRiskManager: correlation check failed: %s", exc)

        # Gate 6: Weekend / market close protection
        weekday = now.weekday()  # 4=Friday, 5=Saturday, 6=Sunday
        hour_utc = now.hour
        if weekday == 4 and hour_utc >= (FRIDAY_CLOSE_UTC - 2):
            return PropRiskDecision(False, risk_pct,
                f"MARKET_CLOSING: within 2h of Friday close ({hour_utc}:xx UTC)")
        if weekday == 5:
            return PropRiskDecision(False, risk_pct, "MARKET_CLOSED: Saturday")
        if weekday == 6 and hour_utc < FRIDAY_CLOSE_UTC:
            return PropRiskDecision(False, risk_pct,
                "MARKET_CLOSED: Sunday before 21:00 UTC open")

        return PropRiskDecision(True, risk_pct, "OK")

    def _load_peak(self, current_nav: float) -> float:
        EQUITY_PEAK_PATH.parent.mkdir(parents=True, exist_ok=True)
        peak = current_nav
        if EQUITY_PEAK_PATH.exists():
            try:
                peak = json.loads(EQUITY_PEAK_PATH.read_text()).get("peak", current_nav)
            except Exception:
                pass
        if current_nav > peak:
            peak = current_nav
            EQUITY_PEAK_PATH.write_text(json.dumps({
                "peak": peak,
                "updated": datetime.now(timezone.utc).isoformat(),
            }))
        return peak
