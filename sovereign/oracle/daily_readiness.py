"""
DailyReadiness — sovereign/oracle/daily_readiness.py

Portfolio-level gate. Runs once at session open before any scanner fires.
Returns TRADE / REDUCE / SIT — no per-trade logic, only portfolio state.

SIT   = no new trades, manage open only
REDUCE = half size, max 1 new position
TRADE  = full size, all scanners run
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
HEALTH_PATH      = ROOT / "data" / "agent" / "health.json"

CRITICAL_SYSTEMS = ["ict_scanner", "backtest_engine", "oanda"]


@dataclass
class ReadinessResult:
    status: str   # TRADE | REDUCE | SIT
    reason: str


class DailyReadiness:
    def __init__(self, bridge=None) -> None:
        self._bridge = bridge  # lazy-loaded if None

    def assess(self) -> ReadinessResult:
        bridge = self._get_bridge()

        # Gate 1: Drawdown — hard stop at 4%, reduce at 3%
        dd = self._get_drawdown(bridge)
        if dd >= 0.04:
            return ReadinessResult("SIT", f"DD {dd:.1%}: capital preservation mode")
        if dd >= 0.03:
            return ReadinessResult("REDUCE", f"DD {dd:.1%}: half size, max 1 trade")

        # Gate 2: System health — SIT if any critical component is RED
        try:
            health = json.loads(HEALTH_PATH.read_text())
            components = health.get("components", {})
            for sys_name in CRITICAL_SYSTEMS:
                status = components.get(sys_name, {}).get("status", "UNKNOWN")
                if status not in ("GREEN", "YELLOW"):
                    return ReadinessResult("SIT", f"{sys_name} degraded ({status}): fix before trading")
        except Exception as exc:
            logger.warning("DailyReadiness: health check failed: %s", exc)

        # Gate 3: Portfolio heat — sum |unrealizedPL| as proxy for deployed risk
        total_heat = self._get_portfolio_heat(bridge)
        if total_heat >= 0.02:
            return ReadinessResult("SIT", f"Portfolio heat {total_heat:.1%}: no new exposure")
        if total_heat >= 0.015:
            return ReadinessResult("REDUCE", f"Portfolio heat {total_heat:.1%}: one more trade max")

        # Gate 4: Losing streak — REDUCE after 5+ trades at ≤20% WR
        try:
            last_10 = bridge.get_closed_trades(limit=10)
            if len(last_10) >= 5:
                recent_wr = sum(1 for t in last_10 if float(t.get("realizedPL", 0)) > 0) / len(last_10)
                if recent_wr <= 0.20:
                    return ReadinessResult("REDUCE", f"Recent WR {recent_wr:.0%}: losing streak protocol")
        except Exception as exc:
            logger.warning("DailyReadiness: recent trades check failed: %s", exc)

        # Gate 5: Regime clarity — REDUCE when ambiguous
        try:
            from sovereign.intelligence.regime_confidence import score_regime_confidence
            rc = score_regime_confidence()
            if rc.confidence < 0.35:
                return ReadinessResult("REDUCE", f"Regime confidence {rc.confidence:.0%}: ambiguous")
        except Exception as exc:
            logger.warning("DailyReadiness: regime check failed: %s", exc)

        return ReadinessResult("TRADE", "All systems nominal")

    # ── private helpers ──────────────────────────────────────────────────────

    def _get_bridge(self):
        if self._bridge is None:
            from sovereign.execution.oanda_bridge import OandaBridge
            self._bridge = OandaBridge()
        return self._bridge

    def _get_drawdown(self, bridge) -> float:
        try:
            nav = bridge.get_account_balance()
            peak = nav
            if EQUITY_PEAK_PATH.exists():
                peak = json.loads(EQUITY_PEAK_PATH.read_text()).get("peak", nav)
            return max(0.0, (peak - nav) / peak) if peak > 0 else 0.0
        except Exception as exc:
            logger.debug("DailyReadiness: drawdown calc failed: %s", exc)
            return 0.0

    def _get_portfolio_heat(self, bridge) -> float:
        try:
            open_trades = bridge.get_open_trades()
            nav = bridge.get_account_balance()
            if nav <= 0 or not open_trades:
                return 0.0
            heat = sum(abs(float(t.get("unrealizedPL", 0))) for t in open_trades)
            return heat / nav
        except Exception as exc:
            logger.debug("DailyReadiness: portfolio heat calc failed: %s", exc)
            return 0.0
