"""
ict/micro_risk.py
=================
Micro-risk engine for the ICT retail subsystem.

ISOLATION RULE
--------------
This module is COMPLETELY ISOLATED from the Sovereign risk engine.
Do NOT import from:  sovereign/, layer2/risk_engine.py, config/parameters.yml

It has its own:
  • config section (ict_params.yml → micro_risk)
  • risk limits (tighter, leverage-aware, small-account-only)
  • sizing formula (fixed fractional, NOT Kelly — Kelly is for Sovereign)
  • position namespace (separate from institutional book)

Design:
  • Fixed-fractional: risk_dollars = account_size × max_risk_per_trade
  • Position size  = risk_dollars / (|entry - stop_loss| × pip_value)
  • Hard caps on concurrent positions, daily loss, leverage
  • Produces PositionSizing (approved) or RiskVeto (rejected with reason)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Defaults (mirrors ict_params.yml) ─────────────────────────────────────── #

_MAX_RISK_PER_TRADE = 0.02     # 2 %
_MAX_CONCURRENT_RISK = 0.06    # 6 %
_MAX_POSITIONS = 3
_MAX_DAILY_LOSS_PCT = 0.05     # 5 %
_MAX_LEVERAGE = 30
_MIN_RR = 2.0
_STOP_ATR_MULT = 1.0
_TP1_R = 2.0
_TP2_R = 4.0
_TP1_SIZE_FRAC = 0.5


# ── Contracts ─────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class MicroRiskParams:
    """Live state snapshot passed in by the caller (from a paper-trade tracker)."""
    account_size: float             # current paper account equity
    open_positions: int = 0         # number of currently open ICT positions
    open_risk_pct: float = 0.0      # total unrealised risk as fraction of account
    daily_loss_pct: float = 0.0     # realised loss today as fraction of account


@dataclass(frozen=True)
class PositionSizing:
    """Approved trade sizing output."""
    direction: str                  # 'LONG' | 'SHORT'
    entry_price: float
    stop_loss: float
    tp1: float                      # first take-profit level
    tp2: float                      # second take-profit level
    risk_pct: float                 # fraction of account being risked
    risk_dollars: float
    position_units: float           # notional units (lots, shares, etc.)
    leverage_used: float            # implicit leverage = notional / account
    rr_ratio: float
    tp1_units: float                # units to close at TP1
    tp2_units: float                # units to hold toward TP2


@dataclass(frozen=True)
class RiskVeto:
    """A rejected trade with the blocking reason."""
    reason: str
    detail: str


# ── Engine ────────────────────────────────────────────────────────────────── #

class MicroRiskEngine:
    """
    Compute position sizing for an ICT paper trade, or veto it.

    Usage::

        engine = MicroRiskEngine()
        params = MicroRiskParams(account_size=10_000, open_positions=1, open_risk_pct=0.02)
        result = engine.size(
            direction='LONG',
            entry=1.0850,
            stop_loss=1.0820,
            atr=0.0045,
            params=params,
        )
        if isinstance(result, PositionSizing):
            print(result.position_units, result.risk_dollars)
        else:
            print("Vetoed:", result.reason)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        cfg = self._load_config(config_path)
        self.max_risk_per_trade: float = cfg.get("max_risk_per_trade", _MAX_RISK_PER_TRADE)
        self.max_concurrent_risk: float = cfg.get("max_concurrent_risk", _MAX_CONCURRENT_RISK)
        self.max_positions: int = cfg.get("max_positions", _MAX_POSITIONS)
        self.max_daily_loss_pct: float = cfg.get("max_daily_loss_pct", _MAX_DAILY_LOSS_PCT)
        self.max_leverage: float = cfg.get("max_leverage", _MAX_LEVERAGE)
        self.min_rr: float = cfg.get("min_rr", _MIN_RR)
        self.stop_atr_mult: float = cfg.get("stop_atr_multiple", _STOP_ATR_MULT)
        self.tp1_r: float = cfg.get("tp1_r", _TP1_R)
        self.tp2_r: float = cfg.get("tp2_r", _TP2_R)
        self.tp1_size_frac: float = cfg.get("tp1_size_fraction", _TP1_SIZE_FRAC)

    # ── Public API ─────────────────────────────────────────────────────── #

    def size(
        self,
        direction: str,
        entry: float,
        stop_loss: float,
        atr: float,
        params: MicroRiskParams,
        pip_value: float = 1.0,
    ) -> "PositionSizing | RiskVeto":
        """
        Compute sizing for a proposed trade.

        Args:
            direction: 'LONG' or 'SHORT'
            entry: proposed entry price
            stop_loss: hard stop-loss price (placed by caller)
            atr: current ATR of the instrument (same price units as entry/stop)
            params: live account state
            pip_value: value of 1 unit × 1 price-unit move in account currency.
                       For 1 standard forex lot: pip_value = 10 per pip.
                       Pass 1.0 if working in dimensionless units.

        Returns:
            PositionSizing if approved, RiskVeto if blocked.
        """
        # ── Gate 1: daily loss limit ─────────────────────────────────────
        if params.daily_loss_pct >= self.max_daily_loss_pct:
            return RiskVeto(
                reason="DAILY_LOSS_LIMIT",
                detail=f"Daily loss {params.daily_loss_pct:.1%} ≥ limit {self.max_daily_loss_pct:.1%}. No more trades today.",
            )

        # ── Gate 2: max concurrent positions ────────────────────────────
        if params.open_positions >= self.max_positions:
            return RiskVeto(
                reason="MAX_POSITIONS",
                detail=f"Already {params.open_positions} open positions (max {self.max_positions}).",
            )

        # ── Gate 3: max concurrent portfolio risk ─────────────────────
        if params.open_risk_pct + self.max_risk_per_trade > self.max_concurrent_risk:
            return RiskVeto(
                reason="MAX_CONCURRENT_RISK",
                detail=(
                    f"Adding {self.max_risk_per_trade:.1%} would bring total risk to "
                    f"{params.open_risk_pct + self.max_risk_per_trade:.1%} "
                    f"(max {self.max_concurrent_risk:.1%})."
                ),
            )

        # ── Gate 4: valid stop-loss placement ───────────────────────────
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return RiskVeto(
                reason="INVALID_STOP",
                detail=f"Stop distance is zero — entry={entry}, stop={stop_loss}.",
            )

        direction_sign = 1 if direction == "LONG" else -1
        if direction == "LONG" and stop_loss >= entry:
            return RiskVeto(
                reason="INVALID_STOP_DIRECTION",
                detail=f"LONG stop {stop_loss} must be below entry {entry}.",
            )
        if direction == "SHORT" and stop_loss <= entry:
            return RiskVeto(
                reason="INVALID_STOP_DIRECTION",
                detail=f"SHORT stop {stop_loss} must be above entry {entry}.",
            )

        # ── Compute take-profit levels ───────────────────────────────────
        tp1 = entry + direction_sign * stop_distance * self.tp1_r
        tp2 = entry + direction_sign * stop_distance * self.tp2_r
        rr_ratio = self.tp1_r  # R:R at first target

        # ── Gate 5: minimum R:R ─────────────────────────────────────────
        if rr_ratio < self.min_rr:
            return RiskVeto(
                reason="MIN_RR",
                detail=f"R:R {rr_ratio:.1f} < minimum {self.min_rr:.1f}.",
            )

        # ── Compute position size ────────────────────────────────────────
        risk_dollars = params.account_size * self.max_risk_per_trade
        position_units = risk_dollars / (stop_distance * pip_value) if pip_value > 0 else 0.0

        # ── Gate 6: leverage cap ─────────────────────────────────────────
        # Notional value of the position in account currency = units × entry price.
        # pip_value is *not* included here because it only scales P&L per unit move,
        # not the face value of the position itself.
        notional = position_units * entry
        leverage = notional / params.account_size if params.account_size > 0 else 0.0
        if leverage > self.max_leverage:
            # Scale down to stay within leverage cap
            position_units = (params.account_size * self.max_leverage) / entry
            risk_dollars = position_units * stop_distance * pip_value
            leverage = self.max_leverage

        tp1_units = position_units * self.tp1_size_frac
        tp2_units = position_units - tp1_units

        return PositionSizing(
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            risk_pct=risk_dollars / params.account_size,
            risk_dollars=round(risk_dollars, 2),
            position_units=round(position_units, 6),
            leverage_used=round(leverage, 2),
            rr_ratio=rr_ratio,
            tp1_units=round(tp1_units, 6),
            tp2_units=round(tp2_units, 6),
        )

    def suggest_stop(self, entry: float, direction: str, atr: float) -> float:
        """
        Suggest an ATR-based stop-loss level (default ICT micro-risk style).

        Caller may override with structural stop; this is just a default.
        """
        mult = self.stop_atr_mult
        if direction == "LONG":
            return entry - mult * atr
        return entry + mult * atr

    # ── Private helpers ────────────────────────────────────────────────── #

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        path = config_path or _default_config_path()
        try:
            with open(path) as f:
                full = yaml.safe_load(f)
            return full.get("micro_risk", {})
        except FileNotFoundError:
            logger.warning("ICT config not found at %s — using defaults", path)
            return {}


def _default_config_path() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config", "ict_params.yml")
