"""Unified challenge ruleset engine (TICK-022 P2).

Day-clock simulation of prop-firm evaluation phases:
  drawdown models   STATIC | EOD_TRAILING | INTRADAY_TRAILING
  daily loss        measured from day-open equity, checked after every trade
  consistency       max single-day profit <= pct of total profit (blocks pass until satisfied)
  min trading days  with_trades basis (FTMO-style) or all-days basis (PropFirmRules-compat)
  phases            chained by funnel.py; this module runs ONE phase
  time caps         real time limits -> TIMEOUT; no-limit firms get a sim cap -> INCOMPLETE

Why a new engine instead of extending sovereign/propfirm/rules_engine.py:
PropFirmRules is trade-clock, single-phase, EOD-trailing-only, and is imported by the
live chain (paper_challenge -> active_challenge.json -> allocation_engine). It also
(a) never enforces its daily_loss_limit field, and (b) its dd_trail_stops_at_starting
flag caps the floor at initial - max_dd from day one, which makes its "trailing" floor
effectively STATIC (peak_eod starts == initial, so the cap always binds). Real-firm EOD
trailing locks at INITIAL balance instead. Both semantics are expressible here via
trailing_lock_mode; the parity presets mirror PropFirmRules-as-implemented, and
tests/test_rulesets.py proves step-level equivalence against the untouched class.

Intraday-trailing honesty: we only have per-trade realized PnL (no MFE/MAE), so the
intraday high-water mark is bracketed: optimistic = realized equity path only;
pessimistic = winners assumed to touch kappa x final PnL before settling (kappa knob,
default 1.25). Verdicts for intraday-trailing firms use the pessimistic bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "firms.yaml"


class DrawdownModel(str, Enum):
    STATIC = "static"
    EOD_TRAILING = "eod_trailing"
    INTRADAY_TRAILING = "intraday_trailing"


class Outcome(str, Enum):
    PASS = "PASS"
    BUST = "BUST"
    TIMEOUT = "TIMEOUT"          # real time limit hit
    INCOMPLETE = "INCOMPLETE"    # sim cap hit on a no-time-limit firm


class TrailingLock(str, Enum):
    NONE = "none"
    AT_INITIAL = "at_initial"                    # real-firm: floor stops at starting balance
    AT_INITIAL_MINUS_DD = "at_initial_minus_dd"  # PropFirmRules-compat: floor never trails
    AT_INITIAL_PLUS_USD = "at_initial_plus_usd"  # APEX-style: floor stops at initial + lock_usd


@dataclass
class PhaseSpec:
    name: str
    profit_target_usd: float
    min_trading_days: int


@dataclass
class FirmSpec:
    name: str
    account_size: float
    drawdown_model: DrawdownModel
    max_dd_usd: float
    daily_loss_usd: Optional[float]
    consistency_pct: Optional[float]
    time_limit_days: Optional[int]
    sim_cap_trading_days: int
    trailing_lock: TrailingLock
    trailing_lock_plus_usd: float
    kappa: float
    trading_day_basis: str            # "with_trades" | "all"
    phases: list[PhaseSpec] = field(default_factory=list)
    fees: dict = field(default_factory=dict)
    funded: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "FirmSpec":
        acct = float(d["account_size"])

        def _usd(frac_key: str, usd_key: str) -> Optional[float]:
            if d.get(usd_key) is not None:
                return float(d[usd_key])
            if d.get(frac_key) is not None:
                return float(d[frac_key]) * acct
            return None

        phases = []
        for p in d["phases"]:
            target = p.get("profit_target_usd")
            if target is None:
                target = float(p["profit_target"]) * acct
            phases.append(PhaseSpec(name=p["name"], profit_target_usd=float(target),
                                    min_trading_days=int(p.get("min_trading_days", 1))))

        if d.get("trailing_lock_mode"):
            lock = TrailingLock(d["trailing_lock_mode"])
        elif d.get("trailing_locks_at_initial_plus_usd") is not None:
            lock = TrailingLock.AT_INITIAL_PLUS_USD
        elif d.get("trailing_locks_at_initial"):
            lock = TrailingLock.AT_INITIAL
        else:
            lock = TrailingLock.NONE

        return cls(
            name=name,
            account_size=acct,
            drawdown_model=DrawdownModel(d["drawdown_model"]),
            max_dd_usd=_usd("max_dd", "trailing_dd_usd") or 0.0,
            daily_loss_usd=_usd("daily_loss_limit", "daily_loss_limit_usd"),
            consistency_pct=(float(d["consistency_pct"]) if d.get("consistency_pct") else None),
            time_limit_days=(int(d["time_limit_days"]) if d.get("time_limit_days") else None),
            sim_cap_trading_days=int(d.get("sim_cap_trading_days", 250)),
            trailing_lock=lock,
            trailing_lock_plus_usd=float(d.get("trailing_locks_at_initial_plus_usd") or 0.0),
            kappa=float(d.get("kappa_intraday_stress", 1.25)),
            trading_day_basis=d.get("trading_day_basis", "with_trades"),
            phases=phases,
            fees=dict(d.get("fees", {})),
            funded=dict(d.get("funded", {})),
        )

    @classmethod
    def load(cls, name: str, path: Path = CONFIG_PATH) -> "FirmSpec":
        raw = yaml.safe_load(path.read_text())
        if name not in raw:
            raise KeyError(f"firm {name!r} not in {path} — have {sorted(raw)}")
        return cls.from_dict(name, raw[name])

    @classmethod
    def load_all(cls, path: Path = CONFIG_PATH) -> dict:
        raw = yaml.safe_load(path.read_text())
        return {k: cls.from_dict(k, v) for k, v in raw.items()}


@dataclass
class PhaseResult:
    outcome: Outcome
    fail_reason: str
    trading_days: int
    days_elapsed: int             # stream days consumed (incl. no-trade days)
    n_trades: int
    equity_end: float
    day_pnls: list = field(default_factory=list)
    trades_blocked: int = 0
    trades_reduced: int = 0
    consistency_blocked_pass: bool = False


class ChallengeEngine:
    """Runs one evaluation phase over a per-day stream of R-multiple lists.

    Sizing: risk = equity * risk_pct per trade, optionally capped at
    buffer_cap_fraction of the remaining drawdown buffer (PropFirmRules-compat
    knob; None disables the cap). kappa_stress=True enables the pessimistic
    intraday-trailing bound (winners touch kappa x final PnL intraday).
    """

    def __init__(self, spec: FirmSpec, risk_pct: float,
                 buffer_cap_fraction: Optional[float] = None,
                 kappa_stress: bool = False,
                 risk_usd: Optional[float] = None):
        self.spec = spec
        self.risk_pct = float(risk_pct)
        self.buffer_cap_fraction = buffer_cap_fraction
        self.kappa_stress = kappa_stress
        self.risk_usd = risk_usd          # fixed-dollar sizing (hand tests + fixed-$ policies)

    # floor candidates never exceed the lock cap
    def _lock_cap(self) -> float:
        s = self.spec
        if s.trailing_lock is TrailingLock.AT_INITIAL:
            return s.account_size
        if s.trailing_lock is TrailingLock.AT_INITIAL_MINUS_DD:
            return s.account_size - s.max_dd_usd
        if s.trailing_lock is TrailingLock.AT_INITIAL_PLUS_USD:
            return s.account_size + s.trailing_lock_plus_usd
        return float("inf")

    def run_phase(self, phase: PhaseSpec,
                  day_r_stream: Iterable[Sequence[float]]) -> PhaseResult:
        s = self.spec
        initial = s.account_size
        dd = s.max_dd_usd
        target = initial + phase.profit_target_usd
        lock_cap = self._lock_cap()

        equity = initial
        floor = initial - dd
        peak_eod = initial
        hwm_intraday = initial
        trading_days = 0
        days_elapsed = 0
        n_trades = 0
        blocked = 0
        reduced = 0
        day_pnls: list[float] = []
        consistency_blocked = False

        for day_trades in day_r_stream:
            if days_elapsed >= s.sim_cap_trading_days:
                break
            days_elapsed += 1
            day_open = equity
            traded = False

            for r in day_trades:
                traded = True
                n_trades += 1
                risk = self.risk_usd if self.risk_usd is not None else equity * self.risk_pct
                if self.buffer_cap_fraction is not None:
                    cap = (equity - floor) * self.buffer_cap_fraction
                    if cap <= 0:
                        blocked += 1
                        risk = 0.0
                    elif risk > cap:
                        reduced += 1
                        risk = cap
                pnl = risk * float(r)

                if s.drawdown_model is DrawdownModel.INTRADAY_TRAILING:
                    touch = pnl * self.spec.kappa if (self.kappa_stress and pnl > 0) else pnl
                    hwm_intraday = max(hwm_intraday, equity + max(touch, 0.0), equity + max(pnl, 0.0))
                    floor = max(floor, min(hwm_intraday - dd, lock_cap))

                equity += pnl

                if s.daily_loss_usd is not None and (day_open - equity) >= s.daily_loss_usd:
                    day_pnls.append(equity - day_open)
                    return PhaseResult(Outcome.BUST, "daily_loss", trading_days, days_elapsed,
                                       n_trades, equity, day_pnls, blocked, reduced)

                cur_floor = (initial - dd) if s.drawdown_model is DrawdownModel.STATIC else floor
                if equity <= cur_floor:
                    day_pnls.append(equity - day_open)
                    return PhaseResult(Outcome.BUST, "max_drawdown", trading_days, days_elapsed,
                                       n_trades, equity, day_pnls, blocked, reduced)

            # ── end of day ──
            if s.drawdown_model is DrawdownModel.EOD_TRAILING:
                peak_eod = max(peak_eod, equity)
                floor = max(floor, min(peak_eod - dd, lock_cap))

            day_pnls.append(equity - day_open)
            if s.trading_day_basis == "all" or traded:
                trading_days += 1

            if equity >= target and trading_days >= phase.min_trading_days:
                if self._consistency_ok(day_pnls, equity - initial):
                    return PhaseResult(Outcome.PASS, "", trading_days, days_elapsed,
                                       n_trades, equity, day_pnls, blocked, reduced,
                                       consistency_blocked_pass=consistency_blocked)
                consistency_blocked = True    # target hit but consistency violated — keep trading

            if s.time_limit_days is not None and trading_days >= s.time_limit_days:
                return PhaseResult(Outcome.TIMEOUT, "time_limit", trading_days, days_elapsed,
                                   n_trades, equity, day_pnls, blocked, reduced,
                                   consistency_blocked_pass=consistency_blocked)

        return PhaseResult(Outcome.INCOMPLETE, "sim_cap", trading_days, days_elapsed,
                           n_trades, equity, day_pnls, blocked, reduced,
                           consistency_blocked_pass=consistency_blocked)

    def _consistency_ok(self, day_pnls: Sequence[float], total_profit: float) -> bool:
        if self.spec.consistency_pct is None or total_profit <= 0:
            return True
        best_day = max((p for p in day_pnls if p > 0), default=0.0)
        return best_day <= self.spec.consistency_pct * total_profit
