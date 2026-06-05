"""Risk engine data structures — built FIRST. Pure data, no logic.

Signal   = the trade being sized (carries grade, stop, strategy).
Position = an open position (for correlated worst-case math).
RiskState = the full world the engine sizes against (equity, drawdown, regime, edge stats…).
RiskDecision = the engine's output (final size/risk, binding constraint, full layer audit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Signal:
    """The trade to be sized. The engine NEVER generates this — it only sizes it."""
    instrument: str
    direction: int            # +1 long / -1 short
    entry: float
    stop: float               # worst-case exit price
    grade: str                # 'A+' | 'A' | 'B' | 'C'
    strategy: str = "forex_macro"
    point_value: float = 1.0  # account-currency value per 1 price unit per 1 size unit
    notes: dict = field(default_factory=dict)  # caller-supplied tags (e.g. below_proven_bar)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Position:
    instrument: str
    direction: int            # +1 long / -1 short
    size: float               # units/lots/contracts
    entry: float
    stop: float               # worst-case exit price
    risk_pct_at_entry: float


@dataclass
class RiskState:
    equity: float
    peak_equity: float                      # for trailing drawdown
    starting_balance: float
    daily_realized_pnl: float
    daily_open_pnl: float
    open_positions: list = field(default_factory=list)   # list[Position]
    drawdown_static: float = 0.0            # (start - equity)/start, >=0
    drawdown_trailing: float = 0.0          # (peak - equity)/peak, >=0
    regime: str = "UNKNOWN"
    threat_score: float = 0.0               # 0..1, higher = worse
    vol_estimates: dict = field(default_factory=dict)    # instrument -> current vol
    vol_baseline: dict = field(default_factory=dict)     # instrument -> target/long-run vol
    edge_stats: dict = field(default_factory=dict)       # strategy -> {win_rate, payoff, n_trades}
    correlation_matrix: dict = field(default_factory=dict)  # {(a,b): rho}
    mc_breach_prob: Optional[float] = None  # from Monte Carlo simulator
    health_ok: bool = True
    timestamp: str = field(default_factory=_now)

    @staticmethod
    def derive_drawdowns(equity, peak_equity, starting_balance):
        """Helper to compute the two drawdown fractions consistently (both >= 0)."""
        dd_static = max(0.0, (starting_balance - equity) / starting_balance) if starting_balance else 0.0
        dd_trailing = max(0.0, (peak_equity - equity) / peak_equity) if peak_equity else 0.0
        return dd_static, dd_trailing


@dataclass
class RiskDecision:
    final_size: float                 # tradable units (after stop conversion, rounded DOWN)
    final_risk_pct: float
    base_risk_pct: float
    binding_constraint: str           # layer that bound, or "halt:<reason>"
    layer_budgets: dict               # every layer's factor/ceiling
    modulators: dict                  # {vol, drawdown, regime}
    halt: bool
    halt_reason: Optional[str]
    reasoning: str
    instrument: str
    strategy: str = "forex_macro"
    timestamp: str = field(default_factory=_now)
