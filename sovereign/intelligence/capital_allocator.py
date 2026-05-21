"""
Capital Allocator — cross-system sizing throttle.

Reads:
  - RegimePerformanceTracker  (rolling Sharpe z-score in current regime)
  - SystemHealthMonitor       (health_score + reliability per system)

Outputs:
  sizing_multiplier per system  →  float [0.0, 1.0]

Mechanism (mechanistic, no ML):
  1. z_sharpe = rolling Sharpe z-score vs historical Sharpe in current regime
  2. health component from SystemHealthMonitor
  3. Hard freeze rules:
       z_sharpe < -2.5            → 0.0  (freeze)
       z_sharpe < -1.5            → 0.5× cap
       reliability == UNRELIABLE  → 0.0  (freeze)
       reliability == LOW         → 0.5× cap
  4. Freeze state persists for 48h and requires ≥5 trades to lift.

State written to: data/intelligence/capital_allocator_state.json

Usage
-----
from sovereign.intelligence.capital_allocator import CapitalAllocator

alloc = CapitalAllocator()

# After new trade + regime update:
multipliers = alloc.compute(
    current_regime="MOMENTUM",
    health_snapshots=monitor.latest_per_system(),
    regime_tracker=tracker,
)
# multipliers["ICT"]   → e.g. 0.5
# multipliers["FOREX"] → e.g. 1.0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parents[2]
_STATE_FILE = _ROOT / "data" / "intelligence" / "capital_allocator_state.json"

SYSTEMS = ("ICT", "FOREX", "EQUITY")

# Z-score thresholds for Sharpe degradation
Z_FREEZE   = -2.5   # full freeze
Z_HALF     = -1.5   # 0.5× cap

# Freeze parameters
FREEZE_HOURS  = 48
FREEZE_MIN_TRADES = 5   # must have ≥5 trades after freeze onset before restore


class AllocationState:
    """Persisted freeze/restore state per system."""

    def __init__(self, freeze_until: Optional[str] = None,
                 freeze_trade_count: int = 0,
                 trades_since_freeze: int = 0) -> None:
        self.freeze_until      = freeze_until        # ISO datetime string or None
        self.freeze_trade_count = freeze_trade_count  # total trades at freeze onset
        self.trades_since_freeze = trades_since_freeze  # trades added since

    def is_frozen(self, total_trade_count: int) -> bool:
        now = datetime.now(timezone.utc)
        if self.freeze_until:
            try:
                until = datetime.fromisoformat(self.freeze_until)
            except ValueError:
                return False
            time_ok   = now < until
            trades_ok = (total_trade_count - self.freeze_trade_count) < FREEZE_MIN_TRADES
            return time_ok or trades_ok
        return False

    def to_dict(self) -> dict:
        return {
            "freeze_until":       self.freeze_until,
            "freeze_trade_count": self.freeze_trade_count,
            "trades_since_freeze": self.trades_since_freeze,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AllocationState":
        return cls(
            freeze_until=d.get("freeze_until"),
            freeze_trade_count=d.get("freeze_trade_count", 0),
            trades_since_freeze=d.get("trades_since_freeze", 0),
        )


class CapitalAllocator:
    """
    Cross-system capital throttle.

    compute() is called after each trade close or regime update.
    It returns a dict of {system: multiplier} where multiplier ∈ [0.0, 1.0].

    The multiplier is multiplicative with the system's own position sizing.
    1.0 = full allocation, 0.5 = half size, 0.0 = frozen (no new trades).
    """

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self._state_file = Path(state_file) if state_file else _STATE_FILE
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._system_states: Dict[str, AllocationState] = {
            s: AllocationState() for s in SYSTEMS
        }
        self._load_state()

    # ── State persistence ──────────────────────────────────────────── #

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text())
            for sys in SYSTEMS:
                if sys in raw:
                    self._system_states[sys] = AllocationState.from_dict(raw[sys])
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        data = {
            sys: self._system_states[sys].to_dict() for sys in SYSTEMS
        }
        self._state_file.write_text(json.dumps(data, indent=2))

    # ── Core logic ─────────────────────────────────────────────────── #

    def compute(
        self,
        current_regime: str,
        health_snapshots: Dict[str, dict],
        regime_tracker,  # RegimePerformanceTracker (avoid circular import)
        trade_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Compute sizing multipliers for all systems.

        Args:
            current_regime:   Active regime label ("MOMENTUM" | "REVERSION" | "FLAT").
            health_snapshots: Output of SystemHealthMonitor.latest_per_system().
            regime_tracker:   RegimePerformanceTracker instance.
            trade_counts:     Total closed trades per system {system: int}.
                              Used for freeze gate evaluation.

        Returns:
            {system: multiplier}  where multiplier ∈ [0.0, 1.0].
        """
        trade_counts = trade_counts or {}
        multipliers: Dict[str, float] = {}
        decisions:   Dict[str, dict]  = {}

        for sys in SYSTEMS:
            total_trades = trade_counts.get(sys, 0)
            state        = self._system_states[sys]

            # --- A. Is the system currently frozen? ---
            if state.is_frozen(total_trades):
                multipliers[sys] = 0.0
                decisions[sys] = {"reason": "FROZEN", "multiplier": 0.0}
                logger.info(f"[Allocator] {sys}: FROZEN (persisting from prior trigger)")
                continue

            # --- B. Health gate ---
            snap = health_snapshots.get(sys, {})
            reliability = snap.get("reliability", "HIGH")
            if reliability == "UNRELIABLE":
                self._trigger_freeze(sys, total_trades)
                multipliers[sys] = 0.0
                decisions[sys] = {"reason": "HEALTH_UNRELIABLE → FREEZE", "multiplier": 0.0}
                logger.warning(f"[Allocator] {sys}: HEALTH_UNRELIABLE → freeze triggered")
                continue

            health_cap = 1.0
            if reliability == "LOW":
                health_cap = 0.5

            # --- C. Regime Sharpe z-score gate ---
            z = regime_tracker.rolling_sharpe_zscore(sys, current_regime)
            z_cap = 1.0
            if z is not None:
                if z < Z_FREEZE:
                    self._trigger_freeze(sys, total_trades)
                    multipliers[sys] = 0.0
                    decisions[sys] = {"reason": f"SHARPE_Z={z:.2f} < {Z_FREEZE} → FREEZE",
                                      "multiplier": 0.0}
                    logger.warning(
                        f"[Allocator] {sys}: z_sharpe={z:.2f} < {Z_FREEZE} → freeze"
                    )
                    continue
                elif z < Z_HALF:
                    z_cap = 0.5

            # --- D. Combine caps (most restrictive wins) ---
            multiplier = min(1.0, health_cap, z_cap)
            multipliers[sys] = round(multiplier, 4)
            decisions[sys] = {
                "reason":     (
                    f"reliability={reliability}, z_sharpe={z:.2f}" if z is not None
                    else f"reliability={reliability}, z_sharpe=insufficient_data"
                ),
                "health_cap": health_cap,
                "z_cap":      z_cap,
                "multiplier": multiplier,
            }
            logger.debug(f"[Allocator] {sys}: multiplier={multiplier:.2f}  {decisions[sys]}")

        self._save_state()
        self._log_decision(current_regime, multipliers, decisions)
        return multipliers

    def _trigger_freeze(self, system: str, trade_count: int) -> None:
        until = (datetime.now(timezone.utc) + timedelta(hours=FREEZE_HOURS)).isoformat()
        self._system_states[system] = AllocationState(
            freeze_until=until,
            freeze_trade_count=trade_count,
        )
        logger.warning(f"[Allocator] {system}: freeze active until {until}")

    def _log_decision(
        self,
        regime: str,
        multipliers: Dict[str, float],
        decisions: Dict[str, dict],
    ) -> None:
        record = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "regime":      regime,
            "multipliers": multipliers,
            "decisions":   decisions,
        }
        log_path = _ROOT / "data" / "intelligence" / "capital_allocator_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def thaw(self, system: str) -> None:
        """Manually thaw a frozen system (human override)."""
        if system.upper() in SYSTEMS:
            self._system_states[system.upper()] = AllocationState()
            self._save_state()
            logger.info(f"[Allocator] {system}: manually thawed")

    def status(self) -> dict:
        """Return current freeze state for all systems."""
        return {
            sys: {
                "frozen":        self._system_states[sys].is_frozen(0),
                "freeze_until":  self._system_states[sys].freeze_until,
            }
            for sys in SYSTEMS
        }

    def summary(
        self,
        current_regime: str,
        health_snapshots: Dict[str, dict],
        regime_tracker,
        trade_counts: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Compute allocations and return a plain-text summary.
        """
        mults = self.compute(current_regime, health_snapshots, regime_tracker, trade_counts)
        lines = [f"Capital Allocator  (regime: {current_regime})"]
        lines.append("-" * 50)
        for sys in SYSTEMS:
            m = mults.get(sys, 1.0)
            bar = "█" * int(m * 10) + "░" * (10 - int(m * 10))
            state = self._system_states[sys]
            frozen_flag = "  ❄ FROZEN" if state.is_frozen(0) else ""
            lines.append(f"  {sys:<10}  [{bar}]  {m:.0%}{frozen_flag}")
        return "\n".join(lines)


# ── Module-level singleton ─────────────────────────────────────────────── #

_allocator: Optional[CapitalAllocator] = None


def get_allocator() -> CapitalAllocator:
    global _allocator
    if _allocator is None:
        _allocator = CapitalAllocator()
    return _allocator
