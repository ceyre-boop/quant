"""sovereign/training/trading_env.py — the self-play gym (spec §4.1).

TradingEnv wraps ONE trade: it steps bar-by-bar, returns S(t) via
state_space.build_state(), accepts one action from A = {HOLD, EXIT_FULL,
EXIT_HALF, ADD_HALF}, and returns (next_state, reward, done, info).

Terminal states are ENUMERATED, not discretionary:
  STOP_LOSS       — the frozen exit_machine's deterministic stop fired
  HOLD_LIMIT      — hold_count reached hold_limit (exit_machine TIME decision)
  EXIT_FULL       — the agent chose to close the position
  CARRY_REVERSAL  — the frozen exit_machine's carry-signal reversal fired

The env reads sovereign/forex/exit_machine.py (frozen) READ-ONLY via import — it
calls decide_exit() to get the objective bar-level signal (stop/reversal/time),
it does not modify that file or reimplement its logic.

NET-COST HARD GUARD: per-bar reward subtracts swap_cost_daily. TICK-024 (the net
carry-cost fix) is STAGED, not landed (frozen until 2026-07-28 per NEXT.md/CLAUDE.md).
`TradingEnv` refuses to emit real training rewards while the cost model in effect
is the known-bad gross one — mirrors value_scorer.GrossReturnError exactly, keyed
off the same ignition-gate check (config/training.yml ignition.tick_024_carry_fix_landed)
so there is exactly one place that decides "is this cost model net or gross."
A caller may explicitly opt into `mode="provisional_gross"` for wiring/unit tests
only; `step()` refuses that mode whenever `for_training=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

from sovereign.forex.exit_machine import (
    BarContext,
    ExitConfig,
    ExitDecision,
    PositionState,
    decide_exit,
)
from sovereign.training.gate import evaluate_gate
from sovereign.training.state_space import (
    build_state,
    carry_alignment as _carry_alignment,
    drawdown_from_peak as _drawdown_from_peak,
)
from sovereign.training.value_scorer import GrossReturnError


class Action(IntEnum):
    HOLD = 0
    EXIT_FULL = 1
    EXIT_HALF = 2
    ADD_HALF = 3


class TerminalReason(IntEnum):
    NONE = 0
    STOP_LOSS = 1
    HOLD_LIMIT = 2
    EXIT_FULL_CHOSEN = 3
    CARRY_REVERSAL = 4


# exit_machine can emit decisions this spec doesn't separately enumerate. They are
# folded into the nearest matching terminal bucket rather than left undocumented:
#   TRAILING_ATR — a stop-type exit (trailing stop hit)              -> STOP_LOSS
#   CB_REFRESH   — carries a re-entry signal but does NOT close       -> not terminal
#   DONCHIAN     — strict_mode only; unreached under live v015 (strict_mode=False)
#                  kept for parity, bucketed defensively               -> HOLD_LIMIT
_DECISION_TO_TERMINAL = {
    ExitDecision.INITIAL_STOP: TerminalReason.STOP_LOSS,
    ExitDecision.TRAILING_ATR: TerminalReason.STOP_LOSS,
    ExitDecision.REVERSAL: TerminalReason.CARRY_REVERSAL,
    ExitDecision.TIME: TerminalReason.HOLD_LIMIT,
    ExitDecision.DONCHIAN: TerminalReason.HOLD_LIMIT,
}

HOLD_COST_PER_BAR = 0.001  # -0.001 * hold_count term in r(t), per spec §4.1


@dataclass
class Bar:
    """One bar's market data. rate_diff_z/cot_z optional (spec's S(t) treats them
    as 0.0 when unavailable, per state_space.py)."""
    close: float
    atr_pct: float
    rsi_14: float
    signal: int             # carry signal this bar (-1/0/+1), feeds decide_exit + carry_alignment
    hold_today: int         # this signal's hold_days, feeds decide_exit TIME check
    rate_diff_z: Optional[float] = None
    cot_z: Optional[float] = None


@dataclass
class TradeRecord:
    pair: str
    direction: int          # +1 long / -1 short
    entry_price: float
    stop_price: float
    hold_limit: int


SwapCostFn = Callable[[str, int, int], float]  # (pair, direction, bar_index) -> daily cost (fraction)


def _refuse_gross_swap_cost(*_a, **_k) -> float:
    raise GrossReturnError(
        "TradingEnv: no swap_cost_fn provided and no default gross-safe cost model "
        "exists here. Pass an explicit swap_cost_fn (net, post-TICK-024) or run in "
        "mode='provisional_gross' with for_training=False for wiring tests only."
    )


class TradingEnv:
    """Gym-style env over ONE trade. `bars` is the sequence of Bar objects starting
    at the entry bar (index 0 = entry). `exit_cfg` is the SAME ExitConfig the live
    exit_machine uses — the env never invents its own stop/reversal/time logic."""

    def __init__(
        self,
        trade: TradeRecord,
        bars: list[Bar],
        exit_cfg: ExitConfig,
        *,
        swap_cost_fn: Optional[SwapCostFn] = None,
        mode: str = "net",
        config_path=None,
    ):
        if mode not in ("net", "provisional_gross"):
            raise ValueError(f"TradingEnv: unknown mode {mode!r}")
        self.trade = trade
        self.bars = bars
        self.exit_cfg = exit_cfg
        self.swap_cost_fn = swap_cost_fn or _refuse_gross_swap_cost
        self.mode = mode
        self._gate_checks = evaluate_gate(config_path).checks
        self._i = 0
        self._done = False
        self._position_frac = 1.0  # 1.0 full, 0.5 after EXIT_HALF, 1.5 after ADD_HALF
        self._pos_state: Optional[PositionState] = None

    # -- net-cost guard -----------------------------------------------------
    def _net_cost_confirmed(self) -> bool:
        return bool(self._gate_checks.get("tick_024_carry_fix_landed", False))

    def _require_net(self, for_training: bool) -> None:
        if self.mode == "provisional_gross":
            if for_training:
                raise GrossReturnError(
                    "TradingEnv refused: mode='provisional_gross' may not be used "
                    "for_training=True. Wiring tests only."
                )
            return  # provisional/gross wiring path — explicitly not for training
        if not self._net_cost_confirmed():
            raise GrossReturnError(
                "TradingEnv refused: ignition.tick_024_carry_fix_landed=false — "
                "the cost model in effect is the known-bad gross one. Refusing to "
                "emit training rewards until TICK-024 lands (see config/training.yml)."
            )

    # -- gym interface --------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._i = 0
        self._done = False
        self._position_frac = 1.0
        entry = self.trade.entry_price
        self._pos_state = PositionState(
            direction=self.trade.direction,
            stop_price=self.trade.stop_price,
            best_price=entry,
            worst_price=entry,
            hold_count=0,
            hold_limit=self.trade.hold_limit,
        )
        return self._state_at(self._i)

    def _record_at(self, i: int) -> dict:
        bar = self.bars[i]
        st = self._pos_state
        entry = self.trade.entry_price
        direction = self.trade.direction
        excursion_pct = direction * (bar.close - entry) / entry
        return {
            "hold_count": st.hold_count,
            "hold_limit": st.hold_limit,
            "excursion_pct": excursion_pct,
            "atr_pct": bar.atr_pct,
            "rsi_14": bar.rsi_14,
            "carry_alignment": _carry_alignment(direction, bar.signal),
            "rate_diff_z": bar.rate_diff_z,
            "cot_z": bar.cot_z,
            "drawdown_from_peak": _drawdown_from_peak(direction, st.best_price, bar.close),
        }

    def _state_at(self, i: int) -> np.ndarray:
        return build_state(self._record_at(i))

    def step(self, action: Action, *, for_training: bool = False):
        """Returns (next_state, reward, done, info). info['terminal_reason'] is a
        TerminalReason (NONE while not done)."""
        if self._done:
            raise RuntimeError("TradingEnv.step called after episode terminated; call reset()")
        self._require_net(for_training)

        if self._i + 1 >= len(self.bars):
            # Data exhausted before any enumerated terminal fired.
            self._done = True
            return None, 0.0, True, {"terminal_reason": TerminalReason.NONE,
                                      "position_frac": self._position_frac}

        prev_close = self.bars[self._i].close
        next_i = self._i + 1
        bar = self.bars[next_i]
        direction = self.trade.direction
        entry = self.trade.entry_price

        if action == Action.EXIT_HALF:
            self._position_frac = max(0.0, self._position_frac - 0.5)
        elif action == Action.ADD_HALF:
            self._position_frac += 0.5

        daily_pnl_frac = direction * (bar.close - prev_close) / entry * self._position_frac
        swap_cost_daily = self.swap_cost_fn(self.trade.pair, direction, next_i)
        reward = daily_pnl_frac - swap_cost_daily - HOLD_COST_PER_BAR * self._pos_state.hold_count

        terminal_reason = TerminalReason.NONE
        if action == Action.EXIT_FULL:
            terminal_reason = TerminalReason.EXIT_FULL_CHOSEN
        else:
            ctx = BarContext(
                close=bar.close, atr_pct=bar.atr_pct, signal=bar.signal,
                hold_today=bar.hold_today, donchian_exit_low=float("nan"),
            )
            res = decide_exit(self._pos_state, ctx, self.exit_cfg)
            self._pos_state = res.state
            if res.decision != ExitDecision.HOLD:
                terminal_reason = _DECISION_TO_TERMINAL.get(res.decision, TerminalReason.HOLD_LIMIT)
                if res.decision == ExitDecision.CB_REFRESH:
                    terminal_reason = TerminalReason.NONE  # re-entry signal only, not a close

        done = terminal_reason != TerminalReason.NONE or next_i >= len(self.bars) - 1
        info = {"terminal_reason": terminal_reason, "position_frac": self._position_frac}

        if done:
            self._done = True
            return None, reward, True, info

        self._i = next_i
        return self._state_at(self._i), reward, False, info
