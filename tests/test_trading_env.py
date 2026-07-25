"""Tests for sovereign/training/trading_env.py — the self-play gym (spec §4.1).

Uses mode='provisional_gross' + for_training=False throughout (wiring tests only,
per the net-cost hard guard) except the two tests that specifically assert the
guard fires.
"""
import pytest

from sovereign.forex.exit_machine import ExitConfig
from sovereign.training.trading_env import (
    Action,
    Bar,
    TerminalReason,
    TradeRecord,
    TradingEnv,
)
from sovereign.training.value_scorer import GrossReturnError

CFG = ExitConfig(stop_atr_mult=1.5, trailing_atr_mult=2.0, strict_mode=False, enable_cb_refresh=False)


def _zero_cost(pair, direction, i):
    return 0.0


def _trade(**kw):
    base = dict(pair="EURUSD", direction=1, entry_price=1.10, stop_price=1.09, hold_limit=5)
    base.update(kw)
    return TradeRecord(**base)


def _bars(closes, atr_pct=0.006, rsi=50.0, signal=1, hold_today=5):
    return [Bar(close=c, atr_pct=atr_pct, rsi_14=rsi, signal=signal, hold_today=hold_today) for c in closes]


def test_reset_returns_8dim_state():
    env = TradingEnv(_trade(), _bars([1.10, 1.101]), CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    s = env.reset()
    assert s.shape == (8,)


def test_hold_then_exit_full_is_terminal():
    env = TradingEnv(_trade(), _bars([1.10, 1.102, 1.104]), CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    env.reset()
    next_s, r, done, info = env.step(Action.HOLD)
    assert not done
    assert info["terminal_reason"] == TerminalReason.NONE
    next_s, r, done, info = env.step(Action.EXIT_FULL)
    assert done
    assert next_s is None
    assert info["terminal_reason"] == TerminalReason.EXIT_FULL_CHOSEN


def test_stop_loss_is_terminal():
    trade = _trade(direction=1, entry_price=1.10, stop_price=1.095)
    # close drops through the stop on bar 1
    env = TradingEnv(trade, _bars([1.10, 1.080]), CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    env.reset()
    _, _, done, info = env.step(Action.HOLD)
    assert done
    assert info["terminal_reason"] == TerminalReason.STOP_LOSS


def test_hold_limit_is_terminal():
    trade = _trade(hold_limit=1)
    bars = _bars([1.10, 1.101, 1.102], hold_today=1)
    env = TradingEnv(trade, bars, CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    env.reset()
    _, _, done, info = env.step(Action.HOLD)
    assert done
    assert info["terminal_reason"] == TerminalReason.HOLD_LIMIT


def test_reward_uses_hold_cost_and_swap_cost():
    def cost_fn(pair, direction, i):
        return 0.0001
    trade = _trade()
    env = TradingEnv(trade, _bars([1.10, 1.11, 1.12]), CFG, swap_cost_fn=cost_fn, mode="provisional_gross")
    env.reset()
    _, r1, _, _ = env.step(Action.HOLD)
    expected = (1.11 - 1.10) / 1.10 - 0.0001 - 0.001 * 0
    assert r1 == pytest.approx(expected)


def test_net_cost_guard_refuses_when_gate_closed():
    """config/training.yml ships with ignition.tick_024_carry_fix_landed=false —
    mode='net' must refuse to step at all while that holds."""
    env = TradingEnv(_trade(), _bars([1.10, 1.101]), CFG, swap_cost_fn=_zero_cost, mode="net")
    env.reset()
    with pytest.raises(GrossReturnError):
        env.step(Action.HOLD, for_training=True)


def test_provisional_gross_mode_refuses_for_training():
    env = TradingEnv(_trade(), _bars([1.10, 1.101]), CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    env.reset()
    with pytest.raises(GrossReturnError):
        env.step(Action.HOLD, for_training=True)


def test_step_after_done_raises():
    env = TradingEnv(_trade(), _bars([1.10, 1.101]), CFG, swap_cost_fn=_zero_cost, mode="provisional_gross")
    env.reset()
    env.step(Action.EXIT_FULL)
    with pytest.raises(RuntimeError):
        env.step(Action.HOLD)
