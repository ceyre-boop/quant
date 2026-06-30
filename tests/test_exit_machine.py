"""tests/test_exit_machine.py — L2 Step 1 regression: the shared decide_exit refactor.

Proves (1) the refactored _simulate_forex_core is byte-identical to the pre-refactor code on a
comprehensive golden fixture, and (2) decide_exit fires each exit state correctly, deterministically,
and preserves the re-entry trap (a bar that is stop_hit AND reversal records STOP yet still re-enters).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sovereign.forex.exit_machine import (
    BarContext, ExitConfig, ExitDecision, PositionState, decide_exit, decide_exit_vec,
)
from sovereign.forex.fast_backtester import _simulate_forex_core

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "exit_machine_golden.npz"
CFG = ExitConfig(stop_atr_mult=2.0, trailing_atr_mult=1.25, strict_mode=False, enable_cb_refresh=True)


# ─── load-bearing: byte-identical parity vs the pre-refactor golden ─────────────

def test_golden_byte_identical():
    if not FIXTURE.exists():
        pytest.skip("golden fixture not captured")
    g = np.load(FIXTURE)
    out = _simulate_forex_core(
        g["opens"], g["closes"], g["signals"], g["hold_days"], 0.04, g["atr_pcts"],
        2.0, 1.25, g["donchian_lows"], False, False, 0.5, 4, True,
    )
    entries, exits, dirs, pnls, holds, reasons, units = out
    np.testing.assert_array_equal(entries, g["g_entries"])
    np.testing.assert_array_equal(exits, g["g_exits"])
    np.testing.assert_array_equal(dirs, g["g_dirs"])
    np.testing.assert_array_equal(pnls, g["g_pnls"])      # exact float equality — same arithmetic
    np.testing.assert_array_equal(holds, g["g_holds"])
    np.testing.assert_array_equal(reasons, g["g_reasons"])
    np.testing.assert_array_equal(units, g["g_units"])
    # the golden must exercise the live-active exit types (not a trivial fixture)
    assert set(np.unique(g["g_reasons"])) >= {1, 2, 3, 4, 5}  # STOP/REVERSAL/CB_REFRESH/TIME/TRAILING


# ─── per-state unit tests (direct decide_exit) ──────────────────────────────────

def _long(stop=0.90, best=1.05, worst=1.0, hold=5, limit=60):
    return PositionState(1, stop, best, worst, hold, limit)


def test_initial_stop():
    r = decide_exit(_long(stop=1.00), BarContext(0.99, 0.01, 1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.INITIAL_STOP


def test_trailing_atr():
    # close below the ATR-trail off best (1.05), but above the far stop (0.90)
    r = decide_exit(_long(stop=0.90, best=1.05), BarContext(1.02, 0.02, 1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.TRAILING_ATR
    assert r.state.best_price == 1.05            # best preserved (1.02 < 1.05)


def test_reversal_and_reentry_signal():
    # best == close so the ATR trail does not fire first (isolates the reversal path)
    r = decide_exit(_long(best=1.01), BarContext(1.01, 0.01, -1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.REVERSAL
    assert r.reentry_signal == -1


def test_time_exit():
    # same-direction signal, hold_today=60 (>=30 so cb_refresh cannot fire), hold reaches limit
    r = decide_exit(_long(best=1.01, hold=59, limit=60), BarContext(1.01, 0.01, 1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.TIME


def test_cb_refresh_and_reentry():
    # same-direction, hold_today<30, hold_count reaches 20
    r = decide_exit(_long(best=1.01, hold=19, limit=60), BarContext(1.01, 0.01, 1, 25, math.nan), CFG)
    assert r.decision == ExitDecision.CB_REFRESH
    assert r.reentry_signal == 1


def test_donchian_strict_only():
    strict = ExitConfig(2.0, 1.25, strict_mode=True, enable_cb_refresh=False)
    # below the Donchian low, but above the stop and not trailing
    r = decide_exit(_long(stop=0.90, best=1.0), BarContext(1.04, 0.001, 1, 60, 1.05), strict)
    assert r.decision == ExitDecision.DONCHIAN
    # and the SAME bar under non-strict config must NOT fire Donchian
    r2 = decide_exit(_long(stop=0.90, best=1.0), BarContext(1.04, 0.001, 1, 60, 1.05), CFG)
    assert r2.decision != ExitDecision.DONCHIAN


def test_hold():
    r = decide_exit(_long(stop=0.90, best=1.0), BarContext(1.005, 0.001, 1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.HOLD
    assert r.reentry_signal == 0


# ─── THE TRAP: stop_hit AND reversal → reason STOP but STILL re-enters ──────────

def test_stop_and_reversal_trap():
    """A bar that is both stop_hit and a reversal records STOP (priority) but the original re-enters
    on (reversal or cb_refresh) regardless. reentry_signal must be the flipped signal."""
    r = decide_exit(_long(stop=1.00), BarContext(0.99, 0.01, -1, 60, math.nan), CFG)
    assert r.decision == ExitDecision.INITIAL_STOP   # stop wins the priority
    assert r.reentry_signal == -1                    # but reversal still triggers re-entry


# ─── determinism ────────────────────────────────────────────────────────────────

def test_determinism():
    a = decide_exit(_long(), BarContext(1.02, 0.02, 1, 25, math.nan), CFG)
    b = decide_exit(_long(), BarContext(1.02, 0.02, 1, 25, math.nan), CFG)
    assert a == b


def test_enum_values_match_reason_constants():
    from sovereign.forex import fast_backtester as fb
    assert int(ExitDecision.INITIAL_STOP) == fb.EXIT_REASON_STOP
    assert int(ExitDecision.REVERSAL) == fb.EXIT_REASON_REVERSAL
    assert int(ExitDecision.CB_REFRESH) == fb.EXIT_REASON_CB_REFRESH
    assert int(ExitDecision.TIME) == fb.EXIT_REASON_TIME
    assert int(ExitDecision.TRAILING_ATR) == fb.EXIT_REASON_TRAILING
    assert int(ExitDecision.DONCHIAN) == fb.EXIT_REASON_DONCHIAN


# ─── parity-gate for the vectorized kernel (HYP-071 Step 2) ─────────────────────

def test_decide_exit_vec_parity():
    """decide_exit_vec must be byte-identical to scalar decide_exit across a large random battery,
    on decision AND advanced (best, worst, hold_count) AND reentry_signal, for both directions and all
    four cfg toggles. If this fails, the vec kernel is disabled and rollouts fall back to scalar."""
    rng = np.random.default_rng(20260630)
    n = 200_000
    direction = rng.choice([-1, 1], size=n)
    # prices around 1.0 so stop/trail crossings actually happen across the battery
    close = rng.uniform(0.80, 1.20, size=n)
    stop = rng.uniform(0.80, 1.20, size=n)
    best = rng.uniform(0.90, 1.20, size=n)
    worst = rng.uniform(0.80, 1.10, size=n)
    atr = rng.uniform(0.0, 0.03, size=n)            # includes 0 → exercises the 1e-6 floor
    signal = rng.choice([-1, 0, 1], size=n)
    hold_today = rng.integers(0, 70, size=n)
    hold_count = rng.integers(0, 65, size=n)
    hold_limit = rng.integers(1, 61, size=n)
    donch = np.where(rng.random(n) < 0.5, rng.uniform(0.80, 1.20, size=n), np.nan)

    for cfg in (
        ExitConfig(2.0, 1.25, False, True),   # canonical v015
        ExitConfig(2.0, 1.25, True, True),    # strict → donchian + no time exit
        ExitConfig(2.0, 0.0, False, False),   # no trail, no cb_refresh
        ExitConfig(2.0, 2.0, False, True),    # GBP-style trail
    ):
        dec_v, best_v, worst_v, hc_v, re_v = decide_exit_vec(
            direction, stop, best, worst, hold_count, hold_limit,
            close, atr, signal, hold_today, donch, cfg,
        )
        # scalar reference, element by element
        dec_s = np.empty(n, dtype=np.int64)
        best_s = np.empty(n); worst_s = np.empty(n); hc_s = np.empty(n, dtype=np.int64); re_s = np.empty(n, dtype=np.int64)
        for i in range(n):
            r = decide_exit(
                PositionState(int(direction[i]), float(stop[i]), float(best[i]), float(worst[i]),
                              int(hold_count[i]), int(hold_limit[i])),
                BarContext(float(close[i]), float(atr[i]), int(signal[i]), int(hold_today[i]), float(donch[i])),
                cfg,
            )
            dec_s[i] = int(r.decision)
            best_s[i] = r.state.best_price
            worst_s[i] = r.state.worst_price
            hc_s[i] = r.state.hold_count
            re_s[i] = r.reentry_signal
        np.testing.assert_array_equal(dec_v, dec_s)
        np.testing.assert_array_equal(best_v, best_s)
        np.testing.assert_array_equal(worst_v, worst_s)
        np.testing.assert_array_equal(hc_v, hc_s)
        np.testing.assert_array_equal(re_v, re_s)
