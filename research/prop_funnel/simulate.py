"""Vectorized phase Monte-Carlo (TICK-022 P4).

Exact numpy reimplementation of ChallengeEngine.run_phase for uncapped sizing
(risk = equity * risk_pct). tests/test_funnel.py proves bit-level agreement with
the scalar engine on shared draw matrices — the scalar engine remains the
readable reference semantics; this module is throughput.

Not vectorized (use ChallengeEngine directly): buffer-capped sizing (parity
presets only — sizing then depends on the floor path, which is circular).

Sampling and evaluation are split so tests can replay identical draws through
both engines: sample_trades() -> DrawSet, evaluate(DrawSet, ...) -> PhaseStats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from research.prop_funnel.feeds import TradePool
from research.prop_funnel.rulesets import DrawdownModel, FirmSpec, PhaseSpec

OUTCOME_PASS, OUTCOME_BUST, OUTCOME_TIMEOUT, OUTCOME_INCOMPLETE = 1, -1, -2, 0


@dataclass
class DrawSet:
    counts: np.ndarray      # (n, D) int — trades per day
    R: np.ndarray           # (n, K) float — padded per-trade R draws
    day_idx: np.ndarray     # (n, K) int — day of each padded trade slot
    valid: np.ndarray       # (n, K) bool — slot < total trades for that attempt

    @property
    def n(self) -> int:
        return self.counts.shape[0]

    @property
    def days(self) -> int:
        return self.counts.shape[1]

    def day_r_lists(self, i: int) -> list:
        """Row i as ChallengeEngine's per-day list-of-R stream (for cross-checks)."""
        out, k = [], 0
        for d in range(self.days):
            c = int(self.counts[i, d])
            out.append([float(r) for r in self.R[i, k:k + c]])
            k += c
        return out


def sample_trades(rng: np.random.Generator, pool: TradePool,
                  n_attempts: int, days: int,
                  trades_per_day: Optional[float] = None) -> DrawSet:
    tpd = pool.trades_per_day if trades_per_day is None else trades_per_day
    counts = rng.poisson(tpd, size=(n_attempts, days)).astype(np.int32)
    totals = counts.sum(axis=1)
    K = max(int(totals.max()), 1)
    R = np.asarray(pool.draw(rng, (n_attempts, K)), dtype=float)
    csum = np.cumsum(counts, axis=1)
    slots = np.arange(K)
    day_idx = np.empty((n_attempts, K), dtype=np.int32)
    for i in range(n_attempts):
        day_idx[i] = np.searchsorted(csum[i], slots, side="right")
    valid = slots[None, :] < totals[:, None]
    # invalid padded slots map past the last day (searchsorted returns D) — clip;
    # their values never contribute (every consumer masks on `valid`)
    np.minimum(day_idx, days - 1, out=day_idx)
    return DrawSet(counts=counts, R=R, day_idx=day_idx, valid=valid)


@dataclass
class PhaseStats:
    outcome: np.ndarray          # (n,) int codes
    event_day: np.ndarray        # (n,) int — 0-based stream day of the deciding event
    trading_days_at_event: np.ndarray
    equity_end: np.ndarray       # (n,) equity at event (pass/fail) or at horizon
    eod_equity: np.ndarray       # (n, D) full EOD equity paths (pre-event semantics)

    def rate(self, code: int) -> float:
        return float(np.mean(self.outcome == code))

    def days_to_pass_pctiles(self, ps=(10, 50, 90)) -> dict:
        d = self.trading_days_at_event[self.outcome == OUTCOME_PASS]
        if d.size == 0:
            return {f"p{p}": None for p in ps}
        return {f"p{p}": float(np.percentile(d, p)) for p in ps}


def evaluate(ds: DrawSet, spec: FirmSpec, phase: PhaseSpec, risk_pct: float,
             kappa_stress: bool = False) -> PhaseStats:
    n, D = ds.counts.shape
    K = ds.R.shape[1]
    initial = spec.account_size
    dd = spec.max_dd_usd
    target = initial + phase.profit_target_usd
    static_floor = initial - dd
    lock_cap = _lock_cap(spec)

    # per-trade equity path (multiplicative; risk = equity_before * risk_pct)
    f = np.where(ds.valid, 1.0 + risk_pct * ds.R, 1.0)
    E = initial * np.cumprod(f, axis=1)
    E_before = np.concatenate([np.full((n, 1), initial), E[:, :-1]], axis=1)

    # EOD equity per day (carry initial before first trade)
    csum = np.cumsum(ds.counts, axis=1)
    last_idx = csum - 1
    gather = np.take_along_axis(E, np.clip(last_idx, 0, K - 1), axis=1)
    eod = np.where(last_idx >= 0, gather, initial)

    # per-trade floor by model
    if spec.drawdown_model is DrawdownModel.STATIC:
        floor_trade = np.full((n, K), static_floor)
    elif spec.drawdown_model is DrawdownModel.EOD_TRAILING:
        peak = np.maximum.accumulate(eod, axis=1)
        floor_eod = np.maximum(static_floor, np.minimum(peak - dd, lock_cap))
        floor_prev = np.concatenate([np.full((n, 1), static_floor), floor_eod[:, :-1]], axis=1)
        floor_trade = np.take_along_axis(floor_prev, ds.day_idx, axis=1)
    else:  # INTRADAY_TRAILING — hwm includes the current trade's (stressed) touch
        touch_R = np.where((ds.R > 0) & kappa_stress, spec.kappa * ds.R, ds.R)
        touch_e = E_before * (1.0 + risk_pct * np.maximum(touch_R, 0.0))
        touch_e = np.where(ds.valid, np.maximum(touch_e, E), initial)
        hwm = np.maximum.accumulate(np.maximum(touch_e, initial), axis=1)
        floor_trade = np.maximum(static_floor, np.minimum(hwm - dd, lock_cap))

    bust_trade = ds.valid & (E <= floor_trade)

    # daily loss from day-open equity, checked after each trade
    if spec.daily_loss_usd is not None:
        eod_prev = np.concatenate([np.full((n, 1), initial), eod[:, :-1]], axis=1)
        day_open_trade = np.take_along_axis(eod_prev, ds.day_idx, axis=1)
        daily_trade = ds.valid & ((day_open_trade - E) >= spec.daily_loss_usd)
        fail_trade = bust_trade | daily_trade
    else:
        fail_trade = bust_trade

    big = K + 1
    first_fail_slot = np.where(fail_trade.any(1), fail_trade.argmax(1), big)
    has_fail = first_fail_slot < big
    fail_day = np.where(has_fail,
                        np.take_along_axis(ds.day_idx, np.clip(first_fail_slot, 0, K - 1)[:, None],
                                           axis=1)[:, 0],
                        D + 1)

    # trading-day counter
    if spec.trading_day_basis == "all":
        tdays = np.tile(np.arange(1, D + 1), (n, 1))
    else:
        tdays = np.cumsum(ds.counts > 0, axis=1)

    # pass condition at EOD (target + min days + consistency)
    pass_ok = (eod >= target) & (tdays >= phase.min_trading_days)
    if spec.consistency_pct is not None:
        eod_prev_full = np.concatenate([np.full((n, 1), initial), eod[:, :-1]], axis=1)
        day_pnl = eod - eod_prev_full
        best_pos = np.maximum.accumulate(np.maximum(day_pnl, 0.0), axis=1)
        profit = eod - initial
        pass_ok &= best_pos <= spec.consistency_pct * np.maximum(profit, 0.0)
    Dbig = D + 1
    pass_day = np.where(pass_ok.any(1), pass_ok.argmax(1), Dbig)

    # timeout at EOD (checked AFTER the pass check on the same day)
    if spec.time_limit_days is not None:
        to_ok = tdays >= spec.time_limit_days
        timeout_day = np.where(to_ok.any(1), to_ok.argmax(1), Dbig)
    else:
        timeout_day = np.full(n, Dbig)

    # first event: intraday fail beats same-day pass; pass beats same-day timeout
    outcome = np.full(n, OUTCOME_INCOMPLETE, dtype=int)
    event_day = np.full(n, D - 1, dtype=int)

    fail_wins = (fail_day <= pass_day) & (fail_day <= timeout_day) & (fail_day <= D)
    pass_wins = ~fail_wins & (pass_day <= timeout_day) & (pass_day <= D)
    to_wins = ~fail_wins & ~pass_wins & (timeout_day <= D)

    outcome[fail_wins] = OUTCOME_BUST
    outcome[pass_wins] = OUTCOME_PASS
    outcome[to_wins] = OUTCOME_TIMEOUT
    event_day[fail_wins] = fail_day[fail_wins]
    event_day[pass_wins] = pass_day[pass_wins]
    event_day[to_wins] = timeout_day[to_wins]

    rows = np.arange(n)
    tdays_at = tdays[rows, np.clip(event_day, 0, D - 1)]
    equity_end = eod[rows, np.clip(event_day, 0, D - 1)]
    # for intraday fails, equity at the failing trade is more honest than that day's EOD
    if has_fail.any():
        fr = rows[fail_wins]
        equity_end[fr] = E[fr, np.clip(first_fail_slot[fail_wins], 0, K - 1)]

    return PhaseStats(outcome=outcome, event_day=event_day,
                      trading_days_at_event=tdays_at, equity_end=equity_end,
                      eod_equity=eod)


def _lock_cap(spec: FirmSpec) -> float:
    from research.prop_funnel.rulesets import TrailingLock
    if spec.trailing_lock is TrailingLock.AT_INITIAL:
        return spec.account_size
    if spec.trailing_lock is TrailingLock.AT_INITIAL_MINUS_DD:
        return spec.account_size - spec.max_dd_usd
    if spec.trailing_lock is TrailingLock.AT_INITIAL_PLUS_USD:
        return spec.account_size + spec.trailing_lock_plus_usd
    return float("inf")


def run_phase_mc(rng: np.random.Generator, pool: TradePool, spec: FirmSpec,
                 phase: PhaseSpec, risk_pct: float, n_attempts: int,
                 kappa_stress: bool = False,
                 trades_per_day: Optional[float] = None) -> PhaseStats:
    days = spec.sim_cap_trading_days
    ds = sample_trades(rng, pool, n_attempts, days, trades_per_day=trades_per_day)
    return evaluate(ds, spec, phase, risk_pct, kappa_stress=kappa_stress)
