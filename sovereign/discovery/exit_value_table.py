#!/usr/bin/env python3
"""sovereign/discovery/exit_value_table.py — HYP-071 tabular exit value function (core).

Networkless, deterministic, seeded. Given per-pair cached daily arrays + the backtester trade ledger +
per-pair ExitConfig, it:
  1. RE-TRACES each historical trade bar-by-bar with the shared `decide_exit` (parity vs the ledger),
     emitting a Member at every HOLD bar (its board cell + carried PositionState + current excursion-R).
  2. Builds a regime-conditional stationary-block-bootstrap return pool (start-conditioned, geometric
     mean L=5), tercile-tagged by the rolling-60-day ATR% percentile.
  3. Computes V(cell, action) = E[R] − 0.5·DownsideDeviation(R) for each evaluated (carry-aligned) cell:
     EXIT_NOW deterministically from member excursion-Rs; HOLD_AND_TRAIL by carrying each sampled member
     state forward through a resampled continuation (vectorized `decide_exit_vec`) to its terminal exit.
  4. Validates: per-cell CPCV sign-stability of (V_hold − V_exit); divergence classification.

Locked design: data/research/preregister/HYP-071_tabular_exit_value.yaml (v2, 3d500bda…) +
HYP-071_interpretation_notes.yaml (c1fab80…). NO labels/Sharpe/edge fitting here beyond the locked V.
The exit arithmetic comes ONLY from exit_machine.decide_exit / decide_exit_vec (live == backtest).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.forex.exit_machine import (
    BarContext, ExitConfig, ExitDecision, PositionState, decide_exit, decide_exit_vec,
)
from sovereign.discovery.cpcv import combinatorial_purged_splits

# ── locked constants ────────────────────────────────────────────────────────────────────────────
BLOCK_L = 5                  # expected geometric block length (v2 lock)
LAMBDA = 0.5                 # downside-penalty weight (v2 lock)
SEED = 20260630             # base seed; per-cell seed = SEED + cell_id + window_id
PCT_WINDOW = 60             # rolling ATR% percentile window (board axis)
ATR_CUTS = (1.0 / 3.0, 2.0 / 3.0)
RSI_PERIOD = 14
MIN_MEMBERS_FOLD = 10       # CPCV: a fold with fewer test members is "insufficient", not "unstable"
CPCV_BUDGET = 2000          # continuations for the sign test (sign doesn't need the full 10k)

# ExitDecision → ledger reason string (for the parity assert)
DECISION_TO_REASON = {
    int(ExitDecision.INITIAL_STOP): "stop",
    int(ExitDecision.REVERSAL): "reversal",
    int(ExitDecision.CB_REFRESH): "cb_refresh",
    int(ExitDecision.TIME): "time",
    int(ExitDecision.TRAILING_ATR): "trailing_stop",
    int(ExitDecision.DONCHIAN): "donchian_exit",
}


# ── indicators (inlined so the core stays dependency-light; match the live definitions) ───────────
def compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder-smoothed RSI, aligned to `closes` (mirror of train_core._compute_rsi)."""
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    safe_loss = np.where(avg_loss > 0, avg_loss, 1e-9)
    rs = avg_gain / safe_loss
    rsi[period:] = 100.0 - 100.0 / (1.0 + rs[period:])
    return rsi


def trailing_pct(s: np.ndarray, window: int = PCT_WINDOW) -> np.ndarray:
    """Trailing percentile rank of each point within its window (no look-ahead). NaN until min_periods.
    Mirrors scripts/research/exit_regime_conditioning.py::_trailing_pct."""
    ser = pd.Series(np.asarray(s, dtype=np.float64))
    return ser.rolling(window, min_periods=30).apply(lambda x: float((x <= x[-1]).mean()), raw=True).to_numpy()


# ── board discretization (all cuts fixed; zero fitted thresholds) ─────────────────────────────────
def _atr_tercile(pct: float) -> int:
    if not np.isfinite(pct):
        return -1
    return 0 if pct < ATR_CUTS[0] else (2 if pct > ATR_CUTS[1] else 1)


def _excursion_bucket(r: float) -> int:
    return 0 if r < 0.0 else (2 if r > 1.0 else 1)        # underwater / modest 0–1R / extended >1R


def _hold_bucket(frac: float) -> int:
    return 0 if frac < 0.33 else (2 if frac > 0.66 else 1)  # early / mid / late


def _rsi_extreme(rsi: float, direction: int) -> int:
    return int((direction == 1 and rsi > 70.0) or (direction == -1 and rsi < 30.0))


def _carry_bucket(signal: int, direction: int) -> int:
    # aligned (0): signal not opposing {+dir, 0}; not-aligned (1): signal == -dir (always REVERSAL → N/A)
    return 1 if signal == -direction else 0


def cell_id(atr_t: int, exc: int, hold: int, rsi_x: int, carry: int) -> int:
    return ((((atr_t * 3 + exc) * 3 + hold) * 2 + rsi_x) * 2 + carry)


def decode_cell(cid: int) -> dict:
    carry = cid % 2; cid //= 2
    rsi_x = cid % 2; cid //= 2
    hold = cid % 3; cid //= 3
    exc = cid % 3; cid //= 3
    atr_t = cid
    names = {
        "atr_tercile": ["low", "mid", "high"][atr_t],
        "excursion": ["underwater", "modest", "extended"][exc],
        "hold_frac": ["early", "mid", "late"][hold],
        "rsi_extreme": bool(rsi_x),
        "carry": ["aligned", "not_aligned"][carry],
    }
    return names


def is_evaluated(cid: int) -> bool:
    return (cid % 2) == 0      # carry == aligned


N_CELLS = 108


# ── members ───────────────────────────────────────────────────────────────────────────────────────
@dataclass
class MemberArrays:
    """Struct-of-arrays for one cell's members (for the vectorized rollout)."""
    pair_idx: np.ndarray
    direction: np.ndarray
    stop_price: np.ndarray
    best_price: np.ndarray
    worst_price: np.ndarray
    hold_count: np.ndarray
    hold_limit: np.ndarray
    entry_price: np.ndarray
    entry_atr: np.ndarray
    close_now: np.ndarray
    excursion_R: np.ndarray
    tercile: np.ndarray
    entry_dt: np.ndarray       # datetime64, for CPCV purging
    exit_dt: np.ndarray

    def __len__(self):
        return len(self.direction)

    def subset(self, idx):
        return MemberArrays(**{f: getattr(self, f)[idx] for f in self.__dataclass_fields__})


# ── re-trace: replay trades, emit HOLD-bar members + parity stats ─────────────────────────────────
def retrace_members(cache: dict, trades: list, cfg: ExitConfig, pair_idx: int) -> tuple[list, dict]:
    """Replay each ledger trade bar-by-bar with the scalar decide_exit. Emit a Member dict at every
    HOLD bar; assert the terminal (exit_pos, reason) matches the ledger byte-for-byte.

    cache keys: idx (DatetimeIndex), pos ({ts:i}), opens, closes, atr (raw ATR%), atr_pct (60d pctile),
                signal, hold, rsi, tercile, pair (str). Returns (members, parity={'n','matched','dropped'}).
    """
    opens, closes = cache["opens"], cache["closes"]
    atr, atr_pct = cache["atr"], cache["atr_pct"]
    signal, hold = cache["signal"], cache["hold"]
    rsi, tercile = cache["rsi"], cache["tercile"]
    idx, pos = cache["idx"], cache["pos"]
    n_bars = len(closes)
    members, matched, dropped = [], 0, 0

    for tr in trades:
        ets = pd.Timestamp(tr["entry_date"])
        epos = pos.get(ets)
        if epos is None or epos < 1:
            dropped += 1
            continue
        direction = int(tr["direction"])
        entry_price = float(opens[epos])
        entry_atr = max(float(atr[epos - 1]), 1e-6)
        risk = cfg.stop_atr_mult * entry_atr * entry_price
        stop_price = entry_price - risk if direction == 1 else entry_price + risk
        hold_limit = max(int(hold[epos - 1]), 1)
        state = PositionState(direction, stop_price, entry_price, entry_price, 0, hold_limit)

        exit_pos, exit_decision = None, None
        for j in range(epos, n_bars):
            res = decide_exit(
                state,
                BarContext(float(closes[j]), float(atr[j]), int(signal[j]), int(hold[j]), float("nan")),
                cfg,
            )
            state = res.state
            if res.decision == ExitDecision.HOLD:
                exc = direction * (closes[j] - entry_price) / risk
                atr_t = _atr_tercile(tercile[j])
                if atr_t < 0:
                    continue                       # tercile undefined (warm-up) → not a board cell
                cid = cell_id(
                    atr_t, _excursion_bucket(exc), _hold_bucket(state.hold_count / state.hold_limit),
                    _rsi_extreme(rsi[j], direction), _carry_bucket(int(signal[j]), direction),
                )
                members.append({
                    "pair_idx": pair_idx, "direction": direction, "stop_price": state.stop_price,
                    "best_price": state.best_price, "worst_price": state.worst_price,
                    "hold_count": state.hold_count, "hold_limit": state.hold_limit,
                    "entry_price": entry_price, "entry_atr": entry_atr, "close_now": float(closes[j]),
                    "excursion_R": exc, "tercile": atr_t, "cell_id": cid,
                    "entry_dt": np.datetime64(ets), "exit_dt": np.datetime64(pd.Timestamp(tr["exit_date"])),
                })
            else:
                exit_pos, exit_decision = j, int(res.decision)
                break

        # parity vs ledger
        if exit_pos is not None:
            same_date = (idx[exit_pos] == pd.Timestamp(tr["exit_date"]))
            same_reason = (DECISION_TO_REASON.get(exit_decision) == tr.get("exit_reason"))
            if same_date and same_reason:
                matched += 1
    return members, {"n": len(trades), "matched": matched, "dropped": dropped}


def group_members_by_cell(members: list) -> dict:
    """list[member dict] → {cell_id: MemberArrays} for evaluated (carry-aligned) cells only."""
    by_cell: dict[int, list] = {}
    for m in members:
        by_cell.setdefault(m["cell_id"], []).append(m)
    out = {}
    for cid, ms in by_cell.items():
        if not is_evaluated(cid):
            continue
        out[cid] = MemberArrays(
            pair_idx=np.array([m["pair_idx"] for m in ms], dtype=np.int64),
            direction=np.array([m["direction"] for m in ms], dtype=np.int64),
            stop_price=np.array([m["stop_price"] for m in ms], dtype=np.float64),
            best_price=np.array([m["best_price"] for m in ms], dtype=np.float64),
            worst_price=np.array([m["worst_price"] for m in ms], dtype=np.float64),
            hold_count=np.array([m["hold_count"] for m in ms], dtype=np.int64),
            hold_limit=np.array([m["hold_limit"] for m in ms], dtype=np.int64),
            entry_price=np.array([m["entry_price"] for m in ms], dtype=np.float64),
            entry_atr=np.array([m["entry_atr"] for m in ms], dtype=np.float64),
            close_now=np.array([m["close_now"] for m in ms], dtype=np.float64),
            excursion_R=np.array([m["excursion_R"] for m in ms], dtype=np.float64),
            tercile=np.array([m["tercile"] for m in ms], dtype=np.int64),
            entry_dt=np.array([m["entry_dt"] for m in ms]),
            exit_dt=np.array([m["exit_dt"] for m in ms]),
        )
    return out


# ── resampling pool (flattened across pairs; start-conditioned by tercile) ─────────────────────────
@dataclass
class Pool:
    ret: np.ndarray          # global flat arrays
    atr: np.ndarray
    signal: np.ndarray
    hold: np.ndarray
    boundary: np.ndarray     # True if g is the last bar of its pair segment (block cannot continue)
    starts_by_tercile: dict  # {tercile: ndarray of global start indices}


def build_return_pool(caches: list, restrict_dates=None) -> Pool:
    """Flatten the per-pair daily arrays into one pool. `caches` = list of cache dicts (one per pair).
    Block STARTS are restricted to bars with a defined tercile (and within `restrict_dates` if given,
    for CPCV train-fold leakage control)."""
    rets, atrs, sigs, holds, bounds, terc = [], [], [], [], [], []
    for c in caches:
        closes = np.asarray(c["closes"], dtype=np.float64)
        r = np.zeros(len(closes))
        r[1:] = closes[1:] / closes[:-1] - 1.0           # close-to-close return; first bar 0
        b = np.zeros(len(closes), dtype=bool); b[-1] = True
        t = np.array([_atr_tercile(p) for p in c["atr_pct"]], dtype=np.int64)
        if restrict_dates is not None:
            lo, hi = restrict_dates
            inwin = np.asarray((c["idx"] >= lo) & (c["idx"] <= hi))
            t = np.where(inwin, t, -1)                    # bars outside the window can't be block starts
        rets.append(r); atrs.append(np.asarray(c["atr"], dtype=np.float64))
        sigs.append(np.asarray(c["signal"], dtype=np.int64)); holds.append(np.asarray(c["hold"], dtype=np.int64))
        bounds.append(b); terc.append(t)
    ret = np.concatenate(rets); atr = np.concatenate(atrs)
    signal = np.concatenate(sigs); hold = np.concatenate(holds)
    boundary = np.concatenate(bounds); tercile = np.concatenate(terc)
    starts = {t: np.where(tercile == t)[0] for t in (0, 1, 2)}
    return Pool(ret, atr, signal, hold, boundary, starts)


def _sample_paths(tercile_per_cont: np.ndarray, max_bars: int, pool: Pool, rng) -> np.ndarray:
    """Pre-sample [n_cont, max_bars] global indices via start-conditioned geometric (mean L) blocks."""
    n = len(tercile_per_cont)
    g = np.empty((n, max_bars), dtype=np.int64)

    def draw(terciles):
        out = np.empty(len(terciles), dtype=np.int64)
        for t in (0, 1, 2):
            mask = terciles == t
            k = int(mask.sum())
            if k:
                starts = pool.starts_by_tercile[t]
                out[mask] = starts[rng.integers(0, len(starts), k)]
        return out

    cur = draw(tercile_per_cont)
    g[:, 0] = cur
    for b in range(1, max_bars):
        jump = rng.random(n) < (1.0 / BLOCK_L)
        nxt = cur + 1
        at_bound = pool.boundary[cur]                    # current bar is last in its pair → must restart
        redraw = jump | at_bound
        if redraw.any():
            nxt = nxt.copy()
            nxt[redraw] = draw(tercile_per_cont[redraw])
        cur = nxt
        g[:, b] = cur
    return g


def rollout_R(ma: MemberArrays, pool: Pool, cfg_by_pair: dict, n_cont: int, rng, signal_mode: str) -> np.ndarray:
    """Draw n_cont (member, continuation) pairs and carry forward to terminal R. Members may span pairs
    (different trailing-mult cfg), so we group by pair and run each group with its own scalar cfg."""
    if len(ma) == 0:
        return np.empty(0)
    midx = rng.integers(0, len(ma), size=n_cont)
    out = np.empty(n_cont)
    for p in np.unique(ma.pair_idx[midx]):
        sel = np.where(ma.pair_idx[midx] == p)[0]
        sub = ma.subset(midx[sel])
        cfg = cfg_by_pair[int(p)]
        # sub is already the drawn members (one continuation each) → rollout 1 cont per member
        out[sel] = _rollout_one_each(sub, pool, cfg, rng, signal_mode)
    return out


def _rollout_one_each(ma: MemberArrays, pool: Pool, cfg: ExitConfig, rng, signal_mode: str) -> np.ndarray:
    """One continuation per member in `ma` (members already drawn). Vectorized."""
    n = len(ma)
    if n == 0:
        return np.empty(0)
    direction = ma.direction
    stop = ma.stop_price.copy(); best = ma.best_price.copy(); worst = ma.worst_price.copy()
    hold_count = ma.hold_count.copy(); hold_limit = ma.hold_limit
    entry_price = ma.entry_price; entry_atr = ma.entry_atr; tercile = ma.tercile
    c = ma.close_now.copy()
    max_bars = int((hold_limit - hold_count).max()) + 1
    g = _sample_paths(tercile, max_bars, pool, rng)
    ret_p, atr_p = pool.ret[g], pool.atr[g]
    sig_p = np.zeros_like(pool.signal[g]) if signal_mode == "frozen" else pool.signal[g]
    hold_p = pool.hold[g]
    nan_col = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool); R = np.full(n, np.nan)
    risk = cfg.stop_atr_mult * entry_atr * entry_price
    for b in range(max_bars):
        c = np.where(alive, c * (1.0 + ret_p[:, b]), c)
        dec, best, worst, hold_count, _ = decide_exit_vec(
            direction, stop, best, worst, hold_count, hold_limit,
            c, atr_p[:, b], sig_p[:, b], hold_p[:, b], nan_col, cfg,
        )
        newly = alive & (dec != int(ExitDecision.HOLD))
        if newly.any():
            R[newly] = direction[newly] * (c[newly] - entry_price[newly]) / risk[newly]
        alive &= (dec == int(ExitDecision.HOLD))
        if not alive.any():
            break
    if alive.any():
        R[alive] = direction[alive] * (c[alive] - entry_price[alive]) / risk[alive]
    return R


# ── value ─────────────────────────────────────────────────────────────────────────────────────────
def downside_dev(R: np.ndarray, mar: float = 0.0) -> float:
    R = np.asarray(R, dtype=np.float64)
    if R.size == 0:
        return 0.0
    d = np.minimum(R - mar, 0.0)
    return float(np.sqrt(np.mean(d * d)))


def value(R: np.ndarray, lam: float = LAMBDA) -> float:
    R = np.asarray(R, dtype=np.float64)
    if R.size == 0:
        return float("nan")
    return float(np.mean(R) - lam * downside_dev(R))


@dataclass
class CellResult:
    cell_id: int
    n_members: int
    V_hold: float
    V_exit: float
    E_hold: float
    DD_hold: float
    margin: float
    optimal_action: str


def compute_table(members_by_cell: dict, pool: Pool, cfg_by_pair: dict, n_cont: int = 10000,
                  window_id: int = 0, signal_mode: str = "live") -> dict:
    """{cell_id: CellResult} for every evaluated cell with members."""
    out = {}
    for cid, ma in members_by_cell.items():
        rng = np.random.default_rng(SEED + cid + window_id)
        V_exit = value(ma.excursion_R)
        R_hold = rollout_R(ma, pool, cfg_by_pair, n_cont, rng, signal_mode)
        V_hold = value(R_hold)
        margin = V_hold - V_exit
        out[cid] = CellResult(
            cell_id=cid, n_members=len(ma), V_hold=V_hold, V_exit=V_exit,
            E_hold=float(np.mean(R_hold)) if R_hold.size else float("nan"),
            DD_hold=downside_dev(R_hold), margin=margin,
            optimal_action="HOLD_AND_TRAIL" if margin >= 0 else "EXIT_NOW",
        )
    return out


# ── CPCV sign-stability per cell ──────────────────────────────────────────────────────────────────
def cpcv_sign_stability(ma: MemberArrays, caches: list, cfg_by_pair: dict, full_margin: float,
                        window_id: int, signal_mode: str) -> dict:
    """Recompute sign(V_hold − V_exit) on each purged-combinatorial test fold of the cell's member
    trades; the continuation pool is restricted to TRAIN-fold dates (no test leakage)."""
    full_sign = 1 if full_margin >= 0 else -1
    if len(ma) < 2 * MIN_MEMBERS_FOLD:
        return {"sign_stable": False, "stability_fraction": None, "signs": [], "status": "CPCV_INSUFFICIENT"}
    splits = list(combinatorial_purged_splits(ma.entry_dt, ma.exit_dt, 6, 2, 0.01))
    signs, insufficient = [], False
    for k, (train_idx, test_idx) in enumerate(splits):
        if len(test_idx) < MIN_MEMBERS_FOLD:
            insufficient = True
            continue
        test = ma.subset(test_idx)
        train_dates = (ma.entry_dt[train_idx].min(), ma.exit_dt[train_idx].max()) if len(train_idx) else None
        pool_k = build_return_pool(caches, restrict_dates=train_dates) if train_dates else build_return_pool(caches)
        rng = np.random.default_rng(SEED + window_id + k + 7919)
        margin_k = value(rollout_R(test, pool_k, cfg_by_pair, CPCV_BUDGET, rng, signal_mode)) - value(test.excursion_R)
        signs.append(1 if margin_k >= 0 else -1)
    if not signs:
        return {"sign_stable": False, "stability_fraction": None, "signs": [], "status": "CPCV_INSUFFICIENT"}
    frac = float(np.mean([s == full_sign for s in signs]))
    status = "CPCV_INSUFFICIENT" if insufficient else "OK"
    return {"sign_stable": bool(frac >= 14.0 / 15.0 and status == "OK"),
            "stability_fraction": frac, "signs": signs, "status": status}


# ── divergence classification (pre-specified sensibility map, addendum) ────────────────────────────
def is_economically_sensible(cid: int, action: str) -> bool:
    if action != "EXIT_NOW":
        return False
    d = decode_cell(cid)
    return (d["atr_tercile"] == "high" or d["hold_frac"] == "late"
            or d["excursion"] == "extended" or d["rsi_extreme"])
