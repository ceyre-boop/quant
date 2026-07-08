#!/usr/bin/env python3
"""scripts/research/exit_policy_evolution.py — HYP-067.

Pre-registered, DATA-ONLY evolutionary search for a regime- AND trade-age-conditioned
forex exit policy. A policy is a 3x3x3 lookup table

    (vix_tercile 0-2, atr_tercile 0-2, hold_count_bucket early/mid/late)
        -> (stop_atr_mult, trailing_atr_mult, hold_limit)

evaluated PER BAR as a trade ages, against the canonical v015 decade trades as a FIXED
entry universe (meta-labeling, AFML ch.3 — only the secondary exit policy is evolved).
A genetic algorithm maximises a CPCV-Sharpe fitness with a robustness penalty; the winner
is then forced through the permutation / deflated-Sharpe / Benjamini-Hochberg gauntlet, a
forward-holdout non-degradation gate, and a mandatory prove reconciliation band.

GATE: this study generalises HYP-066 (exit_regime_conditioning). It only runs against that
prior by default if HYP-066 returned VALID_EDGE. HYP-066 returned NOT_SIGNIFICANT, so the
default run HALTS. Use --standalone to run it as an independent pre-registered hypothesis
whose sole decider is the gauntlet + prove band (with a stricter, search-size deflated-Sharpe
correction). The exit itself is replayed through sovereign.forex.exit_machine.decide_exit —
the SAME function live + backtest use — so any finding is deployable with parity preserved.

NO LIVE TOUCH. No OANDA, no forex_exit_manager import, no config writes, no SHADOW_MODE
change. Reads yfinance + the canonical backtest; writes only under data/research/ and one
append to data/agent/hypothesis_ledger.json.

    python3 scripts/research/exit_policy_evolution.py --sign         # freeze the prereg hash (run once)
    python3 scripts/research/exit_policy_evolution.py                # gated: halts unless HYP-066 VALID_EDGE
    python3 scripts/research/exit_policy_evolution.py --standalone   # run as independent HYP-067
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.forex.exit_machine import BarContext, ExitConfig, ExitDecision, PositionState, decide_exit
from sovereign.forex.forex_backtester import ForexBacktester, RESULTS_PATH
from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.reporting.equity_curve import weighted_portfolio_sharpe
from sovereign.discovery.gate import benjamini_hochberg, deflated_sharpe_ratio
from sovereign.discovery.cpcv import combinatorial_purged_splits, n_backtest_paths, n_cpcv_splits

PREREG = ROOT / "data" / "research" / "preregister" / "HYP-067_exit_policy_evolution.json"
OUT_RESULTS = ROOT / "data" / "research" / "exit_policy_evolution_results.json"
OUT_PARETO = ROOT / "data" / "research" / "exit_policy_evolution_pareto.json"
OUT_WINNER = ROOT / "data" / "research" / "exit_policy_evolution_winner.json"
TRADES_FILE = ROOT / "logs" / "forex_backtest_trades.json"
HYP066_RESULTS = ROOT / "data" / "research" / "exit_regime_conditioning_results.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
DECADE = ("2015-01-01", "2024-12-31")
FORWARD = ("2025-01-01", "2026-06-29")
STATIC_STOP = 2.0
STATIC_TRAIL = {"GBPUSD=X": 2.0, "AUDUSD=X": 1.0, "EURUSD=X": 1.25, "USDJPY=X": 1.25}
SEED = 42
PCT_WINDOW = 252

# regime / age bucketing
HOLD_EDGES = (5, 15)                      # bars-in-trade: early [0,5) mid [5,15) late [15,inf)
N_CELLS = 27                             # 3 vix x 3 atr x 3 hold
MIN_CELL_VISITS = 30                     # below this a cell falls back to its coarser parent
MAX_BARS = 95                            # max in-trade bars to precompute (hold_limit <= 90)

# param bounds + grid
STOP_BOUNDS = (1.0, 3.0)
TRAIL_BOUNDS = (0.75, 2.5)
HOLD_BOUNDS = (10, 90)
GRID_STEP = 0.05                         # snap for stop/trail; hold snapped to integer

# genetic algorithm
POP_DEFAULT = 200
GEN_DEFAULT = 50
TOURNAMENT_K = 5
MUT_RATE = 0.15
MUT_SIGMA_FRAC = 0.2
N_ELITE = 2
EARLY_STOP = 10
SEED_FRAC = 0.20

# cpcv fitness
CPCV_GROUPS = 6
CPCV_TEST = 2
CPCV_EMBARGO = 0.01
ROBUST_PENALTY = 0.3

# gauntlet
N_PERM_DEFAULT = 2000
NON_DEGRADE_TOL = 0.05

# prove reconciliation
RECON_TARGET, RECON_TOL = 0.6886, 0.01
PROVE_FULL_DECADE_BAND = (0.68, 0.70)
PROVE_OOS_BAND = (1.17, 1.25)


# ── pre-registration hash lock (mirrors HYP-066) ─────────────────────────── #

def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sign_preregister() -> None:
    doc = json.loads(PREREG.read_text())
    h = _canonical_hash(doc)
    doc["hash_lock"] = h
    PREREG.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"signed HYP-067  hash_lock = {h}")


def verify_preregister() -> dict:
    doc = json.loads(PREREG.read_text())
    h = _canonical_hash(doc)
    if doc.get("hash_lock") != h:
        raise SystemExit(
            "PREREGISTER HASH MISMATCH — the frozen design was altered after signing.\n"
            f"  stored:   {doc.get('hash_lock')}\n  computed: {h}\n"
            "Revert the change or re-freeze deliberately with --sign."
        )
    print(f"  prereg hash OK ({h[:16]}…)")
    return doc


# ── HYP-066 gate ─────────────────────────────────────────────────────────── #

def hyp066_verdict() -> str:
    if not HYP066_RESULTS.exists():
        return "ABSENT"
    try:
        return str(json.loads(HYP066_RESULTS.read_text()).get("verdict", "ABSENT"))
    except Exception:
        return "ABSENT"


def gate_should_halt(prior_verdict: str, standalone: bool) -> bool:
    """The pre-registered gate: halt unless HYP-066 cleared, OR --standalone overrides."""
    return prior_verdict != "VALID_EDGE" and not standalone


# ── per-pair price / signal / atr cache (reuses ForexBacktester; mirrors HYP-066) ── #

def _trailing_pct(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Percentile rank of each point within its trailing `window` (inclusive). No look-ahead."""
    return s.rolling(window, min_periods=30).apply(lambda x: float((x <= x[-1]).mean()), raw=True)


def pair_arrays(bt: ForexBacktester, pair: str) -> dict | None:
    cfg = PAIR_CONFIG.get(pair)
    df = bt._download_price(pair)
    if df is None or len(df) < 252:
        return None
    base = CB_TO_COUNTRY[cfg.base_central_bank]
    quote = CB_TO_COUNTRY[cfg.quote_central_bank]
    sig = bt._get_pair_signals(df=df, base_country=base, quote_country=quote, pair=pair, hold_days=bt.HOLD_DAYS)
    if pair in bt.PAIR_VIX_GATES:
        sig = bt._apply_vix_regime_gate(sig, pair=pair, start=bt.start, end=bt.end)
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    opens = df["Open"] if "Open" in df.columns else close
    idx = close.index
    atr = pd.Series(np.asarray(bt._signals._compute_atr_pct(close, df), dtype=float), index=idx)
    hold_col = "hold_days" if "hold_days" in sig.columns else "hold"
    return {
        "pair": pair,
        "idx": idx,
        "pos": {ts: i for i, ts in enumerate(idx)},
        "opens": opens.to_numpy(dtype=float),
        "closes": close.to_numpy(dtype=float),
        "atr": atr.to_numpy(dtype=float),
        "atr_pct": _trailing_pct(atr),
        "signal": sig["signal"].reindex(idx).fillna(0).to_numpy(dtype=float).astype(int),
        "hold": sig[hold_col].reindex(idx).fillna(bt.HOLD_DAYS).to_numpy(dtype=float).astype(int),
    }


# ── static single-trade replay + costs (mirrors HYP-066) ─────────────────── #

def replay_exit(arr: dict, entry_pos: int, direction: int, stop_atr_mult: float, trail_atr_mult: float) -> dict | None:
    closes, opens, atr, signal, hold = arr["closes"], arr["opens"], arr["atr"], arr["signal"], arr["hold"]
    n = len(closes)
    if entry_pos <= 0 or entry_pos >= n:
        return None
    entry_price = float(opens[entry_pos])
    entry_atr = max(float(atr[entry_pos - 1]), 1e-6)
    stop_dist = entry_price * stop_atr_mult * entry_atr
    stop_price = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
    hold_limit = max(int(hold[entry_pos - 1]), 1)
    cfg = ExitConfig(stop_atr_mult, trail_atr_mult, strict_mode=False, enable_cb_refresh=True)
    state = PositionState(direction, stop_price, entry_price, entry_price, 0, hold_limit)
    for j in range(entry_pos, n):
        bar = BarContext(float(closes[j]), float(atr[j]), int(signal[j]), int(hold[j]), float("nan"))
        res = decide_exit(state, bar, cfg)
        state = res.state
        if res.decision != ExitDecision.HOLD:
            exit_price = float(closes[j])
            return {"entry": entry_price, "direction": direction,
                    "pnl_pct": direction * (exit_price / max(entry_price, 1e-9) - 1.0),
                    "hold_days": state.hold_count, "exit_pos": j}
    exit_price = float(closes[-1])
    return {"entry": entry_price, "direction": direction,
            "pnl_pct": direction * (exit_price / max(entry_price, 1e-9) - 1.0),
            "hold_days": state.hold_count, "exit_pos": n - 1}


def _apply_costs(pair: str, entry_price: float, direction: int, hold_days: int, raw_pnl: float) -> float:
    t = {"entry": entry_price, "direction": direction, "hold_days": hold_days, "pnl_pct": raw_pnl, "risk_pct": 1.0}
    ForexBacktester._apply_costs([t], pair)
    return float(t["pnl_pct"])


def pooled_sharpe(pnls: np.ndarray, entry_dt: np.ndarray, exit_dt: np.ndarray) -> float:
    """Annualised pooled Sharpe over a set of trades (mirrors HYP-066: sqrt(n/years))."""
    n = len(pnls)
    if n < 2:
        return 0.0
    order = np.argsort(entry_dt)
    p = np.asarray(pnls, dtype=float)[order]
    eq = np.cumprod(1.0 + p)
    ret = np.diff(np.log(eq), prepend=0.0)
    span_years = max((exit_dt.max() - entry_dt.min()) / np.timedelta64(1, "D") / 365.25, 1e-9)
    ann = np.sqrt(max(n, 1) / span_years)
    return float(np.mean(ret) / (np.std(ret) + 1e-9) * ann)


def portfolio_sharpe(idx: np.ndarray, pnls: np.ndarray, pairs: np.ndarray,
                     edt: np.ndarray, xdt: np.ndarray) -> float:
    """√n-weighted portfolio Sharpe over a subset of trades, grouped by pair."""
    per = []
    for p in np.unique(pairs[idx]):
        sel = idx[pairs[idx] == p]
        if len(sel) >= 2:
            per.append((pooled_sharpe(pnls[sel], edt[sel], xdt[sel]), len(sel)))
    return weighted_portfolio_sharpe(per)


# ── regime / age cell helpers ────────────────────────────────────────────── #

def _tercile(x: float, t0: float, t1: float) -> int:
    return 0 if x < t0 else (1 if x < t1 else 2)


def _hold_bucket(hold_count: int) -> int:
    return 0 if hold_count < HOLD_EDGES[0] else (1 if hold_count < HOLD_EDGES[1] else 2)


def cell_index(vix_t: int, atr_t: int, hold_b: int) -> int:
    return vix_t * 9 + atr_t * 3 + hold_b


def cell_marginal(cell: int) -> int:
    """(vix x atr) marginal 0-8 for a flat cell index (drops the hold axis)."""
    return cell // 3


# ── trade-spec precompute (per-bar cells + maximal path) ─────────────────── #

def build_trade_specs(trades_by_pair: dict, caches: dict, vix_pct: pd.Series,
                      vix_terciles: tuple[float, float], atr_terciles: tuple[float, float]) -> list[dict]:
    """One spec per trade: entry, maximal in-trade bar arrays, per-bar cell index, static baseline.

    Per-bar cell uses fixed (train-derived) tercile thresholds applied to that bar's VIX% and
    ATR% plus the bars-in-trade hold bucket — so the same thresholds bin both train and holdout
    with no look-ahead. The static-config exit defines each trade's label interval for CPCV purging.
    """
    specs: list[dict] = []
    for pair, tl in trades_by_pair.items():
        if pair not in caches:
            continue
        arr = caches[pair]
        idx = arr["idx"]
        n = len(idx)
        atr_pct_bar = arr["atr_pct"].reindex(idx).to_numpy(dtype=float)
        vix_bar = vix_pct.reindex(idx, method="ffill").to_numpy(dtype=float)
        for t in tl:
            ed = pd.Timestamp(t["entry_date"])
            epos = arr["pos"].get(ed)
            if epos is None:
                loc = idx.searchsorted(ed)
                if loc <= 0 or loc >= n:
                    continue
                epos = int(loc)
            if epos <= 0 or epos >= n:
                continue
            direction = int(t["direction"])
            entry_price = float(arr["opens"][epos])
            entry_atr = max(float(arr["atr"][epos - 1]), 1e-6)
            end = min(epos + MAX_BARS, n)
            bar_close = arr["closes"][epos:end]
            bar_atr = arr["atr"][epos:end]
            bar_sig = arr["signal"][epos:end]
            bar_hold = arr["hold"][epos:end]
            nb = end - epos
            if nb < 1:
                continue
            cells = np.empty(nb, dtype=np.int16)
            for k in range(nb):
                j = epos + k
                vt = _tercile(vix_bar[j] if not np.isnan(vix_bar[j]) else 0.5, *vix_terciles)
                at = _tercile(atr_pct_bar[j] if not np.isnan(atr_pct_bar[j]) else 0.5, *atr_terciles)
                cells[k] = cell_index(vt, at, _hold_bucket(k))
            entry_cell = int(cells[0])
            # static baseline (label interval + non-degradation comparison)
            st = replay_exit(arr, epos, direction, STATIC_STOP, STATIC_TRAIL.get(pair, 1.5))
            if st is None:
                continue
            static_pnl = _apply_costs(pair, st["entry"], direction, st["hold_days"], st["pnl_pct"])
            exit_dt = idx[st["exit_pos"]].to_datetime64()
            specs.append({
                "index": len(specs), "pair": pair, "direction": direction,
                "entry_dt": np.datetime64(ed), "exit_dt": exit_dt,
                "entry_price": entry_price, "entry_atr": entry_atr,
                "bar_close": bar_close, "bar_atr": bar_atr, "bar_sig": bar_sig, "bar_hold": bar_hold,
                "cells": cells, "entry_cell": entry_cell, "marginal": cell_marginal(entry_cell),
                "visited": np.unique(cells), "static_pnl": static_pnl,
            })
    return specs


def visit_counts(specs: list[dict]) -> np.ndarray:
    """Bar-visits per cell across all specs (drives the min-cell fallback)."""
    counts = np.zeros(N_CELLS, dtype=int)
    for s in specs:
        for c in s["cells"]:
            counts[c] += 1
    return counts


# ── policy representation, fallback, snapping, GA operators ──────────────── #

def _weighted_param_mean(policy: np.ndarray, weights: np.ndarray, cells: list[int]) -> np.ndarray:
    w = weights[cells].astype(float)
    if w.sum() <= 0:
        return policy[cells].mean(axis=0)
    return (policy[cells] * w[:, None]).sum(axis=0) / w.sum()


def resolve_fallback(policy: np.ndarray, counts: np.ndarray, floor: int = MIN_CELL_VISITS) -> np.ndarray:
    """Thin cells (< floor bar-visits) inherit from their coarser parent: the populated hold
    buckets of their (vix x atr) marginal, then the global populated-cell mean."""
    eff = policy.copy()
    populated = [c for c in range(N_CELLS) if counts[c] >= floor]
    global_cfg = _weighted_param_mean(policy, counts, populated) if populated else policy.mean(axis=0)
    for vix in range(3):
        for atr in range(3):
            cells = [cell_index(vix, atr, h) for h in range(3)]
            pop = [c for c in cells if counts[c] >= floor]
            marg = _weighted_param_mean(policy, counts, pop) if pop else global_cfg
            for c in cells:
                if counts[c] < floor:
                    eff[c] = marg
    return snap(eff)


def snap(policy: np.ndarray) -> np.ndarray:
    p = policy.astype(float).copy()
    p[:, 0] = np.round(np.clip(p[:, 0], *STOP_BOUNDS) / GRID_STEP) * GRID_STEP
    p[:, 1] = np.round(np.clip(p[:, 1], *TRAIL_BOUNDS) / GRID_STEP) * GRID_STEP
    p[:, 2] = np.round(np.clip(p[:, 2], *HOLD_BOUNDS))
    return p


def random_policy(rng) -> np.ndarray:
    p = np.empty((N_CELLS, 3), dtype=float)
    p[:, 0] = rng.uniform(*STOP_BOUNDS, N_CELLS)
    p[:, 1] = rng.uniform(*TRAIL_BOUNDS, N_CELLS)
    p[:, 2] = rng.uniform(*HOLD_BOUNDS, N_CELLS)
    return snap(p)


def seed_policy(decade_best_cfg: dict) -> np.ndarray:
    """Map HYP-066's 4 (2x2) best configs onto the 27-cell table (tercile 0/1 -> lo, 2 -> hi;
    hold axis broadcast; hold_limit seeded at the v015 default 60)."""
    p = np.empty((N_CELLS, 3), dtype=float)
    for vix in range(3):
        for atr in range(3):
            key = f"{'Vhi' if vix == 2 else 'Vlo'}_{'Ahi' if atr == 2 else 'Alo'}"
            cfg = decade_best_cfg.get(key) or [STATIC_STOP, 1.25]
            for h in range(3):
                p[cell_index(vix, atr, h)] = [cfg[0], cfg[1], 60.0]
    return snap(p)


def init_population(rng, n_pop: int, decade_best_cfg: dict | None) -> list[np.ndarray]:
    pop = []
    n_seed = int(round(SEED_FRAC * n_pop)) if decade_best_cfg else 0
    base = seed_policy(decade_best_cfg) if n_seed else None
    for _ in range(n_seed):
        jit = base.copy()
        jit[:, 0] += rng.normal(0, 0.1, N_CELLS)
        jit[:, 1] += rng.normal(0, 0.1, N_CELLS)
        jit[:, 2] += rng.normal(0, 5.0, N_CELLS)
        pop.append(snap(jit))
    while len(pop) < n_pop:
        pop.append(random_policy(rng))
    return pop


def mutate(policy: np.ndarray, rng) -> np.ndarray:
    p = policy.copy()
    ranges = np.array([STOP_BOUNDS[1] - STOP_BOUNDS[0], TRAIL_BOUNDS[1] - TRAIL_BOUNDS[0],
                       HOLD_BOUNDS[1] - HOLD_BOUNDS[0]], dtype=float)
    mask = rng.random((N_CELLS, 3)) < MUT_RATE
    noise = rng.normal(0, MUT_SIGMA_FRAC * ranges, (N_CELLS, 3))
    p = np.where(mask, p + noise, p)
    return snap(p)


def crossover(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    mask = rng.random((N_CELLS, 3)) < 0.5
    return np.where(mask, a, b)


def tournament(fits: np.ndarray, rng) -> int:
    k = min(TOURNAMENT_K, len(fits))
    idx = rng.choice(len(fits), k, replace=False)
    return int(idx[int(np.argmax(fits[idx]))])


# ── per-bar policy replay (calls decide_exit — parity with live/backtest) ── #

def replay_policy(spec: dict, policy: np.ndarray) -> float:
    """Costed pnl of one trade under a per-bar policy. stop_atr_mult + hold_limit come from the
    ENTRY cell; trailing_atr_mult is read from the active cell each bar."""
    closes, atrs, sigs, holds, cells = spec["bar_close"], spec["bar_atr"], spec["bar_sig"], spec["bar_hold"], spec["cells"]
    direction, entry_price, entry_atr = spec["direction"], spec["entry_price"], spec["entry_atr"]
    stop_mult = float(policy[spec["entry_cell"], 0])
    hold_limit = max(int(round(policy[spec["entry_cell"], 2])), 1)
    stop_dist = entry_price * stop_mult * entry_atr
    stop_price = entry_price - stop_dist if direction == 1 else entry_price + stop_dist
    state = PositionState(direction, stop_price, entry_price, entry_price, 0, hold_limit)
    nb = len(closes)
    exit_price = float(closes[-1])
    for k in range(nb):
        trail_mult = float(policy[cells[k], 1])
        cfg = ExitConfig(stop_mult, trail_mult, strict_mode=False, enable_cb_refresh=True)
        bar = BarContext(float(closes[k]), float(atrs[k]), int(sigs[k]), int(holds[k]), float("nan"))
        res = decide_exit(state, bar, cfg)
        state = res.state
        if res.decision != ExitDecision.HOLD:
            exit_price = float(closes[k])
            break
    raw = direction * (exit_price / max(entry_price, 1e-9) - 1.0)
    return _apply_costs(spec["pair"], entry_price, direction, state.hold_count, raw)


def evaluate(specs: list[dict], policy: np.ndarray, memo: dict) -> np.ndarray:
    """Costed pnl per spec under the (already-effective) policy, memoised on the visited-cell slice."""
    out = np.empty(len(specs), dtype=float)
    for i, s in enumerate(specs):
        key = (s["index"], policy[s["visited"]].tobytes())
        pnl = memo.get(key)
        if pnl is None:
            pnl = replay_policy(s, policy)
            memo[key] = pnl
        out[i] = pnl
    return out


# ── CPCV fitness ─────────────────────────────────────────────────────────── #

def build_fitness_ctx(specs: list[dict]) -> dict:
    edt = np.array([s["entry_dt"] for s in specs])
    xdt = np.array([s["exit_dt"] for s in specs])
    pairs = np.array([s["pair"] for s in specs])
    splits = list(combinatorial_purged_splits(edt, xdt, CPCV_GROUPS, CPCV_TEST, CPCV_EMBARGO))
    return {"edt": edt, "xdt": xdt, "pairs": pairs, "test_sets": [te for _, te in splits]}


def fitness(policy_eff: np.ndarray, specs: list[dict], ctx: dict, memo: dict) -> tuple[float, float, float]:
    """CPCV Sharpe with robustness penalty. Returns (fitness, mean_path_sharpe, std_path_sharpe)."""
    pnls = evaluate(specs, policy_eff, memo)
    paths = np.array([portfolio_sharpe(te, pnls, ctx["pairs"], ctx["edt"], ctx["xdt"]) for te in ctx["test_sets"]])
    mean, std = float(paths.mean()), float(paths.std())
    return mean - ROBUST_PENALTY * std, mean, std


# ── genetic algorithm ────────────────────────────────────────────────────── #

def run_ga(specs: list[dict], counts: np.ndarray, ctx: dict, *, n_pop: int, n_gen: int,
           decade_best_cfg: dict | None, log=print) -> tuple[np.ndarray, dict, list[float], int]:
    """Evolve exit policies. Returns (winner_effective_policy, archive, best_history, n_unique_evaluated).

    archive maps effective-policy bytes -> (fitness, mean_sharpe, std_sharpe).
    """
    rng = np.random.default_rng(SEED)
    memo: dict = {}
    archive: dict = {}

    def eval_pop(pop):
        fits = np.empty(len(pop))
        for i, raw in enumerate(pop):
            eff = resolve_fallback(snap(raw), counts)
            key = eff.tobytes()
            cached = archive.get(key)
            if cached is None:
                f, mn, sd = fitness(eff, specs, ctx, memo)
                archive[key] = (f, mn, sd)
            else:
                f, mn, sd = cached
            fits[i] = f
        return fits

    pop = init_population(rng, n_pop, decade_best_cfg)
    fits = eval_pop(pop)
    best, stale, history = -np.inf, 0, []
    for gen in range(n_gen):
        gbest = float(fits.max())
        history.append(gbest)
        if gbest > best + 1e-9:
            best, stale = gbest, 0
        else:
            stale += 1
        log(f"  gen {gen + 1:>2}/{n_gen}  best_fit={gbest:+.4f}  unique={len(archive)}  stale={stale}")
        if stale >= EARLY_STOP:
            log(f"  early stop — best fitness unchanged for {EARLY_STOP} generations")
            break
        elite = [pop[i].copy() for i in np.argsort(fits)[-N_ELITE:]]
        children = []
        while len(children) < n_pop - len(elite):
            a, b = pop[tournament(fits, rng)], pop[tournament(fits, rng)]
            children.append(mutate(crossover(a, b, rng), rng))
        pop = elite + children
        fits = eval_pop(pop)

    winner_key = max(archive.items(), key=lambda kv: kv[1][0])[0]
    winner = np.frombuffer(winner_key, dtype=float).reshape(N_CELLS, 3).copy()
    return winner, archive, history, len(archive)


def pareto_front(archive: dict, top_n: int = 20) -> list[dict]:
    """Top-N by fitness, reduced to the non-dominated (max mean_sharpe, min std_sharpe) set."""
    items = sorted(archive.items(), key=lambda kv: -kv[1][0])[:top_n]
    pts = [(mn, sd, key, f) for key, (f, mn, sd) in items]
    front = []
    for mn, sd, key, f in sorted(pts, key=lambda x: (-x[0], x[1])):
        dominated = any(o_mn >= mn and o_sd <= sd and (o_mn > mn or o_sd < sd) for o_mn, o_sd, _, _ in front)
        if not dominated:
            front.append((mn, sd, key, f))
    return [{"mean_sharpe": mn, "robustness_std": sd, "fitness": f,
             "policy_key": key} for mn, sd, key, f in front]


def knee_of(front: list[dict]) -> dict:
    """Knee = front point with min normalised distance to the (max mean, min std) ideal corner."""
    if len(front) == 1:
        return front[0]
    mn = np.array([p["mean_sharpe"] for p in front])
    sd = np.array([p["robustness_std"] for p in front])
    mn_n = (mn - mn.min()) / (np.ptp(mn) + 1e-9)
    sd_n = (sd - sd.min()) / (np.ptp(sd) + 1e-9)
    dist = np.sqrt((1 - mn_n) ** 2 + sd_n ** 2)
    return front[int(np.argmin(dist))]


# ── gauntlet on the winner ───────────────────────────────────────────────── #

def winner_pnls(specs: list[dict], policy: np.ndarray) -> np.ndarray:
    return np.array([replay_policy(s, policy) for s in specs])


def static_pnls(specs: list[dict]) -> np.ndarray:
    return np.array([s["static_pnl"] for s in specs])


def _portfolio_all(specs: list[dict], pnls: np.ndarray) -> float:
    if not specs:
        return 0.0
    idx = np.arange(len(specs))
    pairs = np.array([s["pair"] for s in specs])
    edt = np.array([s["entry_dt"] for s in specs])
    xdt = np.array([s["exit_dt"] for s in specs])
    return portfolio_sharpe(idx, pnls, pairs, edt, xdt)


def holdout_permutation(winner: np.ndarray, hold_specs: list[dict], n_perm: int, log=print) -> dict:
    """Null: the 27-cell mapping carries no OOS information. Shuffle the winner's cell rows,
    recompute the holdout portfolio Sharpe (and per-(vix x atr)-marginal Sharpe). Permuting on
    the holdout — never seen by the GA — is an honest OOS test of the conditioning structure."""
    if len(hold_specs) < 2:
        return {"n": len(hold_specs), "obs_sharpe": 0.0, "portfolio_p": 1.0, "marginal_p": {}, "n_perm": 0}
    idx = np.arange(len(hold_specs))
    pairs = np.array([s["pair"] for s in hold_specs])
    edt = np.array([s["entry_dt"] for s in hold_specs])
    xdt = np.array([s["exit_dt"] for s in hold_specs])
    margs = np.array([s["marginal"] for s in hold_specs])
    pop_margs = [m for m in np.unique(margs) if (margs == m).sum() >= 2]

    def port_and_marg(pol):
        pnls = winner_pnls(hold_specs, pol)
        port = portfolio_sharpe(idx, pnls, pairs, edt, xdt)
        mp = {int(m): pooled_sharpe(pnls[margs == m], edt[margs == m], xdt[margs == m]) for m in pop_margs}
        return port, mp

    obs_port, obs_marg = port_and_marg(winner)
    rng = np.random.default_rng(SEED)
    ge_port = 0
    ge_marg = {int(m): 0 for m in pop_margs}
    for i in range(n_perm):
        shuffled = winner[rng.permutation(N_CELLS)]
        p_port, p_marg = port_and_marg(shuffled)
        if p_port >= obs_port:
            ge_port += 1
        for m in pop_margs:
            if p_marg[int(m)] >= obs_marg[int(m)]:
                ge_marg[int(m)] += 1
        if log and (i + 1) % max(n_perm // 4, 1) == 0:
            log(f"    permutation {i + 1}/{n_perm}")
    port_p = (ge_port + 1) / (n_perm + 1)
    marg_p = {m: (c + 1) / (n_perm + 1) for m, c in ge_marg.items()}
    return {"n": len(hold_specs), "obs_sharpe": obs_port, "portfolio_p": port_p,
            "marginal_p": marg_p, "n_perm": n_perm}


def prove_band_gate(winner_full_decade: float, winner_oos: float,
                    full_band=PROVE_FULL_DECADE_BAND, oos_band=PROVE_OOS_BAND) -> dict:
    fd_ok = full_band[0] <= winner_full_decade <= full_band[1]
    oos_ok = oos_band[0] <= winner_oos <= oos_band[1]
    return {"winner_full_decade": winner_full_decade, "full_decade_band": list(full_band), "full_decade_ok": bool(fd_ok),
            "winner_oos": winner_oos, "oos_band": list(oos_band), "oos_ok": bool(oos_ok),
            "cleared": bool(fd_ok and oos_ok)}


# ── reconcile guard (the prove.py number, in-process) ────────────────────── #

def reconcile_gate() -> float:
    """Canonical decade backtest; weighted portfolio Sharpe must match 0.6886 ± 0.01 (== prove.py)."""
    bt = ForexBacktester(start=DECADE[0], end=DECADE[1])
    results = bt.backtest_all()
    ws = weighted_portfolio_sharpe([(r.sharpe, r.total_trades) for r in results])
    print(f"  reconcile: weighted portfolio Sharpe = {ws}  (target {RECON_TARGET} ± {RECON_TOL})")
    if abs(ws - RECON_TARGET) > RECON_TOL:
        raise SystemExit(f"RECONCILE FAILED — harness Sharpe {ws} != {RECON_TARGET}±{RECON_TOL}. Halting (config/data drift).")
    return ws


# ── ledger append (mirrors research_factory._append_to_ledger) ───────────── #

def append_to_ledger(entry: dict) -> str:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = LEDGER.with_suffix(f".bak-{stamp}.json")
    if LEDGER.exists():
        shutil.copy2(LEDGER, bak)
    ledger.append(entry)
    tmp = LEDGER.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(LEDGER)
    return str(bak)


def policy_to_records(policy: np.ndarray) -> list[dict]:
    out = []
    for vix in range(3):
        for atr in range(3):
            for h in range(3):
                c = cell_index(vix, atr, h)
                out.append({"vix_tercile": vix, "atr_tercile": atr, "hold_bucket": ["early", "mid", "late"][h],
                            "stop_atr_mult": round(float(policy[c, 0]), 3),
                            "trailing_atr_mult": round(float(policy[c, 1]), 3),
                            "hold_limit": int(round(policy[c, 2]))})
    return out


# ── main study ───────────────────────────────────────────────────────────── #

def main() -> int:
    ap = argparse.ArgumentParser(description="HYP-067 evolutionary exit-policy search (data-only).")
    ap.add_argument("--sign", action="store_true", help="freeze the prereg hash (run once before the study)")
    ap.add_argument("--standalone", action="store_true",
                    help="run as an independent hypothesis even though HYP-066 was not VALID_EDGE")
    ap.add_argument("--pop", type=int, default=POP_DEFAULT, help="GA population size")
    ap.add_argument("--generations", type=int, default=GEN_DEFAULT, help="GA generations")
    ap.add_argument("--perms", type=int, default=N_PERM_DEFAULT, help="holdout permutations for the gauntlet")
    args = ap.parse_args()
    if args.sign:
        sign_preregister()
        return 0

    print("HYP-067 — evolutionary exit-policy search (pre-registered, data-only)")
    verify_preregister()

    # ── gate on HYP-066 ──
    prior = hyp066_verdict()
    print(f"  HYP-066 prior verdict: {prior}")
    if gate_should_halt(prior, args.standalone):
        print("\n" + "=" * 70)
        print("  GATE CLOSED — HYP-066 (exit_regime_conditioning) is not VALID_EDGE.")
        print("  The pre-registered gate halts the evolutionary search: there is no validated")
        print("  regime-conditioning signal to build on. This is the expected, disciplined null.")
        print("  To evaluate exit-policy evolution as an INDEPENDENT pre-registered hypothesis,")
        print("  re-run with --standalone (stricter, search-size deflated-Sharpe correction).")
        print("=" * 70)
        return 0
    if args.standalone and prior != "VALID_EDGE":
        print("  --standalone: running as independent HYP-067 (gauntlet + prove band are the sole decider)")

    # ── 1) reconcile + capture decade trades ──
    ws = reconcile_gate()
    decade_backup = TRADES_FILE.read_text()
    results_backup = RESULTS_PATH.read_text() if RESULTS_PATH.exists() else None
    decade_trades = json.loads(decade_backup)

    # ── 2) caches + VIX percentile series (decade) ──
    print("  building per-pair caches (decade)…")
    bt_dec = ForexBacktester(start=DECADE[0], end=DECADE[1])
    caches = {p: a for p in PAIRS if (a := pair_arrays(bt_dec, p)) is not None}
    import yfinance as yf
    vix = yf.download("^VIX", start="2014-01-01", end=FORWARD[1], progress=False, auto_adjust=True)
    if hasattr(vix.columns, "get_level_values"):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_pct = _trailing_pct(vix["Close"])

    # ── 3) tercile thresholds from decade ENTRY values, then build specs ──
    entry_vix, entry_atr = [], []
    for pair, tl in decade_trades.items():
        if pair not in caches:
            continue
        arr = caches[pair]
        for t in tl:
            ed = pd.Timestamp(t["entry_date"])
            vp, ap_ = vix_pct.asof(ed), arr["atr_pct"].asof(ed)
            if not pd.isna(vp) and not pd.isna(ap_):
                entry_vix.append(float(vp))
                entry_atr.append(float(ap_))
    vix_terciles = (float(np.quantile(entry_vix, 1 / 3)), float(np.quantile(entry_vix, 2 / 3)))
    atr_terciles = (float(np.quantile(entry_atr, 1 / 3)), float(np.quantile(entry_atr, 2 / 3)))
    print(f"  vix terciles={tuple(round(x, 3) for x in vix_terciles)}  atr terciles={tuple(round(x, 3) for x in atr_terciles)}")

    print("  building trade specs (decade)…")
    specs = build_trade_specs(decade_trades, caches, vix_pct, vix_terciles, atr_terciles)
    counts = visit_counts(specs)
    ctx = build_fitness_ctx(specs)
    print(f"  N={len(specs)} trades · {n_backtest_paths(CPCV_GROUPS, CPCV_TEST)} CPCV paths · "
          f"populated cells={int((counts >= MIN_CELL_VISITS).sum())}/{N_CELLS}")

    # ── 4) seed + run the GA ──
    decade_best_cfg = None
    if HYP066_RESULTS.exists():
        decade_best_cfg = json.loads(HYP066_RESULTS.read_text()).get("decade_best_cfg")
    print(f"  running GA  pop={args.pop} generations={args.generations} seed={SEED}…")
    winner, archive, history, n_unique = run_ga(
        specs, counts, ctx, n_pop=args.pop, n_gen=args.generations, decade_best_cfg=decade_best_cfg)
    win_fit, win_mean, win_std = archive[winner.tobytes()]
    front = pareto_front(archive)
    knee = knee_of(front)
    winner = np.frombuffer(knee["policy_key"], dtype=float).reshape(N_CELLS, 3).copy()
    win_fit, win_mean, win_std = archive[knee["policy_key"]]
    print(f"  winner (knee): fitness={win_fit:+.4f}  mean_path_sharpe={win_mean:+.4f}  robustness_std={win_std:.4f}")

    # ── 5) winner train pooled Sharpe + deflated Sharpe (n_trials = search size) ──
    train_pnls = winner_pnls(specs, winner)
    winner_full_decade = _portfolio_all(specs, train_pnls)
    static_full_decade = _portfolio_all(specs, static_pnls(specs))
    dsr, dsr_prob = deflated_sharpe_ratio(winner_full_decade, n_trials=max(n_unique, 2), n_obs=1)

    # ── 6) forward holdout (2025-26) — overwrites the trades file; restore after ──
    print("  forward holdout (2025-2026)…")
    bt_fwd = ForexBacktester(start=FORWARD[0], end=FORWARD[1])
    bt_fwd.backtest_all()
    fwd_trades = json.loads(TRADES_FILE.read_text())
    caches_fwd = {p: a for p in PAIRS if (a := pair_arrays(bt_fwd, p)) is not None}
    hold_specs = build_trade_specs(fwd_trades, caches_fwd, vix_pct, vix_terciles, atr_terciles)
    TRADES_FILE.write_text(decade_backup)
    if results_backup is not None:
        RESULTS_PATH.write_text(results_backup)

    hold_winner_pnls = winner_pnls(hold_specs, winner)
    winner_oos = _portfolio_all(hold_specs, hold_winner_pnls)
    static_oos = _portfolio_all(hold_specs, static_pnls(hold_specs))
    non_degrade = bool(winner_oos >= static_oos - NON_DEGRADE_TOL)

    # ── 7) holdout permutation + BH across (vix x atr) marginals ──
    print(f"  holdout permutation (N={args.perms})…")
    perm = holdout_permutation(winner, hold_specs, args.perms)
    marg_items = sorted(perm["marginal_p"].items())
    bh_survive = benjamini_hochberg([p for _, p in marg_items], alpha=0.05) if marg_items else []
    n_bh = int(sum(bh_survive))

    # ── 8) prove reconciliation band ──
    prove = prove_band_gate(winner_full_decade, winner_oos)

    # ── 9) verdict ──
    passes = {
        "holdout_permutation_p<0.05": bool(perm["portfolio_p"] < 0.05),
        "deflated_sr>0": bool(dsr > 0),
        "bh_survivors>=1": bool(n_bh >= 1),
        "forward_non_degrade": non_degrade,
        "prove_band_cleared": bool(prove["cleared"]),
    }
    if not prove["cleared"]:
        verdict = "NOT_ROBUST"
    elif all(passes.values()):
        verdict = "VALID_EDGE"
    else:
        verdict = "NOT_SIGNIFICANT"

    # ── 10) write artifacts ──
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "id": "HYP-067", "verdict": verdict, "passes": passes, "standalone": bool(args.standalone),
        "hyp066_prior": prior, "reconcile_weighted_sharpe": ws,
        "n_trades_decade": len(specs), "n_trades_holdout": len(hold_specs),
        "terciles": {"vix": list(vix_terciles), "atr": list(atr_terciles)},
        "cell_visit_counts": counts.tolist(), "min_cell_visits": MIN_CELL_VISITS,
        "ga": {"pop": args.pop, "generations": args.generations, "n_generations_run": len(history),
               "n_unique_policies": n_unique, "best_history": history, "seed": SEED},
        "winner": {"fitness": win_fit, "mean_path_sharpe": win_mean, "robustness_std": win_std,
                   "full_decade_sharpe": winner_full_decade, "oos_sharpe": winner_oos},
        "static_baseline": {"full_decade_sharpe": static_full_decade, "oos_sharpe": static_oos},
        "deflated_sharpe": {"deflated_sr": dsr, "prob": dsr_prob, "n_trials": int(max(n_unique, 2))},
        "permutation": perm,
        "benjamini_hochberg": {"marginal_p": dict(marg_items),
                               "survive": {m: bool(s) for (m, _), s in zip(marg_items, bh_survive)}, "n_survive": n_bh},
        "forward_gate": {"n": len(hold_specs), "winner_sharpe": winner_oos, "static_sharpe": static_oos,
                         "non_degrade": non_degrade},
        "prove_gate": prove,
    }
    OUT_RESULTS.write_text(json.dumps(result, indent=2, default=str))
    OUT_PARETO.write_text(json.dumps(
        [{k: v for k, v in p.items() if k != "policy_key"} for p in front], indent=2, default=str))
    OUT_WINNER.write_text(json.dumps({
        "id": "HYP-067", "verdict": verdict,
        "fitness": win_fit, "mean_path_sharpe": win_mean, "robustness_std": win_std,
        "full_decade_sharpe": winner_full_decade, "oos_sharpe": winner_oos,
        "policy": policy_to_records(winner),
        "note": "NO live deployment without Colin's explicit sign-off (NON-NEGOTIABLE #4 + prereg).",
    }, indent=2, default=str))

    # ── 11) ledger ──
    bak = append_to_ledger({
        "id": "HYP-067", "name": "exit_policy_evolution (GA over 3x3x3 regime/age-keyed exit policy)",
        "status": verdict, "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "manual", "validator": "exit_policy_evolution",
        "mechanism": "GA-evolved per-bar regime+age-conditioned exit policy vs static; CPCV fitness, "
                     "holdout permutation, deflated-Sharpe at search size, prove band.",
        "result": {"perm_p": perm["portfolio_p"], "oos_sharpe": winner_oos,
                   "walkforward": "ROBUST" if non_degrade else "DEGRADED", "nperm": args.perms},
        "oos_sharpe": winner_oos, "is_sharpe": winner_full_decade,
        "p_value": perm["portfolio_p"], "bh_survives": bool(n_bh >= 1),
        "standalone": bool(args.standalone), "auto_generated": False,
    })

    # ── 12) one-screen print ──
    print("\n" + "=" * 70)
    print(f"  HYP-067 — EVOLUTIONARY EXIT POLICY — {verdict}")
    print("=" * 70)
    print(f"  Reconcile decade Sharpe : {ws}  (target {RECON_TARGET})")
    print(f"  Trades decade / holdout : {len(specs)} / {len(hold_specs)}")
    print(f"  Winner fitness          : {win_fit:+.4f}  (mean path {win_mean:+.3f}, robustness {win_std:.3f})")
    print(f"  Full-decade Sharpe      : winner {winner_full_decade:+.3f}  vs static {static_full_decade:+.3f}")
    print(f"  Holdout 2025-26 Sharpe  : winner {winner_oos:+.3f}  vs static {static_oos:+.3f}  non-degrade={non_degrade}")
    print(f"  Holdout permutation p   : {perm['portfolio_p']:.4f}  (N={args.perms})")
    print(f"  Deflated SR (n={int(max(n_unique, 2))}) : {dsr:+.3f}  P(SR>0)={dsr_prob:.3f}")
    print(f"  BH survivors            : {n_bh}/{len(marg_items)} marginals")
    print(f"  Prove band              : full-decade {prove['full_decade_ok']}  oos {prove['oos_ok']}  → cleared={prove['cleared']}")
    print("  Gates:", "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in passes.items()))
    print("=" * 70)
    print(f"  → {OUT_RESULTS}")
    print(f"  → {OUT_WINNER}  /  {OUT_PARETO}")
    print(f"  ledger += HYP-067 ({verdict})  backup {bak}")
    if verdict != "VALID_EDGE":
        print("  Honest null (the pre-registered expectation). Static config stays; no deployment.")
    else:
        print("  VALID_EDGE — see winner policy. Deploy ONLY via a separate logged param_change + sign-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
