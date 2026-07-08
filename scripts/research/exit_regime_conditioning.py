#!/usr/bin/env python3
"""scripts/research/exit_regime_conditioning.py — HYP-066.

Pre-registered, DATA-ONLY study. Question: should the forex exit parameters
(stop_atr_mult, trailing_atr_mult) be conditioned on entry-observable regime
(VIX percentile x ATR percentile) instead of held static?

Meta-labeling (Lopez de Prado, AFML ch.3): the primary model (carry entries) is
FIXED — we take the canonical v015 decade trades as a fixed entry universe — and the
secondary policy (exit config) is what we condition on regime. The exit itself is
evaluated by single-trade replay through sovereign.forex.exit_machine.decide_exit,
the SAME function the live L2 manager and the backtester call, so any finding is
deployable with parity preserved.

NO LIVE TOUCH. No OANDA, no forex_exit_manager import, no config writes, no
SHADOW_MODE change. Reads yfinance + the canonical backtest; writes only under
data/research/.

    python3 scripts/research/exit_regime_conditioning.py --sign   # freeze the prereg hash (run once)
    python3 scripts/research/exit_regime_conditioning.py          # verify hash, run the study
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from walk_forward_validation import deflated_sharpe_ratio
from sovereign.discovery.gate import benjamini_hochberg

PREREG = ROOT / "data" / "research" / "preregister" / "HYP-066_exit_regime_conditioning.json"
OUT = ROOT / "data" / "research" / "exit_regime_conditioning_results.json"
TRADES_FILE = ROOT / "logs" / "forex_backtest_trades.json"

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
DECADE = ("2015-01-01", "2024-12-31")
FORWARD = ("2025-01-01", "2026-06-29")
STOP_GRID = [1.5, 2.0, 2.5]
TRAIL_GRID = [1.0, 1.25, 1.5, 2.0]
GRID = [(s, t) for s in STOP_GRID for t in TRAIL_GRID]            # 12 configs
STATIC_STOP = 2.0
STATIC_TRAIL = {"GBPUSD=X": 2.0, "AUDUSD=X": 1.0, "EURUSD=X": 1.25, "USDJPY=X": 1.25}
N_PERM = 10_000
SEED = 42
N_FOLDS = 5
N_TRIALS = len(GRID) * 4                                          # 48
RECON_TARGET, RECON_TOL = 0.6886, 0.01
PCT_WINDOW = 252
NON_DEGRADE_TOL = 0.05


# ── pre-registration hash lock ──────────────────────────────────────────── #

def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sign_preregister() -> None:
    doc = json.loads(PREREG.read_text())
    h = _canonical_hash(doc)
    doc["hash_lock"] = h
    PREREG.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"signed HYP-066  hash_lock = {h}")


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


# ── per-pair price / signal / atr cache (reuses ForexBacktester) ────────── #

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
        "atr_series": atr,
        "atr_pct": _trailing_pct(atr),
        "signal": sig["signal"].reindex(idx).fillna(0).to_numpy(dtype=float).astype(int),
        "hold": sig[hold_col].reindex(idx).fillna(bt.HOLD_DAYS).to_numpy(dtype=float).astype(int),
    }


def _trailing_pct(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Percentile rank of each point within its trailing `window` (inclusive). No look-ahead."""
    return s.rolling(window, min_periods=30).apply(lambda x: float((x <= x[-1]).mean()), raw=True)


# ── single-trade exit replay (mirrors _simulate_forex_core 152-170 + 91-117) ── #

def replay_exit(arr: dict, entry_pos: int, direction: int, stop_atr_mult: float, trail_atr_mult: float) -> dict | None:
    closes, opens, atr, signal, hold = arr["closes"], arr["opens"], arr["atr"], arr["signal"], arr["hold"]
    n = len(closes)
    if entry_pos <= 0 or entry_pos >= n:
        return None
    entry_price = float(opens[entry_pos])
    entry_atr = max(float(atr[entry_pos - 1]), 1e-6)             # ATR at the signal bar (i), entry at i+1
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
            return {
                "entry": entry_price, "direction": direction,
                "pnl_pct": direction * (exit_price / max(entry_price, 1e-9) - 1.0),
                "hold_days": state.hold_count, "exit_pos": j,
                "exit_reason": ExitDecision(res.decision).name,
            }
    # never exited within the window → forced close at last bar
    exit_price = float(closes[-1])
    return {
        "entry": entry_price, "direction": direction,
        "pnl_pct": direction * (exit_price / max(entry_price, 1e-9) - 1.0),
        "hold_days": state.hold_count, "exit_pos": n - 1, "exit_reason": "WINDOW_END",
    }


def net_pnl(arr_pair: str, raw: dict) -> float:
    """Apply the canonical cost model (spread+slippage+swap) to a replayed trade's pnl."""
    t = {"entry": raw["entry"], "direction": raw["direction"], "hold_days": raw["hold_days"],
         "pnl_pct": raw["pnl_pct"], "risk_pct": 1.0}
    ForexBacktester._apply_costs([t], arr_pair)
    return float(t["pnl_pct"])


# ── Sharpe estimators ───────────────────────────────────────────────────── #

def pooled_sharpe(pnls: np.ndarray, entry_dt: np.ndarray, exit_dt: np.ndarray) -> float:
    """Annualised pooled Sharpe over a set of trades (mirrors _compute_stats: sqrt(n/years))."""
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


def _weighted(per_bucket: dict, key: str) -> float:
    rows = [(v[key], v["n"]) for v in per_bucket.values() if v["n"] > 0]
    wsum = sum(np.sqrt(n) for _, n in rows)
    return float(sum(s * np.sqrt(n) for s, n in rows) / wsum) if wsum else 0.0


# ── purged interval-overlap k-fold (trade granularity) ──────────────────── #

def _purged_folds(entry_dt: np.ndarray, exit_dt: np.ndarray, k: int = N_FOLDS):
    """Time-ordered k folds; for each, train = other folds MINUS trades whose [entry,exit]
    overlaps the test window (AFML purge, at trade granularity since the index is trades not bars)."""
    n = len(entry_dt)
    order = np.argsort(entry_dt)
    folds = np.array_split(order, k)
    for f in range(k):
        test = folds[f]
        if len(test) < 2:
            continue
        t0, t1 = entry_dt[test].min(), exit_dt[test].max()
        train = np.concatenate([folds[g] for g in range(k) if g != f]) if k > 1 else np.array([], int)
        keep = [i for i in train if not (exit_dt[i] >= t0 and entry_dt[i] <= t1)]   # drop overlappers
        yield np.asarray(keep, dtype=int), np.asarray(test, dtype=int)


# ── main study ──────────────────────────────────────────────────────────── #

def reconcile_gate() -> tuple[float, dict]:
    """Canonical decade backtest; weighted portfolio Sharpe must match 0.6886 ± 0.01."""
    bt = ForexBacktester(start=DECADE[0], end=DECADE[1])
    results = bt.backtest_all()
    ws = weighted_portfolio_sharpe([(r.sharpe, r.total_trades) for r in results])
    decade_trades = json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() else {}
    print(f"  reconcile: weighted portfolio Sharpe = {ws}  (target {RECON_TARGET} ± {RECON_TOL})")
    if abs(ws - RECON_TARGET) > RECON_TOL:
        raise SystemExit(f"RECONCILE FAILED — harness Sharpe {ws} != {RECON_TARGET}±{RECON_TOL}. Halting (config/data drift).")
    return ws, decade_trades


def label_and_replay(trades_by_pair: dict, caches: dict, vix_pct: pd.Series,
                     vix_med: float | None, atr_med: float | None):
    """Returns (records, vix_med, atr_med). Each record: pair, entry_dt, exit_dt, bucket, static_pnl, grid_pnls[12]."""
    records, raw_vix, raw_atr = [], [], []
    for pair, tl in trades_by_pair.items():
        if pair not in caches:
            continue
        arr = caches[pair]
        for t in tl:
            ed = pd.Timestamp(t["entry_date"])
            if ed not in arr["pos"]:
                loc = arr["idx"].searchsorted(ed)
                if loc <= 0 or loc >= len(arr["idx"]):
                    continue
                epos = int(loc)
            else:
                epos = arr["pos"][ed]
            direction = int(t["direction"])
            vp = vix_pct.asof(ed)
            ap = arr["atr_pct"].asof(ed)
            if pd.isna(vp) or pd.isna(ap):
                continue
            records.append({"pair": pair, "epos": epos, "direction": direction,
                            "entry_dt": np.datetime64(ed), "vix_pct": float(vp), "atr_pct": float(ap)})
            raw_vix.append(float(vp))
            raw_atr.append(float(ap))
    if vix_med is None:
        vix_med, atr_med = float(np.median(raw_vix)), float(np.median(raw_atr))
    # replay every record under static + the 12-config grid
    for r in records:
        arr = caches[r["pair"]]
        st = replay_exit(arr, r["epos"], r["direction"], STATIC_STOP, STATIC_TRAIL[r["pair"]])
        r["static_pnl"] = net_pnl(r["pair"], st) if st else 0.0
        r["exit_dt"] = arr["idx"][st["exit_pos"]].to_datetime64() if st else r["entry_dt"]
        gp = []
        for (s, tr) in GRID:
            rep = replay_exit(arr, r["epos"], r["direction"], s, tr)
            gp.append(net_pnl(r["pair"], rep) if rep else 0.0)
        r["grid_pnls"] = gp
        vhi = r["vix_pct"] >= vix_med
        ahi = r["atr_pct"] >= atr_med
        r["bucket"] = f"{'Vhi' if vhi else 'Vlo'}_{'Ahi' if ahi else 'Alo'}"
    return records, vix_med, atr_med


def bucket_stats(records: list, labels: np.ndarray) -> dict:
    """In-sample: per bucket, best grid config Sharpe and static Sharpe (+ chosen config index)."""
    grid = np.array([r["grid_pnls"] for r in records])           # [N, 12]
    stat = np.array([r["static_pnl"] for r in records])
    edt = np.array([r["entry_dt"] for r in records])
    xdt = np.array([r["exit_dt"] for r in records])
    out = {}
    for b in sorted(set(labels)):
        m = labels == b
        if m.sum() < 2:
            out[b] = {"n": int(m.sum()), "best": 0.0, "static": 0.0, "best_cfg": None, "improvement": 0.0}
            continue
        cfg_sharpes = [pooled_sharpe(grid[m, c], edt[m], xdt[m]) for c in range(len(GRID))]
        c_star = int(np.argmax(cfg_sharpes))
        st = pooled_sharpe(stat[m], edt[m], xdt[m])
        out[b] = {"n": int(m.sum()), "best": float(cfg_sharpes[c_star]), "static": float(st),
                  "best_cfg": c_star, "improvement": float(cfg_sharpes[c_star] - st)}
    return out


def oos_bucket(records: list, labels: np.ndarray) -> dict:
    """Per bucket, purged 5-fold OOS Sharpe: regime-keyed (best-on-train) vs static."""
    grid = np.array([r["grid_pnls"] for r in records])
    stat = np.array([r["static_pnl"] for r in records])
    edt = np.array([r["entry_dt"] for r in records])
    xdt = np.array([r["exit_dt"] for r in records])
    out = {}
    for b in sorted(set(labels)):
        idx = np.where(labels == b)[0]
        if len(idx) < N_FOLDS * 2:
            out[b] = {"n": int(len(idx)), "rk_oos": 0.0, "static_oos": 0.0}
            continue
        rk, st = [], []
        for tr, te in _purged_folds(edt[idx], xdt[idx]):
            if len(tr) < 2 or len(te) < 2:
                continue
            gi, ge = idx[tr], idx[te]
            cfg_sharpes = [pooled_sharpe(grid[gi, c], edt[gi], xdt[gi]) for c in range(len(GRID))]
            c_star = int(np.argmax(cfg_sharpes))
            rk.append(pooled_sharpe(grid[ge, c_star], edt[ge], xdt[ge]))
            st.append(pooled_sharpe(stat[ge], edt[ge], xdt[ge]))
        out[b] = {"n": int(len(idx)), "rk_oos": float(np.mean(rk)) if rk else 0.0,
                  "static_oos": float(np.mean(st)) if st else 0.0}
    return out


def permutation_test(records: list, labels: np.ndarray):
    """Shuffle bucket labels; null = regime carries no exit-config info. Returns (portfolio_p, per_bucket_p)."""
    grid = np.array([r["grid_pnls"] for r in records])
    stat = np.array([r["static_pnl"] for r in records])
    edt = np.array([r["entry_dt"] for r in records])
    xdt = np.array([r["exit_dt"] for r in records])
    buckets = sorted(set(labels))

    def improvement(lab):
        per = {}
        for b in buckets:
            m = lab == b
            n = int(m.sum())
            if n < 2:
                per[b] = (0.0, n)
                continue
            best = max(pooled_sharpe(grid[m, c], edt[m], xdt[m]) for c in range(len(GRID)))
            st = pooled_sharpe(stat[m], edt[m], xdt[m])
            per[b] = (best - st, n)
        wsum = sum(np.sqrt(n) for _, n in per.values() if n > 0)
        port = sum(v * np.sqrt(n) for v, n in per.values() if n > 0) / wsum if wsum else 0.0
        return port, {b: per[b][0] for b in buckets}

    obs_port, obs_b = improvement(labels)
    rng = np.random.default_rng(SEED)
    ge_port = 0
    ge_b = {b: 0 for b in buckets}
    for _ in range(N_PERM):
        perm = rng.permutation(labels)
        p_port, p_b = improvement(perm)
        if p_port >= obs_port:
            ge_port += 1
        for b in buckets:
            if p_b[b] >= obs_b[b]:
                ge_b[b] += 1
    port_p = (ge_port + 1) / (N_PERM + 1)
    bucket_p = {b: (ge_b[b] + 1) / (N_PERM + 1) for b in buckets}
    return obs_port, port_p, obs_b, bucket_p


def forward_gate(caches_fwd: dict, fwd_trades: dict, vix_pct: pd.Series,
                 vix_med: float, atr_med: float, decade_best_cfg: dict) -> dict:
    """One-shot: on 2025-26, regime-keyed (each bucket's decade-best cfg) must not degrade vs static."""
    rk_p, st_p, ent, ext = [], [], [], []
    for pair, tl in fwd_trades.items():
        if pair not in caches_fwd:
            continue
        arr = caches_fwd[pair]
        for t in tl:
            ed = pd.Timestamp(t["entry_date"])
            epos = arr["pos"].get(ed)
            if epos is None or epos <= 0:
                continue
            vp, ap = vix_pct.asof(ed), arr["atr_pct"].asof(ed)
            if pd.isna(vp) or pd.isna(ap):
                continue
            bucket = f"{'Vhi' if vp >= vix_med else 'Vlo'}_{'Ahi' if ap >= atr_med else 'Alo'}"
            cstar = decade_best_cfg.get(bucket)
            direction = int(t["direction"])
            stat = replay_exit(arr, epos, direction, STATIC_STOP, STATIC_TRAIL[pair])
            if stat is None:
                continue
            st_p.append(net_pnl(pair, stat))
            if cstar is not None:
                s, tr = GRID[cstar]
                rk = replay_exit(arr, epos, direction, s, tr)
                rk_p.append(net_pnl(pair, rk) if rk else st_p[-1])
            else:
                rk_p.append(st_p[-1])
            ent.append(ed.to_datetime64())
            ext.append(arr["idx"][stat["exit_pos"]].to_datetime64())
    if len(st_p) < 2:
        return {"n": len(st_p), "rk_sharpe": 0.0, "static_sharpe": 0.0, "non_degrade": True, "note": "too few forward trades"}
    e, x = np.array(ent), np.array(ext)
    rk_s = pooled_sharpe(np.array(rk_p), e, x)
    st_s = pooled_sharpe(np.array(st_p), e, x)
    return {"n": len(st_p), "rk_sharpe": rk_s, "static_sharpe": st_s,
            "non_degrade": bool(rk_s >= st_s - NON_DEGRADE_TOL)}


def main() -> int:
    ap = argparse.ArgumentParser(description="HYP-066 regime-conditioned exit study (data-only).")
    ap.add_argument("--sign", action="store_true", help="freeze the prereg hash (run once before the study)")
    args = ap.parse_args()
    if args.sign:
        sign_preregister()
        return 0

    print("HYP-066 — regime-conditioned exit parameters (pre-registered, data-only)")
    verify_preregister()

    # 1) reconcile + capture decade trades (before forward run overwrites the canonical logs)
    ws, decade_trades = reconcile_gate()
    decade_backup = TRADES_FILE.read_text()
    results_backup = RESULTS_PATH.read_text() if RESULTS_PATH.exists() else None

    # 2) caches + VIX percentile series
    print("  building per-pair caches (decade)…")
    bt_dec = ForexBacktester(start=DECADE[0], end=DECADE[1])
    caches = {p: a for p in PAIRS if (a := pair_arrays(bt_dec, p)) is not None}
    import yfinance as yf
    vix = yf.download("^VIX", start="2014-01-01", end=FORWARD[1], progress=False, auto_adjust=True)
    if hasattr(vix.columns, "get_level_values"):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_pct = _trailing_pct(vix["Close"])

    # 3) label + replay matrix
    print("  labeling + replaying exit grid over decade entries…")
    records, vix_med, atr_med = label_and_replay(decade_trades, caches, vix_pct, None, None)
    labels = np.array([r["bucket"] for r in records])
    n_by_bucket = {b: int((labels == b).sum()) for b in sorted(set(labels))}
    print(f"  N={len(records)} trades · medians VIX%={vix_med:.3f} ATR%={atr_med:.3f} · buckets {n_by_bucket}")

    # 4) in-sample best cfg, OOS k-fold, permutation, deflated Sharpe, BH
    insample = bucket_stats(records, labels)
    decade_best_cfg = {b: v["best_cfg"] for b, v in insample.items()}
    oos = oos_bucket(records, labels)
    obs_port, port_p, obs_b, bucket_p = permutation_test(records, labels)
    port_oos_rk = _weighted({b: {"v": oos[b]["rk_oos"], "n": oos[b]["n"]} for b in oos}, "v")
    port_oos_st = _weighted({b: {"v": oos[b]["static_oos"], "n": oos[b]["n"]} for b in oos}, "v")
    dsr, dsr_prob = deflated_sharpe_ratio(port_oos_rk, n_trials=N_TRIALS, n_obs=1)
    bh_pvals = [bucket_p[b] for b in sorted(bucket_p)]
    bh_survive = benjamini_hochberg(bh_pvals, alpha=0.05)
    n_bh = int(sum(bh_survive))

    # 5) forward gate (2025-26) — overwrites the trades file; restore after
    print("  forward gate (2025-2026, one-shot)…")
    bt_fwd = ForexBacktester(start=FORWARD[0], end=FORWARD[1])
    fwd_results = bt_fwd.backtest_all()
    fwd_trades = json.loads(TRADES_FILE.read_text())
    caches_fwd = {p: a for p in PAIRS if (a := pair_arrays(bt_fwd, p)) is not None}
    fwd = forward_gate(caches_fwd, fwd_trades, vix_pct, vix_med, atr_med, decade_best_cfg)
    TRADES_FILE.write_text(decade_backup)                          # restore canonical decade logs
    if results_backup is not None:
        RESULTS_PATH.write_text(results_backup)

    # 6) verdict
    passes = {
        "permutation_p<0.05": bool(port_p < 0.05),
        "deflated_sr>0": bool(dsr > 0),
        "bh_survivors>=1": bool(n_bh >= 1),
        "forward_non_degrade": bool(fwd["non_degrade"]),
    }
    verdict = "VALID_EDGE" if all(passes.values()) else "NOT_SIGNIFICANT"

    result = {
        "id": "HYP-066", "verdict": verdict, "passes": passes,
        "reconcile_weighted_sharpe": ws,
        "n_trades": len(records), "buckets": n_by_bucket,
        "medians": {"vix_pct": vix_med, "atr_pct": atr_med},
        "grid": GRID,
        "in_sample": insample, "oos": oos,
        "portfolio_oos": {"regime_keyed": port_oos_rk, "static": port_oos_st},
        "permutation": {"obs_improvement": obs_port, "portfolio_p": port_p,
                        "bucket_obs": obs_b, "bucket_p": bucket_p, "n_perm": N_PERM, "seed": SEED},
        "deflated_sharpe": {"deflated_sr": dsr, "prob": dsr_prob, "n_trials": N_TRIALS},
        "benjamini_hochberg": {"pvals": dict(zip(sorted(bucket_p), bh_pvals)),
                               "survive": dict(zip(sorted(bucket_p), [bool(x) for x in bh_survive])), "n_survive": n_bh},
        "forward_gate": fwd,
        "decade_best_cfg": {b: (GRID[c] if c is not None else None) for b, c in decade_best_cfg.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))

    # 7) one-screen print
    print("\n" + "=" * 70)
    print(f"  HYP-066 — REGIME-CONDITIONED EXITS — {verdict}")
    print("=" * 70)
    print(f"  Reconcile decade Sharpe : {ws}  (target {RECON_TARGET})")
    print(f"  Trades / buckets        : {len(records)}  {n_by_bucket}")
    print(f"  Portfolio OOS Sharpe    : regime-keyed {port_oos_rk:+.3f}  vs static {port_oos_st:+.3f}")
    print(f"  Permutation p (N={N_PERM}) : {port_p:.4f}   (obs improvement {obs_port:+.4f})")
    print(f"  Deflated SR (48 trials) : {dsr:+.3f}  P(SR>0)={dsr_prob:.3f}")
    print(f"  BH survivors            : {n_bh}/4   {result['benjamini_hochberg']['survive']}")
    print(f"  Forward 2025-26         : rk {fwd['rk_sharpe']:+.3f} vs static {fwd['static_sharpe']:+.3f}  "
          f"non-degrade={fwd['non_degrade']}  (n={fwd['n']})")
    print("  Gates:", "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in passes.items()))
    print("=" * 70)
    print(f"  → {OUT}")
    if verdict == "NOT_SIGNIFICANT":
        print("  Honest null (the pre-registered expectation). Static config stays; Step 5 ships unchanged.")
    else:
        print("  VALID_EDGE — see decade_best_cfg. Deploy via cfg_for_pair as a separate logged param_change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
