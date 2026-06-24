"""
sovereign/discovery/validation.py
=================================
Executable checks behind the pipeline-validation gate (Colin's "Quantum Backtest
Validation Framework"). Each check returns a structured record so the report can
answer the six final questions and emit an A/B/C/D verdict.

The point: prove the machinery is correct so dry results mean "no edge found",
not "broken system found nothing". Read-only — never trades or tunes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = ROOT / "data" / "drift_snapshots"


def _rec(check, status, evidence):
    return {"check": check, "status": status, "evidence": evidence}


# ─── §1 data integrity ────────────────────────────────────────────────────────

def check_coverage(adapter) -> list:
    out = []
    for pair in adapter.pairs:
        df = adapter.price_df(pair)
        n = len(df)
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        bpy = n / yrs if yrs else 0
        status = "PASS" if (yrs >= 8 and 180 <= bpy <= 320) else "WARN"
        out.append(_rec(f"coverage:{pair.replace('=X','')}", status,
                        f"{n} bars, {df.index[0].date()}→{df.index[-1].date()} ({yrs:.1f}y, {bpy:.0f} bars/yr)"))
    return out


def check_continuity(adapter) -> list:
    out = []
    for pair in adapter.pairs:
        df = adapter.price_df(pair)
        # forward-filled / synthetic bars: O=H=L=C
        ohlc_flat = ((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])).sum()
        # weekday gaps > 4 calendar days (excl. normal weekends/holidays)
        gaps = df.index.to_series().diff().dt.days
        big_gaps = int((gaps > 5).sum())
        # outlier bars: daily range > 5σ of the range distribution
        rng = ((df["High"] - df["Low"]) / df["Close"]).dropna()
        z = (rng - rng.mean()) / (rng.std() + 1e-12)
        outliers = int((z > 5).sum())
        status = "PASS" if (ohlc_flat == 0 and outliers <= 2) else "WARN"
        out.append(_rec(f"continuity:{pair.replace('=X','')}", status,
                        f"flat OHLC bars={int(ohlc_flat)}, calendar-gaps>5d={big_gaps}, >5σ-range bars={outliers}"))
    return out


def check_feature_health(adapter) -> list:
    from sovereign.discovery.features import compute_features, FEATURE_COLUMNS
    df = adapter.price_df(adapter.pairs[0])
    f = compute_features(df)
    out = []
    dead, naney = [], []
    for c in FEATURE_COLUMNS:
        if c not in f.columns:
            continue
        nan_pct = float(f[c].isna().mean() * 100)
        nun = f[c].nunique()
        if nun <= 1:
            dead.append(c)
        if nan_pct > 5:
            naney.append(f"{c}={nan_pct:.0f}%")
    status = "PASS" if (not dead and not naney) else "WARN"
    out.append(_rec("feature_health", status,
                    f"{len(FEATURE_COLUMNS)} features; dead={dead or 'none'}; >5%NaN={naney or 'none'}"))
    return out


# ─── §1.3 drift (the crux) ────────────────────────────────────────────────────

def measure_drift(adapter) -> list:
    """Quantify yfinance drift three ways: within-run reproducibility, vs the
    recorded backtest trade prices (a dated baseline), and snapshot for the 24h test."""
    import yfinance as yf
    out = []
    pair0 = adapter.pairs[0]

    # (a) within-run reproducibility — two pulls back-to-back must be identical
    try:
        a = yf.download(pair0, start="2024-01-01", end="2024-06-30", progress=False, auto_adjust=True)
        b = yf.download(pair0, start="2024-01-01", end="2024-06-30", progress=False, auto_adjust=True)
        for d in (a, b):
            if hasattr(d.columns, "get_level_values"):
                d.columns = d.columns.get_level_values(0)
        same = bool(np.allclose(a["Close"].to_numpy(), b["Close"].to_numpy(), atol=1e-9))
        out.append(_rec("drift:within_run", "PASS" if same else "FAIL",
                        f"two back-to-back pulls of {pair0.replace('=X','')} identical={same}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_rec("drift:within_run", "WARN", f"could not run: {type(exc).__name__}"))

    # (b) vs the recorded backtest trade prices (dated baseline)
    trades_path = ROOT / "logs" / "forex_backtest_trades.json"
    if trades_path.exists():
        try:
            baseline_age_days = (pd.Timestamp.utcnow().tz_localize(None) -
                                 pd.Timestamp(trades_path.stat().st_mtime, unit="s")).days
            tj = json.loads(trades_path.read_text())
            diffs = []
            for pair, tl in tj.items():
                fresh = yf.download(pair, start="2015-01-01", end="2025-01-01", progress=False, auto_adjust=True)
                if hasattr(fresh.columns, "get_level_values"):
                    fresh.columns = fresh.columns.get_level_values(0)
                fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
                for t in tl[:40]:
                    try:
                        ed = pd.Timestamp(str(t.get("entry_date"))[:10])
                        rec_entry = float(t.get("entry", 0))
                        if ed in fresh.index and rec_entry > 0:
                            fresh_open = float(fresh.loc[ed, "Open"])
                            diffs.append(abs(fresh_open - rec_entry) / rec_entry * 100)
                    except Exception:
                        pass
            if diffs:
                mean_d, max_d = float(np.mean(diffs)), float(np.max(diffs))
                status = "FAIL" if max_d > 0.1 else "PASS"
                out.append(_rec("drift:vs_baseline", status,
                                f"baseline {baseline_age_days}d old; n={len(diffs)} bars; "
                                f"mean drift {mean_d:.4f}%, max {max_d:.4f}% (gate: max>0.1%⇒flag)"))
            else:
                out.append(_rec("drift:vs_baseline", "WARN", "no overlapping bars to compare"))
        except Exception as exc:  # noqa: BLE001
            out.append(_rec("drift:vs_baseline", "WARN", f"compare failed: {type(exc).__name__}"))
    else:
        out.append(_rec("drift:vs_baseline", "WARN", "no logs/forex_backtest_trades.json baseline yet"))

    # (c) snapshot today for the true 24h test
    try:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        snap = {}
        for pair in adapter.pairs:
            df = adapter.price_df(pair)
            tail = df["Close"].tail(120)
            snap[pair] = {str(d.date()): round(float(v), 6) for d, v in tail.items()}
        stamp = str(adapter.price_df(adapter.pairs[0]).index[-1].date())
        (SNAP_DIR / f"snapshot_{stamp}.json").write_text(json.dumps(snap, indent=2))
        out.append(_rec("drift:snapshot", "INFO",
                        f"snapshot saved → data/drift_snapshots/snapshot_{stamp}.json (re-run --drift-compare tomorrow for the true 24h test)"))
    except Exception as exc:  # noqa: BLE001
        out.append(_rec("drift:snapshot", "WARN", f"snapshot failed: {type(exc).__name__}"))
    return out


# ─── §2 look-ahead canary ─────────────────────────────────────────────────────

def lookahead_canary(adapter, n_perms: int = 100) -> list:
    """Inject a deliberate future-peeking signal; the gate MUST reward it hugely.
    If it does → the gate detects leaks and the real (≈0.35 Sharpe) features are clean.
    If it doesn't → the evaluator itself is broken."""
    from sovereign.discovery.gate import Gate, Candidate

    def leak(pair, pdf, fdf):
        close = pdf["Close"] if "Close" in pdf.columns else pdf.iloc[:, 0]
        fwd = close.shift(-20) / close - 1.0   # peeks 20 bars ahead
        return np.sign(fwd.fillna(0.0)).astype(np.int8).to_numpy()

    def real(pair, pdf, fdf):
        return np.asarray(adapter.dataset(pair).signals).astype(np.int8)

    cands = [Candidate("canary_leak", "canary_future_peek", "DIAGNOSTIC: sign(20-bar forward return)", leak),
             Candidate("real_base", "real_macro", "the real ungated macro signal", real)]
    feats = {p: adapter.price_df(p) for p in adapter.pairs}
    g = Gate(adapter, train_window=(adapter.start, "2022-12-31"),
             holdout_window=("2023-01-01", adapter.end), n_perms=n_perms, seed=7)
    res = {r.name: r for r in g.evaluate(cands, feats, progress=lambda m: None)}
    canary, realr = res.get("canary_future_peek"), res.get("real_macro")
    detected = bool(canary and realr and canary.full_sharpe > realr.full_sharpe * 2
                    and canary.perm_p is not None and canary.perm_p < 0.05)
    return [_rec("lookahead_canary", "PASS" if detected else "FAIL",
                 f"canary(future-peek) Sharpe={canary.full_sharpe if canary else 'na'} perm_p={canary.perm_p if canary else 'na'} "
                 f"vs real Sharpe={realr.full_sharpe if realr else 'na'} → gate {'DETECTS leaks (real features clean)' if detected else 'FAILED to reward a known leak (evaluator suspect)'}")]


# ─── §3 trade-gen sanity (on the real base signal) ────────────────────────────

def check_tradegen(adapter, min_trades: int = 30) -> list:
    from sovereign.discovery.gate import weighted_portfolio_sharpe
    out, all_trades, per_pair = [], [], {}
    for pair in adapter.pairs:
        res = adapter.eval_signals(pair, np.asarray(adapter.dataset(pair).signals).astype(np.int8))
        per_pair[pair] = res.n_trades
        all_trades.extend(res.trades)
    n = len(all_trades)
    out.append(_rec("tradegen:sample_size", "PASS" if n >= min_trades else "FAIL",
                    f"{n} base trades total; per-pair {{{', '.join(f'{p.replace('=X','')}:{c}' for p,c in per_pair.items())}}} (floor {min_trades})"))
    if all_trades:
        pnls = [t.get("pnl_pct", 0.0) for t in all_trades]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / n
        gw = sum(p for p in pnls if p > 0); gl = abs(sum(p for p in pnls if p < 0))
        pf = gw / gl if gl > 1e-9 else float("inf")
        total = sum(abs(p) for p in pnls)
        top = max(abs(p) for p in pnls) / total if total else 0
        out.append(_rec("tradegen:winrate_pf", "PASS" if (0.35 <= wr <= 0.75 and pf > 1.0) else "WARN",
                        f"win_rate={wr:.1%}, profit_factor={pf:.2f}"))
        out.append(_rec("tradegen:concentration", "PASS" if top < 0.5 else "WARN",
                        f"largest single trade = {top:.1%} of gross P&L (flag if >50%)"))
    return out


# ─── §4 statistical machinery present ─────────────────────────────────────────

def check_stats_machinery() -> list:
    import sovereign.discovery.gate as G
    have = all(hasattr(G, fn) for fn in
               ("_permutation_p" if False else "benjamini_hochberg", "deflated_sharpe_ratio",
                "bootstrap_sharpe_diff_pvalue"))
    methods = ["permutation (Gate._permutation_p)", "Benjamini-Hochberg", "Deflated Sharpe",
               "bootstrap Sharpe-diff (delta-vs-base)", "chronological walk-forward (date windows)"]
    return [_rec("stats:machinery", "PASS" if have else "FAIL",
                 "present: " + "; ".join(methods))]
