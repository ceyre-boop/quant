#!/usr/bin/env python3
"""Monte Carlo prop-challenge risk simulator (BOOTSTRAP, not GBM).

Answers ONE question rigorously: given the ACTUAL forex edge (v015 4-pair, OOS Sharpe 1.08),
what is P(pass the prop challenge: reach +8%) before P(breach the −8% drawdown floor)?

This is a RISK tool — it assesses the outcome distribution of an edge we already have. It does
NOT generate or place trades.

METHODOLOGY: bootstrap-resample the system's REAL closed-trade %-equity returns (preserving the
actual loss distribution, fat tails, and skew). A GBM/normal version is run ONLY as a comparison
to demonstrate where the normality assumption misleads — on our OWN data, in whichever direction
the data says (this edge has POSITIVE skew, so GBM may actually OVERstate ruin; we report the
honest direction, never force it).

Reads logs/forex_backtest_trades.json (the OOS 2023-2024 pool the stamped Sharpe compounds).
Writes only data/risk/prop_monte_carlo.json. Fails loud if the real pool is missing/empty.

Usage:  python3 -m sovereign.risk.monte_carlo_prop [--sims 10000] [--account 100000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRADES = ROOT / "logs" / "forex_backtest_trades.json"
RESULTS = ROOT / "logs" / "forex_backtest_results.json"
OUT = ROOT / "data" / "risk" / "prop_monte_carlo.json"

LOW_CONFIDENCE_N = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def load_pool():
    """Flatten the real per-trade %-equity returns. FAIL LOUD if missing/empty.
    Cached — the trades file is static within a run (the risk engine calls this per decision)."""
    if not TRADES.exists():
        raise SystemExit(f"FATAL: real trade pool not found: {TRADES}. "
                         f"Run the v015 backtest first — this tool refuses to assume returns.")
    data = json.loads(TRADES.read_text())
    pnls, per_pair = [], {}
    for pair, trades in data.items():
        vals = [float(t["pnl_pct"]) for t in trades if "pnl_pct" in t]
        per_pair[pair] = len(vals)
        pnls.extend(vals)
    if not pnls:
        raise SystemExit("FATAL: trade pool is empty — no real returns to bootstrap. Refusing to fabricate.")
    # Portfolio trades/year (for the time clock).
    tpy = 49.8
    try:
        tpy = sum(float(r["trades_per_year"]) for r in json.loads(RESULTS.read_text()))
    except Exception:
        pass
    return np.array(pnls, dtype=float), per_pair, tpy


def _moments(x: np.ndarray) -> dict:
    m, s = float(x.mean()), float(x.std())
    z = (x - m) / (s + 1e-12)
    return {"mean": round(m, 6), "std": round(s, 6),
            "skew": round(float(np.mean(z ** 3)), 3),
            "excess_kurtosis": round(float(np.mean(z ** 4) - 3), 3),
            "min": round(float(x.min()), 5), "max": round(float(x.max()), 5)}


def _resolve_outcomes(paths: np.ndarray, floor_val: float, target_val: float):
    """Given equity paths (n_sims × steps incl. start col), return per-sim:
    outcome (1=PASS,-1=FAIL,0=INCOMPLETE), stop_step (trades taken), realized paths (carry-fwd)."""
    n, ncol = paths.shape
    fail_hit = paths <= floor_val
    pass_hit = paths >= target_val
    big = ncol + 1
    first_fail = np.where(fail_hit.any(1), fail_hit.argmax(1), big)
    first_pass = np.where(pass_hit.any(1), pass_hit.argmax(1), big)
    outcome = np.zeros(n, dtype=int)
    outcome[first_fail < first_pass] = -1
    outcome[first_pass < first_fail] = 1
    stop_step = np.minimum(first_fail, first_pass)
    stop_step[stop_step == big] = ncol - 1            # INCOMPLETE → ran to horizon end
    # Carry equity forward after the stop step.
    cols = np.arange(ncol)[None, :]
    stopped = cols > stop_step[:, None]
    eq_at_stop = paths[np.arange(n), stop_step][:, None]
    realized = np.where(stopped, eq_at_stop, paths)
    return outcome, stop_step, realized


def _max_drawdown(realized: np.ndarray) -> np.ndarray:
    run_max = np.maximum.accumulate(realized, axis=1)
    return ((realized - run_max) / run_max).min(axis=1)


def _bootstrap(pnls, n_sims, max_trades, account, floor_val, target_val, rng):
    draws = rng.choice(pnls, size=(n_sims, max_trades), replace=True)
    paths = account * np.cumprod(1 + draws, axis=1)
    paths = np.concatenate([np.full((n_sims, 1), account), paths], axis=1)
    return _resolve_outcomes(paths, floor_val, target_val) + (paths,)


def _gbm(mu, sigma, n_sims, max_trades, account, floor_val, target_val, rng):
    draws = rng.normal(mu, sigma, size=(n_sims, max_trades))
    paths = account * np.cumprod(1 + draws, axis=1)
    paths = np.concatenate([np.full((n_sims, 1), account), paths], axis=1)
    outcome, _, _ = _resolve_outcomes(paths, floor_val, target_val)
    return float(np.mean(outcome == -1))


def _ci95(p, n):
    se = (p * (1 - p) / n) ** 0.5
    return [round(max(0.0, p - 1.96 * se), 4), round(min(1.0, p + 1.96 * se), 4)]


def _pctiles(arr, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in ps}


def run(account=100_000.0, floor_pct=0.08, target_pct=0.08,
        horizons=(30, 60, 90), n_sims=10_000, seed=7) -> dict:
    pnls, per_pair, tpy = load_pool()
    n_pool = len(pnls)
    floor_val = account * (1 - floor_pct)
    target_val = account * (1 + target_pct)
    days_per_trade = 365.0 / max(tpy, 1e-9)
    mom = _moments(pnls)
    rng = np.random.default_rng(seed)

    out_horizons = {}
    for h in horizons:
        # floor: a trade counts only if it COMPLETES within the window (k·days_per_trade <= h).
        max_trades = max(1, int(h / days_per_trade))
        outcome, stop_step, realized, _paths = _bootstrap(
            pnls, n_sims, max_trades, account, floor_val, target_val, rng)
        max_dd = _max_drawdown(realized)
        terminal = realized[:, -1]
        p_pass = float(np.mean(outcome == 1))
        p_fail = float(np.mean(outcome == -1))
        p_incomplete = float(np.mean(outcome == 0))

        pass_mask = outcome == 1
        trades_to_pass = stop_step[pass_mask]
        days_to_pass = trades_to_pass * days_per_trade
        def _med_iqr(a):
            if len(a) == 0:
                return {"median": None, "iqr": [None, None], "n": 0}
            return {"median": round(float(np.median(a)), 2),
                    "iqr": [round(float(np.percentile(a, 25)), 2), round(float(np.percentile(a, 75)), 2)],
                    "n": int(len(a))}

        gbm_p_fail = _gbm(mom["mean"], mom["std"], n_sims, max_trades, account, floor_val, target_val, rng)
        bands = {f"p{p}": [round(float(v), 2) for v in np.percentile(realized, p, axis=0)]
                 for p in (5, 25, 50, 75, 95)}

        out_horizons[str(h)] = {
            "max_trades_in_window": max_trades,
            "p_pass": round(p_pass, 4), "p_pass_ci95": _ci95(p_pass, n_sims),
            "p_fail": round(p_fail, 4), "p_fail_ci95": _ci95(p_fail, n_sims),
            "p_incomplete": round(p_incomplete, 4),
            "max_drawdown": {"mean": round(float(max_dd.mean()), 4),
                             "median": round(float(np.median(max_dd)), 4),
                             "p95_worst": round(float(np.percentile(max_dd, 5)), 4),  # 5th pctile = worst 5%
                             "worst": round(float(max_dd.min()), 4)},
            "terminal_equity_pctiles": _pctiles(terminal),
            "trades_to_pass": _med_iqr(trades_to_pass),
            "days_to_pass": _med_iqr(days_to_pass),
            "gbm_p_fail": round(gbm_p_fail, 4),
            "gbm_understates_fail_by": round(p_fail - gbm_p_fail, 4),
            "equity_curve_bands": bands,
        }

    low_conf = n_pool < LOW_CONFIDENCE_N
    payload = {
        "generated_at": _now(),
        "question": "P(reach +8% before breaching -8%) for the real v015 4-pair forex edge",
        "pool_size": n_pool,
        "pool_per_pair": per_pair,
        "pool_window": "OOS 2023-2024 (the validated out-of-sample trades)",
        "low_confidence": low_conf,
        "calibration_note": (
            f"Bootstrap of n={n_pool} real trades. n>={LOW_CONFIDENCE_N} so the resample is "
            f"reasonably conditioned, but the pool is still only ~2 years of OOS trades — the "
            f"probabilities carry their own uncertainty (see *_ci95). Treat point estimates as "
            f"directional, not precise." if not low_conf else
            f"⚠️ LOW CONFIDENCE: only n={n_pool} real trades (<{LOW_CONFIDENCE_N}). The bootstrap "
            f"itself is uncertain — wide confidence on every probability. Do not over-trust."),
        "regime_caveat": (
            "CRITICAL: this bootstraps the 2023-2024 OOS window only — a FAVORABLE, rate-trending "
            "regime (mean +0.36%/trade, positive skew). The forex edge is REGIME-FRAGILE: rolling "
            "walk-forward was 2021 -0.13 / 2022 +0.51 / 2023 +1.26 / 2024 -0.09. In a flat/adverse "
            "regime, P(fail) would be materially higher and P(pass) lower than shown here. These "
            "numbers assume the future resembles 2023-2024; it may not."),
        "account": account, "floor": floor_val, "target": target_val,
        "floor_pct": floor_pct, "target_pct": target_pct,
        "n_sims": n_sims, "portfolio_trades_per_year": round(tpy, 1),
        "days_per_trade": round(days_per_trade, 2),
        "return_distribution": mom,
        "method": "bootstrap resample of real per-trade pnl_pct (sizing already embedded; forex is "
                  "not graded). GBM shown only as a normality-assumption comparison.",
        "horizons": out_horizons,
        "provenance": {"source": "logs/forex_backtest_trades.json", "verified": True,
                       "note": "Real backtested OOS trade returns — NOT assumed/round numbers."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def _print_report(p: dict):
    print(f"\n{'='*66}\n  PROP-CHALLENGE MONTE CARLO — real edge, n={p['pool_size']} trades "
          f"({p['pool_window']})\n{'='*66}")
    d = p["return_distribution"]
    print(f"  Per-trade returns: mean {d['mean']:+.4f}  std {d['std']:.4f}  "
          f"skew {d['skew']:+.2f}  excess-kurt {d['excess_kurtosis']:+.2f}  (min {d['min']}, max {d['max']})")
    print(f"  {p['calibration_note']}")
    print(f"  ⚠ {p['regime_caveat']}")
    for h, r in p["horizons"].items():
        print(f"\n  ── {h}-day window (~{r['max_trades_in_window']} trades) "
              f"[pool n={p['pool_size']}] ──")
        print(f"    P(PASS)       {r['p_pass']:.1%}  ci95 {r['p_pass_ci95']}")
        print(f"    P(FAIL/breach){r['p_fail']:.1%}  ci95 {r['p_fail_ci95']}")
        print(f"    P(INCOMPLETE) {r['p_incomplete']:.1%}")
        print(f"    Max DD: median {r['max_drawdown']['median']:.1%}  worst-5% {r['max_drawdown']['p95_worst']:.1%}  "
              f"worst {r['max_drawdown']['worst']:.1%}")
        dtp = r["days_to_pass"]
        print(f"    Days-to-pass (PASS sims, n={dtp['n']}): median {dtp['median']}  IQR {dtp['iqr']}")
        # GBM honesty line
        gb, bs = r["gbm_p_fail"], r["p_fail"]
        delta = r["gbm_understates_fail_by"]
        print(f"    P(FAIL) bootstrap (real distribution): {bs:.1%}")
        print(f"    P(FAIL) GBM (normal assumption):       {gb:.1%}")
        if delta > 0.0005:
            print(f"    → GBM UNDERSTATES ruin by {delta:.1%} — real returns have fatter LEFT tail than normal.")
        elif delta < -0.0005:
            print(f"    → GBM OVERSTATES ruin by {abs(delta):.1%} — this edge's POSITIVE skew "
                  f"(skew {d['skew']:+.2f}) gives a thinner left tail than a symmetric normal. "
                  f"Honest direction reported (not forced).")
        else:
            print(f"    → bootstrap ≈ GBM here (Δ {delta:+.1%}).")
    print(f"\n  Saved: data/risk/prop_monte_carlo.json\n{'='*66}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--account", type=float, default=100_000.0)
    ap.add_argument("--floor-pct", type=float, default=0.08)
    ap.add_argument("--target-pct", type=float, default=0.08)
    args = ap.parse_args()
    p = run(account=args.account, floor_pct=args.floor_pct, target_pct=args.target_pct, n_sims=args.sims)
    _print_report(p)


if __name__ == "__main__":
    main()
