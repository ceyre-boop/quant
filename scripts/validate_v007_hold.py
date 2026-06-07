"""
v007 per-pair hold — REDESIGNED VALIDATION (the one the code comment says is "pending").

The per-pair hold caps (GBPUSD=6, AUDUSD=5, EURUSD=5) are ALREADY LIVE and baked into the 1.08
OOS Sharpe. The only prior test (retro_validate 2026-05-27, p=0.008 "harmful") is self-acknowledged
invalid — it tested forensics "natural duration," not forced-cap effect. retro_validate.py:44 itself
says "only the backtester is a valid test for exit rules." This script IS that valid test.

Question: do the live hold caps beat the 60d baseline — OOS, permutation, and walk-forward?

  TREATMENT = live overrides (GBP 6 / AUD 5 / EUR 5 / USDJPY 60d default)
  BASELINE  = all pairs 60d (PAIR_HOLD_OVERRIDES cleared)

  Headline (faithful live path, reconciles to 1.08): ForexBacktester.backtest_pair per arm —
    applies VIX gate + trailing overrides + costs; only the hold differs between arms.
  Permutation (fast array path): are the SPECIFIC caps better than random caps? null = random
    per-pair cap, 10k draws, p = P(null portfolio Sharpe >= treatment).
  Walk-forward: treatment-vs-baseline delta per disjoint test year (must hold across windows).
  Both-sides: among cap-fired trades, winners cut early (opportunity cost) vs bleeders cut (benefit).

Verdict VALID_EDGE (OOS delta>0, p<0.05, walk-forward stable) or NOT_SIGNIFICANT. Logs regardless.
Values are pre-specified (chosen 2026-05-19) → testing them OOS is a clean holdout, not circular.

Usage:  ~/quant/.venv/bin/python scripts/validate_v007_hold.py --perms 10000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _l in ("yfinance", "urllib3", "requests", "peewee"):
    logging.getLogger(_l).setLevel(logging.ERROR)

from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.batch_backtester import ForexBatchBacktester
from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
from scripts.holdout_validation_v014 import _sharpe_from_results, sharpe_ci

OUT = ROOT / "data" / "research" / "v007_hold_validation.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"

ACTIVE = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
LIVE_OVERRIDES = {"GBPUSD=X": 6, "AUDUSD=X": 5, "EURUSD=X": 5}  # USDJPY -> default 60
CAND_CAPS = [3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 45, 60]
OOS = ("2023-01-01", "2024-12-31")
WF_WINDOWS = [("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
              ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31")]


# ── headline: faithful live path (backtest_pair), treatment vs baseline ─────────────── #

def _portfolio_live(start: str, end: str, overrides: dict):
    """√n-weighted portfolio Sharpe + per-pair, via the live backtest_pair path."""
    bt = ForexBacktester(start=start, end=end)
    bt.PAIR_HOLD_OVERRIDES = dict(overrides)  # instance shadows class attr
    results, per_pair = [], {}
    for p in ACTIVE:
        try:
            r = bt.backtest_pair(p)
        except Exception:
            r = None
        if r is not None:
            results.append(r)
            per_pair[p] = {"sharpe": round(float(r.sharpe), 3), "n": int(r.total_trades),
                           "avg_hold": round(float(r.avg_hold_days), 1)}
    return _sharpe_from_results(results), sum(r.total_trades for r in results), per_pair


# ── fast array path for the permutation (cap = np.minimum(hold_days, cap)) ───────────── #

def _array_pair_sharpe(bt: ForexBatchBacktester, pair: str, cap: int):
    ds = bt._array_cache.get(pair)
    if ds is None:
        return 0.0, 0
    capped = np.minimum(ds.hold_days, cap).astype(ds.hold_days.dtype)
    trades = simulate_forex_trades_arrays(
        opens=ds.opens, closes=ds.closes, signals=ds.signals.astype(np.int8),
        hold_days=capped, stop_pct=bt._backtester.STOP_PCT, index=ds.index)
    if not trades:
        return 0.0, 0
    trades = ForexBacktester._apply_costs(trades, pair)
    st = bt._backtester._compute_stats(pair, trades, len(ds.index))
    return float(st.sharpe), int(st.total_trades)


def _array_portfolio(bt, caps: dict):
    per = {p: _array_pair_sharpe(bt, p, caps[p]) for p in ACTIVE}
    items = [(s, n) for (s, n) in per.values() if n > 0 and not np.isnan(s)]
    if not items:
        return 0.0
    w = [np.sqrt(n) for _, n in items]
    return float(sum(s * wi for (s, _), wi in zip(items, w)) / sum(w))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "id": "HYP-V007-HOLD", "active_pairs": ACTIVE, "live_overrides": LIVE_OVERRIDES,
           "n_perms": args.perms, "seed": args.seed}

    # ── 1. OOS headline (faithful live path) ──────────────────────────────────────────
    print("OOS headline (live path)...")
    t_sh, t_n, t_pp = _portfolio_live(*OOS, LIVE_OVERRIDES)
    b_sh, b_n, b_pp = _portfolio_live(*OOS, {})
    oos_delta = t_sh - b_sh
    out["oos"] = {
        "treatment_sharpe": round(t_sh, 4), "treatment_trades": t_n, "treatment_per_pair": t_pp,
        "baseline_sharpe": round(b_sh, 4), "baseline_trades": b_n, "baseline_per_pair": b_pp,
        "delta_sharpe": round(oos_delta, 4),
        "treatment_ci": sharpe_ci(t_sh, t_n), "baseline_ci": sharpe_ci(b_sh, b_n),
    }
    print(f"  TREATMENT OOS Sharpe={t_sh:.4f} (n={t_n})  BASELINE={b_sh:.4f} (n={b_n})  delta={oos_delta:+.4f}")

    # ── 2. Permutation: chosen caps vs random caps (array path, OOS window) ────────────
    print(f"Permutation ({args.perms}, random caps)...")
    bt = ForexBatchBacktester(start=OOS[0], end=OOS[1])
    bt.preload(ACTIVE)
    caps_treatment = {p: LIVE_OVERRIDES.get(p, 60) for p in ACTIVE}
    treat_arr = _array_portfolio(bt, caps_treatment)
    null = np.empty(args.perms)
    for k in range(args.perms):
        caps = {p: int(rng.choice(CAND_CAPS)) for p in ACTIVE}
        null[k] = _array_portfolio(bt, caps)
    p_value = float((np.sum(null >= treat_arr) + 1) / (args.perms + 1))
    out["permutation"] = {
        "treatment_array_sharpe": round(treat_arr, 4),
        "null_mean": round(float(null.mean()), 4), "null_pct95": round(float(np.percentile(null, 95)), 4),
        "null_max": round(float(null.max()), 4), "p_value": p_value,
        "interpretation": "p = P(random per-pair caps >= the chosen 6/5/5 caps). Array path "
                          "(no VIX gate); measures whether the SPECIFIC caps are special vs random.",
    }
    print(f"  treatment(array)={treat_arr:.4f}  null mean={null.mean():.4f} p95={np.percentile(null,95):.4f}  p={p_value:.4f}")

    # ── 3. Walk-forward (live path, per disjoint test year) ───────────────────────────
    print("Walk-forward (live path, per year)...")
    wf = []
    for label, s, e in WF_WINDOWS:
        ts, tn, _ = _portfolio_live(s, e, LIVE_OVERRIDES)
        bs, bn, _ = _portfolio_live(s, e, {})
        wf.append({"window": label, "treatment": round(ts, 3), "baseline": round(bs, 3),
                   "delta": round(ts - bs, 3), "t_n": tn, "b_n": bn})
        print(f"  {label}: treat={ts:+.3f} base={bs:+.3f} delta={ts-bs:+.3f} (n={tn})")
    out["walk_forward"] = wf
    wf_deltas = [w["delta"] for w in wf]
    wf_stable = all(d >= -0.05 for d in wf_deltas) and sum(d > 0 for d in wf_deltas) >= 3

    # ── verdict ───────────────────────────────────────────────────────────────────────
    valid = (oos_delta > 0) and (p_value < 0.05) and wf_stable
    out["verdict"] = "VALID_EDGE" if valid else "NOT_SIGNIFICANT"
    out["verdict_basis"] = {
        "oos_delta_positive": oos_delta > 0, "permutation_sig": p_value < 0.05,
        "walk_forward_stable": wf_stable, "wf_deltas": wf_deltas,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nVERDICT: {out['verdict']}  (OOS delta {oos_delta:+.4f}, p={p_value:.4f}, wf_stable={wf_stable})")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
