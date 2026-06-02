"""
Canonical Hypothesis Runner — costed IS/OOS framework.

Every hypothesis MUST go through this script before touching live code.
Uses ForexBacktester (spread + slippage + swap, √(n/years) annualization)
with a mandatory IS/OOS split. Reports OOS p-value; gates on OOS, not IS.

Usage:
  PYTHONPATH=. python3 scripts/run_hypothesis.py \\
      --id HYP-045 \\
      --name "AUDNZD exclusion test" \\
      --config '{"EXCLUDE_PAIRS": ["AUDNZD=X"]}' \\
      [--perms 500]    # 500 = fast dev run; 1000 = final verdict

Config keys supported:
  VIX_GATES       dict[pair, float]   e.g. {"USDJPY=X": 13}
  HOLD_OVERRIDES  dict[pair, int]     e.g. {"GBPUSD=X": 8}
  TRAILING_OVERRIDES dict[pair, float]
  EXCLUDE_PAIRS   list[str]           pairs to drop from portfolio entirely

Decision gate (all must pass to CONFIRM):
  1. OOS Sharpe delta > 0
  2. OOS permutation p < 0.05
  3. Benjamini-Hochberg FDR=5% survives across all stored p-values
  4. Decay ratio (OOS / IS) >= 0.50

Writes:
  data/agent/canonical/{id}.json   — full result
  data/agent/hypothesis_ledger.json — updates entry status/p_value
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from sovereign.forex.batch_backtester import ForexBatchBacktester
from sovereign.forex.fast_backtester import simulate_forex_trades_arrays
from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.pair_universe import ALL_PAIRS

IS_START  = "2015-01-01"
IS_END    = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END   = "2024-12-31"

LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
OUT_DIR     = ROOT / "data" / "agent" / "canonical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── shared helpers (mirror holdout_validation_v014.py) ──────────────────────

def _weighted_sharpe(results) -> float:
    """√n-weighted mean of per-pair Sharpes."""
    items = [(r.sharpe, r.total_trades) for r in results
             if r is not None and not np.isnan(r.sharpe) and r.total_trades > 0]
    if not items:
        return 0.0
    w = [np.sqrt(n) for _, n in items]
    return float(sum(s * wi for (s, _), wi in zip(items, w)) / sum(w))


def _total_trades(results) -> int:
    return int(sum(r.total_trades for r in results if r is not None))


def sharpe_ci(sharpe: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI: Lo 2002 / Jorion SE = √((1 + ½·SR²) / n)."""
    se = float(np.sqrt((1 + 0.5 * sharpe ** 2) / max(n, 1)))
    return round(sharpe - z * se, 3), round(sharpe + z * se, 3)


# ─── apply config overrides to a ForexBacktester instance ────────────────────

def _patch_backtester(bt: ForexBacktester, cfg: dict) -> None:
    """Mutate bt class-level dicts to reflect hypothesis config."""
    if "VIX_GATES" in cfg:
        bt.PAIR_VIX_GATES = {**bt.PAIR_VIX_GATES, **cfg["VIX_GATES"]}
    if "HOLD_OVERRIDES" in cfg:
        bt.PAIR_HOLD_OVERRIDES = {**bt.PAIR_HOLD_OVERRIDES, **cfg["HOLD_OVERRIDES"]}
    if "TRAILING_OVERRIDES" in cfg:
        bt.PAIR_TRAILING_OVERRIDES = {**bt.PAIR_TRAILING_OVERRIDES, **cfg["TRAILING_OVERRIDES"]}


def _run_window(start: str, end: str, cfg: dict) -> tuple[list, float, int]:
    """Run costed backtest for a date window with config overrides applied."""
    bt = ForexBacktester(start=start, end=end)
    _patch_backtester(bt, cfg)
    exclude = set(cfg.get("EXCLUDE_PAIRS", []))
    results = [r for r in bt.backtest_all() if r is not None and r.pair not in exclude]
    sharpe = _weighted_sharpe(results)
    n = _total_trades(results)
    return results, sharpe, n


# ─── permutation test (OOS window only) ──────────────────────────────────────

def _perm_run_signals(bt_instance: ForexBatchBacktester, pair: str, ds, signals: np.ndarray):
    trades = simulate_forex_trades_arrays(
        opens=ds.opens, closes=ds.closes,
        signals=signals.astype(np.int8),
        hold_days=ds.hold_days,
        stop_pct=bt_instance._backtester.STOP_PCT,
        index=ds.index,
    )
    if not trades:
        return 0.0, 0
    trades = ForexBacktester._apply_costs(trades, pair)
    stats = bt_instance._backtester._compute_stats(pair, trades, len(ds.index))
    return float(stats.sharpe), int(stats.total_trades)


def _perm_weighted_sharpe(per_pair: dict) -> float:
    items = [(s, n) for (s, n) in per_pair.values() if n > 0 and not np.isnan(s)]
    if not items:
        return 0.0
    w = [np.sqrt(n) for _, n in items]
    return float(sum(s * wi for (s, _), wi in zip(items, w)) / sum(w))


def run_permutation_test(cfg: dict, n_perms: int = 500, seed: int = 7) -> dict:
    """
    Permutation test on OOS window only.
    Null: random entries at same frequency, same costs, same hold_days.
    Returns dict with real_sharpe, p_value, verdict.
    """
    rng = np.random.default_rng(seed)
    exclude = set(cfg.get("EXCLUDE_PAIRS", []))

    bt = ForexBatchBacktester(start=OOS_START, end=OOS_END)
    # Apply VIX gate overrides to the batch backtester's underlying instance
    if "VIX_GATES" in cfg:
        bt._backtester.PAIR_VIX_GATES = {**bt._backtester.PAIR_VIX_GATES, **cfg["VIX_GATES"]}
    if "HOLD_OVERRIDES" in cfg:
        bt._backtester.PAIR_HOLD_OVERRIDES = {**bt._backtester.PAIR_HOLD_OVERRIDES, **cfg["HOLD_OVERRIDES"]}

    pairs_to_load = [p for p in ALL_PAIRS if p not in exclude]
    bt.preload(pairs_to_load)
    pairs = [p for p in sorted(bt._array_cache.keys()) if p not in exclude]

    if not pairs:
        return {"real_sharpe": 0.0, "p_value": 1.0, "n_perms": 0, "verdict": "NO_DATA"}

    # Real signals
    real_per_pair = {}
    sig_counts = {}
    for pair in pairs:
        ds = bt._array_cache[pair]
        sig = np.asarray(ds.signals)
        sig_counts[pair] = int(np.count_nonzero(sig))
        real_per_pair[pair] = _perm_run_signals(bt, pair, ds, sig)

    real_portfolio = _perm_weighted_sharpe(real_per_pair)

    # Null distribution
    null_portfolio = []
    bar_counts = {p: len(np.asarray(bt._array_cache[p].signals)) for p in pairs}
    for k in range(n_perms):
        perm = {}
        for pair in pairs:
            ds = bt._array_cache[pair]
            n_bars = bar_counts[pair]
            n_sig  = sig_counts[pair]
            rand_sig = np.zeros(n_bars, dtype=np.int8)
            if n_sig > 0:
                idx = rng.choice(n_bars, size=min(n_sig, n_bars), replace=False)
                rand_sig[idx] = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx))
            perm[pair] = _perm_run_signals(bt, pair, ds, rand_sig)
        null_portfolio.append(_perm_weighted_sharpe(perm))
        if (k + 1) % 100 == 0:
            print(f"    perm {k+1}/{n_perms} ...")

    null_arr = np.asarray(null_portfolio)
    p_value  = float(np.mean(null_arr >= real_portfolio))
    verdict  = ("REAL" if p_value < 0.05 else
                "SUGGESTIVE" if p_value < 0.10 else "NOT_PROVEN")

    return {
        "real_sharpe":   round(real_portfolio, 4),
        "null_mean":     round(float(null_arr.mean()), 4),
        "null_std":      round(float(null_arr.std()), 4),
        "null_p95":      round(float(np.percentile(null_arr, 95)), 4),
        "p_value":       round(p_value, 4),
        "n_perms":       n_perms,
        "verdict":       verdict,
    }


# ─── Benjamini-Hochberg across ledger p-values ───────────────────────────────

def _bh_check(new_p: float, fdr: float = 0.05) -> bool:
    """
    Apply BH correction including all stored p-values plus this new one.
    Returns True if the new p-value survives FDR=5%.
    """
    ledger = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists() else {"hypotheses": []}
    stored = [
        h.get("p_value") for h in ledger.get("hypotheses", [])
        if isinstance(h.get("p_value"), (int, float))
    ]
    all_p = sorted(stored + [new_p])
    m = len(all_p)
    for rank, p in enumerate(all_p, start=1):
        if p == new_p:
            threshold = fdr * rank / m
            return new_p <= threshold
    return False


# ─── ledger update ────────────────────────────────────────────────────────────

def _update_ledger(hyp_id: str, name: str, result: dict) -> None:
    if not LEDGER_PATH.exists():
        ledger = {"hypotheses": []}
    else:
        ledger = json.loads(LEDGER_PATH.read_text())

    entry = next((h for h in ledger["hypotheses"] if h.get("id") == hyp_id), None)
    if entry is None:
        entry = {"id": hyp_id, "name": name}
        ledger["hypotheses"].append(entry)

    entry.update({
        "name":            name,
        "status":          result["verdict"],
        "date_tested":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "methodology":     "canonical_costed_is_oos",
        "is_sharpe":       result["is_sharpe"],
        "oos_sharpe":      result["oos_sharpe"],
        "oos_delta":       result["oos_delta"],
        "decay_ratio":     result["decay_ratio"],
        "p_value":         result["perm"]["p_value"],
        "bh_survives":     result["bh_survives"],
        "oos_ci":          result["oos_ci"],
        "config":          result["config"],
    })
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=float))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical costed IS/OOS hypothesis runner")
    ap.add_argument("--id",     required=True,  help="Hypothesis ID, e.g. HYP-045")
    ap.add_argument("--name",   default="",     help="Short description")
    ap.add_argument("--config", default="{}",   help="JSON config override dict")
    ap.add_argument("--perms",  type=int, default=500,
                    help="Permutation count: 500=fast, 1000=final")
    args = ap.parse_args()

    cfg = json.loads(args.config)

    print(f"\n{'═'*64}")
    print(f"  CANONICAL HYPOTHESIS TEST: {args.id}")
    print(f"  {args.name}")
    print(f"  Config: {json.dumps(cfg)}")
    print(f"  IS: {IS_START}→{IS_END}  |  OOS: {OOS_START}→{OOS_END}")
    print(f"  Permutations (OOS only): {args.perms}")
    print(f"{'═'*64}\n")

    # ── Baseline (no overrides, same windows) ─────────────────────────────
    print("  Running BASELINE IS...")
    _, base_is_sharpe, _ = _run_window(IS_START, IS_END, {})
    print(f"  Baseline IS Sharpe: {base_is_sharpe:.4f}")

    print("  Running BASELINE OOS...")
    _, base_oos_sharpe, _ = _run_window(OOS_START, OOS_END, {})
    print(f"  Baseline OOS Sharpe: {base_oos_sharpe:.4f}")

    # ── Hypothesis ────────────────────────────────────────────────────────
    print("\n  Running HYPOTHESIS IS...")
    is_results, is_sharpe, is_n = _run_window(IS_START, IS_END, cfg)
    print(f"  Hypothesis IS Sharpe:  {is_sharpe:.4f}  (n={is_n})")

    print("  Running HYPOTHESIS OOS...")
    oos_results, oos_sharpe, oos_n = _run_window(OOS_START, OOS_END, cfg)
    print(f"  Hypothesis OOS Sharpe: {oos_sharpe:.4f}  (n={oos_n})")

    oos_delta   = round(oos_sharpe - base_oos_sharpe, 4)
    is_delta    = round(is_sharpe  - base_is_sharpe,  4)
    decay_ratio = round(oos_sharpe / is_sharpe, 3) if is_sharpe > 0 else 0.0
    oos_ci      = sharpe_ci(oos_sharpe, oos_n)
    is_ci       = sharpe_ci(is_sharpe,  is_n)

    print(f"\n  IS  delta vs baseline: {is_delta:+.4f}  CI {is_ci}")
    print(f"  OOS delta vs baseline: {oos_delta:+.4f}  CI {oos_ci}")
    print(f"  Decay ratio (OOS/IS):  {decay_ratio:.3f}  (≥0.50 required)")

    # ── Permutation test on OOS only ──────────────────────────────────────
    print(f"\n  Running permutation test on OOS window ({args.perms} perms)...")
    perm = run_permutation_test(cfg, n_perms=args.perms)
    print(f"  Real OOS Sharpe (perm engine): {perm['real_sharpe']:.4f}")
    print(f"  Null mean: {perm['null_mean']:.4f}  std: {perm['null_std']:.4f}")
    print(f"  p-value: {perm['p_value']:.4f}  verdict: {perm['verdict']}")

    # ── Benjamini-Hochberg ────────────────────────────────────────────────
    bh_ok = _bh_check(perm["p_value"])
    print(f"  BH FDR=5% survives: {'YES' if bh_ok else 'NO'}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    gates = {
        "oos_delta_positive": oos_delta > 0,
        "oos_p_lt_005":       perm["p_value"] < 0.05,
        "bh_survives":        bh_ok,
        "decay_ok":           decay_ratio >= 0.50,
    }
    confirmed = all(gates.values())
    verdict_str = "CONFIRMED" if confirmed else (
        "SUGGESTIVE" if perm["p_value"] < 0.10 and oos_delta > 0 else "REJECTED"
    )

    print(f"  Gate checks:")
    for g, v in gates.items():
        print(f"    {'✓' if v else '✗'} {g}")
    print(f"\n  VERDICT: {verdict_str}")
    if confirmed:
        print(f"  → OOS costed Sharpe: {oos_sharpe:.4f} (+{oos_delta:.4f} vs baseline)")
        print(f"  → Qualifies for version bump if OOS Sharpe > v015 gate (0.85)")
    print(f"{'═'*64}\n")

    result = {
        "id":             args.id,
        "name":           args.name,
        "run_at":         datetime.now(timezone.utc).isoformat(),
        "config":         cfg,
        "baseline":       {"is_sharpe": round(base_is_sharpe, 4), "oos_sharpe": round(base_oos_sharpe, 4)},
        "is_sharpe":      round(is_sharpe, 4),
        "is_n":           is_n,
        "is_ci":          is_ci,
        "is_delta":       is_delta,
        "oos_sharpe":     round(oos_sharpe, 4),
        "oos_n":          oos_n,
        "oos_ci":         oos_ci,
        "oos_delta":      oos_delta,
        "decay_ratio":    decay_ratio,
        "perm":           perm,
        "bh_survives":    bh_ok,
        "gates":          gates,
        "verdict":        verdict_str,
    }

    out_path = OUT_DIR / f"{args.id.replace('-', '_').lower()}.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"  Saved to {out_path}")

    _update_ledger(args.id, args.name, result)
    print(f"  Ledger updated.")


if __name__ == "__main__":
    main()
