"""
RQ-REST-004 — Rate Differential Gate Suite (HYP-052b, HYP-052c, HYP-054)
==========================================================================

Updated in REST-004 to:
  - Fix PairConfig.get() bug (use dataclass attributes)
  - Actually run backtests via run_with_signals() for real Sharpe comparison
  - Test all gate variants A/B/D with proper vectorized gates

GATE VARIANTS:
  A. Rate vol gate:   block when 30d rate diff vol < threshold
  B. Rate level gate: block when |rate differential| < threshold  [HYP-054]
  D. Pair trend gate: block when 30d rate diff trend <= 0         [HYP-052c]
  E. Combined B+D:    require both level AND trend > 0

Usage:
    python3 scripts/rq_rest_004_rate_vol_gate.py [--start 2015-01-01] [--end 2024-12-31]

Output: data/research/hyp_052b_rate_vol_gate.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests", "charset_normalizer"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from sovereign.forex.batch_backtester import ForexBatchBacktester
from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.data_fetcher import FALLBACK_RATES

CACHE_DIR = ROOT / "data" / "cache" / "macro"
OUT_PATH = ROOT / "data" / "research" / "hyp_052b_rate_vol_gate.json"


# ─── Rate data helpers ───────────────────────────────────────────────────────

def _load_rates(country: str) -> pd.Series:
    for name in [f"{country}_rates.parquet", f"{country.upper()}_rates.parquet"]:
        p = CACHE_DIR / name
        if p.exists():
            df = pd.read_parquet(p)
            s = df["rate"] if "rate" in df.columns else df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)


def _load_cpi(country: str) -> pd.Series:
    for name in [f"{country}_cpi.parquet", f"{country.upper()}_cpi.parquet"]:
        p = CACHE_DIR / name
        if p.exists():
            df = pd.read_parquet(p)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)


def _pair_countries(pair: str) -> tuple[str, str]:
    """Return (base_country, quote_country) from PairConfig (not .get())."""
    cfg = PAIR_CONFIG.get(pair)
    if cfg is None:
        return "US", "US"
    base_cb  = cfg.base_central_bank
    quote_cb = cfg.quote_central_bank
    return CB_TO_COUNTRY.get(base_cb, "US"), CB_TO_COUNTRY.get(quote_cb, "US")


def build_rate_diff_series(pair: str, start: str, end: str) -> pd.Series:
    """Daily real rate differential: (base_real - quote_real)."""
    base, quote = _pair_countries(pair)
    idx = pd.date_range(start, end, freq="D")

    br = _load_rates(base).reindex(idx, method="ffill").fillna(FALLBACK_RATES.get(base, 2.0))
    qr = _load_rates(quote).reindex(idx, method="ffill").fillna(FALLBACK_RATES.get(quote, 2.0))

    bc_raw = _load_cpi(base)
    qc_raw = _load_cpi(quote)
    bc = bc_raw.reindex(idx, method="ffill").fillna(2.0) if len(bc_raw) else pd.Series(2.0, index=idx)
    qc = qc_raw.reindex(idx, method="ffill").fillna(2.0) if len(qc_raw) else pd.Series(2.0, index=idx)

    return (br - bc) - (qr - qc)


# ─── Gate functions (vectorized) ─────────────────────────────────────────────

def gate_rate_vol(signals: np.ndarray, dates: pd.DatetimeIndex,
                  rate_diff: pd.Series, vol_window: int = 30,
                  threshold: float = 0.15) -> np.ndarray:
    """Variant A: block if 30d rate diff vol < threshold."""
    rate_vol = rate_diff.rolling(vol_window).std().reindex(dates).fillna(0).to_numpy()
    mask = rate_vol >= threshold
    return np.where(mask, signals, 0).astype(signals.dtype)


def gate_rate_level(signals: np.ndarray, dates: pd.DatetimeIndex,
                    rate_diff: pd.Series, threshold: float = 1.0) -> np.ndarray:
    """Variant B: block if |real_rate_diff| < threshold. [HYP-054]"""
    levels = rate_diff.reindex(dates).fillna(0).to_numpy()
    mask = np.abs(levels) >= threshold
    return np.where(mask, signals, 0).astype(signals.dtype)


def gate_pair_trend(signals: np.ndarray, dates: pd.DatetimeIndex,
                    rate_diff: pd.Series, trend_window: int = 30) -> np.ndarray:
    """Variant D: block when 30d rate diff trend <= 0. [HYP-052c]"""
    trend = rate_diff.diff(trend_window).reindex(dates).fillna(0).to_numpy()
    mask = trend > 0
    return np.where(mask, signals, 0).astype(signals.dtype)


def gate_level_and_trend(signals: np.ndarray, dates: pd.DatetimeIndex,
                          rate_diff: pd.Series, level_threshold: float = 1.0,
                          trend_window: int = 30) -> np.ndarray:
    """Variant E: require BOTH level > threshold AND trend > 0."""
    levels = rate_diff.reindex(dates).fillna(0).to_numpy()
    trend  = rate_diff.diff(trend_window).reindex(dates).fillna(0).to_numpy()
    mask   = (np.abs(levels) >= level_threshold) & (trend > 0)
    return np.where(mask, signals, 0).astype(signals.dtype)


# ─── Sharpe helpers ──────────────────────────────────────────────────────────

def weighted_portfolio_sharpe(per_pair: dict) -> float:
    valid = [(s, n) for s, n in per_pair.values()
             if n > 0 and not np.isnan(s) and not np.isinf(s)]
    if not valid:
        return float("nan")
    w = [np.sqrt(n) for _, n in valid]
    return float(sum(s * wi for (s, _), wi in zip(valid, w)) / sum(w))


def run_gated(bt: ForexBatchBacktester, gate_fn, rate_diffs: dict) -> dict:
    """Apply gate and run run_with_signals() for each pair. Returns per-pair results."""
    results = {}
    for pair in ALL_PAIRS:
        ds = bt._array_cache.get(pair)
        if ds is None:
            print(f"    {pair}: no cached dataset, skip")
            results[pair] = (float("nan"), 0, 0)
            continue

        gated_sig = gate_fn(ds.signals, ds.index, rate_diffs[pair])
        n_orig  = int((ds.signals != 0).sum())
        n_gated = int((gated_sig != 0).sum())

        result = bt._backtester.run_with_signals(
            pair=pair,
            opens=ds.opens,
            closes=ds.closes,
            signals=gated_sig,
            hold_days=ds.hold_days,
            index=ds.index,
        )
        sharpe = result.sharpe if result else float("nan")
        n_trades = result.total_trades if result else 0
        pct_pass = round(n_gated / n_orig * 100, 1) if n_orig else 0.0
        print(f"    {pair}: Sharpe={sharpe:.3f}  trades={n_trades}  "
              f"signals: {n_orig}→{n_gated} ({pct_pass:.0f}% pass)")
        results[pair] = (sharpe, n_trades, pct_pass)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--vol-threshold",   type=float, default=0.15)
    parser.add_argument("--level-threshold", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Rate Differential Gate Suite (HYP-052b / HYP-052c / HYP-054)")
    print(f"Period: {args.start} → {args.end}  |  Pairs: {ALL_PAIRS}")
    print()

    # Build rate diff series from cached parquets
    print("Loading rate differential series...")
    rate_diffs = {}
    for pair in ALL_PAIRS:
        rd = build_rate_diff_series(pair, args.start, args.end)
        rate_diffs[pair] = rd
        print(f"  {pair}: {len(rd)} days  avg={rd.mean():+.2f}%  "
              f"30d-vol-mean={rd.rolling(30).std().mean():.3f}%")

    # Preload the batch backtester (no yfinance — uses cached parquets via ForexDataFetcher)
    print("\nPreloading batch backtester...")
    bt = ForexBatchBacktester(start=args.start, end=args.end)
    bt.preload()

    # Base run (no gate)
    print("\n--- BASELINE (no gate) ---")
    base_results = {}
    for pair in ALL_PAIRS:
        ds = bt._array_cache.get(pair)
        if ds is None:
            base_results[pair] = (float("nan"), 0)
            continue
        result = bt._backtester.run_with_signals(
            pair=pair,
            opens=ds.opens,
            closes=ds.closes,
            signals=ds.signals,
            hold_days=ds.hold_days,
            index=ds.index,
        )
        s = result.sharpe if result else float("nan")
        n = result.total_trades if result else 0
        base_results[pair] = (s, n)
        print(f"  {pair}: Sharpe={s:.3f}  trades={n}")
    base_port = weighted_portfolio_sharpe(base_results)
    print(f"  Portfolio Sharpe (baseline): {base_port:.4f}")

    # Gate variants
    gate_variants = {
        "A_vol_gate": lambda sig, dates, rd: gate_rate_vol(
            sig, dates, rd, threshold=args.vol_threshold),
        "B_level_gate_1pct": lambda sig, dates, rd: gate_rate_level(
            sig, dates, rd, threshold=1.0),
        "B_level_gate_15pct": lambda sig, dates, rd: gate_rate_level(
            sig, dates, rd, threshold=1.5),
        "D_trend_gate": lambda sig, dates, rd: gate_pair_trend(sig, dates, rd),
        "E_level_and_trend": lambda sig, dates, rd: gate_level_and_trend(
            sig, dates, rd, level_threshold=1.0),
    }

    all_gate_results = {}
    for gate_name, gate_fn in gate_variants.items():
        print(f"\n--- GATE: {gate_name} ---")
        gated = run_gated(bt, gate_fn, rate_diffs)
        port_sharpe = weighted_portfolio_sharpe(
            {p: (s, n) for p, (s, n, _) in gated.items()}
        )
        delta = port_sharpe - base_port
        print(f"  Portfolio Sharpe: {port_sharpe:.4f}  (Δ {delta:+.4f} vs baseline)")
        all_gate_results[gate_name] = {
            "portfolio_sharpe": round(port_sharpe, 4),
            "delta_vs_baseline": round(delta, 4),
            "per_pair": {
                p: {"sharpe": round(s, 4), "n_trades": n, "pct_signals_passed": pct}
                for p, (s, n, pct) in gated.items()
            },
        }

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"{'Gate':<24} {'Portfolio Sharpe':>16} {'Delta':>8}")
    print("-" * 65)
    print(f"{'Baseline':24} {base_port:16.4f} {'—':>8}")
    for gname, gres in all_gate_results.items():
        marker = " ←" if gres["delta_vs_baseline"] > 0.15 else ""
        print(f"{gname:<24} {gres['portfolio_sharpe']:16.4f} "
              f"{gres['delta_vs_baseline']:>+8.4f}{marker}")
    print("=" * 65)

    results = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "run_by": "REST-004",
        "hypotheses": ["HYP-052b", "HYP-052c", "HYP-054"],
        "start": args.start,
        "end": args.end,
        "baseline": {
            "portfolio_sharpe": round(base_port, 4),
            "per_pair": {p: {"sharpe": round(s, 4), "n": n}
                         for p, (s, n) in base_results.items()},
        },
        "gate_results": all_gate_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
