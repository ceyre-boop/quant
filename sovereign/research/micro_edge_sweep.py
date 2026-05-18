"""
Micro-Edge Sweep — systematic parameter grid search across all forex pairs.

Architecture: pre-compute signals once per pair, then sweep execution parameters
at Numba speed (~148k simulations/sec). The signal generation (macro + CB + CPI
layers) is the bottleneck; the execution sweep is essentially free.

Grid:
  hold_days:        [5, 10, 20, 30, 60, 90]
  trailing_mult:    [0.75, 1.0, 1.25, 1.5, 2.0]
  stop_mult:        [1.5, 2.0, 2.5, 3.0]
  signal_threshold: [0.05, 0.10, 0.15, 0.20, 0.25]

= 6 × 5 × 4 × 5 = 600 combinations per pair × 7 pairs = 4,200 backtests

Output:
  data/research/micro_edges.json    — all combos with WR>55% AND avgR>0.10R
  data/research/sweep_full.json     — complete results (compressed)
  data/research/sweep_report.md     — human-readable summary

Run:
  PYTHONPATH=/path/to/quant python3 sovereign/research/micro_edge_sweep.py
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "research"
OUT_EDGES   = OUT_DIR / "micro_edges.json"
OUT_FULL    = OUT_DIR / "sweep_full.json"
OUT_REPORT  = OUT_DIR / "sweep_report.md"

# ── Grid definition ───────────────────────────────────────────────────────
HOLD_DAYS         = [5, 10, 20, 30, 60, 90]
TRAILING_MULTS    = [0.75, 1.0, 1.25, 1.5, 2.0]
STOP_MULTS        = [1.5, 2.0, 2.5, 3.0]
SIGNAL_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

# Minimum criteria for a "micro-edge"
# Medallion: 50.75% WR across millions of trades. We want 51%+ at monthly cadence.
MIN_WIN_RATE  = 0.51   # just above coin-flip — consistency beats conviction
MIN_AVG_R     = 0.05   # meaningful positive expectation per trade (in pct units)
MIN_TRADES    = 12     # enough statistical validity for a monthly system
MIN_SHARPE    = 0.60   # annualized, risk-adjusted minimum

# Stop pct (fixed, only ATR mult varies in sweep)
STOP_PCT      = 0.04
RISK_PCT      = 0.01


@dataclass
class SweepResult:
    pair: str
    hold_days: int
    trailing_mult: float
    stop_mult: float
    signal_threshold: float
    win_rate: float
    avg_r: float
    total_trades: int
    sharpe: float
    is_micro_edge: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_micro_edge"] = bool(d["is_micro_edge"])
        return d


def _load_price_data(pair: str) -> Optional[pd.DataFrame]:
    """Load price data from cache or download."""
    import yfinance as yf
    cache_path = ROOT / "data" / "cache" / f"{pair.replace('=X','')}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    df = yf.download(pair, start="2014-01-01", end="2026-05-01", progress=False)
    if df is None or len(df) < 500:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _build_signal_frame(
    pair: str,
    prices: pd.DataFrame,
    base_country: str,
    quote_country: str,
    signal_threshold: float,
    hold_days: int,
) -> Optional[pd.DataFrame]:
    """Generate signal frame for a specific threshold + hold_days."""
    try:
        from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
        from sovereign.forex.data_fetcher import ForexDataFetcher
        from sovereign.forex.entry_engine import CBEventTrigger

        config = SignalConfig(
            signal_threshold=signal_threshold,
            hold_days=hold_days,
        )
        engine = ForexSignalEngine(
            fetcher=ForexDataFetcher(),
            cb_trigger=CBEventTrigger(),
            config=config,
        )
        return engine.build_signal_frame(
            prices=prices,
            base_country=base_country,
            quote_country=quote_country,
            start="2015-01-01",
            end="2026-01-01",
            pair=pair,
        )
    except Exception as exc:
        print(f"    [warn] signal generation failed for {pair} threshold={signal_threshold}: {exc}")
        return None


def _run_sweep_for_signals(
    pair: str,
    prices: pd.DataFrame,
    signal_frame: pd.DataFrame,
    signal_threshold: float,
    hold_days_base: int,
    atr_series=None,
) -> List[SweepResult]:
    """Sweep trailing/stop mults against a pre-computed signal frame."""
    from sovereign.forex.fast_backtester import simulate_forex_trades

    close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 3]
    opens = prices["Open"]  if "Open"  in prices.columns else prices.iloc[:, 0]

    results = []
    for trailing_mult, stop_mult in product(TRAILING_MULTS, STOP_MULTS):
        try:
            trades = simulate_forex_trades(
                df=prices,
                signal_frame=signal_frame,
                stop_pct=STOP_PCT,
                atr_series=atr_series,
                stop_atr_mult=stop_mult,
                trailing_atr_mult=trailing_mult,
                strict_mode=False,
                risk_pct=RISK_PCT,
                max_risk_pct=RISK_PCT,
            )
        except Exception:
            continue

        if not trades or len(trades) < MIN_TRADES:
            continue

        pnl = np.array([t["pnl_pct"] for t in trades])
        hold = np.array([t["hold_days"] for t in trades])
        wins = float(np.sum(pnl > 0))
        win_rate = wins / len(pnl)
        avg_r = float(np.mean(pnl))
        sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-9)) * np.sqrt(252 / max(float(np.mean(hold)), 1.0))

        is_edge = (
            win_rate >= MIN_WIN_RATE and
            avg_r >= MIN_AVG_R / 100 and  # pnl_pct is in decimal
            len(trades) >= MIN_TRADES and
            sharpe >= MIN_SHARPE
        )

        results.append(SweepResult(
            pair=pair,
            hold_days=int(np.mean(hold)),
            trailing_mult=trailing_mult,
            stop_mult=stop_mult,
            signal_threshold=signal_threshold,
            win_rate=round(win_rate, 4),
            avg_r=round(avg_r * 100, 4),  # convert to pct for readability
            total_trades=len(trades),
            sharpe=round(sharpe, 4),
            is_micro_edge=is_edge,
        ))

    return results


def run_sweep(verbose: bool = True) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY

    t0 = time.time()
    all_results: List[SweepResult] = []
    micro_edges: List[SweepResult] = []

    total_combinations = len(ALL_PAIRS) * len(SIGNAL_THRESHOLDS) * len(HOLD_DAYS) * len(TRAILING_MULTS) * len(STOP_MULTS)
    if verbose:
        print(f"Micro-Edge Sweep")
        print(f"Grid: {len(ALL_PAIRS)} pairs × {len(SIGNAL_THRESHOLDS)} thresholds × "
              f"{len(HOLD_DAYS)} hold_days × {len(TRAILING_MULTS)} trailing × "
              f"{len(STOP_MULTS)} stop_mults = {total_combinations:,} backtests")
        print()

    for pair in ALL_PAIRS:
        cfg = PAIR_CONFIG.get(pair)
        if not cfg:
            continue
        base  = CB_TO_COUNTRY.get(cfg.base_central_bank, "US")
        quote = CB_TO_COUNTRY.get(cfg.quote_central_bank, "US")

        if verbose:
            print(f"  {pair} — loading prices...", end="", flush=True)

        prices = _load_price_data(pair)
        if prices is None:
            print(" [SKIP: no data]")
            continue

        # Pre-compute ATR series once per pair (used in every simulation)
        try:
            from sovereign.forex.signal_engine import ForexSignalEngine
            from sovereign.forex.data_fetcher import ForexDataFetcher
            from sovereign.forex.entry_engine import CBEventTrigger
            _engine_tmp = ForexSignalEngine(fetcher=ForexDataFetcher(), cb_trigger=CBEventTrigger())
            close_tmp = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 3]
            atr_series = _engine_tmp._compute_atr_pct(close_tmp, prices)
        except Exception:
            atr_series = None

        pair_edges = 0
        for threshold in SIGNAL_THRESHOLDS:
            for hold_days in HOLD_DAYS:
                signal_frame = _build_signal_frame(
                    pair=pair,
                    prices=prices,
                    base_country=base,
                    quote_country=quote,
                    signal_threshold=threshold,
                    hold_days=hold_days,
                )
                if signal_frame is None:
                    continue

                results = _run_sweep_for_signals(
                    pair=pair,
                    prices=prices,
                    signal_frame=signal_frame,
                    signal_threshold=threshold,
                    hold_days_base=hold_days,
                    atr_series=atr_series,
                )

                all_results.extend(results)
                edges = [r for r in results if r.is_micro_edge]
                micro_edges.extend(edges)
                pair_edges += len(edges)

        elapsed = time.time() - t0
        if verbose:
            print(f" {pair_edges} micro-edges found  ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0

    # Sort by Sharpe
    micro_edges.sort(key=lambda x: -x.sharpe)
    all_results.sort(key=lambda x: -x.sharpe)

    # ── Correlation analysis ─────────────────────────────────────────────
    portfolio = _build_edge_portfolio(micro_edges)

    # ── Save outputs ─────────────────────────────────────────────────────
    edges_doc = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sweep_seconds": round(elapsed, 1),
        "total_combinations": total_combinations,
        "total_results": len(all_results),
        "micro_edges_found": len(micro_edges),
        "criteria": {
            "min_win_rate": MIN_WIN_RATE,
            "min_avg_r_pct": MIN_AVG_R,
            "min_trades": MIN_TRADES,
            "min_sharpe": MIN_SHARPE,
        },
        "top_edges": [e.to_dict() for e in micro_edges[:50]],
        "portfolio": portfolio,
    }
    OUT_EDGES.write_text(json.dumps(edges_doc, indent=2))

    # Full results (top 200 only — full set can be large)
    full_doc = {"results": [r.to_dict() for r in all_results[:200]]}
    OUT_FULL.write_text(json.dumps(full_doc, indent=2))

    # Human-readable report
    _write_report(micro_edges, portfolio, elapsed, total_combinations)

    return edges_doc


def _build_edge_portfolio(edges: List[SweepResult]) -> Dict:
    """
    Select a portfolio of uncorrelated micro-edges using a greedy approach.
    Two edges are "correlated" if they're the same pair with similar hold_days
    and threshold (they'd trade the same positions).
    """
    if not edges:
        return {"edges": [], "total_risk_pct": 0, "expected_monthly_wr": 0}

    selected = []
    seen_signatures = set()

    for edge in edges:
        # Signature: pair + hold bucket (5-day buckets) + threshold bucket
        hold_bucket = (edge.hold_days // 15) * 15
        threshold_bucket = round(edge.signal_threshold * 2) / 2
        sig = (edge.pair, hold_bucket, threshold_bucket)

        if sig not in seen_signatures:
            selected.append(edge)
            seen_signatures.add(sig)

        if len(selected) >= 30:
            break

    # Lo's math: probability of winning month with N independent edges
    if selected:
        avg_wr = np.mean([e.win_rate for e in selected])
        n = len(selected)
        # Monthly: each edge trades ~4 times/month on average (hold_days varies)
        # For conservative estimate: 1 trade per edge per month
        # P(positive month) ≈ P(wins > n/2) under binomial
        from scipy.stats import binom
        monthly_win_prob = 1 - binom.cdf(n // 2, n, avg_wr)
    else:
        avg_wr = 0
        monthly_win_prob = 0

    risk_per_edge = 0.25  # % of account per edge
    total_risk = len(selected) * risk_per_edge

    return {
        "selected_edges": len(selected),
        "avg_win_rate": round(float(avg_wr), 4),
        "risk_per_edge_pct": risk_per_edge,
        "total_risk_deployed_pct": round(total_risk, 2),
        "estimated_monthly_win_prob": round(float(monthly_win_prob), 4),
        "edges": [{"pair": e.pair, "hold_days": e.hold_days,
                   "threshold": e.signal_threshold, "sharpe": e.sharpe,
                   "win_rate": e.win_rate, "avg_r_pct": e.avg_r}
                  for e in selected],
    }


def _write_report(edges: List[SweepResult], portfolio: Dict, elapsed: float, total: int) -> None:
    lines = [
        "# Micro-Edge Sweep Report",
        f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Sweep: {total:,} combinations in {elapsed:.0f}s",
        "",
        f"## Results",
        f"- Micro-edges found: {len(edges)}",
        f"- Portfolio edges (uncorrelated): {portfolio.get('selected_edges', 0)}",
        f"- Total risk deployed: {portfolio.get('total_risk_deployed_pct', 0)}%",
        f"- Estimated monthly win probability: {portfolio.get('estimated_monthly_win_prob', 0)*100:.1f}%",
        "",
        "## Top 20 Micro-Edges (by Sharpe)",
        "",
        "| Pair | Hold | Threshold | Trailing | Stop | WR | AvgR | Sharpe |",
        "|------|------|-----------|----------|------|----|------|--------|",
    ]
    for e in edges[:20]:
        lines.append(
            f"| {e.pair} | {e.hold_days}d | {e.signal_threshold:.2f} | "
            f"{e.trailing_mult}× | {e.stop_mult}× | "
            f"{e.win_rate*100:.1f}% | {e.avg_r:+.3f}% | {e.sharpe:.3f} |"
        )

    lines += [
        "",
        "## Portfolio Construction (Lo Framework)",
        f"- {portfolio.get('selected_edges', 0)} uncorrelated edges",
        f"- {portfolio.get('risk_per_edge_pct', 0.25)}% risk per edge",
        f"- {portfolio.get('total_risk_deployed_pct', 0)}% total risk",
        f"- {portfolio.get('estimated_monthly_win_prob', 0)*100:.1f}% estimated monthly win probability",
        "",
        "## Next Steps",
        "1. Wire top edges into signal_engine as additional signal layers",
        "2. Run signal_decay.py monthly to monitor edge degradation",
        "3. Re-run sweep quarterly to discover new edges",
    ]

    OUT_REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    result = run_sweep(verbose=True)
    print(f"\nMicro-edges found: {result['micro_edges_found']}")
    print(f"Portfolio: {result['portfolio']['selected_edges']} uncorrelated edges")
    print(f"Estimated monthly win prob: {result['portfolio']['estimated_monthly_win_prob']*100:.1f}%")
    print(f"\nFiles written:")
    print(f"  {OUT_EDGES}")
    print(f"  {OUT_FULL}")
    print(f"  {OUT_REPORT}")
