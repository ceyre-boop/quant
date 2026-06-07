"""
RQ-REST-003 — Conviction Gate for Short-Hold Prevention
=========================================================

HYP-049 confirmed that 1-3d holds are systematic losers (p<0.0001).
This script asks: does entry conviction (abs macro_score) predict
which trades become short-hold losers BEFORE they are entered?

Method:
  1. Re-run the ForexBacktester across all 4 pairs (2015-2024) with a
     patched version that captures macro_score per signal.
  2. For each resulting trade, we know:
       - entry_date, pair, direction, hold_days, pnl_pct
       - macro_score at entry (the raw signal magnitude)
  3. Ask: does conviction < THRESHOLD predict short holds AND losses?
  4. Sweep CONVICTION_NEUTRAL_THRESHOLD from 0.20 to 0.70.
     At each threshold, compare:
       - Sharpe of trades above threshold ("allowed")
       - Sharpe of trades below threshold ("gated out")
       - Net portfolio Sharpe if we gate out low-conviction entries
  5. Output: conviction_gate_results.json

Usage:
    python3 scripts/rq_rest_003_conviction_gate.py [--threshold 0.50]

Key question: does raising threshold from 0.35 → 0.50 improve Sharpe
without cutting too many trades?
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
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.pair_universe import ALL_PAIRS, PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig
from sovereign.forex.data_fetcher import ForexDataFetcher
from sovereign.forex.entry_engine import CBEventTrigger

OUT_PATH = ROOT / "data" / "research" / "conviction_gate_results.json"
TRADE_LOG_PATH = ROOT / "data" / "research" / "conviction_gate_trades.json"


def _sharpe(returns: list[float]) -> float:
    """Annualized Sharpe from raw per-trade returns (not daily)."""
    if len(returns) < 5:
        return float("nan")
    a = np.array(returns)
    return float(np.mean(a) / np.std(a, ddof=1) * np.sqrt(252)) if np.std(a, ddof=1) > 0 else 0.0


def _portfolio_sharpe(per_pair_results: list[tuple[float, int]]) -> float:
    """sqrt(n)-weighted portfolio Sharpe."""
    valid = [(s, n) for s, n in per_pair_results if n > 0 and not np.isnan(s)]
    if not valid:
        return float("nan")
    weights = [np.sqrt(n) for _, n in valid]
    return float(sum(s * w for (s, _), w in zip(valid, weights)) / sum(weights))


class ConvictionLoggingEngine(ForexSignalEngine):
    """Wraps ForexSignalEngine to expose macro_score per signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_log: dict[str, float] = {}  # date_str → macro_score

    def _macro_signal_for_date(self, close, date, base_country, quote_country,
                                base_rates, quote_rates, base_cpi_h, quote_cpi_h) -> int:
        """Override to capture macro_score before thresholding."""
        import numpy as np
        from sovereign.forex.data_fetcher import FALLBACK_RATES, FALLBACK_CPI

        spot = float(close.asof(date))
        hist = close.loc[:date]

        b_rate = (float(base_rates.asof(date)) if len(base_rates) and date >= base_rates.index[0]
                  else FALLBACK_RATES.get(base_country, 2.0))
        q_rate = (float(quote_rates.asof(date)) if len(quote_rates) and date >= quote_rates.index[0]
                  else FALLBACK_RATES.get(quote_country, 2.0))
        b_cpi = (float(base_cpi_h.asof(date)) if len(base_cpi_h) and date >= base_cpi_h.index[0]
                 else FALLBACK_CPI.get(base_country, 2.0))
        q_cpi = (float(quote_cpi_h.asof(date)) if len(quote_cpi_h) and date >= quote_cpi_h.index[0]
                 else FALLBACK_CPI.get(quote_country, 2.0))

        real_rate_diff = (b_rate - b_cpi) - (q_rate - q_cpi)
        irp_fv = spot * (1 + q_rate / 100) / (1 + b_rate / 100)
        irp_dev = (spot - irp_fv) / irp_fv if irp_fv != 0 else 0.0
        irp_z = (irp_dev / (hist.pct_change().std() * np.sqrt(252) + 1e-8)
                 if len(hist) > 252 else 0.0)

        macro_score = (
            self.config.irp_weight * np.clip(-irp_z / 1.5, -1, 1) +
            self.config.rate_weight * np.clip(real_rate_diff / 4.0, -1, 1)
        )

        # Log the raw score for this date
        self._score_log[str(date.date())] = float(macro_score)

        macro_sign = int(np.sign(macro_score)) if abs(macro_score) > self.config.signal_threshold else 0
        if not self.config.use_momentum_filter:
            return macro_sign

        mom_sign = 0
        if len(hist) > 63:
            mom = float(hist.iloc[-1] / hist.iloc[-63] - 1)
            mom_sign = int(np.sign(mom)) if abs(mom) > 0.005 else 0

        if macro_sign != 0 and (mom_sign == 0 or mom_sign == macro_sign):
            return macro_sign
        return 0


def run_conviction_analysis(start: str = "2015-01-01", end: str = "2024-12-31") -> dict:
    """Full conviction sweep across all 4 pairs."""
    print(f"Conviction gate analysis: {start} → {end}")
    print(f"Pairs: {ALL_PAIRS}")

    fetcher = ForexDataFetcher()
    cb_trigger = CBEventTrigger()

    all_trades: list[dict] = []

    for pair in ALL_PAIRS:
        cfg = PAIR_CONFIG.get(pair)
        if cfg is None:
            continue
        base_country = CB_TO_COUNTRY.get(cfg.base_central_bank, "US")
        quote_country = CB_TO_COUNTRY.get(cfg.quote_central_bank, "US")

        print(f"  {pair} ({base_country} vs {quote_country}) ...", end=" ", flush=True)

        # Use the logging engine
        engine = ConvictionLoggingEngine(
            fetcher=fetcher,
            cb_trigger=cb_trigger,
            config=SignalConfig()
        )

        try:
            bt = ForexBacktester(start=start, end=end)
            result, trades = bt.run_pair_with_trades(
                pair=pair,
                base_country=base_country,
                quote_country=quote_country,
                signal_engine_override=engine,
            )
            print(f"{len(trades)} trades, Sharpe={result.sharpe:.3f}")
        except (AttributeError, TypeError):
            # ForexBacktester may not have run_pair_with_trades yet
            # Fall back to signal re-extraction from the score log
            result = bt.run_pair(pair, base_country, quote_country)
            trades = []
            print(f"(signal log only, no trade objects — upgrade backtester)")

        # Attach conviction scores to trades
        for t in trades:
            entry_date = str(t.get("entry_date", ""))[:10]
            conviction = abs(engine._score_log.get(entry_date, float("nan")))
            all_trades.append({
                "pair": pair,
                "entry_date": entry_date,
                "direction": t.get("direction", 0),
                "hold_days": t.get("hold_days", 0),
                "pnl_pct": t.get("pnl_pct", 0.0),
                "macro_score": engine._score_log.get(entry_date, float("nan")),
                "conviction": conviction,
            })

    # Sweep thresholds
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    sweep_results = []

    for thresh in thresholds:
        allowed = [t for t in all_trades if t["conviction"] >= thresh]
        blocked = [t for t in all_trades if t["conviction"] < thresh]

        allowed_returns = [t["pnl_pct"] for t in allowed if not np.isnan(t["pnl_pct"])]
        blocked_returns = [t["pnl_pct"] for t in blocked if not np.isnan(t["pnl_pct"])]

        # Short-hold analysis within allowed
        allowed_short = [t for t in allowed if t["hold_days"] <= 3]
        allowed_long = [t for t in allowed if t["hold_days"] > 3]

        sweep_results.append({
            "threshold": thresh,
            "n_allowed": len(allowed),
            "n_blocked": len(blocked),
            "allowed_sharpe": _sharpe(allowed_returns),
            "blocked_sharpe": _sharpe(blocked_returns),
            "allowed_wr": float(np.mean([r > 0 for r in allowed_returns])) if allowed_returns else 0,
            "blocked_wr": float(np.mean([r > 0 for r in blocked_returns])) if blocked_returns else 0,
            "short_hold_pct_in_allowed": len(allowed_short) / len(allowed) if allowed else 0,
        })

        print(f"  thresh={thresh:.2f}: allowed={len(allowed)} Sharpe={_sharpe(allowed_returns):.3f} | "
              f"blocked={len(blocked)} Sharpe={_sharpe(blocked_returns):.3f}")

    # Short-hold conviction analysis
    if all_trades:
        short_holds = [t for t in all_trades if t["hold_days"] <= 3]
        long_holds = [t for t in all_trades if t["hold_days"] > 3]
        short_conv = [t["conviction"] for t in short_holds if not np.isnan(t["conviction"])]
        long_conv = [t["conviction"] for t in long_holds if not np.isnan(t["conviction"])]

        print(f"\n  Short holds (1-3d): n={len(short_holds)}, avg_conviction={np.mean(short_conv):.3f}")
        print(f"  Long holds (4d+):   n={len(long_holds)}, avg_conviction={np.mean(long_conv):.3f}")

    return {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "n_pairs": len(ALL_PAIRS),
        "n_total_trades": len(all_trades),
        "threshold_sweep": sweep_results,
        "verdict": "See threshold_sweep — pick threshold that maximizes Sharpe without dropping >20% of trades",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    results = run_conviction_analysis(args.start, args.end)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUT_PATH}")

    # Print threshold recommendations
    print("\nTHRESHOLD RECOMMENDATIONS:")
    print(f"{'Threshold':12} | {'N Allowed':10} | {'Sharpe':8} | {'WR':6}")
    for r in results.get("threshold_sweep", []):
        marker = " ← current" if abs(r["threshold"] - 0.35) < 0.01 else ""
        print(f"  {r['threshold']:.2f}:       {r['n_allowed']:6d}     {r['allowed_sharpe']:+.3f}  {r['allowed_wr']:.2%}{marker}")


if __name__ == "__main__":
    main()
