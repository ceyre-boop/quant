"""
Task 3 — Live Trade Enricher

Loads all live trade ledger CSVs, joins each trade with:
  - SPY 5-day return at entry date (from yfinance)
  - Regime label (Hurst-based, from the trade date's Hurst value)
  - ATR% at entry
  - Predicted failure cluster (via ClusterVeto KNN)
  - Cluster description

Saves: logs/live_trades_enriched.csv
Prints: how many live trades matched a known failure cluster
        (= trades that should have been blocked by the veto)

Designed to run weekly as the feedback mechanism.
Run: python3 scripts/enrich_live_trades.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

LOGS         = Path("logs")
LEDGER_DIR   = Path("data/ledger")
ENRICHED_OUT = LOGS / "live_trades_enriched.csv"


# ── Schema of enriched output ─────────────────────────────────────────────────
ENRICHED_COLS = [
    "trade_id", "symbol", "direction", "entry_price",
    "entry_date", "strategy", "confidence",
    # Enriched fields
    "regime", "atr_pct", "spy_week_return",
    "failure_cluster", "cluster_description", "would_have_been_blocked",
]


def _find_ledger_files() -> List[Path]:
    """Return all trade_ledger_*.csv files under data/ledger/."""
    if not LEDGER_DIR.exists():
        return []
    files = sorted(LEDGER_DIR.glob("trade_ledger*.csv"))
    return files


def _load_ledger(files: List[Path]) -> Optional[pd.DataFrame]:
    """Load and concatenate all ledger CSVs."""
    if not files:
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  [warn] Could not read {f}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _fetch_spy_weekly(start_date: str, end_date: str) -> pd.Series:
    """Fetch SPY daily closes and return a Series indexed by date → 5-day return."""
    import yfinance as yf
    raw = yf.download("SPY", start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("SPY", axis=1, level=1)
    raw.columns = [str(c).lower() for c in raw.columns]
    closes = raw["close"]
    weekly = closes.pct_change(5).fillna(0.0)  # decimal fraction (e.g. -0.021 = -2.1%)
    return weekly  # DatetimeIndex → float


def _rolling_hurst_for_date(
    ticker: str, trade_date: str, window: int = 63
) -> float:
    """Download ~3 months of data before trade_date and return Hurst estimate."""
    try:
        import yfinance as yf
        from universe_sweep import _rolling_hurst

        end   = pd.Timestamp(trade_date) + pd.Timedelta(days=5)
        start = pd.Timestamp(trade_date) - pd.Timedelta(days=window * 2)

        raw = yf.download(ticker, start=str(start.date()), end=str(end.date()),
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(ticker, axis=1, level=1)
        raw.columns = [str(c).lower() for c in raw.columns]
        closes = raw["close"].dropna().to_numpy(dtype=np.float64)
        if len(closes) < 10:
            return 0.5

        hurst_arr = _rolling_hurst(closes)
        return float(hurst_arr[-1])
    except Exception:
        return 0.5


def _hurst_label(h: float) -> str:
    if h > 0.55:
        return "MOMENTUM"
    if h < 0.45:
        return "REVERSION"
    return "NEUTRAL"


def _enrich_trade(row: pd.Series, spy_weekly: pd.Series, veto) -> dict:
    """Enrich one trade row. Returns dict with extra fields."""
    entry_date_str = str(row.get("entry_date", row.get("timestamp", "2024-01-01")))
    try:
        entry_ts = pd.Timestamp(entry_date_str)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")
    except Exception:
        entry_ts = pd.Timestamp("2024-01-01", tz="UTC")

    # SPY weekly return on that date
    spy_ret = 0.0
    if not spy_weekly.empty:
        # Align tz-awareness between entry_ts and spy_weekly.index
        if spy_weekly.index.tz is None and entry_ts.tzinfo is not None:
            entry_ts_cmp = entry_ts.tz_localize(None)
        elif spy_weekly.index.tz is not None and entry_ts.tzinfo is None:
            entry_ts_cmp = entry_ts.tz_localize("UTC").tz_convert(spy_weekly.index.tz)
        else:
            entry_ts_cmp = entry_ts
        idx = spy_weekly.index.asof(entry_ts_cmp) if len(spy_weekly) > 0 else None
        if idx is not None and idx in spy_weekly.index:
            spy_ret = float(spy_weekly[idx])

    # Regime from Hurst (would need real OHLCV — use cached or default)
    # For enrichment we use 0.5 as fallback if no feature cube available
    symbol = str(row.get("symbol", "SPY"))

    feature_cube = Path("logs/feature_cube.parquet")
    regime = "NEUTRAL"
    atr_pct = 0.0

    if feature_cube.exists():
        try:
            cube = pd.read_parquet(feature_cube)
            cube.index = pd.to_datetime(cube.index)
            closest = cube.index.asof(entry_ts)
            if closest in cube.index:
                sym_data = cube.loc[closest]
                regime = _hurst_label(float(sym_data.get("hurst", 0.5)))
                atr_pct = float(sym_data.get("atr_pct", 0.0))
        except Exception:
            pass
    else:
        # Fallback: mild estimate from price if ATR is in the row
        atr_raw = float(row.get("atr", 0))
        price   = float(row.get("entry_price", 1))
        atr_pct = (atr_raw / price * 100.0) if price > 0 else 0.0

    strategy = str(row.get("strategy", row.get("specialist", "momentum")))

    # Cluster prediction
    cluster_id   = -1
    cluster_desc = "—"
    blocked      = False

    if veto is not None and veto.ready:
        blk, reason = veto.should_block(
            strategy_name=strategy,
            regime=regime,
            atr_pct=atr_pct,
            spy_week_return=spy_ret,
        )
        blocked = blk
        if blk:
            cluster_desc = reason

    return {
        "trade_id":              row.get("trade_id", ""),
        "symbol":                symbol,
        "direction":             row.get("direction", ""),
        "entry_price":           row.get("entry_price", ""),
        "entry_date":            entry_date_str,
        "strategy":              strategy,
        "confidence":            row.get("confidence", ""),
        "regime":                regime,
        "atr_pct":               round(atr_pct, 4),
        "spy_week_return":       round(spy_ret, 4),
        "failure_cluster":       cluster_id,
        "cluster_description":   cluster_desc,
        "would_have_been_blocked": blocked,
    }


def main():
    LOGS.mkdir(exist_ok=True)

    # Load veto
    from sovereign.risk.cluster_veto import ClusterVeto
    try:
        veto = ClusterVeto()
        print(f"[Veto] Loaded — {len(veto.cluster_info)} clusters")
    except Exception as e:
        veto = None
        print(f"[Veto] Could not load: {e}")

    # Find ledger files
    ledger_files = _find_ledger_files()

    if not ledger_files:
        print(f"\n[Ledger] No trade ledger files found at {LEDGER_DIR}/")
        print("  → No live trades to enrich yet.")
        print("  → Enricher is ready — re-run after trades accumulate.\n")

        # Write empty enriched file with correct schema so downstream tools don't break
        empty = pd.DataFrame(columns=ENRICHED_COLS)
        empty.to_csv(ENRICHED_OUT, index=False)
        print(f"LIVE ENRICHER: 0 live trades, 0 would have been blocked by cluster veto")
        print(f"  (empty enriched file written to {ENRICHED_OUT})")
        return

    print(f"[Ledger] Found {len(ledger_files)} file(s): {[f.name for f in ledger_files]}")

    trades = _load_ledger(ledger_files)
    if trades is None or len(trades) == 0:
        print("[Ledger] Files found but no rows loaded.")
        empty = pd.DataFrame(columns=ENRICHED_COLS)
        empty.to_csv(ENRICHED_OUT, index=False)
        print("LIVE ENRICHER: 0 live trades")
        return

    n_trades = len(trades)
    print(f"[Ledger] {n_trades} trades loaded")

    # Date range for SPY
    date_cols = [c for c in trades.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        dates = pd.to_datetime(trades[date_cols[0]], errors="coerce").dropna()
        start = str((dates.min() - pd.Timedelta(days=10)).date())
        end   = str((dates.max() + pd.Timedelta(days=10)).date())
    else:
        start = "2024-01-01"
        end   = "2025-01-01"

    print(f"[SPY] Fetching weekly returns {start} → {end}...")
    try:
        spy_weekly = _fetch_spy_weekly(start, end)
    except Exception as e:
        print(f"  [warn] SPY fetch failed: {e}")
        spy_weekly = pd.Series(dtype=float)

    # Enrich each trade
    enriched_rows = []
    for _, row in trades.iterrows():
        enriched_rows.append(_enrich_trade(row, spy_weekly, veto))

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(ENRICHED_OUT, index=False)

    n_blocked  = enriched_df["would_have_been_blocked"].sum()
    pct_blocked = n_blocked / n_trades * 100 if n_trades > 0 else 0

    print("\n" + "═" * 60)
    print(f"LIVE ENRICHER: {n_trades} live trades enriched")
    print(f"  Would have been blocked by cluster veto: {n_blocked}/{n_trades} ({pct_blocked:.0f}%)")
    if n_blocked > 0:
        blocked_trades = enriched_df[enriched_df["would_have_been_blocked"]]
        print("\n  Blocked trades:")
        for _, t in blocked_trades.iterrows():
            print(f"    {t['entry_date']}  {t['symbol']:<6}  {t['strategy']:<16}  "
                  f"{t['cluster_description'][:50]}")
    print("═" * 60)
    print(f"\nEnriched file: {ENRICHED_OUT}")


if __name__ == "__main__":
    main()
