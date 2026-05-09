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
    """Return all trade_ledger_*.csv and trade_ledger_*.jsonl files under data/ledger/."""
    if not LEDGER_DIR.exists():
        return []
    files = sorted(LEDGER_DIR.glob("trade_ledger*.csv")) + \
            sorted(LEDGER_DIR.glob("trade_ledger*.jsonl"))
    return files


def _load_ledger(files: List[Path]) -> Optional[pd.DataFrame]:
    """Load and concatenate all ledger CSVs and JSONLs.

    JSONL rows use entry_time and strategy; normalise to CSV column names so
    _enrich_trade() can consume either format without branching.
    """
    if not files:
        return None
    frames = []
    for f in files:
        try:
            if f.suffix == ".jsonl":
                import json as _json
                rows = []
                for line in f.read_text().splitlines():
                    line = line.strip()
                    if line:
                        rows.append(_json.loads(line))
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                # Normalise field names to match CSV schema
                if "entry_time" in df.columns and "entry_date" not in df.columns:
                    df = df.rename(columns={"entry_time": "entry_date"})
                if "strategy" not in df.columns:
                    df["strategy"] = "momentum"
            else:
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


def _hurst_label(h: float) -> str:
    if h > 0.55:
        return "MOMENTUM"
    if h < 0.45:
        return "REVERSION"
    return "NEUTRAL"


def _build_symbol_features(symbols: list, start_date: str, end_date: str) -> dict:
    """Batch-download OHLCV and compute rolling ATR-14 + Hurst-63 per symbol.

    Returns:
        {symbol: pd.DataFrame with DatetimeIndex and columns ['atr_pct', 'hurst']}
    """
    import yfinance as yf

    result = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            raw = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            if raw.empty or len(raw) < 20:
                continue
            raw.columns = [str(c).lower() for c in raw.columns]
            df = raw[["high", "low", "close"]].dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            n = len(df)
            c = df["close"].to_numpy(np.float64)
            h = df["high"].to_numpy(np.float64)
            lo = df["low"].to_numpy(np.float64)
            # Wilder ATR-14
            prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
            tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
            atr = np.zeros(n)
            for i in range(1, n):
                atr[i] = atr[i-1] * (1 - 1/14) + tr[i] * (1/14) if i >= 14 else tr[i]
            atr_pct = np.where(c > 0, atr / c * 100, 0.0)
            # Rolling Hurst (63-bar window)
            log_c = np.log(np.maximum(c, 1e-9))
            hurst_arr = np.full(n, 0.5)
            window = 63
            for i in range(window, n):
                seg = log_c[i-window:i]
                ret = np.diff(seg)
                if len(ret) < 4 or ret.std() < 1e-12:
                    continue
                dev = np.cumsum(ret - ret.mean())
                rs = (dev.max() - dev.min()) / ret.std()
                if rs > 0:
                    hurst_arr[i] = np.log(rs) / np.log(window)
            feat = pd.DataFrame({"atr_pct": atr_pct, "hurst": hurst_arr}, index=df.index)
            result[sym] = feat
        except Exception:
            pass
    return result


def _enrich_trade(row: pd.Series, spy_weekly: pd.Series, veto,
                  sym_features: dict) -> dict:
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

    symbol = str(row.get("symbol", "SPY"))
    regime  = "NEUTRAL"
    atr_pct = 1.5  # sensible fallback (avg equity ATR ~1.5%)

    feat_df = sym_features.get(symbol)
    if feat_df is not None and not feat_df.empty:
        try:
            entry_date_naive = entry_ts.tz_localize(None) if entry_ts.tzinfo else entry_ts
            idx = feat_df.index.asof(entry_date_naive)
            if idx in feat_df.index:
                row_feat = feat_df.loc[idx]
                atr_pct  = float(row_feat["atr_pct"]) or 1.5
                regime   = _hurst_label(float(row_feat["hurst"]))
        except Exception:
            pass

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

    # Batch-fetch ATR + Hurst for all unique symbols (30-day buffer on either side)
    unique_symbols = trades["symbol"].dropna().unique().tolist() if "symbol" in trades.columns else []
    feat_start = str((pd.Timestamp(start) - pd.Timedelta(days=90)).date())
    print(f"[Features] Fetching ATR/Hurst for {len(unique_symbols)} symbols ({feat_start}→{end})...")
    sym_features = _build_symbol_features(unique_symbols, feat_start, end)
    print(f"[Features] Loaded data for {len(sym_features)}/{len(unique_symbols)} symbols")

    # Enrich each trade
    enriched_rows = []
    for _, row in trades.iterrows():
        enriched_rows.append(_enrich_trade(row, spy_weekly, veto, sym_features))

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
