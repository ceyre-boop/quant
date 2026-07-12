"""P1: freeze every market input to parquet + sha256 manifest (HYP-090).

The study runs ONLY from these frozen files after P1 — yfinance/FRED drift is a
documented repo issue, and a pre-registered study must be reproducible.

Frozen: 4 pair OHLCV (yf, auto_adjust, replicating ForexBacktester._download_price
semantics), SPY + ^VIX (for the causal external VIX-gate mask and regime vector),
and per-pair FRED rate differentials (for the regime vector's rate_diff_mom).

Run: python3 -m research.modern.data_freeze
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

from research.modern._lib import CACHE_DIR, gate_zero, sha256_file, write_json

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
FREEZE_START = "2014-06-01"          # margin for 200d SMA / 252d percentiles
FREEZE_END = "2026-06-30"
MANIFEST = CACHE_DIR / "freeze_manifest.json"

PAIR_COUNTRIES = {                    # mirrors PAIR_CONFIG/CB_TO_COUNTRY resolution
    "EURUSD=X": ("euro_area", "united_states"),
    "GBPUSD=X": ("united_kingdom", "united_states"),
    "USDJPY=X": ("united_states", "japan"),
    "AUDUSD=X": ("australia", "united_states"),
}


def _dl(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=FREEZE_START, end=FREEZE_END,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise SystemExit(f"FREEZE FAIL: empty download for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def main() -> None:
    gate_zero()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}

    for sym in PAIRS + ["SPY", "^VIX"]:
        df = _dl(sym)
        fname = sym.replace("=X", "").replace("^", "") + ".parquet"
        path = CACHE_DIR / fname
        df.to_parquet(path)
        entries[fname] = {"symbol": sym, "rows": len(df),
                          "span": [str(df.index[0].date()), str(df.index[-1].date())],
                          "sha256": sha256_file(path)}
        print(f"frozen {sym:10s} -> {fname} ({len(df)} rows)")

    from sovereign.forex.data_fetcher import ForexDataFetcher
    fetcher = ForexDataFetcher()
    for pair, (base, quote) in PAIR_COUNTRIES.items():
        diff = fetcher.get_pair_differentials(base, quote, start=FREEZE_START)
        fname = pair.replace("=X", "") + "_differentials.parquet"
        path = CACHE_DIR / fname
        diff.to_parquet(path)
        entries[fname] = {"pair": pair, "rows": len(diff),
                          "span": [str(diff.index[0].date()), str(diff.index[-1].date())],
                          "sha256": sha256_file(path)}
        print(f"frozen {pair} differentials ({len(diff)} rows)")

    write_json(MANIFEST, {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_start": FREEZE_START, "freeze_end": FREEZE_END,
        "files": entries,
        "note": ("study inputs frozen; every downstream step reads ONLY these parquets. "
                 "Reconcile (canonical, unfrozen) runs separately by design — it measures "
                 "whether the canonical harness still reproduces 0.6886 on live data."),
    })
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
