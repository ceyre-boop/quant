"""
sovereign/discovery/regime.py
=============================
Macro-state regime features for the regime-router sub-track, aligned per-pair to
the price index. Mirrors the inputs of the deployed HYP-027 VIX gate
(sovereign/forex/signal_engine._apply_vix_regime_gate) + the rate-differential
trend from data_fetcher.get_pair_differentials.

The discovery substrate's base signals (ForexBatchBacktester via build_signal_arrays)
are already UNGATED — so `adapter.dataset(pair).signals` IS the ungated base the
delta test compares against. No reconstruction needed.

Per-pair regime features: spy_bull (SPY>200SMA), vix, vix_z, spy_ret20,
rate_diff, rate_diff_mom (21d).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY


def load_macro(start: str, end: str) -> pd.DataFrame:
    """Daily SPY/VIX macro state (mirrors the deployed VIX gate's inputs)."""
    import yfinance as yf
    spy = yf.download("SPY", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    for d in (spy, vix):
        if hasattr(d.columns, "get_level_values"):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
    m = pd.DataFrame(index=spy.index)
    m["spy"] = spy["Close"]
    m["spy_sma200"] = spy["Close"].rolling(200).mean()
    m["spy_bull"] = (spy["Close"] > m["spy_sma200"]).astype(float)
    m["spy_ret20"] = spy["Close"].pct_change(20)
    m["vix"] = vix["Close"].reindex(m.index).ffill()
    m["vix_z"] = (m["vix"] - m["vix"].rolling(252).mean()) / m["vix"].rolling(252).std()
    return m


def regime_features(adapter) -> dict:
    """Per-pair regime feature frames aligned to each pair's price index."""
    macro = load_macro(adapter.start, adapter.end)
    fetcher = adapter.batch._backtester._fetcher
    out = {}
    for pair in adapter.pairs:
        idx = adapter.index(pair)
        if idx is None:
            continue
        ma = macro.reindex(idx, method="ffill")
        f = pd.DataFrame(index=idx)
        for c in ("spy_bull", "vix", "vix_z", "spy_ret20"):
            f[c] = ma[c].to_numpy()
        cfg = PAIR_CONFIG.get(pair)
        try:
            bc = CB_TO_COUNTRY[cfg.base_central_bank]
            qc = CB_TO_COUNTRY[cfg.quote_central_bank]
            diffs = fetcher.get_pair_differentials(bc, qc, start=adapter.start)
            diffs.index = pd.to_datetime(diffs.index).tz_localize(None)
            di = diffs.reindex(idx, method="ffill")
            f["rate_diff"] = di["rate_differential"].to_numpy()
            f["rate_diff_mom"] = di["rate_diff_momentum"].to_numpy()
        except Exception:
            f["rate_diff"] = 0.0
            f["rate_diff_mom"] = 0.0
        out[pair] = f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
