"""P1: the 64 ungated signal builds + external causal VIX-gate masks (HYP-090).

Why 64: signals depend on theta; the hold array depends on (theta, hold) jointly
(build_signal_arrays stamps config.hold_days at macro-signal indices, while
calendar/CPI/CB layers carry their own event holds). Gate-state is NOT a build —
build_signal_arrays is UNGATED (confirmed: discovery/regime.py relies on this),
so the Bull+VIX gate is applied later as a per-pair boolean mask replicating
ForexSignalEngine._apply_vix_regime_gate semantics exactly:
  - SPY/VIX sliced from the STUDY window start (truncated SMA200 warmup — during
    warmup sma is NaN, Close > NaN is False, gate inactive; replicated).
  - .asof lookups for both series; gate suppresses a signal bar iff
    is_bull.asof(date) AND vix.asof(date) > PAIR_VIX_GATES[pair].

All inputs come from the frozen parquets. Output: data/research/modern/signals.npz
holding, per (pair, theta_idx, hold_idx): signal int8 + hold int32 arrays, plus
per-pair: gate mask (bool), atr_pct (float64), close/open (float64), and the
shared date index — everything P2's kernel runs need, network-free.

Run: python3 -m research.modern.signal_arrays          (study span 2015..2026-06)
     python3 -m research.modern.signal_arrays --parity (2015..2024 build, parity test input)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.modern._lib import CACHE_DIR, OUT_DIR, gate_zero, sha256_file
from research.modern.data_freeze import MANIFEST, PAIR_COUNTRIES, PAIRS

THETAS = [0.10, 0.15, 0.20, 0.25]
HOLDS = [30, 45, 60, 90]
PAIR_VIX_GATES = {"USDJPY=X": 15.0, "EURUSD=X": 18.0, "GBPUSD=X": 18.0, "AUDUSD=X": 20.0}

STUDY_START, STUDY_END = "2015-01-01", "2026-06-30"
PARITY_START, PARITY_END = "2015-01-01", "2024-12-31"


def load_frozen(name: str) -> pd.DataFrame:
    path = CACHE_DIR / name
    if not path.exists():
        raise SystemExit(f"frozen input missing: {path} — run data_freeze first")
    return pd.read_parquet(path)


def pair_frame(pair: str, start: str, end: str) -> pd.DataFrame:
    df = load_frozen(pair.replace("=X", "") + ".parquet")
    return df.loc[(df.index >= start) & (df.index <= end)]


def gate_mask(pair: str, index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """True where the Bull+VIX gate would suppress a signal on that date."""
    thr = PAIR_VIX_GATES[pair]
    spy = load_frozen("SPY.parquet")
    vix = load_frozen("VIX.parquet")
    spy = spy.loc[(spy.index >= start) & (spy.index <= end)].copy()
    vix = vix.loc[(vix.index >= start) & (vix.index <= end)]
    spy["sma200"] = spy["Close"].rolling(200).mean()
    is_bull = spy["Close"] > spy["sma200"]              # NaN sma -> False (warmup inactive)
    mask = np.zeros(len(index), dtype=bool)
    for i, date in enumerate(index):
        b = is_bull.asof(date)
        v = vix["Close"].asof(date)
        if (b is not None and b is not np.nan and bool(b)) and (v == v) and float(v) > thr:
            mask[i] = True
    return mask


def build_all(start: str, end: str, out_name: str) -> Path:
    gate_zero()
    from sovereign.forex.data_fetcher import ForexDataFetcher
    from sovereign.forex.forex_backtester import CBEventTrigger  # re-exported (recon-verified)
    from sovereign.forex.signal_engine import ForexSignalEngine, SignalConfig

    fetcher = ForexDataFetcher()
    cb = CBEventTrigger()
    arrays: dict[str, np.ndarray] = {}

    for pair in PAIRS:
        df = pair_frame(pair, start, end)
        close = df["Close"]
        key = pair.replace("=X", "")
        arrays[f"{key}__index"] = close.index.astype("int64").to_numpy()
        arrays[f"{key}__close"] = close.to_numpy(dtype=np.float64)
        arrays[f"{key}__open"] = (df["Open"] if "Open" in df.columns else close).to_numpy(np.float64)
        arrays[f"{key}__gate_mask"] = gate_mask(pair, close.index, start, end)

        base, quote = PAIR_COUNTRIES[pair]
        for ti, theta in enumerate(THETAS):
            for hi, hold in enumerate(HOLDS):
                eng = ForexSignalEngine(
                    fetcher=fetcher, cb_trigger=cb,
                    config=SignalConfig(hold_days=hold, signal_threshold=theta,
                                        cb_surprise_threshold=20),
                )
                sig, hd = eng.build_signal_arrays(
                    close=close, base_country=base, quote_country=quote,
                    start=start, end=end, pair=pair, prices_df=df,
                )
                arrays[f"{key}__sig_t{ti}_h{hi}"] = sig
                arrays[f"{key}__hold_t{ti}_h{hi}"] = hd
                if sig.size:
                    print(f"{pair} θ={theta} H={hold}: {int((sig != 0).sum())} signal bars")

        # ATR from the engine (identical to canonical _simulate_trades usage)
        eng0 = ForexSignalEngine(fetcher=fetcher, cb_trigger=cb, config=SignalConfig())
        atr = eng0._compute_atr_pct(close, df)
        arrays[f"{key}__atr_pct"] = np.asarray(atr, dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    np.savez_compressed(out, **arrays)
    print(f"saved {out} ({len(arrays)} arrays, sha256 {sha256_file(out)[:16]}…)")
    return out


if __name__ == "__main__":
    if "--parity" in sys.argv:
        build_all(PARITY_START, PARITY_END, "signals_parity_2015_2024.npz")
    else:
        build_all(STUDY_START, STUDY_END, "signals.npz")
