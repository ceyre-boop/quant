"""P3: causal regime features from frozen inputs only (HYP-090).

Five features on the union calendar, each standardized by a TRAILING 252-day
rolling mean/std with min 60 observations (full-sample standardization is
look-ahead and forbidden — prereg lock):

  vix_close        ^VIX close (frozen), asof-aligned
  vix_z            252d trailing z of vix_close (z of a z is ~idempotent; kept
                   as locked — two vol features at different horizons)
  spy_bull         SPY Close > SMA200 (trailing) as 0/1
  rate_diff_mom    mean over the 4 pairs of frozen rate_diff_momentum
  atr_pctile       mean over pairs of the trailing-252d percentile rank of ATR%

The A2 regime VECTOR at day t (per window W) is the trailing-W calendar-day mean
of the standardized features — window-aligned with the performance scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.modern._lib import CACHE_DIR, OUT_DIR, to_dtindex
from research.modern.signal_arrays import PAIRS, load_frozen

FEATURES = ["vix_close", "vix_z", "spy_bull", "rate_diff_mom", "atr_pctile"]


def _trailing_pct(s: pd.Series, window: int = 252) -> pd.Series:
    """Trailing percentile rank of the last value in each window (no look-ahead).
    Copied semantics from scripts/research/exit_policy_evolution._trailing_pct."""
    return s.rolling(window, min_periods=60).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True)


def build_features(union: pd.DatetimeIndex) -> pd.DataFrame:
    vix = load_frozen("VIX.parquet")["Close"]
    spy = load_frozen("SPY.parquet")["Close"]

    vix_u = vix.reindex(union.union(vix.index)).ffill().reindex(union)
    spy_u = spy.reindex(union.union(spy.index)).ffill().reindex(union)
    sma200 = spy_u.rolling(200, min_periods=200).mean()
    spy_bull = (spy_u > sma200).astype(float)
    vix_z = (vix_u - vix_u.rolling(252, min_periods=60).mean()) / (
        vix_u.rolling(252, min_periods=60).std() + 1e-12)

    rate_moms = []
    for pair in PAIRS:
        d = load_frozen(pair.replace("=X", "") + "_differentials.parquet")
        rate_moms.append(d["rate_diff_momentum"].reindex(
            union.union(d.index)).ffill().reindex(union))
    rate_diff_mom = pd.concat(rate_moms, axis=1).mean(axis=1)

    sig = np.load(OUT_DIR / "signals.npz")
    atr_pcts = []
    for pair in PAIRS:
        key = pair.replace("=X", "")
        atr = pd.Series(sig[f"{key}__atr_pct"], index=to_dtindex(sig[f"{key}__index"]))
        atr_pcts.append(_trailing_pct(atr).reindex(union).ffill())
    atr_pctile = pd.concat(atr_pcts, axis=1).mean(axis=1)

    df = pd.DataFrame({"vix_close": vix_u, "vix_z": vix_z, "spy_bull": spy_bull,
                       "rate_diff_mom": rate_diff_mom, "atr_pctile": atr_pctile},
                      index=union)

    # trailing standardization (expanding-window guard via min_periods)
    z = pd.DataFrame(index=union)
    for col in FEATURES:
        m = df[col].rolling(252, min_periods=60).mean()
        s = df[col].rolling(252, min_periods=60).std()
        z[col] = (df[col] - m) / (s + 1e-12)
    return z


def regime_vectors(z: pd.DataFrame, window_days: int) -> np.ndarray:
    """Trailing-W calendar-day mean of standardized features; NaN rows where
    insufficient history. Shape (n_days, 5)."""
    rolled = z.rolling(f"{window_days}D", min_periods=20).mean()
    return rolled.to_numpy(dtype=float)
