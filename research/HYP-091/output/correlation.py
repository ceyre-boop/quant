"""Phase 2 — monthly correlation of TSMOM vs the ACTUAL v015 carry returns, plus
the confirmatory 50/50 combined-portfolio Sharpe.

Primary adjudication is on the ratediff (correct-financing) TSMOM correlation with
v015; the broken-model leg is the apples-to-apples cross-check (v015's CSV was
costed with the broken swap), so its ρ is reported alongside to expose the
financing-regime mismatch's effect. Correlation is scale-invariant; the combined
Sharpe blends at EQUAL VOL (z-scores) so it isn't dominated by the larger-scale leg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign.reporting.equity_curve import _sharpe


def _align(tsmom: pd.Series, v015: pd.Series) -> pd.DataFrame:
    """Align two monthly series on calendar month (Period M), inner join."""
    a = tsmom.copy(); a.index = a.index.to_period("M")
    b = v015.copy(); b.index = b.index.to_period("M")
    df = pd.concat({"tsmom": a, "v015": b}, axis=1).dropna()
    return df


def _corr(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float | None:
    if len(x) < min_n or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _years(idx) -> float:
    return max((idx.max().to_timestamp() - idx.min().to_timestamp()).days / 365.25, 1e-9)


def analyze(tsmom_monthly: pd.Series, v015_monthly: pd.Series, label: str) -> dict:
    df = _align(tsmom_monthly, v015_monthly)
    x, y = df["tsmom"].values, df["v015"].values
    full = _corr(x, y)
    # 2022 rate-shock window
    m2022 = df.index.year == 2022
    r2022 = _corr(x[m2022], y[m2022], min_n=6)
    # rolling 12-month correlation (descriptive)
    roll = df["tsmom"].rolling(12).corr(df["v015"])
    # combined 50/50 equal-vol blend — scale by VOL ONLY (keep the mean, else Sharpe==0)
    sx = df["tsmom"] / (df["tsmom"].std() or 1.0)
    sy = df["v015"] / (df["v015"].std() or 1.0)
    blend = 0.5 * sx + 0.5 * sy
    yrs = _years(df.index)
    s_tsmom = _sharpe(df["tsmom"].tolist(), yrs)
    s_v015 = _sharpe(df["v015"].tolist(), yrs)
    s_blend = _sharpe(blend.tolist(), yrs)
    return {
        "label": label,
        "n_overlap_months": int(len(df)),
        "corr_full": full,
        "corr_full_abs": abs(full) if full is not None else None,
        "corr_2022": r2022,
        "corr_rolling12_last": float(roll.dropna().iloc[-1]) if roll.dropna().size else None,
        "corr_rolling12_mean": float(roll.dropna().mean()) if roll.dropna().size else None,
        "corr_SE_approx": round(1.0 / np.sqrt(len(df)), 4) if len(df) else None,
        "sharpe_tsmom_overlap": s_tsmom,
        "sharpe_v015_overlap": s_v015,
        "sharpe_5050_blend": s_blend,
        "diversification_lift": (s_blend - max(s_tsmom, s_v015)) if None not in (s_tsmom, s_v015, s_blend) else None,
        "null_corr_triggered": (abs(full) > 0.5) if full is not None else None,
    }
