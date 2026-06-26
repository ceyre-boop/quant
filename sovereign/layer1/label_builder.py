"""sovereign/layer1/label_builder.py — forward-direction labels for the Layer-1 model (HYP-064).

fwd_direction_Hd per pair = 1 if close[t+H] > close[t] else 0 (NaN where the forward close is
unavailable). LEAK-FREE BY CONSTRUCTION: data_loader fetches only through 2023-12-31 (the 2024+
holdout is never fetched), so close.shift(-H) is NaN at the series tail and those rows drop out —
the label's forward window can never reach into holdout. No holdout-boundary arithmetic needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign.layer1 import data_loader as dl
from sovereign.layer1.data_loader import LoadReport

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
HORIZONS = (5, 10)


def _forward_direction(close: pd.Series, h: int) -> pd.Series:
    fwd = close.shift(-h)
    diff = fwd - close
    label = pd.Series(np.where(diff.to_numpy() > 0, 1.0, 0.0), index=close.index)
    label[diff.isna()] = np.nan          # tail rows with no forward close stay unlabelled
    return label


def build_labels(pairs: list[str] = PAIRS, report: LoadReport | None = None):
    """Return (labels_df with MultiIndex [date, pair], report)."""
    report = report or LoadReport()
    frames = []
    for pair in pairs:
        close = dl.get_pair_prices(pair, report)
        if close is None:
            continue
        cols = {f"fwd_direction_{h}d": _forward_direction(close, h) for h in HORIZONS}
        df = pd.DataFrame(cols, index=close.index)
        df["pair"] = pair
        df = df.set_index("pair", append=True)        # MultiIndex (date, pair)
        frames.append(df)
    labels = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    return labels, report
