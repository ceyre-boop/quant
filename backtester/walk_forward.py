"""Rolling walk-forward validation for the backtester stack.

Why this exists alongside backtest/walk_forward.py: that module is a
calendar-month, symbol-at-a-time harness bolted to BacktestRunner. The
backtester/ stack (engine.py, daily_engine.py, megascan*) works on bar frames
and a strategy config, and until now validated on a SINGLE date split —
megascan's dirty/holdout cut at 2025-07-17, hyp105's 70/30 cut at 2026-04-08.
A single split reports one draw from the parameter-stability distribution; it
cannot distinguish "the edge is stable" from "the split landed kindly".

The contract here is deliberately narrow:

    strategy_fn(train_df, test_df) -> {"rets": [...], "params": {...}}

`strategy_fn` may fit whatever it likes on train_df — thresholds, hold periods,
a model — but it may only *score* on test_df. Everything this module reports is
computed from the concatenated test slices, so an in-sample number cannot reach
the summary even by accident.

Windows are in TRADING ROWS, not calendar days: the frame is assumed to be one
row per bar, sorted ascending, which is what load_daily() returns. Using row
counts avoids the `timedelta(days=30 * months)` calendar drift that the older
harness accumulates over a decade of windows.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import holdout_guard as _hg


def _date_col(data: pd.DataFrame) -> str | None:
    for c in ("date", "session", "ts_event", "ts"):
        if c in data.columns:
            return c
    return None


def _bounds(data: pd.DataFrame) -> tuple[str | None, str | None]:
    col = _date_col(data)
    if col is None or not len(data):
        return None, None
    s = data[col].astype(str)
    return s.iloc[0], s.iloc[-1]


def generate_windows(n: int, train_window: int, test_window: int,
                     min_train: int, anchored: bool = False
                     ) -> list[tuple[int, int, int, int]]:
    """(train_lo, train_hi, test_lo, test_hi) row-index slices, hi exclusive.

    Rolls forward by `test_window` so test slices tile the data without overlap
    — every out-of-sample bar is scored exactly once, which is what makes the
    concatenated series a valid equity curve rather than a resampling.

    anchored=True keeps train_lo pinned at 0 so the training set genuinely
    expands. (The EXPANDING mode in backtest/walk_forward.py sets train_start to
    the global start but leaves train_end = train_start + train_window, so its
    window never actually grows — see that module's generate_windows.)
    """
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive")
    if min_train > train_window:
        raise ValueError(f"min_train {min_train} > train_window {train_window}")

    windows = []
    train_hi = train_window
    while train_hi + test_window <= n:
        train_lo = 0 if anchored else max(0, train_hi - train_window)
        if train_hi - train_lo >= min_train:
            windows.append((train_lo, train_hi, train_hi, train_hi + test_window))
        train_hi += test_window
    return windows


def _metrics(rets: np.ndarray, periods_per_year: float = 252.0) -> dict:
    if not len(rets):
        return dict(n=0, mean=0.0, sharpe=0.0, max_dd=0.0, win=0.0, total=0.0)
    sd = float(rets.std())
    sharpe = float(rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in rets:
        eq = max(eq * (1 + max(float(x), -1.0)), 0.0)
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak if peak > 0 else 1.0)
    return dict(n=int(len(rets)), mean=float(rets.mean()), sharpe=round(sharpe, 4),
                max_dd=round(mdd, 5), win=round(float((rets > 0).mean()), 4),
                total=round(eq - 1, 5))


def walk_forward_backtest(
    strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
    data: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
    min_train: int = 126,
    anchored: bool = False,
    periods_per_year: float = 252.0,
    dataset: str = "equities_daily",
    check_holdout: bool = True,
    context: str = "walk_forward",
) -> dict:
    """Roll a train/test window forward and report pooled out-of-sample results.

    Train on rows [t, t+train_window), test on [t+train_window,
    t+train_window+test_window), roll forward by test_window, concatenate the
    test results.

    Args:
        strategy_fn: fits on train slice, scores on test slice. Must return
            {"rets": sequence of per-trade/per-bar net returns from the TEST
            slice, "params": optional dict of what it fitted}.
        data: bar frame, one row per period, ascending, with a date-like column.
        train_window / test_window / min_train: window sizes in rows.
        anchored: expanding train window pinned at row 0 instead of rolling.
        check_holdout: validate the frame's own date span against the sealed
            holdout before running. Leave on for mining.

    Returns a dict with per-window rows, pooled out-of-sample metrics, and a
    parameter-stability summary (how often the fitted params changed — a
    strategy that re-fits to something different every window is describing
    noise, whatever its pooled Sharpe says).
    """
    data = data.reset_index(drop=True)
    if check_holdout:
        lo, hi = _bounds(data)
        _hg.validate_date_range(lo, hi, context=context, dataset=dataset)

    windows = generate_windows(len(data), train_window, test_window, min_train,
                               anchored=anchored)
    if not windows:
        return {"error": "no windows — data shorter than train_window + "
                         f"test_window ({len(data)} rows < "
                         f"{train_window + test_window})",
                "n_windows": 0, "windows": [], "out_of_sample": _metrics(np.array([]))}

    date_col = _date_col(data)
    rows, pooled, param_sets = [], [], []
    for i, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(windows):
        train_df = data.iloc[tr_lo:tr_hi]
        test_df = data.iloc[te_lo:te_hi]
        out = strategy_fn(train_df, test_df) or {}
        rets = np.asarray(list(out.get("rets") or []), dtype=float)
        params = out.get("params") or {}
        param_sets.append(tuple(sorted((str(k), str(v)) for k, v in params.items())))
        pooled.append(rets)
        m = _metrics(rets, periods_per_year)
        rows.append({
            "window": i + 1,
            "train_rows": [tr_lo, tr_hi], "test_rows": [te_lo, te_hi],
            "train_start": (str(data[date_col].iloc[tr_lo]) if date_col else None),
            "train_end": (str(data[date_col].iloc[tr_hi - 1]) if date_col else None),
            "test_start": (str(data[date_col].iloc[te_lo]) if date_col else None),
            "test_end": (str(data[date_col].iloc[te_hi - 1]) if date_col else None),
            "params": params, **m,
        })

    all_rets = np.concatenate(pooled) if pooled else np.array([])
    oos = _metrics(all_rets, periods_per_year)
    per_window_sharpes = [r["sharpe"] for r in rows if r["n"] > 0]
    scored = [r for r in rows if r["n"] > 0]

    return {
        "config": {"train_window": train_window, "test_window": test_window,
                   "min_train": min_train, "anchored": anchored,
                   "periods_per_year": periods_per_year, "dataset": dataset},
        "n_windows": len(windows),
        "n_windows_scored": len(scored),
        "out_of_sample": oos,
        "window_sharpe": {
            "mean": round(float(np.mean(per_window_sharpes)), 4) if per_window_sharpes else 0.0,
            "std": round(float(np.std(per_window_sharpes)), 4) if per_window_sharpes else 0.0,
            "positive_frac": (round(sum(1 for s in per_window_sharpes if s > 0)
                                    / len(per_window_sharpes), 4)
                              if per_window_sharpes else 0.0),
        },
        "param_stability": {
            "distinct_param_sets": len(set(param_sets)),
            "refit_churn": (round(sum(1 for a, b in zip(param_sets, param_sets[1:])
                                      if a != b) / max(len(param_sets) - 1, 1), 4)),
        },
        "windows": rows,
    }
