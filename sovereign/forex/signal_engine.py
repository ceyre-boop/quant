"""
Forex signal engine — builds per-bar signal arrays without constructing a
pandas DataFrame in the hot path.

Public API
----------
build_signal_arrays(pair, prices_df, base_country, quote_country, fetcher=None)
    Returns (signal_arr, hold_arr) as compact numpy arrays.

build_signal_frame(pair, prices_df, base_country, quote_country, fetcher=None)
    Compatibility wrapper — returns a pandas DataFrame with 'signal' / 'hold'
    columns backed by the same arrays.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Strategy constants kept in sync with ForexBacktester and fast_backtester
HOLD_DAYS: int = 60
SIGNAL_THRESHOLD: float = 0.20


def build_signal_arrays(
    pair: str,
    prices_df: pd.DataFrame,
    base_country: str,
    quote_country: str,
    fetcher=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build compact numpy signal arrays for one pair.

    Parameters
    ----------
    pair          : Yahoo Finance ticker (e.g. ``'EURUSD=X'``)
    prices_df     : OHLCV DataFrame with at least a ``'Close'`` column
    base_country  : Two-letter country code for the base currency (e.g. ``'EU'``)
    quote_country : Two-letter country code for the quote currency (e.g. ``'US'``)
    fetcher       : optional :class:`~sovereign.forex.data_fetcher.ForexDataFetcher`
                    instance; one is created on demand when *None*.

    Returns
    -------
    signal_arr : ``np.ndarray`` shape ``(n_bars,)`` dtype ``float64``
                 Per-bar signal: ``+1.0`` (long), ``-1.0`` (short), ``0.0`` (flat).
                 Only monthly business-month-start bars carry a non-zero value.
    hold_arr   : ``np.ndarray`` shape ``(n_bars,)`` dtype ``int32``
                 Maximum hold duration in bars for each bar (currently constant
                 at :data:`HOLD_DAYS` for all bars).
    """
    from sovereign.forex.data_fetcher import (
        ForexDataFetcher,
        FALLBACK_RATES,
        FALLBACK_CPI,
    )

    if fetcher is None:
        fetcher = ForexDataFetcher()

    close = (
        prices_df["Close"]
        if "Close" in prices_df.columns
        else prices_df.iloc[:, 0]
    )
    n_bars = len(close)

    signal_arr = np.zeros(n_bars, dtype=np.float64)
    hold_arr = np.full(n_bars, HOLD_DAYS, dtype=np.int32)

    if n_bars < 2:
        return signal_arr, hold_arr

    # --- Fetch macro histories once ------------------------------------------
    base_rates = fetcher.get_rate_history(base_country, start="2014-01-01")
    quote_rates = fetcher.get_rate_history(quote_country, start="2014-01-01")
    base_cpi_h = fetcher.get_cpi_history(base_country, start="2014-01-01")
    quote_cpi_h = fetcher.get_cpi_history(quote_country, start="2014-01-01")

    # --- Compute monthly signals -----------------------------------------------
    monthly_dates = close.resample("BMS").first().index
    monthly_signals: dict = {}

    for date in monthly_dates:
        sig = _compute_monthly_signal(
            date,
            close,
            base_rates,
            quote_rates,
            base_cpi_h,
            quote_cpi_h,
            FALLBACK_RATES,
            FALLBACK_CPI,
            base_country,
            quote_country,
        )
        if sig != 0.0:
            monthly_signals[date] = sig

    # --- Expand to per-bar array -----------------------------------------------
    signal_set = set(monthly_signals.keys())
    for i, date in enumerate(close.index):
        if date in signal_set:
            signal_arr[i] = monthly_signals[date]

    return signal_arr, hold_arr


def build_signal_frame(
    pair: str,
    prices_df: pd.DataFrame,
    base_country: str,
    quote_country: str,
    fetcher=None,
) -> pd.DataFrame:
    """
    Compatibility wrapper around :func:`build_signal_arrays`.

    Returns a :class:`pandas.DataFrame` with columns ``'signal'`` and ``'hold'``
    indexed by the price index.
    """
    close = (
        prices_df["Close"]
        if "Close" in prices_df.columns
        else prices_df.iloc[:, 0]
    )
    signal_arr, hold_arr = build_signal_arrays(
        pair, prices_df, base_country, quote_country, fetcher
    )
    return pd.DataFrame(
        {"signal": signal_arr, "hold": hold_arr},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_monthly_signal(
    date,
    close: pd.Series,
    base_rates: pd.Series,
    quote_rates: pd.Series,
    base_cpi_h: pd.Series,
    quote_cpi_h: pd.Series,
    fallback_rates: dict,
    fallback_cpi: dict,
    base_country: str,
    quote_country: str,
) -> float:
    """Return +1, -1, or 0 for the given monthly signal date.

    Replicates the per-date logic from
    :meth:`~sovereign.forex.forex_backtester.ForexBacktester._generate_monthly_signals`.
    """
    # Spot price at this month's date
    try:
        spot = float(
            close.asof(date)
            if hasattr(close, "asof")
            else float(close.loc[:date].iloc[-1])
        )
    except (IndexError, KeyError):
        return 0.0

    if np.isnan(spot) or spot == 0.0:
        return 0.0

    # Central-bank rates
    b_rate = _asof_or_fallback(base_rates, date, fallback_rates.get(base_country, 2.0))
    q_rate = _asof_or_fallback(quote_rates, date, fallback_rates.get(quote_country, 2.0))
    b_cpi = _asof_or_fallback(base_cpi_h, date, fallback_cpi.get(base_country, 2.0))
    q_cpi = _asof_or_fallback(quote_cpi_h, date, fallback_cpi.get(quote_country, 2.0))

    real_rate_diff = (b_rate - b_cpi) - (q_rate - q_cpi)

    # IRP fair value and z-score
    irp_fv = spot * (1 + q_rate / 100.0) / (1 + b_rate / 100.0)
    irp_dev = (spot - irp_fv) / irp_fv if irp_fv != 0.0 else 0.0

    hist = close.loc[:date]
    if len(hist) > 252:
        sigma = hist.pct_change().std() * np.sqrt(252)
        irp_z = irp_dev / (sigma + 1e-8)
    else:
        irp_z = 0.0

    rdm = float(np.clip(real_rate_diff / 4.0, -1.0, 1.0))
    irp_component = float(np.clip(-irp_z / 1.5, -1.0, 1.0))
    macro_score = 0.50 * irp_component + 0.50 * rdm

    # Momentum confirmation (3-month / 63-bar price trend)
    if len(hist) > 63:
        mom = float(hist.iloc[-1] / hist.iloc[-63] - 1.0)
        mom_sign = int(np.sign(mom)) if abs(mom) > 0.005 else 0
    else:
        mom_sign = 0

    macro_sign = int(np.sign(macro_score)) if abs(macro_score) > SIGNAL_THRESHOLD else 0
    if macro_sign != 0 and (mom_sign == 0 or mom_sign == macro_sign):
        return float(macro_sign)
    return 0.0


def _asof_or_fallback(series: pd.Series, date, fallback: float) -> float:
    """Return the most recent value from *series* on or before *date*, or *fallback*."""
    if series is None or len(series) == 0:
        return fallback
    try:
        if date >= series.index[0]:
            val = series.asof(date) if hasattr(series, "asof") else series.loc[:date].iloc[-1]
            if not np.isnan(val):
                return float(val)
    except (KeyError, IndexError):
        pass
    return fallback
