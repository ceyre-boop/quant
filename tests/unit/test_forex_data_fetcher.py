"""Unit tests for sovereign/forex/data_fetcher.py"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.data_fetcher import (
    ForexDataFetcher,
    FALLBACK_RATES,
    FALLBACK_CPI,
    FALLBACK_GDP_GROWTH,
    RATE_TRAJECTORY,
)


# ---------------------------------------------------------------------------
# get_country_macro
# ---------------------------------------------------------------------------

def test_get_country_macro_returns_required_keys():
    fetcher = ForexDataFetcher()
    with patch.object(fetcher, "_fetch_macro", return_value={
        "country": "US",
        "rate": 4.33,
        "cpi_yoy": 2.4,
        "gdp_growth": 2.5,
        "real_rate": 1.93,
        "rate_trajectory": [-1, -1, 0],
        "as_of": "2026-04-01",
    }), patch(
        "sovereign.forex.data_fetcher.Path.exists", return_value=False
    ):
        macro = fetcher.get_country_macro("US")

    for key in ("rate", "cpi_yoy", "gdp_growth", "rate_trajectory"):
        assert key in macro


def test_get_country_macro_fallback_values():
    """When FRED is unavailable, returned values must come from fallback dicts."""
    fetcher = ForexDataFetcher()
    assert not fetcher._fred_ok  # FRED_API_KEY not set in test environment

    with patch(
        "sovereign.forex.data_fetcher.Path.exists", return_value=False
    ):
        macro = fetcher.get_country_macro("EU")

    assert macro["rate"] == pytest.approx(FALLBACK_RATES["EU"])
    assert macro["cpi_yoy"] == pytest.approx(FALLBACK_CPI["EU"])


# ---------------------------------------------------------------------------
# get_rate_history
# ---------------------------------------------------------------------------

def test_get_rate_history_returns_series():
    fetcher = ForexDataFetcher()
    # Mock away the parquet write so we don't need pyarrow
    with patch(
        "sovereign.forex.data_fetcher.Path.exists", return_value=False
    ), patch(
        "pandas.core.frame.DataFrame.to_parquet"
    ):
        series = fetcher.get_rate_history("US", start="2020-01-01")

    assert isinstance(series, pd.Series)
    assert len(series) > 0


def test_get_rate_history_fallback_flat_value():
    """Without FRED the history should be a flat fallback series."""
    fetcher = ForexDataFetcher()
    assert not fetcher._fred_ok

    with patch(
        "sovereign.forex.data_fetcher.Path.exists", return_value=False
    ), patch(
        "pandas.core.frame.DataFrame.to_parquet"
    ):
        series = fetcher.get_rate_history("US", start="2023-01-01")

    assert float(series.iloc[-1]) == pytest.approx(FALLBACK_RATES["US"])


# ---------------------------------------------------------------------------
# get_cpi_history
# ---------------------------------------------------------------------------

def test_get_cpi_history_returns_series():
    fetcher = ForexDataFetcher()
    with patch(
        "sovereign.forex.data_fetcher.Path.exists", return_value=False
    ), patch(
        "pandas.core.frame.DataFrame.to_parquet"
    ):
        series = fetcher.get_cpi_history("EU", start="2020-01-01")

    assert isinstance(series, pd.Series)
    assert len(series) > 0


# ---------------------------------------------------------------------------
# Fallback constants sanity checks
# ---------------------------------------------------------------------------

def test_fallback_rates_all_countries_covered():
    expected = {"US", "EU", "UK", "JP", "CH", "AU", "CA", "NZ"}
    assert expected.issubset(set(FALLBACK_RATES.keys()))


def test_fallback_cpi_all_countries_covered():
    expected = {"US", "EU", "UK", "JP", "CH", "AU", "CA", "NZ"}
    assert expected.issubset(set(FALLBACK_CPI.keys()))


def test_rate_trajectory_lists():
    for country, traj in RATE_TRAJECTORY.items():
        assert len(traj) == 3, f"{country} trajectory should have 3 entries"
        assert all(v in (-1, 0, 1) for v in traj), f"{country} trajectory values invalid"
