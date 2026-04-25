"""Unit tests for forex macro engine."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sovereign.forex.cycle_detector import CycleState, PairCycleSignal
from sovereign.forex.fair_value import FairValueSignal
from sovereign.forex.macro_engine import ForexMacroEngine, ForexSignal
from sovereign.forex.pair_universe import ALL_PAIRS


def make_macro(country: str, rate: float, cpi_yoy: float) -> dict:
    """Build a canned macro snapshot."""
    return {
        "country": country,
        "rate": rate,
        "cpi_yoy": cpi_yoy,
        "gdp_growth": 2.0,
        "rate_trajectory": [1, 0, 0],
    }


def make_fv_signal(
    pair: str = "EURUSD=X",
    spot: float = 1.15,
    irp_z_score: float = -2.4,
    ppp_z_score: float = -1.5,
    composite_direction: str = "LONG",
) -> FairValueSignal:
    """Build a fair-value signal with configurable z-scores."""
    return FairValueSignal(
        pair=pair,
        spot=spot,
        irp_fair_value=spot * 0.98,
        irp_z_score=irp_z_score,
        irp_direction="LONG" if irp_z_score < -1.5 else "NEUTRAL",
        ppp_fair_value=spot * 0.99,
        ppp_z_score=ppp_z_score,
        ppp_direction="LONG" if ppp_z_score < -1.5 else "NEUTRAL",
        rate_differential=1.5,
        real_rate_differential=1.0,
        composite_direction=composite_direction,
        composite_strength=0.7,
    )


def make_cycle_signal(
    pair: str = "EURUSD=X",
    divergence_score: float = 0.6,
    direction: str = "LONG",
) -> PairCycleSignal:
    """Build a pair cycle signal with plausible cycle states."""
    base_cycle = CycleState(
        country="EU",
        phase="MID_EXP",
        rate_trajectory="HIKING",
        inflation_trajectory="STABLE",
        gdp_trajectory="ACCELERATING",
        phase_score=1.8,
    )
    quote_cycle = CycleState(
        country="US",
        phase="CONTRACTION",
        rate_trajectory="CUTTING",
        inflation_trajectory="FALLING",
        gdp_trajectory="DECELERATING",
        phase_score=-1.8,
    )
    return PairCycleSignal(
        pair=pair,
        base_cycle=base_cycle,
        quote_cycle=quote_cycle,
        divergence_score=divergence_score,
        direction=direction,
    )


@pytest.fixture
def synthetic_prices():
    """Return 120 synthetic daily close prices."""
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    values = 1.05 + np.linspace(0.0, 0.12, len(index))
    return pd.Series(values, index=index, name="Close")


@pytest.fixture
def macro_lookup():
    """Return canned country macro data used by mocked fetcher calls."""
    return {
        "EU": make_macro("EU", rate=4.0, cpi_yoy=2.0),
        "US": make_macro("US", rate=2.0, cpi_yoy=3.0),
        "UK": make_macro("UK", rate=4.5, cpi_yoy=2.1),
        "JP": make_macro("JP", rate=0.5, cpi_yoy=1.0),
        "AU": make_macro("AU", rate=3.5, cpi_yoy=2.4),
        "NZ": make_macro("NZ", rate=3.8, cpi_yoy=2.2),
        "CA": make_macro("CA", rate=3.0, cpi_yoy=2.0),
        "CH": make_macro("CH", rate=1.5, cpi_yoy=1.2),
    }


@pytest.fixture
def engine_ctx(synthetic_prices, macro_lookup):
    """Create a fully mocked macro engine and expose key mocks."""
    price_frame = pd.DataFrame({"Close": synthetic_prices})

    with patch(
        "sovereign.forex.macro_engine.ForexDataFetcher.get_country_macro"
    ) as mock_get_country_macro, patch(
        "sovereign.forex.macro_engine.yf.download", return_value=price_frame
    ) as mock_download, patch(
        "sovereign.forex.macro_engine.RiskSentimentEngine.override_for_pair",
        return_value=None,
    ) as mock_override, patch(
        "sovereign.forex.macro_engine.FairValueModel.score_pair",
        return_value=make_fv_signal(),
    ) as mock_fv, patch(
        "sovereign.forex.macro_engine.CycleDetector.score_pair",
        return_value=make_cycle_signal(),
    ) as mock_cycle, patch.object(
        ForexMacroEngine,
        "_real_rate_diff_momentum",
        return_value=0.9,
    ) as mock_rdm:
        mock_get_country_macro.side_effect = lambda country: macro_lookup[country]
        engine = ForexMacroEngine()
        yield SimpleNamespace(
            engine=engine,
            mock_get_country_macro=mock_get_country_macro,
            mock_download=mock_download,
            mock_override=mock_override,
            mock_fv=mock_fv,
            mock_cycle=mock_cycle,
            mock_rdm=mock_rdm,
        )


def test_score_pair_long(engine_ctx):
    with patch.object(ForexMacroEngine, "_compute_hurst", return_value=0.8):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is not None
    assert signal.direction == "LONG"
    assert signal.conviction >= 0.35


def test_score_pair_short(engine_ctx):
    engine_ctx.mock_rdm.return_value = -0.9
    engine_ctx.mock_fv.return_value = make_fv_signal(
        irp_z_score=2.4,
        ppp_z_score=1.5,
        composite_direction="SHORT",
    )
    engine_ctx.mock_cycle.return_value = make_cycle_signal(
        divergence_score=-0.6,
        direction="SHORT",
    )

    with patch.object(ForexMacroEngine, "_compute_hurst", return_value=0.2):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is not None
    assert signal.direction == "SHORT"


def test_score_pair_neutral_low_conviction(engine_ctx):
    engine_ctx.mock_rdm.return_value = 0.02
    engine_ctx.mock_fv.return_value = make_fv_signal(
        irp_z_score=0.3,
        ppp_z_score=-0.3,
        composite_direction="NEUTRAL",
    )
    engine_ctx.mock_cycle.return_value = make_cycle_signal(
        divergence_score=0.02,
        direction="NEUTRAL",
    )

    with patch.object(ForexMacroEngine, "_compute_hurst", return_value=0.5):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is not None
    assert signal.direction == "NEUTRAL"
    assert signal.conviction < 0.35


def test_score_pair_risk_override(engine_ctx):
    engine_ctx.mock_rdm.return_value = 0.3
    engine_ctx.mock_override.return_value = "SHORT"
    engine_ctx.mock_fv.return_value = make_fv_signal(
        irp_z_score=0.0,
        ppp_z_score=0.0,
        composite_direction="NEUTRAL",
    )
    engine_ctx.mock_cycle.return_value = make_cycle_signal(
        divergence_score=0.0,
        direction="NEUTRAL",
    )

    with patch.object(ForexMacroEngine, "_compute_hurst", return_value=0.5):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is not None
    assert signal.direction == "SHORT"
    assert signal.conviction == pytest.approx(0.39, rel=1e-6)


def test_score_pair_unknown_pair(engine_ctx):
    assert engine_ctx.engine.score_pair("UNKNOWN=X") is None


def test_score_pair_insufficient_price_history(engine_ctx, synthetic_prices):
    short_history = synthetic_prices.head(40)

    with patch.object(ForexMacroEngine, "_get_price_history", return_value=short_history):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is None


def test_score_pair_fv_skip(engine_ctx):
    engine_ctx.mock_fv.return_value = make_fv_signal(composite_direction="SKIP")

    with patch.object(ForexMacroEngine, "_compute_hurst", return_value=0.8):
        signal = engine_ctx.engine.score_pair("EURUSD=X")

    assert signal is not None
    assert signal.direction == "LONG"
    assert signal.conviction == pytest.approx(0.35, rel=1e-6)


def test_compute_hurst_trending():
    prices = pd.Series(np.cumsum(np.ones(120)) + 100.0)

    hurst = ForexMacroEngine._compute_hurst(prices)
    score = 0.3 if hurst > 0.55 else 0.0

    assert hurst > 0.55
    assert score == 0.3


def test_compute_hurst_mean_reverting():
    returns = np.array([1 if i % 2 == 0 else -1 for i in range(119)], dtype=float)
    prices = pd.Series(100 + np.cumsum(returns))

    hurst = ForexMacroEngine._compute_hurst(prices)
    score = -0.1 if hurst < 0.45 else 0.0

    assert hurst < 0.45
    assert score == -0.1


def test_estimate_hold_rate_diff():
    hold = ForexMacroEngine._estimate_hold("rate_diff_momentum", conviction=1.0)

    assert hold > 0
    assert hold == 58


def test_scan_all_pairs_returns_top3():
    engine = ForexMacroEngine()
    score_map = {
        ALL_PAIRS[0]: ForexSignal(
            pair=ALL_PAIRS[0],
            direction="LONG",
            conviction=0.91,
            hold_period_estimate=45,
            primary_driver="rate_diff_momentum",
            rate_differential=1.5,
            irp_z=-1.8,
            ppp_z=-1.0,
            cycle_divergence=0.7,
            hurst=0.6,
            spot=1.12,
            base_cycle="MID_EXP",
            quote_cycle="CONTRACTION",
        ),
        ALL_PAIRS[1]: ForexSignal(
            pair=ALL_PAIRS[1],
            direction="SHORT",
            conviction=0.82,
            hold_period_estimate=40,
            primary_driver="cycle_divergence",
            rate_differential=-1.0,
            irp_z=1.7,
            ppp_z=0.8,
            cycle_divergence=-0.9,
            hurst=0.4,
            spot=1.27,
            base_cycle="LATE_EXP",
            quote_cycle="MID_EXP",
        ),
        ALL_PAIRS[2]: ForexSignal(
            pair=ALL_PAIRS[2],
            direction="LONG",
            conviction=0.71,
            hold_period_estimate=55,
            primary_driver="ppp_deviation",
            rate_differential=2.0,
            irp_z=-0.9,
            ppp_z=-2.2,
            cycle_divergence=0.3,
            hurst=0.58,
            spot=149.2,
            base_cycle="MID_EXP",
            quote_cycle="CONTRACTION",
        ),
        ALL_PAIRS[3]: ForexSignal(
            pair=ALL_PAIRS[3],
            direction="LONG",
            conviction=0.41,
            hold_period_estimate=47,
            primary_driver="rate_diff_momentum",
            rate_differential=0.9,
            irp_z=-0.6,
            ppp_z=-0.5,
            cycle_divergence=0.2,
            hurst=0.56,
            spot=0.88,
            base_cycle="EARLY_EXP",
            quote_cycle="CONTRACTION",
        ),
        ALL_PAIRS[4]: ForexSignal(
            pair=ALL_PAIRS[4],
            direction="NEUTRAL",
            conviction=0.99,
            hold_period_estimate=30,
            primary_driver="irp_mean_reversion",
            rate_differential=0.0,
            irp_z=0.0,
            ppp_z=0.0,
            cycle_divergence=0.0,
            hurst=0.5,
            spot=1.0,
            base_cycle="EARLY_EXP",
            quote_cycle="EARLY_EXP",
        ),
    }

    with patch.object(engine, "score_pair", side_effect=lambda pair: score_map.get(pair)):
        results = engine.scan_all_pairs()

    assert len(results) <= 3
    assert [signal.pair for signal in results] == [ALL_PAIRS[0], ALL_PAIRS[1], ALL_PAIRS[2]]
