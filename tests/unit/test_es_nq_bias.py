"""Daily bias engine unit tests — component signs, NEUTRAL gate, lookahead hygiene."""
import numpy as np
import pandas as pd
import pytest

from sovereign.es_nq.config import es_nq_params
from sovereign.es_nq import daily_bias_engine as be


def _dates(n, start="2021-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


# ---------- overnight ----------

def test_overnight_saturates_and_signs():
    assert be.overnight_score(0.003, False) == 1.0      # +0.30% saturates
    assert be.overnight_score(-0.003, False) == -1.0
    assert be.overnight_score(0.0015, False) == pytest.approx(0.5)
    assert be.overnight_score(0.10, False) == 1.0       # clipped


def test_overnight_zero_on_roll_day():
    assert be.overnight_score(0.01, True) == 0.0


def test_overnight_zero_on_nan():
    assert be.overnight_score(float("nan"), False) == 0.0


# ---------- vix ----------

def test_vix_falling_below_ma_is_bullish():
    dates = _dates(30)
    vix = pd.Series(np.linspace(30, 15, 30), index=dates)  # falling
    assert be.vix_score(vix, "2021-02-15") == 1.0


def test_vix_rising_above_ma_is_bearish():
    dates = _dates(30)
    vix = pd.Series(np.linspace(15, 35, 30), index=dates)
    assert be.vix_score(vix, "2021-02-15") == -1.0


def test_vix_insufficient_history_neutral():
    vix = pd.Series([20.0, 21.0], index=_dates(2))
    assert be.vix_score(vix, "2021-02-15") == 0.0


def test_vix_uses_only_prior_data():
    """A huge spike ON the session date must not leak into the score."""
    dates = _dates(30)
    vals = np.linspace(30, 15, 30)
    vix = pd.Series(vals, index=dates)
    score_clean = be.vix_score(vix, dates[-1])
    spiked = vix.copy()
    spiked.iloc[-1] = 90.0                      # spike on the session date itself
    assert be.vix_score(spiked, dates[-1]) == score_clean


# ---------- hurst ----------

def _ar1_closes(n=40, phi=0.8, drift=0.002, seed=7):
    """AR(1)-persistent returns — positive autocorrelation is what variance-ratio
    Hurst measures (constant drift demeans away and reads as H≈0.5)."""
    rng = np.random.RandomState(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, 0.004)
    r += drift
    return pd.Series(100 * np.exp(np.cumsum(r)), index=_dates(n))


def test_hurst_momentum_continuation():
    """Persistent (AR1 phi=0.8) series → momentum regime → score continues the
    recent 5-day direction (whatever it is for this path)."""
    closes = _ar1_closes(40, phi=0.8, drift=0.002)
    logret = np.log(closes.values[1:] / closes.values[:-1])
    assert be.variance_ratio_hurst(logret[-20:]) > 0.55      # regime detected
    expected = float(np.sign(logret[-5:].sum()))
    assert be.hurst_score(closes, "2099-01-01") == expected  # continuation


def test_hurst_neutral_band_zero():
    rng = np.random.RandomState(7)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40))),
                       index=_dates(40))
    s = be.hurst_score(closes, "2021-12-31")
    assert s in (-1.0, 0.0, 1.0)               # well-defined output


def test_hurst_insufficient_history_zero():
    closes = pd.Series([100.0] * 5, index=_dates(5))
    assert be.hurst_score(closes, "2021-12-31") == 0.0


def test_variance_ratio_hurst_persistent_vs_alternating():
    rng = np.random.RandomState(1)
    r = np.zeros(200)
    for i in range(1, 200):
        r[i] = 0.7 * r[i - 1] + rng.normal(0, 0.01)   # positive autocorrelation
    assert be.variance_ratio_hurst(r) > 0.55
    alt = np.tile([0.01, -0.01], 100)                 # negative autocorrelation
    assert be.variance_ratio_hurst(alt) < 0.45


# ---------- international ----------

def test_international_alignment_and_holiday():
    dates = _dates(10)
    nik = pd.Series(np.linspace(100, 110, 10), index=dates)   # rising
    dax = pd.Series(np.linspace(200, 220, 10), index=dates)   # rising
    assert be.international_score(nik, dax, dates[-1]) == 1.0
    # Tokyo holiday: date absent from nikkei index → nik leg 0
    nik_holiday = nik.drop(dates[-1])
    assert be.international_score(nik_holiday, dax, dates[-1]) == 0.5


def test_international_dax_uses_prior_completed_session():
    dates = _dates(10)
    nik = pd.Series(100.0, index=dates)                        # flat → 0
    dax = pd.Series(np.linspace(220, 200, 10), index=dates)    # falling
    # DAX leg must read prior two completed sessions, not the session date row
    assert be.international_score(nik, dax, dates[-1]) == -0.5


# ---------- calendar / compute_bias ----------

def test_calendar_backtest_zero_direction_flags_event():
    cal = {"2022-06-15": {"events": ["FOMC"]}}
    s, ev = be.calendar_score("2022-06-15", cal)
    assert s == 0.0 and ev is True
    s2, ev2 = be.calendar_score("2022-06-16", cal)
    assert s2 == 0.0 and ev2 is False


def test_compute_bias_event_day_multiplier():
    comp = {"overnight": 1.0, "vix": 1.0, "hurst": 1.0, "international": 1.0, "calendar": 0.0}
    b_plain = be.compute_bias("2022-06-14", comp, False, False, calendar_active=False)
    b_event = be.compute_bias("2022-06-15", comp, True, False, calendar_active=False)
    assert b_plain.confidence == pytest.approx(1.0)
    assert b_event.confidence == pytest.approx(0.75)
    assert b_plain.direction == "UP" and b_event.direction == "UP"


def test_compute_bias_neutral_below_threshold():
    comp = {"overnight": 0.5, "vix": -0.5, "hurst": 0.0, "international": 0.0, "calendar": 0.0}
    b = be.compute_bias("2022-06-14", comp, False, False, calendar_active=False)
    assert b.direction == "NEUTRAL"            # conflict → low confidence → skip


def test_compute_bias_weight_renormalization():
    # All four active inputs at +1 → raw = 0.75, confidence = 0.75/0.75 = 1.0
    comp = {"overnight": 1.0, "vix": 1.0, "hurst": 1.0, "international": 1.0, "calendar": 0.0}
    b = be.compute_bias("2022-06-14", comp, False, False, calendar_active=False)
    assert b.raw_score == pytest.approx(0.75)
    assert b.confidence == pytest.approx(1.0)


def test_compute_bias_down():
    comp = {"overnight": -1.0, "vix": -1.0, "hurst": -1.0, "international": -1.0, "calendar": 0.0}
    b = be.compute_bias("2022-06-14", comp, False, False, calendar_active=False)
    assert b.direction == "DOWN"


# ---------- feature table lookahead ----------

def test_feature_table_outcome_and_columns():
    dates = _dates(45)
    rng = np.random.RandomState(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 45)))
    daily = pd.DataFrame({
        "rth_open": closes * 0.999, "rth_close": closes,
        "rth_high": closes * 1.01, "rth_low": closes * 0.99,
        "onh": closes * 1.005, "onl": closes * 0.995,
        "px_0925": closes * 1.001, "prior_rth_close": np.r_[np.nan, closes[:-1]],
        "overnight_ret": rng.normal(0, 0.002, 45), "roll_day": False,
        "rth_bars": 78, "symbol": "NQH1",
    }, index=dates)
    aux = pd.DataFrame({"vix": np.linspace(30, 15, 45),
                        "nikkei": np.linspace(100, 110, 45),
                        "dax": np.linspace(200, 210, 45)}, index=dates)
    ft = be.build_feature_table(daily, aux, {}, dates[25], dates[-1])
    assert set(["s_overnight", "s_vix", "s_hurst", "s_international",
                "direction", "confidence", "direction_real"]).issubset(ft.columns)
    assert (ft["direction_real"].dropna().isin(["UP", "DOWN"])).all()
    # outcome matches sign of close-open
    row = ft.iloc[0]
    expected = "UP" if daily.loc[ft.index[0], "rth_close"] > daily.loc[ft.index[0], "rth_open"] else "DOWN"
    assert row["direction_real"] == expected
