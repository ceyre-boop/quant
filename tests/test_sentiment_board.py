"""tests/test_sentiment_board.py — sentiment board-state pipeline (Step 0).

Offline & deterministic: a seeded in-memory DuckDB drives the board (no network). Proves the board
fuses news+FRED+VIX correctly, the schema/contract holds, the VIX regime classifier is correct, there's
no forward-look, and the module stays isolation-clean (no ict/layer imports).
"""
from __future__ import annotations

import ast
import inspect
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config.loader import params
from sovereign.sentiment import (
    store, board_state, vix_feed, news_feed, macro_feed, gdelt_feed, surprise_feed, cot_feed, vrp_feed,
)

NOW = datetime(2026, 6, 30, tzinfo=timezone.utc)
PAIRS = board_state.PAIRS
# (date, vix_level) chosen to span every regime; macro_curve seeded DISTINCT per date for the no-look test.
SEED = [
    ("2023-03-01", 12.0, -0.10),   # LOW
    ("2023-06-01", 18.0, -0.20),   # NORMAL
    ("2023-09-01", 24.0, -0.30),   # NORMAL
    ("2024-01-02", 27.0, -0.40),   # HIGH
    ("2024-06-03", 33.0, -0.50),   # SPIKE
    ("2024-11-01", 14.0, -0.60),   # LOW
]


@pytest.fixture
def con():
    """In-memory DuckDB seeded with synthetic macro/vix/news for 2023-2024, board rebuilt."""
    c = store.connect(path=":memory:")
    dates = [pd.Timestamp(d).date() for d, _, _ in SEED]

    macro_rows = []
    for (d, _vix, curve) in SEED:
        dd = pd.Timestamp(d).date()
        for sid, val in [("T10Y2Y", curve), ("BAMLH0A0HYM2", 4.2), ("T10YIE", 2.3)]:
            macro_rows.append((dd, sid, val, 0.01, 0.05, NOW))
    macro = pd.DataFrame(macro_rows, columns=["date", "series", "value", "delta_1d", "delta_5d", "fetched_at"])
    store.upsert(c, "sentiment_macro_daily", macro, ["date", "series"])

    vix = pd.DataFrame({"date": dates, "vix_close": [v for _, v, _ in SEED]})
    vix["vix_5d_ago"] = vix["vix_close"] - 1.0
    vix["vix_momentum"] = 1.0
    vix["vix_regime"] = [vix_feed.classify_regime(v) for v in vix["vix_close"]]
    vix["fetched_at"] = NOW
    store.upsert(c, "sentiment_vix_daily", vix, ["date"])

    # news only on a couple of (date, pair) — the rest stay NULL (mirrors the ~30d coverage reality)
    news = pd.DataFrame(
        [(dates[0], "EURUSD", 10, 7, 2, 0.5, NOW), (dates[1], "EURUSD", 8, 2, 5, -0.375, NOW),
         (dates[0], "GBPUSD", 6, 3, 3, 0.0, NOW)],
        columns=["date", "pair", "n_articles", "n_pos", "n_neg", "news_score", "fetched_at"],
    )
    store.upsert(c, "sentiment_news_daily", news, ["date", "pair"])

    # GDELT texture (tone in [-1,1]) on a couple of (date, pair); rest NULL (gdelt starts ~2017)
    gdelt = pd.DataFrame(
        [(dates[0], "EURUSD", 22.0, 0.22, 0.05, 6.1, NOW), (dates[1], "EURUSD", -15.0, -0.15, -0.10, 5.4, NOW)],
        columns=["date", "pair", "tone_raw", "tone_score", "tone_5d", "volume", "fetched_at"],
    )
    store.upsert(c, "sentiment_gdelt_daily", gdelt, ["date", "pair"])

    # daily econ_surprise_z on the seeded trading days (broadcast across pairs)
    surprise = pd.DataFrame({"date": dates, "econ_surprise_z": [0.0, 1.2, 0.6, -0.8, 2.1, 0.1], "fetched_at": NOW})
    store.upsert(c, "sentiment_surprise_daily", surprise, ["date"])

    # COT weekly (EURUSD) — Friday-published; a later row (Nov-29) must NOT be visible to earlier board dates
    cot = pd.DataFrame(
        [(pd.Timestamp("2023-01-03").date(), pd.Timestamp("2023-01-06").date(), "EURUSD", 200000, 150000, 50000, 0.06, 0.70, 800000, NOW),
         (pd.Timestamp("2023-07-04").date(), pd.Timestamp("2023-07-07").date(), "EURUSD", 180000, 200000, -20000, -0.02, 0.20, 850000, NOW),
         (pd.Timestamp("2024-11-26").date(), pd.Timestamp("2024-11-29").date(), "EURUSD", 250000, 100000, 150000, 0.18, 0.95, 820000, NOW)],
        columns=["measurement_date", "publish_date", "pair", "noncomm_long", "noncomm_short",
                 "net_spec", "net_oi", "net_pct", "open_interest", "fetched_at"],
    )
    store.upsert(c, "sentiment_cot_weekly", cot, ["measurement_date", "pair"])

    # VRP (EURUSD) — dated to the observable EOD close (iv_obs_date == rv_last_date == date). A later
    # obs (2024-11-29) must NOT be visible to earlier board dates (ASOF). 2023-07-07 shows RV>IV → negative.
    vrp = pd.DataFrame(
        [(pd.Timestamp("2023-01-06").date(), "EURUSD", "FXE", pd.Timestamp("2023-02-03").date(), 28, 102.0,
          0.085, 0.070, 0.015, 0.60, "bs_invert", pd.Timestamp("2023-01-06").date(), pd.Timestamp("2023-01-06").date(), NOW),
         (pd.Timestamp("2023-07-07").date(), "EURUSD", "FXE", pd.Timestamp("2023-08-04").date(), 28, 106.0,
          0.070, 0.090, -0.020, 0.10, "bs_invert", pd.Timestamp("2023-07-07").date(), pd.Timestamp("2023-07-07").date(), NOW),
         (pd.Timestamp("2024-11-29").date(), "EURUSD", "FXE", pd.Timestamp("2024-12-27").date(), 28, 104.0,
          0.095, 0.060, 0.035, 0.90, "bs_invert", pd.Timestamp("2024-11-29").date(), pd.Timestamp("2024-11-29").date(), NOW)],
        columns=["date", "pair", "symbol", "expiry", "dte", "atm_strike", "iv_atm", "rv_trailing",
                 "vrp_signal", "vrp_pct", "iv_source", "iv_obs_date", "rv_last_date", "fetched_at"],
    )
    store.upsert(c, "sentiment_vrp_daily", vrp, ["date", "pair"])

    board_state.rebuild(con=c)
    yield c
    c.close()


# 1 — get_history non-empty for EURUSD, 2023-2024
def test_get_history_nonempty(con):
    h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
    assert len(h) == len(SEED) > 0


# 2 — all required columns present
def test_required_columns(con):
    h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
    for col in board_state.REQUIRED_COLUMNS:
        assert col in h.columns, f"missing board column {col!r}"


# 3 — news_score (non-null) in [-1, 1]  (NaN-tolerant: historical dates have no NewsAPI coverage)
def test_news_score_bounded(con):
    h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
    nn = h["news_score"].dropna()
    assert len(nn) >= 1                                   # at least the seeded ones
    assert nn.between(-1.0, 1.0).all()
    assert h["news_score"].isna().any()                  # uncovered dates are NULL, not fabricated


# 4 — vix_regime values in the allowed set (+ classifier boundaries)
def test_vix_regime_values(con):
    h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
    assert set(h["vix_regime"].dropna()) <= {"LOW", "NORMAL", "HIGH", "SPIKE"}


def test_classify_regime_boundaries():
    assert vix_feed.classify_regime(14.9) == "LOW"
    assert vix_feed.classify_regime(15.0) == "NORMAL"
    assert vix_feed.classify_regime(25.0) == "NORMAL"
    assert vix_feed.classify_regime(25.1) == "HIGH"
    assert vix_feed.classify_regime(30.0) == "HIGH"
    assert vix_feed.classify_regime(30.1) == "SPIKE"
    assert vix_feed.classify_regime(float("nan")) is None


# 5 — no forward-look: the value seeded at date D is retrievable at D (not shifted to D+1)
def test_no_forward_look(con):
    for (d, _vix, curve) in SEED:
        st = board_state.get_state(pd.Timestamp(d).date(), "EURUSD", con=con)
        assert st is not None
        assert pd.Timestamp(st["date"]).date() == pd.Timestamp(d).date()   # date index == signal date
        assert st["macro_curve"] == pytest.approx(curve)                   # value belongs to ITS date


# 6 — isolation: no sentiment module imports ict / ict-engine / layer1 / layer2 (AST-checked)
def test_sentiment_does_not_import_ict():
    forbidden_roots = {"ict", "ict_engine", "layer1", "layer2"}
    for mod in (store, news_feed, macro_feed, vix_feed, gdelt_feed, surprise_feed, cot_feed, vrp_feed, board_state):
        tree = ast.parse(inspect.getsource(mod))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        for name in names:
            assert name.split(".")[0] not in forbidden_roots, \
                f"{mod.__name__} imports forbidden module {name!r}"


# AV NEWS_SENTIMENT scorer — pair→ticker mapping, directional per-article score, bounded aggregate
def test_pair_tickers_mapping():
    assert news_feed.pair_tickers("EURUSD") == ("FOREX:EUR", "FOREX:USD")
    assert news_feed.pair_tickers("USD_JPY") == ("FOREX:USD", "FOREX:JPY")   # OANDA underscore form
    assert news_feed.pair_tickers("GBPJPY") == ("FOREX:GBP", "FOREX:JPY")
    assert news_feed.pair_tickers("NZDCAD") == ("FOREX:NZD", "FOREX:CAD")    # auto-decomposed, not listed


def _art(**tickers):
    """AV-shaped article: tickers={'FOREX:EUR': (relevance, sentiment), ...}."""
    return {"time_published": "20230115T120000",
            "ticker_sentiment": [{"ticker": t, "relevance_score": str(r), "ticker_sentiment_score": str(s)}
                                 for t, (r, s) in tickers.items()]}


def test_article_pair_score_directional():
    B, Q = "FOREX:EUR", "FOREX:USD"
    # bullish base → pair up (+); bullish quote → pair down (−); neither present → None
    assert news_feed.article_pair_score(_art(**{"FOREX:EUR": (0.8, 0.5)}), B, Q) > 0
    assert news_feed.article_pair_score(_art(**{"FOREX:USD": (0.8, 0.5)}), B, Q) < 0
    assert news_feed.article_pair_score(_art(**{"FOREX:JPY": (0.8, 0.5)}), B, Q) is None
    # relevance weighting: exact formula rel_base*sent_base − rel_quote*sent_quote
    s = news_feed.article_pair_score(_art(**{"FOREX:EUR": (0.5, 0.4), "FOREX:USD": (0.5, 0.2)}), B, Q)
    assert s == pytest.approx(0.5 * 0.4 - 0.5 * 0.2)


def test_news_aggregate_bounded():
    B, Q = "FOREX:EUR", "FOREX:USD"
    arts = [_art(**{"FOREX:EUR": (1.0, 0.3)}), _art(**{"FOREX:USD": (1.0, 0.4)}),
            _art(**{"FOREX:JPY": (1.0, 0.9)})]                      # last mentions neither → ignored
    n_total, n_pos, n_neg, score = news_feed._aggregate(arts, B, Q)
    assert n_total == 2 and n_pos == 1 and n_neg == 1
    assert -1.0 <= score <= 1.0
    # empty / no-coverage → NULL score, not fabricated
    assert news_feed._aggregate([], B, Q) == (0, 0, 0, None)


# ── Step 1: GDELT texture ─────────────────────────────────────────────────────
class TestGdelt:
    def test_normalize_tone_bounded(self):
        for raw in (-100.0, -42.0, 0.0, 37.0, 100.0):
            assert -1.0 <= gdelt_feed.normalize_tone(raw) <= 1.0
        assert gdelt_feed.normalize_tone(50.0) == 0.5

    def test_parse_timeline(self):
        payload = {"timeline": [{"data": [{"date": "20230301T000000Z", "value": 2.5},
                                          {"date": "20230302T000000Z", "value": -1.0}]}]}
        pts = gdelt_feed.parse_timeline(payload)
        assert len(pts) == 2 and pts[0]["value"] == 2.5

    def test_parse_timeline_garbage(self):
        # the plain-text throttle body / malformed payloads must degrade to [] (not crash)
        assert gdelt_feed.parse_timeline({}) == []
        assert gdelt_feed.parse_timeline("Please limit requests to one every 5 seconds") == []
        assert gdelt_feed.parse_timeline({"timeline": []}) == []

    def test_board_gdelt_tone(self, con):
        h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
        nn = h["gdelt_tone"].dropna()
        assert len(nn) >= 1 and nn.between(-1.0, 1.0).all()
        assert h["gdelt_tone"].isna().any()      # NULL where uncovered, not fabricated


# ── Step 1: economic surprise (release innovation) ────────────────────────────
class TestSurprise:
    def test_innovation_and_zscore(self):
        # VARYING month-over-month innovation → defined z-score (a constant innovation has 0 variance → NaN, correctly)
        steps = ([1, 2, 1, 3, 2, 1, 2, 3, 1, 2] * 4)[:40]
        vals = (100 + np.cumsum(steps)).astype(float)
        fp = pd.DataFrame({
            "ref_date": pd.date_range("2018-01-01", periods=40, freq="MS"),
            "publish_date": pd.date_range("2018-02-05", periods=40, freq="MS"),
            "first_print": vals,
        })
        out = surprise_feed.compute_surprise(fp, "prior", zscore_window=12)
        assert out["surprise"].dropna().iloc[-1] == pytest.approx(vals[-1] - vals[-2])  # first_print − prior
        assert out["surprise_z"].notna().any()

    def test_constant_innovation_is_nan_not_garbage(self):
        fp = pd.DataFrame({
            "ref_date": pd.date_range("2018-01-01", periods=30, freq="MS"),
            "publish_date": pd.date_range("2018-02-05", periods=30, freq="MS"),
            "first_print": [100 + i for i in range(30)],   # constant +1 innovation → zero variance
        })
        out = surprise_feed.compute_surprise(fp, "prior", 12)
        assert out["surprise_z"].isna().all()              # no fabricated z on a zero-variance series

    def test_no_forward_look_publish_after_ref(self):
        # the innovation must be dated on the PUBLISH date, which is AFTER the reference period
        fp = pd.DataFrame({
            "ref_date": pd.to_datetime(["2026-05-01"]),
            "publish_date": pd.to_datetime(["2026-06-05"]),   # BLS publishes May data in June
            "first_print": [4.3],
        })
        out = surprise_feed.compute_surprise(fp, "prior", 12)
        assert (pd.to_datetime(out["publish_date"]) > pd.to_datetime(out["ref_date"])).all()

    def test_decay_accumulate(self):
        cal = [f"d{i}" for i in range(11)]
        pulse = {"d0": 2.0}                                   # single pulse at t=0
        z = surprise_feed._decay_accumulate(cal, pulse, halflife_days=5)
        assert z[0] == pytest.approx(2.0)
        assert z[5] == pytest.approx(1.0, abs=0.05)           # halved after one half-life
        assert z[-1] < z[5]                                   # keeps decaying, never re-spikes

    def test_board_econ_surprise(self, con):
        h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
        assert "econ_surprise_z" in h.columns
        assert h["econ_surprise_z"].notna().sum() >= 1

    def test_usd_sign_map_and_no_cancellation(self):
        # the documented sign map: UNRATE inverted, the other 5 USD-positive on a beat
        sign = params["sentiment"]["surprise"]["usd_sign"]
        assert sign["UNRATE"] == -1
        assert all(sign[s] == 1 for s in ["CPIAUCSL", "CPILFESL", "PCEPILFE", "PAYEMS", "RSAFS"])
        # the feeder aggregates surprise_z × usd_sign — verify signs + that a hot jobs report ADDS (no cancel)
        rel = pd.DataFrame({
            "publish_date": ["2023-02-03", "2023-02-03", "2023-01-12"],
            "series": ["PAYEMS", "UNRATE", "UNRATE"],
            "surprise_z": [2.0, -1.0, 2.0],   # hot jobs: payrolls +2, unemployment fell -1 | separate: unemployment rose +2
            "usd_sign": [sign["PAYEMS"], sign["UNRATE"], sign["UNRATE"]],
        })
        rel["usd_z"] = rel["surprise_z"] * rel["usd_sign"]
        assert rel.loc[0, "usd_z"] > 0                    # hot payrolls → USD-positive
        assert rel.loc[2, "usd_z"] < 0                    # rising unemployment → USD-negative (was +ve before fix)
        day = rel[rel["publish_date"] == "2023-02-03"]
        assert day["usd_z"].sum() == pytest.approx(3.0)   # +2 (payrolls) + +1 (unemployment fell) — constructive
        assert day["surprise_z"].sum() == pytest.approx(1.0)  # raw sum cancels to +1 — the bug we fixed


# ── ITEM 3: COT positioning (CFTC, Friday-published, no look-ahead) ───────────
class TestCOT:
    def test_forward_look_publish_after_measurement(self, con):
        bad = con.execute(
            "SELECT COUNT(*) FROM sentiment_cot_weekly WHERE publish_date <= measurement_date").fetchone()[0]
        assert bad == 0                                   # every row dated to its Friday publish, AFTER the Tuesday

    def test_net_pct_bounded(self, con):
        df = con.execute("SELECT net_pct FROM sentiment_cot_weekly WHERE net_pct IS NOT NULL").df()
        assert len(df) >= 1 and df["net_pct"].between(0.0, 1.0).all()

    def test_board_has_cot_columns(self, con):
        h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
        assert "cot_net_pct" in h.columns and "cot_net_oi" in h.columns

    def test_board_asof_no_lookahead(self, con):
        # board sees only COT published on/before its date; a future-published row stays invisible
        assert board_state.get_state(pd.Timestamp("2023-03-01").date(), "EURUSD", con=con)["cot_net_pct"] == pytest.approx(0.70)
        assert board_state.get_state(pd.Timestamp("2023-09-01").date(), "EURUSD", con=con)["cot_net_pct"] == pytest.approx(0.20)
        # 2024-11-01 must still see the 2023-07-07 row (0.20) — the 2024-11-29-published 0.95 is NOT yet public
        assert board_state.get_state(pd.Timestamp("2024-11-01").date(), "EURUSD", con=con)["cot_net_pct"] == pytest.approx(0.20)

    def test_trailing_percentile_no_lookahead(self):
        # the percentile uses TRAILING data only — current value's rank within its own trailing window
        s = pd.Series([10, 20, 30, 40, 5], dtype=float)
        pct = s.rolling(5, min_periods=2).apply(lambda x: float((x <= x[-1]).mean()), raw=True)
        assert pct.iloc[3] == pytest.approx(1.0)          # 40 is the max SO FAR → 100th pct (ignores the future 5)
        assert pct.iloc[4] == pytest.approx(0.2)          # 5 is the min of the 5 → 20th pct

    # ── positioning-layer extension: TFF + 1y/3y percentiles, z-scores, flush, crosses ──

    def test_feature_helpers_causal_pure(self):
        # truncating the future must not change past feature values (trailing-only guarantee)
        s = pd.Series(np.linspace(-5, 7, 80) + np.sin(np.arange(80)), dtype=float)
        for fn in (lambda x: cot_feed._trailing_pct(x, 52, 26), lambda x: cot_feed._trailing_z(x, 52, 26)):
            full, cut = fn(s), fn(s.iloc[:60])
            pd.testing.assert_series_equal(full.iloc[:60], cut, check_names=False)

    def test_cross_frame_leg_difference(self):
        dts = pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-16"])
        base = pd.DataFrame({"measurement_date": dts, "long": [10, 12, 14], "short": [2, 2, 2],
                             "net": [8, 10, 12], "net_oi": [0.4, 0.5, 0.6], "open_interest": [20, 20, 20]})
        quote = pd.DataFrame({"measurement_date": dts[:2], "long": [5, 5], "short": [1, 2],
                              "net": [4, 3], "net_oi": [0.2, 0.15], "open_interest": [20, 20]})
        x = cot_feed._cross_frame(base, quote)
        assert len(x) == 2                                    # inner join — unmatched leg dates drop
        assert list(x["net"]) == [4, 7]                       # base_net − quote_net
        assert x["net_oi"].tolist() == pytest.approx([0.2, 0.35])
        assert (x["open_interest"] == 0).all()                # OI is meaningless for a leg difference

    def test_release_ts_provenance(self):
        df = pd.DataFrame({"measurement_date": pd.to_datetime(["2024-06-04"]),  # a Tuesday
                           "net": [5], "net_oi": [0.1], "long": [6], "short": [1], "open_interest": [50]})
        out = cot_feed._date_stamp(df, lag=3, release_time_et="15:30")
        assert out["publish_date"].iloc[0] == pd.Timestamp("2024-06-07").date()   # the Friday
        assert out["release_ts"].iloc[0] == pd.Timestamp("2024-06-07 15:30")      # 3:30pm ET provenance
        assert out["release_ts"].iloc[0].date() > out["measurement_date"].iloc[0]  # never the Tuesday

    def test_new_columns_schema_roundtrip(self, con):
        cot = pd.DataFrame(
            [(pd.Timestamp("2024-05-28").date(), pd.Timestamp("2024-05-31").date(), "AUDNZD",
              0, 0, 1500, 0.12, 0.97, 0.99, 2.3, 1.8, -2.4, pd.Timestamp("2024-05-31 15:30"), 0, NOW)],
            columns=["measurement_date", "publish_date", "pair", "noncomm_long", "noncomm_short",
                     "net_spec", "net_oi", "net_pct", "net_pct_1y", "net_z_1y", "net_z_3y",
                     "flush_1w", "release_ts", "open_interest", "fetched_at"])
        store.upsert(con, "sentiment_cot_weekly", cot, ["measurement_date", "pair"])
        r = con.execute("SELECT net_pct_1y, net_z_1y, flush_1w, release_ts FROM sentiment_cot_weekly "
                        "WHERE pair='AUDNZD'").fetchone()
        assert r[0] == pytest.approx(0.99) and r[2] == pytest.approx(-2.4)
        assert r[3] == pd.Timestamp("2024-05-31 15:30")

    def test_look_ahead_auditor_clean_on_seeded_db(self, con):
        # the standalone auditor must report zero violations on the honest seeded fixture
        from scripts.audit_look_ahead import audit
        results = audit(con)
        assert results, "auditor returned no checks"
        assert sum(r["violations"] for r in results) == 0, results

    def test_look_ahead_auditor_catches_seeded_leak(self, con):
        # seed a row PUBLISHED BEFORE ITS MEASUREMENT — the auditor must flag exactly that
        from scripts.audit_look_ahead import audit
        leak = pd.DataFrame(
            [(pd.Timestamp("2024-06-04").date(), pd.Timestamp("2024-06-03").date(), "LEAKPAIR",
              1, 1, 0, 0.0, 0.5, 100, NOW)],
            columns=["measurement_date", "publish_date", "pair", "noncomm_long", "noncomm_short",
                     "net_spec", "net_oi", "net_pct", "open_interest", "fetched_at"])
        store.upsert(con, "sentiment_cot_weekly", leak, ["measurement_date", "pair"])
        viol = {(r["table"], r["check"]): r["violations"] for r in audit(con)}
        assert viol[("sentiment_cot_weekly", "publish_after_measurement")] == 1
        con.execute("DELETE FROM sentiment_cot_weekly WHERE pair='LEAKPAIR'")

    def test_tff_table_exists_and_guarded(self, con):
        tff = pd.DataFrame(
            [(pd.Timestamp("2024-05-28").date(), pd.Timestamp("2024-05-31").date(), "EURUSD",
              90000, 40000, 50000, 0.2, 0.88, 0.91, pd.Timestamp("2024-05-31 15:30"), 250000, NOW)],
            columns=["measurement_date", "publish_date", "pair", "lev_long", "lev_short", "lev_net",
                     "lev_net_oi", "lev_net_pct", "lev_net_pct_1y", "release_ts",
                     "open_interest", "fetched_at"])
        store.upsert(con, "sentiment_cot_tff_weekly", tff, ["measurement_date", "pair"])
        bad = con.execute("SELECT COUNT(*) FROM sentiment_cot_tff_weekly "
                          "WHERE publish_date <= measurement_date").fetchone()[0]
        assert bad == 0
        pcts = con.execute("SELECT lev_net_pct, lev_net_pct_1y FROM sentiment_cot_tff_weekly").df()
        assert pcts.stack().between(0, 1).all()


# ── Options SURFACE: ATM term structure + 25Δ RR/BF (HYP-074/075/078/079 substrate) ────────────
class TestOptionsSurface:
    """Offline: the FXE fixture chains were PRICED from a known smile
    (data/fixtures/thetadata/FXE.json smile_params) — the surface math must recover it."""

    def test_smile_recovery_from_known_fixture(self):
        import json
        from sovereign.sentiment.options_surface_feed import FixtureLoader, smile_read
        L = FixtureLoader()
        doc = json.load(open("data/fixtures/thetadata/FXE.json"))
        near = doc["smile_params"]["2026-07-24"]
        out = smile_read(L.get_option_chain("FXE", "2026-06-26", "2026-07-24"),
                         100.0, near["dte"], 0.04, 5)
        assert out is not None and out["n_strikes"] >= 5
        assert out["atm_iv"] == pytest.approx(near["atm"], abs=1e-3)
        # analytic rr25 ≈ skew·(x_c25 − x_p25) with x ≈ ±σ√T·z(0.25); for these params ≈ −0.0031
        assert out["rr25"] == pytest.approx(-0.0031, abs=4e-4)
        assert abs(out["bf25"]) < 5e-4                      # wings ~flat at these tight moneyness levels

    def test_fixture_mode_is_loud_and_stamped(self, con, capsys):
        from sovereign.sentiment import options_surface_feed as osf
        cov = osf.update(con=con, fixture=True)
        printed = capsys.readouterr().out
        assert "FIXTURE MODE — NOT REAL DATA" in printed
        assert any(k.startswith("FIXTURE:") for k in cov)   # coverage keys stamped
        src = con.execute("SELECT DISTINCT iv_source FROM sentiment_options_surface").df()
        assert src["iv_source"].str.startswith("FIXTURE:").all()   # every row stamped

    def test_surface_provenance_and_term_slope(self, con):
        from sovereign.sentiment import options_surface_feed as osf
        osf.update(con=con, fixture=True)
        bad = con.execute("SELECT COUNT(*) FROM sentiment_options_surface WHERE iv_obs_date > date").fetchone()[0]
        assert bad == 0
        row = con.execute("SELECT atm_iv_1m, atm_iv_3m, term_slope, rr25 FROM sentiment_options_surface "
                          "WHERE pair='EURUSD'").fetchone()
        assert row is not None
        assert row[2] == pytest.approx(row[0] - row[1], abs=1e-12)  # slope = 1m − 3m
        assert row[2] < 0                                            # fixture term structure is upward (10% < 12%)
        assert row[3] == pytest.approx(-0.0031, abs=4e-4)

    def test_look_ahead_auditor_covers_surface(self, con):
        from sovereign.sentiment import options_surface_feed as osf
        from scripts.audit_look_ahead import audit
        osf.update(con=con, fixture=True)
        checks = {(r["table"], r["check"]) for r in audit(con)}
        assert ("sentiment_options_surface", "provenance_not_after_date") in checks


# ── VRP-001: implied vol − realized vol FEATURE (ThetaData FX-ETF options) ─────────────────────
class TestVRP:
    # ── the MANDATORY, load-bearing look-ahead guard ──
    def test_forward_look_guard_mandatory(self, con):
        # no IV or RV term may use data dated AFTER the feature's own date — 0 violations, always
        bad = con.execute(
            "SELECT COUNT(*) FROM sentiment_vrp_daily WHERE iv_obs_date > date OR rv_last_date > date"
        ).fetchone()[0]
        assert bad == 0

    def test_feature_is_forward_vol_free_by_construction(self, con):
        # provenance is pinned to the observation date — the forward realized vol (the thing a VRP
        # trade bets on) is structurally absent from the feature; it may only ever be a later OUTCOME.
        df = con.execute("SELECT date, iv_obs_date, rv_last_date FROM sentiment_vrp_daily").df()
        assert (pd.to_datetime(df["iv_obs_date"]) <= pd.to_datetime(df["date"])).all()
        assert (pd.to_datetime(df["rv_last_date"]) <= pd.to_datetime(df["date"])).all()

    # ── RV leg is a PURE function of past+present (truncation-invariant) ──
    def test_rv_trailing_is_causal_pure(self):
        from sovereign.research.vrp.vrp_calculator import realized_variance
        rng = np.random.default_rng(7)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)),
                          index=pd.date_range("2020-01-01", periods=400, freq="B"))
        D = close.index[300]
        full = np.sqrt(realized_variance(close, window=20, forward=False)).asof(D)
        truncated = np.sqrt(realized_variance(close.loc[:D], window=20, forward=False)).asof(D)
        assert full == pytest.approx(truncated)           # deleting the future does not change RV at D

    # ── Black-76 ATM IV inverter round-trips a known sigma (pure, no network) ──
    def test_bs_inversion_roundtrip_atm(self):
        T, r, sigma = 30 / 365.0, 0.04, 0.12
        F = K = 50.0
        price = vrp_feed._bs76_call(F, K, T, r, sigma)     # ATM: call == put (parity, F=K)
        iv = vrp_feed.implied_vol_atm(price, price, K, T, r)
        assert iv == pytest.approx(sigma, abs=1e-3)

    def test_bs_inversion_roundtrip_forward_from_parity(self):
        T, r, sigma = 45 / 365.0, 0.03, 0.09
        F, K = 51.0, 50.0
        call = vrp_feed._bs76_call(F, K, T, r, sigma)
        put = call - math.exp(-r * T) * (F - K)            # put-call parity
        iv = vrp_feed.implied_vol_atm(call, put, K, T, r)  # recovers F=51 from (C−P), then sigma
        assert iv == pytest.approx(sigma, abs=1e-3)

    def test_bs_inversion_unusable_inputs_return_nan(self):
        assert math.isnan(vrp_feed.implied_vol_atm(float("nan"), 1.0, 50.0, 0.1, 0.04))
        assert math.isnan(vrp_feed.implied_vol_atm(0.0, 0.0, 50.0, 0.1, 0.04))     # no time value

    # ── board fusion + ASOF visibility (no forward leak) ──
    def test_board_has_vrp_columns(self, con):
        h = board_state.get_history("2023-01-01", "2024-12-31", "EURUSD", con=con)
        for col in ("vrp_signal", "vrp_pct", "vrp_iv_atm", "vrp_rv_trailing"):
            assert col in h.columns

    def test_board_asof_no_lookahead(self, con):
        # a board date sees only the VRP obs dated on/before it — the future obs stays invisible
        assert board_state.get_state(pd.Timestamp("2023-03-01").date(), "EURUSD", con=con)["vrp_signal"] == pytest.approx(0.015)
        assert board_state.get_state(pd.Timestamp("2023-09-01").date(), "EURUSD", con=con)["vrp_signal"] == pytest.approx(-0.020)
        # 2024-11-01 must still see the 2023-07-07 obs (−0.020) — the 2024-11-29 obs is NOT yet observable
        assert board_state.get_state(pd.Timestamp("2024-11-01").date(), "EURUSD", con=con)["vrp_signal"] == pytest.approx(-0.020)

    def test_rv_spike_compresses_vrp(self, con):
        # eyeball the mechanism: when RV spikes past IV (2023-07-07: rv 0.09 > iv 0.07) VRP goes negative
        st = board_state.get_state(pd.Timestamp("2023-09-01").date(), "EURUSD", con=con)
        assert st["vrp_iv_atm"] < st["vrp_rv_trailing"] and st["vrp_signal"] < 0

    # ── idempotency: delete-then-insert upsert leaves the table identical on a re-run ──
    def test_upsert_idempotent(self, con):
        before = con.execute("SELECT COUNT(*), SUM(vrp_signal) FROM sentiment_vrp_daily").fetchone()
        dup = con.execute("SELECT * FROM sentiment_vrp_daily").df()
        store.upsert(con, "sentiment_vrp_daily", dup, ["date", "pair"])   # re-write the same rows
        after = con.execute("SELECT COUNT(*), SUM(vrp_signal) FROM sentiment_vrp_daily").fetchone()
        assert before == after
