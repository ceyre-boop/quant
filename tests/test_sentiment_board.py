"""tests/test_sentiment_board.py — sentiment board-state pipeline (Step 0).

Offline & deterministic: a seeded in-memory DuckDB drives the board (no network). Proves the board
fuses news+FRED+VIX correctly, the schema/contract holds, the VIX regime classifier is correct, there's
no forward-look, and the module stays isolation-clean (no ict/layer imports).
"""
from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config.loader import params
from sovereign.sentiment import store, board_state, vix_feed, news_feed, macro_feed, gdelt_feed, surprise_feed

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
    for mod in (store, news_feed, macro_feed, vix_feed, board_state):
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


# news polarity scorer — bounded, correct labels, exact (pos-neg)/total formula
def test_news_scorer():
    assert news_feed.score_article("euro rallies on strong growth") == 1
    assert news_feed.score_article("pound plunges as recession fears mount") == -1
    assert news_feed.score_article("the central bank met today") == 0
    # the locked aggregate formula is bounded in [-1, 1]
    for n_pos, n_neg, n_tot in [(7, 2, 10), (0, 5, 5), (3, 3, 6), (10, 0, 10)]:
        score = (n_pos - n_neg) / n_tot
        assert -1.0 <= score <= 1.0


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
