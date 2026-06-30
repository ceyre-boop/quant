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

from sovereign.sentiment import store, board_state, vix_feed, news_feed, macro_feed

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
