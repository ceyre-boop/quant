"""sovereign/sentiment/ — sentiment board-state data pipeline (Step 0, data layer only).

A SEPARATE machine that sits ALONGSIDE carry — it does not replace it. This package builds the daily
per-pair "board state" vector that fuses three sources (NewsAPI sentiment, FRED macro, VIX regime) into
one row per (trading-day, pair). No model lives here — pipeline only.

Isolation: this package MUST NOT import from ict/ or ict-engine/ (horizontal wall), nor from layer1/
layer2 (vertical wall). It imports only config (thresholds), duckdb (storage), and external data libs.
Enforced by tests/test_sentiment_board.py::test_sentiment_does_not_import_ict.

Coverage note: NewsAPI (free/developer tier) serves only ~30 days of history, so `news_score` is
populatable only for recent dates; FRED + VIX backfill to 2015. The board carries NULL news_score for
uncovered (historical) dates.
"""
from sovereign.sentiment import (
    store, news_feed, macro_feed, vix_feed, gdelt_feed, surprise_feed, cot_feed, board_state,
)

__all__ = ["store", "news_feed", "macro_feed", "vix_feed", "gdelt_feed", "surprise_feed",
           "cot_feed", "board_state"]
