"""Sentiment Engine - News sentiment analysis."""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Analyzes news sentiment for market symbols."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    def fetch_news_sentiment(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """Fetch news sentiment for a symbol.

        Args:
            symbol: Ticker symbol
            limit: Max number of articles

        Returns:
            Dict with sentiment metrics
        """
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not set, returning mock data")
            return self._get_mock_sentiment(symbol)

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.api_key,
                "limit": str(limit),
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse sentiment
            feed = data.get("feed", [])

            if not feed:
                return self._get_mock_sentiment(symbol)

            # Calculate aggregate sentiment
            sentiment_scores = []
            for article in feed:
                ticker_sentiment = article.get("ticker_sentiment", [])
                for ts in ticker_sentiment:
                    if ts.get("ticker") == symbol:
                        try:
                            score = float(ts.get("ticker_sentiment_score", 0))
                            sentiment_scores.append(score)
                        except (ValueError, TypeError):
                            continue

            if not sentiment_scores:
                return self._get_mock_sentiment(symbol)

            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            # Categorize
            if avg_sentiment > 0.25:
                sentiment_label = "bullish"
            elif avg_sentiment < -0.25:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            return {
                "symbol": symbol,
                "average_sentiment": avg_sentiment,
                "sentiment_label": sentiment_label,
                "article_count": len(feed),
                "bullish_count": len([s for s in sentiment_scores if s > 0.25]),
                "bearish_count": len([s for s in sentiment_scores if s < -0.25]),
                "neutral_count": len([s for s in sentiment_scores if -0.25 <= s <= 0.25]),
                "raw_feed": feed[:5],  # Keep only first 5 for storage
            }

        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {symbol}: {e}")
            return self._get_mock_sentiment(symbol)

    def _get_mock_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Return mock sentiment for testing."""
        return {
            "symbol": symbol,
            "average_sentiment": 0.0,
            "sentiment_label": "neutral",
            "article_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "raw_feed": [],
        }

    def is_sentiment_extreme(self, sentiment_data: Dict[str, Any]) -> bool:
        """Check if sentiment is at extreme levels.

        Returns True if sentiment is strongly bullish or bearish.
        """
        avg = sentiment_data.get("average_sentiment", 0)
        return abs(avg) > 0.5
