"""sovereign/sentiment/news_feed.py — per-pair NewsAPI ingestion + deterministic polarity scoring.

For each configured G10 pair, queries NewsAPI /v2/everything over the rolling window for the pair's topic
list, scores each article with a finance polarity LEXICON (rule-based, NOT an ML model), and upserts the
rolling sentiment to sentiment_news_daily:  news_score = (n_pos − n_neg) / n_total  ∈ [−1, 1].

COVERAGE: NewsAPI free/developer tier serves only ~30 days of history (the `from` param is blocked
beyond that). earliest_article_date() probes the key to report the actual earliest date available.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, env_key, upsert

# ── finance polarity lexicon (deterministic; the model later does the real NLP) ───────────────────
POS_WORDS = {
    "rally", "rallies", "surge", "surges", "gain", "gains", "rise", "rises", "rose", "jump", "jumps",
    "soar", "soars", "strengthen", "strengthens", "bullish", "beat", "beats", "upgrade", "upgraded",
    "optimism", "recovery", "rebound", "boost", "record high", "outperform", "growth", "grows",
    "expand", "expands", "climb", "climbs", "advance", "advances", "upbeat", "resilient", "firmer",
}
NEG_WORDS = {
    "fall", "falls", "fell", "drop", "drops", "plunge", "plunges", "slump", "slumps", "decline",
    "declines", "weaken", "weakens", "bearish", "miss", "misses", "downgrade", "downgraded", "fear",
    "fears", "recession", "crash", "selloff", "sell-off", "tumble", "tumbles", "loss", "losses",
    "slowdown", "contraction", "crisis", "default", "plummet", "plummets", "sink", "sinks", "retreat",
    "worries", "gloom", "downturn", "pressure",
}


def score_article(text: str) -> int:
    """+1 if net-positive, −1 if net-negative, 0 if neutral/tie. Substring match on lowercased text."""
    t = (text or "").lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    return 1 if pos > neg else (-1 if neg > pos else 0)


def fetch_pair(pair: str, topics: list, max_articles: int, window_hours: int) -> list:
    """Articles for one pair over the rolling window. Empty on missing key / failure."""
    try:
        key = env_key("NEWS_API_KEY")
    except RuntimeError:
        return []
    try:
        import requests
        frm = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%S")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": " OR ".join(f'"{t}"' for t in topics), "language": "en",
                    "from": frm, "sortBy": "publishedAt", "pageSize": max_articles, "apiKey": key},
            timeout=20,
        )
        j = r.json()
        if j.get("status") != "ok":
            print(f"  [news] {pair}: API status={j.get('status')} code={j.get('code')} msg={j.get('message')}")
            return []
        return j.get("articles", [])
    except Exception as exc:
        print(f"  [news] {pair}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return []


def update(con=None, as_of: str | None = None) -> dict:
    """Fetch + score the rolling window for every configured pair; upsert today's row per pair."""
    cfg = params["sentiment"]["news"]
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    day = as_of or now.date().isoformat()
    rows, coverage = [], {}
    for pair, topics in cfg["pairs"].items():
        arts = fetch_pair(pair, topics, int(cfg.get("max_articles", 100)), int(cfg.get("rolling_window_hours", 24)))
        n_total = len(arts)
        n_pos = sum(1 for a in arts if score_article(f"{a.get('title') or ''} {a.get('description') or ''}") > 0)
        n_neg = sum(1 for a in arts if score_article(f"{a.get('title') or ''} {a.get('description') or ''}") < 0)
        score = (n_pos - n_neg) / n_total if n_total else None      # NULL when no coverage (honest)
        rows.append((day, pair, n_total, n_pos, n_neg, score, now))
        coverage[pair] = {"n_articles": n_total, "score": score}
    news_df = pd.DataFrame(rows, columns=["date", "pair", "n_articles", "n_pos", "n_neg", "news_score", "fetched_at"])
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    news_df["news_score"] = pd.to_numeric(news_df["news_score"], errors="coerce")  # None → NaN → NULL
    upsert(con, "sentiment_news_daily", news_df, ["date", "pair"])
    if own:
        con.close()
    return coverage


def earliest_article_date() -> dict:
    """Probe how far back NewsAPI serves on this key (the required coverage check)."""
    try:
        key = env_key("NEWS_API_KEY")
    except RuntimeError:
        return {"available": False, "reason": "NEWS_API_KEY not set"}
    try:
        import requests
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "forex", "from": "2015-01-01", "pageSize": 1, "sortBy": "publishedAt",
                    "language": "en", "apiKey": key},
            timeout=20,
        )
        j = r.json()
        if j.get("status") == "error":
            # NewsAPI's error message literally states the earliest date the plan permits.
            return {"available": True, "tier_limited": True, "code": j.get("code"), "message": j.get("message")}
        arts = j.get("articles", [])
        return {"available": True, "tier_limited": False,
                "oldest_returned": arts[-1].get("publishedAt") if arts else None}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
