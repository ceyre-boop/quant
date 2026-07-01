"""sovereign/sentiment/news_feed.py — per-pair Alpha Vantage NEWS_SENTIMENT ingestion.

For each configured pair we query AV's NEWS_SENTIMENT endpoint with the pair's two
FOREX: currency tickers and use AV's pre-scored ``ticker_sentiment_score`` (∈ [-1, 1]
per ticker, weighted by ``relevance_score``) — no lexicon. The pair's daily score is
DIRECTIONAL: base-currency sentiment lifts the pair, quote-currency sentiment lowers it:

    article_score = relevance(BASE)·sentiment(BASE) − relevance(QUOTE)·sentiment(QUOTE)
    news_score(day, pair) = clamp( mean(article_score over the day), −1, +1 )   ∈ [−1, 1]

so a positive news_score means "bullish for BASE/QUOTE", matching the prior convention.

COVERAGE: AV NEWS_SENTIMENT serves history back to ~2022. ``time_from``/``time_to``
windowing (see ``backfill``) pages the full 2022→present range one month at a time,
which comfortably fits AV's 1000-article-per-call cap for FX news volume.

RATE LIMITS: free-tier AV returns an ``Information``/``Note`` payload once the daily
quota (25 req) is hit; we detect it, stop, and report rather than hammer. A premium key
lifts the per-minute cap; we still throttle between calls (``_THROTTLE_SECONDS``).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.loader import params
from sovereign.sentiment.store import connect, env_key, upsert

AV_URL = "https://www.alphavantage.co/query"
AV_KEY_NAME = "ALPHA_VANTAGE_API_KEY"

# Politeness delay between AV calls (seconds). Free keys enforce ~1 req/s; premium
# lifts it. Kept ≥1s so a premium upgrade needs no change and free keys stay clean.
_THROTTLE_SECONDS = 1.2

# AV NEWS_SENTIMENT history begins here.
BACKFILL_START = "2022-01-01"

# Explicit pair → (base ticker, quote ticker). Any pair not listed is auto-decomposed
# from a 6-letter symbol (e.g. "EURUSD" → FOREX:EUR / FOREX:USD), so new config pairs
# work with no change here. Listed for the pairs called out in the wiring spec.
PAIR_TICKERS: dict[str, tuple[str, str]] = {
    "USDJPY": ("FOREX:USD", "FOREX:JPY"),
    "EURUSD": ("FOREX:EUR", "FOREX:USD"),
    "GBPUSD": ("FOREX:GBP", "FOREX:USD"),
    "GBPJPY": ("FOREX:GBP", "FOREX:JPY"),
    "AUDUSD": ("FOREX:AUD", "FOREX:USD"),
    "AUDNZD": ("FOREX:AUD", "FOREX:NZD"),
}


def pair_tickers(pair: str) -> tuple[str, str]:
    """(base_ticker, quote_ticker) for a pair, e.g. 'EURUSD' → ('FOREX:EUR','FOREX:USD')."""
    p = (pair or "").upper().replace("_", "").replace("/", "").replace("=X", "")
    if p in PAIR_TICKERS:
        return PAIR_TICKERS[p]
    if len(p) == 6:
        return f"FOREX:{p[:3]}", f"FOREX:{p[3:]}"
    raise ValueError(f"cannot derive FOREX tickers for pair {pair!r}")


def _av_time(dt: datetime) -> str:
    """AV wants YYYYMMDDTHHMM (UTC)."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def _rate_limited(payload: dict) -> bool:
    """True if AV returned a quota/note payload instead of a feed."""
    return any(k in payload for k in ("Note", "Information")) and "feed" not in payload


def fetch_pair(
    pair: str,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 1000,
    sort: str = "EARLIEST",
) -> list:
    """AV NEWS_SENTIMENT feed for one pair's two FOREX tickers. [] on missing key / rate-limit / error.

    Repoints the old NewsAPI call: queries by the pair's currency tickers (not free-text
    topics) and returns AV's pre-scored article feed for downstream ticker_sentiment use.
    """
    try:
        key = env_key(AV_KEY_NAME)
    except RuntimeError:
        return []
    try:
        base, quote = pair_tickers(pair)
    except ValueError as exc:
        print(f"  [news] {pair}: {exc}")
        return []

    p = {
        "function": "NEWS_SENTIMENT",
        "tickers": f"{base},{quote}",
        "limit": str(min(int(limit), 1000)),
        "sort": sort,
        "apikey": key,
    }
    if time_from is not None:
        p["time_from"] = _av_time(time_from)
    if time_to is not None:
        p["time_to"] = _av_time(time_to)

    try:
        import requests
        r = requests.get(AV_URL, params=p, timeout=30)
        j = r.json()
    except Exception as exc:
        print(f"  [news] {pair}: FETCH FAILED ({type(exc).__name__}: {exc})")
        return []

    if _rate_limited(j):
        print(f"  [news] {pair}: AV rate-limited — {j.get('Note') or j.get('Information')}")
        return []
    if "Error Message" in j:
        print(f"  [news] {pair}: AV error — {j.get('Error Message')}")
        return []
    return j.get("feed", []) or []


# ── directional per-pair scoring from AV ticker_sentiment (no lexicon) ──────────────

def _num(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ticker_entry(article: dict, ticker: str) -> tuple[float, float] | None:
    """(relevance_score, ticker_sentiment_score) for `ticker` in this article, or None."""
    for t in article.get("ticker_sentiment", []) or []:
        if t.get("ticker") == ticker:
            return _num(t.get("relevance_score")), _num(t.get("ticker_sentiment_score"))
    return None


def article_pair_score(article: dict, base_ticker: str, quote_ticker: str) -> float | None:
    """Directional push of one article on BASE/QUOTE, relevance-weighted; None if neither ticker present.

    base strengthens → pair up (+); quote strengthens → pair down (−).
    """
    b = _ticker_entry(article, base_ticker)
    q = _ticker_entry(article, quote_ticker)
    if b is None and q is None:
        return None
    score = 0.0
    if b is not None:
        score += b[0] * b[1]
    if q is not None:
        score -= q[0] * q[1]
    return score


def _aggregate(articles: list, base_ticker: str, quote_ticker: str) -> tuple[int, int, int, float | None]:
    """(n_articles, n_pos, n_neg, news_score) for a set of articles about one pair."""
    scores = [s for a in articles if (s := article_pair_score(a, base_ticker, quote_ticker)) is not None]
    if not scores:
        return 0, 0, 0, None
    eps = 1e-9
    n_pos = sum(1 for s in scores if s > eps)
    n_neg = sum(1 for s in scores if s < -eps)
    mean = sum(scores) / len(scores)
    news_score = max(-1.0, min(1.0, mean))     # keep the schema's [-1, 1] contract
    return len(scores), n_pos, n_neg, news_score


def _art_date(article: dict) -> str | None:
    """UTC date (YYYY-MM-DD) from AV time_published 'YYYYMMDDTHHMMSS'."""
    tp = str(article.get("time_published") or "")
    if len(tp) < 8:
        return None
    return f"{tp[0:4]}-{tp[4:6]}-{tp[6:8]}"


# ── rolling-window daily update (the scheduled path) ───────────────────────────────

def update(con=None, as_of: str | None = None) -> dict:
    """Fetch + score the rolling window for every configured pair; upsert today's row per pair."""
    cfg = params["sentiment"]["news"]
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    day = as_of or now.date().isoformat()
    window_hours = int(cfg.get("rolling_window_hours", 48))
    limit = int(cfg.get("max_articles", 1000))
    frm = now - timedelta(hours=window_hours)

    rows, coverage = [], {}
    for pair in cfg["pairs"].keys():
        base, quote = pair_tickers(pair)
        arts = fetch_pair(pair, time_from=frm, time_to=now, limit=limit, sort="LATEST")
        n_total, n_pos, n_neg, score = _aggregate(arts, base, quote)
        rows.append((day, pair, n_total, n_pos, n_neg, score, now))
        coverage[pair] = {"n_articles": n_total, "score": score}
        time.sleep(_THROTTLE_SECONDS)

    news_df = pd.DataFrame(rows, columns=["date", "pair", "n_articles", "n_pos", "n_neg", "news_score", "fetched_at"])
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    news_df["news_score"] = pd.to_numeric(news_df["news_score"], errors="coerce")   # None → NaN → NULL
    upsert(con, "sentiment_news_daily", news_df, ["date", "pair"])
    if own:
        con.close()
    return coverage


# ── historical backfill 2022 → present (monthly windows, rate-limited) ─────────────

def _month_windows(start: datetime, end: datetime):
    """Yield (window_start, window_end) UTC pairs, one per calendar month, start→end."""
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        nxt = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt - timedelta(seconds=1), end)
        cur = nxt


def _months_with_data(con, pair: str) -> set:
    """{'YYYY-MM'} months already populated for this pair (so resume skips them)."""
    try:
        rows = con.execute(
            "SELECT DISTINCT strftime(date, '%Y-%m') FROM sentiment_news_daily "
            "WHERE pair = ? AND news_score IS NOT NULL", [pair]
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def backfill(con=None, start: str = BACKFILL_START, end: str | None = None,
             pairs: list | None = None, resume: bool = True) -> dict:
    """Backfill daily news_score per pair from `start` to `end` via monthly AV windows.

    One AV call per (pair, month); articles are bucketed to their UTC date and each date
    with coverage gets a row. Idempotent via upsert on (date, pair). Days with no articles
    get no row (stay NULL in the board join — honest, not fabricated).

    RESUME (default): skip (pair, month) windows already populated in the DB, so on a
    free AV key (25 calls/day) each run makes forward progress instead of re-spending the
    daily quota on months already fetched. The run stops itself once the quota is hit
    (fetch_pair returns [] on the rate-limit payload); just run again the next day, or
    upgrade the key to finish in one pass. `resume=False` forces a full re-fetch.
    """
    cfg = params["sentiment"]["news"]
    own = con is None
    con = con or connect()
    now = datetime.now(timezone.utc)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else now)
    pair_list = pairs or list(cfg["pairs"].keys())
    limit = int(cfg.get("max_articles", 1000))

    summary: dict = {}
    for pair in pair_list:
        base, quote = pair_tickers(pair)
        done_months = _months_with_data(con, pair) if resume else set()
        rows, n_days, n_arts, n_skipped = [], 0, 0, 0
        for w_start, w_end in _month_windows(start_dt, end_dt):
            if w_start.strftime("%Y-%m") in done_months:
                n_skipped += 1
                continue
            arts = fetch_pair(pair, time_from=w_start, time_to=w_end, limit=limit, sort="EARLIEST")
            time.sleep(_THROTTLE_SECONDS)
            if not arts:
                continue
            by_day: dict[str, list] = {}
            for a in arts:
                d = _art_date(a)
                if d:
                    by_day.setdefault(d, []).append(a)
            for d, day_arts in by_day.items():
                n_total, n_pos, n_neg, score = _aggregate(day_arts, base, quote)
                if score is None:
                    continue
                rows.append((d, pair, n_total, n_pos, n_neg, score, now))
                n_days += 1
                n_arts += n_total
        if rows:
            df = pd.DataFrame(rows, columns=["date", "pair", "n_articles", "n_pos", "n_neg", "news_score", "fetched_at"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["news_score"] = pd.to_numeric(df["news_score"], errors="coerce")
            upsert(con, "sentiment_news_daily", df, ["date", "pair"])
        summary[pair] = {"days": n_days, "articles": n_arts, "months_skipped": n_skipped}
        print(f"  [news-backfill] {pair}: {n_days} days, {n_arts} articles, "
              f"{n_skipped} months already done  {start}→{end_dt.date().isoformat()}")
    if own:
        con.close()
    return summary


def earliest_article_date() -> dict:
    """Probe how far back AV NEWS_SENTIMENT serves on this key (coverage check)."""
    try:
        env_key(AV_KEY_NAME)
    except RuntimeError:
        return {"available": False, "reason": f"{AV_KEY_NAME} not set"}
    arts = fetch_pair("EURUSD", time_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
                      limit=1, sort="EARLIEST")
    if not arts:
        return {"available": True, "oldest_returned": None, "note": "no articles / rate-limited"}
    return {"available": True, "oldest_returned": arts[0].get("time_published")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AV NEWS_SENTIMENT feed — update or backfill")
    ap.add_argument("--backfill", action="store_true", help="Backfill 2022→present (monthly windows)")
    ap.add_argument("--start", default=BACKFILL_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--pair", action="append", help="Limit to specific pair(s); repeatable")
    ap.add_argument("--no-resume", action="store_true", help="Re-fetch months already in the DB")
    args = ap.parse_args()
    if args.backfill:
        print(backfill(start=args.start, end=args.end, pairs=args.pair, resume=not args.no_resume))
    else:
        print(update())
