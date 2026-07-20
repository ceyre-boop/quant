"""
sovereign/data/reddit_scraper.py
Alta Investments — Reddit Sentiment Engine

Scrapes public Reddit JSON for trading sentiment signals via Reddit's public
.json endpoint.

Uses Reddit's public JSON API — no credentials required.
UA: quant-sentiment/1.0 by Potential-Peanut-695

If Reddit returns 403, it may be a temporary datacenter block. The scraper logs
the failure, writes data/health/reddit_status.json with status FAILED, leaves
the cache untouched, and exits 1.

Subreddits monitored:
  r/wallstreetbets  — equity flow, ticker mentions, momentum signals
  r/Forex           — currency pair sentiment, retail positioning
  r/investing       — broader equity sentiment, macro discussion
  r/stocks          — individual stock flow

Output: data/cache/reddit_sentiment.json
  Per ticker:  mention_count, bull_score, bear_score, net_score, top_posts
  Per pair:    mention_count, bull_score, bear_score, net_score
  Metadata:    last_updated, posts_scanned, subreddit breakdown

RUN:
  python3 -m sovereign.data.reddit_scraper
  python3 sovereign/data/reddit_scraper.py [--verbose]

SCHEDULE: Every 30-60 minutes via agent_scheduler or launchd
"""

from __future__ import annotations

import json
import re
import sys
import time
import logging
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ROOT        = Path(__file__).parent.parent.parent
CACHE_PATH  = ROOT / "data" / "cache" / "reddit_sentiment.json"
LOG_PATH    = ROOT / "logs" / "reddit_scraper.log"
HEALTH_PATH = ROOT / "data" / "health" / "reddit_status.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [reddit] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="a")],
)
log = logging.getLogger(__name__)

UA = "quant-sentiment/1.0 by Potential-Peanut-695"

# ── Subreddits to monitor ─────────────────────────────────────────────────────

SUBREDDITS = {
    "wallstreetbets": {"limit": 100, "focus": "equity",  "weight": 1.5},
    "Forex":          {"limit": 50,  "focus": "forex",   "weight": 2.0},
    "investing":      {"limit": 50,  "focus": "equity",  "weight": 0.8},
    "stocks":         {"limit": 50,  "focus": "equity",  "weight": 0.8},
}

# ── Ticker universe to track ──────────────────────────────────────────────────

EQUITY_TICKERS = {
    "SPY","QQQ","AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD",
    "PLTR","GME","AMC","BBBY","MSTR","COIN","SOFI","RIVN","LCID","F",
    "GLD","SLV","USO","TLT","VIX","UVXY","SQQQ","TQQQ","SPXL","SPXS",
}

FOREX_PAIRS = {
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","GBPJPY","AUDNZD",
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","NZD/USD","GBP/JPY",
    "EURO","POUND","YEN","AUSSIE","LOONIE","KIWI","CABLE","FIBER",
    "EUR","GBP","JPY","AUD","NZD","CAD","CHF",
}

# Canonical pair names for normalization
PAIR_ALIASES = {
    "EURO":"EURUSD","FIBER":"EURUSD","EUR":"EURUSD",
    "CABLE":"GBPUSD","POUND":"GBPUSD","GBP":"GBPUSD",
    "YEN":"USDJPY","JPY":"USDJPY",
    "AUSSIE":"AUDUSD","AUD":"AUDUSD",
    "LOONIE":"USDCAD","CAD":"USDCAD",
    "KIWI":"NZDUSD","NZD":"NZDUSD",
    "EUR/USD":"EURUSD","GBP/USD":"GBPUSD","USD/JPY":"USDJPY",
    "AUD/USD":"AUDUSD","USD/CAD":"USDCAD","NZD/USD":"NZDUSD","GBP/JPY":"GBPJPY",
}

# ── Sentiment keywords ─────────────────────────────────────────────────────────

BULL_WORDS = {
    "buy","bull","bullish","long","calls","moon","mooning","breakout","squeeze",
    "gamma","rip","yolo","all in","loading","accumulate","support","bounce",
    "higher","rally","pumping","up","green","ATH","strong","hold","hodl",
    "buy the dip","BTD","undervalued","cheap","oversold",
}
BEAR_WORDS = {
    "sell","bear","bearish","short","puts","dump","crash","collapse","recession",
    "correction","drop","falling","down","red","bleeding","tank","tanking",
    "overbought","resistance","rejection","overvalued","bubble","hedge","hedging",
    "puts","covered calls","risk off","risk-off",
}

# Words that must accompany a ticker to count (avoids false positives)
# e.g. "AMD" as a standalone word in "I AMD going" — skip it
CONTEXT_REQUIRED = {"F", "A", "C"}  # tickers easily confused with words


# ── Scraper ───────────────────────────────────────────────────────────────────

# Populated by _fetch_subreddit; consumed by run() to decide exit status.
# Maps subreddit -> failure reason string. Empty dict == every fetch succeeded.
FETCH_FAILURES: Dict[str, str] = {}


def _fetch_subreddit(subreddit: str, limit: int, sort: str = "hot") -> List[dict]:
    """Fetch posts from a subreddit JSON endpoint. Returns list of post dicts.

    Tries old.reddit.com first (more permissive with bots), then falls back to
    www.reddit.com. Retries on 429 with exponential backoff (up to 3 attempts).

    Records any failure in FETCH_FAILURES so run() can fail loudly rather than
    writing an empty cache and exiting 0 ("green but empty").
    """
    headers = {
        "User-Agent": UA,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # old.reddit.com is more permissive for JSON bot access than www.reddit.com
    url_candidates = [
        f"https://old.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&raw_json=1",
        f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&raw_json=1",
    ]

    last_error: str = "unknown"
    for url in url_candidates:
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 429:
                    wait = 2 ** attempt * 2  # 2s, 4s, 8s
                    log.warning(f"r/{subreddit}: 429 rate-limited — waiting {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                    last_error = "429"
                    continue
                if r.status_code == 403:
                    log.debug(f"r/{subreddit}: 403 on {url} — trying next URL")
                    last_error = "403"
                    break  # try next url_candidate
                r.raise_for_status()
                data = r.json()
                children = data["data"]["children"]
                posts = [c["data"] for c in children]
                if posts:
                    log.info(f"r/{subreddit}: fetched {len(posts)} posts from {url.split('/')[2]}")
                return posts
            except requests.exceptions.JSONDecodeError as e:
                log.warning(f"r/{subreddit}: JSON decode error on {url} — Reddit returned HTML? {e}")
                last_error = f"JSONDecodeError: {e}"
                break  # try next url_candidate
            except Exception as e:
                log.warning(f"r/{subreddit}: {type(e).__name__} on attempt {attempt+1}: {e}")
                last_error = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            # exhausted retries on this url_candidate
            continue

    log.error(f"REDDIT_FETCH_FAILED: {last_error} — r/{subreddit}: all URL/retry combinations exhausted")
    FETCH_FAILURES[subreddit] = last_error
    return []


def _write_health(status: str, error: str | None, posts: int) -> None:
    """Write data/health/reddit_status.json. Best-effort; never raises."""
    try:
        HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_PATH.write_text(json.dumps({
            "status": status,
            "error":  error,
            "ts":     datetime.now(timezone.utc).isoformat(),
            "posts":  posts,
            "failed_subreddits": dict(FETCH_FAILURES),
        }, indent=2))
    except Exception as e:
        log.warning(f"Failed to write reddit health file: {e}")


def _score_text(text: str) -> Tuple[float, float]:
    """Return (bull_score, bear_score) for a block of text."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    bull = sum(1 for w in BULL_WORDS if w in words)
    bear = sum(1 for w in BEAR_WORDS if w in words)
    return float(bull), float(bear)


def _find_tickers(text: str, focus: str) -> List[str]:
    """Extract ticker/pair mentions from text."""
    found = []
    upper = text.upper()

    if focus in ("equity", "both"):
        # Match $TICKER or standalone uppercase words
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})', upper)
        for t in dollar_tickers:
            if t in EQUITY_TICKERS:
                found.append(t)
        # Standalone words (more careful)
        for ticker in EQUITY_TICKERS:
            if ticker in CONTEXT_REQUIRED:
                # Require $ prefix for ambiguous tickers
                if f"${ticker}" in upper:
                    found.append(ticker)
            else:
                # Word boundary match
                if re.search(rf'\b{re.escape(ticker)}\b', upper):
                    found.append(ticker)

    if focus in ("forex", "both"):
        for pair in FOREX_PAIRS:
            if pair.upper() in upper:
                canonical = PAIR_ALIASES.get(pair.upper(), pair.upper().replace("/",""))
                found.append(canonical)

    return list(set(found))


def scrape_all(verbose: bool = False) -> dict:
    """Scrape all subreddits and aggregate sentiment. Returns structured output."""
    equity_data: Dict[str, dict] = defaultdict(lambda: {
        "mentions": 0, "bull": 0.0, "bear": 0.0, "posts": [], "subreddits": defaultdict(int)
    })
    forex_data: Dict[str, dict] = defaultdict(lambda: {
        "mentions": 0, "bull": 0.0, "bear": 0.0, "posts": [], "subreddits": defaultdict(int)
    })

    total_posts = 0

    for subreddit, cfg in SUBREDDITS.items():
        focus  = cfg["focus"]
        weight = cfg["weight"]
        posts  = _fetch_subreddit(subreddit, cfg["limit"])
        log.info(f"r/{subreddit}: {len(posts)} posts fetched")

        for post in posts:
            title   = post.get("title", "")
            body    = post.get("selftext", "")
            score   = post.get("score", 0)
            text    = f"{title} {body}"
            bull, bear = _score_text(text)
            bull   *= weight
            bear   *= weight
            tickers = _find_tickers(text, focus)

            if not tickers:
                continue

            total_posts += 1
            for ticker in tickers:
                target = forex_data if focus == "forex" or ticker in {
                    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","GBPJPY","AUDNZD"
                } else equity_data

                target[ticker]["mentions"] += 1
                target[ticker]["bull"] += bull
                target[ticker]["bear"] += bear
                target[ticker]["subreddits"][subreddit] += 1

                if score > 50 and title:
                    target[ticker]["posts"].append({
                        "title": title[:100],
                        "score": score,
                        "sub":   subreddit,
                    })

            if verbose and tickers:
                direction = "🟢" if bull > bear else "🔴" if bear > bull else "⬜"
                log.info(f"  {direction} [{subreddit}] {tickers} | {title[:60]}")

        time.sleep(1.5)  # be a good citizen — 1 req/1.5s

    # ── Format output ──────────────────────────────────────────────────────────
    def _format(data: dict) -> dict:
        out = {}
        for ticker, d in sorted(data.items(), key=lambda x: -x[1]["mentions"]):
            net = d["bull"] - d["bear"]
            out[ticker] = {
                "mentions":    d["mentions"],
                "bull_score":  round(d["bull"], 1),
                "bear_score":  round(d["bear"], 1),
                "net_score":   round(net, 1),
                "sentiment":   "BULLISH" if net > 2 else "BEARISH" if net < -2 else "NEUTRAL",
                "top_posts":   sorted(d["posts"], key=lambda p: -p["score"])[:3],
                "subreddits":  dict(d["subreddits"]),
            }
        return out

    result = {
        "last_updated":   datetime.now(timezone.utc).isoformat(),
        "posts_scanned":  total_posts,
        "equity":         _format(equity_data),
        "forex":          _format(forex_data),
        "top_equity":     [],
        "top_forex":      [],
        "summary":        {},
    }

    # Top movers by mention count
    eq_sorted = sorted(result["equity"].items(), key=lambda x: -x[1]["mentions"])
    fx_sorted = sorted(result["forex"].items(),  key=lambda x: -x[1]["mentions"])
    result["top_equity"] = [t for t, _ in eq_sorted[:10]]
    result["top_forex"]  = [t for t, _ in fx_sorted[:5]]

    # Human-readable summary for Oracle context
    eq_summary = ", ".join(
        f"{t}({d['sentiment'][0]}{d['mentions']})"
        for t, d in eq_sorted[:8]
    )
    fx_summary = ", ".join(
        f"{t}({d['sentiment'][0]}{d['mentions']})"
        for t, d in fx_sorted[:5]
    )
    result["summary"] = {
        "equity": eq_summary or "no signals",
        "forex":  fx_summary or "no signals",
    }

    return result


def run(verbose: bool = False) -> dict:
    log.info("Reddit sentiment scrape starting...")
    t0 = time.time()
    data = scrape_all(verbose=verbose)
    elapsed = time.time() - t0

    total_failed = len(FETCH_FAILURES) == len(SUBREDDITS)
    if total_failed:
        # Every source failed. Do NOT overwrite the cache with an empty payload —
        # that is exactly the silent-success bug: a fresh mtime on garbage keeps
        # downstream freshness checks GREEN while sentiment is really dead.
        err = sorted(set(FETCH_FAILURES.values()))[0]
        log.error(f"REDDIT_FETCH_FAILED: all {len(SUBREDDITS)} subreddits failed "
                  f"({FETCH_FAILURES}) — cache left untouched, exiting non-zero.")
        _write_health("FAILED", err, 0)
        data["_failed"] = True
        return data

    status = "DEGRADED" if FETCH_FAILURES else "OK"
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    _write_health(status, sorted(set(FETCH_FAILURES.values()))[0] if FETCH_FAILURES else None,
                  data["posts_scanned"])
    data["_failed"] = False

    log.info(
        f"Done in {elapsed:.1f}s — {data['posts_scanned']} posts | "
        f"equity: {len(data['equity'])} tickers | forex: {len(data['forex'])} pairs"
    )
    log.info(f"Top equity: {data['summary']['equity']}")
    log.info(f"Top forex:  {data['summary']['forex']}")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = run(verbose=args.verbose)
    if result.get("_failed"):
        # Exit non-zero so launchctl records the failure instead of a green run.
        print("REDDIT_FETCH_FAILED — see data/health/reddit_status.json", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nTop equity tickers: {result['top_equity']}")
    print(f"Top forex pairs:    {result['top_forex']}")
