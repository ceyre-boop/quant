"""
sovereign/data/reddit_scraper.py
Alta Investments — Reddit Sentiment Engine

Scrapes public Reddit JSON (no API key, no auth) for trading sentiment signals.
Uses Reddit's undocumented but stable public .json endpoint.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [reddit] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="a")],
)
log = logging.getLogger(__name__)

UA = "sovereign-quant/1.0 (autonomous trading system research)"

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

def _fetch_subreddit(subreddit: str, limit: int, sort: str = "hot") -> List[dict]:
    """Fetch posts from a subreddit JSON endpoint. Returns list of post dicts."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12)
        r.raise_for_status()
        children = r.json()["data"]["children"]
        return [c["data"] for c in children]
    except Exception as e:
        log.warning(f"r/{subreddit} fetch failed: {e}")
        return []


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

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))

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
    print(json.dumps(result["summary"], indent=2))
    print(f"\nTop equity tickers: {result['top_equity']}")
    print(f"Top forex pairs:    {result['top_forex']}")
