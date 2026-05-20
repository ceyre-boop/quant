"""
News Intelligence Engine — Phase 1: Lookahead Research
sovereign/intelligence/news_engine.py

Uses Qwen3 (local Ollama) to classify financial news sentiment.
Phase 1: full lookahead bias — finding what matters.
Phase 2: live pipeline (after Phase 1 shows IC > 0.10).

Run Phase 1 research:
    python3 sovereign/intelligence/news_engine.py --research
    python3 sovereign/intelligence/news_engine.py --research --days 90
    python3 sovereign/intelligence/news_engine.py --status

Cost: $0.00 (Qwen3 local inference via Ollama)
"""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

ROOT      = Path(__file__).resolve().parents[2]
NEWS_DIR  = ROOT / "data" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen3:0.6b"

# Universes to test
EQUITY_TICKERS  = ["META", "UNH", "AMD", "BAC", "JPM", "SPG", "PFE"]
FOREX_PAIRS     = ["GBPUSD", "EURUSD", "AUDUSD", "AUDNZD", "USDJPY", "GBPJPY", "USDCAD"]
ICT_PAIRS       = ["GBPUSD", "EURUSD", "AUDUSD", "AUDNZD"]


# ── Qwen3 classifier ─────────────────────────────────────────────────────────

def _load_vader():
    """Load VADER sentiment analyzer (primary classifier — free, fast, financial-aware)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def classify_headline(headline: str, timeout: int = 10) -> Optional[dict]:
    """
    Classify a headline using VADER (primary) + financial keyword boosters.
    Returns dict with direction, magnitude, confidence, score.
    Falls back to Ollama Qwen3 if VADER unavailable.
    """
    # Financial booster words that VADER misses
    BULLISH_TERMS = {"surge", "surges", "beat", "beats", "jumps", "jump", "soars",
                     "rallies", "rally", "upgrade", "upgraded", "raised guidance",
                     "record", "growth", "profit", "outperform", "strong"}
    BEARISH_TERMS = {"crash", "crashes", "miss", "misses", "plunge", "plunges",
                     "falls", "drop", "drops", "downgrade", "downgraded", "cut",
                     "warns", "warning", "loss", "layoff", "layoffs", "recall"}

    vader = _load_vader()
    if vader:
        scores = vader.polarity_scores(headline)
        compound = scores["compound"]

        # Boost with domain-specific terms
        lower = headline.lower()
        boost = sum(0.15 for t in BULLISH_TERMS if t in lower)
        boost -= sum(0.15 for t in BEARISH_TERMS if t in lower)
        compound = max(-1.0, min(1.0, compound + boost))

        if compound >= 0.1:
            direction, magnitude = "BULLISH", ("HIGH" if compound > 0.5 else "MEDIUM")
        elif compound <= -0.1:
            direction, magnitude = "BEARISH", ("HIGH" if compound < -0.5 else "MEDIUM")
        else:
            direction, magnitude = "NEUTRAL", "LOW"

        return {
            "direction": direction,
            "magnitude": magnitude,
            "confidence": round(abs(compound), 3),
            "score": round(compound, 4),
            "method": "VADER",
        }

    # Fallback: Ollama (if available and Qwen3 works)
    try:
        prompt = (f"Is this headline bullish, bearish, or neutral for the market? "
                  f"Reply one word only. Headline: {headline}")
        resp = requests.post(OLLAMA_URL, json={
            "model": QWEN_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0, "num_predict": 5}
        }, timeout=timeout)
        answer = resp.json().get("response", "").strip().upper()
        if "BULL" in answer:
            return {"direction": "BULLISH", "magnitude": "MEDIUM", "confidence": 0.6, "score": 0.5, "method": "OLLAMA"}
        elif "BEAR" in answer:
            return {"direction": "BEARISH", "magnitude": "MEDIUM", "confidence": 0.6, "score": -0.5, "method": "OLLAMA"}
    except Exception:
        pass

    return {"direction": "NEUTRAL", "magnitude": "LOW", "confidence": 0.0, "score": 0.0, "method": "FALLBACK"}


def _sentiment_score(classification: Optional[dict]) -> float:
    """Extract numeric sentiment score from classification."""
    if not classification:
        return 0.0
    return float(classification.get("score", 0.0))


# ── News fetcher (Yahoo Finance via yfinance) ────────────────────────────────

def _fetch_yahoo_news(ticker: str, days: int = 90) -> list[dict]:
    """Fetch news headlines for a ticker from Yahoo Finance."""
    try:
        import yfinance as yf
        from datetime import datetime as dt
        t = yf.Ticker(ticker)
        news = t.news or []
        cutoff = time.time() - days * 86400
        results = []
        for n in news:
            # Support both old schema (flat) and new schema (nested under 'content')
            content = n.get("content", n)
            title = content.get("title", "")
            pub_date = content.get("pubDate", "") or content.get("displayTime", "")
            if not title:
                continue
            # Parse date
            if pub_date:
                try:
                    ts = dt.fromisoformat(pub_date.replace("Z", "+00:00"))
                    if ts.timestamp() < cutoff:
                        continue
                    date_str = ts.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            results.append({"date": date_str, "headline": title, "ticker": ticker})
        return results
    except Exception:
        return []


def _fetch_price_returns(ticker: str, days: int = 90) -> dict[str, float]:
    """Returns next-day returns keyed by date string."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days+5}d", progress=False)
        if hasattr(df.columns, 'get_level_values'):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        returns = {}
        for i in range(len(close) - 1):
            date_str = str(close.index[i].date())
            ret = (float(close.iloc[i+1]) - float(close.iloc[i])) / float(close.iloc[i])
            returns[date_str] = ret
        return returns
    except Exception:
        return {}


# ── Phase 1 Research ─────────────────────────────────────────────────────────

def run_research(tickers: list[str], days: int = 90) -> dict:
    """
    Phase 1 lookahead research: classify news and compute IC against next-day returns.

    IC = rank correlation between news_sentiment_score and next_day_return.
    IC > 0.10: feature is meaningful. Move to Phase 2.
    IC < 0.05: feature is noise. Don't build it.
    """
    print(f"News Intelligence Research — {len(tickers)} tickers, {days}d window")
    print(f"Model: {QWEN_MODEL} (local Ollama)\n")

    # Check Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:
        print("ERROR: Ollama not running. Start with: ollama serve")
        return {}

    all_results = {}
    combined_pairs: list[tuple[float, float]] = []  # (sentiment, next_day_return)

    for ticker in tickers:
        print(f"  {ticker}: fetching news...")
        articles = _fetch_yahoo_news(ticker, days=days)
        returns  = _fetch_price_returns(ticker, days=days)

        if not articles:
            print(f"  {ticker}: no news found — skipping")
            continue

        print(f"  {ticker}: classifying {len(articles)} headlines...")
        sentiments_by_date: dict[str, list[float]] = {}

        for art in articles:
            cls = classify_headline(art["headline"])
            score = _sentiment_score(cls)
            if score != 0.0:
                sentiments_by_date.setdefault(art["date"], []).append(score)

        # Average sentiment per day, pair with next-day return
        pairs: list[tuple[float, float]] = []
        for date, scores in sentiments_by_date.items():
            if date in returns:
                avg_sent = sum(scores) / len(scores)
                pairs.append((avg_sent, returns[date]))

        if len(pairs) < 10:
            print(f"  {ticker}: only {len(pairs)} matched day/return pairs — low confidence")
            continue

        # Compute Information Coefficient (Spearman rank correlation)
        ic = _rank_correlation([p[0] for p in pairs], [p[1] for p in pairs])
        combined_pairs.extend(pairs)

        result = {
            "ticker": ticker,
            "n_articles": len(articles),
            "n_classified": len(sentiments_by_date),
            "n_matched_days": len(pairs),
            "ic": round(ic, 4),
            "verdict": "SIGNAL" if abs(ic) > 0.10 else ("WEAK" if abs(ic) > 0.05 else "NOISE"),
        }
        all_results[ticker] = result
        verdict_icon = "✅" if result["verdict"] == "SIGNAL" else ("🟡" if result["verdict"] == "WEAK" else "⬜")
        print(f"  {ticker}: IC={ic:+.4f} n={len(pairs)} {verdict_icon} {result['verdict']}")

    # Combined IC across all tickers
    if combined_pairs:
        combined_ic = _rank_correlation(
            [p[0] for p in combined_pairs],
            [p[1] for p in combined_pairs]
        )
        print(f"\n  Combined IC ({len(combined_pairs)} day/ticker pairs): {combined_ic:+.4f}")
        all_results["_combined"] = {
            "n_pairs": len(combined_pairs),
            "ic": round(combined_ic, 4),
            "verdict": "SIGNAL" if abs(combined_ic) > 0.10 else ("WEAK" if abs(combined_ic) > 0.05 else "NOISE"),
        }

    # Save
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": QWEN_MODEL,
        "days_window": days,
        "tickers": tickers,
        "results": all_results,
        "phase1_conclusion": _summarize(all_results),
    }
    out_path = NEWS_DIR / f"phase1_research_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved: {out_path}")
    return out


def _rank_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation — the Information Coefficient."""
    if len(x) < 5:
        return 0.0
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return float(corr) if corr == corr else 0.0  # nan check
    except ImportError:
        # Manual rank correlation fallback
        def rank(arr):
            sorted_arr = sorted(range(len(arr)), key=lambda i: arr[i])
            ranks = [0] * len(arr)
            for rank_i, orig_i in enumerate(sorted_arr):
                ranks[orig_i] = rank_i
            return ranks
        rx, ry = rank(x), rank(y)
        n = len(rx)
        mean_rx = sum(rx) / n
        mean_ry = sum(ry) / n
        cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n)) / n
        std_rx = (sum((r - mean_rx)**2 for r in rx) / n) ** 0.5
        std_ry = (sum((r - mean_ry)**2 for r in ry) / n) ** 0.5
        return cov / (std_rx * std_ry) if std_rx * std_ry > 0 else 0.0


def _summarize(results: dict) -> str:
    signals = [k for k, v in results.items() if v.get("verdict") == "SIGNAL" and not k.startswith("_")]
    weak    = [k for k, v in results.items() if v.get("verdict") == "WEAK" and not k.startswith("_")]
    noise   = [k for k, v in results.items() if v.get("verdict") == "NOISE" and not k.startswith("_")]
    combined_ic = results.get("_combined", {}).get("ic", 0)

    if len(signals) >= 3 or abs(combined_ic) > 0.10:
        return f"PROCEED TO PHASE 2. {len(signals)} tickers show SIGNAL IC>0.10. Combined IC={combined_ic:+.4f}."
    elif len(signals) >= 1 or abs(combined_ic) > 0.05:
        return f"INVESTIGATE FURTHER. {len(signals)} signals, {len(weak)} weak. Combined IC={combined_ic:+.4f}. Test with more data before Phase 2."
    else:
        return f"SKIP NEWS LAYER. No tickers show IC>0.05. Combined IC={combined_ic:+.4f}. News doesn't predict next-day returns for this universe."


# ── Daily live sentiment (Phase 2, when IC validated) ───────────────────────

def get_daily_sentiment(tickers: list[str], date: Optional[str] = None) -> dict[str, float]:
    """
    Phase 2: Get today's news sentiment for each ticker.
    Returns dict of ticker → sentiment_score [-1, +1].
    Called by equity orchestrator before signal generation.
    """
    today = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_path = NEWS_DIR / f"sentiment_{today}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    result = {}
    for ticker in tickers:
        articles = _fetch_yahoo_news(ticker, days=2)
        today_articles = [a for a in articles if a["date"] == today]
        if today_articles:
            scores = [_sentiment_score(classify_headline(a["headline"])) for a in today_articles]
            scores = [s for s in scores if s != 0.0]
            result[ticker] = round(sum(scores) / len(scores), 4) if scores else 0.0
        else:
            result[ticker] = 0.0

    cache_path.write_text(json.dumps(result, indent=2))
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News Intelligence Engine")
    parser.add_argument("--research",  action="store_true", help="Run Phase 1 lookahead research")
    parser.add_argument("--equity",    action="store_true", help="Research on equity universe only")
    parser.add_argument("--forex",     action="store_true", help="Research on forex pairs")
    parser.add_argument("--days",      type=int, default=90, help="Lookback window in days")
    parser.add_argument("--status",    action="store_true", help="Show last research results")
    parser.add_argument("--classify",  type=str, help="Classify a single headline")
    args = parser.parse_args()

    if args.classify:
        result = classify_headline(args.classify)
        print(json.dumps(result, indent=2))
        print(f"Sentiment score: {_sentiment_score(result):+.4f}")

    elif args.status:
        results = sorted(NEWS_DIR.glob("phase1_research_*.json"), reverse=True)
        if results:
            data = json.loads(results[0].read_text())
            print(f"\nLast research: {data['generated_at'][:10]}")
            print(f"Conclusion: {data['phase1_conclusion']}")
            print("\nBy ticker:")
            for t, r in data["results"].items():
                if not t.startswith("_"):
                    print(f"  {t:<8} IC={r['ic']:+.4f}  {r['verdict']}")
        else:
            print("No research results yet. Run: --research")

    elif args.research:
        if args.equity:
            tickers = EQUITY_TICKERS
        elif args.forex:
            tickers = FOREX_PAIRS
        else:
            tickers = EQUITY_TICKERS  # default: equity first
        run_research(tickers, days=args.days)

    else:
        parser.print_help()
