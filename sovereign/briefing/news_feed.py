#!/usr/bin/env python3
"""C4 — News retriever + mapper.

Pulls recent market-moving headlines from NewsAPI (NEWS_API_KEY, already configured),
filters to the categories that move ES/NQ — Fed/rates, inflation prints, geopolitics
(Iran/oil), major tech earnings, AI capex — and tags each with a category + likely-affected
instrument.

Degrades gracefully to an empty feed on any API failure (a scheduled job must never die).
Writes data/briefing/news_feed.json.

Usage:  python3 -m sovereign.briefing.news_feed
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _lib in ("urllib3", "requests"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

OUT = ROOT / "data" / "briefing" / "news_feed.json"

# keyword → (category, instrument) ; instrument NQ = tech/AI-led, ES = broad, BOTH = macro
_RULES = [
    (("federal reserve", "fed ", "fomc", "powell", "rate cut", "rate hike", "interest rate"), "FED_RATES", "BOTH"),
    (("inflation", "cpi", "pce", "ppi", "core inflation"), "INFLATION", "BOTH"),
    (("jobs report", "payroll", "nonfarm", "unemployment", "labor market"), "LABOR", "BOTH"),
    (("iran", "israel", "oil", "crude", "opec", "geopolit", "war", "strike"), "GEOPOLITICS", "BOTH"),
    (("nvidia", "nvda", "amd", "broadcom", "dell", "microsoft", "earnings", "guidance"), "TECH_EARNINGS", "NQ"),
    (("ai capex", "data center", "artificial intelligence", "ai spending", "ai chip", "gpu"), "AI_CAPEX", "NQ"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag(text: str):
    t = text.lower()
    for keys, cat, inst in _RULES:
        if any(k in t for k in keys):
            return cat, inst
    return "OTHER", "BOTH"


def _write(items: list) -> dict:
    payload = {"as_of": _now(), "count": len(items), "items": items}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def fetch(max_items: int = 20) -> dict:
    try:
        from sovereign.oracle.oracle_agent import _load_dotenv
        _load_dotenv()
    except Exception:
        pass
    key = os.environ.get("NEWS_API_KEY", "")
    if not key:
        return _write([])

    import requests
    topics = ['"Federal Reserve"', '"inflation"', '"interest rates"', '"Nvidia"',
              '"AI capex"', '"oil"', '"earnings"', '"Iran"']
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": " OR ".join(topics), "language": "en",
                    "sortBy": "publishedAt", "pageSize": max_items, "apiKey": key},
            timeout=20,
        )
        articles = r.json().get("articles", []) if r.ok else []
    except Exception:
        articles = []

    items = []
    for a in articles:
        title = a.get("title") or ""
        cat, inst = _tag(f"{title} {a.get('description') or ''}")
        # Keep only market-relevant items (drop OTHER to control noise/tokens).
        if cat == "OTHER":
            continue
        items.append({
            "title": title,
            "source": (a.get("source") or {}).get("name"),
            "published_at": a.get("publishedAt"),
            "category": cat,
            "instrument": inst,
            "url": a.get("url"),
        })
    return _write(items[:max_items])


if __name__ == "__main__":
    p = fetch()
    print(f"News feed: {p['count']} tagged market items")
    for it in p["items"][:8]:
        print(f"  [{it['category']}/{it['instrument']}] {it['title'][:90]}")
    print(f"  Saved: {OUT.relative_to(ROOT)}")
