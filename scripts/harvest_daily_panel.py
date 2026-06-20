#!/usr/bin/env python3
"""
Daily multi-source research panel — HARVEST ONLY.

Pings/assembles every market-relevant data source each day and records its domain as a unified
daily panel. This is RAW RECORDED DATA, not analysis: relationships between variables are NOT
inferred here. Hypothesis testing about how these variables relate (independent/dependent given
triggers/trends) goes through the existing research_factory — pre-registration + 10k permutation
+ both-sides + Benjamini-Hochberg. The panel FEEDS the factory; it never bypasses its discipline.
(TRADING_PHILOSOPHY Tenet 4: "collect the data first." Tenet 1: features earn their place by OOS
expectancy, not by appearing in a correlation scan.)

Sources (market-relevant only; OPENWEATHER/NASDAQ_DATA_LINK/TIINGO deliberately excluded):
  macro_fred       READ data/macro/fred_economic_latest.json   (economy: GDP/CPI/labor/sentiment)
  markets          READ data/macro/macro_snapshot.json         (VIX/yields/curve/credit)
  sentiment_reddit READ data/cache/reddit_sentiment.json
  news             READ data/briefing/news_feed.json           (count + by-category)
  fx_macro         READ data/cache/macro/{country}_macro.json  (rate/CPI/GDP/real-rate + diffs)
  equities         ACTIVE yfinance pull, SOVEREIGN_UNIVERSE     (daily close + 1d return)
  vrp_proxy        compute VIX − SPY 20d realized vol           (full options VRP = separate track)
  positioning      READ data/cache/cb_decisions.json           (latest CB decision meta)

Writes:
  data/research/panel/YYYY-MM-DD.json   full structured daily snapshot (per-source ok/null status)
  data/research/panel/panel.parquet     append long-form tidy rows (date, source, domain, metric,
                                         value, unit) — the regression-ready research panel

Each source is isolated (try/except): a dead source records null and never blocks the others.
Runnable standalone:  python3 scripts/harvest_daily_panel.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "data" / "research" / "panel"
PANEL_PARQUET = PANEL_DIR / "panel.parquet"


def _load_env() -> None:
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _num(x):
    """Coerce to float or None (drops non-numeric so the parquet stays clean)."""
    try:
        if x is None or isinstance(x, bool):
            return None
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ── per-domain collectors: each returns (structured_dict, [long_rows]) ──
def collect_macro_fred() -> tuple[dict, list]:
    d = _read(ROOT / "data" / "macro" / "fred_economic_latest.json") or {}
    s = d.get("summary", {})
    rows = [("macro_fred", "macro", k, _num(v), "") for k, v in s.items() if _num(v) is not None]
    return {"status": "ok" if s else "null", "summary": s, "snapshot_date": d.get("date")}, rows


def collect_markets() -> tuple[dict, list]:
    d = _read(ROOT / "data" / "macro" / "macro_snapshot.json") or {}
    s = d.get("summary", {})
    rows = [("markets", "markets", k, _num(v), "") for k, v in s.items() if _num(v) is not None]
    return {"status": "ok" if s else "null", "summary": s, "fetched_at": d.get("fetched_at")}, rows


def collect_sentiment_reddit() -> tuple[dict, list]:
    d = _read(ROOT / "data" / "cache" / "reddit_sentiment.json") or {}
    metrics = {
        "posts_scanned": d.get("posts_scanned", 0),
        "n_equity_signals": len(d.get("equity", {}) or {}),
        "n_forex_signals": len(d.get("forex", {}) or {}),
    }
    rows = [("sentiment_reddit", "sentiment", k, _num(v), "count") for k, v in metrics.items()]
    return {"status": "ok" if d else "null", "metrics": metrics, "last_updated": d.get("last_updated")}, rows


def collect_news() -> tuple[dict, list]:
    d = _read(ROOT / "data" / "briefing" / "news_feed.json") or {}
    items = d.get("items", []) or []
    by_cat: dict = {}
    for it in items:
        c = (it.get("category") or "UNCATEGORIZED").upper()
        by_cat[c] = by_cat.get(c, 0) + 1
    rows = [("news", "sentiment", "count", _num(d.get("count", len(items))), "count")]
    rows += [("news", "sentiment", f"cat_{c}", _num(n), "count") for c, n in by_cat.items()]
    return {"status": "ok" if d else "null", "count": d.get("count"), "by_category": by_cat,
            "as_of": d.get("as_of")}, rows


def collect_fx_macro() -> tuple[dict, list]:
    countries, per = ["US", "EU", "JP", "UK", "AU", "NZ"], {}
    rows = []
    for c in countries:
        d = _read(ROOT / "data" / "cache" / "macro" / f"{c}_macro.json")
        if not d:
            continue
        per[c] = {k: d.get(k) for k in ("rate", "cpi_yoy", "gdp_growth", "real_rate")}
        for k in ("rate", "cpi_yoy", "gdp_growth", "real_rate"):
            if _num(d.get(k)) is not None:
                rows.append(("fx_macro", "fx", f"{c}_{k}", _num(d.get(k)), "%"))
    # key real-rate differentials vs USD (the carry driver)
    us = per.get("US", {}).get("real_rate")
    if _num(us) is not None:
        for c in ("EU", "JP", "UK", "AU", "NZ"):
            o = per.get(c, {}).get("real_rate")
            if _num(o) is not None:
                rows.append(("fx_macro", "fx", f"realrate_diff_US_{c}", round(_num(us) - _num(o), 4), "%"))
    return {"status": "ok" if per else "null", "countries": per}, rows


def collect_equities() -> tuple[dict, list]:
    try:
        import yfinance as yf
        from sovereign.data.feeds.alpaca_feed import SOVEREIGN_UNIVERSE as UNIV
    except Exception:
        UNIV = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META",
                "GOOGL", "AMD", "TLT", "GLD", "HYG", "IWM", "XLF"]
    import yfinance as yf
    px = yf.download(UNIV, period="6d", interval="1d", auto_adjust=True, progress=False)["Close"]
    per, rows = {}, []
    for t in UNIV:
        try:
            ser = px[t].dropna()
            if len(ser) < 2:
                continue
            close, prev = float(ser.iloc[-1]), float(ser.iloc[-2])
            ret = round((close / prev - 1) * 100, 4)
            per[t] = {"close": round(close, 4), "ret_1d_pct": ret}
            rows.append(("equities", "equities", f"{t}_close", round(close, 4), "$"))
            rows.append(("equities", "equities", f"{t}_ret_1d_pct", ret, "%"))
        except Exception:
            continue
    return {"status": "ok" if per else "null", "tickers": per}, rows


def collect_vrp_proxy(markets: dict, fred: dict) -> tuple[dict, list]:
    """VIX − SPY 20d realized vol (annualized). A cheap IV-RV gap proxy; the real options VRP
    lives in the separate ThetaData track (VRP-001-OPTIONS)."""
    try:
        import yfinance as yf
        import numpy as np
        spy = yf.Ticker("SPY").history(period="40d", interval="1d", auto_adjust=True)["Close"].dropna()
        r = np.diff(np.log(spy.to_numpy(float)))[-20:]
        realized = float(np.std(r, ddof=1) * math.sqrt(252) * 100)
        vix = (markets.get("summary", {}).get("vix")
               or fred.get("summary", {}).get("vix"))
        vix = _num(vix)
        if vix is None:
            return {"status": "null"}, []
        gap = round(vix - realized, 3)
        m = {"vix": vix, "spy_realized_20d": round(realized, 3), "iv_rv_gap": gap}
        rows = [("vrp_proxy", "volatility", k, _num(v), "vol_pts") for k, v in m.items()]
        return {"status": "ok", "metrics": m}, rows
    except Exception:
        return {"status": "null"}, []


def collect_positioning() -> tuple[dict, list]:
    d = _read(ROOT / "data" / "cache" / "cb_decisions.json")
    if not isinstance(d, list) or not d:
        return {"status": "null", "note": "cb_decisions unavailable (CFTC NQ feed currently 404)"}, []
    latest = d[-1]
    return {"status": "ok", "n_records": len(d), "latest": latest}, []   # reference data, no daily numeric


def _isolate(name, fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {"status": "null", "error": type(e).__name__}, []


def harvest() -> dict:
    _load_env()
    today = date.today().isoformat()
    sources, all_rows = {}, []

    for name, fn, args in [
        ("macro_fred", collect_macro_fred, ()),
        ("markets", collect_markets, ()),
        ("sentiment_reddit", collect_sentiment_reddit, ()),
        ("news", collect_news, ()),
        ("fx_macro", collect_fx_macro, ()),
        ("equities", collect_equities, ()),
        ("positioning", collect_positioning, ()),
    ]:
        struct, rows = _isolate(name, fn, *args)
        sources[name] = struct
        all_rows += rows

    # vrp_proxy depends on markets + fred already collected
    struct, rows = _isolate("vrp_proxy", collect_vrp_proxy, sources["markets"], sources["macro_fred"])
    sources["vrp_proxy"] = struct
    all_rows += rows

    ok = [k for k, v in sources.items() if v.get("status") == "ok"]
    snapshot = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "sources_ok": ok,
        "sources_null": [k for k in sources if k not in ok],
        "provenance": {
            "kind": "research_panel_harvest",
            "note": ("Raw recorded variables across data domains. Relationships are NOT inferred "
                     "here — hypothesis testing goes through research_factory (pre-registration + "
                     "10k permutation + both-sides + Benjamini-Hochberg). Do NOT treat panel "
                     "columns as findings or trading inputs."),
        },
    }
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    (PANEL_DIR / f"{today}.json").write_text(json.dumps(snapshot, indent=2, default=str))
    _append_parquet(today, all_rows)
    return snapshot


def _append_parquet(today: str, rows: list) -> None:
    """Append (date, source, domain, metric, value, unit) tidy rows; idempotent per date."""
    if not rows:
        return
    try:
        import pandas as pd
    except Exception:
        return
    new = pd.DataFrame(
        [{"date": today, "source": s, "domain": d, "metric": m, "value": v, "unit": u}
         for (s, d, m, v, u) in rows if v is not None]
    )
    if PANEL_PARQUET.exists():
        try:
            old = pd.read_parquet(PANEL_PARQUET)
            old = old[old["date"] != today]      # idempotent: drop today, re-append
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    new.to_parquet(PANEL_PARQUET, index=False)


if __name__ == "__main__":
    s = harvest()
    print(f"Panel {s['date']}: {len(s['sources_ok'])}/{len(s['sources'])} sources OK "
          f"-> {s['sources_ok']}")
    if s["sources_null"]:
        print(f"  null: {s['sources_null']}")
    try:
        import pandas as pd
        df = pd.read_parquet(PANEL_PARQUET)
        print(f"  panel.parquet: {len(df)} rows total, {df['date'].nunique()} day(s), "
              f"{df['metric'].nunique()} distinct metrics")
    except Exception:
        pass
