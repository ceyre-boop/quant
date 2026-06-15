#!/usr/bin/env python3
"""
Daily Bias Engine for MES/MNQ.

Synthesizes overnight Globex action, economic calendar, and macro regime
into a structured directional lean before each session.

Output: printed bias statement + appended entry in data/futures/bias_log.jsonl

Usage:
    python3.13 scripts/futures_bias.py
    python3.13 scripts/futures_bias.py --instrument MNQ
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BIAS_LOG  = ROOT / "data" / "futures" / "bias_log.jsonl"
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"

HIGH_IMPACT_KEYWORDS = {
    "fomc", "federal reserve", "interest rate decision",
    "cpi", "consumer price", "pce",
    "nfp", "nonfarm payroll", "jobs report",
    "gdp", "jobless claims",
    "ism manufacturing", "ism services",
}


def _fetch_prices(instrument: str = "ES") -> dict:
    """Pull overnight and prior-day data via yfinance."""
    import yfinance as yf

    ticker_map = {"ES": "ES=F", "NQ": "NQ=F"}
    ticker = ticker_map.get(instrument.upper(), "ES=F")

    now_et = datetime.now(timezone.utc) - timedelta(hours=4)  # rough ET offset
    # Pull last 5 days of daily bars for prior-day reference
    daily = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
    # Pull last 24h of 5-minute bars for overnight analysis
    intra = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True)

    if daily.empty or intra.empty:
        raise RuntimeError(f"No price data returned for {ticker}")

    # Prior session (last complete day)
    prior = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    prior_high = float(prior["High"].item())
    prior_low  = float(prior["Low"].item())
    prior_close = float(prior["Close"].item())

    # Current session / overnight
    current_high = float(intra["High"].max().item())
    current_low  = float(intra["Low"].min().item())
    last_price   = float(intra["Close"].iloc[-1].item())

    # Overnight direction vs prior close
    overnight_change = last_price - prior_close
    overnight_pct    = overnight_change / prior_close * 100

    return {
        "ticker":          ticker,
        "instrument":      instrument.upper(),
        "prior_day_high":  round(prior_high, 2),
        "prior_day_low":   round(prior_low, 2),
        "prior_close":     round(prior_close, 2),
        "overnight_high":  round(current_high, 2),
        "overnight_low":   round(current_low, 2),
        "last_price":      round(last_price, 2),
        "overnight_chg":   round(overnight_change, 2),
        "overnight_pct":   round(overnight_pct, 3),
    }


def _fetch_macro() -> dict:
    """Pull VIX, 10yr yield, DXY for regime context."""
    import yfinance as yf

    tickers = {"^VIX": "vix", "^TNX": "yield_10yr", "DX-Y.NYB": "dxy"}
    out = {}
    for symbol, key in tickers.items():
        try:
            d = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=True)
            if not d.empty:
                prev = float(d["Close"].iloc[-2].item()) if len(d) >= 2 else float(d["Close"].iloc[-1].item())
                curr = float(d["Close"].iloc[-1].item())
                out[key]              = round(curr, 3)
                out[f"{key}_chg"]     = round(curr - prev, 3)
                out[f"{key}_pct_chg"] = round((curr - prev) / prev * 100, 3)
        except Exception:
            out[key] = None
    return out


def _fetch_calendar() -> list[dict]:
    """
    Fetch today's high-impact US economic events.
    Tries econdb API (free, no auth). Falls back to empty list with a warning.
    """
    try:
        import urllib.request
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://www.econdb.com/api/events/?date={today}&country=US&format=json"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        events = []
        for item in data.get("results", []):
            name = (item.get("name") or "").lower()
            impact = item.get("importance", 0)
            if impact >= 2 or any(kw in name for kw in HIGH_IMPACT_KEYWORDS):
                events.append({
                    "name":   item.get("name"),
                    "time":   item.get("time", ""),
                    "impact": impact,
                })
        return events

    except Exception as e:
        print(f"  [calendar] econdb unavailable ({type(e).__name__}), proceeding without calendar data")
        return []


def _score_bias(prices: dict, macro: dict, events: list[dict]) -> dict:
    """
    Derive directional lean and conviction from the inputs.
    Returns: {bias, conviction, reasoning_parts, risk_flags, invalidation}
    """
    bull_points = 0
    bear_points = 0
    reasoning   = []
    risk_flags  = []

    # 1. Overnight direction
    pct = prices["overnight_pct"]
    if pct > 0.20:
        bull_points += 2
        reasoning.append(f"Overnight +{pct:.2f}% — buyers in control")
    elif pct < -0.20:
        bear_points += 2
        reasoning.append(f"Overnight {pct:.2f}% — sellers in control")
    else:
        reasoning.append(f"Overnight flat ({pct:+.2f}%) — no clear overnight bias")

    # 2. Overnight position vs prior-day levels
    last = prices["last_price"]
    pdh  = prices["prior_day_high"]
    pdl  = prices["prior_day_low"]
    if last > pdh:
        bull_points += 1
        reasoning.append(f"Trading ABOVE prior high ({pdh}) — bullish continuation")
    elif last < pdl:
        bear_points += 1
        reasoning.append(f"Trading BELOW prior low ({pdl}) — bearish continuation")
    else:
        reasoning.append(f"Inside prior day range ({pdl}–{pdh})")

    # 3. VIX regime
    vix = macro.get("vix")
    vix_chg = macro.get("vix_chg", 0) or 0
    if vix is not None:
        if vix > 25:
            risk_flags.append(f"VIX {vix:.1f} — high volatility regime, reduce size")
            bear_points += 1
        elif vix > 20:
            risk_flags.append(f"VIX {vix:.1f} — elevated, use conviction 1 max")
        elif vix < 14:
            risk_flags.append(f"VIX {vix:.1f} — complacency zone, gaps can reverse fast")
        else:
            reasoning.append(f"VIX {vix:.1f} — normal regime ({'+' if vix_chg>0 else ''}{vix_chg:.2f})")

    # 4. 10yr yield direction (relevant for NQ especially)
    tnx = macro.get("yield_10yr")
    tnx_chg = macro.get("yield_10yr_chg", 0) or 0
    if tnx is not None:
        if tnx_chg > 0.05:
            bear_points += 1
            reasoning.append(f"10yr yield rising ({tnx:.3f}, +{tnx_chg:.3f}) — headwind for growth")
        elif tnx_chg < -0.05:
            bull_points += 1
            reasoning.append(f"10yr yield falling ({tnx:.3f}, {tnx_chg:.3f}) — tailwind")
        else:
            reasoning.append(f"10yr yield flat ({tnx:.3f})")

    # 5. DXY (dollar strength)
    dxy_chg = macro.get("dxy_pct_chg", 0) or 0
    if abs(dxy_chg) > 0.20:
        if dxy_chg > 0:
            bear_points += 1
            reasoning.append(f"DXY rising +{dxy_chg:.2f}% — dollar strength pressure")
        else:
            bull_points += 1
            reasoning.append(f"DXY falling {dxy_chg:.2f}% — dollar weakness supportive")

    # 6. High-impact calendar events
    if events:
        event_names = ", ".join(e["name"] for e in events)
        risk_flags.append(f"HIGH-IMPACT EVENTS TODAY: {event_names}")
        # Reduce conviction for scheduled volatility
        bull_points = max(0, bull_points - 1)
        bear_points = max(0, bear_points - 1)

    # Derive bias and conviction
    net = bull_points - bear_points
    if net >= 3:
        bias, conviction = "LONG", 3
    elif net == 2:
        bias, conviction = "LONG", 2
    elif net == 1:
        bias, conviction = "LONG", 1
    elif net == -1:
        bias, conviction = "SHORT", 1
    elif net == -2:
        bias, conviction = "SHORT", 2
    elif net <= -3:
        bias, conviction = "SHORT", 3
    else:
        bias, conviction = "NEUTRAL", 1

    # Hard cap: conviction=1 if high-impact events or VIX>20
    if events or (vix and vix > 20):
        conviction = min(conviction, 1)

    # Invalidation level
    if bias == "LONG":
        invalidation = f"Below overnight low {prices['overnight_low']:.2f}"
    elif bias == "SHORT":
        invalidation = f"Above overnight high {prices['overnight_high']:.2f}"
    else:
        invalidation = "Bias unclear — stand down or reduce to minimum size"

    return {
        "bias":         bias,
        "conviction":   conviction,
        "reasoning":    reasoning,
        "risk_flags":   risk_flags,
        "invalidation": invalidation,
        "bull_points":  bull_points,
        "bear_points":  bear_points,
    }


def _print_bias(bias_record: dict) -> None:
    s = bias_record
    conviction_stars = "★" * s["conviction"] + "☆" * (3 - s["conviction"])

    print(f"\n{'═'*60}")
    print(f"  DAILY BIAS — {s['date']}  |  {s['instrument']}")
    print(f"{'═'*60}")
    print(f"  Direction:    {s['bias']}  ({conviction_stars}  {s['conviction']}/3)")
    print(f"  Invalidation: {s['invalidation']}")
    print(f"\n  Key Levels:")
    kl = s["key_levels"]
    print(f"    Prior day high: {kl['prior_day_high']}")
    print(f"    Prior day low:  {kl['prior_day_low']}")
    print(f"    Prior close:    {kl['prior_close']}")
    print(f"    Overnight high: {kl['overnight_high']}")
    print(f"    Overnight low:  {kl['overnight_low']}")
    print(f"    Last price:     {kl['last_price']}")

    print(f"\n  Macro:")
    m = s["macro"]
    for k, v in m.items():
        if "_chg" not in k and v is not None:
            chg_key = f"{k}_pct_chg"
            chg = m.get(chg_key)
            chg_str = f"  ({'+' if chg and chg>0 else ''}{chg:.2f}%)" if chg is not None else ""
            print(f"    {k:15s}: {v:.3f}{chg_str}")

    print(f"\n  Reasoning:")
    for r in s["reasoning"]:
        print(f"    • {r}")

    if s["risk_flags"]:
        print(f"\n  ⚠ Risk Flags:")
        for f in s["risk_flags"]:
            print(f"    ! {f}")

    if s["events"]:
        print(f"\n  Calendar Events:")
        for e in s["events"]:
            print(f"    {e.get('time','?'):6s}  {e['name']}")

    print(f"\n  Conviction guide:")
    print(f"    1 = minimum size, stand down if invalidated immediately")
    print(f"    2 = standard probe size")
    print(f"    3 = multiple factors aligned, press T2 if confirmed")
    print(f"{'═'*60}\n")


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="ES", choices=["ES", "NQ"],
                    help="ES (S&P 500 micro) or NQ (Nasdaq micro)")
    ap.add_argument("--no-log", action="store_true", help="Don't write to bias_log.jsonl")
    args = ap.parse_args()

    print(f"Fetching data for {args.instrument} bias...", end=" ", flush=True)

    try:
        prices = _fetch_prices(args.instrument)
    except RuntimeError as e:
        # No market data (markets closed, or pre-open data not yet posted). Skip
        # cleanly — write no bias, exit 0. The monitor runs NEUTRAL until a later
        # run produces data; a missing bias is not a hard failure and must not
        # surface as a traceback / non-zero exit (which would trip loop_health).
        print(f"\n  ⚠️  {e} — markets likely closed; no bias written this run.")
        return {"instrument": args.instrument.upper(), "bias": "NEUTRAL",
                "skipped": True, "reason": str(e)}
    macro  = _fetch_macro()
    events = _fetch_calendar()
    scored = _score_bias(prices, macro, events)

    print("done.")

    now = datetime.now(timezone.utc)
    bias_record = {
        "date":        now.strftime("%Y-%m-%d"),
        "timestamp":   now.isoformat(),
        "instrument":  args.instrument.upper(),
        "bias":        scored["bias"],
        "conviction":  scored["conviction"],
        "reasoning":   scored["reasoning"],
        "risk_flags":  scored["risk_flags"],
        "invalidation": scored["invalidation"],
        "events":      events,
        "key_levels":  {
            "prior_day_high":  prices["prior_day_high"],
            "prior_day_low":   prices["prior_day_low"],
            "prior_close":     prices["prior_close"],
            "overnight_high":  prices["overnight_high"],
            "overnight_low":   prices["overnight_low"],
            "last_price":      prices["last_price"],
        },
        "macro": macro,
        "scoring": {
            "bull_points": scored["bull_points"],
            "bear_points": scored["bear_points"],
        },
    }

    _print_bias(bias_record)

    if not args.no_log:
        BIAS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(BIAS_LOG, "a") as f:
            f.write(json.dumps(bias_record) + "\n")
        print(f"  Logged to {BIAS_LOG.relative_to(ROOT)}")

    return bias_record


if __name__ == "__main__":
    main()
