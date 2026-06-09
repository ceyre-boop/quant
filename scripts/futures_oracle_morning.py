#!/usr/bin/env python3
"""
Futures morning oracle — Opus-4 session briefing.

Synthesizes overnight Globex action, macro context, and calendar into a
structured, falsifiable pre-session prediction with stated probability.
Logs to data/futures/oracle_mornings.jsonl for calibration tracking.

Does NOT place orders. Does NOT import IBBridge. Pure cognition only.

Usage:
    python3.13 scripts/futures_oracle_morning.py
    python3.13 scripts/futures_oracle_morning.py --instrument MNQ
    python3.13 scripts/futures_oracle_morning.py --no-log   # display only
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

ORACLE_LOG = ROOT / "data" / "futures" / "oracle_mornings.jsonl"
BIAS_LOG   = ROOT / "data" / "futures" / "bias_log.jsonl"
TRADE_LOG  = ROOT / "data" / "futures" / "trade_log.jsonl"

TICKER_MAP = {"MES": "ES=F", "MNQ": "NQ=F"}

MODEL = "claude-opus-4-7"
SONNET_MODEL = "claude-sonnet-4-6"          # intraday killzone synthesis (Option 2)
# Killzone open times in ET — the institutional order-flow windows.
KILLZONES = {"LONDON": "03:00", "NY_AM": "09:30", "NY_PM": "14:00"}

SYSTEM_PROMPT = """\
You are a futures session analyst for a paper-trading learning project on ES/NQ micro contracts.
Your job is to produce one structured, falsifiable pre-session prediction and state your confidence.

Rules — non-negotiable:
1. Every prediction must include an explicit falsifier: the specific price or condition that proves you wrong today.
2. State a probability between 40% and 80%. Never state exactly 50% — that is noise, not analysis.
3. If market conditions are genuinely unreadable (high-impact event imminent, ADR already >150%, contradicting signals with no resolution), output "NO_PREDICTION" as the bias value.
4. Keep reasoning_summary to 3 lines maximum. Specific factors only. No eloquent filler.
5. session_type must be one of: NORMAL, TRENDING, EXTENDED, CHOP, PRE_EVENT.

Your output will be scored for calibration over time. Eloquence is not measured. Accuracy is.
A well-reasoned wrong prediction is better than vague hedging. Commit or say NO_PREDICTION.

Output ONLY valid JSON matching this exact schema — no preamble, no explanation:
{
  "bias": "LONG" | "SHORT" | "NEUTRAL" | "NO_PREDICTION",
  "conviction": 1 | 2 | 3,
  "stated_probability": 0.40-0.80,
  "falsifier": "specific price/condition that invalidates this bias",
  "key_levels": {
    "invalidation": float,
    "t1_target": float,
    "t2_target": float
  },
  "reasoning_summary": "line1\\nline2\\nline3",
  "session_type": "NORMAL" | "TRENDING" | "EXTENDED" | "CHOP" | "PRE_EVENT",
  "adr_pct": float,
  "confidence_note": "one sentence on anything that reduces confidence, or empty string"
}"""


# ── context gathering ────────────────────────────────────────────────────────

def _fetch_overnight(instrument: str) -> dict:
    import yfinance as yf
    ticker = TICKER_MAP[instrument]
    # 5-day daily for ADR baseline
    daily = yf.download(ticker, period="20d", interval="1d", progress=False, auto_adjust=True)
    # 2-day 5m for overnight detail
    intra = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=True)
    if daily.empty or intra.empty:
        return {}

    # ADR: average of last 14 days' high-low range
    if len(daily) >= 14:
        ranges = (daily["High"] - daily["Low"]).iloc[-14:]
        avg_range = float(ranges.mean().item())
    else:
        avg_range = float((daily["High"] - daily["Low"]).mean().item())

    prior = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    prior_high  = float(prior["High"].item())
    prior_low   = float(prior["Low"].item())
    prior_close = float(prior["Close"].item())
    prior_range = prior_high - prior_low
    adr_pct     = round(prior_range / avg_range * 100, 1) if avg_range else 0.0

    curr_high = float(intra["High"].max().item())
    curr_low  = float(intra["Low"].min().item())
    last_px   = float(intra["Close"].iloc[-1].item())
    ov_chg    = round(last_px - prior_close, 2)
    ov_pct    = round(ov_chg / prior_close * 100, 3)
    ov_range  = round(curr_high - curr_low, 2)

    return {
        "instrument":      instrument,
        "ticker":          ticker,
        "prior_high":      round(prior_high, 2),
        "prior_low":       round(prior_low, 2),
        "prior_close":     round(prior_close, 2),
        "overnight_high":  round(curr_high, 2),
        "overnight_low":   round(curr_low, 2),
        "last_price":      round(last_px, 2),
        "overnight_chg":   ov_chg,
        "overnight_pct":   ov_pct,
        "overnight_range": ov_range,
        "avg_daily_range": round(avg_range, 2),
        "adr_pct":         adr_pct,
    }


def _fetch_macro() -> dict:
    import yfinance as yf
    tickers = {"^VIX": "vix", "^TNX": "yield_10yr", "DX-Y.NYB": "dxy"}
    out = {}
    for sym, key in tickers.items():
        try:
            d = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=True)
            if not d.empty:
                prev = float(d["Close"].iloc[-2].item()) if len(d) >= 2 else float(d["Close"].iloc[-1].item())
                curr = float(d["Close"].iloc[-1].item())
                out[key]          = round(curr, 3)
                out[f"{key}_chg"] = round(curr - prev, 3)
                out[f"{key}_pct"] = round((curr - prev) / prev * 100, 3)
        except Exception:
            out[key] = None
    return out


def _fetch_calendar() -> list[dict]:
    try:
        import urllib.request
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://www.econdb.com/api/events/?date={today}&country=US&format=json"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        HIGH = {"fomc","federal reserve","interest rate","cpi","consumer price","pce",
                "nfp","nonfarm","jobs report","gdp","jobless","ism"}
        events = []
        for item in data.get("results", []):
            name = (item.get("name") or "").lower()
            imp  = item.get("importance", 0)
            if imp >= 2 or any(kw in name for kw in HIGH):
                events.append({"name": item.get("name"), "time": item.get("time",""), "impact": imp})
        return events
    except Exception:
        return []


def _load_recent_sessions(instrument: str, n: int = 10) -> list[dict]:
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("instrument") == instrument and t.get("r_realized") is not None:
                    trades.append(t)
            except Exception:
                pass
    return trades[-n:]


def _load_prior_oracle_sessions(instrument: str, n: int = 5) -> list[dict]:
    if not ORACLE_LOG.exists():
        return []
    sessions = []
    with open(ORACLE_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("instrument") == instrument:
                    sessions.append(rec)
            except Exception:
                pass
    return sessions[-n:]


# ── prompt assembly ──────────────────────────────────────────────────────────

def _build_user_prompt(instrument: str, overnight: dict, macro: dict,
                       events: list[dict], trades: list[dict],
                       prior_calls: list[dict]) -> str:
    lines = []
    lines.append(f"=== PRE-SESSION CONTEXT: {instrument} ===")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    lines.append("")

    lines.append("OVERNIGHT PRICE ACTION:")
    if overnight:
        lines.append(f"  Prior close:    {overnight.get('prior_close')}")
        lines.append(f"  Prior high/low: {overnight.get('prior_high')} / {overnight.get('prior_low')}")
        lines.append(f"  Overnight high: {overnight.get('overnight_high')}")
        lines.append(f"  Overnight low:  {overnight.get('overnight_low')}")
        lines.append(f"  Last price:     {overnight.get('last_price')}")
        lines.append(f"  Overnight move: {overnight.get('overnight_chg'):+.2f} ({overnight.get('overnight_pct'):+.3f}%)")
        lines.append(f"  Overnight range: {overnight.get('overnight_range')} pts")
        lines.append(f"  ADR%: {overnight.get('adr_pct')}% of {overnight.get('avg_daily_range'):.2f} avg range")
        above_prior = overnight.get('last_price', 0) > overnight.get('prior_high', 0)
        below_prior = overnight.get('last_price', 0) < overnight.get('prior_low', 0)
        inside = not above_prior and not below_prior
        if above_prior:
            lines.append(f"  Position vs prior day: ABOVE prior high — bullish continuation signal")
        elif below_prior:
            lines.append(f"  Position vs prior day: BELOW prior low — bearish continuation signal")
        else:
            lines.append(f"  Position vs prior day: INSIDE prior range — no clear overnight continuation")
    else:
        lines.append("  [price data unavailable]")

    lines.append("")
    lines.append("MACRO CONTEXT:")
    vix = macro.get("vix")
    if vix:
        regime = "HIGH VOLATILITY" if vix > 25 else ("ELEVATED" if vix > 20 else ("COMPLACENCY ZONE" if vix < 14 else "NORMAL"))
        lines.append(f"  VIX: {vix:.1f} ({regime}) | chg: {macro.get('vix_chg',0):+.3f}")
    tnx = macro.get("yield_10yr")
    if tnx:
        lines.append(f"  10yr yield: {tnx:.3f}% | chg: {macro.get('yield_10yr_chg',0):+.3f}")
    dxy = macro.get("dxy")
    if dxy:
        lines.append(f"  DXY: {dxy:.3f} | chg: {macro.get('dxy_chg',0):+.3f} ({macro.get('dxy_pct',0):+.2f}%)")

    if events:
        lines.append("")
        lines.append("HIGH-IMPACT CALENDAR EVENTS TODAY:")
        for e in events:
            lines.append(f"  {e.get('time','?'):6s} — {e.get('name','?')}")

    if trades:
        lines.append("")
        lines.append(f"RECENT SESSION OUTCOMES ({len(trades)} closed trades):")
        wins = sum(1 for t in trades if (t.get("r_realized") or 0) > 0)
        total_r = sum((t.get("r_realized") or 0) for t in trades)
        aligned = [t for t in trades if t.get("bias_aligned")]
        aligned_wins = sum(1 for t in aligned if (t.get("r_realized") or 0) > 0)
        lines.append(f"  Win rate: {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)")
        lines.append(f"  Total R: {total_r:+.2f} | Avg R/trade: {total_r/len(trades):+.3f}")
        if aligned:
            lines.append(f"  Bias-aligned win rate: {aligned_wins}/{len(aligned)} ({aligned_wins/len(aligned)*100:.0f}%)")
    else:
        lines.append("")
        lines.append("RECENT SESSION OUTCOMES: No trades logged yet (first sessions)")

    if prior_calls:
        lines.append("")
        lines.append("PRIOR ORACLE CALLS (for context, not to anchor you):")
        for c in prior_calls[-3:]:
            scored = c.get("outcome_scored")
            score_str = f" → {'HIT' if scored == 1 else ('MISS' if scored == 0 else 'UNSCORED')}" if scored is not None else " → unscored"
            lines.append(f"  {c.get('date','')} | {c.get('bias','')} {c.get('stated_probability',0):.0%}{score_str}")

    lines.append("")
    lines.append("TASK: Produce your pre-session call. Output JSON only. No preamble.")

    return "\n".join(lines)


# ── Opus call ────────────────────────────────────────────────────────────────

def _call_opus(user_prompt: str) -> dict:
    """Daily Opus synthesis (back-compat wrapper)."""
    return _call_model(user_prompt, MODEL)


def _call_model(user_prompt: str, model: str) -> dict:
    from dotenv import load_dotenv
    import os
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ── display ──────────────────────────────────────────────────────────────────

def _print_oracle(result: dict, instrument: str, overnight: dict) -> None:
    bias = result.get("bias", "?")
    prob = result.get("stated_probability", 0)
    conv = result.get("conviction", 1)
    stars = "★" * conv + "☆" * (3 - conv)
    sess  = result.get("session_type", "?")
    adr   = result.get("adr_pct", 0)
    date  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    color = "\033[92m" if bias == "LONG" else ("\033[91m" if bias == "SHORT" else "\033[93m")
    rs    = "\033[0m"
    bd    = "\033[1m"

    print(f"\n{'═'*62}")
    print(f"  {bd}ORACLE MORNING CALL — {date} | {instrument}{rs}")
    print(f"{'═'*62}")
    print(f"  {bd}Direction:{rs}    {color}{bias}{rs}  ({stars}  {conv}/3)")
    print(f"  {bd}Probability:{rs}  {prob:.0%}")
    print(f"  {bd}Session type:{rs} {sess}  |  ADR: {adr:.0f}%")
    print(f"\n  {bd}Falsifier:{rs}")
    print(f"    {result.get('falsifier', '?')}")
    kl = result.get("key_levels", {})
    if kl:
        print(f"\n  {bd}Key levels:{rs}")
        print(f"    Invalidation: {kl.get('invalidation','?')}")
        print(f"    T1 target:    {kl.get('t1_target','?')}")
        print(f"    T2 target:    {kl.get('t2_target','?')}")
    if overnight:
        print(f"\n  {bd}Overnight context:{rs}")
        print(f"    Last price:   {overnight.get('last_price')}  "
              f"(chg: {overnight.get('overnight_chg'):+.2f} / {overnight.get('overnight_pct'):+.3f}%)")
    rsm = result.get("reasoning_summary", "")
    if rsm:
        print(f"\n  {bd}Reasoning:{rs}")
        for line in rsm.split("\n"):
            print(f"    {line}")
    note = result.get("confidence_note", "")
    if note:
        print(f"\n  {bd}⚠ Confidence note:{rs} {note}")
    print(f"\n  {bd}Calibration:{rs} This call will be scored after the session.")
    print(f"  Run: python3.13 scripts/futures_calibration.py --oracle --score-today")
    print(f"{'═'*62}\n")


# ── structural context (Increment 4): prior-day volume profile + CVD trend ────

def _prior_structure(instrument: str, overnight: dict) -> dict:
    """Prior-day POC/VAH/VAL + CVD trend — the structure a 1% scalper reads pre-open.
    Best-effort off yfinance 5m; marked untrusted because futures volume there is weak.
    (Today's ORB level can't be known pre-open — that alignment is logged live in the replay.)"""
    try:
        from sovereign.futures import volume_profile as vp
        from sovereign.futures import cvd as cvd_mod
        from sovereign.futures.config import contract_spec
        import yfinance as yf
        ticker = TICKER_MAP[instrument]
        h = yf.download(ticker, period="3d", interval="5m", progress=False, auto_adjust=True)
        if h is None or h.empty:
            return {"available": False}
        import pandas as pd
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        days = sorted({d.strftime("%Y-%m-%d") for d in h.index})
        if len(days) < 2:
            return {"available": False}
        prior = h[h.index.strftime("%Y-%m-%d") == days[-2]]
        prof = vp.compute_profile(prior)
        cstate = cvd_mod.cvd_state(prior)
        tick = contract_spec(instrument)["tick"]
        last = overnight.get("last_price")
        dist = None
        if prof and last:
            dist = {k: round((last - prof[k]) / tick, 1) for k in ("poc", "vah", "val")}
        return {
            "available": prof is not None,
            "prior_poc": prof["poc"] if prof else None,
            "prior_vah": prof["vah"] if prof else None,
            "prior_val": prof["val"] if prof else None,
            "dist_to_levels_ticks": dist,
            "prior_cvd_trend": (None if cstate is None else ("up" if cstate["slope"] > 0 else "down")),
            "untrusted": True,   # yfinance volume — confirm on IB
        }
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> dict:
    ap = argparse.ArgumentParser(description="Futures oracle synthesis (daily Opus / killzone Sonnet)")
    ap.add_argument("--instrument", default="MES", choices=["MES", "MNQ"])
    ap.add_argument("--no-log", action="store_true", help="Display only, don't write to jsonl")
    ap.add_argument("--killzone", choices=list(KILLZONES), default=None,
                    help="Intraday killzone synthesis (Sonnet) instead of the daily Opus call")
    ap.add_argument("--model", default=None, help="Override the model id")
    args = ap.parse_args()

    is_kz = args.killzone is not None
    synthesis_type = "killzone_sonnet" if is_kz else "daily_opus"
    model = args.model or (SONNET_MODEL if is_kz else MODEL)
    label = f"{args.killzone} killzone (Sonnet)" if is_kz else "daily (Opus)"

    print(f"Gathering context for {args.instrument} {label} call...", end=" ", flush=True)

    overnight   = _fetch_overnight(args.instrument)
    macro       = _fetch_macro()
    events      = _fetch_calendar()
    trades      = _load_recent_sessions(args.instrument)
    prior_calls = _load_prior_oracle_sessions(args.instrument)

    print(f"done. Calling {model}...", end=" ", flush=True)

    user_prompt = _build_user_prompt(args.instrument, overnight, macro, events, trades, prior_calls)
    if is_kz:
        user_prompt = (f"INTRADAY KILLZONE UPDATE ({args.killzone}, {KILLZONES[args.killzone]} ET). "
                       f"This is a fast big-move read at a live order-flow window, not the full daily "
                       f"plan — focus on what has changed and the immediate directional read.\n\n"
                       + user_prompt)

    try:
        result = _call_model(user_prompt, model)
    except Exception as e:
        print(f"\n  [error] {model} call failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print("done.\n")

    _print_oracle(result, args.instrument, overnight)

    if not args.no_log:
        record = {
            "date":              datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "synthesis_type":    synthesis_type,    # daily_opus | killzone_sonnet
            "killzone":          args.killzone,     # LONDON | NY_AM | NY_PM | None
            "instrument":        args.instrument,
            "bias":              result.get("bias"),
            "conviction":        result.get("conviction"),
            "stated_probability": result.get("stated_probability"),
            "falsifier":         result.get("falsifier"),
            "key_levels":        result.get("key_levels", {}),
            "reasoning_summary": result.get("reasoning_summary"),
            "session_type":      result.get("session_type"),
            "adr_pct":           result.get("adr_pct"),
            "confidence_note":   result.get("confidence_note", ""),
            "outcome_scored":    None,   # filled by futures_calibration.py --score-today
            "outcome_hit_t1":    None,
            "brier_contribution": None,
            "model":             model,
            "context_snapshot": {
                "last_price":    overnight.get("last_price"),
                "overnight_pct": overnight.get("overnight_pct"),
                "vix":           macro.get("vix"),
                "yield_10yr":    macro.get("yield_10yr"),
                "events_count":  len(events),
            },
            "structure": _prior_structure(args.instrument, overnight),
        }
        ORACLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ORACLE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Logged to {ORACLE_LOG.relative_to(ROOT)}")

    return result


if __name__ == "__main__":
    main()
