#!/usr/bin/env python3
"""ALTA — ES/NQ daily session brief. Runs 08:45 ET via launchd (com.alta.esnq.brief).

Cognition-only: runs even under a soft freeze (like Oracle briefings — it proposes,
it never trades). The paper runner recomputes the bias at 09:25 with the
pre-registered cutoff; this 08:45 brief is advisory.

Data: this morning's Globex slice from Databento (fail loud — never silently
degrade to synthetic volume), prior closes for ^VIX/^N225/^GDAXI via yfinance,
today's events from the static calendar + CalendarFetcher when available.

Output: data/es_nq/daily_brief_YYYY-MM-DD.json + console + session_log (mode=BRIEF).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.es_nq import data_store                              # noqa: E402
from sovereign.es_nq.config import es_nq_params                     # noqa: E402
from sovereign.es_nq.daily_bias_engine import (                     # noqa: E402
    calendar_score, compute_bias, hurst_score, international_score,
    overnight_score, vix_score,
)
from sovereign.es_nq.session_logger import log_brief, read_sessions  # noqa: E402

ET = ZoneInfo("America/New_York")
CAL_PATH = ROOT / "data" / "es_nq" / "econ_calendar_2018_2026.json"


def live_calendar_tone(date: str) -> tuple[float | None, list[str]]:
    """Today's pre-09:30 event tone from CalendarFetcher, if reachable.
    Returns (tone or None, event names). Tone None → A1 risk-flag path."""
    try:
        from data.calendar_fetcher import CalendarFetcher
        events = CalendarFetcher().fetch_events(country="united states", days_ahead=0)
    except Exception as e:
        print(f"calendar fetcher unavailable ({e}) — using static calendar risk flag")
        return None, []
    names = [e.get("Event", "?") for e in events
             if e.get("Impact", "").lower() in ("high", "3")]
    return None, names    # surprise tone needs actual-vs-forecast; risk-flag for now


def main() -> None:
    p = es_nq_params()
    now_et = datetime.now(ET)
    date = now_et.strftime("%Y-%m-%d")

    # Morning Globex slice: prior day 18:00 ET → now. Databento, fail loud.
    start = (now_et - timedelta(days=1)).replace(hour=18, minute=0, second=0)
    df = data_store.pull_globex_history(
        start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        now_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        chunk_days=2)
    overnight_px = float(df["Close"].iloc[-1])

    daily = data_store.load_daily()
    if date in daily.index:
        raise SystemExit(f"FATAL: daily table already has {date} — clock/cache confusion")
    prior = daily.iloc[-1]
    roll_day = str(df["symbol"].iloc[-1]) != str(prior["symbol"])
    overnight_ret = overnight_px / float(prior["rth_close"]) - 1.0

    aux = data_store.load_aux_daily()
    cal = json.loads(CAL_PATH.read_text())
    static_events = (cal.get(date) or {}).get("events", [])
    live_tone, live_events = live_calendar_tone(date)

    comp = {
        "overnight": overnight_score(overnight_ret, roll_day, p),
        "vix": vix_score(aux["vix"], date, p),
        "hurst": hurst_score(daily["rth_close"], date, p),
        "international": international_score(aux["nikkei"], aux["dax"], date),
    }
    cal_s, event_day = calendar_score(date, cal, live_tone)
    comp["calendar"] = cal_s
    event_day = event_day or bool(live_events)
    bias = compute_bias(date, comp, event_day, roll_day,
                        calendar_active=live_tone is not None, params=p)

    levels = {
        "pdh": float(prior["rth_high"]), "pdl": float(prior["rth_low"]),
        "onh": float(df["High"].max()), "onl": float(df["Low"].min()),
        "last": overnight_px,
    }
    prior_recap = _prior_session_recap()
    conf_band = ("HIGH" if bias.confidence >= 0.7 else
                 "MEDIUM" if bias.confidence >= 0.4 else "LOW")
    arrow = {"UP": "↑", "DOWN": "↓", "NEUTRAL": "─"}[bias.direction]
    plan = _plan_text(bias, levels, roll_day, static_events + live_events)

    brief = {
        "date": date, "generated_at": datetime.now(timezone.utc).isoformat(),
        "bias": {"direction": bias.direction, "confidence": bias.confidence,
                 "raw_score": bias.raw_score, "components": bias.components,
                 "event_day": bias.event_day, "roll_day": roll_day,
                 "reasoning": bias.reasoning},
        "levels": levels, "events": static_events + live_events,
        "overnight_ret": overnight_ret, "plan": plan, "prior_session": prior_recap,
        "note": "advisory — paper runner recomputes bias at 09:25 ET (pre-registered cutoff)",
    }
    out = data_store.DATA_DIR / f"daily_brief_{date}.json"
    out.write_text(json.dumps(brief, indent=1, default=str))
    log_brief(date, brief["bias"], levels, plan, prior_recap)

    print("═" * 47)
    print("ALTA — DAILY SESSION BRIEF")
    print(f"{date} {now_et.strftime('%A')}")
    print("═" * 47)
    print(f"BIAS: {bias.direction} {arrow}")
    print(f"CONFIDENCE: {bias.confidence:.2f} ({conf_band})")
    print("\nINPUTS:")
    print(f"  Overnight Globex:   {overnight_ret:+.2%} "
          f"({'ROLL DAY — zeroed' if roll_day else f'score {comp['overnight']:+.2f}'})")
    print(f"  Calendar:           {', '.join(static_events + live_events) or 'clear'}"
          f"{' → confidence ×0.75' if event_day and live_tone is None else ''}")
    print(f"  VIX regime:         score {comp['vix']:+.2f}")
    print(f"  Momentum regime:    score {comp['hurst']:+.2f}")
    print(f"  Intl tone:          score {comp['international']:+.2f}")
    print(f"\nLEVELS: PDH {levels['pdh']:.2f} | PDL {levels['pdl']:.2f} | "
          f"ONH {levels['onh']:.2f} | ONL {levels['onl']:.2f} | last {levels['last']:.2f}")
    print(f"\nTODAY'S PLAN:\n  {plan}")
    if prior_recap:
        print(f"\nPRIOR SESSION: bias {prior_recap.get('bias_direction')} | "
              f"correct: {prior_recap.get('bias_was_correct')} | "
              f"R: {prior_recap.get('session_r_total')}")
    print("═" * 47)
    print(f"Wrote {out}")


def _plan_text(bias, levels, roll_day, events) -> str:
    if roll_day:
        return "ROLL DAY — no structure trading (spliced continuous series). Observe only."
    if bias.direction == "NEUTRAL":
        return ("Inputs conflict — confidence below 0.40. NO TRADE. The skip is a trade. "
                "Re-evaluate tomorrow.")
    if bias.direction == "UP":
        watch = f"a sweep below PDL {levels['pdl']:.2f} or ONL {levels['onl']:.2f}"
        invalid = "a sustained break below the swept level after reclaim"
    else:
        watch = f"a sweep above PDH {levels['pdh']:.2f} or ONH {levels['onh']:.2f}"
        invalid = "a sustained break above the swept level after reclaim"
    ev = f" Event risk today ({', '.join(events)}): expect the sweep around the release." \
        if events else ""
    return (f"Bias {bias.direction}. Wait for {watch}, reclaim within 3 bars, return to "
            f"VWAP, then one {'up' if bias.direction == 'UP' else 'down'} bar on >1.2x "
            f"volume before 12:00 ET. Probe 0.5% first. Invalidation: {invalid}.{ev}")


def _prior_session_recap() -> dict | None:
    sessions = [s for s in read_sessions(mode="PAPER")]
    if not sessions:
        return None
    s = sessions[-1]
    return {"date": s.get("session_date"),
            "bias_direction": (s.get("bias") or {}).get("direction"),
            "bias_was_correct": s.get("bias_was_correct"),
            "session_r_total": s.get("session_r_total")}


if __name__ == "__main__":
    main()
