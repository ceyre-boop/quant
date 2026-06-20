#!/usr/bin/env python3
"""Morning market briefing ENGINE — orchestrator (Oracle's daily journal).
===========================================================================

Runs the full pipeline each morning before US cash open:
  scorecard.score_yesterday  →  market_data  →  lead_lag  →  volume_profile  →
  news_feed  →  event_calendar  →  synthesize (Opus 4.8, or deterministic fallback)

Writes the briefing to data/oracle/market_briefings/{date}.json + latest.json (the store
Oracle's reflect_cycle / session_open already read), and a heartbeat for loop_health.

PROVENANCE DISCIPLINE: the briefing is Oracle's ANALYTICAL JOURNAL, flagged
`provenance.verified=false`. It is qualitative regime context + a self-scored directional
call — INTELLIGENCE, not a validated edge. Its numbers must never be ingested as inputs the
system computes trades on. ES/NQ are CME futures (OANDA is forex-only) — no execution path.

Deterministic-capable: every step degrades gracefully, and if the Opus synthesis is
unavailable (missing key / API error) it falls back to a deterministic narrative so a
scheduled run can never die.

Usage:
    python3 scripts/morning_market_briefing.py
    python3 scripts/morning_market_briefing.py --narrative-file data/oracle/market_briefings/_seed.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sovereign.briefing import market_data, lead_lag, volume_profile, news_feed, event_calendar, scorecard, synthesize

BRIEF_DIR = ROOT / "data" / "oracle" / "market_briefings"
MACRO_LATEST = ROOT / "data" / "macro" / "fred_economic_latest.json"
FOREX_PROX = ROOT / "data" / "agent" / "forex_proximity.json"
BIG_MOVE_PATH = ROOT / "data" / "agent" / "big_move.json"
HEARTBEAT = ROOT / "logs" / ".heartbeat_morning_briefing"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)} if isinstance(default, dict) else default


def _deterministic_narrative(ll: dict, vp: dict, cal: dict, prox: dict, score: str) -> str:
    regime = ll.get("regime", "UNKNOWN")
    nq_vp = (vp.get("instruments", {}).get("NQ=F", {}) or {})
    poc = nq_vp.get("poc")
    events = ", ".join(f"{e['event']} ({e['note']})" for e in cal.get("events", [])[:3]) or "none in window"
    return (
        f"Session open {date.today().isoformat()}. NQ/ES regime: {regime} — {ll.get('read','')} "
        f"NQ {ll.get('nq_last','?')} / ES {ll.get('es_last','?')}, corr {ll.get('contemporaneous_corr','?')}, "
        f"lead-lag leader {ll.get('lead_lag',{}).get('leader','?')}. "
        f"Volume profile NQ POC ~{poc}. "
        f"Forex: {prox.get('verdict','n/a')}. "
        f"Event watch: {events}. "
        f"{score} "
        f"Regime read is an INPUT for sizing/trust, not a trade signal (deterministic fallback — "
        f"Opus synthesis unavailable this run)."
    )


def _big_move_headline() -> dict:
    """
    The 'answer of the day': the highest-conviction Big-Move-of-the-Day call from
    the pulse-written estimate. Deterministic — reads data/agent/big_move.json, no
    LLM call. DISPLAY-ONLY: this is a hypothesis the validation gate hasn't cleared,
    so it is flagged unverified and never ingested as a trading input.
    """
    bm = _read(BIG_MOVE_PATH)
    ests = bm.get("estimates") if isinstance(bm, dict) else None
    if not ests:
        return {}
    # Rank non-NEUTRAL calls by p_big * confidence.
    ranked = sorted(
        [(p, e) for p, e in ests.items() if e.get("direction") in ("LONG", "SHORT")],
        key=lambda x: (x[1].get("p_big", 0) * x[1].get("confidence", 0)),
        reverse=True,
    )
    if not ranked:
        return {"headline": "No directional big-move call today — drivers are mixed/neutral.",
                "validation_status": bm.get("validation_status", "UNVALIDATED — display only")}
    pair, e = ranked[0]
    pct = round(e.get("p_big", 0) * 100)
    return {
        "headline": (f"Today's likeliest institutional move: {e['direction']} {pair} "
                     f"(P {pct}%, ~{e.get('expected_magnitude_pct', 0):.2f}% range, "
                     f"conf {round(e.get('confidence', 0) * 100)}%)."),
        "pair": pair,
        "direction": e["direction"],
        "p_big": e.get("p_big"),
        "confidence": e.get("confidence"),
        "expected_magnitude_pct": e.get("expected_magnitude_pct"),
        "session": e.get("session"),
        "drivers": e.get("drivers", [])[:3],
        "validation_status": bm.get("validation_status", "UNVALIDATED — display only"),
    }


def build(narrative_file: str | None = None) -> dict:
    today = date.today().isoformat()

    # C8 — score any past unscored briefings against reality FIRST.
    _safe(lambda: scorecard.score_yesterday(), 0)
    score_summary = _safe(lambda: scorecard.summary_line(), "scorecard unavailable")

    # C1-C5 — collectors (each degrades gracefully).
    ms = _safe(lambda: market_data.collect(), {})
    ll = _safe(lambda: lead_lag.classify(), {})
    vp = _safe(lambda: volume_profile.build_all(), {})
    nw = _safe(lambda: news_feed.fetch(), {})
    cal = _safe(lambda: event_calendar.build(), {})
    prox = _read(FOREX_PROX)

    # C7 — daily FRED macro backdrop (same formatter Oracle uses; context only, never a trading input).
    from sovereign.oracle.reflect_cycle import _load_daily_macro
    macro_text = _safe(_load_daily_macro, "No macro snapshot (run scripts/fetch_fred_economic.py).")
    macro_summary = _read(MACRO_LATEST).get("summary", {})

    # C6 — Opus synthesis, with deterministic fallback.
    synth = _safe(lambda: synthesize.synthesize(ms, ll, vp, nw, cal, score_summary), None)
    if synth:
        directional_bias = synth["directional_bias"]
        confidence = synth["confidence"]
        regime_call = synth.get("regime_call") or ll.get("regime")
        key_level = synth.get("key_level")
        narrative = synth["narrative"]
        synth_source = synth.get("model", "opus")
    else:
        directional_bias = "NEUTRAL"
        confidence = 0
        regime_call = ll.get("regime")
        key_level = None
        narrative = _deterministic_narrative(ll, vp, cal, prox, score_summary)
        synth_source = "deterministic_fallback"

    if narrative_file and Path(narrative_file).exists():
        narrative = (Path(narrative_file).read_text().strip()
                     + "\n\n---\n[auto-appended signal read]\n" + narrative)

    # Surface the macro backdrop at the top of the readable narrative.
    narrative = f"Macro backdrop —\n{macro_text}\n\n---\n{narrative}"

    briefing = {
        "date": today,
        "generated_at": _now(),
        # --- keys reflect_cycle._load_market_briefing reads ---
        "meta_regime": ll.get("regime"),
        "regime_read": ll.get("read", ""),
        "narrative": narrative,
        # --- answer of the day: Big-Move-of-the-Day headline (display-only) ---
        "big_move_headline": _big_move_headline(),
        # --- structured call (scored by scorecard.py) ---
        "directional_bias": directional_bias,
        "confidence": confidence,
        "regime_call": regime_call,
        "key_level": key_level,
        "synthesis_source": synth_source,
        # --- component snapshots ---
        "lead_lag": {
            "regime": ll.get("regime"), "spread_avg": ll.get("nq_es_return_spread_avg"),
            "leader": ll.get("lead_lag", {}).get("leader"), "corr": ll.get("contemporaneous_corr"),
            "divergence": ll.get("divergence"),
        },
        "volume_profile": vp.get("instruments", {}),
        # --- daily FRED macro backdrop (context only) ---
        "macro_economic": {"text": macro_text, "summary": macro_summary},
        "news_count": nw.get("count", 0),
        "event_calendar": cal.get("events", []),
        "scorecard_summary": score_summary,
        "bias": ("size to regime: trust direction more in BREADTH, less in CONCENTRATION; "
                 "ROTATION_WARN = treat tech-led longs as suspect"),
        "provenance": {
            "source": "oracle_morning_briefing",
            "verified": False,
            "note": ("Analyst/Oracle journal — qualitative regime context + a self-scored "
                     "directional call, NOT a verified data feed or validated edge. Numbers here "
                     "are never ingested as trading inputs."),
        },
    }

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    (BRIEF_DIR / f"{today}.json").write_text(json.dumps(briefing, indent=2))
    (BRIEF_DIR / "latest.json").write_text(json.dumps(briefing, indent=2))

    # Heartbeat for loop_health (written every run, before nothing gates it).
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(_now())
    except Exception:
        pass
    return briefing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrative-file", default=None, help="optional richer narrative to embed for today")
    args = ap.parse_args()
    b = build(args.narrative_file)
    print(f"Morning briefing {b['date']}: regime={b['meta_regime']} bias={b['directional_bias']} "
          f"conf={b['confidence']} ({b['synthesis_source']}, verified={b['provenance']['verified']})")
    print(f"  {b['scorecard_summary']}")
    print(f"  Saved: data/oracle/market_briefings/{b['date']}.json + latest.json")


if __name__ == "__main__":
    main()
