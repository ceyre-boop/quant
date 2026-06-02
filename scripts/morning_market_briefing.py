#!/usr/bin/env python3
"""
Morning market briefing generator (Oracle's daily journal).
===========================================================

Each morning, synthesize a market-context briefing from the signals the system actually has —
the NQ/ES lead-lag regime, forex proximity-to-trigger, and the near-term event calendar — and
store it where Oracle (reflect_cycle / session_open) and the dashboard read it.

PROVENANCE DISCIPLINE: the briefing is Oracle's ANALYTICAL JOURNAL, flagged
`provenance.verified=false`. It is qualitative regime context, NOT a verified data feed — its
narrative claims must never be ingested as numbers the system computes trades on. (The whole
system's trustworthiness rests on not letting confident-but-unverified claims drive decisions.)

Deterministic core (no API-key dependency, so a scheduled run can't fail on a missing key).
Optional `--narrative-file PATH` injects a richer human/Oracle-written narrative for that day.

Usage:
    python3 scripts/morning_market_briefing.py
    python3 scripts/morning_market_briefing.py --narrative-file data/oracle/market_briefings/_seed.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIEF_DIR = ROOT / "data" / "oracle" / "market_briefings"
NQES = ROOT / "data" / "research" / "nqes_regime.json"
FOREX_PROX = ROOT / "data" / "agent" / "forex_proximity.json"

# Near-term known events (seed; flagged unverified — replace with an econ-calendar feed later).
EVENT_CALENDAR = [
    {"date": "2026-06-16/17", "event": "FOMC meeting", "action": "reduce size / widen stops into the window"},
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build(narrative_file: str | None = None) -> dict:
    nqes = _read(NQES)
    prox = _read(FOREX_PROX)
    today = date.today().isoformat()

    regime = nqes.get("regime", "UNKNOWN")
    nqes_read = nqes.get("read", "NQ/ES regime not computed yet (run nqes_regime.py).")
    forex_verdict = prox.get("verdict", "forex proximity not computed yet")

    # Deterministic narrative synthesis from real signals.
    synth = (
        f"Session open {today}. NQ/ES regime: {regime} — {nqes_read} "
        f"NQ {nqes.get('nq_last','?')} / ES {nqes.get('es_last','?')}, "
        f"contemporaneous corr {nqes.get('contemporaneous_corr','?')}. "
        f"Forex: {forex_verdict}. "
        f"Event watch: {', '.join(e['event']+' ('+e['date']+')' for e in EVENT_CALENDAR)} — "
        f"reduce size / widen stops into known event windows (that is when unpredictable shocks cluster). "
        f"Regime read is an INPUT for sizing/trust, not a trade signal."
    )
    narrative = synth
    if narrative_file and Path(narrative_file).exists():
        narrative = Path(narrative_file).read_text().strip() + "\n\n---\n[auto-appended signal read]\n" + synth

    briefing = {
        "date": today,
        "generated_at": _now(),
        "meta_regime": regime,
        "regime_read": nqes_read,
        "nqes_leadlag_class": {
            "regime": regime,
            "spread_avg": nqes.get("nq_es_return_spread_avg"),
            "lead_lag": nqes.get("lead_lag"),
            "corr": nqes.get("contemporaneous_corr"),
        },
        "forex_proximity": forex_verdict,
        "event_calendar": EVENT_CALENDAR,
        "bias": "size to regime: trust direction more in BREADTH, less in CONCENTRATION; "
                "ROTATION_DIVERGENCE = treat tech-led longs as suspect",
        "provenance": {
            "source": "oracle_morning_briefing",
            "verified": False,
            "note": "Analyst/Oracle narrative — qualitative regime context, NOT a verified data feed. "
                    "Numbers here are not ingested as trading inputs.",
        },
        "narrative": narrative,
    }

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    (BRIEF_DIR / f"{today}.json").write_text(json.dumps(briefing, indent=2))
    (BRIEF_DIR / "latest.json").write_text(json.dumps(briefing, indent=2))
    return briefing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrative-file", default=None, help="optional richer narrative to embed for today")
    args = ap.parse_args()
    b = build(args.narrative_file)
    print(f"Morning briefing {b['date']}: meta_regime={b['meta_regime']} (verified={b['provenance']['verified']})")
    print(f"  {b['narrative'][:200]}...")
    print(f"  Saved: data/oracle/market_briefings/{b['date']}.json + latest.json")


if __name__ == "__main__":
    main()
