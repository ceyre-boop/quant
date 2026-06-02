#!/usr/bin/env python3
"""C6 — Opus 4.8 synthesis (produces the briefing narrative + structured call).

Feeds the five collectors (market_state, lead_lag, volume_profile, news, event_calendar)
plus the scorecard track-record into ONE Claude Opus 4.8 call and asks for the exact
morning-briefing format: day-by-day moves mapped to catalysts, the volume-profile read,
the NQ/ES lead-lag regime, event-calendar awareness, and a synthesis.

NON-NEGOTIABLE PROMPT RULES:
  - Grade its OWN confidence (high only when news is clear AND lead-lag is clean; say "mixed"
    otherwise; never manufacture certainty).
  - State the bias as a PROBABILITY with an INVALIDATION level, not a prediction.

Returns a structured dict {directional_bias, confidence, regime_call, key_level, narrative,
model, cost_usd} — or None on any failure (missing key / API error / parse error), so the
orchestrator falls back to its deterministic synthesis and a scheduled run never dies.

Reuses the reflect_cycle anthropic pattern (model="claude-opus-4-8", _load_dotenv).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COST_LOG = ROOT / "logs" / "oracle_cost.json"

MODEL = "claude-opus-4-8"
# Opus pricing (verify against current Anthropic docs): $15/1M input, $75/1M output.
_IN_RATE = 0.000015
_OUT_RATE = 0.000075

PROMPT_TEMPLATE = """You are the morning market analyst for a disciplined quant trader who trades the ES/NQ \
(S&P 500 / Nasdaq-100 futures) correlated pair. Produce a pre-US-open market briefing.

You are given the system's REAL signals (JSON below). Build the briefing ONLY from these plus \
widely-known market context — do not invent specific numbers.

MARKET STATE (ES + NQ):
{market_state}

NQ/ES LEAD-LAG REGIME (the core read — concentration vs breadth vs rotation):
{lead_lag}

VOLUME PROFILE (where volume traded — NOT order-flow delta; respect that limitation):
{volume_profile}

NEWS (tagged headlines):
{news}

EVENT CALENDAR (FOMC is seed; CPI/PCE/NFP/ISM are cadence estimates — treat as approximate):
{event_calendar}

BRIEFING TRACK RECORD (your own scorecard so far — be humble if it's unproven):
{scorecard}

Write a briefing with these sections:
1. Recent move-by-move read, each move tied to its likely catalyst.
2. Volume-profile read: accumulation zone / POC / the line in the sand.
3. NQ/ES lead-lag regime classification and what it implies (size/trust).
4. Event-calendar awareness (what's coming, how to size into it).
5. SYNTHESIS: the meta-regime + a NARROWED-probability bias.

RULES:
- GRADE YOUR OWN CONFIDENCE. High confidence ONLY when news is clear AND lead-lag is clean. \
If signals are mixed, say so and lower confidence. Never manufacture certainty.
- State the bias as a PROBABILITY with an INVALIDATION LEVEL, e.g. "long-the-dip bias, ~65%, \
invalidated below 30,375".
- This briefing is INTELLIGENCE, not a validated edge. It must not pretend to be one.

Respond with ONLY a JSON object (no prose outside it):
{{
  "directional_bias": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": <integer 0-100>,
  "regime_call": "CONCENTRATION" | "BREADTH" | "ROTATION_WARN",
  "key_level": <number — the invalidation price for NQ, or null>,
  "narrative": "<the full multi-section briefing as readable prose>"
}}"""


def _log_cost(in_tok: int, out_tok: int, cost: float) -> None:
    try:
        COST_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = json.loads(COST_LOG.read_text()) if COST_LOG.exists() else {"entries": []}
        if not isinstance(log, dict) or "entries" not in log:
            log = {"entries": []}
        log["entries"].append({
            "at": datetime.now(timezone.utc).isoformat(), "source": "morning_briefing_synthesis",
            "model": MODEL, "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": round(cost, 6),
        })
        log["entries"] = log["entries"][-500:]
        COST_LOG.write_text(json.dumps(log, indent=2))
    except Exception:
        pass


def _compact(obj, limit: int = 2500) -> str:
    s = json.dumps(obj, default=str)
    return s[:limit]


def _parse(raw: str) -> dict | None:
    """Parse the model JSON, tolerating ```json fences and truncated narratives.

    The structured scalars come FIRST in the schema, so even a narrative truncated by
    max_tokens still yields a usable call — we salvage the head fields + whatever
    narrative text arrived."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    # 1) clean parse
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "narrative" in d:
            return d
    except Exception:
        pass
    # 2) full-object regex
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, dict) and "narrative" in d:
                return d
        except Exception:
            pass
    # 3) salvage from a truncated object (scalars are first; narrative is last)
    bias = re.search(r'"directional_bias"\s*:\s*"([^"]+)"', raw)
    conf = re.search(r'"confidence"\s*:\s*(\d+)', raw)
    regime = re.search(r'"regime_call"\s*:\s*"([^"]+)"', raw)
    klvl = re.search(r'"key_level"\s*:\s*([0-9.]+|null)', raw)
    narr = re.search(r'"narrative"\s*:\s*"(.*)', raw, re.DOTALL)
    if not (bias or narr):
        return None
    narrative = ""
    if narr:
        narrative = narr.group(1)
        # strip a clean trailing close if present; otherwise keep the truncated text
        narrative = re.sub(r'"\s*\}?\s*$', "", narrative)
        narrative = narrative.encode().decode("unicode_escape", errors="ignore")
    return {
        "directional_bias": bias.group(1) if bias else "NEUTRAL",
        "confidence": int(conf.group(1)) if conf else 0,
        "regime_call": regime.group(1) if regime else None,
        "key_level": (None if (not klvl or klvl.group(1) == "null") else float(klvl.group(1))),
        "narrative": narrative.strip() + ("  [note: narrative truncated at token limit]"
                                          if narr and not raw.rstrip().endswith("}") else ""),
    }


def synthesize(market_state: dict, lead_lag: dict, volume_profile: dict,
               news: dict, event_calendar: dict, scorecard_summary: str) -> dict | None:
    """Run the Opus synthesis. Returns the structured briefing, or None to trigger fallback."""
    try:
        from sovereign.oracle.oracle_agent import _load_dotenv
        _load_dotenv()
    except Exception:
        pass
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    prompt = PROMPT_TEMPLATE.format(
        market_state=_compact(market_state),
        lead_lag=_compact(lead_lag),
        volume_profile=_compact(volume_profile),
        news=_compact(news, 2000),
        event_calendar=_compact(event_calendar, 1500),
        scorecard=scorecard_summary,
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        cost = in_tok * _IN_RATE + out_tok * _OUT_RATE
        _log_cost(in_tok, out_tok, cost)
    except Exception:
        return None

    data = _parse(raw)
    if data is None:
        return None

    bias = str(data.get("directional_bias", "NEUTRAL")).upper()
    if bias not in ("LONG", "SHORT", "NEUTRAL"):
        bias = "NEUTRAL"
    try:
        conf = int(data.get("confidence", 0))
    except Exception:
        conf = 0
    return {
        "directional_bias": bias,
        "confidence": max(0, min(100, conf)),
        "regime_call": data.get("regime_call") or lead_lag.get("regime"),
        "key_level": data.get("key_level"),
        "narrative": data.get("narrative", ""),
        "model": MODEL,
        "cost_usd": round(cost, 6),
    }
