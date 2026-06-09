"""The Big Move Oracle — predict the day's ONE institutional move (cognition only).

Fuses everything the sandbox already pulls into a single forecast:
  MAGNITUDE  = VIX-implied expected daily move (implied_move)
  DIRECTION  = ICT structure (prior H/L, overnight, POC/VAH/VAL) + order flow (CVD) + regime
  TIMING     = the catalyst calendar (FOMC/CPI/NFP/open)
  CONVICTION = the Oracle's own calibrated track record (recent pulse hit-rate)

`gather_context()` builds the fused snapshot; `forecast()` asks Sonnet 4.6 for a falsifiable
BigMoveForecast. This is a clean, callable cognition unit — the exact shape an Agents-SDK
sub-agent slots into when it ships. It does NOT place orders. Sandbox-local.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sovereign.futures import bar_feed as bf
from sovereign.futures import cvd as cvd_mod
from sovereign.futures import regime as regime_mod
from sovereign.futures import volume_profile as vp
from sovereign.futures import implied_move as im

ROOT = Path(__file__).resolve().parents[2]
PULSE_LOG = ROOT / "data" / "futures" / "big_move_pulse.jsonl"
MODEL = "claude-sonnet-4-6"          # fast + cheap for the recurring intraday pulse

SYSTEM_PROMPT = """\
You are the BIG MOVE analyst for an ES/NQ futures learning system. Markets make ONE primary
directional move per day — a draw on liquidity toward resting stops, often triggered by a
catalyst. Your job: predict THAT move, not the noise.

You are given a fused snapshot: VIX-implied expected daily move (the MAGNITUDE the market is
pricing), structure (prior high/low, overnight range, volume POC/VAH/VAL = where institutions
transact), order-flow CVD, regime, today's range-so-far, and the catalyst calendar.

Rules — non-negotiable:
1. Anchor expected_move_pts to the implied expected move; do not invent a magnitude.
2. drawn_to_level = the specific price liquidity is most likely drawn toward today (a prior
   high/low, POC/VAH/VAL, or overnight extreme) — name a real number from the snapshot.
3. Every call needs a falsifier: the price/condition that proves you wrong today.
4. stated_probability between 0.40 and 0.80. Never exactly 0.50.
5. If genuinely unreadable (conflicting signals, imminent high-impact event with no lean),
   set direction = "NO_PREDICTION".
6. reasoning: 3 lines max, specific factors only, no filler.

Output ONLY valid JSON, this exact schema:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL" | "NO_PREDICTION",
  "expected_move_pts": float,
  "drawn_to_level": float,
  "trigger_window": "e.g. 09:30-10:30 ET or post-CPI 08:30",
  "catalyst": "the main driver in a few words",
  "conviction": 1 | 2 | 3,
  "stated_probability": 0.40-0.80,
  "falsifier": "specific price/condition that invalidates this",
  "reasoning": "line1\\nline2\\nline3"
}"""


@dataclass
class BigMoveForecast:
    instrument: str
    direction: str
    expected_move_pts: Optional[float]
    drawn_to_level: Optional[float]
    trigger_window: str
    catalyst: str
    conviction: int
    stated_probability: float
    falsifier: str
    reasoning: str
    # echoed context (for the dashboard + calibration)
    last_price: Optional[float] = None
    vix: Optional[float] = None
    implied_move_pts: Optional[float] = None
    regime_state: Optional[str] = None
    cvd_slope: Optional[float] = None
    model: str = MODEL
    ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── catalyst calendar (self-contained, best-effort) ───────────────────────────

def _fetch_calendar() -> list[dict]:
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://www.econdb.com/api/events/?date={today}&country=US&format=json"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        HIGH = {"fomc", "federal reserve", "interest rate", "cpi", "consumer price", "pce",
                "nfp", "nonfarm", "jobs report", "gdp", "jobless", "ism", "retail sales"}
        out = []
        for item in data.get("results", []):
            name = (item.get("name") or "").lower()
            if item.get("importance", 0) >= 2 or any(k in name for k in HIGH):
                out.append({"name": item.get("name"), "time": item.get("time", "")})
        return out
    except Exception:
        return []


def _recent_calibration() -> Optional[dict]:
    """Hit-rate of recently scored pulses (the self-calibration signal)."""
    if not PULSE_LOG.exists():
        return None
    scored = []
    for line in PULSE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("direction_hit") is not None:
            scored.append(r["direction_hit"])
    if not scored:
        return None
    s = scored[-50:]
    return {"n": len(s), "direction_hit_rate": round(sum(s) / len(s), 2)}


# ── context fusion ────────────────────────────────────────────────────────────

def gather_context(instrument: str, source: str = "ib") -> dict:
    """Fuse structure + CVD + regime + implied move + catalyst + calibration. Null-safe."""
    ctx: dict = {"instrument": instrument, "ts": datetime.now(timezone.utc).isoformat(),
                 "errors": []}
    try:
        df = bf.load_history(instrument, source=source, lookback="5d")
    except Exception as e:
        ctx["errors"].append(f"bars: {type(e).__name__}: {e}")
        df = None

    last_price = None
    if df is not None and len(df):
        days = bf.session_days(df)
        et = df.index.tz_convert(bf.ET).strftime("%Y-%m-%d")
        today_df = df[et == days[-1]]
        prior_df = df[et == days[-2]] if len(days) >= 2 else None
        if len(today_df):
            last_price = float(today_df["Close"].iloc[-1])
            ctx["today"] = {
                "last_price": round(last_price, 2),
                "rth_high": round(float(today_df["High"].max()), 2),
                "rth_low": round(float(today_df["Low"].min()), 2),
                "bars_so_far": int(len(today_df)),
            }
            if len(today_df) >= 5:
                ctx["orb"] = {"high": round(float(today_df["High"].iloc[:5].max()), 2),
                              "low": round(float(today_df["Low"].iloc[:5].min()), 2)}
            cst = cvd_mod.cvd_state(today_df)
            ctx["cvd"] = ({"slope": round(cst["slope"], 1),
                           "quality": "HIGH" if cvd_mod.is_strong(cst) else "LOW"} if cst else None)
        if prior_df is not None and len(prior_df):
            ctx["prior_day"] = {
                "high": round(float(prior_df["High"].max()), 2),
                "low": round(float(prior_df["Low"].min()), 2),
                "close": round(float(prior_df["Close"].iloc[-1]), 2),
            }
            ctx["volume_profile"] = vp.compute_profile(prior_df)
        # regime + ADR-used
        ranges = []
        for d in days[:-1]:
            ddf = df[et == d]
            if len(ddf):
                ranges.append(float(ddf["High"].max() - ddf["Low"].min()))
        adr = sum(ranges) / len(ranges) if ranges else None
        if len(today_df) and adr:
            today_range = float(today_df["High"].max() - today_df["Low"].min())
            ctx["adr_used_pct"] = round(today_range / adr, 2)
            ctx["regime"] = regime_mod.classify_session(today_df, adr_used_pct=ctx["adr_used_pct"])

    ctx["implied_move"] = im.expected_daily_move(last_price) if last_price else {"vix": im.get_vix()}
    ctx["catalysts"] = _fetch_calendar()
    ctx["calibration"] = _recent_calibration()
    return ctx


def _build_user_prompt(ctx: dict) -> str:
    return ("=== BIG MOVE SNAPSHOT ===\n" + json.dumps(ctx, indent=2, default=str) +
            "\n\nProduce the BigMoveForecast JSON for today's primary move. JSON only.")


def _call_sonnet(user_prompt: str) -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not in .env")
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=MODEL, max_tokens=900,
                                 system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": user_prompt}])
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def forecast(ctx: dict) -> Optional[BigMoveForecast]:
    """Ask Sonnet 4.6 for the day's big move. Returns None on API/parse failure."""
    try:
        r = _call_sonnet(_build_user_prompt(ctx))
    except Exception:
        return None
    im_ctx = ctx.get("implied_move", {}) or {}
    today = ctx.get("today", {}) or {}
    regime = ctx.get("regime", {}) or {}
    cvd_ctx = ctx.get("cvd", {}) or {}
    return BigMoveForecast(
        instrument=ctx.get("instrument", "MES"),
        direction=r.get("direction", "NO_PREDICTION"),
        expected_move_pts=r.get("expected_move_pts"),
        drawn_to_level=r.get("drawn_to_level"),
        trigger_window=r.get("trigger_window", ""),
        catalyst=r.get("catalyst", ""),
        conviction=int(r.get("conviction", 1) or 1),
        stated_probability=float(r.get("stated_probability", 0.5) or 0.5),
        falsifier=r.get("falsifier", ""),
        reasoning=r.get("reasoning", ""),
        last_price=today.get("last_price"),
        vix=im_ctx.get("vix"),
        implied_move_pts=im_ctx.get("expected_move_pts"),
        regime_state=regime.get("trend_state"),
        cvd_slope=cvd_ctx.get("slope"),
        ts=datetime.now(timezone.utc).isoformat(),
    )
