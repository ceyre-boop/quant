"""Layer 2 — daily directional bias. RECORDED, SCORED, AND NEVER GATING.

THE WALL THIS RESPECTS
----------------------
`docs/ARCHITECTURE.md:61-65` splits the system in two:

    Layer 1 — PREDICTIVE.  What to bet on and which way. Ceiling ~55%.
    Layer 2 — EVALUATIVE.  What to do about a position already held.
                           Deterministic. No AI, no prediction, post-entry.

and `:284-286` fixes the handoff:

    "Only the FACT of a setup crosses L1 -> L2: direction, the entry trigger, and
     the risk geometry the setup implies. NO REASONING, NO PROBABILITY, NO
     NARRATIVE CROSSES."

`ARCHITECTURE.md:362-367` names the one component that violates this —
`futures/decision_engine.evaluate_entry()`, Anti-pattern §6.3 — where prediction
and evaluation are "fused at birth, so neither can be tested or replaced in
isolation." It is the only component classified as violating, and it is also the
only live bias->entry path in the system.

This module deliberately does not extend it.

WHY NON-GATING IS NOT TIMIDITY
-------------------------------
The predictive layer currently has 23+ clean nulls and zero confirmed edges. Its
own architecture doc calls it "inherently humble; ceiling ~55%". Wiring an
unvalidated predictor upstream of the only two surviving edges would let it veto
signals that survived preregistration, on the strength of nothing.

So the bias is written down, published to Obsidian, and SCORED against realised
outcomes every day. That builds the track record that does not exist today. If it
earns predictive value over a meaningful sample, promoting it to a gate is then a
preregistration away — with evidence, in the direction the ledger discipline
already demands.

ENFORCEMENT: `tests/unit/test_bias_isolation.py` asserts that no signal-layer
module imports this one. The wall is a test, not a comment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from execution.context import Status, build_morning_context
from sovereign.utils.timestamps import canonical_timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "bias"
UTC = timezone.utc

VALID_DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL")


@dataclass
class Bias:
    """A recorded directional opinion. Carries its own provenance and honesty."""
    date: str
    direction: str
    confidence: float                      # 0.0 - 1.0
    reasoning: list[str] = field(default_factory=list)
    inputs_used: list[str] = field(default_factory=list)
    inputs_missing: list[str] = field(default_factory=list)
    context_fraction_fresh: float = 0.0
    generated_at: str = ""
    # filled in later by score_bias()
    realised: dict[str, Any] | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["generated_at"] = d["generated_at"] or canonical_timestamp()
        return d


def derive_bias(ctx: dict) -> Bias:
    """Derive a directional bias from the morning context.

    Deliberately simple and transparent — this is a recorder, not a model. It
    reads the briefing's own `directional_bias` when present and downgrades
    confidence in proportion to how much of the context was actually FRESH.

    A bias built on 2 of 7 live sources should not be stated with the same
    confidence as one built on 7 of 7, and here that is arithmetic rather than
    a footnote.
    """
    fields = ctx.get("fields", {})
    fresh = [n for n, f in fields.items() if f["status"] == Status.FRESH.value]
    missing = [n for n, f in fields.items()
               if f["status"] in (Status.UNAVAILABLE.value, Status.SILENT_NULL.value,
                                  Status.ERROR.value)]
    frac = ctx.get("health", {}).get("fraction_fresh", 0.0)

    reasoning: list[str] = []
    direction, base_conf = "NEUTRAL", 0.0

    brief = fields.get("briefing", {})
    if brief.get("status") == Status.FRESH.value and brief.get("value"):
        v = brief["value"]
        raw = (v.get("directional_bias") or "").upper()
        if raw in VALID_DIRECTIONS:
            direction = raw
            base_conf = float(v.get("confidence") or 0.0)
            reasoning.append(f"briefing directional_bias={raw} conf={base_conf}")
        if v.get("regime_read"):
            reasoning.append(f"regime: {v['regime_read']}")
    elif brief.get("status") == Status.STALE.value:
        reasoning.append("briefing STALE — not used for direction")
    else:
        reasoning.append("briefing unavailable — no directional input")

    macro = fields.get("fred_macro", {})
    if macro.get("status") == Status.FRESH.value and macro.get("value"):
        reasoning.append(f"macro context present ({macro['value'].get('date')})")
    else:
        reasoning.append("macro context absent or stale")

    # Confidence is scaled by how much of the context was real. A bias derived
    # from a mostly-dark morning is stated weakly, by construction.
    confidence = round(base_conf * frac, 4)
    if direction != "NEUTRAL" and confidence == 0.0:
        reasoning.append("confidence floored to 0 — no FRESH inputs supported it")

    return Bias(
        date=ctx["date"], direction=direction, confidence=confidence,
        reasoning=reasoning, inputs_used=sorted(fresh), inputs_missing=sorted(missing),
        context_fraction_fresh=frac, generated_at=canonical_timestamp(),
    )


#: Broad-market proxy the bias is scored against. The bias is a market-direction
#: call, so it must be scored against the market — not against whether the day's
#: gapper trades happened to win, which would conflate direction with execution.
REALISED_PROXY = "SPY"

#: Daily move smaller than this counts as NEUTRAL rather than a direction.
#: DEFINITIONAL, NOT FITTED. 0.25% is roughly a quiet SPY session; it was chosen
#: before any scoring was run and must not be tuned afterwards to improve the hit
#: rate. Changing it invalidates every prior score, so it is asserted in a test.
NEUTRAL_BAND = 0.0025


def realised_direction(day: date, *, proxy: str = REALISED_PROXY) -> tuple[str, dict]:
    """What the market actually did on `day`, as BULLISH / BEARISH / NEUTRAL.

    Returns (direction, detail). Direction is "UNKNOWN" when the proxy bar is
    unavailable — an unscoreable day must never be silently recorded as NEUTRAL,
    because NEUTRAL is a real answer and UNKNOWN is the absence of one.
    """
    from execution import alpaca
    try:
        alpaca.load_env()
        prev_close = alpaca.daily_prev_close(proxy, day)
        bars = alpaca.minute_bars(proxy, day)
        if not bars or not prev_close:
            return "UNKNOWN", {"reason": "no proxy bars", "proxy": proxy}
        close = float(bars[-1]["c"])
        ret = close / float(prev_close) - 1.0
    except Exception as e:                                   # noqa: BLE001
        return "UNKNOWN", {"reason": f"{type(e).__name__}: {e}", "proxy": proxy}

    if abs(ret) < NEUTRAL_BAND:
        direction = "NEUTRAL"
    else:
        direction = "BULLISH" if ret > 0 else "BEARISH"
    return direction, {"proxy": proxy, "return": round(ret, 6),
                       "neutral_band": NEUTRAL_BAND, "close": close,
                       "prev_close": float(prev_close)}


def bias_path(day: date | str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUT_DIR) / f"bias_{day}.json"


def write_bias(b: Bias, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = bias_path(b.date, out_dir)
    p.write_text(json.dumps(b.to_json(), indent=2))
    return p


def load_bias(day: date | str, out_dir: Path | None = None) -> Bias | None:
    p = bias_path(day, out_dir)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Bias(**d)


def score_bias(day: date | str, realised_direction: str,
               realised_detail: dict | None = None,
               out_dir: Path | None = None) -> Bias | None:
    """Append the realised outcome next to a previously recorded bias.

    This is the whole point of the layer. Without scoring, a bias is an opinion
    that never has to be right. With it, the component accumulates a record that
    can later justify — or refuse — promotion to a gate.
    """
    b = load_bias(day, out_dir)
    if b is None:
        return None
    correct = None
    if b.direction in ("BULLISH", "BEARISH") and realised_direction in ("BULLISH", "BEARISH"):
        correct = (b.direction == realised_direction)
    b.realised = {
        "direction": realised_direction,
        "correct": correct,          # None when either side was NEUTRAL
        "scored_at": canonical_timestamp(),
        **(realised_detail or {}),
    }
    write_bias(b, out_dir)
    return b


def track_record(out_dir: Path | None = None) -> dict:
    """Accumulated hit rate over all scored biases.

    Reports `n_scored` alongside every rate. A hit rate without its sample size
    is the kind of number this repo has been burned by before.
    """
    out_dir = out_dir or OUT_DIR
    if not out_dir.exists():
        return {"n_total": 0, "n_scored": 0, "hit_rate": None}
    rows = []
    for p in sorted(out_dir.glob("bias_*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    scored = [r for r in rows
              if r.get("realised") and r["realised"].get("correct") is not None]
    hits = sum(1 for r in scored if r["realised"]["correct"])
    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_directional": sum(1 for r in rows if r.get("direction") != "NEUTRAL"),
        "hit_rate": round(hits / len(scored), 4) if scored else None,
        "note": ("Hit rate is meaningless below ~30 scored days. This is a track "
                 "record being built, not evidence of an edge."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Daily bias — recorded, scored, non-gating")
    ap.add_argument("--day", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", metavar="DIRECTION",
                    help="score a prior day's bias against the realised direction")
    ap.add_argument("--record", action="store_true", help="show the track record")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else None
    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()

    if args.record:
        print(json.dumps(track_record(out_dir), indent=2))
        return 0

    if args.score:
        b = score_bias(day, args.score.upper(), out_dir=out_dir)
        if b is None:
            print(f"no bias recorded for {day}")
            return 1
        print(f"{day}: predicted {b.direction}, realised {args.score.upper()}, "
              f"correct={b.realised['correct']}")
        return 0

    ctx = build_morning_context(day)
    b = derive_bias(ctx)
    p = write_bias(b, out_dir)
    print(f"{b.date}  {b.direction}  confidence {b.confidence:.3f} "
          f"(context {b.context_fraction_fresh:.0%} fresh)")
    for r in b.reasoning:
        print(f"    · {r}")
    if b.inputs_missing:
        print(f"    missing: {', '.join(b.inputs_missing)}")
    print(f"\nwrote {p}")
    print("NOTE: this bias gates nothing. The signal layer does not read it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
