"""Layer 6 — end-of-day reconciliation, written to Obsidian.

WHAT IT RECONCILES
------------------
Four comparisons, each of which can fail silently today:

  signals fired  vs  GO list      — did every GO produce an attempt?
  fills taken    vs  signals      — did every attempt produce a fill, or was it
                                    skipped, and for what stated reason?
  net return     vs  backtest     — does reality match the model that will be
                                    used to decide whether this is tradeable?
  risk gates     triggered        — did any ratified article fire?

plus the bias scoring loop from Layer 2, so the predictor's record accumulates
whether or not anyone reads it.

WHY UNFILLED GO SIGNALS ARE THE HEADLINE
-----------------------------------------
The most useful number here is GO signals that produced no fill. A system that
reports "0 fills" looks identical whether nothing qualified, or twelve things
qualified and every one was unfillable. The ICT bottleneck was exactly this shape
— roughly 94 setups per 90 days of which about 98% expired unfilled — and it went
unnoticed because the reporting counted signals, not conversions.

So the note leads with conversion, and every gap carries its reason.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from execution import bias as bias_mod
from execution import obsidian, signals as signals_mod
from execution.context import Status
from sovereign.utils.timestamps import canonical_timestamp

ROOT = Path(__file__).resolve().parents[1]
FILL_DIR = ROOT / "data" / "execution"
SIGNAL_DIR = ROOT / "data" / "signals"
CONTEXT_DIR = ROOT / "data" / "context"
UTC = timezone.utc


def _load_fills(day: date | str, fill_dir: Path) -> list[dict]:
    p = fill_dir / "fill_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("date") == str(day):
            out.append(r)
    return out


def _load_context(day: date | str, ctx_dir: Path) -> dict | None:
    p = ctx_dir / f"morning_context_{day}.json"
    return json.loads(p.read_text()) if p.exists() else None


def reconcile(day: date | str, *, fill_dir: Path | None = None,
              signal_dir: Path | None = None, ctx_dir: Path | None = None,
              bias_dir: Path | None = None) -> dict:
    """Build the reconciliation. Pure — no writes, no network."""
    fill_dir = fill_dir or FILL_DIR
    signal_dir = signal_dir or SIGNAL_DIR
    ctx_dir = ctx_dir or CONTEXT_DIR

    sigs = signals_mod.load_signals(day, signal_dir)
    gos = [s for s in sigs if s["decision"] == "GO"]
    fills = _load_fills(day, fill_dir)

    filled = [f for f in fills if not f["signal_type"].startswith("SKIP_")]
    skipped = [f for f in fills if f["signal_type"].startswith("SKIP_")]

    attempted_ids = {f.get("signal_id") for f in fills if f.get("signal_id")}
    unfilled_gos = [s for s in gos if s["signal_id"] not in attempted_ids]

    skip_reasons: dict[str, int] = {}
    for f in skipped:
        skip_reasons[f["signal_type"]] = skip_reasons.get(f["signal_type"], 0) + 1

    risk_fired = [f for f in fills if f.get("risk_breached")]

    nets = [f["net_return"] for f in filled if f.get("net_return") is not None]
    exps = [f["backtest_expected_return"] for f in filled
            if f.get("backtest_expected_return") is not None]
    spreads = [f["spread_cost"] for f in filled if f.get("spread_cost") is not None]

    ctx = _load_context(day, ctx_dir)
    b = bias_mod.load_bias(day, bias_dir)

    return {
        "date": str(day),
        "generated_at": canonical_timestamp(),
        "signals": {
            "n_total": len(sigs), "n_go": len(gos),
            "n_no_go": len(sigs) - len(gos),
            "by_hypothesis": {h: sum(1 for s in gos if s["hypothesis"] == h)
                              for h in {s["hypothesis"] for s in sigs}} if sigs else {},
        },
        "conversion": {
            "n_go": len(gos),
            "n_attempted": len(attempted_ids),
            "n_filled": len(filled),
            "n_skipped": len(skipped),
            "n_go_unfilled": len(unfilled_gos),
            "fill_rate": round(len(filled) / len(gos), 4) if gos else None,
            "unfilled_detail": [{"signal_id": s["signal_id"], "ticker": s["ticker"],
                                 "hypothesis": s["hypothesis"]} for s in unfilled_gos],
            "skip_reasons": skip_reasons,
        },
        "performance": {
            "n": len(nets),
            "median_net": round(median(nets), 6) if nets else None,
            "median_spread_cost": round(median(spreads), 6) if spreads else None,
            "median_backtest_expected": round(median(exps), 6) if exps else None,
            "vs_backtest_delta": (round(median(nets) - median(exps), 6)
                                  if nets and exps else None),
        },
        "risk": {
            "n_gates_fired": len(risk_fired),
            "detail": [{"ticker": f["ticker"], "action": f.get("risk_action"),
                        "breached": f.get("risk_breached")} for f in risk_fired],
        },
        "bias": ({"direction": b.direction, "confidence": b.confidence,
                  "context_fraction_fresh": b.context_fraction_fresh,
                  "realised": b.realised} if b else None),
        "context_health": (ctx or {}).get("health"),
    }


def render(rec: dict) -> str:
    """Markdown body for the vault note."""
    c, p, s, r = rec["conversion"], rec["performance"], rec["signals"], rec["risk"]
    L: list[str] = []

    L.append("## Conversion — the number that matters")
    L.append("")
    if c["n_go"] == 0:
        L.append("**0 GO signals.** No candidate passed a frozen filter today. "
                 "This is a real zero, not a silent failure — see the signal "
                 "table below for every rejection and its reason.")
    else:
        L.append(f"| GO | attempted | filled | skipped | **unfilled GO** |")
        L.append(f"|---|---|---|---|---|")
        L.append(f"| {c['n_go']} | {c['n_attempted']} | {c['n_filled']} | "
                 f"{c['n_skipped']} | **{c['n_go_unfilled']}** |")
        L.append("")
        if c["fill_rate"] is not None:
            L.append(f"Fill rate **{c['fill_rate']:.0%}**.")
        if c["n_go_unfilled"]:
            L.append("")
            L.append("**Unfilled GO signals** (a signal that never becomes a fill "
                     "is the failure mode that hid the ICT bottleneck — ~98% of "
                     "setups expiring unfilled, invisible because reporting "
                     "counted signals rather than conversions):")
            for u in c["unfilled_detail"]:
                L.append(f"- `{u['signal_id']}` — {u['ticker']} {u['hypothesis']}")
    if c["skip_reasons"]:
        L.append("")
        L.append("Skips by reason: " + ", ".join(
            f"`{k}` × {v}" for k, v in sorted(c["skip_reasons"].items())))

    L += ["", "## Performance vs backtest", ""]
    if p["n"]:
        L.append(f"- Fills: **{p['n']}**")
        L.append(f"- Median net: **{p['median_net']:+.4%}**")
        L.append(f"- Median spread cost: {p['median_spread_cost']:.4%}")
        L.append(f"- Median backtest expectation: {p['median_backtest_expected']:+.4%}"
                 if p["median_backtest_expected"] is not None else
                 "- Median backtest expectation: n/a")
        if p["vs_backtest_delta"] is not None:
            L.append(f"- **vs_backtest_delta: {p['vs_backtest_delta']:+.4%}**")
    else:
        L.append("No fills — nothing to compare.")

    L += ["", "## Risk gates", ""]
    if r["n_gates_fired"] == 0:
        L.append("No ratified article fired.")
    else:
        for d in r["detail"]:
            L.append(f"- **{d['ticker']}** — {d['action']}: {', '.join(d['breached'])}")

    L += ["", "## Bias (recorded, non-gating)", ""]
    if rec["bias"]:
        b = rec["bias"]
        L.append(f"- Direction **{b['direction']}**, confidence {b['confidence']:.3f} "
                 f"(context {b['context_fraction_fresh']:.0%} fresh)")
        L.append(f"- Realised: {b['realised']}" if b["realised"] else
                 "- Realised: not yet scored")
        L.append("- This bias gated nothing. It is recorded so it can earn — or "
                 "fail to earn — a track record.")
    else:
        L.append("No bias recorded.")

    L += ["", "## Information health", ""]
    h = rec["context_health"]
    if h:
        L.append(f"- **{h['n_fresh']}/{h['n_sources']} sources FRESH "
                 f"({h['fraction_fresh']:.0%})**")
        for status, names in sorted(h.get("by_status", {}).items()):
            if status != Status.FRESH.value:
                L.append(f"- {status}: {', '.join(names)}")
        if h["fraction_fresh"] < 0.6:
            L.append("")
            L.append("> Most inputs were not live today. Read every conclusion "
                     "above against that.")
    else:
        L.append("No morning context was built.")

    L += ["", "## Signals", ""]
    L.append(f"{s['n_go']} GO / {s['n_no_go']} NO_GO of {s['n_total']} scored.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Layer 6 — EOD reconciliation")
    ap.add_argument("--day", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="render without writing to the vault")
    ap.add_argument("--fills", default=None)
    ap.add_argument("--signals", default=None)
    ap.add_argument("--score-bias", metavar="DIRECTION", default=None,
                    help="score today's bias against the realised direction")
    args = ap.parse_args(argv)

    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()

    if args.score_bias:
        bias_mod.score_bias(day, args.score_bias.upper())

    rec = reconcile(day,
                    fill_dir=Path(args.fills) if args.fills else None,
                    signal_dir=Path(args.signals) if args.signals else None)
    body = render(rec)
    p, text = obsidian.write_eod_note(day, body, dry_run=args.dry_run)

    print(text if args.dry_run else f"wrote {p}")
    if args.dry_run:
        print(f"\n(dry run — would write to {p})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
