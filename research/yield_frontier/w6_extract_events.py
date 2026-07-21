#!/usr/bin/env python3
"""
w6_extract_events.py — reproduce the sealed HYP-093 holdout event paths for W6.

WHY THIS EXISTS
---------------
The W6 sizing simulator bootstraps from the 559 sealed HYP-093 holdout events, but
the gauntlet stored only summary statistics in verdicts.json — not the per-event
return stream. Those events are, however, *deterministically reproducible* from the
raw Alpaca holdout via the frozen fill rule in `gauntlet_run.run_hyp093`.

This script re-runs that exact rule (transcribed verbatim, same thresholds, same
cost model) and emits one row per event with the fields W6 needs:
    date, ticker, prev_close, tier (T10/T20), gain_1030, entry, exit, ret_event

It changes nothing. It reads the sealed holdout and writes a derived event table to
optimization/W6_inputs/hyp093_events.json. The reproduction is validated against the
sealed verdicts.json: n_events, event_mean, event_median, event_p5 and worst_event
must all match, or the script refuses to write.

Tier rule (from HYP-097, frozen): T10 = prev_close >= $3.00 (LULD 10% bands),
T20 = 0.75 <= prev_close < $3.00 (20% bands).
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
HD = REPO / "data" / "research" / "yield_frontier" / "holdout_equities"
PREREG = REPO / "data" / "research" / "preregister" / "HYP-093_yield_frontier.json"
SEALED = REPO / "data" / "research" / "yield_frontier" / "gauntlet" / "verdicts.json"
OUT_DIR = REPO / "data" / "research" / "yield_frontier" / "optimization" / "W6_inputs"
ET = ZoneInfo("America/New_York")

# News classifier — transcribed VERBATIM from gauntlet_run.py (BUCKETS + classify).
# Order matters: MERGER_ACQ is excluded only if it is the FIRST matching bucket, so
# the whole ordered list and the first-match rule must be reproduced exactly, not a
# simplified keyword scan. FDA_CLINICAL precedes MERGER_ACQ deliberately.
BUCKETS = [
    ("FDA_CLINICAL", ["fda", "phase 1", "phase 2", "phase 3", "clinical",
                      "trial", "approval", "clearance", "breakthrough", "nda",
                      "510(k)", "orphan drug"]),
    ("MERGER_ACQ", ["merger", "acquisition", "acquire", "buyout", "takeover",
                    "definitive agreement", "letter of intent", "strategic alternatives",
                    "going private"]),
    ("OFFERING", ["offering", "private placement", "registered direct",
                  "dilution", "warrant", "at-the-market"]),
    ("EARNINGS", ["earnings", "quarterly results", "q1 ", "q2 ", "q3 ", "q4 ",
                  "revenue", "eps", "guidance", "fiscal"]),
    ("CRYPTO", ["bitcoin", "crypto", "ethereum", "solana", "token",
                "digital asset", "treasury strategy"]),
    ("AI", [" ai ", "artificial intelligence", "ai-powered", "genai"]),
    ("CONTRACT_PARTNER", ["contract", "partnership", "collaboration",
                          "purchase order", "agreement", "award", "deal with"]),
    ("ANALYST", ["upgrade", "price target", "initiates"]),
]


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def classify(heads) -> str:
    if not heads:
        return "NO_NEWS_PRE1030"
    blob = " " + " ".join(h.lower() for h in heads) + " "
    for name, kws in BUCKETS:
        if any(k in blob for k in kws):
            return name
    return "OTHER_NEWS"


def apr(gain: float, sched: dict) -> float:
    if gain < 0.5:
        return sched["gap_0.3_0.5"]
    if gain < 1.0:
        return sched["gap_0.5_1.0"]
    if gain < 1.5:
        return sched["gap_1.0_1.5"]
    return sched["gap_1.5_plus"]


def tier_of(prev_close: float) -> str:
    return "T10" if prev_close >= 3.00 else "T20"


def extract() -> list[dict]:
    doc = json.loads(PREREG.read_text())
    sched = doc["costs"]["borrow_apr_schedule_pessimistic"]
    slip = doc["costs"]["slippage_per_side"]

    cands = defaultdict(list)
    with open(HD / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc = line.strip().split(",")
            cands[d].append((t, float(pc)))

    events: list[dict] = []
    guards = defaultdict(int)

    for fp in sorted((HD / "grouped").glob("*.json.gz")):
        day = fp.stem.replace(".json", "")
        afp = HD / f"alpaca/{day}.json.gz"
        nfp = HD / f"news/{day}.json"
        if day not in cands or not afp.exists():
            continue
        with gzip.open(afp, "rt") as f:
            payload = json.load(f)
        news = json.loads(nfp.read_text()) if nfp.exists() else {}

        for t, pc_poly in cands[day]:
            bars = payload["intraday"].get(t, [])
            daily = payload["daily"].get(t, [])
            if not bars or not daily:
                guards["no_data"] += 1
                continue
            pc = daily[-1]["c"]  # prior close from Alpaca daily (the sealed source)
            sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
            if len(sl) < 8 or et_t(sl[-1]) < dtime(10, 15):
                guards["sparse"] += 1
                continue
            P = sl[-1]["c"]
            vol = sum(b["v"] for b in sl)
            if not (P >= 1.30 * pc and P >= 2.00 and vol >= 500_000):
                continue
            gain = P / pc - 1
            if gain < 0.50:
                continue
            if classify(news.get(t, [])) == "MERGER_ACQ":
                guards["mna_excluded"] += 1
                continue
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            if not post:
                guards["no_entry"] += 1
                continue
            entry = post[0]["o"]
            stop_px = entry * 1.30
            exit_px = None
            for b in post[1:]:
                if b["o"] >= stop_px:
                    exit_px = b["o"]
                    break
                if b["h"] >= stop_px:
                    exit_px = stop_px
                    break
            if exit_px is None:
                exit_px = post[-1]["c"]
            ret = (entry - exit_px) / entry - 2 * slip - apr(gain, sched) / 365
            events.append({
                "date": day, "ticker": t,
                "prev_close": round(pc, 4), "tier": tier_of(pc),
                "gain_1030": round(gain, 5),
                "entry": round(entry, 4), "exit": round(exit_px, 4),
                "ret_event": round(ret, 6),
            })
    return events, dict(guards)


def validate(events: list[dict]) -> tuple[bool, list[str]]:
    sealed = json.loads(SEALED.read_text())["HYP-093"]
    r = np.array([e["ret_event"] for e in events], dtype=float)
    checks = [
        ("n_events", len(events), sealed["n_events"], 0),
        ("event_mean", round(float(r.mean()), 5), sealed["event_mean"], 0.0005),
        ("event_median", round(float(np.median(r)), 5), sealed["event_median"], 0.0005),
        ("event_p5", round(float(np.percentile(r, 5)), 5), sealed["event_p5"], 0.002),
        ("worst_event", round(float(r.min()), 5), sealed["worst_event"], 0.002),
    ]
    msgs, ok = [], True
    for name, got, want, tol in checks:
        match = abs(got - want) <= tol
        ok = ok and match
        msgs.append(f"  {'OK ' if match else 'XXX'} {name}: reproduced {got} vs sealed {want}"
                    + ("" if match else f"  (tol {tol})"))
    return ok, msgs


def main() -> int:
    events, guards = extract()
    ok, msgs = validate(events)
    print(f"Reproduced {len(events)} HYP-093 holdout events.")
    print("Validation against sealed verdicts.json:")
    print("\n".join(msgs))
    print(f"Guards: {guards}")
    n_t10 = sum(1 for e in events if e["tier"] == "T10")
    print(f"Tier split: T10={n_t10}  T20={len(events) - n_t10}")

    if not ok:
        print("\nVALIDATION FAILED — reproduction does not match the sealed verdict. "
              "Refusing to write. The bootstrap must not run on an unvalidated event set.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source": "reproduced from sealed HYP-093 holdout via frozen fill rule",
        "validated_against": "gauntlet/verdicts.json HYP-093",
        "n_events": len(events),
        "tier_split": {"T10": n_t10, "T20": len(events) - n_t10},
        "guards": guards,
        "events": events,
    }
    (OUT_DIR / "hyp093_events.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'hyp093_events.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
