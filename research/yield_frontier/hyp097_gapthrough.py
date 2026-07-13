#!/usr/bin/env python3
"""HYP-097 — evidence-based worst-case gap-through for the frozen fade signal.

Replaces HYP-093's DECLARED sizing worst-case (2x stop distance = 60%) with a
per-LULD-tier bound W* = safety x max(structural collar bound, realized max
overshoot), then re-derives the constitutional yield as pure arithmetic on the
sealed event stream. HYP-093's significance verdict is untouched.

STRUCTURAL SCENARIO (locked): price approaches the +30% stop from below; a
limit-up halt fires with pre-halt reference at the stop; the reopen prints at
the collar with ONE 5% extension (DERA: reversion is the norm, extensions the
tail; deeper cascades are covered by the 1.25 safety multiplier):
    T10 (prev close >= $3, 10% bands): 1.30*1.10*1.05 - 1 = 0.5015
    T20 (prev close  < $3, 20% bands): 1.30*1.20*1.05 - 1 = 0.6380
NOTE recorded pre-lock: this physics floor means the operator's thesis (~30%
worst case, income doubles) may NOT survive; the null (NOT_CLEARED, premise
confirmed-or-worse) is fully live. The rule locks anyway — that is the point.

Phases:  --prereg  write + hash-lock BEFORE measurement
         --run     measure, re-derive, seal verdict (hash verified pre/post)
"""
import gzip
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ._lib import REPO, canonical_hash

ET = ZoneInfo("America/New_York")
PREREG = REPO / "data/research/preregister/HYP-097_gapthrough_sizing.json"
LEDGER = REPO / "data/agent/hypothesis_ledger.json"
OUT = REPO / "data/research/yield_frontier/hyp097"
MINING = REPO / "data/research/gapper"
HOLDOUT = REPO / "data/research/yield_frontier/holdout_equities"

SPEC = {
    "id": "HYP-097",
    "name": "Evidence-based worst-case gap-through replaces the declared 2x-stop sizing premise",
    "registered": "2026-07-13", "ticket": "TICK-033",
    "not_holdout_reuse": "Significance (HYP-093 p=0.031) sealed and untouched; only the sizing premise (a mechanics assumption) is re-derived; event returns re-weighted by arithmetic, never re-tested.",
    "event_population": "All signal events per the frozen HYP-093 spec, reconstructed identically from mining-year AND holdout-year caches (same code path as the sealed runs).",
    "tiers": {"T10": "prev_close >= 3.00 (LULD 10% bands)",
              "T20": "0.75 <= prev_close < 3.00 (LULD 20% bands)"},
    "structural_total_adverse": {"T10": 0.5015, "T20": 0.6380},
    "structural_scenario": "single limit-up halt crossing the stop, reopen at collar with one 5% extension; DERA reversion-is-the-norm justifies one extension as the tail scenario; deeper cascades covered by safety_mult",
    "empirical_component": "max realized total adverse move from entry among STOPPED events per tier across both years: max(exit_px/entry - 1)",
    "derivation_locked": "W*_tier = min(1.25 * max(structural_total_adverse_tier, empirical_max_tier), 1.00)",
    "safety_mult": 1.25,
    "re_derivation": "constitutional daily yield = per-day sum of sealed event returns * (0.0075 / W*_tier(event)) * locate 0.50 over the holdout trading-day calendar (verdict basis); mining calendar reported as disclosure",
    "sensitivities_non_verdict": ["borrow at flat 300% APR replacing tiered schedule", "risk budget 0.82%"],
    "verdict_rule": "holdout-calendar yield >= 0.0005/day at the 0.75% budget -> SIZING_CLEARED; else NOT_CLEARED",
    "priors_recorded": {"operator": "SIZING_CLEARED (~30% worst-case thesis)",
                        "analyst_pre_lock": "NOT_CLEARED live — LULD collar math yields structural floors of 50-64% before safety"},
    "disclosures": [
        "5-min bar resolution: within-bar sequencing approximated by the sealed fill rule (open-through, then high-touch).",
        "Broker risk-desk discretionary liquidation remains unmodeled tail (W3/W4).",
        "SSR entry effects unmodeled here, as in HYP-093 (bind on entries, not stops).",
        "Fixed-fractional Art.1 framework only; RCK/heat-budget sizing is W6's question, not this one.",
    ],
}

MNA = ["merger", "acquisition", "acquire", "buyout", "takeover",
       "definitive agreement", "letter of intent", "strategic alternatives",
       "going private"]


def prereg():
    doc = dict(SPEC)
    doc["hash_lock"] = canonical_hash(doc)
    PREREG.write_text(json.dumps(doc, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy(LEDGER, str(LEDGER) + f".bak-{stamp}")
    led = json.loads(LEDGER.read_text())
    assert not any(e.get("id") == "HYP-097" for e in led)
    led.append({"id": "HYP-097", "name": doc["name"], "status": "PREREGISTERED",
                "date_registered": "2026-07-13",
                "prereg_file": str(PREREG.relative_to(REPO)),
                "hash_lock": doc["hash_lock"], "ticket": "TICK-033",
                "prior_expectation": "operator SIZING_CLEARED / analyst NOT_CLEARED — both recorded pre-lock",
                "methodology_note": "W* = 1.25 x max(LULD structural bound [T10 .5015 / T20 .638], realized max overshoot) per tier; deterministic re-weighting of the sealed HYP-093 stream; floor 0.05%/day unchanged."})
    LEDGER.write_text(json.dumps(led, indent=2))
    print(f"HYP-097 locked {doc['hash_lock'][:8]} + PREREGISTERED (backup .bak-{stamp})")


def verify():
    doc = json.loads(PREREG.read_text())
    lock = doc.pop("hash_lock")
    assert canonical_hash(doc) == lock, "HYP-097 hash broken"
    return doc


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def trading_days(grouped_dir):
    days = []
    for fp in sorted(Path(grouped_dir).glob("*.json.gz")):
        with gzip.open(fp, "rt") as f:
            d = json.load(f)
        if d.get("n", 0) > 0 and not d.get("denied_403"):
            days.append(d["date"])
    return days


def events_from(alpaca_dir, cand_file, news_dir, dates):
    cands = defaultdict(list)
    with open(cand_file) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            cands[p[0]].append(p[1])
    evs = []
    for day in dates:
        fp = Path(alpaca_dir) / f"{day}.json.gz"
        if day not in cands or not fp.exists():
            continue
        with gzip.open(fp, "rt") as f:
            payload = json.load(f)
        nfp = Path(news_dir) / f"{day}.json"
        news = json.loads(nfp.read_text()) if nfp.exists() else {}
        for t in cands[day]:
            bars = payload["intraday"].get(t, [])
            daily = payload["daily"].get(t, [])
            if not bars or not daily:
                continue
            pc = daily[-1]["c"]
            sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
            if len(sl) < 8 or et_t(sl[-1]) < dtime(10, 15) or pc <= 0:
                continue
            P = sl[-1]["c"]
            gain = P / pc - 1
            if not (P >= 1.30 * pc and P >= 2.00 and
                    sum(b["v"] for b in sl) >= 500_000 and gain >= 0.50):
                continue
            if any(k in " ".join(h.lower() for h in news.get(t, [])) for k in MNA):
                continue
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            if not post:
                continue
            entry = post[0]["o"]
            stop_px = entry * 1.30
            exit_px, stopped = None, False
            for b in post[1:]:
                if b["o"] >= stop_px:
                    exit_px, stopped = b["o"], True
                    break
                if b["h"] >= stop_px:
                    exit_px, stopped = stop_px, True
                    break
            if exit_px is None:
                exit_px = post[-1]["c"]
            apr = 6.0 if gain >= 1.5 else 4.0 if gain >= 1.0 else 2.0
            evs.append({"date": day, "tier": "T10" if pc >= 3.00 else "T20",
                        "adverse_total": exit_px / entry - 1, "stopped": stopped,
                        "ret": (entry - exit_px) / entry - 0.01 - apr / 365,
                        "apr": apr})
    return evs


def daily_yield(evs, dates, W, budget=0.0075, borrow_flat=None):
    day = {d: 0.0 for d in dates}
    for e in evs:
        if e["date"] not in day:
            continue
        r = e["ret"]
        if borrow_flat is not None:
            r = r + e["apr"] / 365 - borrow_flat / 365
        day[e["date"]] += r * (budget / W[e["tier"]]) * 0.50
    return float(np.mean(list(day.values())))


def run():
    doc = verify()
    m_days = trading_days(MINING / "cache/grouped")
    h_days = trading_days(HOLDOUT / "grouped")
    ev_m = events_from(MINING / "cache/alpaca", MINING / "candidates.csv",
                       MINING / "cache/news", m_days)
    ev_h = events_from(HOLDOUT / "alpaca", HOLDOUT / "candidates.csv",
                       HOLDOUT / "news", h_days)
    allev = ev_m + ev_h
    struct = doc["structural_total_adverse"]
    W, emp, nstop = {}, {}, {}
    for tier in ("T10", "T20"):
        stopped = [e["adverse_total"] for e in allev
                   if e["tier"] == tier and e["stopped"]]
        nstop[tier] = len(stopped)
        emp[tier] = max(stopped) if stopped else 0.30
        W[tier] = min(doc["safety_mult"] * max(struct[tier], emp[tier]), 1.00)
    y_hold = daily_yield(ev_h, h_days, W)
    res = {
        "verdict": "SIZING_CLEARED" if y_hold >= 0.0005 else "NOT_CLEARED",
        "W_star": {k: round(v, 4) for k, v in W.items()},
        "empirical_max_adverse": {k: round(emp[k], 4) for k in emp},
        "n_stopped": nstop,
        "structural_bounds": struct,
        "tier_mix_holdout": {t: sum(1 for e in ev_h if e["tier"] == t)
                             for t in ("T10", "T20")},
        "yield_holdout_calendar": round(y_hold, 6),
        "yield_mining_calendar_disclosure": round(daily_yield(ev_m, m_days, W), 6),
        "floor": 0.0005,
        "sens_borrow300": round(daily_yield(ev_h, h_days, W, borrow_flat=3.0), 6),
        "sens_budget_0082": round(y_hold * 0.82 / 0.75, 6),
        "sealed_baseline_at_060": 0.00023,
        "events": {"mining": len(ev_m), "holdout": len(ev_h)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    verify()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if "--prereg" in sys.argv:
        prereg()
    elif "--run" in sys.argv:
        run()
    else:
        raise SystemExit("--prereg first, then --run")
