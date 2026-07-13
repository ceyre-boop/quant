#!/usr/bin/env python3
"""G2 — evaluate HYP-093/094/095 on untouched holdouts, verbatim to their preregs.

Gate-zero first. Every constant read FROM the prereg JSON (no free parameters
here). Daily constitutional net series over the full holdout calendar; stats:
stationary block bootstrap vs zero (L=5, 10k, seed 42), DSR at n_trials=809
(non-annualized Sharpe input), BH across the family, floors + tail conditions.
Verdicts -> data/research/yield_frontier/gauntlet/ + ledger (backup first).
Run: python3 -m research.yield_frontier.gauntlet_run
"""
import gzip
import json
import shutil
from collections import defaultdict
from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._lib import REPO, canonical_hash, block_bootstrap_sharpe_diff_p
from sovereign.discovery.gate import benjamini_hochberg, deflated_sharpe_ratio

ET = ZoneInfo("America/New_York")
HD = REPO / "data/research/yield_frontier/holdout_equities"
OUT = REPO / "data/research/yield_frontier/gauntlet"
LEDGER = REPO / "data/agent/hypothesis_ledger.json"

# --- news classifier: verbatim copy of posthoc_scan (prereg names it) ---
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


def classify(heads):
    if not heads:
        return "NO_NEWS_PRE1030"
    blob = " " + " ".join(h.lower() for h in heads) + " "
    for name, kws in BUCKETS:
        if any(k in blob for k in kws):
            return name
    return "OTHER_NEWS"


def prereg(hyp):
    doc = json.loads((REPO / f"data/research/preregister/{hyp}_yield_frontier.json").read_text())
    lock = dict(doc)
    lock.pop("hash_lock")
    assert canonical_hash(lock) == doc["hash_lock"], f"{hyp} hash broken"
    return doc


def gate_zero():
    ledger = json.loads(LEDGER.read_text())
    for hyp in ("HYP-093", "HYP-094", "HYP-095"):
        prereg(hyp)
        e = next(x for x in ledger if x.get("id") == hyp)
        assert e["status"] == "PREREGISTERED", f"{hyp} not PREREGISTERED"
    print("[G2] gate-zero OK")


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def apr(gain, sched):
    if gain < 0.5:
        return sched["gap_0.3_0.5"]
    if gain < 1.0:
        return sched["gap_0.5_1.0"]
    if gain < 1.5:
        return sched["gap_1.0_1.5"]
    return sched["gap_1.5_plus"]


def load_holdout_grouped():
    days = {}
    denied = 0
    for fp in sorted((HD / "grouped").glob("*.json.gz")):
        with gzip.open(fp, "rt") as f:
            d = json.load(f)
        if d.get("denied_403"):
            denied += 1
            continue
        if d["n"] > 0:
            days[d["date"]] = d["bars"]
    return days, denied


def stats_block(daily_series, doc, event_rets, extra_tail_ok=True):
    """Common R3 evaluation. daily_series over the FULL holdout calendar."""
    a = np.asarray(daily_series, dtype=float)
    boot = block_bootstrap_sharpe_diff_p(a, np.zeros_like(a))
    sr_daily = float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0
    _, dsr_prob = deflated_sharpe_ratio(sr_daily, n_trials=doc["thresholds"]["mined_n"],
                                        n_obs=len(a))
    ev = np.asarray(event_rets, dtype=float)
    return {
        "n_days": len(a), "n_events": len(ev),
        "mean_pct_day_constitutional": round(float(a.mean()), 6),
        "boot_p": boot["p_one_sided"], "boot_detail": boot,
        "sharpe_daily": round(sr_daily, 4),
        "dsr_prob": round(float(dsr_prob), 4),
        "event_mean": round(float(ev.mean()), 5) if len(ev) else None,
        "event_median": round(float(np.median(ev)), 5) if len(ev) else None,
        "event_p5": round(float(np.percentile(ev, 5)), 5) if len(ev) else None,
        "worst_event": round(float(ev.min()), 5) if len(ev) else None,
        "floor": doc["thresholds"]["constitutional_floor_pct_day"],
        "floor_met": bool(len(a) and a.mean() >= doc["thresholds"]["constitutional_floor_pct_day"]),
        "tail_ok": bool(extra_tail_ok),
    }


def run_hyp093(grouped):
    doc = prereg("HYP-093")
    sched = doc["costs"]["borrow_apr_schedule_pessimistic"]
    slip = doc["costs"]["slippage_per_side"]
    w = doc["sizing_constitutional"]["notional_per_event"] * \
        doc["sizing_constitutional"]["locate_weight"]
    cands = defaultdict(list)
    with open(HD / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc = line.strip().split(",")
            cands[d].append(t)
    day_rets, event_rets, guards = {}, [], defaultdict(int)
    for day in sorted(grouped):
        day_rets[day] = 0.0
        fp = HD / f"alpaca/{day}.json.gz"
        nfp = HD / f"news/{day}.json"
        if day not in cands or not fp.exists():
            continue
        with gzip.open(fp, "rt") as f:
            payload = json.load(f)
        news = json.loads(nfp.read_text()) if nfp.exists() else {}
        for t in cands[day]:
            bars = payload["intraday"].get(t, [])
            daily = payload["daily"].get(t, [])
            if not bars or not daily:
                guards["no_data"] += 1
                continue
            pc = daily[-1]["c"]
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
            event_rets.append(ret)
            day_rets[day] += ret * w
    tail_ok = (len(event_rets) == 0 or
               (np.percentile(event_rets, 5) >= -0.60 and min(event_rets) >= -0.60))
    res = stats_block(list(day_rets.values()), doc, event_rets, tail_ok)
    res["guards"] = dict(guards)
    return doc, res


def run_hyp094(grouped):
    doc = prereg("HYP-094")
    sched = doc["costs"]["borrow_apr_schedule_pessimistic"]
    slip = doc["costs"]["slippage_per_side"]
    w = doc["sizing_constitutional"]["notional_per_event"] * \
        doc["sizing_constitutional"]["locate_weight"]
    sig = doc["signal"]
    loc_q1 = float(sig["close_location"].split("<=")[1].split("(")[0])
    dv = sig["dollar_volume"]
    dv_q1 = float(dv.split("(")[1].split(",")[0])
    dv_q2 = float(dv.split(",")[1].split("]")[0])
    cands = defaultdict(dict)
    with open(HD / "candidates.csv") as f:
        next(f)
        for line in f:
            d, t, pc = line.strip().split(",")
            cands[d][t] = float(pc)
    dates = sorted(grouped)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    day_rets, event_rets = {d: 0.0 for d in dates}, []
    absent = signals = 0
    for day in dates:
        for t, pc in cands.get(day, {}).items():
            bar = grouped[day].get(t)
            if bar is None or day not in nxt:
                continue
            o, h, l, c, v = bar
            if None in (o, h, l, c, v) or pc <= 0:
                continue
            g = c / pc - 1
            if not (0.30 <= g < 0.50):
                continue
            loc = (c - l) / (h - l) if h > l else np.nan
            dvol = c * v
            if not (np.isfinite(loc) and loc <= loc_q1 and dv_q1 < dvol <= dv_q2):
                continue
            signals += 1
            nb = grouped[nxt[day]].get(t)
            if nb is None or nb[3] is None:
                absent += 1
                continue
            ret = (c - nb[3]) / c - 2 * slip - 2 * sched["gap_0.3_0.5"] / 365
            event_rets.append(ret)
            day_rets[day] += ret * w
    tail_ok = (len(event_rets) == 0 or
               (min(event_rets) >= -1.00 and np.percentile(event_rets, 5) >= -0.50))
    res = stats_block(list(day_rets.values()), doc, event_rets, tail_ok)
    res["guards"] = {"signals": signals, "absent_next_day": absent,
                     "absent_rate": round(absent / signals, 4) if signals else 0}
    if signals and absent / signals > 0.05:
        res["data_insufficient"] = "absent_next_day rate > 5%"
    return doc, res


def run_hyp095():
    doc = prereg("HYP-095")
    vix_cut = float(doc["signal"].split(">=")[1].split("(")[0])
    daily = pd.read_parquet(REPO / "data/es_nq/nq_daily.parquet")   # unfenced: G2 only
    daily.index = pd.to_datetime(daily.index)
    aux = pd.read_parquet(REPO / "data/es_nq/aux_daily.parquet")
    aux.index = pd.to_datetime(aux.index)
    j = daily.join(aux["vix"].shift(1).rename("vix_prior"), how="left")
    j["oc"] = j["rth_close"] / j["rth_open"] - 1
    j["prior_oc"] = j["oc"].shift(1)
    hold = j[(j.index >= "2024-07-01") & (~j["roll_day"])]
    w = doc["sizing_constitutional"]["notional_per_event"]
    day_rets, event_rets = [], []
    for _, r in hold.iterrows():
        sig = (np.isfinite(r["vix_prior"]) and r["vix_prior"] >= vix_cut
               and np.isfinite(r["prior_oc"]) and r["prior_oc"] < -0.002)
        if sig and np.isfinite(r["oc"]):
            ret = r["oc"] - 0.625 / r["rth_open"]
            event_rets.append(ret)
            day_rets.append(ret * w)
        else:
            day_rets.append(0.0)
    tail_ok = len(event_rets) == 0 or min(event_rets) >= -0.10
    res = stats_block(day_rets, doc, event_rets, tail_ok)
    res["guards"] = {"holdout_sessions": len(hold), "signal_sessions": len(event_rets)}
    if len(event_rets) < 40:
        res["data_insufficient"] = "fewer than 40 signal sessions"
    return doc, res


def verdict_for(res, bh_ok):
    if res.get("data_insufficient"):
        return "DATA_INSUFFICIENT"
    if not res["tail_ok"]:
        return "NOT_ROBUST"
    sig = res["boot_p"] < 0.05 and bh_ok and res["dsr_prob"] >= 0.95
    if sig and res["floor_met"]:
        return "CONFIRMED"
    if sig:
        return "VALID_BUT_BELOW_FLOOR"
    return "NOT_SIGNIFICANT"


def main():
    gate_zero()
    grouped, denied = load_holdout_grouped()
    print(f"[G2] holdout grouped: {len(grouped)} trading days ({denied} denied-403, "
          "partial-window clause invoked)")
    results = {}
    d93, r93 = run_hyp093(grouped)
    d94, r94 = run_hyp094(grouped)
    d95, r95 = run_hyp095()
    ps = [r93["boot_p"], r94["boot_p"], r95["boot_p"]]
    bh = benjamini_hochberg(ps, alpha=0.05)
    for hyp, res in (("HYP-093", r93), ("HYP-094", r94), ("HYP-095", r95)):
        i = ("HYP-093", "HYP-094", "HYP-095").index(hyp)
        res["bh_survivor"] = bool(bh[i])
        res["verdict"] = verdict_for(res, bh[i])
        results[hyp] = res
        print(f"{hyp}: {res['verdict']} | p={res['boot_p']:.4f} bh={bh[i]} "
              f"dsr={res['dsr_prob']:.3f} %/day={res['mean_pct_day_constitutional']:+.5f} "
              f"floor_met={res['floor_met']} events={res['n_events']}")
    OUT.mkdir(parents=True, exist_ok=True)
    results["_meta"] = {
        "run": datetime.now(timezone.utc).isoformat(), "denied_403_days": denied,
        "partial_window": "2024-07-14..2025-06-30 effective (free-tier 2y lookback)",
        "bh_alpha": 0.05, "family_p": ps,
    }
    (OUT / "verdicts.json").write_text(json.dumps(results, indent=2, default=str))
    # seal ledger
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy(LEDGER, str(LEDGER) + f".bak-{stamp}")
    ledger = json.loads(LEDGER.read_text())
    for hyp in ("HYP-093", "HYP-094", "HYP-095"):
        prereg(hyp)   # hash re-verified post-run
        e = next(x for x in ledger if x.get("id") == hyp)
        r = results[hyp]
        e.update({"status": "ADJUDICATED", "verdict": r["verdict"],
                  "date_tested": "2026-07-13", "p_value": r["boot_p"],
                  "result": (f"boot_p={r['boot_p']:.4f}, dsr={r['dsr_prob']:.3f}, "
                             f"constitutional %/day={r['mean_pct_day_constitutional']:+.5f} "
                             f"(floor {r['floor']}), events={r['n_events']}, "
                             f"event_median={r['event_median']}, verdict={r['verdict']}. "
                             "Details data/research/yield_frontier/gauntlet/verdicts.json")})
    LEDGER.write_text(json.dumps(ledger, indent=2))
    print(f"[G2] sealed to ledger (backup .bak-{stamp}); hashes verified pre/post")


if __name__ == "__main__":
    main()
