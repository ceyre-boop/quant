#!/usr/bin/env python3
"""M1 — equities gapper mining (families F-EQ1..F-EQ5, ~272 cells).

STAMP: MINING. Look-back allowed and used deliberately; outputs are candidates,
never evidence. All data from the 2025-07→2026-06 caches via holdout_guard.
Run: python3 -m research.yield_frontier.m1_equities
"""
import gzip
import json
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._lib import OUT, record_mined
from .frictions import (EQ_PARTICIPATION_CAP, EQ_SLIPPAGE, locate_fill_prob,
                        short_borrow_cost)
from .holdout_guard import GAPPER_DIR, assert_no_equities_holdout_on_disk, \
    gapper_grouped_files
from .yield_board import row, write_session

ET = ZoneInfo("America/New_York")
N_DAYS = 251


def load_grouped():
    days = {}
    for fp in gapper_grouped_files():
        with gzip.open(fp, "rt") as f:
            d = json.load(f)
        if d["n"] > 0:
            days[d["date"]] = d["bars"]
    return days


def load_intraday(date_str):
    fp = GAPPER_DIR / f"cache/alpaca/{date_str}.json.gz"
    if not fp.exists():
        return None
    with gzip.open(fp, "rt") as f:
        return json.load(f)


def et_t(bar):
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def tercile_labels(values):
    q1, q2 = np.nanpercentile(values, [33.3, 66.7])
    return np.where(values <= q1, "T1", np.where(values <= q2, "T2", "T3")), (q1, q2)


# ---------------- F-EQ1 overnight continuation (108 cells) ----------------
def f_eq1(grouped, candidates):
    dates = sorted(grouped)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    ev = []
    absent = 0
    for _, r in candidates.iterrows():
        d, t, pc = r["date"], r["ticker"], r["prev_close_polygon"]
        bar = grouped.get(d, {}).get(t)
        if bar is None or d not in nxt:
            continue
        o, h, l, c, v = bar
        if None in (o, h, l, c, v) or pc <= 0 or c / pc - 1 < 0.30:
            continue
        nbar = grouped[nxt[d]].get(t)
        if nbar is None or nbar[0] is None or nbar[3] is None:
            absent += 1
            continue
        ev.append({"date": d, "gain": c / pc - 1,
                   "loc": (c - l) / (h - l) if h > l else np.nan,
                   "dvol": c * v,
                   "to_open": nbar[0] / c - 1, "to_close": nbar[3] / c - 1})
    df = pd.DataFrame(ev)
    df["loc_t"], _ = tercile_labels(df["loc"].to_numpy())
    df["vol_t"], _ = tercile_labels(df["dvol"].to_numpy())
    rows = []
    gain_b = [("g30-50", (0.30, 0.50)), ("g50-100", (0.50, 1.00)), ("g100+", (1.00, 99))]
    for gname, (lo, hi) in gain_b:
        for loc_t in ("T1", "T2", "T3"):
            for vol_t in ("T1", "T2", "T3"):
                sub = df[(df.gain >= lo) & (df.gain < hi) &
                         (df.loc_t == loc_t) & (df.vol_t == vol_t)]
                cap = EQ_PARTICIPATION_CAP * float(sub["dvol"].median()) if len(sub) else 0
                for exit_, col in (("next_open", "to_open"), ("next_close", "to_close")):
                    r_ = sub[col].to_numpy()
                    days_held = 1.0 if exit_ == "next_open" else 2.0
                    for dir_ in ("long", "short"):
                        if dir_ == "long":
                            rets = r_ - 2 * EQ_SLIPPAGE
                            fp_, adj = 1.0, 0.0
                        else:
                            gm = float(sub["gain"].mean()) if len(sub) else 1.0
                            rets = -r_ - 2 * EQ_SLIPPAGE
                            adj = short_borrow_cost(gm, days_held)
                            fp_ = locate_fill_prob(gm)
                        rows.append(row("equities", "F-EQ1_overnight",
                                        f"{gname}|loc{loc_t}|vol{vol_t}|{dir_}|{exit_}",
                                        rets, len(rets) / N_DAYS, cap,
                                        net_adjust=adj, fill_prob=fp_))
    return rows, 108, {"next_day_absent": absent, "events": len(df)}


# ---------------- F-EQ2 parabolic fade shorts (96 cells) ----------------
def walk_short(bars_post, entry, stop_frac, end_time):
    stop_px = entry * (1 + stop_frac) if stop_frac else None
    last = None
    for b in bars_post:
        t = et_t(b)
        if t >= end_time:
            break
        if stop_px is not None:
            if b["o"] >= stop_px:
                return (entry - b["o"]) / entry
            if b["h"] >= stop_px:
                return (entry - stop_px) / entry
        last = b
    if last is None:
        return np.nan
    return (entry - last["c"]) / entry


def f_eq2(enriched):
    rows = []
    by_date = defaultdict(list)
    for _, r in enriched.iterrows():
        by_date[r["date"]].append(r)
    cache = {}
    events = defaultdict(list)   # (thr, stop, exit, mna) -> list of (ret, gain, dollar_vol)
    for d, rs in by_date.items():
        payload = load_intraday(d)
        if payload is None:
            continue
        for r in rs:
            bars = payload["intraday"].get(r["ticker"], [])
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            if not post:
                continue
            entry = r["entry_open_1030"]
            for thr in (0.5, 1.0, 1.5):
                if r["gain_1030"] < thr:
                    continue
                for mna in (False, True):
                    if mna and r["catalyst"] == "MERGER_ACQ":
                        continue
                    for stop in (0.10, 0.20, 0.30, None):
                        for exit_, endt in (("1530", dtime(15, 30)), ("close", dtime(16, 0))):
                            ret = walk_short(post, entry, stop, endt) - 2 * EQ_SLIPPAGE
                            events[(thr, stop, exit_, mna)].append(
                                (ret, r["gain_1030"], r["price_1030"] * r["cum_vol_1030"]))
    for (thr, stop, exit_, mna), evs in events.items():
        rets = np.array([e[0] for e in evs])
        gm = float(np.mean([e[1] for e in evs]))
        cap = EQ_PARTICIPATION_CAP * float(np.median([e[2] for e in evs]))
        rows.append(row("equities", "F-EQ2_fade_short",
                        f"thr{thr}|stop{stop}|{exit_}|mna_excl={mna}",
                        rets, len(rets) / N_DAYS, cap,
                        fill_prob=locate_fill_prob(gm)))
    return rows, 96, {}


# ---------------- F-EQ3 halt-runner re-entry longs (36 cells) ----------------
def f_eq3(grouped, candidates):
    dates = sorted(grouped)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    events = defaultdict(list)
    for d, sub in candidates.groupby("date"):
        payload = load_intraday(d)
        if payload is None:
            continue
        for _, r in sub.iterrows():
            t = r["ticker"]
            bars = payload["intraday"].get(t, [])
            sl = [b for b in bars if dtime(9, 30) <= et_t(b) <= dtime(10, 25)]
            sparse = len(sl) < 8 or (sl and et_t(sl[-1]) < dtime(10, 15)) or not sl
            if not sparse:
                continue
            daily = payload["daily"].get(t, [])
            if not daily:
                continue
            pc = daily[-1]["c"]
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            for entry_ix, ename in ((0, "first"), (1, "second")):
                if len(post) <= entry_ix:
                    continue
                eb = post[entry_ix]
                entry = eb["o"]
                gap = entry / pc - 1
                walk = post[entry_ix:]
                for thr in (0.3, 0.5, 1.0):
                    if gap < thr:
                        continue
                    for stop in (-0.10, -0.20, None):
                        stopped = None
                        if stop is not None:
                            spx = entry * (1 + stop)
                            for b in walk[1:]:
                                if b["o"] <= spx:
                                    stopped = (b["o"] / entry - 1)
                                    break
                                if b["l"] <= spx:
                                    stopped = stop
                                    break
                        eod = walk[-1]["c"] / entry - 1
                        nb = grouped.get(nxt.get(d, ""), {}).get(t)
                        for exit_ in ("close", "next_open"):
                            if stopped is not None:
                                ret = stopped
                            elif exit_ == "close":
                                ret = eod
                            else:
                                if nb is None or nb[0] is None:
                                    continue
                                ret = nb[0] / entry - 1
                            events[(thr, ename, exit_, stop)].append(
                                (ret - 2 * EQ_SLIPPAGE, entry * eb["v"]))
    rows = []
    for (thr, ename, exit_, stop), evs in events.items():
        rets = np.array([e[0] for e in evs])
        cap = EQ_PARTICIPATION_CAP * float(np.median([e[1] for e in evs]))
        rows.append(row("equities", "F-EQ3_halt_runner",
                        f"gap{thr}|entry_{ename}|{exit_}|stop{stop}",
                        rets, len(rets) / N_DAYS, cap))
    return rows, 36, {}


# ---------------- F-EQ4 no-news recipe + F-EQ5 catalyst longs (32 cells) ----------------
def f_eq45(enriched, grouped):
    dates = sorted(grouped)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    rows = []
    # F-EQ4
    base = enriched[(enriched.catalyst == "NO_NEWS_PRE1030") &
                    (enriched.gain_1030 >= 0.3) & (enriched.gain_1030 < 0.5)]
    events = defaultdict(list)
    for d, sub in base.groupby("date"):
        payload = load_intraday(d)
        if payload is None:
            continue
        for _, r in sub.iterrows():
            bars = payload["intraday"].get(r["ticker"], [])
            post = [b for b in bars if dtime(10, 30) <= et_t(b) < dtime(16, 0)]
            if not post:
                continue
            entries = {"1030open": post[0]["o"]}
            if r["price_1030"] < r["vwap_1030"]:
                rec = next((b for b in post if b["c"] >= r["vwap_1030"]), None)
                entries["vwap_reclaim"] = rec["c"] if rec else None
            else:
                entries["vwap_reclaim"] = post[0]["o"]
            nb = grouped.get(nxt.get(d, ""), {}).get(r["ticker"])
            for ename, entry in entries.items():
                if entry is None:
                    continue
                eod = post[-1]["c"] / entry - 1
                for read in ("CONT", "any"):
                    if read == "CONT" and r["read"] != "CONT":
                        continue
                    for stop in (-0.10, None):
                        stopped = None
                        if stop is not None:
                            spx = entry * (1 + stop)
                            for b in post[1:]:
                                if b["o"] <= spx:
                                    stopped = b["o"] / entry - 1
                                    break
                                if b["l"] <= spx:
                                    stopped = stop
                                    break
                        for exit_ in ("close", "next_open"):
                            if stopped is not None:
                                ret = stopped
                            elif exit_ == "close":
                                ret = eod
                            elif nb is not None and nb[0] is not None:
                                ret = nb[0] / entry - 1
                            else:
                                continue
                            events[(read, ename, exit_, stop)].append(
                                (ret - 2 * EQ_SLIPPAGE,
                                 r["price_1030"] * r["cum_vol_1030"]))
    for (read, ename, exit_, stop), evs in events.items():
        rets = np.array([e[0] for e in evs])
        cap = EQ_PARTICIPATION_CAP * float(np.median([e[1] for e in evs]))
        rows.append(row("equities", "F-EQ4_nonews_recipe",
                        f"read{read}|{ename}|{exit_}|stop{stop}",
                        rets, len(rets) / N_DAYS, cap))
    # F-EQ5 catalyst-conditioned 10:30->close longs
    for cat, sub in enriched.groupby("catalyst"):
        for read in ("CONT", "any"):
            s = sub if read == "any" else sub[sub.read == "CONT"]
            rets = (s["close_eod"] / s["entry_open_1030"] - 1 - 2 * EQ_SLIPPAGE).to_numpy()
            cap = EQ_PARTICIPATION_CAP * float(
                (s["price_1030"] * s["cum_vol_1030"]).median()) if len(s) else 0
            rows.append(row("equities", "F-EQ5_catalyst_long",
                            f"{cat}|read{read}", rets, len(rets) / N_DAYS, cap))
    return rows, 16 + 20, {}


def main():
    assert_no_equities_holdout_on_disk()
    grouped = load_grouped()
    candidates = pd.read_csv(GAPPER_DIR / "candidates.csv")
    enriched = pd.read_csv(GAPPER_DIR / "per_candidate_enriched.csv")
    all_rows, meta = [], {}
    for fn, args in ((f_eq1, (grouped, candidates)), (f_eq2, (enriched,)),
                     (f_eq3, (grouped, candidates)), (f_eq45, (enriched, grouped))):
        rows, n_cells, m = fn(*args)
        fam = rows[0]["family"] if rows else fn.__name__
        record_mined("equities", fam.split("_")[0] + fn.__name__, n_cells)
        all_rows += rows
        meta.update(m)
    path = write_session("m1_equities", all_rows)
    ranked = [r for r in all_rows if r.get("n", 0) >= 40 and "net_pct_day" in r]
    ranked.sort(key=lambda r: -r["net_pct_day"])
    print(f"[M1] {len(all_rows)} rows ({len(ranked)} rankable) -> {path}; meta={meta}")
    for r in ranked[:10]:
        print(f"  {r['family']:<22} {r['config']:<44} n={r['n']:>4} "
              f"net/day={r['net_pct_day']:+.4f} med={r['median_pct']:+.4f} "
              f"p5={r['tail_p5']:+.3f} cap=${r['capacity_usd']:,}")


if __name__ == "__main__":
    main()
