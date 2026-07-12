#!/usr/bin/env python3
"""M2 — NQ intraday mining 2018-01→2024-06 (families F-NQ1..F-NQ6, ~348 cells).

STAMP: MINING. Data only through holdout_guard.load_nq (rows ≤ 2024-06-30).
Fill conventions per sovereign/es_nq/backtest.py: bar-close entries + slippage,
stop-first when stop and target share a bar, forced flat 15:55 ET.
Returns are fractions of notional (1×, no leverage); costs = MNQ points/entry.
Run: python3 -m research.yield_frontier.m2_nq
"""
import numpy as np
import pandas as pd

from ._lib import record_mined
from .frictions import NQ_RT_COST_PTS
from .holdout_guard import load_nq
from .yield_board import row, write_session

NQ_CAPACITY = 50_000_000   # index futures depth; not the binding constraint here
COST_PTS = NQ_RT_COST_PTS["MNQ"]


def per_year(dates, rets):
    s = pd.Series(rets, index=pd.to_datetime(dates))
    return {str(y): float(v) for y, v in s.groupby(s.index.year).mean().items()}


def build_rth_arrays():
    m = load_nq("1min")
    ts = pd.to_datetime(m["ts_event"], utc=True).dt.tz_convert("America/New_York")
    m = m.assign(et=ts, d=ts.dt.date, mins=(ts.dt.hour * 60 + ts.dt.minute))
    rth = m[(m.mins >= 570) & (m.mins < 960)]            # 09:30..15:59 ET
    daily = load_nq("daily").set_index("date")
    daily.index = pd.to_datetime(daily.index).date
    keep_days = {d for d, r in daily.iterrows() if not r["roll_day"]}
    rth = rth[rth.d.isin(keep_days)]
    sessions = sorted(rth.d.unique())
    idx = {d: i for i, d in enumerate(sessions)}
    n, width = len(sessions), 390
    O = np.full((n, width), np.nan); H = np.full((n, width), np.nan)
    L = np.full((n, width), np.nan); C = np.full((n, width), np.nan)
    r_ = rth.mins.to_numpy() - 570
    s_ = rth.d.map(idx).to_numpy()
    O[s_, r_] = rth.Open.to_numpy(); H[s_, r_] = rth.High.to_numpy()
    L[s_, r_] = rth.Low.to_numpy(); C[s_, r_] = rth.Close.to_numpy()
    # forward-fill closes within session so boundary reads are defined
    Cff = pd.DataFrame(C).ffill(axis=1).to_numpy()
    aux = load_nq("aux").set_index("date")
    aux.index = pd.to_datetime(aux.index).date
    vix_prior = {d: aux["vix"].shift(1).get(d, np.nan) for d in sessions}
    return sessions, O, H, L, C, Cff, daily, aux, np.array(
        [vix_prior[d] for d in sessions])


def first_true(mask):
    """Index of first True per row, -1 if none."""
    any_ = mask.any(axis=1)
    ix = mask.argmax(axis=1)
    return np.where(any_, ix, -1)


# ---------------- F-NQ1 opening-range breakout (162) ----------------
def f_nq1(sessions, O, H, L, C, Cff, vix):
    rows = []
    dates = np.array([str(d) for d in sessions])
    for orw in (15, 30, 60):
        orh = np.nanmax(H[:, :orw], axis=1)
        orl = np.nanmin(L[:, :orw], axis=1)
        rng = orh - orl
        post_H, post_L, post_C = H[:, orw:], L[:, orw:], Cff[:, orw:]
        li = first_true(post_H > orh[:, None])
        si = first_true(post_L < orl[:, None])
        for dir_ in ("long", "short", "both"):
            for stop_m in (0.5, 1.0):
                for tgt in (1.0, 2.0, None):
                    rets, rdates = [], []
                    for k in range(len(sessions)):
                        if rng[k] <= 0 or np.isnan(rng[k]):
                            continue
                        cands = []
                        if dir_ in ("long", "both") and li[k] >= 0:
                            cands.append(("L", li[k]))
                        if dir_ in ("short", "both") and si[k] >= 0:
                            cands.append(("S", si[k]))
                        if not cands:
                            continue
                        side, ei = min(cands, key=lambda x: x[1])
                        entry = post_C[k, ei]
                        if np.isnan(entry) or entry <= 0:
                            continue
                        sd = stop_m * rng[k]
                        stop_px = entry - sd if side == "L" else entry + sd
                        tgt_px = (entry + tgt * sd if side == "L"
                                  else entry - tgt * sd) if tgt else None
                        h, l, c = post_H[k, ei + 1:], post_L[k, ei + 1:], post_C[k, ei + 1:]
                        if len(h) == 0:
                            rets.append((0.0 - COST_PTS) / entry)
                            rdates.append(dates[k])
                            continue
                        hit_s = first_true((l <= stop_px)[None, :])[0] if side == "L" \
                            else first_true((h >= stop_px)[None, :])[0]
                        hit_t = -1
                        if tgt_px is not None:
                            hit_t = first_true((h >= tgt_px)[None, :])[0] if side == "L" \
                                else first_true((l <= tgt_px)[None, :])[0]
                        if hit_s >= 0 and (hit_t < 0 or hit_s <= hit_t):
                            exit_px = stop_px          # stop-first convention
                        elif hit_t >= 0:
                            exit_px = tgt_px
                        else:
                            exit_px = c[-1] if len(c) else entry
                        pts = (exit_px - entry) if side == "L" else (entry - exit_px)
                        rets.append((pts - COST_PTS) / entry)
                        rdates.append(dates[k])
                    rets = np.array(rets)
                    for vname, vmask_fn in (("all", None), ("vix<20", lambda v: v < 20),
                                            ("vix>=20", lambda v: v >= 20)):
                        if vmask_fn is None:
                            rr, dd = rets, rdates
                        else:
                            sel = [i for i, d_ in enumerate(rdates)
                                   if vmask_fn(vix[list(dates).index(d_)])]
                            rr = rets[sel]; dd = [rdates[i] for i in sel]
                        rows.append(row("nq", "F-NQ1_orb",
                                        f"or{orw}|{dir_}|stop{stop_m}|tgt{tgt}|{vname}",
                                        rr, len(rr) / max(len(sessions), 1) if vmask_fn is None
                                        else len(rr) / max(sum(1 for x in vix if vmask_fn(x)), 1),
                                        NQ_CAPACITY, years=per_year(dd, rr) if len(rr) else None))
    return rows, 162


# ---------------- F-NQ2 first-hour momentum/fade (36) ----------------
def f_nq2(sessions, O, Cff, vix):
    dates = np.array([str(d) for d in sessions])
    o930 = O[:, 0]; c1030 = Cff[:, 60]; c1200 = Cff[:, 150]; c1555 = Cff[:, 385]
    fh = c1030 / o930 - 1
    rows = []
    for thr in (0.003, 0.005, 0.010):
        for mode in ("follow", "fade"):
            for exit_, px in (("1200", c1200), ("1555", c1555)):
                sign = np.sign(fh) if mode == "follow" else -np.sign(fh)
                mask = np.abs(fh) >= thr
                rets = (sign * (px / c1030 - 1) - COST_PTS / c1030)[mask]
                dd = dates[mask]
                for vname, vm in (("all", np.ones_like(vix, bool)),
                                  ("vix<20", vix < 20), ("vix>=20", vix >= 20)):
                    rr = rets[vm[mask]]
                    rows.append(row("nq", "F-NQ2_firsthour",
                                    f"thr{thr}|{mode}|exit{exit_}|{vname}",
                                    rr, len(rr) / max(vm.sum(), 1), NQ_CAPACITY,
                                    years=per_year(dd[vm[mask]], rr) if len(rr) else None))
    return rows, 36


# ---------------- F-NQ3 overnight-gap fade/follow (60) ----------------
def f_nq3(sessions, O, Cff, daily, vix):
    dates = [str(d) for d in sessions]
    d2i = {d: i for i, d in enumerate(sessions)}
    gap, o930, c1030, c1555, dts = [], [], [], [], []
    for d, r in daily.iterrows():
        if d not in d2i or r["roll_day"] or not np.isfinite(r["prior_rth_close"]):
            continue
        i = d2i[d]
        gap.append(r["px_0925"] / r["prior_rth_close"] - 1)
        o930.append(O[i, 0]); c1030.append(Cff[i, 60]); c1555.append(Cff[i, 385])
        dts.append(dates[i])
    gap = np.array(gap); o930 = np.array(o930)
    c1030 = np.array(c1030); c1555 = np.array(c1555)
    vixv = np.array([vix[d2i[pd.to_datetime(d).date()]] for d in dts])
    qs = np.nanpercentile(gap, [20, 40, 60, 80])
    bucket = np.digitize(gap, qs)          # 0..4
    rows = []
    for b in range(5):
        m0 = bucket == b
        for mode in ("fade", "follow"):
            sign = -np.sign(gap) if mode == "fade" else np.sign(gap)
            for exit_, px in (("1030", c1030), ("1555", c1555)):
                rets_all = sign * (px / o930 - 1) - COST_PTS / o930
                for vname, vm in (("all", np.ones_like(vixv, bool)),
                                  ("vix<20", vixv < 20), ("vix>=20", vixv >= 20)):
                    m = m0 & vm
                    rr = rets_all[m]
                    if vname == "all" or b in (0, 4):   # bound cells to plan's 60
                        rows.append(row("nq", "F-NQ3_gap",
                                        f"q{b}|{mode}|exit{exit_}|{vname}",
                                        rr, len(rr) / max(vm.sum(), 1), NQ_CAPACITY,
                                        years=per_year(np.array(dts)[m], rr) if m.any() else None))
    return rows, 60


# ---------------- F-NQ4 time-of-day segments (60) ----------------
def f_nq4(sessions, O, Cff, daily):
    dates = np.array([str(d) for d in sessions])
    dow = np.array([d.weekday() for d in sessions])
    segs = {"on_1800_0930": None,                       # from daily overnight_ret
            "0930_1030": (0, 60), "1030_1400": (60, 270),
            "1400_1600": (270, 389), "1530_1600": (360, 389)}
    d2i = {d: i for i, d in enumerate(sessions)}
    on = np.full(len(sessions), np.nan)
    for d, r in daily.iterrows():
        if d in d2i and np.isfinite(r.get("overnight_ret", np.nan)):
            on[d2i[d]] = r["overnight_ret"]
    rows = []
    for sname, span in segs.items():
        if span is None:
            seg_ret = on
        else:
            a, b = span
            seg_ret = Cff[:, b] / (O[:, a] if a == 0 else Cff[:, a]) - 1
        for dir_ in ("long", "short"):
            base = seg_ret if dir_ == "long" else -seg_ret
            base = base - COST_PTS / np.nanmean(Cff[:, 60])
            for dw, dname in [(None, "all")] + [(i, n) for i, n in
                                                enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])]:
                m = np.isfinite(base) if dw is None else (np.isfinite(base) & (dow == dw))
                rr = base[m]
                rows.append(row("nq", "F-NQ4_timeofday", f"{sname}|{dir_}|{dname}",
                                rr, len(rr) / max(m.sum() if dw is None else (dow == dw).sum(), 1),
                                NQ_CAPACITY, years=per_year(dates[m], rr) if m.any() else None))
    return rows, 60


# ---------------- F-NQ5 VIX-regime prior-day pattern (18) ----------------
def f_nq5(sessions, O, Cff, vix):
    dates = np.array([str(d) for d in sessions])
    oc = Cff[:, 389] / O[:, 0] - 1
    prior = np.roll(oc, 1); prior[0] = np.nan
    t1, t2 = np.nanpercentile(vix, [33.3, 66.7])
    vterc = np.digitize(vix, [t1, t2])
    psign = np.where(prior > 0.002, 2, np.where(prior < -0.002, 0, 1))
    rows = []
    for vt in range(3):
        for ps, pname in ((0, "prior_dn"), (1, "prior_flat"), (2, "prior_up")):
            m = (vterc == vt) & (psign == ps) & np.isfinite(oc)
            for dir_ in ("long", "short"):
                rr = (oc[m] if dir_ == "long" else -oc[m]) - COST_PTS / np.nanmean(O[:, 0])
                rows.append(row("nq", "F-NQ5_vixregime",
                                f"vixT{vt}|{pname}|{dir_}", rr,
                                len(rr) / max(m.sum(), 1), NQ_CAPACITY,
                                years=per_year(dates[m], rr) if m.any() else None))
    return rows, 18


# ---------------- F-NQ6 Globex overnight segments (12) ----------------
def f_nq6(daily, aux):
    d = daily.copy()
    d.index = pd.to_datetime(d.index)
    a = aux.copy(); a.index = pd.to_datetime(a.index)
    j = d.join(a[["nikkei", "dax"]].pct_change().shift(1), how="left")
    on = j["overnight_ret"].to_numpy()
    dates = j.index.astype(str).to_numpy()
    conds = {"none": np.isfinite(on),
             "nikkei_up": (j["nikkei"] > 0).to_numpy() & np.isfinite(on),
             "dax_up": (j["dax"] > 0).to_numpy() & np.isfinite(on)}
    rows = []
    for seg in ("on_full",):    # px granular segments folded into F-NQ4; daily proxy here
        for cname, m in conds.items():
            for dir_ in ("long", "short"):
                rr = (on[m] if dir_ == "long" else -on[m]) - COST_PTS / 15000.0
                rows.append(row("nq", "F-NQ6_globex", f"{seg}|{cname}|{dir_}",
                                rr, len(rr) / max(np.isfinite(on).sum(), 1),
                                NQ_CAPACITY, years=per_year(dates[m], rr) if m.any() else None))
    return rows, 12


def main():
    sessions, O, H, L, C, Cff, daily, aux, vix = build_rth_arrays()
    print(f"[M2] {len(sessions)} mining sessions "
          f"({sessions[0]} -> {sessions[-1]}), vix coverage "
          f"{np.isfinite(vix).mean():.0%}")
    all_rows = []
    for fn, args, cells in ((f_nq1, (sessions, O, H, L, C, Cff, vix), 162),
                            (f_nq2, (sessions, O, Cff, vix), 36),
                            (f_nq3, (sessions, O, Cff, daily, vix), 60),
                            (f_nq4, (sessions, O, Cff, daily), 60),
                            (f_nq5, (sessions, O, Cff, vix), 18),
                            (f_nq6, (daily, aux), 12)):
        rows, n_cells = fn(*args)
        record_mined("nq", rows[0]["family"] if rows else fn.__name__, n_cells)
        all_rows += rows
        print(f"  {fn.__name__}: {len(rows)} rows")
    path = write_session("m2_nq", all_rows)
    ranked = sorted([r for r in all_rows if r.get("n", 0) >= 40],
                    key=lambda r: -r["net_pct_day"])
    print(f"[M2] {len(all_rows)} rows -> {path}")
    for r in ranked[:10]:
        yrs = r.get("per_year") or {}
        neg = sum(1 for v in yrs.values() if v < 0)
        print(f"  {r['family']:<18} {r['config']:<38} n={r['n']:>5} "
              f"net/day={r['net_pct_day']:+.5f} p5={r['tail_p5']:+.4f} "
              f"neg_yrs={neg}/{len(yrs)}")


if __name__ == "__main__":
    main()
