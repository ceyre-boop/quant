#!/usr/bin/env python3
"""HYP-098 Step 1 — MINING pass over the FVG x corridor grid (2018-01→2024-06 only).

STAMP: MINING — not evidence. 720 cells: 3 entry families x 5 corridor conditions
x 4 killzones x 6 managements x 2 FVG sizes. Every cell counted in mined_n.json.
Entries at bar CLOSE (causal); costs MNQ 0.87 pts/RT; returns are fractions of
notional. Run: python3 -m research.fvg_corridor.mine
"""
from collections import defaultdict
from datetime import time as dtime

import numpy as np
import pandas as pd

from research.yield_frontier._lib import record_mined
from research.yield_frontier.frictions import NQ_RT_COST_PTS
from research.yield_frontier.yield_board import row, write_session
from . import core

COST = NQ_RT_COST_PTS["MNQ"]
RETEST_K = 100          # bars (~8.3h) to wait for retest/mitigation
D2 = core.DEPTHS[1]     # corridor condition depth (50)


def flat_index(df):
    """For each bar, index of the next bar at/after 15:55 ET (session flat)."""
    et = df.index.tz_convert("America/New_York")
    is_flat = np.array([(t.hour, t.minute) >= (15, 55) and (t.hour, t.minute) < (16, 0)
                        for t in et.time])
    n = len(df)
    nxt = np.full(n, n - 1, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if is_flat[i]:
            last = i
        nxt[i] = last
    return nxt


def build_events(df, atr, feats, min_size):
    """Per family: arrays of (entry_idx, direction, gap_size). Entry at close of entry_idx."""
    idx, side, top, bot = core.detect_fvgs(df, atr, min_size)
    h, l, c = df["h"].to_numpy(), df["l"].to_numpy(), df["c"].to_numpy()
    n = len(df)
    fams = {"F2_continuation": ([], [], [])}
    for k in range(len(idx)):
        i, s_, t_, b_ = idx[k], side[k], top[k], bot[k]
        fams["F2_continuation"][0].append(i)
        fams["F2_continuation"][1].append(s_)
        fams["F2_continuation"][2].append(t_ - b_)
    # retest + mitigation need a forward scan per FVG (bounded)
    fams["F1_retest"] = ([], [], [])
    fams["F3_mitigation_fade"] = ([], [], [])
    for k in range(len(idx)):
        i, s_, t_, b_ = idx[k], side[k], top[k], bot[k]
        end = min(i + RETEST_K, n - 1)
        if end <= i + 1:
            continue
        sl_l, sl_c = l[i + 1:end], c[i + 1:end]
        sl_h = h[i + 1:end]
        if s_ == 1:
            touch = sl_l <= t_
            fill = sl_c < b_
        else:
            touch = sl_h >= b_
            fill = sl_c > t_
        j = int(np.argmax(touch)) if touch.any() else -1
        if j >= 0:
            fams["F1_retest"][0].append(i + 1 + j)
            fams["F1_retest"][1].append(s_)
            fams["F1_retest"][2].append(t_ - b_)
        jf = int(np.argmax(fill)) if fill.any() else -1
        if jf >= 0:
            fams["F3_mitigation_fade"][0].append(i + 1 + jf)
            fams["F3_mitigation_fade"][1].append(-s_)   # fade the failed gap
            fams["F3_mitigation_fade"][2].append(t_ - b_)
    return {f: (np.array(a), np.array(b_), np.array(g)) for f, (a, b_, g) in fams.items()}


def corridor_condition_mask(feats, entries, dirs, cond):
    pos = feats[D2]["pos"][entries]
    slope = feats[D2]["slope_sign"][entries]
    if cond == "none":
        return np.ones(len(entries), bool)
    if cond == "inside_d2":
        return (pos >= 0.2) & (pos <= 0.8)
    if cond == "beyond_with":
        return np.where(dirs == 1, pos > 1.0, pos < 0.0)
    if cond == "beyond_against":
        return np.where(dirs == 1, pos < 0.0, pos > 1.0)
    if cond == "slope_agree":
        return slope == dirs
    raise ValueError(cond)


def walk(df, entries, dirs, gaps, stop_mult, target_r, flat_ix):
    """Vector-ish walk: per event, stop/target/flat exit. Returns fractional rets."""
    h, l, c = df["h"].to_numpy(), df["l"].to_numpy(), df["c"].to_numpy()
    rets = np.empty(len(entries))
    for k in range(len(entries)):
        e, d_, g = entries[k], dirs[k], gaps[k]
        entry = c[e]
        sd = stop_mult * g
        stop_px = entry - d_ * sd
        tgt_px = entry + d_ * target_r * sd if target_r else None
        fi = flat_ix[e] if flat_ix[e] > e else min(e + 288, len(c) - 1)
        hh, ll = h[e + 1:fi + 1], l[e + 1:fi + 1]
        if d_ == 1:
            hit_s = np.argmax(ll <= stop_px) if (ll <= stop_px).any() else -1
            hit_t = (np.argmax(hh >= tgt_px) if (hh >= tgt_px).any() else -1) \
                if tgt_px else -1
        else:
            hit_s = np.argmax(hh >= stop_px) if (hh >= stop_px).any() else -1
            hit_t = (np.argmax(ll <= tgt_px) if (ll <= tgt_px).any() else -1) \
                if tgt_px else -1
        if hit_s >= 0 and (hit_t < 0 or hit_s <= hit_t):
            exit_px = stop_px                       # stop-first convention
        elif hit_t >= 0:
            exit_px = tgt_px
        else:
            exit_px = c[fi]
        rets[k] = d_ * (exit_px - entry) / entry - COST / entry
    return rets


def main():
    df = core.load_nq_5min("mining")
    atr = core.causal_atr(df)
    feats = core.corridor_features(df)
    flat_ix = flat_index(df)
    dates = pd.Series(df.index.tz_convert("America/New_York").date)
    n_days = dates.nunique()
    kz_masks = {z: core.killzone_mask(df, z) for z in
                ("LONDON", "NY_OPEN", "NY_PM", "ALL")}
    rows = []
    total_cells = 0
    for min_size in (0.5, 1.5):
        fams = build_events(df, atr, feats, min_size)
        for fam, (entries, dirs, gaps) in fams.items():
            ok = gaps > 0
            entries, dirs, gaps = entries[ok], dirs[ok], gaps[ok]
            for cond in ("none", "inside_d2", "beyond_with", "beyond_against",
                         "slope_agree"):
                cmask = corridor_condition_mask(feats, entries, dirs, cond)
                cmask &= np.isfinite(feats[D2]["pos"][entries]) | (cond == "none")
                for kz, kmask in kz_masks.items():
                    m = cmask & kmask[entries]
                    e2, d2_, g2 = entries[m], dirs[m], gaps[m]
                    for stop_mult in (0.5, 1.0):
                        rets_by_tgt = {}
                        for tgt in (1.0, 2.0, None):
                            total_cells += 1
                            if len(e2) < 30:
                                continue
                            r = walk(df, e2, d2_, g2, stop_mult, tgt, flat_ix)
                            yrs = pd.Series(r, index=pd.to_datetime(
                                df.index[e2])).groupby(
                                lambda ix: ix.year).mean().round(5).to_dict()
                            rows.append(row(
                                "nq_fvg", fam,
                                f"sz{min_size}|{cond}|{kz}|stop{stop_mult}|tgt{tgt}",
                                r, len(r) / n_days, 50_000_000,
                                years={str(k): v for k, v in yrs.items()}))
    record_mined("nq_fvg", "HYP098_grid_round1", total_cells)
    path = write_session("fvg_corridor_m1", rows)
    ranked = sorted([x for x in rows if x.get("n", 0) >= 40],
                    key=lambda x: -x["net_pct_day"])
    print(f"[M] {total_cells} cells, {len(rows)} populated, {len(ranked)} rankable "
          f"-> {path}")
    for x in ranked[:15]:
        yrs = x.get("per_year") or {}
        neg = sum(1 for v in yrs.values() if v < 0)
        print(f"  {x['family']:<20} {x['config']:<44} n={x['n']:>6} "
              f"net/day={x['net_pct_day']:+.5f} med={x['median_pct']:+.5f} "
              f"neg_yrs={neg}/{len(yrs)}")


if __name__ == "__main__":
    main()
