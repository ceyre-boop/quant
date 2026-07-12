#!/usr/bin/env python3
"""M3 — SPY options premium mining, quote dates 2022-01→2023-09 (~185 cells).

STAMP: MINING. Chains via holdout_guard.chain_files() (fenced ≤ 2023-09-30).
Fills at mid ∓ 0.5×half-spread + $0.65/leg/side (k-sensitivity disclosed, not
extra trials). Returns are % of collateral; net %/day treats capital as one
continuously-recycled slot (mean event return ÷ mean days held).
Daily cadence + fixed-moneyness strikes — structurally distinct from the gated
VRP-001-v2 spec (weekly Monday, 1σ-RV20 strikes), which is never run.
Run: python3 -m research.yield_frontier.m3_options
"""
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ._lib import REPO, record_mined
from .frictions import OPT_COMMISSION_PER_LEG_SIDE as COMM
from .frictions import option_fill
from .holdout_guard import chain_files, load_nq
from .yield_board import row, write_session

K = 0.5   # headline fill; k∈{0.25,1.0} sensitivity disclosed in report

# NOTE: chain cache has delta/iv columns entirely NaN (backfill never filled them)
# -> strike_rule is fixed OTM moneyness. Distinct from VRP-001-v2 (1sigma-RV20,
# weekly Monday cadence) on both axes; collision test asserts this.
# cache reality: VRP-era backfill holds ~30-DTE monthly chains ONLY (dte 1/7/14
# have no expiries within tolerance) -> OP1 runs at dte 30; reduction disclosed.
OP1_GRID = [{"cadence": "daily", "strike_rule": "moneyness", "otm": d, "width": w,
             "dte": 30, "mgmt": m}
            for d in (0.02, 0.04, 0.06) for w in (1, 2, 5)
            for m in ("expiry", "pt50", "next_eod")]
OP2_GRID = [{"cadence": "daily", "strike_rule": "moneyness", "otm": d, "width": w,
             "dte": t, "mgmt": m}
            for d in (0.02, 0.04) for w in (1, 2, 5)
            for t in (7, 14, 30) for m in ("expiry", "pt50", "dte21")]


class Chains:
    def __init__(self):
        self.by_quote = defaultdict(dict)
        for fp in chain_files():
            q, e = fp.name.replace(".parquet", "").split("_")
            self.by_quote[q][e] = fp
        self.quotes = sorted(self.by_quote)
        self._cache = {}
        self._parity = {}   # spot cache is dividend-ADJUSTED (5%+ off actual) —
                            # derive spot from the chains themselves via put-call parity

    def load(self, q, e):
        key = (q, e)
        if key not in self._cache:
            fp = self.by_quote.get(q, {}).get(e)
            df = pd.read_parquet(fp) if fp else None
            if df is not None and (df.empty or not np.isfinite(df["put_mid"]).any()):
                df = None
            self._cache[key] = df
            if len(self._cache) > 4000:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def pick_expiry(self, q, target_dte, tol):
        qd = date.fromisoformat(q)
        best, bdiff = None, 99
        for e in self.by_quote[q]:
            dte = (date.fromisoformat(e) - qd).days
            if target_dte == 1 and dte < 1:
                continue
            diff = abs(dte - target_dte) if target_dte > 1 else (dte - 1)
            if 0 <= diff < bdiff and diff <= tol:
                best, bdiff = e, diff
        return best

    def parity_spot(self, q):
        if q in self._parity:
            return self._parity[q]
        S = None
        for e in sorted(self.by_quote.get(q, {}),
                        key=lambda e_: (date.fromisoformat(e_) - date.fromisoformat(q)).days):
            ch = self.load(q, e)
            if ch is None:
                continue
            d = ch[np.isfinite(ch.call_mid) & np.isfinite(ch.put_mid) &
                   (ch.call_mid > 0) & (ch.put_mid > 0)]
            if len(d) < 3:
                continue
            r = d.iloc[(d.call_mid - d.put_mid).abs().argmin()]
            S = float(r.strike + r.call_mid - r.put_mid)
            break
        self._parity[q] = S
        return S


def leg(df, otm, side, spot):
    d = df[np.isfinite(df[f"{side}_mid"]) &
           (df[f"{side}_ask"] >= df[f"{side}_bid"]) & (df[f"{side}_mid"] > 0)]
    if d.empty or spot is None:
        return None
    target = spot * (1 - otm) if side == "put" else spot * (1 + otm)
    r = d.iloc[(d["strike"] - target).abs().argmin()]
    if abs(r["strike"] - target) > 0.02 * spot:   # nothing near the target strike
        return None
    return r


def strike_row(df, strike, side, tol=2.6):
    d = df[np.isfinite(df[f"{side}_mid"]) & (df[f"{side}_mid"] > 0)]
    if d.empty:
        return None
    r = d.iloc[(d["strike"] - strike).abs().argmin()]
    if abs(r["strike"] - strike) > tol:
        return None
    return r


def spread_value(ch, s_short, s_long, side, k):
    """Cost to CLOSE a short vertical (buy back short leg, sell long leg)."""
    a, b = strike_row(ch, s_short, side), strike_row(ch, s_long, side)
    if a is None or b is None:
        return None
    close_short = option_fill(a[f"{side}_mid"], (a[f"{side}_ask"] - a[f"{side}_bid"]) / 2, k, selling=False)
    close_long = option_fill(b[f"{side}_mid"], (b[f"{side}_ask"] - b[f"{side}_bid"]) / 2, k, selling=True)
    return close_short - close_long


def run_verticals(C, grid, family, both_sides=False):
    events = defaultdict(list)
    for q in C.quotes:
        for cell in grid:
            e = C.pick_expiry(q, cell["dte"], tol={1: 4, 7: 3, 14: 5, 30: 7}[cell["dte"]])
            if e is None:
                continue
            ch = C.load(q, e)
            if ch is None:
                continue
            spot = C.parity_spot(q)
            legs = []
            for side in (("put", "call") if both_sides else ("put",)):
                sl = leg(ch, cell["otm"], side, spot)
                if sl is None:
                    continue
                width = cell["width"]
                s_short = sl["strike"]
                target_long = s_short - width if side == "put" else s_short + width
                lr = strike_row(ch, target_long, side)
                if lr is None or lr["strike"] == s_short:
                    continue
                s_long = lr["strike"]
                cr = (option_fill(sl[f"{side}_mid"], (sl[f"{side}_ask"] - sl[f"{side}_bid"]) / 2, K, True)
                      - option_fill(lr[f"{side}_mid"], (lr[f"{side}_ask"] - lr[f"{side}_bid"]) / 2, K, False))
                legs.append((side, s_short, s_long, cr, abs(s_long - s_short)))
            if not legs or (both_sides and len(legs) < 2):
                continue
            credit = sum(l[3] for l in legs) - 2 * len(legs) * COMM / 100.0
            width_actual = max(l[4] for l in legs)
            collateral = width_actual - sum(l[3] for l in legs)
            if credit <= 0 or collateral <= 0:
                continue
            # management walk
            exit_cost, days = None, None
            qd = date.fromisoformat(q)
            horizon = [(d_, (date.fromisoformat(d_) - qd).days) for d_ in C.quotes
                       if q < d_ <= e]
            if cell["mgmt"] in ("pt50", "next_eod", "dte21"):
                for d_, dd in horizon:
                    ch2 = C.load(d_, e)
                    if ch2 is None:
                        continue
                    vals = [spread_value(ch2, l[1], l[2], l[0], K) for l in legs]
                    if any(v is None for v in vals):
                        continue
                    val = sum(vals)
                    if cell["mgmt"] == "next_eod":
                        exit_cost, days = val, dd
                        break
                    if cell["mgmt"] == "pt50" and val <= 0.5 * sum(l[3] for l in legs):
                        exit_cost, days = val, dd
                        break
                    dte_left = (date.fromisoformat(e) - date.fromisoformat(d_)).days
                    if cell["mgmt"] == "dte21" and dte_left <= 21:
                        exit_cost, days = val, dd
                        break
            if exit_cost is None:
                # settle at the FINAL available chain mark <= expiry (no external
                # price source — the adjusted spot cache mispriced settlement)
                for d_, dd in reversed(horizon):
                    ch2 = C.load(d_, e)
                    if ch2 is None:
                        continue
                    vals = [spread_value(ch2, l[1], l[2], l[0], K) for l in legs]
                    if any(v is None for v in vals):
                        continue
                    exit_cost, days = sum(vals), dd
                    break
                if exit_cost is None:
                    continue
            pnl = credit - exit_cost - 2 * len(legs) * COMM / 100.0
            key = tuple(sorted(cell.items()))
            events[key].append((pnl / collateral, max(days, 1), q))
    rows = []
    for key, evs in events.items():
        cell = dict(key)
        rets = np.array([e[0] for e in evs])
        hold = float(np.mean([e[1] for e in evs]))
        yrs = {}
        s = pd.Series(rets, index=pd.to_datetime([e[2] for e in evs]))
        yrs = {str(y): float(v) for y, v in s.groupby(s.index.year).mean().items()}
        rows.append(row("options", family,
                        f"otm{cell['otm']}|w{cell['width']}|dte{cell['dte']}|{cell['mgmt']}",
                        rets, 1.0 / hold, 5_000_000, years=yrs))
    return rows


def run_lottery(C):
    events = defaultdict(list)
    for q in C.quotes:
        for dlt in (0.05, 0.08):
            for t in (7, 14):
                e = C.pick_expiry(q, t, tol=5)
                if e is None:
                    continue
                ch = C.load(q, e)
                if ch is None:
                    continue
                for side in ("call", "put"):
                    sl = leg(ch, dlt, side, C.parity_spot(q))
                    if sl is None:
                        continue
                    debit = option_fill(sl[f"{side}_mid"],
                                        (sl[f"{side}_ask"] - sl[f"{side}_bid"]) / 2,
                                        K, selling=False) + COMM / 100.0
                    if debit <= 0.01:
                        continue
                    qd = date.fromisoformat(q)
                    sell_px, days = None, None
                    for d_ in reversed([x for x in C.quotes if q < x <= e]):
                        ch2 = C.load(d_, e)
                        if ch2 is None:
                            continue
                        r2 = strike_row(ch2, sl["strike"], side)
                        if r2 is None:
                            continue
                        sell_px = option_fill(r2[f"{side}_mid"],
                                              (r2[f"{side}_ask"] - r2[f"{side}_bid"]) / 2,
                                              K, selling=True)
                        days = max((date.fromisoformat(d_) - qd).days, 1)
                        break
                    if sell_px is None:
                        continue
                    events[(dlt, t, side)].append(((sell_px - debit) / debit, days, q))
    rows = []
    for (dlt, t, side), evs in events.items():
        rets = np.array([e[0] for e in evs])
        hold = float(np.mean([e[1] for e in evs]))
        rows.append(row("options", "F-OP5_lottery", f"d{dlt}|dte{t}|{side}",
                        rets, 1.0 / hold, 500_000))
    return rows


def main():
    C = Chains()
    print(f"[M3] {len(C.quotes)} quote dates ({C.quotes[0]} -> {C.quotes[-1]})")
    all_rows = []
    r1 = run_verticals(C, OP1_GRID, "F-OP1_putspread")
    record_mined("options", "F-OP1_putspread", 81); all_rows += r1
    print(f"  OP1: {len(r1)} rows")
    r2 = run_verticals(C, OP2_GRID, "F-OP2_condor", both_sides=True)
    record_mined("options", "F-OP2_condor", 54); all_rows += r2
    print(f"  OP2: {len(r2)} rows")
    r5 = run_lottery(C)
    record_mined("options", "F-OP5_lottery", 8); all_rows += r5
    print(f"  OP5: {len(r5)} rows")
    # F-OP4 VIX overlay on top-10 premium cells
    aux = load_nq("aux").set_index("date")
    aux.index = pd.to_datetime(aux.index).strftime("%Y-%m-%d")
    record_mined("options", "F-OP4_vix_overlay", 30)
    print("  OP4: overlay computed at synthesis from stored events (30 cells counted)")
    record_mined("options", "F-OP3_strangle", 12)
    print("  OP3: strangles SKIPPED (undefined-risk margin model too coarse to rank "
          "honestly) — 12 cells counted as mined, no rows emitted; disclosed")
    path = write_session("m3_options", all_rows)
    ranked = sorted([r for r in all_rows if r.get("n", 0) >= 40],
                    key=lambda r: -r["net_pct_day"])
    print(f"[M3] {len(all_rows)} rows -> {path}")
    for r in ranked[:8]:
        print(f"  {r['family']:<18} {r['config']:<28} n={r['n']:>4} "
              f"net/day={r['net_pct_day']:+.5f} med={r['median_pct']:+.4f} "
              f"p5={r['tail_p5']:+.3f}")


if __name__ == "__main__":
    main()
