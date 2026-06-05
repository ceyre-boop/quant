#!/usr/bin/env python3
"""Edge test — the OVERNIGHT EFFECT on QQQ (first non-FX, orthogonal-source hypothesis).

Decades-documented equity-index anomaly (Cooper/Cliff/Gulen; Lou/Polk/Skouras 'tug of war';
Branch & Ma): index gains accrue close->open (overnight) while open->close (intraday) is flat/
negative. Tested on QQQ (Nasdaq-100 ETF — the literature's instrument; NQ futures RTH data is
unavailable here and its 24h structure weakens the gap). FULLY COSTED — costs are the #1 killer.

Same discipline as the FX research: both sides reported, buy-and-hold baseline, 10k permutation,
no threshold tuning. Read-only: writes data/research/edge_research_overnight.json + appends the ledger.

VERDICT VALID_EDGE only if: overnight positive NET of 1.5x costs AND present in 2023-2025 walk-forward
AND permutation p < 0.05. Else DECAYED (older-only) or NOT_SIGNIFICANT.

Usage:  python3 scripts/edge_research_overnight_qqq.py [--perms 10000] [--cost-bp 1.5]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.getLogger("yfinance").setLevel(logging.ERROR)

OUT = ROOT / "data" / "research" / "edge_research_overnight.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CITATION = "Cooper/Cliff/Gulen; Lou, Polk & Skouras 'A Tug of War' (2019); Branch & Ma — overnight/intraday return decomposition"
ANN = np.sqrt(252.0)


def _load_qqq():
    """QQQ daily, corporate-action ADJUSTED (auto_adjust=True → Open/Close adjusted consistently,
    so close->open is clean of split/dividend gaps). yfinance gives full history."""
    import yfinance as yf
    h = yf.Ticker("QQQ").history(period="max", interval="1d", auto_adjust=True)
    if h is None or len(h) < 500:
        raise SystemExit("FATAL: QQQ daily history unavailable/too short — refusing to test on thin data.")
    return h


def _sharpe(x):
    sd = float(np.std(x))
    return round(float(np.mean(x) / sd * ANN), 3) if sd > 0 else None


def _tstat(x):
    sd = float(np.std(x, ddof=1))
    return round(float(np.mean(x) / (sd / np.sqrt(len(x)))), 2) if sd > 0 and len(x) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bp", type=float, default=1.5, help="base round-trip cost in basis points")
    args = ap.parse_args()
    base_cost = args.cost_bp / 1e4

    h = _load_qqq()
    O = h["Open"].to_numpy(float)
    C = h["Close"].to_numpy(float)
    dates = [d.date().isoformat() for d in h.index]
    years = np.array([int(d[:4]) for d in dates[1:]])    # aligned to t>=1

    on = O[1:] / C[:-1] - 1.0          # overnight: today open / prior close
    intra = C[1:] / O[1:] - 1.0        # intraday: today close / today open
    bh = C[1:] / C[:-1] - 1.0          # buy & hold daily
    N = len(on)

    # ── TEST 1 — existence ────────────────────────────────────────────────────
    def _cum(x):
        return round(float(np.prod(1 + x) - 1), 4)
    test1 = {
        "n_days": N, "window": f"{dates[1]}..{dates[-1]}",
        "overnight": {"mean_bp": round(float(on.mean()) * 1e4, 3), "cum_return": _cum(on),
                      "ann_sharpe": _sharpe(on), "t_stat": _tstat(on)},
        "intraday": {"mean_bp": round(float(intra.mean()) * 1e4, 3), "cum_return": _cum(intra),
                     "ann_sharpe": _sharpe(intra), "t_stat": _tstat(intra)},
        "buy_hold": {"mean_bp": round(float(bh.mean()) * 1e4, 3), "cum_return": _cum(bh),
                     "ann_sharpe": _sharpe(bh)},
        "pre_registered_overnight_gt0": bool(on.mean() > 0),
        "pre_registered_intraday_le0": bool(intra.mean() <= 0),
    }

    # ── TEST 2 — net of costs ─────────────────────────────────────────────────
    test2 = {"base_cost_bp": args.cost_bp,
             "note": "overnight-only = 1 round trip/day (buy MOC, sell MOO); cost applied to every day"}
    for mult in (1.0, 1.5, 2.0):
        net = on - base_cost * mult
        test2[f"{mult}x"] = {"ann_sharpe": _sharpe(net),
                             "ann_return_pct": round(float(np.mean(net) * 252) * 100, 2),
                             "cost_drag_pct_yr": round(base_cost * mult * 252 * 100, 2)}

    # ── TEST 3 — walk-forward (per-year overnight Sharpe, net 1x cost) ────────
    net1 = on - base_cost
    test3 = {}
    for y in sorted(set(years.tolist())):
        m = years == y
        if m.sum() >= 20:
            test3[str(y)] = {"n": int(m.sum()), "overnight_net_sharpe": _sharpe(net1[m]),
                             "mean_bp": round(float(on[m].mean()) * 1e4, 2)}
    recent = [test3[str(y)]["overnight_net_sharpe"] for y in (2023, 2024, 2025)
              if str(y) in test3 and test3[str(y)]["overnight_net_sharpe"] is not None]
    recent_alive = bool(recent) and all(s > 0 for s in recent)

    # ── TEST 4 — reversal (tug of war) ────────────────────────────────────────
    corr = float(np.corrcoef(on, intra)[0, 1])
    slope = float(np.polyfit(on, intra, 1)[0])
    test4 = {"overnight_intraday_corr": round(corr, 3), "intraday_on_overnight_slope": round(slope, 3),
             "interpretation": ("positive overnight predicts negative intraday (reversal present)"
                                if slope < -0.1 else "no strong reversal")}

    # ── TEST 5 — permutation (segment-label shuffle) ──────────────────────────
    pool = np.concatenate([on, intra])
    observed = float(on.mean() - intra.mean())
    rng = np.random.default_rng(args.seed)
    n_tot = len(pool)
    null = np.empty(args.perms)
    for i in range(args.perms):
        idx = rng.permutation(n_tot)[:N]
        sel = np.zeros(n_tot, dtype=bool); sel[idx] = True
        null[i] = pool[sel].mean() - pool[~sel].mean()
    p_value = float((np.sum(null >= observed) + 1) / (args.perms + 1))
    test5 = {"observed_overnight_minus_intraday_bp": round(observed * 1e4, 3), "p_value": p_value}

    # ── VERDICT ───────────────────────────────────────────────────────────────
    survives_cost = (test2["1.5x"]["ann_sharpe"] or -1) > 0
    significant = p_value < 0.05
    if survives_cost and recent_alive and significant:
        verdict = "VALID_EDGE"
    elif significant and not recent_alive:
        verdict = "DECAYED"
    elif significant and not survives_cost:
        verdict = "DECAYED"   # real but not tradeable net of costs
    else:
        verdict = "NOT_SIGNIFICANT"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "id": "OVERNIGHT-QQQ",
        "instrument": "QQQ (Nasdaq-100 ETF), corporate-action adjusted (yfinance auto_adjust)",
        "citation": CITATION,
        "verdict": verdict,
        "gates": {"survives_1.5x_cost": survives_cost, "recent_walkforward_alive_2023_25": recent_alive,
                  "permutation_significant": significant},
        "test1_existence": test1, "test2_costs": test2, "test3_walkforward": test3,
        "test4_reversal": test4, "test5_permutation": test5,
        "caveat": ("QQQ tests the PHENOMENON on the Nasdaq source. NOT yet a tradeable NQ strategy — "
                   "live capture needs a CME futures feed + broker, and futures' 24h Globex structure "
                   "may weaken the RTH overnight gap. A VALID_EDGE here is a candidate second edge that "
                   "must clear its own execution validation downstream."),
        "meta": {"perms": args.perms, "seed": args.seed,
                 "cost_rationale": f"{args.cost_bp}bp round trip = half-spread + slippage/side x2 + "
                                   f"negligible ETF commission (QQQ is among the most liquid ETFs)."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # Ledger append (idempotent).
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    led = [e for e in led if e.get("id") != "OVERNIGHT-QQQ"]
    led.append({
        "id": "OVERNIGHT-QQQ", "name": "Overnight effect on QQQ (orthogonal non-FX source)",
        "status": verdict, "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "result": f"overnight {test1['overnight']['mean_bp']}bp/day (cum {test1['overnight']['cum_return']:.0%}, "
                  f"Sharpe {test1['overnight']['ann_sharpe']}) vs intraday {test1['intraday']['mean_bp']}bp; "
                  f"net-1.5x Sharpe {test2['1.5x']['ann_sharpe']}; perm p={p_value}; recent_alive={recent_alive}",
        "p_value": p_value,
        "methodology_note": ("QQQ daily adj OHLC, overnight=open/prior-close, intraday=close/open; fully "
                             "costed (1x/1.5x/2x); per-year walk-forward; segment-label permutation. "
                             "Grounded in " + CITATION + ". CAVEAT: phenomenon on Nasdaq source, not yet a "
                             "tradeable NQ strategy (needs futures feed/broker; 24h structure may weaken)."),
    })
    LEDGER.write_text(json.dumps(led, indent=2))

    # Print.
    print(f"\n{'='*78}\n  OVERNIGHT EFFECT — QQQ adj daily ({test1['window']}, n={N})\n{'='*78}")
    print(f"  T1 EXISTENCE      mean/day    cum_return   ann_Sharpe   t-stat")
    for k in ("overnight", "intraday", "buy_hold"):
        v = test1[k]
        print(f"    {k:11s}    {v['mean_bp']:+7.2f}bp   {v['cum_return']:+8.1%}   "
              f"{str(v['ann_sharpe']):>8s}   {str(v.get('t_stat','')):>6s}")
    print(f"  T2 NET OF COSTS (overnight-only, 1 round trip/day @ base {args.cost_bp}bp):")
    for mult in ("1.0x", "1.5x", "2.0x"):
        v = test2[mult]
        print(f"    {mult}: ann_Sharpe {str(v['ann_sharpe']):>6s}  ann_return {v['ann_return_pct']:+.1f}%  "
              f"(cost drag {v['cost_drag_pct_yr']:.1f}%/yr)")
    print(f"  T3 WALK-FORWARD (overnight net-1x Sharpe by year):")
    print("    " + "  ".join(f"{y}:{test3[y]['overnight_net_sharpe']}" for y in test3))
    print(f"    recent (2023-25) all positive: {recent_alive}")
    print(f"  T4 REVERSAL: corr(on,intra)={test4['overnight_intraday_corr']} slope={test4['intraday_on_overnight_slope']} — {test4['interpretation']}")
    print(f"  T5 PERMUTATION: overnight−intraday = {test5['observed_overnight_minus_intraday_bp']}bp, p={p_value}")
    print(f"\n  GATES: cost✓={survives_cost}  recent_alive✓={recent_alive}  significant✓={significant}")
    print(f"  VERDICT: {verdict}")
    print(f"  CAVEAT: QQQ = the phenomenon on the Nasdaq source; NOT yet a tradeable NQ strategy.")
    print(f"\n  Logged to hypothesis_ledger.json. Saved: data/research/edge_research_overnight.json\n{'='*78}\n")


if __name__ == "__main__":
    main()
