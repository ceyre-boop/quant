#!/usr/bin/env python3
"""TICK-039: fit a spread model on MINING data, then rerun the sealed holdout.

DESIGN — why it is arranged this way
------------------------------------
1. The parametric spread model is fit on MINING-period events only. Fitting it
   on the holdout and then evaluating on the holdout would contaminate the
   evaluation with the cost model, which is the subtler cousin of the look-ahead
   bug that killed HYP-105/106.

2. The holdout rerun's PRIMARY result uses DIRECTLY MEASURED spreads per event,
   not the fitted model. Where real quotes exist there is no reason to model
   them. The fitted model is reported alongside as a validation of the model,
   and is what `realistic_fills` will use when quotes are not fetched.

3. NO VERDICT IS EMITTED. Per operator decision 2026-07-18, this ticket measures
   and reports; the verdict rule is set afterwards with full information. The
   report includes constitutional %/day against the 0.05% floor because that —
   not median return per trade — is the gate that actually binds (HYP-093 posted
   a +4.87% median and still failed at 0.023%/day).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median, mean, stdev

import numpy as np

from execution.config import frozen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "research" / "gapper" / "tick039"
C107 = frozen("hyp107")

# Published marks from the sealed reconstruction, used to validate that our
# rebuild of the holdout matches the original (the event list was never
# committed — only the thresholds survived).
PUBLISHED = {"all_gapups_n": 269, "all_gapups_median": -0.002, "all_gapups_win": 0.47,
             "filtered_n": 57, "filtered_median": 0.054, "filtered_mean": 0.153,
             "filtered_win": 0.70, "filtered_tail": 4.4}

TRADING_DAYS = 252


def load(which: str) -> list[dict]:
    p = DATA / f"{which}_events.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def passes_filter(r: dict) -> bool:
    og, lv = r.get("overnight_gap"), r.get("log_vol")
    if og is None or lv is None:
        return False
    return (og >= C107["gap_floor"] and og <= C107["og_max"]
            and lv <= C107["logvol_max"])


def is_gapup(r: dict) -> bool:
    og = r.get("overnight_gap")
    return og is not None and og >= C107["gap_floor"]


def tail_ratio(rets: list[float]) -> float | None:
    w = [x for x in rets if x > 0]
    l = [x for x in rets if x < 0]
    if not w or not l:
        return None
    return (sum(w) / len(w)) / (-sum(l) / len(l))


def sharpe(rets: list[float]) -> float | None:
    """Per-trade Sharpe annualised by observed trade frequency.

    NOT annualised as if trading daily — that is the error that produced the
    repo's discredited 2.097 headline.
    """
    if len(rets) < 3:
        return None
    sd = stdev(rets)
    if sd == 0:
        return None
    return mean(rets) / sd


def describe(rets: list[float], label: str) -> dict:
    if not rets:
        return {"label": label, "n": 0}
    return {
        "label": label, "n": len(rets),
        "median": round(median(rets), 6),
        "mean": round(mean(rets), 6),
        "win_rate": round(sum(1 for x in rets if x > 0) / len(rets), 4),
        "tail_ratio": round(tail_ratio(rets), 3) if tail_ratio(rets) else None,
        "per_trade_sharpe": round(sharpe(rets), 4) if sharpe(rets) else None,
    }


# ── Spread model, fit on MINING only ──────────────────────────────────────────

def fit_spread_model(mining: list[dict]) -> dict:
    """Regress log(half-spread) on log(price), log(minute $vol), log(bar range).

    Half-spread is the quantity `realistic_fills._half_spread` returns, so the
    fitted model is a drop-in replacement for it.
    """
    rows = [r for r in mining
            if r.get("entry_spread_pct") is not None
            and r.get("entry_mid", 0) > 0
            and r.get("entry_minute_dollar_vol", 0) > 0
            and r.get("entry_bar_range_pct") is not None]
    if len(rows) < 30:
        return {"ok": False, "n": len(rows), "reason": "insufficient mining data"}

    y = np.log(np.array([max(r["entry_spread_pct"] / 2.0, 1e-6) for r in rows]))
    X = np.column_stack([
        np.ones(len(rows)),
        np.log(np.array([r["entry_mid"] for r in rows])),
        np.log(np.array([max(r["entry_minute_dollar_vol"], 1e3) for r in rows])),
        np.log(np.array([max(r["entry_bar_range_pct"], 1e-4) for r in rows])),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    obs = sorted(r["entry_spread_pct"] / 2.0 for r in rows)
    return {
        "ok": True, "n": len(rows),
        "coef": {"intercept": round(float(coef[0]), 6),
                 "log_price": round(float(coef[1]), 6),
                 "log_dollar_vol": round(float(coef[2]), 6),
                 "log_bar_range": round(float(coef[3]), 6)},
        "r2_log": round(r2, 4),
        "residual_sd_log": round(float(np.std(y - pred)), 4),
        # caps derived from OBSERVED percentiles, not hand-picked
        "half_spread_floor": round(obs[int(0.01 * len(obs))], 6),
        "half_spread_p50": round(obs[len(obs) // 2], 6),
        "half_spread_p95": round(obs[int(0.95 * len(obs))], 6),
        "half_spread_cap": round(obs[int(0.99 * len(obs))], 6),
    }


def model_half_spread(price: float, dollar_vol: float, bar_range: float,
                      m: dict) -> float:
    c = m["coef"]
    lv = (c["intercept"] + c["log_price"] * math.log(max(price, 1e-6))
          + c["log_dollar_vol"] * math.log(max(dollar_vol, 1e3))
          + c["log_bar_range"] * math.log(max(bar_range, 1e-4)))
    return float(np.clip(math.exp(lv), m["half_spread_floor"], m["half_spread_cap"]))


# ── Legacy model, for the A/B ─────────────────────────────────────────────────

def legacy_half_spread(price, bar_range_pct, minute_dollar_vol, scenario="base"):
    from backtester.realistic_fills import SCENARIOS
    sc = SCENARIOS[scenario]
    illiq = sc["k_illiq"] / math.sqrt(max(minute_dollar_vol, 1e4) / 1e6)
    s = sc["k_range"] * bar_range_pct + illiq
    return float(np.clip(s, sc["floor"], sc["cap"]) / 2.0)


def main() -> int:
    mining, holdout = load("mining"), load("holdout")
    print(f"mining events: {len(mining)}   holdout events: {len(holdout)}\n")

    # ── reconstruction fidelity ──────────────────────────────────────────────
    gapups = [r for r in holdout if is_gapup(r)]
    filt = [r for r in holdout if passes_filter(r)]
    g_bar = [r["gross_bar"] for r in gapups if r.get("gross_bar") is not None]
    f_bar = [r["gross_bar"] for r in filt if r.get("gross_bar") is not None]

    print("=" * 72)
    print("RECONSTRUCTION FIDELITY (bar-based gross, vs published marks)")
    print("=" * 72)
    print(f"  all gap-ups : n={len(g_bar):4d} (published {PUBLISHED['all_gapups_n']})  "
          f"median {median(g_bar):+.4f} (published {PUBLISHED['all_gapups_median']:+.4f})  "
          f"win {sum(1 for x in g_bar if x>0)/len(g_bar):.3f} "
          f"(published {PUBLISHED['all_gapups_win']:.2f})" if g_bar else "  no gap-ups")
    if f_bar:
        print(f"  FILTERED    : n={len(f_bar):4d} (published {PUBLISHED['filtered_n']})  "
              f"median {median(f_bar):+.4f} (published {PUBLISHED['filtered_median']:+.4f})  "
              f"win {sum(1 for x in f_bar if x>0)/len(f_bar):.3f} "
              f"(published {PUBLISHED['filtered_win']:.2f})")

    # ── spread model, fit on MINING only ─────────────────────────────────────
    m = fit_spread_model(mining)
    print("\n" + "=" * 72)
    print("SPREAD MODEL — fit on MINING events only (holdout untouched)")
    print("=" * 72)
    if not m["ok"]:
        print(f"  FIT FAILED: {m}")
    else:
        print(f"  n={m['n']}  R^2(log)={m['r2_log']}  resid_sd(log)={m['residual_sd_log']}")
        print(f"  coefficients: {json.dumps(m['coef'])}")
        print(f"  observed half-spread: p1={m['half_spread_floor']:.5f} "
              f"p50={m['half_spread_p50']:.5f} p95={m['half_spread_p95']:.5f} "
              f"p99(cap)={m['half_spread_cap']:.5f}")
        print(f"  => round-trip median {2*m['half_spread_p50']:.4%}, "
              f"cap {2*m['half_spread_cap']:.4%}   (legacy cap was 8.0000%)")

    # ── holdout rerun ────────────────────────────────────────────────────────
    usable = [r for r in filt if r.get("net_quoted") is not None]
    print("\n" + "=" * 72)
    print("SEALED HOLDOUT RERUN — HYP-107 filtered set")
    print("=" * 72)
    print(f"  filtered events: {len(filt)}   with real quotes: {len(usable)}")

    results = {}
    if usable:
        gross = [r["gross_quoted"] for r in usable]
        net_measured = [r["net_quoted"] for r in usable]
        spreads = [r["spread_cost"] for r in usable]

        net_legacy, net_model = [], []
        for r in usable:
            g = r["gross_quoted"]
            hs_e = legacy_half_spread(r["entry_mid"], r["entry_bar_range_pct"],
                                      r["entry_minute_dollar_vol"])
            hs_x = legacy_half_spread(r["exit_mid"], r["exit_bar_range_pct"],
                                      r["exit_minute_dollar_vol"])
            net_legacy.append(g - hs_e - hs_x)
            if m["ok"]:
                me = model_half_spread(r["entry_mid"], r["entry_minute_dollar_vol"],
                                       r["entry_bar_range_pct"], m)
                mx = model_half_spread(r["exit_mid"], r["exit_minute_dollar_vol"],
                                       r["exit_bar_range_pct"], m)
                net_model.append(g - me - mx)

        for label, rets in [("GROSS (mid-to-mid)", gross),
                            ("NET — measured spreads (PRIMARY)", net_measured),
                            ("NET — fitted model (mining-fit)", net_model),
                            ("NET — legacy model (old, for A/B)", net_legacy)]:
            if not rets:
                continue
            d = describe(rets, label)
            results[label] = d
            print(f"\n  {label}")
            print(f"    n={d['n']}  median={d['median']:+.4%}  mean={d['mean']:+.4%}  "
                  f"win={d['win_rate']:.1%}  tail={d['tail_ratio']}  "
                  f"per-trade Sharpe={d['per_trade_sharpe']}")

        print(f"\n  median measured round-trip spread: {median(spreads):.4%}")

        # ── constitutional yield — the gate that actually binds ──────────────
        span_days = len({r["date"] for r in holdout})
        n_tr = len(usable)
        med_net = median(net_measured)
        # yield %/day = median net per trade * trades per day, at full weight
        trades_per_day = n_tr / max(span_days, 1)
        raw_yield = med_net * trades_per_day
        print("\n" + "=" * 72)
        print("CONSTITUTIONAL YIELD (Art. 1) — the gate that actually binds")
        print("=" * 72)
        print(f"  holdout span: {span_days} trading dates, {n_tr} filtered trades")
        print(f"  trades/day: {trades_per_day:.4f}")
        print(f"  UNSIZED yield: {raw_yield:.5%}/day   vs floor 0.05000%/day")
        print(f"  NOTE: unsized. Art. 1 conviction sizing reduces this. HYP-093 "
              f"posted a +4.87% median and still failed at 0.023%/day.")

        results["_constitutional"] = {
            "span_days": span_days, "n_trades": n_tr,
            "trades_per_day": round(trades_per_day, 5),
            "unsized_yield_pct_per_day": round(raw_yield * 100, 6),
            "floor_pct_per_day": 0.05,
        }

    print("\n" + "=" * 72)
    print("NO VERDICT EMITTED. HYP-107 remains "
          "REAL_BUT_MARGINAL_EXECUTION_UNRESOLVED.")
    print("Verdict rule to be set by the operator with these numbers in hand.")
    print("=" * 72)

    out = {"spread_model": m, "results": results,
           "reconstruction": {"gapups_n": len(g_bar), "filtered_n": len(f_bar),
                              "published": PUBLISHED}}
    (DATA / "tick039_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DATA / 'tick039_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
