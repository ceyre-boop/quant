#!/usr/bin/env python3
"""
RQ-REST-014 — Does the HYP-058 signed-velocity defense overlay act THROUGH the
exit channel (HYP-059)? i.e. is the overlay's drawdown reduction mostly achieved
by suppressing the trailing-stop loss cluster, or is it an orthogonal lever?

If the overlay preferentially throttles trailing-stop losers (low mult on the
-0.5R cluster), then HYP-058 (size overlay) and an exit-rule fix (RQ-010) are
PARTLY REDUNDANT -> pick the cheaper lever (the overlay: a pure size reducer,
no live exit-logic change, no Colin sign-off on exit math).

If the multiplier distribution is the SAME across exit_reasons, the overlay is
NOT working through the exit channel -> the two levers are complementary and you
would want BOTH.

Method (forensic, no network, no look-ahead):
  - costed 4-pair v015/HYP-045 log (logs/forex_backtest_trades.json, 318 trades).
  - portfolio signed/abs 90d rate-spread velocity from cached macro parquets.
  - CAUSAL expanding-window percentile (min 252 prior obs) -> HYP-058 buckets
    {LOW:0.25, MID:0.5, HIGH:1.0} via lo_q=0.25, hi_q=0.50 (matches RQ-REST-011).
  - tag each trade by its velocity bucket at entry; cross-tab exit_reason x bucket.
  - compare mean applied multiplier across exits; permutation test
    mean_mult(trailing) vs mean_mult(time). Decompose overlay Delta-R by exit.
"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

RNG = np.random.default_rng(14)
ROOT = Path("/sessions/brave-sweet-rubin/mnt/quant")
SRC = ROOT / "logs/forex_backtest_trades.json"
OUT = ROOT / "data/agent/rq_rest_014_results.json"

RATE_MAP = {  # pair -> (base ccy, quote ccy)
    "EURUSD=X": ("EU", "US"), "GBPUSD=X": ("GB", "US"),
    "USDJPY=X": ("US", "JP"), "AUDUSD=X": ("AU", "US"),
}


def load_rate(c):
    return pd.read_parquet(ROOT / f"data/cache/macro/{c}_rates.parquet")["rate"]


def build_velocity(lookback=90, signed=True):
    cal = pd.date_range("2014-01-01", "2023-06-30", freq="D")
    ccys = set(sum([list(v) for v in RATE_MAP.values()], []))
    rates = {c: load_rate(c).reindex(cal).ffill() for c in ccys}
    spread = pd.DataFrame({p: rates[b] - rates[q] for p, (b, q) in RATE_MAP.items()},
                          index=cal)
    chg = spread - spread.shift(lookback)
    chg = chg if signed else chg.abs()
    return chg.sum(axis=1)  # portfolio velocity (sum across pairs)


def causal_pct_bucket(vel, lo_q=0.25, hi_q=0.50, min_obs=252):
    """Expanding-window percentile rank of each day's |velocity| -> bucket+mult.
    Magnitude of trend (|signed velocity|) is what 'flat regime' means, so rank on
    abs value even when the underlying series is signed."""
    mag = vel.abs()
    out = {}
    vals = mag.values
    idx = mag.index
    for i in range(len(mag)):
        if np.isnan(vals[i]) or i < min_obs:
            continue
        hist = vals[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_obs:
            continue
        rank = (hist < vals[i]).mean()  # causal percentile in [0,1)
        if rank < lo_q:
            b, m = "LOW", 0.25
        elif rank < hi_q:
            b, m = "MID", 0.50
        else:
            b, m = "HIGH", 1.0
        out[idx[i]] = (rank, b, m)
    return out


def net_R(t):
    return (t["pnl_pct"] - t.get("cost_spread_frac", 0.0)
            + t.get("cost_swap_frac", 0.0)) / t["risk_pct"]


def load_trades():
    d = json.load(open(SRC))
    rows = []
    for pair, tl in d.items():
        for t in tl:
            rows.append(dict(
                pair=pair.replace("=X", ""),
                entry=pd.Timestamp(t["entry_date"]).normalize(),
                year=t["entry_date"][:4],
                exit=t["exit_reason"],
                nR=net_R(t)))
    return rows


def tag(rows, bucketmap):
    tagged = []
    for r in rows:
        d = r["entry"]
        info = bucketmap.get(d)
        if info is None:  # fall back to most recent prior tagged day
            prior = [k for k in bucketmap if k <= d]
            if not prior:
                continue
            info = bucketmap[max(prior)]
        rank, b, m = info
        tagged.append({**r, "rank": rank, "bucket": b, "mult": m})
    return tagged


def grp_stats(rs):
    a = np.array([r["nR"] for r in rs], float)
    return dict(n=len(a), wr=float(100 * np.mean(a > 0)) if len(a) else 0.0,
                meanR=float(a.mean()) if len(a) else 0.0,
                sumR=float(a.sum()) if len(a) else 0.0,
                mean_mult=float(np.mean([r["mult"] for r in rs])) if rs else 0.0)


def main():
    rows = load_trades()
    results = {}
    for signed in (True, False):
        vel = build_velocity(90, signed=signed)
        bmap = causal_pct_bucket(vel)
        tg = tag(rows, bmap)

        cross = defaultdict(list)
        for r in tg:
            cross[(r["exit"], r["bucket"])].append(r)
        ctab = {f"{e}|{b}": grp_stats(v) for (e, b), v in sorted(cross.items())}

        by_exit = defaultdict(list)
        for r in tg:
            by_exit[r["exit"]].append(r)
        exit_summary = {}
        for e, v in by_exit.items():
            bs = [r["bucket"] for r in v]
            exit_summary[e] = {
                **grp_stats(v),
                "bucket_mix": {b: bs.count(b) for b in ("LOW", "MID", "HIGH")},
                "low_share": float(bs.count("LOW") / len(bs)),
            }

        delta_by_exit = {}
        for e, v in by_exit.items():
            base = sum(r["nR"] for r in v)
            gated = sum(r["nR"] * r["mult"] for r in v)
            delta_by_exit[e] = dict(base_sumR=float(base), gated_sumR=float(gated),
                                    delta=float(gated - base))
        tot_base = sum(r["nR"] for r in tg)
        tot_gated = sum(r["nR"] * r["mult"] for r in tg)
        tot_delta = tot_gated - tot_base
        trail_delta = delta_by_exit.get("trailing_stop", {}).get("delta", 0.0)
        improve_share_trailing = (trail_delta / tot_delta) if tot_delta else float("nan")

        tr = np.array([r["mult"] for r in tg if r["exit"] == "trailing_stop"])
        ti = np.array([r["mult"] for r in tg if r["exit"] == "time"])
        obs = tr.mean() - ti.mean()
        pool = np.concatenate([tr, ti])
        n_tr = len(tr)
        cnt = 0
        for _ in range(10000):
            RNG.shuffle(pool)
            if (pool[:n_tr].mean() - pool[n_tr:].mean()) <= obs:
                cnt += 1
        perm_p = cnt / 10000

        results["signed" if signed else "abs"] = dict(
            n_tagged=len(tg),
            cross_tab=ctab,
            exit_summary=exit_summary,
            overlay_delta_by_exit=delta_by_exit,
            overlay_total=dict(base_sumR=float(tot_base), gated_sumR=float(tot_gated),
                               delta=float(tot_delta),
                               improve_share_from_trailing=float(improve_share_trailing)),
            mult_test=dict(mean_mult_trailing=float(tr.mean()),
                           mean_mult_time=float(ti.mean()),
                           diff=float(obs), perm_p_trailing_more_throttled=float(perm_p)),
        )

    OUT.write_text(json.dumps(results, indent=2))

    s = results["signed"]
    print("=" * 70)
    print("RQ-REST-014 — velocity overlay x exit-mechanism redundancy")
    print("=" * 70)
    print(f"tagged trades: {s['n_tagged']}\n")
    print("Mean applied multiplier by exit_reason (|signed-vel| buckets):")
    for e, d in sorted(s["exit_summary"].items(), key=lambda kv: -kv[1]["n"]):
        mix = d["bucket_mix"]
        print(f"  {e:13s} n={d['n']:3d}  meanR={d['meanR']:+.3f}  "
              f"mean_mult={d['mean_mult']:.3f}  LOWshare={d['low_share']:.2f}  "
              f"mix L/M/H={mix['LOW']}/{mix['MID']}/{mix['HIGH']}")
    mt = s["mult_test"]
    print(f"\nmean_mult trailing={mt['mean_mult_trailing']:.3f} vs "
          f"time={mt['mean_mult_time']:.3f}  diff={mt['diff']:+.3f}  "
          f"perm_p(trailing throttled more)={mt['perm_p_trailing_more_throttled']:.4f}")
    ot = s["overlay_total"]
    print(f"\noverlay total R: base={ot['base_sumR']:+.1f} -> gated={ot['gated_sumR']:+.1f} "
          f"(Delta={ot['delta']:+.1f})")
    print(f"share of overlay R-Delta from trailing cluster = "
          f"{ot['improve_share_from_trailing']:+.2f}")
    print("\noverlay Delta by exit:")
    for e, d in sorted(s["overlay_delta_by_exit"].items(),
                       key=lambda kv: kv[1]["delta"]):
        print(f"  {e:13s} base={d['base_sumR']:+7.1f} gated={d['gated_sumR']:+7.1f} "
              f"Delta={d['delta']:+6.1f}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
