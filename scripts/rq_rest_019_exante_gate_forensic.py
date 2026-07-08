#!/usr/bin/env python3
"""
RQ-REST-019 (no-network) — Ex-ante gate forensic for the trailing-whipsaw drag.

Context: RQ-REST-013 partial (FIND-REST-018-a) localized the -49R trailing-stop
drag to a 3-5 day hold-bucket whipsaw cluster (-42.2R, 86% of the drag). But
hold_days is only known at EXIT, so it is NOT a usable gate. The open question
the prior cycle flagged: is there an ENTRY-KNOWN feature that proxies for that
doomed early-whipsaw cluster, so we could veto those trades up front?

This script answers that with the correct ex-ante framing: a real gate vetoes
the WHOLE trade at entry (before the exit type is known), so the unit of
analysis is ALL 318 trades, not just the trailing subset. We score every
entry-known cut by its effect on TOTAL portfolio R / Sharpe, with a permutation
test (shuffle the gate label, hold the veto count fixed) and a walk-forward
split (early vs late half) to test the known regime-fragility weakness.

Entry-known features available in logs/forex_backtest_trades.json:
  pair, direction, entry_date -> DOW, DOM-bucket, month, quarter, year.
NO look-ahead. NO live config change. NO network.

Run: python3 scripts/rq_rest_019_exante_gate_forensic.py
"""
import json
import datetime as dt
import numpy as np
from collections import defaultdict
from pathlib import Path

RNG = np.random.default_rng(19)
SRC = Path("logs/forex_backtest_trades.json")
OUT = Path("data/agent/rq_rest_019_results.json")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_rows():
    d = json.load(open(SRC))
    rows = []
    for pair, tl in d.items():
        for t in tl:
            net_R = (t["pnl_pct"] - t.get("cost_spread_frac", 0.0)
                     + t.get("cost_swap_frac", 0.0)) / t["risk_pct"]
            ed = dt.datetime.fromisoformat(t["entry_date"])
            dom = ed.day
            rows.append(dict(
                pair=pair.replace("=X", ""),
                date=ed,
                year=ed.year,
                month=ed.month,
                quarter=f"Q{(ed.month - 1) // 3 + 1}",
                dow=ed.weekday(),            # 0=Mon
                dom_bucket=("1-7" if dom <= 7 else "8-15" if dom <= 15
                            else "16-23" if dom <= 23 else "24-31"),
                dirn="long" if t.get("direction", 1) >= 0 else "short",
                exit=t["exit_reason"],
                hold=int(t.get("hold_days", 0)),
                nR=net_R,
            ))
    rows.sort(key=lambda r: r["date"])
    return rows


def sharpe(a):
    a = np.asarray(a, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")


def stats(rs):
    a = np.array([r["nR"] for r in rs], float)
    if len(a) == 0:
        return dict(n=0, wr=float("nan"), meanR=float("nan"), sharpe=float("nan"), sumR=0.0)
    return dict(n=int(len(a)), wr=round(float(100 * np.mean(a > 0)), 1),
                meanR=round(float(a.mean()), 4), sharpe=round(sharpe(a), 4),
                sumR=round(float(a.sum()), 3))


def boot_ci(a, n=5000):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float("nan"), float("nan")]
    bs = [RNG.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def perm_test_keep_minus_veto(rows, mask_fn, n=10000):
    """Permutation: does the *kept* set's meanR beat the *vetoed* set's meanR by
    more than chance? Shuffle which trades carry the gate label, hold count fixed.
    Statistic = mean(kept) - mean(vetoed). One-sided (gate should raise kept)."""
    R = np.array([r["nR"] for r in rows], float)
    veto = np.array([mask_fn(r) for r in rows], bool)
    k = int(veto.sum())
    if k == 0 or k == len(rows):
        return None
    obs = R[~veto].mean() - R[veto].mean()
    idx = np.arange(len(rows))
    cnt = 0
    for _ in range(n):
        v = RNG.choice(idx, k, replace=False)
        m = np.zeros(len(rows), bool)
        m[v] = True
        if (R[~m].mean() - R[m].mean()) >= obs:
            cnt += 1
    return dict(obs_keep_minus_veto=round(float(obs), 4), p_value=round((cnt + 1) / (n + 1), 4),
                n_vetoed=k)


def gate_effect(rows, mask_fn):
    """mask_fn(r)->True means VETO. Report portfolio before/after + what's removed."""
    kept = [r for r in rows if not mask_fn(r)]
    vetoed = [r for r in rows if mask_fn(r)]
    base = stats(rows)
    aft = stats(kept)
    # how much of the trailing 3-5d whipsaw drag does this veto remove?
    whip = [r for r in vetoed if r["exit"] == "trailing_stop" and 3 <= r["hold"] <= 5]
    whip_R_removed = round(sum(r["nR"] for r in whip), 3)
    winners_removed = round(sum(r["nR"] for r in vetoed if r["nR"] > 0), 3)
    return dict(
        base_sumR=base["sumR"], base_meanR=base["meanR"], base_sharpe=base["sharpe"], base_n=base["n"],
        kept_sumR=aft["sumR"], kept_meanR=aft["meanR"], kept_sharpe=aft["sharpe"], kept_n=aft["n"],
        d_sumR=round(aft["sumR"] - base["sumR"], 3),
        d_meanR=round(aft["meanR"] - base["meanR"], 4),
        d_sharpe=round(aft["sharpe"] - base["sharpe"], 4),
        n_vetoed=len(vetoed), vetoed_sumR=round(sum(r["nR"] for r in vetoed), 3),
        whip_3_5d_R_removed=whip_R_removed, winner_R_removed=winners_removed,
    )


def walk_forward(rows, mask_fn):
    mid = rows[len(rows) // 2]["date"]
    early = [r for r in rows if r["date"] < mid]
    late = [r for r in rows if r["date"] >= mid]
    out = {}
    for name, sub in (("early", early), ("late", late)):
        kept = [r for r in sub if not mask_fn(r)]
        out[name] = dict(split_from=str(early[0]["date"].date()) if name == "early" else str(mid.date()),
                         base_meanR=stats(sub)["meanR"], kept_meanR=stats(kept)["meanR"],
                         d_meanR=round(stats(kept)["meanR"] - stats(sub)["meanR"], 4),
                         base_sharpe=stats(sub)["sharpe"], kept_sharpe=stats(kept)["sharpe"])
    out["sign_consistent"] = bool(out["early"]["d_meanR"] > 0 and out["late"]["d_meanR"] > 0)
    return out


def main():
    rows = load_rows()
    res = {"meta": dict(src=str(SRC), n_trades=len(rows),
                        date_range=[str(rows[0]["date"].date()), str(rows[-1]["date"].date())],
                        baseline_portfolio=stats(rows))}

    # ---- Reference: where the doomed cluster actually sits (exit-known) ------
    whip = [r for r in rows if r["exit"] == "trailing_stop" and 3 <= r["hold"] <= 5]
    res["whipsaw_cluster_ref"] = dict(stats=stats(whip),
        by_dow={DOW[k]: stats([r for r in whip if r["dow"] == k]) for k in range(5)},
        by_pair={p: stats([r for r in whip if r["pair"] == p]) for p in sorted(set(r["pair"] for r in whip))},
        by_dirn={d: stats([r for r in whip if r["dirn"] == d]) for d in ("long", "short")},
        by_dom={b: stats([r for r in whip if r["dom_bucket"] == b]) for b in ("1-7", "8-15", "16-23", "24-31")},
    )

    # ---- Entry-known descriptive cuts on ALL trades -------------------------
    res["all_trades_by_feature"] = dict(
        by_dow={DOW[k]: stats([r for r in rows if r["dow"] == k]) for k in range(5)},
        by_pair={p: stats([r for r in rows if r["pair"] == p]) for p in sorted(set(r["pair"] for r in rows))},
        by_dirn={d: stats([r for r in rows if r["dirn"] == d]) for d in ("long", "short")},
        by_dom={b: stats([r for r in rows if r["dom_bucket"] == b]) for b in ("1-7", "8-15", "16-23", "24-31")},
        by_quarter={q: stats([r for r in rows if r["quarter"] == q]) for q in ("Q1", "Q2", "Q3", "Q4")},
    )

    # ---- Candidate ex-ante gates (veto = mask True) -------------------------
    gates = {
        "veto_Tue": lambda r: r["dow"] == 1,
        "veto_Thu": lambda r: r["dow"] == 3,
        "veto_Tue_Thu": lambda r: r["dow"] in (1, 3),
        "veto_Mon": lambda r: r["dow"] == 0,
        "veto_Fri": lambda r: r["dow"] == 4,
        "veto_short": lambda r: r["dirn"] == "short",
        "veto_dom_1_7": lambda r: r["dom_bucket"] == "1-7",
        "veto_dom_24_31": lambda r: r["dom_bucket"] == "24-31",
        "keep_only_dom_8_15": lambda r: r["dom_bucket"] != "8-15",
    }
    res["gates"] = {}
    for name, fn in gates.items():
        eff = gate_effect(rows, fn)
        perm = perm_test_keep_minus_veto(rows, fn)
        wf = walk_forward(rows, fn)
        # bootstrap CI on the vetoed set's meanR (is the vetoed bucket truly negative?)
        vR = [r["nR"] for r in rows if fn(r)]
        eff["vetoed_meanR_ci95"] = boot_ci(vR)
        res["gates"][name] = dict(effect=eff, perm_keep_minus_veto=perm, walk_forward=wf)

    OUT.write_text(json.dumps(res, indent=2))
    print("WROTE", OUT)
    # console summary
    b = res["meta"]["baseline_portfolio"]
    print(f"\nBASELINE 318-trade portfolio: sumR={b['sumR']} meanR={b['meanR']} sharpe={b['sharpe']} WR={b['wr']}%")
    print("\nGate ranking by d_sharpe (kept vs base):")
    rank = sorted(res["gates"].items(), key=lambda kv: kv[1]["effect"]["d_sharpe"], reverse=True)
    print(f"{'gate':18s} {'n_veto':>6s} {'d_sumR':>8s} {'d_meanR':>8s} {'d_sharpe':>9s} {'perm_p':>7s} {'wf_ok':>6s} {'whipR_rm':>9s} {'winR_rm':>8s}")
    for name, g in rank:
        e, p, w = g["effect"], g["perm_keep_minus_veto"], g["walk_forward"]
        pv = p["p_value"] if p else float("nan")
        print(f"{name:18s} {e['n_vetoed']:6d} {e['d_sumR']:8.2f} {e['d_meanR']:8.4f} "
              f"{e['d_sharpe']:9.4f} {pv:7.4f} {str(w['sign_consistent']):>6s} "
              f"{e['whip_3_5d_R_removed']:9.2f} {e['winner_R_removed']:8.2f}")


if __name__ == "__main__":
    main()
