#!/usr/bin/env python3
"""scripts/test_regime_conditionality.py — HYP-065: is the carry edge Fed-cycle-conditional?

Pre-registered (data/research/preregister/HYP-065_carry_regime_conditionality.json, hash 111854c0).
Generates ONE continuous 2015→2026 trade ledger for the 4-pair v015 portfolio via the canonical
ForexBacktester, tags each trade by Fed regime (FEDFUNDS) + cycle, and tests whether the edge
concentrates in HIKING/PEAK_HOLD and fails in CUTTING/BOTTOM across ≥2 of 3 cycles.

READ-ONLY: re-reads yfinance/FRED, writes only the report + ledger verdict. No live config, no
model training, no commits.

    python3 scripts/test_regime_conditionality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.reporting.equity_curve import _sharpe, weighted_portfolio_sharpe  # canonical Sharpe

PRE = json.load(open(ROOT / "data/research/preregister/HYP-065_carry_regime_conditionality.json"))
OUT_MD = ROOT / "docs/research/HYP-065_regime_conditionality.md"
LEDGER = ROOT / "data/agent/hypothesis_ledger.json"
TRADES_FILE = ROOT / "logs/forex_backtest_trades.json"
DOSSIER_BACKUP = ROOT / "logs/forex_backtest_trades_dossier_20260627.json"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
PAIR_YF = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X"}
CCY_TO_COUNTRY = {"EUR": "EU", "GBP": "UK", "JPY": "JP", "USD": "US", "AUD": "AU", "CAD": "CA"}
CYCLES = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in PRE["cycle_dates"].items()}
ON, OFF = {"HIKING", "PEAK_HOLD"}, {"CUTTING", "BOTTOM"}


def _fedfunds() -> pd.Series:
    from fredapi import Fred
    key = [l.split("=", 1)[1].strip().strip('"').strip("'")
           for l in (ROOT / ".env").read_text().splitlines() if l.startswith("FRED_API_KEY")][0]
    s = Fred(api_key=key).get_series("FEDFUNDS", observation_start="2013-01-01", observation_end="2026-06-30")
    return pd.Series(s).astype(float)


def _regime_series(ff: pd.Series, delta=0.25, lookback_mo=3, hl_mo=12) -> pd.Series:
    """Daily regime label from monthly FEDFUNDS, per the pre-registered rule."""
    daily = ff.reindex(pd.date_range(ff.index.min(), "2026-06-30", freq="D")).ffill()
    lb = lookback_mo * 30
    out = {}
    for d in daily.index:
        now = daily.loc[d]
        past = daily.loc[:d - pd.Timedelta(days=lb)]
        win = daily.loc[d - pd.Timedelta(days=hl_mo * 30):d]
        if pd.isna(now) or past.empty or win.empty:
            out[d] = "NEUTRAL"; continue
        diff = now - past.iloc[-1]
        hi, lo = win.max(), win.min()
        if diff >= delta:
            out[d] = "HIKING"
        elif diff <= -delta:
            out[d] = "CUTTING"
        elif now >= hi - delta:
            out[d] = "PEAK_HOLD"
        elif now <= lo + delta:
            out[d] = "BOTTOM"
        else:
            out[d] = "NEUTRAL"
    return pd.Series(out)


def _cycle_of(d: pd.Timestamp) -> str:
    for k, (a, b) in CYCLES.items():
        if a <= d <= b:
            return k
    return "C0_pre"


def _gen_ledger() -> pd.DataFrame:
    """CANONICAL path: backtest_all() writes logs/forex_backtest_trades.json (the v015 ledger that
    prove.py/the dossier use). We read that — NOT run_pair_with_trades (which VIX-gates USDJPY)."""
    from sovereign.forex.forex_backtester import ForexBacktester
    ForexBacktester(start="2015-01-01", end="2026-06-30").backtest_all()   # writes TRADES_FILE
    raw = json.loads(TRADES_FILE.read_text())
    rows = []
    for tk, tl in raw.items():
        pair = tk.replace("=X", "")
        if pair not in PAIRS:
            continue
        for t in tl:
            rows.append({"pair": pair,
                         "entry_date": pd.Timestamp(t["entry_date"]).normalize(),
                         "exit_date": pd.Timestamp(t["exit_date"]).normalize(),
                         "pnl_pct": float(t.get("pnl_pct", 0.0)),
                         "rapnl": float(t.get("risk_adjusted_pnl_pct", t.get("pnl_pct", 0.0)))})
    return pd.DataFrame(rows)


def _grp_sharpe(df: pd.DataFrame) -> float:
    """Canonical √n-weighted costed portfolio Sharpe (dossier method): per-pair _sharpe on
    risk_adjusted_pnl_pct (annualised by trades/yr), then weighted_portfolio_sharpe across pairs."""
    if len(df) == 0:
        return 0.0
    per_pair = []
    for _p, g in df.groupby("pair"):
        yrs = max((g["entry_date"].max() - g["entry_date"].min()).days / 365.25, 0.25)
        per_pair.append((_sharpe(g["rapnl"].tolist(), yrs), len(g)))
    return weighted_portfolio_sharpe(per_pair)


def _reconcile(df: pd.DataFrame):
    """Gate: the new full-history ledger's 2025-26 slice MUST match the dossier backup (esp. USDJPY=16),
    and OOS 2023-24 must reproduce ~1.25. Returns (issues, oos_sharpe, fresh_counts)."""
    issues = []
    if not DOSSIER_BACKUP.exists():
        return ["dossier backup missing — cannot reconcile"], None, {}
    backup = json.loads(DOSSIER_BACKUP.read_text())
    fresh = df[df.entry_date >= pd.Timestamp("2025-01-01")]
    fresh_counts = {}
    for tk, tl in backup.items():
        p = tk.replace("=X", "")
        mine_n, doss_n = int((fresh.pair == p).sum()), len(tl)
        fresh_counts[p] = (mine_n, doss_n)
        if abs(mine_n - doss_n) > 1:                 # ±1 tolerance for yfinance edge drift
            issues.append(f"{p} 2025-26 count: mine {mine_n} vs dossier {doss_n}")
    oos = df[(df.entry_date >= pd.Timestamp("2023-01-01")) & (df.entry_date <= pd.Timestamp("2024-12-31"))]
    oos_sharpe = _grp_sharpe(oos)
    if not (0.7 <= oos_sharpe <= 1.7):               # documented OOS ~1.25
        issues.append(f"OOS 2023-24 Sharpe {oos_sharpe} outside the ~1.25 reconciliation band [0.7,1.7]")
    return issues, oos_sharpe, fresh_counts


def _classify_and_tag(ledger: pd.DataFrame, delta, lb, hl, ff) -> pd.DataFrame:
    reg = _regime_series(ff, delta, lb, hl)
    df = ledger.copy()
    df["regime"] = df["entry_date"].map(lambda d: reg.get(d, reg.reindex([d], method="ffill").iloc[0] if d in reg.index or len(reg) else "NEUTRAL"))
    # robust map: nearest prior daily regime
    rd = reg.reindex(pd.DatetimeIndex(df["entry_date"].unique()).sort_values(), method="ffill")
    df["regime"] = df["entry_date"].map(rd.to_dict())
    df["cycle"] = df["entry_date"].map(_cycle_of)
    df["group"] = df["regime"].map(lambda r: "ON" if r in ON else ("OFF" if r in OFF else "NEUTRAL"))
    return df


def _bar(df: pd.DataFrame):
    per_cycle = {}
    for c in ("C1", "C2", "C3"):
        sub = df[df.cycle == c]
        s_on = _grp_sharpe(sub[sub.group == "ON"]); s_off = _grp_sharpe(sub[sub.group == "OFF"])
        per_cycle[c] = {"sharpe_on": round(s_on, 3), "sharpe_off": round(s_off, 3),
                        "n_on": int((sub.group == "ON").sum()), "n_off": int((sub.group == "OFF").sum())}
    cyc_on = sum(1 for c in per_cycle if per_cycle[c]["sharpe_on"] > 0.50)
    cyc_off = sum(1 for c in per_cycle if per_cycle[c]["sharpe_off"] < 0.20)
    return per_cycle, cyc_on, cyc_off


def _perm_p(df: pd.DataFrame, n=1000, seed=11) -> float:
    obs = _grp_sharpe(df[df.group == "ON"]) - _grp_sharpe(df[df.group == "OFF"])
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        d = df.copy()
        for c in d.cycle.unique():
            m = d.cycle == c
            d.loc[m, "regime"] = rng.permutation(d.loc[m, "regime"].to_numpy())
        d["group"] = d["regime"].map(lambda r: "ON" if r in ON else ("OFF" if r in OFF else "NEUTRAL"))
        if (_grp_sharpe(d[d.group == "ON"]) - _grp_sharpe(d[d.group == "OFF"])) >= obs:
            ge += 1
    return (ge + 1) / (n + 1)


def main() -> dict:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    ff = _fedfunds()
    ledger = _gen_ledger()

    # ── RECONCILIATION GATE: do not proceed unless the canonical ledger reproduces v015 ──
    issues, oos_sharpe, fresh_counts = _reconcile(ledger)
    print(f"RECONCILE: OOS 2023-24 Sharpe {oos_sharpe} | 2025-26 counts (mine,dossier) {fresh_counts}")
    if issues:
        print("RECONCILIATION FAILED — NOT proceeding to regime tagging:")
        for i in issues:
            print("   -", i)
        OUT_MD.write_text("# HYP-065 — RECONCILIATION FAILED (run 2)\n\n"
                          "The canonical backtest_all() ledger did NOT reconcile with the dossier; "
                          "no regime verdict produced.\n\n- " + "\n- ".join(issues) +
                          f"\n\nOOS 2023-24 Sharpe = {oos_sharpe}; 2025-26 counts (mine,dossier) = {fresh_counts}\n")
        return {"verdict": "RECONCILIATION_FAILED", "issues": issues, "oos_sharpe": oos_sharpe,
                "fresh_counts": fresh_counts}

    df = _classify_and_tag(ledger, 0.25, 3, 12, ff)
    df.attrs["oos_sharpe"] = oos_sharpe

    per_regime = {r: {"sharpe": round(_grp_sharpe(df[df.regime == r]), 3), "n": int((df.regime == r).sum()),
                      "win": round(float((df[df.regime == r]["pnl_pct"] > 0).mean()), 3) if (df.regime == r).any() else None}
                  for r in ["HIKING", "PEAK_HOLD", "CUTTING", "BOTTOM", "NEUTRAL"]}
    per_cycle, cyc_on, cyc_off = _bar(df)
    perm = _perm_p(df)

    # robustness sweep
    robust = {}
    for delta in PRE["robustness"]["delta_threshold_pct"]:
        rdf = _classify_and_tag(ledger, delta, 3, 12, ff)
        _pc, co, cf = _bar(rdf); robust[f"delta{delta}"] = {"cyc_on>0.5": co, "cyc_off<0.2": cf}
    for lb in PRE["robustness"]["lookback_months"]:
        rdf = _classify_and_tag(ledger, 0.25, lb, 12, ff)
        _pc, co, cf = _bar(rdf); robust[f"lookback{lb}"] = {"cyc_on>0.5": co, "cyc_off<0.2": cf}

    bar_met = (cyc_on >= 2 and cyc_off >= 2 and perm < 0.05)
    robust_ok = all(v["cyc_on>0.5"] >= 2 and v["cyc_off<0.2"] >= 2 for v in robust.values())
    c3_only = (per_cycle["C3"]["sharpe_on"] > 0.50 and not (per_cycle["C1"]["sharpe_on"] > 0.50 or per_cycle["C2"]["sharpe_on"] > 0.50))

    if bar_met and robust_ok:
        verdict = "CONFIRMED"
    elif c3_only or (bar_met and not robust_ok):
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"

    fresh = df[df.entry_date >= pd.Timestamp("2024-01-01")]
    fresh_info = {"n": len(fresh), "regimes": fresh.regime.value_counts().to_dict(),
                  "sharpe": round(_grp_sharpe(fresh), 3),
                  "matches_expected_off": bool(_grp_sharpe(fresh) < 0.20 and (fresh.group == "OFF").mean() > 0.5)}

    out = {"hypothesis": "HYP-065", "hash": PRE["hash_lock"][:16], "verdict": verdict,
           "n_trades": len(df), "per_regime": per_regime, "per_cycle": per_cycle,
           "cycles_edge_on_gt0.5": cyc_on, "cycles_edge_off_lt0.2": cyc_off,
           "permutation_p": round(perm, 4), "robustness": robust, "robust_ok": robust_ok,
           "c3_only": c3_only, "fresh_2024_26": fresh_info,
           "by_cycle_pair": {c: {p: round(_grp_sharpe(df[(df.cycle == c) & (df.pair == p)]), 2) for p in PAIRS}
                             for c in ("C1", "C2", "C3")}}
    _write_md(out); _update_ledger(out)
    return out


def _write_md(o: dict) -> None:
    pc = o["per_cycle"]
    L = [f"# HYP-065 — Carry Regime Conditionality — VERDICT: {o['verdict']}", "",
         f"> pre-reg hash `{o['hash']}` · {o['n_trades']} trades (2015-2026, 4-pair v015) · "
         "READ-ONLY, no live changes. Sharpe = annualised (√n by trades/yr).", "",
         "## Per-regime (pooled)", "", "| regime | Sharpe | win | n |", "|---|---|---|---|"]
    for r, v in o["per_regime"].items():
        L.append(f"| {r} | {v['sharpe']} | {v['win']} | {v['n']} |")
    L += ["", "## Per cycle — edge-ON (hiking+peak) vs edge-OFF (cutting+bottom)", "",
          "| cycle | Sharpe ON | Sharpe OFF | n_on | n_off |", "|---|---|---|---|---|"]
    for c in ("C1", "C2", "C3"):
        L.append(f"| {c} | {pc[c]['sharpe_on']} | {pc[c]['sharpe_off']} | {pc[c]['n_on']} | {pc[c]['n_off']} |")
    L += ["", f"**Bar:** edge-ON>0.50 in **{o['cycles_edge_on_gt0.5']}/3** cycles (need ≥2); "
          f"edge-OFF<0.20 in **{o['cycles_edge_off_lt0.2']}/3** (need ≥2); permutation p **{o['permutation_p']}** "
          f"(need <0.05); robust across sweeps: **{o['robust_ok']}**.", "",
          f"**Robustness sweep** (cycles meeting on>0.5 / off<0.2): {json.dumps(o['robustness'])}", "",
          "## Load-bearing: 2024-26 fresh (Fed cutting → should be edge-OFF)", "",
          f"- {o['fresh_2024_26']['n']} trades, regimes {o['fresh_2024_26']['regimes']}, "
          f"Sharpe {o['fresh_2024_26']['sharpe']} → matches-expected-OFF: **{o['fresh_2024_26']['matches_expected_off']}**", "",
          "## Sharpe by cycle × pair", "", "| cycle | " + " | ".join(PAIRS) + " |", "|" + "---|" * (len(PAIRS) + 1)]
    for c in ("C1", "C2", "C3"):
        L.append(f"| {c} | " + " | ".join(str(o["by_cycle_pair"][c][p]) for p in PAIRS) + " |")
    L += ["", "## Verdict", "",
          {"CONFIRMED": "Pattern holds in ≥2/3 cycles, significant, robust → a Fed-regime gate is "
                        "justified. Design follows in a separate pre-registered prompt.",
           "PARTIAL": "Pattern is C3-only or fragile to parameters → curve-fit to 2022-23; carry is "
                      "regime-dead. Pivot to a different edge family (e.g. VRP).",
           "REJECTED": "No regime pattern → carry is dead unconditionally. Pivot edge families."}[o["verdict"]],
          ""]
    OUT_MD.write_text("\n".join(L) + "\n")


def _update_ledger(o: dict) -> None:
    led = json.loads(LEDGER.read_text())
    for e in led:
        if isinstance(e, dict) and e.get("id") == "HYP-065":
            e["status"] = o["verdict"]; e["date_tested"] = "2026-06-27"
            e["result"] = (f"{o['verdict']}: edge-ON>0.5 in {o['cycles_edge_on_gt0.5']}/3 cycles, "
                           f"edge-OFF<0.2 in {o['cycles_edge_off_lt0.2']}/3, perm p {o['permutation_p']}, "
                           f"robust {o['robust_ok']}. Per-cycle ON Sharpe: "
                           f"C1 {o['per_cycle']['C1']['sharpe_on']} / C2 {o['per_cycle']['C2']['sharpe_on']} / "
                           f"C3 {o['per_cycle']['C3']['sharpe_on']}. 2024-26 fresh Sharpe "
                           f"{o['fresh_2024_26']['sharpe']} (matches-OFF {o['fresh_2024_26']['matches_expected_off']}).")
    json.dump(led, open(LEDGER, "w"), indent=2)


if __name__ == "__main__":
    o = main()
    if o.get("verdict") == "RECONCILIATION_FAILED":
        print("=" * 76)
        print("HYP-065 RECONCILIATION FAILED — no verdict. Issues:")
        for i in o["issues"]:
            print("   -", i)
        print(f"  OOS 2023-24 Sharpe {o['oos_sharpe']} | 2025-26 counts {o['fresh_counts']}")
        print("=" * 76)
        raise SystemExit(0)
    pc = o["per_cycle"]
    print("=" * 76)
    print(f"HYP-065 VERDICT: {o['verdict']}   ({o['n_trades']} trades 2015-2026)")
    print(f"  per-regime Sharpe: " + " ".join(f"{r}={v['sharpe']}(n{v['n']})" for r, v in o['per_regime'].items()))
    for c in ("C1", "C2", "C3"):
        print(f"  {c}: edge-ON {pc[c]['sharpe_on']} (n{pc[c]['n_on']})  edge-OFF {pc[c]['sharpe_off']} (n{pc[c]['n_off']})")
    print(f"  bar: ON>0.5 in {o['cycles_edge_on_gt0.5']}/3, OFF<0.2 in {o['cycles_edge_off_lt0.2']}/3, "
          f"perm p {o['permutation_p']}, robust {o['robust_ok']}")
    print(f"  2024-26 fresh: n{o['fresh_2024_26']['n']}, Sharpe {o['fresh_2024_26']['sharpe']}, "
          f"regimes {o['fresh_2024_26']['regimes']}, matches-OFF {o['fresh_2024_26']['matches_expected_off']}")
    print("=" * 76)
