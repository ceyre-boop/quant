"""Reconcile a broker fill record into measured returns: TWR, Modified Dietz MWR, vs SPY.

Input: data/thesis/fills/public_<acct>.jsonl (one row per trade or cash flow).
Nothing here reads the network except through the injected price loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILLS = ROOT / "data" / "thesis" / "fills"
Loader = Callable[[str], pd.Series]
EXTERNAL = {"DEPOSIT", "DEPOSIT_RETURNED", "WITHDRAWAL", "UNIDENTIFIED"}


def load_fills(account: str, fills_dir: Path = FILLS) -> list[dict[str, Any]]:
    rows = []
    f = fills_dir / (f"{account}.jsonl" if account.startswith("SIM-") else f"public_{account}.jsonl")
    for line in f.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if not d.get("_meta"):
                rows.append(d)
    return sorted(rows, key=lambda r: r["ts"])


def daily_frame(rows: list[dict[str, Any]], loader: Loader, asof: str,
                tickers: set[str] | None = None, cash: bool = True) -> pd.DataFrame:
    """Daily holdings value, cash, external flows. If `tickers` is given, restrict to that
    basket and treat its trades as the external flows (sub-portfolio view)."""
    start = pd.Timestamp(rows[0]["ts"]).normalize()
    end = pd.Timestamp(asof).normalize()
    days = pd.bdate_range(start, end)
    px = {}
    syms = {r["symbol"] for r in rows if r["type"] == "TRADE"}
    if tickers is not None:
        syms &= tickers
    for s in syms:
        px[s] = loader(s).reindex(days).ffill()
    qty = {s: 0.0 for s in syms}
    cash_bal = 0.0
    flow = pd.Series(0.0, index=days)
    value = pd.Series(0.0, index=days)
    by_day: dict[pd.Timestamp, list] = {}
    for r in rows:
        d = pd.Timestamp(r["ts"]).normalize()
        if d not in flow.index:
            d = days[days >= d][0] if (days >= d).any() else days[-1]
        by_day.setdefault(d, []).append(r)
    for d in days:
        for r in by_day.get(d, []):
            if r["type"] == "TRADE":
                if tickers is not None and r["symbol"] not in tickers:
                    continue
                side = 1 if r["side"] == "BUY" else -1
                qty[r["symbol"]] += side * float(r["qty"])
                if tickers is None:
                    cash_bal += float(r["net"])
                else:
                    flow[d] += -float(r["net"])  # buy = money in, sell = money out
            elif tickers is None and r["type"] == "CASH":
                if r.get("subtype") in EXTERNAL:
                    flow[d] += float(r["net"])
                cash_bal += float(r["net"])
        hold = sum(qty[s] * float(px[s].loc[d]) for s in syms if qty[s] > 1e-9 and pd.notna(px[s].loc[d]))
        value[d] = hold + (cash_bal if (cash and tickers is None) else 0.0)
    return pd.DataFrame({"value": value, "flow": flow})


def twr(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Daily-linked time-weighted return. Flows are assumed start-of-day."""
    v, f = df["value"], df["flow"]
    growth = []
    prev = None
    for d in df.index:
        if prev is None:
            prev = v[d]
            growth.append(1.0)
            continue
        base = prev + f[d]
        growth.append(v[d] / base if base > 1e-9 else 1.0)
        prev = v[d]
    g = pd.Series(growth, index=df.index).cumprod()
    return float(g.iloc[-1] - 1.0), g


def modified_dietz(df: pd.DataFrame) -> float:
    """Money-weighted return. Index 0 is the opening value (its flow is the seed capital);
    a flow at index k occurs at the start of period k and is weighted (n-k+1)/n."""
    v, f = df["value"], df["flow"]
    n = len(df) - 1
    if n <= 0:
        return 0.0
    v0 = float(v.iloc[0])
    wsum = sum(float(f.iloc[k]) * (n - k + 1) / n for k in range(1, n + 1))
    denom = v0 + wsum
    gain = float(v.iloc[-1]) - v0 - float(f.iloc[1:].sum())
    return float(gain / denom) if denom > 1e-9 else 0.0


def reconcile(account: str, loader: Loader, asof: str, baskets: dict[str, set[str]] | None = None,
              fills_dir: Path = FILLS) -> dict[str, Any]:
    rows = load_fills(account, fills_dir)
    df = daily_frame(rows, loader, asof)
    t, g = twr(df)
    spy = loader("SPY").reindex(df.index).ffill()
    ext = [float(r["net"]) for r in rows if r["type"] == "CASH" and r.get("subtype") in EXTERNAL]
    contributed = sum(x for x in ext if x > 0)
    withdrawn = -sum(x for x in ext if x < 0)
    unidentified = sum(float(r["net"]) for r in rows if r["type"] == "CASH" and r.get("subtype") == "UNIDENTIFIED")
    out: dict[str, Any] = {
        "account": account, "start": str(df.index[0].date()), "asof": str(df.index[-1].date()),
        "contributed": round(contributed, 2), "withdrawn": round(withdrawn, 2),
        "net_contributed": round(contributed - withdrawn, 2),
        "ending_value_modelled": round(float(df["value"].iloc[-1]), 2),
        "net_gain": round(float(df["value"].iloc[-1]) - (contributed - withdrawn), 2),
        "unidentified_flows": round(unidentified, 2),
        "twr": t, "mwr_modified_dietz": modified_dietz(df),
        "spy_same_window": float(spy.iloc[-1] / spy.iloc[0] - 1.0),
        "ending_over_net_contributed": float(df["value"].iloc[-1] / (contributed - withdrawn) - 1.0) if contributed - withdrawn > 0 else None,
        "baskets": {},
    }
    for name, tk in (baskets or {}).items():
        sub = daily_frame(rows, loader, asof, tickers=tk)
        nz = sub[(sub["value"] > 0) | (sub["flow"] != 0)]
        if nz.empty:
            continue
        sub = sub.loc[nz.index[0]:]
        # Sub-basket TWR is ill-posed: its "flows" are the trades themselves, whose intraday
        # timing we don't have, and the chain restarts from a near-zero base after a trim or
        # liquidation. Report money-weighted only; TWR is an account-level number.
        liquidated = bool((sub["value"].iloc[1:-1] < 0.02 * sub["value"].max()).any())
        st = None
        sp = spy.loc[sub.index]
        out["baskets"][name] = {
            "tickers": sorted(tk), "start": str(sub.index[0].date()),
            "money_in": round(float(sub["flow"][sub["flow"] > 0].sum()), 2),
            "money_out": round(float(-sub["flow"][sub["flow"] < 0].sum()), 2),
            "ending_value": round(float(sub["value"].iloc[-1]), 2),
            "net_gain": round(float(sub["value"].iloc[-1] - sub["flow"].sum()), 2),
            "twr": st, "liquidated_midway": liquidated, "mwr_modified_dietz": modified_dietz(sub),
            "spy_same_window": float(sp.iloc[-1] / sp.iloc[0] - 1.0),
        }
    return out


def format_reconciliation(r: dict[str, Any]) -> str:
    L = [f"ACCOUNT {r['account']}  {r['start']} → {r['asof']}",
         f"  contributed ${r['contributed']:,.2f}  withdrawn ${r['withdrawn']:,.2f}  net ${r['net_contributed']:,.2f}",
         f"  ending value (modelled from fills × prices) ${r['ending_value_modelled']:,.2f}   net gain ${r['net_gain']:,.2f}"
         + (f"   [includes ${r['unidentified_flows']:+,.2f} unidentified external flow]" if r.get('unidentified_flows') else ""),
         f"  TWR {r['twr']:+.1%}   MWR (Modified Dietz) {r['mwr_modified_dietz']:+.1%}   SPY same window {r['spy_same_window']:+.1%}"]
    if r.get("ending_over_net_contributed") is not None:
        L.append(f"  ending ÷ net-contributed − 1 = {r['ending_over_net_contributed']:+.1%}  (the number people quote; not a return)")
    for n, b in r["baskets"].items():
        L.append(f"  BASKET {n}: {', '.join(b['tickers'])}  from {b['start']}")
        L.append(f"    in ${b['money_in']:,.2f} out ${b['money_out']:,.2f} ending ${b['ending_value']:,.2f} gain ${b['net_gain']:,.2f}")
        L.append(f"    MWR {b['mwr_modified_dietz']:+.1%}  SPY {b['spy_same_window']:+.1%}"
                 + ("  (liquidated mid-window)" if b.get("liquidated_midway") else ""))
    return "\n".join(L)
