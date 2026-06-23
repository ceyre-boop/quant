#!/usr/bin/env python3
"""
Cross-pair regime screen (REGIME-ROUTER-SCREEN) — RESEARCH ONLY, produces a TABLE.

Question: does the proven USDJPY Bull+VIX>15 gate (HYP-027) generalize across pairs? For each
pair, measure √n Sharpe IN vs OUT of the Bull+VIX regime (SPY>200SMA AND VIX>15), on IS and OOS.
The signal we're hunting: a pair that THRIVES in the regime that kills USDJPY → a cross-pair
router candidate (route freed capital there when USDJPY is gated out).

Discipline:
  - Pre-registered + frozen (data/research/preregister/regime_router_screen.json). This script
    ASSERTS the frozen constants match (tripwire) and refuses to run a tuned spec.
  - Both-sides (NN#2): every pair reports Sharpe in BOTH regime states.
  - SINGLE frozen VIX threshold (15) — not a swept grid (the HYP-044 multiple-comparisons trap).
  - Output is a TABLE. No gating decision, no live change. A candidate becomes a hypothesis ONLY
    via scripts/run_hypothesis.py (permutation + IS/OOS + BH + decay gate).

Reuses sovereign/forex/forex_backtester (backtest_pair, the Bull+VIX gate). Touches no live config.

Run:  python3 scripts/regime_screen.py
"""
from __future__ import annotations

import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

PREREG = ROOT / "data" / "research" / "preregister" / "regime_router_screen.json"
OUT = ROOT / "data" / "research" / "regime_screen.json"


def _inverse_vix_gate(self, signals, pair, start, end):
    """Inverse of _apply_vix_regime_gate: keep ONLY in-regime trades (Bull AND VIX>threshold),
    zero everything else. The both-sides counterpart used to measure in-regime Sharpe."""
    import pandas as pd
    vix_threshold = self.PAIR_VIX_GATES.get(pair)
    if vix_threshold is None:
        return signals
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False)
        vix = yf.download("^VIX", start=start, end=end, progress=False)
        for df_ in (spy, vix):
            if hasattr(df_.columns, "get_level_values"):
                df_.columns = df_.columns.get_level_values(0)
            df_.index = pd.to_datetime(df_.index).tz_localize(None)
        spy["sma200"] = spy["Close"].rolling(200).mean()
        spy["is_bull"] = spy["Close"] > spy["sma200"]
        signals = signals.copy()
        for date in signals[signals["signal"] != 0].index:
            try:
                in_regime = bool(spy["is_bull"].asof(date)) and float(vix["Close"].asof(date)) > vix_threshold
                if not in_regime:
                    signals.loc[date, "signal"] = 0.0
            except Exception:
                pass
    except Exception:
        pass
    return signals


def _sharpe(pair: str, start: str, end: str, threshold: float | None, inverse: bool) -> dict:
    """Run one pair backtest. threshold=None -> ungated; else gated at threshold; inverse -> in-regime only."""
    from sovereign.forex.forex_backtester import ForexBacktester
    bt = ForexBacktester(start=start, end=end)
    if threshold is None:
        bt.PAIR_VIX_GATES = {}                      # ungated: all trades
    else:
        bt.PAIR_VIX_GATES = {pair: float(threshold)}
        if inverse:
            bt._apply_vix_regime_gate = types.MethodType(_inverse_vix_gate, bt)
    try:
        r = bt.backtest_pair(pair)
    except Exception as e:  # noqa: BLE001
        return {"sharpe": None, "n": 0, "err": type(e).__name__}
    if r is None:
        return {"sharpe": None, "n": 0}
    return {"sharpe": round(float(r.sharpe), 3), "n": int(r.total_trades),
            "win_rate": round(float(r.win_rate), 3)}


def main() -> dict:
    p = json.loads(PREREG.read_text())
    thr = float(p["vix_threshold"])
    pairs = p["pairs"]
    IS, OOS = p["splits"]["IS"], p["splits"]["OOS"]
    # tripwire — refuse a tuned spec
    assert thr == 15.0, "vix_threshold drifted from pre-registration"
    assert IS == ["2015-01-01", "2022-12-31"] and OOS == ["2023-01-01", "2024-12-31"], "splits drifted"

    rows = []
    for pair in pairs:
        cell: dict = {"pair": pair}
        for win_name, (s, e) in (("IS", IS), ("OOS", OOS)):
            ung = _sharpe(pair, s, e, None, False)
            gat = _sharpe(pair, s, e, thr, False)       # out-of-regime trades (the gate keeps these)
            inr = _sharpe(pair, s, e, thr, True)        # in-regime trades only (both-sides)
            delta = (round(gat["sharpe"] - ung["sharpe"], 3)
                     if gat["sharpe"] is not None and ung["sharpe"] is not None else None)
            cell[win_name] = {
                "sharpe_ungated": ung["sharpe"], "n_ungated": ung["n"],
                "sharpe_gated_out_regime": gat["sharpe"], "n_out_regime": gat["n"],
                "sharpe_in_regime": inr["sharpe"], "n_in_regime": inr["n"],
                "delta_gated_minus_ungated": delta,
            }
        rows.append(cell)

    payload = {
        "id": "REGIME-ROUTER-SCREEN",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "regime": f"Bull (SPY>200SMA) AND VIX>{thr}",
        "splits": {"IS": IS, "OOS": OOS},
        "rows": rows,
        "provenance": {
            "kind": "regime_screen_table",
            "note": "Research table only. No gating decision. A cross-pair gate becomes a hypothesis "
                    "ONLY via scripts/run_hypothesis.py (permutation>=1000 + IS/OOS + BH + decay>=0.50). "
                    "Not labelled v016 pre-gauntlet. No live config touched.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def _fmt(x):
    return f"{x:>6}" if x is not None else "   n/a"


if __name__ == "__main__":
    pl = main()
    print(f"\n{'='*100}\n  REGIME SCREEN — {pl['regime']}  (both-sides; research table only)\n{'='*100}")
    for win in ("IS", "OOS"):
        print(f"\n  [{win}]  {'pair':10} {'ungated':>8} {'out_regime':>11} {'in_regime':>10} {'delta':>7}   n(ung/out/in)")
        for r in pl["rows"]:
            c = r[win]
            print(f"  {'':9}{r['pair']:10} {_fmt(c['sharpe_ungated']):>8} "
                  f"{_fmt(c['sharpe_gated_out_regime']):>11} {_fmt(c['sharpe_in_regime']):>10} "
                  f"{_fmt(c['delta_gated_minus_ungated']):>7}   "
                  f"{c['n_ungated']}/{c['n_out_regime']}/{c['n_in_regime']}")
    print(f"\n  READ: a pair with strong +Sharpe in_regime is a router candidate (thrives when USDJPY is gated).")
    print(f"        positive 'delta' = gating helps that pair (out-regime > ungated). NEXT: gauntlet, not a decision.")
    print(f"  Saved: data/research/regime_screen.json\n{'='*100}\n")
