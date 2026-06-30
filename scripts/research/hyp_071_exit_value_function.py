#!/usr/bin/env python3
"""scripts/research/hyp_071_exit_value_function.py — HYP-071 Step 2 (the table).

Computes the tabular exit value function locked by:
  data/research/preregister/HYP-071_tabular_exit_value.yaml          (v2, hash 3d500bda…)
  data/research/preregister/HYP-071_interpretation_notes.yaml        (addendum, hash c1fab80…)

HARD, NON-NEGOTIABLE SEQUENCE:
  1. verify BOTH hashes  → HALT on mismatch
  2. reconcile gate: the unbucketed, un-resampled full-decade portfolio Sharpe MUST equal 0.6886 ± 0.01
     out of this harness, else HALT — not one cell is computed
  3. only then: build the regime-conditional block bootstrap (L=5) and compute V(cell, action)
  4. validation: CPCV sign-stability, forward agreement, separability, static-config divergence,
     regime-window robustness, signal-frozen sensitivity
  5. write data/research/HYP-071_tabular_exit_value_results.json (PROVISIONAL verdict; read with Colin)

NO LIVE TOUCH: no OANDA, no forex_exit_manager state writes, no config changes. Reads yfinance + the
canonical backtest; the exit arithmetic comes ONLY from exit_machine (live == backtest by construction).

    python3 scripts/research/hyp_071_exit_value_function.py            # verify + reconcile + run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.forex.exit_machine import ExitConfig
from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY
from sovereign.reporting.equity_curve import weighted_portfolio_sharpe
from sovereign.discovery import exit_value_table as evt

PREREG = ROOT / "data" / "research" / "preregister" / "HYP-071_tabular_exit_value.yaml"
ADDENDUM = ROOT / "data" / "research" / "preregister" / "HYP-071_interpretation_notes.yaml"
OUT = ROOT / "data" / "research" / "HYP-071_tabular_exit_value_results.json"
TRADES_FILE = ROOT / "logs" / "forex_backtest_trades.json"
RESULTS_JSON = ROOT / "logs" / "forex_backtest_results.json"

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
DECADE = ("2015-01-01", "2024-12-31")
FULL = ("2015-01-01", "2026-06-30")
RECON_TARGET, RECON_TOL = 0.6886, 0.01
N_CONT = 10_000

# member-population windows (sliced by entry-date year), per the v2 prereg + addendum
WINDOWS = {
    "decade":   (2015, 2024),
    "oos":      (2023, 2024),
    "forward":  (2025, 2026),
    "hiking":   (2022, 2023),
    "cutting_2021": (2021, 2021),
    "cutting_2024": (2024, 2024),
}
WIN_ID = {k: i for i, k in enumerate(["decade", "oos", "forward", "hiking", "cutting", "decade252", "frozen"])}


# ── prereg / addendum verification (YAML-aware, same canonical-hash math as the JSON preregs) ─────
def _canonical_hash(path: Path) -> str:
    doc = yaml.safe_load(path.read_text())
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_locked():
    for path, name in ((PREREG, "prereg v2"), (ADDENDUM, "addendum")):
        doc = yaml.safe_load(path.read_text())
        h = _canonical_hash(path)
        if doc.get("hash_lock") != h:
            raise SystemExit(f"HASH MISMATCH on {name} ({path.name}) — frozen design altered.\n"
                             f"  stored:   {doc.get('hash_lock')}\n  computed: {h}")
        print(f"  {name} hash OK ({h[:16]}…)")


# ── reconcile gate (canonical decade backtest; HALT on drift) ─────────────────────────────────────
def reconcile_gate() -> float:
    bt = ForexBacktester(start=DECADE[0], end=DECADE[1])
    results = bt.backtest_all()
    ws = weighted_portfolio_sharpe([(r.sharpe, r.total_trades) for r in results])
    print(f"  reconcile: weighted portfolio Sharpe = {ws}  (target {RECON_TARGET} ± {RECON_TOL})")
    if abs(ws - RECON_TARGET) > RECON_TOL:
        raise SystemExit(f"RECONCILE FAILED — harness Sharpe {ws} != {RECON_TARGET}±{RECON_TOL}. "
                         f"Halting (config/data drift). NOT ONE CELL COMPUTED.")
    return ws


# ── per-pair cache (mirror of HYP-066 pair_arrays; 60-day percentile + RSI + tercile) ─────────────
def build_cache(bt: ForexBacktester, pair: str, pct_window: int, pair_idx: int) -> dict | None:
    cfg = PAIR_CONFIG.get(pair)
    df = bt._download_price(pair)
    if df is None or len(df) < 252:
        return None
    base = CB_TO_COUNTRY[cfg.base_central_bank]
    quote = CB_TO_COUNTRY[cfg.quote_central_bank]
    sig = bt._get_pair_signals(df=df, base_country=base, quote_country=quote, pair=pair, hold_days=bt.HOLD_DAYS)
    if pair in bt.PAIR_VIX_GATES:
        sig = bt._apply_vix_regime_gate(sig, pair=pair, start=bt.start, end=bt.end)
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    opens = df["Open"] if "Open" in df.columns else close
    idx = close.index
    atr = np.asarray(bt._signals._compute_atr_pct(close, df), dtype=float)
    pct = evt.trailing_pct(atr, pct_window)
    hold_col = "hold_days" if "hold_days" in sig.columns else "hold"
    return {
        "pair": pair, "pair_idx": pair_idx, "idx": idx, "pos": {ts: i for i, ts in enumerate(idx)},
        "opens": opens.to_numpy(dtype=float), "closes": close.to_numpy(dtype=float),
        "atr": atr, "atr_pct": pct, "tercile": pct,
        "rsi": evt.compute_rsi(close.to_numpy(dtype=float)),
        "signal": sig["signal"].reindex(idx).fillna(0).to_numpy(dtype=float).astype(np.int64),
        "hold": sig[hold_col].reindex(idx).fillna(bt.HOLD_DAYS).to_numpy(dtype=float).astype(np.int64),
    }


def cfg_from_bt(bt: ForexBacktester) -> dict:
    """ExitConfig per pair-index, sourced from the live backtester (parity-safe)."""
    out = {}
    for i, p in enumerate(PAIRS):
        out[i] = ExitConfig(bt.STOP_ATR_MULT, bt.PAIR_TRAILING_OVERRIDES.get(p, bt.TRAILING_ATR_MULT),
                            bt.strict_mode, not bt.strict_mode)
    return out


def _in_window(members, lo_yr, hi_yr):
    """Filter a list of member dicts by entry-date year."""
    return [m for m in members if lo_yr <= pd.Timestamp(m["entry_dt"]).year <= hi_yr]


def main() -> int:
    ap = argparse.ArgumentParser(description="HYP-071 tabular exit value function (Step 2).")
    ap.add_argument("--standalone", action="store_true", help="(HYP-071 is independent; flag kept for house symmetry)")
    ap.parse_args()

    print("HYP-071 Step 2 — tabular exit value function")
    print("1) verify locked design")
    verify_locked()

    print("2) reconcile gate (HALT on drift)")
    trades_backup = TRADES_FILE.read_text() if TRADES_FILE.exists() else None
    results_backup = RESULTS_JSON.read_text() if RESULTS_JSON.exists() else None
    ws = reconcile_gate()

    try:
        print("3) full-history backtest for the member population")
        bt = ForexBacktester(start=FULL[0], end=FULL[1])
        bt.backtest_all()                                   # writes the full ledger
        ledger = json.loads(TRADES_FILE.read_text())
        cfg_by_pair = cfg_from_bt(bt)

        print("4) build caches (60-day + 252-day) and re-trace members (parity HALT)")
        caches60, caches252 = [], []
        members60, members252 = [], []
        parity = {"n": 0, "matched": 0, "dropped": 0}
        for i, pair in enumerate(PAIRS):
            c60 = build_cache(bt, pair, 60, i)
            c252 = build_cache(bt, pair, 252, i)
            if c60 is None or c252 is None:
                raise SystemExit(f"cache build failed for {pair}")
            caches60.append(c60); caches252.append(c252)
            trades = ledger.get(pair, [])
            m60, par = evt.retrace_members(c60, trades, cfg_by_pair[i], i)
            m252, _ = evt.retrace_members(c252, trades, cfg_by_pair[i], i)
            members60 += m60; members252 += m252
            for k in parity:
                parity[k] += par[k]
        scored = parity["n"] - parity["dropped"]
        match_rate = parity["matched"] / scored if scored else 0.0
        print(f"   parity: {parity['matched']}/{scored} match (dropped {parity['dropped']}); rate {match_rate:.4f}")
        if match_rate < 0.99:
            raise SystemExit(f"RETRACE PARITY FAILED ({match_rate:.4f} < 0.99) — re-trace diverged from "
                             f"the canonical ledger. Halting; the table would not be backtest-faithful.")

        print("5) pools + per-window tables")
        pool60 = evt.build_return_pool(caches60)
        pool252 = evt.build_return_pool(caches252)

        def table_for(members, lo, hi, pool, win_id, signal_mode="live"):
            by_cell = evt.group_members_by_cell(_in_window(members, lo, hi))
            return by_cell, evt.compute_table(by_cell, pool, cfg_by_pair, N_CONT, win_id, signal_mode)

        decade_cells, decade_tbl = table_for(members60, *WINDOWS["decade"], pool60, WIN_ID["decade"])
        _, oos_tbl = table_for(members60, *WINDOWS["oos"], pool60, WIN_ID["oos"])
        _, fwd_tbl = table_for(members60, *WINDOWS["forward"], pool60, WIN_ID["forward"])
        _, hik_tbl = table_for(members60, *WINDOWS["hiking"], pool60, WIN_ID["hiking"])
        cut_members = _in_window(members60, *WINDOWS["cutting_2021"]) + _in_window(members60, *WINDOWS["cutting_2024"])
        cut_cells = evt.group_members_by_cell(cut_members)
        cut_tbl = evt.compute_table(cut_cells, pool60, cfg_by_pair, N_CONT, WIN_ID["cutting"])
        d252_cells, d252_tbl = table_for(members252, *WINDOWS["decade"], pool252, WIN_ID["decade252"])
        _, frozen_tbl = table_for(members60, *WINDOWS["decade"], pool60, WIN_ID["frozen"], signal_mode="frozen")

        print("6) CPCV sign-stability per decade cell")
        cpcv = {}
        for cid, ma in decade_cells.items():
            cpcv[cid] = evt.cpcv_sign_stability(ma, caches60, cfg_by_pair, decade_tbl[cid].margin,
                                                WIN_ID["decade"], "live")

        print("7) assemble results")
        cells_out = {}
        exit_now_cells, sensible_stable = [], []
        for cid in range(evt.N_CELLS):
            if not evt.is_evaluated(cid):
                cells_out[str(cid)] = {"evaluated": False, "reason": "carry not-aligned → REVERSAL / N/A"}
                continue
            r = decade_tbl.get(cid)
            if r is None:
                cells_out[str(cid)] = {"evaluated": True, "n_members": 0, "note": "no members"}
                continue
            cp = cpcv.get(cid, {})
            action = r.optimal_action
            sensible = evt.is_economically_sensible(cid, action) and bool(cp.get("sign_stable"))
            if action == "EXIT_NOW":
                exit_now_cells.append(cid)
            if sensible:
                sensible_stable.append(cid)
            oa = oos_tbl[cid].optimal_action if cid in oos_tbl else None
            fa = fwd_tbl[cid].optimal_action if cid in fwd_tbl else None
            cells_out[str(cid)] = {
                "evaluated": True, "coords": evt.decode_cell(cid), "n_members": r.n_members,
                "V_hold": round(r.V_hold, 5), "V_exit": round(r.V_exit, 5), "E_hold": round(r.E_hold, 5),
                "DD_hold": round(r.DD_hold, 5), "margin": round(r.margin, 5), "optimal_action": action,
                "static_action": "HOLD_AND_TRAIL", "agree_with_static": action == "HOLD_AND_TRAIL",
                "cpcv": cp, "economically_sensible": sensible,
                "forward": {"oos_2023_24": oa, "forward_2025_26": fa, "agree": (oa is not None and oa == fa)},
                "per_regime": {"hiking": hik_tbl[cid].optimal_action if cid in hik_tbl else None,
                               "cutting": cut_tbl[cid].optimal_action if cid in cut_tbl else None},
                "regime_window_252": d252_tbl[cid].optimal_action if cid in d252_tbl else None,
                "signal_frozen_action": frozen_tbl[cid].optimal_action if cid in frozen_tbl else None,
            }

        # summary fractions over cells present in both compared windows
        def agreement(t_a, t_b):
            common = set(t_a) & set(t_b)
            return (float(np.mean([t_a[c].optimal_action == t_b[c].optimal_action for c in common])), len(common)) \
                if common else (None, 0)
        fwd_frac, fwd_n = agreement(oos_tbl, fwd_tbl)
        sep_frac, sep_n = agreement(hik_tbl, cut_tbl)
        rw_frac, rw_n = agreement(decade_tbl, d252_tbl)
        # sensible+stable divergences whose forward action also agrees
        div_fwd_agree = [c for c in sensible_stable
                         if cells_out[str(c)]["forward"]["agree"] and cells_out[str(c)]["forward"]["oos_2023_24"] == "EXIT_NOW"]
        verdict = "PASS" if div_fwd_agree else "FAIL"

        result = {
            "id": "HYP-071", "prereg_hash": yaml.safe_load(PREREG.read_text())["hash_lock"],
            "addendum_hash": yaml.safe_load(ADDENDUM.read_text())["hash_lock"],
            "verdict": verdict, "verdict_status": "PROVISIONAL — read with Colin against the locked §7 gate",
            "reconcile_weighted_sharpe": ws, "block_length_L": evt.BLOCK_L, "n_continuations": N_CONT,
            "pct_window": evt.PCT_WINDOW, "lambda": evt.LAMBDA,
            "parity": {**parity, "match_rate": round(match_rate, 5)},
            "n_members": {"decade": sum(len(v) for v in decade_cells.values()),
                          "forward": len(_in_window(members60, *WINDOWS["forward"]))},
            "cells": cells_out,
            "summary": {
                "forward_agreement_fraction": fwd_frac, "forward_cells_compared": fwd_n,
                "separability_agreement": sep_frac, "separability_cells_compared": sep_n,
                "regime_window_agreement": rw_frac, "regime_window_cells_compared": rw_n,
                "regime_window_robust": (rw_frac is not None and rw_frac >= 0.9),
                "n_exit_now_cells": len(exit_now_cells),
                "n_cpcv_stable_sensible_divergences": len(sensible_stable),
                "n_divergences_forward_consistent": len(div_fwd_agree),
                "gross_R_caveat": "R is gross (locked); swap/carry would shift marginal EXIT_NOW cells toward EXIT.",
                "forward_window_thin_note": f"forward (2025-26) members = {len(_in_window(members60, *WINDOWS['forward']))}; treat its agreement cautiously.",
                "interpretation_caveat": "Vol innovations are ~white; any structure is excursion geometry "
                                         "conditioned on entry regime, NOT intra-trade vol momentum.",
                "pass_fail_sentence": (
                    f"{verdict} — {len(sensible_stable)} CPCV-stable economically-sensible EXIT_NOW divergence(s); "
                    f"{len(div_fwd_agree)} of them agree across the 2023-24 and 2025-26 windows."),
            },
        }
        OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"\n   wrote {OUT.relative_to(ROOT)}")
        print(f"   VERDICT (provisional): {verdict}")
        print(f"   EXIT_NOW cells: {len(exit_now_cells)} | CPCV-stable sensible: {len(sensible_stable)} | "
              f"forward-consistent: {len(div_fwd_agree)}")
        print(f"   forward agree {fwd_frac} (n={fwd_n}) | separability {sep_frac} (n={sep_n}) | "
              f"regime-window {rw_frac} (n={rw_n})")
    finally:
        if trades_backup is not None:
            TRADES_FILE.write_text(trades_backup)            # restore the canonical decade ledger
        if results_backup is not None:
            RESULTS_JSON.write_text(results_backup)
        print("   restored canonical decade ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
