#!/usr/bin/env python3
"""scripts/measure_loss_side.py — loss-side exit measurement (exploratory, multi-mechanism).

The excursion analysis rejected partial-exit (carry winners run). The loss mechanism is slow
sideways drift that consumes a position slot, not violent adverse moves. This replays three
LOSS-SIDE exit mechanisms counterfactually on the same 407 carry trades and reports which (if any)
improve net R/trade WITHOUT cutting winners. EXPLORATORY / in-sample (parameters are excursion-
informed) — a promising combo is a CANDIDATE for separate pre-registered OOS validation, not a
validated edge.

READ-ONLY: re-reads the same yfinance OHLC (writes nothing to production), no model training, no
holdout (2024+), no commits. R = realized pnl_pct / (2.0 × ATR%@entry).

    python3 scripts/measure_loss_side.py
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
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from measure_excursion import _load_ohlc, STOP_ATR_MULT  # noqa: E402  (reuse exact OHLC+ATR logic)

DATA = ROOT / "data" / "layer1" / "meta_dataset_v1.parquet"
DOCS = ROOT / "docs" / "layer1"
MD = DOCS / "LOSS_SIDE_ANALYSIS.md"
A_PROGRESS = [0.2, 0.3, 0.4]
A_DAYS = [15, 20, 25, 30]
B_THRESH = [0.15, 0.25, 0.35]
C_MULT = [1.0, 1.25, 1.5]
CAND_BAR = 0.10   # flag combos improving mean R/trade by >= this (in-sample candidate)


def _build_trades(md: pd.DataFrame, ohlc: dict) -> list:
    out = []
    for _, row in md.iterrows():
        pair = row["pair"]; df = ohlc.get(pair)
        if df is None:
            continue
        ed = pd.Timestamp(row["entry_date"]).normalize(); xd = pd.Timestamp(row["exit_date"]).normalize()
        if ed not in df.index:
            continue
        path = df.loc[ed:xd]
        if path.empty:                          # only truly-empty (exit before entry); keep 1-bar
            continue                            # same-day reversal exits (hold_days=1) — they belong
        #                                         in the baseline; loss-side gates simply never fire.
        entry_px = float(path["Open"].iloc[0]); atr = float(df.at[ed, "atr_pct"])
        if not (entry_px > 0 and atr and atr > 0):
            continue
        stop_dist = STOP_ATR_MULT * atr
        d = int(row["direction"])
        closes = path["Close"].to_numpy(float); highs = path["High"].to_numpy(float); lows = path["Low"].to_numpy(float)
        fav = (closes - entry_px) / entry_px * d                       # close-based favorable (pct)
        fav_hi = ((highs - entry_px) if d == 1 else (entry_px - lows)) / entry_px
        mfe_run_R = np.maximum.accumulate(fav_hi) / stop_dist          # running MFE in R
        out.append({
            "pair": pair, "dir": d, "entry_px": entry_px, "stop_dist": stop_dist,
            "closes": closes, "highs": highs, "lows": lows, "fav": fav, "mfe_run_R": mfe_run_R,
            "n": len(path), "actual_R": float(row["realized_r"]) / stop_dist,
            "entry_date": ed, "exit_date": xd,
        })
    return out


def _mech_A(t, X, N):
    if (t["n"] - 1) > N and t["mfe_run_R"][N] < X:
        return t["fav"][N] / t["stop_dist"], N, True            # exit at day N close
    return t["actual_R"], t["n"] - 1, False


def _mech_B(t, X):
    c = t["closes"]; sd = t["stop_dist"]; d = t["dir"]
    for i in range(5, t["n"] - 1):
        ret5 = (c[i] - c[i - 5]) / c[i - 5] * d
        if ret5 / sd < -X:
            return t["fav"][i] / sd, i, True
    return t["actual_R"], t["n"] - 1, False


def _mech_C(t, M):
    sd = t["stop_dist"]; d = t["dir"]; e = t["entry_px"]
    stop_px = e * (1 - M * (sd / STOP_ATR_MULT)) if d == 1 else e * (1 + M * (sd / STOP_ATR_MULT))
    for i in range(1, t["n"] - 1):
        breached = (t["lows"][i] <= stop_px) if d == 1 else (t["highs"][i] >= stop_px)
        if breached:
            return -(M / STOP_ATR_MULT), i, True                # exit at the tighter stop = -M/2 R
    return t["actual_R"], t["n"] - 1, False


def _agg(trades, fn, base_mean):
    cfR, cut, cf_hold, orig_R_cut, days_freed = [], [], [], [], []
    by_pair_cut = {}
    for t in trades:
        r, hold, c = fn(t)
        cfR.append(r); cut.append(c); cf_hold.append(hold)
        if c:
            orig_R_cut.append(t["actual_R"]); days_freed.append((t["n"] - 1) - hold)
            by_pair_cut[t["pair"]] = by_pair_cut.get(t["pair"], 0) + 1
    cfR = np.array(cfR)
    n_cut = int(sum(cut))
    return {
        "mean_R": round(float(cfR.mean()), 4), "delta_R": round(float(cfR.mean() - base_mean), 4),
        "win_rate": round(float((cfR > 0).mean()), 3), "n_cut": n_cut,
        "pct_cut": round(n_cut / len(trades) * 100, 1),
        "cut_winners_mean_origR": round(float(np.mean(orig_R_cut)), 3) if orig_R_cut else None,
        "cut_winner_rate": round(float((np.array(orig_R_cut) > 0).mean()), 3) if orig_R_cut else None,
        "slot_days_freed": int(sum(days_freed)),
        "by_pair_cut": by_pair_cut,
    }


def _slot_occupancy(trades):
    if not trades:
        return {}
    lo = min(t["entry_date"] for t in trades); hi = max(t["exit_date"] for t in trades)
    days = pd.bdate_range(lo, hi)
    conc = np.zeros(len(days), int)
    idx = {d: i for i, d in enumerate(days)}
    for t in trades:
        a, b = idx.get(t["entry_date"]), idx.get(t["exit_date"])
        if a is not None and b is not None:
            conc[a:b + 1] += 1
    occ = conc[conc > 0]
    return {
        "pct_days_ge2_open": round(float((occ >= 2).mean()) * 100, 1),
        "pct_days_ge3_open": round(float((occ >= 3).mean()) * 100, 1),
        "max_concurrent": int(conc.max()),
        "_conc": conc,
    }


def main() -> dict:
    DOCS.mkdir(parents=True, exist_ok=True)
    md = pd.read_parquet(DATA).reset_index()
    ohlc = _load_ohlc()
    trades = _build_trades(md, ohlc)
    base = np.array([t["actual_R"] for t in trades])
    base_mean = float(base.mean()); base_win = float((base > 0).mean())
    n = len(trades); unmatched = len(md) - n
    slot = _slot_occupancy(trades)

    A = {f"X{X}_d{N}": _agg(trades, lambda t, X=X, N=N: _mech_A(t, X, N), base_mean) for X in A_PROGRESS for N in A_DAYS}
    B = {f"thr{X}": _agg(trades, lambda t, X=X: _mech_B(t, X), base_mean) for X in B_THRESH}
    C = {f"atr{M}": _agg(trades, lambda t, M=M: _mech_C(t, M), base_mean) for M in C_MULT}

    def best(d):
        return max(d.items(), key=lambda kv: kv[1]["delta_R"])

    out = {
        "n_trades": n, "n_unmatched": unmatched,
        "baseline": {"mean_R": round(base_mean, 4), "win_rate": round(base_win, 3)},
        "slot": {k: v for k, v in slot.items() if k != "_conc"},
        "mech_A": A, "mech_B": B, "mech_C": C,
        "best": {"A": best(A), "B": best(B), "C": best(C)},
        "candidates": sorted(
            [(f"A:{k}", v["delta_R"]) for k, v in A.items() if v["delta_R"] >= CAND_BAR]
            + [(f"B:{k}", v["delta_R"]) for k, v in B.items() if v["delta_R"] >= CAND_BAR]
            + [(f"C:{k}", v["delta_R"]) for k, v in C.items() if v["delta_R"] >= CAND_BAR],
            key=lambda x: -x[1]),
        "framing": "EXPLORATORY / in-sample (params excursion-informed). Candidates need separate OOS validation.",
    }

    # ── figures ──
    f, ax = plt.subplots(figsize=(6, 4))
    grid = np.array([[A[f"X{X}_d{N}"]["delta_R"] for N in A_DAYS] for X in A_PROGRESS])
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(A_DAYS))); ax.set_xticklabels(A_DAYS); ax.set_yticks(range(len(A_PROGRESS))); ax.set_yticklabels(A_PROGRESS)
    ax.set_xlabel("day_N"); ax.set_ylabel("no_progress_R"); ax.set_title("Mech A: ΔR/trade vs baseline")
    for i in range(len(A_PROGRESS)):
        for j in range(len(A_DAYS)):
            ax.text(j, i, f"{grid[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    f.colorbar(im); f.tight_layout(); f.savefig(DOCS / "loss_A_heatmap.png", dpi=90); plt.close(f)

    f, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(slot["_conc"]); ax.axhline(2, color="r", ls="--", lw=1, label="live max-concurrent cap (2)")
    ax.set_title(f"Concurrent open positions over time (max {slot['max_concurrent']}; "
                 f"{slot['pct_days_ge2_open']}% of active days ≥2)"); ax.set_ylabel("# open"); ax.legend()
    f.tight_layout(); f.savefig(DOCS / "loss_slot_occupancy.png", dpi=90); plt.close(f)

    _write_md(out)
    return out


def _row(name, v):
    return (f"| {name} | {v['delta_R']:+} | {v['mean_R']} | {v['win_rate']} | {v['pct_cut']}% | "
            f"{v['cut_winners_mean_origR']} | {v['slot_days_freed']} |")


def _write_md(o: dict) -> None:
    b = o["baseline"]; s = o["slot"]
    L = ["# Loss-side exit measurement — HYP-064 carry trades", "",
         f"> {o['framing']}  READ-ONLY; holdout untouched. R = pnl_pct / (2·ATR%@entry).", "",
         f"**Trades:** {o['n_trades']} (unmatched {o['n_unmatched']})  ·  **baseline mean R "
         f"{b['mean_R']}** / win {b['win_rate']}", "",
         f"**Slot contention:** max concurrent {s.get('max_concurrent')}; "
         f"{s.get('pct_days_ge2_open')}% of active days had ≥2 open, {s.get('pct_days_ge3_open')}% ≥3 "
         "(how often the live 2-position cap would bind → whether freeing a slot has value).", "",
         "## Decision matrix (Δ vs baseline; cut-winner-R>0 means we're cutting winners — bad)", "",
         "| mechanism | ΔR/trade | mean R | win rate | % cut | cut-trades mean orig-R | slot-days freed |",
         "|---|---|---|---|---|---|---|"]
    for k, v in o["mech_A"].items():
        L.append(_row(f"A {k}", v))
    for k, v in o["mech_B"].items():
        L.append(_row(f"B {k}", v))
    for k, v in o["mech_C"].items():
        L.append(_row(f"C {k}", v))
    bestA, bestB, bestC = o["best"]["A"], o["best"]["B"], o["best"]["C"]
    L += ["", "## Best per mechanism", "",
          f"- **A (time-velocity):** {bestA[0]} → ΔR {bestA[1]['delta_R']:+}/trade, "
          f"cut {bestA[1]['pct_cut']}%, cut-trades orig-R mean {bestA[1]['cut_winners_mean_origR']}",
          f"- **B (adverse-velocity):** {bestB[0]} → ΔR {bestB[1]['delta_R']:+}/trade, cut {bestB[1]['pct_cut']}%, "
          f"cut-trades orig-R mean {bestB[1]['cut_winners_mean_origR']}",
          f"- **C (tighter ATR stop, control):** {bestC[0]} → ΔR {bestC[1]['delta_R']:+}/trade "
          f"(expected ~0 per the MAE table)", "",
          "## Candidates for OOS validation (ΔR ≥ +0.10, IN-SAMPLE — not validated edge)", ""]
    if o["candidates"]:
        for name, dr in o["candidates"]:
            L.append(f"- {name}: ΔR {dr:+}/trade — flag for separate pre-registered OOS test")
    else:
        L.append("- **NONE.** No mechanism clears +0.10R/trade in-sample. The loss-side lever, as "
                 "formulated, does not show meaningful uplift on this dataset — like partial-exit, the "
                 "intuition does not survive measurement. Re-think the mechanism or accept the edge as-is.")
    L += ["", "## Figures", "", "- ![Mech A heatmap](loss_A_heatmap.png)",
          "- ![Slot occupancy](loss_slot_occupancy.png)", ""]
    MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    o = main()
    bA, bB, bC = o["best"]["A"], o["best"]["B"], o["best"]["C"]
    print("=" * 74)
    print(f"LOSS-SIDE MEASUREMENT — {o['n_trades']} trades (unmatched {o['n_unmatched']})")
    print(f"  baseline mean R {o['baseline']['mean_R']} / win {o['baseline']['win_rate']}")
    print(f"  slot contention: {o['slot'].get('pct_days_ge2_open')}% active days ≥2 open, "
          f"{o['slot'].get('pct_days_ge3_open')}% ≥3 (max {o['slot'].get('max_concurrent')})")
    print(f"  BEST A (time-velocity)  {bA[0]}: ΔR {bA[1]['delta_R']:+}/trade, cut {bA[1]['pct_cut']}%, "
          f"cut-trades orig-R {bA[1]['cut_winners_mean_origR']}")
    print(f"  BEST B (adverse-vel)    {bB[0]}: ΔR {bB[1]['delta_R']:+}/trade, cut {bB[1]['pct_cut']}%")
    print(f"  BEST C (tighter stop)   {bC[0]}: ΔR {bC[1]['delta_R']:+}/trade (control)")
    print(f"  CANDIDATES (ΔR≥+0.10, in-sample): {[c[0] for c in o['candidates']] or 'NONE'}")
    print("=" * 74)
