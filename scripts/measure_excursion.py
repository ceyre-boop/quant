#!/usr/bin/env python3
"""scripts/measure_excursion.py — MFE/MAE excursion analysis on the HYP-064 carry trades.

READ-ONLY descriptive measurement to confirm/reject the L2 partial-exit-at-intermediate-R thesis.
Reconstructs each of the 408 carry trades' intra-trade price path from the SAME yfinance daily OHLC
HYP-064 used (read-only; writes nothing to production), and measures max-favorable / max-adverse
excursion in R-multiples (R = excursion / (2.0 × ATR%@entry)). No live changes, no model training,
no backtester re-run, no production edits.

    python3 scripts/measure_excursion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "data" / "layer1" / "meta_dataset_v1.parquet"
DOCS = ROOT / "docs" / "layer1"
MD = DOCS / "EXCURSION_ANALYSIS.md"
STOP_ATR_MULT = 2.0
THRESHOLDS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
PAIR_YF = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
           "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X"}


def _load_ohlc() -> dict:
    """Daily OHLC + ATR%(14) per pair (read-only; identical bounds to HYP-064)."""
    import yfinance as yf
    out = {}
    for pair, tk in PAIR_YF.items():
        df = yf.download(tk, start="2015-01-01", end="2024-01-01", auto_adjust=True, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index = pd.to_datetime(df.index).normalize()
        prev = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                        (df["Low"] - prev).abs()], axis=1).max(axis=1)
        df["atr_pct"] = (tr.rolling(14).mean() / df["Close"])
        out[pair] = df
    return out


def _excursions(row, ohlc: dict):
    pair = row["pair"]
    df = ohlc.get(pair)
    if df is None:
        return None
    entry_d = pd.Timestamp(row["entry_date"]).normalize()
    exit_d = pd.Timestamp(row["exit_date"]).normalize()
    if entry_d not in df.index:
        return None
    entry_px = float(df.at[entry_d, "Open"])
    atr_pct = float(df.at[entry_d, "atr_pct"])
    if not (entry_px > 0 and atr_pct and atr_pct > 0):
        return None
    stop_dist = STOP_ATR_MULT * atr_pct                      # in fractional price terms
    path = df.loc[entry_d:exit_d]
    if path.empty:
        return None
    d = int(row["direction"])
    if d == 1:
        mfe = ((path["High"] - entry_px) / entry_px).max()
        mae = ((entry_px - path["Low"]) / entry_px).max()
        mfe_close = ((path["Close"] - entry_px) / entry_px).max()
    else:
        mfe = ((entry_px - path["Low"]) / entry_px).max()
        mae = ((path["High"] - entry_px) / entry_px).max()
        mfe_close = ((entry_px - path["Close"]) / entry_px).max()
    return {
        "mfe_R": float(mfe / stop_dist), "mae_R": float(mae / stop_dist),
        "mfe_close_R": float(mfe_close / stop_dist),
        "actual_R": float(row["realized_r"] / stop_dist),
    }


def _pct_reaching(series: pd.Series, ths) -> dict:
    return {t: round(float((series >= t).mean()) * 100, 1) for t in ths}


def main() -> dict:
    DOCS.mkdir(parents=True, exist_ok=True)
    md = pd.read_parquet(DATA).reset_index()   # entry_date + pair become columns
    ohlc = _load_ohlc()

    recs, unmatched = [], 0
    for idx, row in md.iterrows():
        r = _excursions(row, ohlc)
        if r is None:
            unmatched += 1
            continue
        recs.append({**r, "exit_reason": row["exit_reason"], "pair": row["pair"],
                     "realized_r": float(row["realized_r"]), "win": int(row["meta_win"])})
    t = pd.DataFrame(recs)

    # ── plots ──
    def hist(ax, data, label=None, bins=30, rng=(-1, 4)):
        ax.hist(np.clip(data, *rng), bins=bins, alpha=0.6, label=label)

    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    hist(ax[0], t["mfe_R"]); ax[0].axvline(0.5, color="r", ls="--", lw=1, label="+0.5R")
    ax[0].set_title(f"MFE_R (all {len(t)} trades)"); ax[0].set_xlabel("MFE (R)"); ax[0].legend()
    hist(ax[1], t["mae_R"], rng=(0, 3)); ax[1].set_title("MAE_R (all trades)"); ax[1].set_xlabel("MAE (R)")
    f.tight_layout(); f.savefig(DOCS / "excursion_mfe_overall.png", dpi=90); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 4))
    for er, g in t.groupby("exit_reason"):
        hist(ax, g["mfe_R"], label=f"{er} (n={len(g)})")
    ax.axvline(0.5, color="r", ls="--", lw=1); ax.set_title("MFE_R by exit_reason"); ax.set_xlabel("MFE (R)"); ax.legend()
    f.tight_layout(); f.savefig(DOCS / "excursion_mfe_by_exit.png", dpi=90); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 4))
    hist(ax, t.loc[t.win == 1, "mfe_R"], label="winners"); hist(ax, t.loc[t.win == 0, "mfe_R"], label="losers")
    ax.axvline(0.5, color="r", ls="--", lw=1); ax.set_title("MFE_R winners vs losers"); ax.set_xlabel("MFE (R)"); ax.legend()
    f.tight_layout(); f.savefig(DOCS / "excursion_mfe_by_winloss.png", dpi=90); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 4))
    for p, g in t.groupby("pair"):
        hist(ax, g["mfe_R"], label=f"{p} (n={len(g)})")
    ax.axvline(0.5, color="r", ls="--", lw=1); ax.set_title("MFE_R by pair"); ax.set_xlabel("MFE (R)"); ax.legend()
    f.tight_layout(); f.savefig(DOCS / "excursion_mfe_by_pair.png", dpi=90); plt.close(f)

    # ── load-bearing: time-exit trades that reached MFE>=0.5R ──
    te = t[t.exit_reason == "time"]
    te_touch = te[te.mfe_R >= 0.5]
    frac_touch = float(len(te_touch) / len(te)) if len(te) else 0.0

    f, ax = plt.subplots(figsize=(9, 4))
    if len(te_touch):
        hist(ax, te_touch["actual_R"], rng=(-1, 3), bins=25)
    ax.axvline(0.5, color="g", ls="--", lw=1, label="+0.5R (TP)"); ax.axvline(0.3, color="orange", ls=":", lw=1, label="+0.3R")
    ax.set_title(f"Time-exit trades w/ MFE>=0.5R (n={len(te_touch)}): FINAL realized R")
    ax.set_xlabel("final realized R"); ax.legend()
    f.tight_layout(); f.savefig(DOCS / "excursion_timeexit_faded.png", dpi=90); plt.close(f)

    # ── first-order counterfactual: half out at +0.5R when MFE>=0.5R, half held to actual ──
    cf = np.where(t.mfe_R >= 0.5, 0.25 + 0.5 * t.actual_R, t.actual_R)
    cf_te = np.where(te.mfe_R >= 0.5, 0.25 + 0.5 * te.actual_R, te.actual_R)

    out = {
        "n_trades": int(len(t)), "n_unmatched": unmatched,
        "mfe_R_median": round(float(t.mfe_R.median()), 3), "mfe_R_mean": round(float(t.mfe_R.mean()), 3),
        "mae_R_median": round(float(t.mae_R.median()), 3),
        "pct_reaching_mfe": _pct_reaching(t.mfe_R, THRESHOLDS),
        "pct_reaching_mae": _pct_reaching(t.mae_R, THRESHOLDS),
        "time_exit": {
            "n": int(len(te)), "frac_reaching_0.5R": round(frac_touch, 3),
            "subset_final_R_mean": round(float(te_touch.actual_R.mean()), 3) if len(te_touch) else None,
            "subset_final_R_median": round(float(te_touch.actual_R.median()), 3) if len(te_touch) else None,
            "subset_pct_closing_below_0.3R": round(float((te_touch.actual_R < 0.3).mean()) * 100, 1) if len(te_touch) else None,
        },
        "counterfactual": {
            "actual_mean_R_all": round(float(t.actual_R.mean()), 4), "partial_mean_R_all": round(float(cf.mean()), 4),
            "uplift_R_all": round(float(cf.mean() - t.actual_R.mean()), 4),
            "actual_mean_R_timeexit": round(float(te.actual_R.mean()), 4), "partial_mean_R_timeexit": round(float(cf_te.mean()), 4),
            "uplift_R_timeexit": round(float(cf_te.mean() - te.actual_R.mean()), 4),
        },
    }

    # ── verdict (data-driven). The COUNTERFACTUAL uplift is the decision number — a partial-exit is
    #    only worth building if it RAISES realized R. Threshold %s alone are misleading: trades can
    #    reach +0.5R and keep running, in which case banking early CAPS winners. ──
    cfd, ted = out["counterfactual"], out["time_exit"]
    p03 = out["pct_reaching_mfe"][0.3]
    p05, p10 = out["pct_reaching_mfe"][0.5], out["pct_reaching_mfe"][1.0]
    uplift = cfd["uplift_R_all"]
    te_final = ted["subset_final_R_mean"]
    if uplift <= 0:
        world = ("**THESIS REJECTED — partial-exit-at-0.5R HURTS, don't build it.** Trades that reach "
                 f"+0.5R mostly KEEP RUNNING, they don't fade: the time-exit trades that touched +0.5R "
                 f"close at mean **{te_final}R** (only {ted['subset_pct_closing_below_0.3R']}% fade below "
                 f"+0.3R). Taking half off at +0.5R caps those winners → counterfactual uplift "
                 f"**{uplift:+}R/trade** (negative). The edge is carry's winners RUNNING; partial exit "
                 "kills the asymmetry. **L2's lever is NOT exit-banking — it's the LOSS side** (53% of "
                 "trades lose; the MAE table shows how much heat they take). New hypothesis to measure "
                 "next: cut losers faster (tighter/structural stop) without clipping the runners.")
    elif p03 < 40:
        world = ("**WORLD 4 — barely-drift, thesis wrong.** Only {p03}% reach +0.3R; nothing to bank. "
                 "L2 lever is entry filtering / time exit, not exit refinement.").format(p03=p03)
    else:
        world = (f"**Partial exit ADDS {uplift:+}R/trade** ({p05}% reach +0.5R; touched-0.5R time-exits "
                 f"close at mean {te_final}R). Worth building — tune the threshold around the MFE mass.")
    out["world_verdict"] = world
    _write_md(out)
    return out


def _write_md(o: dict) -> None:
    L = ["# Excursion Analysis — HYP-064 carry trades (L2 partial-exit thesis test)", "",
         "> READ-ONLY measurement. Price paths re-read from the same yfinance OHLC HYP-064 used; "
         "nothing written to production. R = excursion / (2.0 × ATR%@entry). MFE is intrabar "
         "(high/low) — what a take-profit order would actually catch.", "",
         f"**Trades:** {o['n_trades']} (unmatched dropped: {o['n_unmatched']})  ·  "
         f"MFE_R median **{o['mfe_R_median']}** / mean {o['mfe_R_mean']}  ·  MAE_R median {o['mae_R_median']}", "",
         "## Which world are we in?", "", o["world_verdict"], "",
         "## % of trades reaching each MFE / MAE threshold (R)", "",
         "| threshold | % reaching MFE | % reaching MAE |", "|---|---|---|"]
    for th in THRESHOLDS:
        L.append(f"| {th}R | {o['pct_reaching_mfe'][th]}% | {o['pct_reaching_mae'][th]}% |")
    te = o["time_exit"]; cf = o["counterfactual"]
    L += ["", "## Load-bearing: the 203 time-exit trades", "",
          f"- reached MFE ≥ 0.5R during their life: **{round(te['frac_reaching_0.5R']*100,1)}%** "
          f"({int(te['frac_reaching_0.5R']*te['n'])} of {te['n']})",
          f"- that subset's FINAL realized R: mean **{te['subset_final_R_mean']}**, median {te['subset_final_R_median']}",
          f"- of that subset, **{te['subset_pct_closing_below_0.3R']}% closed below +0.3R** "
          "(i.e. touched +0.5R then faded — money a partial close would have banked)", "",
          "## First-order counterfactual (close half at +0.5R when MFE≥0.5R, hold the rest)", "",
          "> First-order only — the remaining half's path is held fixed. The true test is L2 re-running "
          "the exit machine with partials; this just sizes the prize.", "",
          f"- ALL trades: actual mean {cf['actual_mean_R_all']}R → partial {cf['partial_mean_R_all']}R "
          f"(**uplift {cf['uplift_R_all']:+}R/trade**)",
          f"- time-exit subset: actual {cf['actual_mean_R_timeexit']}R → partial {cf['partial_mean_R_timeexit']}R "
          f"(**uplift {cf['uplift_R_timeexit']:+}R/trade**)", "",
          "## Figures", "",
          "- ![MFE/MAE overall](excursion_mfe_overall.png)",
          "- ![MFE by exit reason](excursion_mfe_by_exit.png)",
          "- ![MFE winners vs losers](excursion_mfe_by_winloss.png)",
          "- ![MFE by pair](excursion_mfe_by_pair.png)",
          "- ![Time-exit faded](excursion_timeexit_faded.png)", ""]
    MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    o = main()
    cf, te = o["counterfactual"], o["time_exit"]
    print("=" * 72)
    print(f"EXCURSION ANALYSIS — {o['n_trades']} trades (unmatched {o['n_unmatched']})")
    print(f"  MFE_R median {o['mfe_R_median']} / mean {o['mfe_R_mean']}")
    print(f"  % reaching MFE: " + " ".join(f"{k}R={v}%" for k, v in o['pct_reaching_mfe'].items()))
    print(f"  TIME-EXIT (n={te['n']}): {round(te['frac_reaching_0.5R']*100,1)}% reached +0.5R; "
          f"that subset final R mean {te['subset_final_R_mean']}, {te['subset_pct_closing_below_0.3R']}% closed <+0.3R")
    print(f"  COUNTERFACTUAL uplift: all {cf['uplift_R_all']:+}R/trade · time-exit {cf['uplift_R_timeexit']:+}R/trade")
    print(f"  VERDICT: {o['world_verdict'].split('—')[0].strip()}")
    print("=" * 72)
