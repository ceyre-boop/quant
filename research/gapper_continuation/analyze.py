#!/usr/bin/env python3
"""HYP-092 stage 3 — pre-registered adjudication + report.

Primary: MWU one-tailed (CONT > EX) on all qualifying ticker-days.
Robustness: same test on the run-deduped sample (drop a candidate if the same
ticker had a qualifying candidate within the prior 10 TRADING days).
Verdict rule (prereg): VALID_SEPARATION iff primary p<0.05 AND robustness p<0.10
same direction; UNDERPOWERED if either bucket n<30; else NOT_SIGNIFICANT.
Everything else is descriptive and labeled so.
"""
import gzip
import json
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/research/gapper"
GROUPED = OUT / "cache/grouped"


def trading_day_index() -> dict:
    days = []
    for fp in sorted(GROUPED.glob("*.json.gz")):
        with gzip.open(fp, "rt") as f:
            if json.load(f)["n"] > 0:
                days.append(fp.name.split(".")[0])  # .stem leaves ".json" on .json.gz
    return {d: i for i, d in enumerate(days)}


def run_dedup(df: pd.DataFrame, tdi: dict) -> pd.DataFrame:
    df = df.sort_values(["date", "ticker"]).copy()
    df["tdi"] = df["date"].map(tdi)
    keep = []
    last_seen: dict = {}
    for _, r in df.iterrows():
        prior = last_seen.get(r["ticker"])
        ok = prior is None or (r["tdi"] - prior) > 10
        keep.append(ok)
        last_seen[r["ticker"]] = r["tdi"]  # any qualifying candidate resets the clock
    return df[pd.Series(keep, index=df.index)]


def bucket_stats(df: pd.DataFrame) -> dict:
    out = {}
    for name, sub in [("CONT", df[df.read == "CONT"]), ("EX", df[df.read == "EX"]),
                      ("UNC", df[df.read == "UNC"]), ("ALL", df)]:
        o = sub["outcome_pct"]
        out[name] = {
            "n": int(len(sub)),
            "unique_tickers": int(sub["ticker"].nunique()),
            "median_pct": round(float(o.median()), 4) if len(sub) else None,
            "mean_pct": round(float(o.mean()), 4) if len(sub) else None,
            "continued_gt3": round(float((o > 0.03).mean()), 4) if len(sub) else None,
            "reversed_lt3": round(float((o < -0.03).mean()), 4) if len(sub) else None,
            "stagnated": round(float(((o >= -0.03) & (o <= 0.03)).mean()), 4) if len(sub) else None,
        }
    return out


def mwu(df: pd.DataFrame) -> dict:
    cont = df[df.read == "CONT"]["outcome_pct"]
    ex = df[df.read == "EX"]["outcome_pct"]
    if len(cont) == 0 or len(ex) == 0:
        return {"n_cont": len(cont), "n_ex": len(ex), "U": None, "p": None}
    u, p = mannwhitneyu(cont, ex, alternative="greater")
    return {"n_cont": int(len(cont)), "n_ex": int(len(ex)),
            "U": float(u), "p": round(float(p), 6)}


def main():
    df = pd.read_csv(OUT / "per_candidate.csv")
    guards = json.loads((OUT / "stage2_guards.json").read_text())
    tdi = trading_day_index()

    primary = mwu(df)
    dedup = run_dedup(df, tdi)
    robust = mwu(dedup)

    powered = primary["n_cont"] >= 30 and primary["n_ex"] >= 30
    if not powered:
        verdict = "UNDERPOWERED"
    elif primary["p"] is not None and primary["p"] < 0.05 and \
            robust["p"] is not None and robust["p"] < 0.10:
        verdict = "VALID_SEPARATION"
    else:
        verdict = "NOT_SIGNIFICANT"

    monthly = df.assign(month=df["date"].str[:7]).groupby("month").size().to_dict()
    cont_path = df[df.read == "CONT"].sort_values("date")["outcome_pct"].cumsum()
    ex_path = (-df[df.read == "EX"].sort_values("date")["outcome_pct"]).cumsum()

    results = {
        "hypothesis": "HYP-092", "verdict": verdict,
        "primary_mwu": primary, "robustness_mwu": robust,
        "robustness_sample_n": int(len(dedup)),
        "bucket_stats_primary": bucket_stats(df),
        "bucket_stats_robustness": bucket_stats(dedup),
        "monthly_candidate_counts": monthly,
        "trading_days_in_window": len(tdi),
        "descriptive_cumulative_pct_long_CONT": round(float(cont_path.iloc[-1]), 4) if len(cont_path) else None,
        "descriptive_cumulative_pct_short_EX": round(float(ex_path.iloc[-1]), 4) if len(ex_path) else None,
        "stage2_guards": guards,
        "vote_marginals": {c: round(float(df[c].mean()), 4)
                           for c in ["C1", "C2", "C3", "C4", "E1", "E2", "E3", "E4"]},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in
                      ["verdict", "primary_mwu", "robustness_mwu"]}, indent=2))


if __name__ == "__main__":
    main()
