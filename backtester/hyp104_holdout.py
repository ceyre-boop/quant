"""HYP-104 holdout test — frozen rule on 2025-07-17..2026-07-17. Refuses to run
unless the prereg is committed. Seals the verdict."""
import json
import subprocess
from pathlib import Path

import numpy as np

from . import daily_engine as de

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "research/hyp_104_downgap_short_prereg.md"
PREREG_SHA = "a984372183d7b7d3affa7a04415dcfabc095ccdc993a7a36c717356cb8f18dbf"
EXCLUDE = {"UVXY", "VXX", "SVXY", "TQQQ", "SQQQ", "SPXL", "SPXS", "TNA", "TZA",
           "SOXL", "SOXS", "ARKK", "GDXJ"}
CFG = dict(gap_dir="down", trade_dir="short", thr=0.05, hold_days=2,
           stop_pct=0.15, direction="short")
HOLD_LO, HOLD_HI = "2025-07-17", "2026-07-17"
BENCH = dict(annual=0.18, sharpe=2.0, max_dd=0.15, per_year=50)
N_PERM = 2000


def guard():
    sha = subprocess.run(["shasum", "-a", "256", str(PREREG)],
                         capture_output=True, text=True).stdout.split()[0]
    assert sha == PREREG_SHA, f"prereg hash mismatch: {sha}"
    log = subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "--",
                          str(PREREG)], capture_output=True, text=True).stdout
    assert log.strip(), "prereg not committed — holdout refused"


def main():
    guard()
    tickers = sorted(p.stem for p in de.UNIVERSE.glob("*.parquet"))
    rets, per_asset = [], {}
    for t in tickers:
        if t in EXCLUDE:
            continue
        r = de.backtest_daily(de.load_daily(t), "gap", CFG, HOLD_LO, HOLD_HI)["rets"]
        if r:
            per_asset[t] = (round(float(np.mean(r)), 4), len(r))
            rets.extend(r)
    r = np.array(rets)
    yrs = 1.0
    m = de._metrics(list(r), yrs, 0.10)

    rng = np.random.default_rng(42)
    obs = m["sharpe"]
    ge = 0
    for _ in range(N_PERM):
        flip = rng.choice([-1, 1], size=len(r))
        sr = r * flip
        sh = sr.mean() / sr.std() * np.sqrt(len(sr)) if sr.std() > 0 else 0.0
        if sh >= obs:
            ge += 1
    p = (ge + 1) / (N_PERM + 1)

    beat = (m["annual"] > BENCH["annual"] and m["sharpe"] > BENCH["sharpe"]
            and m["max_dd"] < BENCH["max_dd"] and m["per_year"] >= BENCH["per_year"])
    if len(r) < 50:
        verdict = "DATA_INSUFFICIENT"
    elif beat and p < 0.05:
        verdict = "CONFIRMED"
    else:
        verdict = "NOT_CONFIRMED"

    out = {"hyp": "HYP-104", "holdout": [HOLD_LO, HOLD_HI],
           "n_trades": int(len(r)), "metrics": m, "perm_p": round(p, 5),
           "beats_benchmark": bool(beat), "verdict": verdict,
           "benchmark": BENCH,
           "top_contributors": dict(sorted(per_asset.items(),
                                           key=lambda x: -x[1][1])[:12])}
    (REPO / "research/hyp_104_downgap_short_results.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
