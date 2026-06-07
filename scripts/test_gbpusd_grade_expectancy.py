"""
STEP 3 — Is GBPUSD grade-A genuinely different from grade-B? (diagnostic only)

The user asked whether the GBPUSD conviction-grading carries real signal — originally framed as
"the 2x size on grade-A," but no grade->position-size multiplier exists in the code (GBPUSD's "2x"
is the trailing-ATR mult). So this tests the underlying question directly: do grade-A GBPUSD trades
have materially different expectancy than grade-B?

Grade is a function of |real_rate_diff| (strategy.py grade_from_signal): A >= 1.5, B 0.5-1.5.
We grade each full-period (2015-2024) GBPUSD backtest trade by the real_rate_diff at its entry date
(from the cached macro parquets, same inputs the signal engine uses), then compare A vs B mean
expectancy (pnl_pct) + Sharpe, with a 10k label-shuffle permutation.

NO implementation — there is nothing to gate on (no size multiplier). If significant, it's a
candidate for a FUTURE grade-sizing rule (separate work). Logs to the ledger regardless.

Usage:  ~/quant/.venv/bin/python scripts/test_gbpusd_grade_expectancy.py --perms 10000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)
for _l in ("yfinance", "urllib3", "requests", "peewee"):
    logging.getLogger(_l).setLevel(logging.ERROR)

from sovereign.forex.forex_backtester import ForexBacktester
from sovereign.forex.pair_universe import PAIR_CONFIG, CB_TO_COUNTRY

OUT = ROOT / "data" / "research" / "gbpusd_grade_expectancy.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
MACRO = ROOT / "data" / "cache" / "macro"
PAIR = "GBPUSD=X"


def _macro(name: str, col: str) -> pd.Series:
    df = pd.read_parquet(MACRO / f"{name}.parquet").sort_index()
    s = df[col]
    s.index = pd.to_datetime(s.index)
    return s


def _grade(rate_diff: float) -> str:
    a = abs(rate_diff)
    return "A" if a >= 1.5 else ("B" if a >= 0.5 else "C")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = PAIR_CONFIG[PAIR]
    base_country = CB_TO_COUNTRY[cfg.base_central_bank]   # GBP -> UK
    quote_country = CB_TO_COUNTRY[cfg.quote_central_bank]  # USD -> US

    bt = ForexBacktester(start="2015-01-01", end="2024-12-31")
    bt.PAIR_HOLD_OVERRIDES = {}  # post-rollback live config (60d)
    result, trades = bt.run_pair_with_trades(PAIR, base_country, quote_country)
    if not trades:
        print("No GBPUSD trades"); return 1

    # macro real-rate-diff series (UK base, US quote)
    b_rate = _macro(f"{base_country}_rates", "rate"); b_cpi = _macro(f"{base_country}_cpi", "cpi")
    q_rate = _macro(f"{quote_country}_rates", "rate"); q_cpi = _macro(f"{quote_country}_cpi", "cpi")

    rows = []
    for t in trades:
        d = pd.Timestamp(str(t["entry_date"])[:10])
        try:
            rd = (float(b_rate.asof(d)) - float(b_cpi.asof(d))) - (float(q_rate.asof(d)) - float(q_cpi.asof(d)))
        except Exception:
            continue
        rows.append({"pnl": float(t["pnl_pct"]), "grade": _grade(rd), "rate_diff": round(rd, 3),
                     "hold": int(t.get("hold_days", 0))})
    df = pd.DataFrame(rows)

    A = df[df.grade == "A"]["pnl"].to_numpy()
    B = df[df.grade == "B"]["pnl"].to_numpy()

    def _sharpe(x):
        return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252 / max(1, np.mean([r["hold"] for r in rows])))) if len(x) > 1 else 0.0

    obs_diff = (A.mean() - B.mean()) if len(A) and len(B) else 0.0

    # permutation: shuffle grade labels across A+B trades
    pooled = np.concatenate([A, B]) if len(A) and len(B) else np.array([])
    nA = len(A)
    p_value = None
    if nA and len(B) and len(pooled) >= 4:
        null = np.empty(args.perms)
        for k in range(args.perms):
            perm = rng.permutation(pooled)
            null[k] = perm[:nA].mean() - perm[nA:].mean()
        p_value = float((np.sum(np.abs(null) >= abs(obs_diff)) + 1) / (args.perms + 1))

    grade_counts = df.grade.value_counts().to_dict()
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "id": "GBPUSD-GRADE-EXPECTANCY",
        "pair": PAIR, "period": "2015-01-01..2024-12-31", "n_trades": len(df),
        "grade_counts": grade_counts,
        "grade_A": {"n": int(len(A)), "mean_pnl_pct": round(float(A.mean()) * 100, 4) if len(A) else None,
                    "win_rate": round(float((A > 0).mean()), 3) if len(A) else None},
        "grade_B": {"n": int(len(B)), "mean_pnl_pct": round(float(B.mean()) * 100, 4) if len(B) else None,
                    "win_rate": round(float((B > 0).mean()), 3) if len(B) else None},
        "observed_mean_diff_pct": round(obs_diff * 100, 4),
        "permutation_p": p_value, "n_perms": args.perms, "seed": args.seed,
    }
    sig = p_value is not None and p_value < 0.05
    out["verdict"] = "SIGNIFICANT" if sig else "NOT_SIGNIFICANT"
    out["note"] = ("Diagnostic only — no grade->size multiplier exists to gate. "
                   + ("Grade-A expectancy materially differs from B; candidate for a future "
                      "grade-sizing rule (separate work)." if sig else
                      "Grade-A and grade-B GBPUSD expectancy not distinguishable; grading carries "
                      "no actionable size signal on this sample."))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"GBPUSD trades: {len(df)}  grades={grade_counts}")
    print(f"  A: n={len(A)} mean_pnl={out['grade_A']['mean_pnl_pct']}%  B: n={len(B)} mean_pnl={out['grade_B']['mean_pnl_pct']}%")
    print(f"  obs diff={out['observed_mean_diff_pct']}%  perm p={p_value}  -> {out['verdict']}")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
