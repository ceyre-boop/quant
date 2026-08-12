"""Phases 1-4 orchestrator for HYP-091. Loads data once, runs the backtest for the
primary (ratediff) + robustness (broken, none) financing modes, computes the
per-year Sharpe table, the monthly correlation vs actual v015 carry + combined
Sharpe, and the gauntlet; writes JSON artifacts + the report under
data/research/tsmom_hyp091/. Does NOT write the ledger — that is verdict_to_ledger.py.

    python3 research/tsmom_hyp091/run_study.py
"""
from __future__ import annotations

import pandas as pd

from research.tsmom_hyp091 import backtest as bt
from research.tsmom_hyp091 import correlation, feeds, gauntlet, report
from research.tsmom_hyp091._lib import (
    OOS_START, OUT_DIR, PAIRS, SEED, V015_DECADE_CSV, env_record, write_json,
)
from sovereign.reporting.equity_curve import _sharpe, weighted_portfolio_sharpe


def _years(idx) -> float:
    return max((idx.max() - idx.min()).days / 365.25, 1e-9)


def _per_year(port: pd.Series) -> list[dict]:
    rows = []
    for y, g in port.groupby(port.index.year):
        rows.append({"year": int(y), "sharpe": _sharpe(g.tolist(), 1.0), "n": int(len(g))})
    return rows


def _v015_loader_sanity() -> dict:
    """Reproduce the v015 decade headline (per-pair weighted ~0.6886) to prove the CSV loader."""
    df = pd.read_csv(V015_DECADE_CSV)
    df["entry_date"] = pd.to_datetime(df["entry_date"]); df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["risk_adjusted_pnl_pct"] = pd.to_numeric(df["risk_adjusted_pnl_pct"], errors="coerce")
    rows = []
    for _, g in df.groupby("pair"):
        yrs = max((g["exit_date"].max() - g["entry_date"].min()).days / 365.25, 1e-9)
        rows.append((_sharpe(g["risk_adjusted_pnl_pct"].tolist(), yrs), len(g)))
    return {"per_pair_weighted_sharpe": weighted_portfolio_sharpe(rows), "target_decade": 0.6886}


def run() -> dict:
    prices, diffs, calib = feeds.load_prices(), feeds.load_rate_differentials(), feeds.load_swap_calibration()
    v015 = feeds.load_v015_monthly()

    runs = {mode: bt.backtest(mode, prices, diffs, calib) for mode in ("ratediff", "broken", "none")}
    prim = runs["ratediff"]
    port = prim["monthly"]

    def sh(s):
        s = s.dropna()
        return {"full": _sharpe(s.tolist(), _years(s.index)),
                "is": _sharpe(s[s.index < OOS_START].tolist(), _years(s[s.index < OOS_START].index)),
                "oos": _sharpe(s[s.index >= OOS_START].tolist(), _years(s[s.index >= OOS_START].index)),
                "mean_monthly": round(float(s.mean()), 6), "n": int(len(s))}

    backtest_out = {
        "sharpe_by_mode": {m: sh(r["monthly"]) for m, r in runs.items()},
        "per_year_ratediff": _per_year(port),
        "positive_years": sum(1 for r in _per_year(port) if r["sharpe"] > 0),
        "total_years": port.index.year.nunique(),
        "window": {"first_month": prim["first_month"], "last_month": prim["last_month"], "n_months": prim["n_months"]},
        "v015_loader_sanity": _v015_loader_sanity(),
    }

    corr_primary = correlation.analyze(port, v015, "primary_ratediff")
    corr_broken = correlation.analyze(runs["broken"]["monthly"], v015, "robustness_broken")
    correlation_out = {"primary": corr_primary, "robustness_broken_model": corr_broken}

    gaunt = gauntlet.run(prim["dec"], port, corr_primary["corr_full_abs"], seed=SEED)

    results = {
        "hypothesis": "HYP-091",
        "title": "TSMOM diversification of the v015 carry book",
        "ticket": "TICK-027",
        "prereg": "data/research/preregister/HYP-091_tsmom_carry_diversification.json",
        "run_utc": pd.Timestamp.utcnow().isoformat(),
        "env": env_record(),
        "prior_expectation": "NOT_SIGNIFICANT",
        "verdict": gaunt["verdict"],
        "verdict_reasons": gaunt["verdict_reasons"],
        "null": "OOS(2023-24) Sharpe <= 0 OR monthly |corr| with v015 > 0.5",
        "backtest": backtest_out,
        "correlation": correlation_out,
        "gauntlet": gaunt,
        "relationship_to_HYP089": (
            "Corrected instrument vs the parallel HYP-089 quick-look (proxy corr + no financing + daily "
            "rebalance): monthly rebalance, correlation vs ACTUAL v015 returns, correct rate-differential "
            "financing. Reaches NOT_SIGNIFICANT on the OOS-Sharpe leg with real financing (a cleaner kill "
            "than HYP-089's boundary-close 0.277-vs-0.30)."
        ),
    }

    write_json(OUT_DIR / "backtest.json", backtest_out)
    write_json(OUT_DIR / "correlation.json", correlation_out)
    write_json(OUT_DIR / "gauntlet.json", gaunt)
    write_json(OUT_DIR / "results.json", results)
    # human-readable equity/return series for the primary
    port.rename("R").to_csv(OUT_DIR / "monthly_returns_ratediff.csv")
    report.write_report(results)
    return results


if __name__ == "__main__":
    r = run()
    print(f"VERDICT: {r['verdict']}")
    for why in r["verdict_reasons"]:
        print(f"  - {why}")
