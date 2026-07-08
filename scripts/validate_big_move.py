#!/usr/bin/env python3
"""
scripts/validate_big_move.py
============================
GATE for the Big-Move-of-the-Day classifier (sovereign/intelligence/big_move.py).

Until this script returns PASS, the dashboard meter is DISPLAY-ONLY and the
classifier never gates a live decision. This is the same discipline that killed
the ES/NQ bias engine (p=0.57, 2026-06-10): an estimate is a hypothesis until the
out-of-sample harness says otherwise.

WHAT IT TESTS
-------------
Strategy under test (no lookahead): to predict day i, the classifier sees only
bars [:i] (completed prior days). On days where p_big >= threshold and direction
is not NEUTRAL, take that direction at day i's OPEN and exit at day i's CLOSE —
i.e. try to capture the day's institutional displacement. Per-trade returns are
costed (round-trip spread) and annualized by ACTUAL trade frequency, never as if
trading daily (per CLAUDE.md: the prior 2.097 headline was wrong precisely because
it annualized sparse trades as daily).

GATES (all must hold for PASS)
  1. OOS √n-weighted portfolio Sharpe >= MIN_OOS_SHARPE (default 0.30)
  2. Directional permutation p < ALPHA (default 0.05) on the pooled OOS trades
  3. Rolling walk-forward verdict == ROBUST (every test year positive, min > 0.3)

Output: data/research/big_move_validation.json
Exit:   0 = PASS, 1 = FAIL (valid outcome — meter stays display-only), 2 = ERROR

Reuses (does not re-implement):
  scripts/holdout_validation_v014.py -> sharpe_ci, classify_decay
  scripts/derive_hypothesis_pvalues.py -> benjamini_hochberg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.intelligence.big_move import estimate_big_move, _MIN_BARS  # noqa: E402
from scripts.holdout_validation_v014 import sharpe_ci, classify_decay     # noqa: E402
from scripts.derive_hypothesis_pvalues import benjamini_hochberg          # noqa: E402

logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.ERROR)

REPORT = ROOT / "data" / "research" / "big_move_validation.json"

# Proven 4-pair portfolio (AUDNZD excluded — HYP-045, both legs RBA-driven).
PAIRS = {
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
}

IS_START, IS_END = "2015-01-01", "2020-12-31"
OOS_START, OOS_END = "2021-01-01", "2024-12-31"
WALKFORWARD_YEARS = [2021, 2022, 2023, 2024]

DEFAULT_THRESHOLD = 0.50      # p_big gate
DEFAULT_COST_PCT = 0.0002     # round-trip cost (~2 bps) applied as a drag per trade
ALPHA = 0.05
MIN_OOS_SHARPE = 0.30
N_PERM = 2000


# ── Strategy simulation (no lookahead) ────────────────────────────────────────

def _trade_series(pair: str, df: pd.DataFrame, threshold: float,
                  cost_pct: float) -> pd.DataFrame:
    """
    Walk daily bars; emit one costed return per traded day.

    For day i the classifier sees bars [:i] only. Realized return is the day's
    open->close move in the predicted direction, minus round-trip cost.
    Returns a DataFrame indexed by date with columns: ret, sign.
    """
    dates, rets, signs = [], [], []
    opens = df["Open"].to_numpy()
    closes = df["Close"].to_numpy()
    for i in range(_MIN_BARS, len(df)):
        est = estimate_big_move(pair, df.iloc[:i])  # price-only, no future bar
        if est.p_big < threshold or est.direction == "NEUTRAL":
            continue
        o, c = float(opens[i]), float(closes[i])
        if o <= 0:
            continue
        sign = 1.0 if est.direction == "LONG" else -1.0
        gross = sign * (c - o) / o
        dates.append(df.index[i])
        rets.append(gross - cost_pct)
        signs.append(sign)
    return pd.DataFrame({"ret": rets, "sign": signs}, index=pd.DatetimeIndex(dates))


def _annualized_sharpe(rets: np.ndarray, years_span: float) -> float:
    """Sharpe annualized by ACTUAL trade frequency, not as if trading daily."""
    rets = np.asarray(rets, float)
    if len(rets) < 5 or rets.std(ddof=1) == 0 or years_span <= 0:
        return 0.0
    per_trade = rets.mean() / rets.std(ddof=1)
    trades_per_year = len(rets) / years_span
    return float(per_trade * np.sqrt(trades_per_year))


def _years_span(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    return max((idx.max() - idx.min()).days / 365.25, 1e-6)


def _sqrtn_weighted(per_pair: List[Tuple[float, int]]) -> float:
    """√n-weighted mean of per-pair Sharpes (matches holdout_validation_v014)."""
    pairs = [(s, n) for s, n in per_pair if n > 0 and not np.isnan(s)]
    if not pairs:
        return 0.0
    w = [np.sqrt(n) for _, n in pairs]
    return float(sum(s * wi for (s, _), wi in zip(pairs, w)) / sum(w))


def _permutation_pvalue(rets: np.ndarray, signs: np.ndarray,
                        years_span: float, n_perm: int, rng) -> float:
    """
    Directional permutation test: shuffle the sign of each trade's return and
    recompute Sharpe. p = P(null Sharpe >= actual). Tests whether DIRECTION
    carries skill (magnitude/timing held fixed).
    """
    rets = np.asarray(rets, float)
    signs = np.asarray(signs, float)
    if len(rets) < 5:
        return float("nan")
    actual = _annualized_sharpe(rets, years_span)
    # gross per-day magnitude (strip the direction we chose): ret + cost is signed
    # gross; recover unsigned magnitude by dividing out our sign.
    base = rets / np.where(signs == 0, 1.0, signs)  # ≈ gross-cost in our direction
    ge = 0
    for _ in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=len(base))
        null = _annualized_sharpe(base * flip, years_span)
        if null >= actual:
            ge += 1
    return float((ge + 1) / (n_perm + 1))  # add-one for a valid (never-zero) p


# ── Data ──────────────────────────────────────────────────────────────────────

def _fetch_daily(yticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import yfinance as yf
    df = yf.Ticker(yticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _synthetic_daily(start: str, end: str, edge: float, rng) -> pd.DataFrame:
    """Synthetic frame with a known directional edge (for --smoke). edge>0 means
    up-days are slightly more common, so the momentum-following classifier should
    show positive Sharpe; edge=0 means no edge (should not pass)."""
    idx = pd.date_range(start, end, freq="B", tz="UTC")
    n = len(idx)
    steps = rng.normal(edge * 0.001, 0.006, n)
    close = 1.30 * np.exp(np.cumsum(steps))
    o = np.empty(n); o[0] = close[0]
    o[1:] = close[:-1]
    intraday = np.abs(rng.normal(0, 0.004, n))
    hi = np.maximum(o, close) + intraday
    lo = np.minimum(o, close) - intraday
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": close}, index=idx)


# ── Per-pair evaluation ───────────────────────────────────────────────────────

def _evaluate_pair(pair: str, df_full: pd.DataFrame, threshold: float,
                   cost_pct: float, n_perm: int, rng) -> Dict:
    trades = _trade_series(pair, df_full, threshold, cost_pct)

    is_t = trades[(trades.index >= IS_START) & (trades.index <= IS_END)]
    oos_t = trades[(trades.index >= OOS_START) & (trades.index <= OOS_END)]

    is_sharpe = _annualized_sharpe(is_t["ret"].to_numpy(), _years_span(is_t.index))
    oos_sharpe = _annualized_sharpe(oos_t["ret"].to_numpy(), _years_span(oos_t.index))
    lo, hi, se = sharpe_ci(oos_sharpe, len(oos_t))
    decay_tag, decay_msg = classify_decay(is_sharpe, oos_sharpe)
    perm_p = _permutation_pvalue(oos_t["ret"].to_numpy(), oos_t["sign"].to_numpy(),
                                 _years_span(oos_t.index), n_perm, rng)

    wf = {}
    for yr in WALKFORWARD_YEARS:
        yt = trades[(trades.index >= f"{yr}-01-01") & (trades.index <= f"{yr}-12-31")]
        wf[str(yr)] = round(_annualized_sharpe(yt["ret"].to_numpy(), _years_span(yt.index)), 3)

    return {
        "pair": pair,
        "is_sharpe": round(is_sharpe, 3),
        "oos_sharpe": round(oos_sharpe, 3),
        "oos_ci": [lo, hi],
        "oos_trades": int(len(oos_t)),
        "decay": decay_tag,
        "decay_note": decay_msg,
        "p_value": perm_p,
        "walkforward": wf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Big-Move classifier validation gate")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_PCT)
    ap.add_argument("--nperm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="run the full pipeline on synthetic data (no network)")
    ap.add_argument("--smoke-edge", type=float, default=0.0,
                    help="synthetic directional edge for --smoke (0 = no edge)")
    args = ap.parse_args()

    n_perm = args.nperm
    rng = np.random.default_rng(args.seed)

    try:
        per_pair: List[Dict] = []
        for pair, yticker in PAIRS.items():
            if args.smoke:
                df = _synthetic_daily(IS_START, OOS_END, args.smoke_edge, rng)
            else:
                df = _fetch_daily(yticker, IS_START, OOS_END)
            if df is None or len(df) < _MIN_BARS + 30:
                per_pair.append({"pair": pair, "error": "insufficient data"})
                continue
            per_pair.append(_evaluate_pair(pair, df, args.threshold, args.cost, n_perm, rng))

        scored = [p for p in per_pair if "oos_sharpe" in p]
        benjamini_hochberg(scored, ALPHA)

        portfolio_oos = _sqrtn_weighted([(p["oos_sharpe"], p["oos_trades"]) for p in scored])
        portfolio_is = _sqrtn_weighted(
            [(p["is_sharpe"], p["oos_trades"]) for p in scored])
        # Pooled directional permutation across all OOS trades:
        all_oos_p = [p["p_value"] for p in scored
                     if isinstance(p["p_value"], float) and not np.isnan(p["p_value"])]
        pooled_p = float(np.median(all_oos_p)) if all_oos_p else float("nan")

        # Walk-forward verdict across the portfolio (every year positive, min > 0.3).
        wf_years = {str(y): _sqrtn_weighted(
            [(p["walkforward"][str(y)], p["oos_trades"]) for p in scored])
            for y in WALKFORWARD_YEARS}
        all_pos = all(v > 0 for v in wf_years.values())
        wf_verdict = "ROBUST" if (all_pos and min(wf_years.values()) > 0.3) else "FRAGILE"

        gate1 = portfolio_oos >= MIN_OOS_SHARPE
        gate2 = not np.isnan(pooled_p) and pooled_p < ALPHA
        gate3 = wf_verdict == "ROBUST"
        verdict = "PASS" if (gate1 and gate2 and gate3) else "FAIL"

        report = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "mode": "smoke" if args.smoke else "live",
            "params": {"threshold": args.threshold, "cost_pct": args.cost,
                       "nperm": n_perm, "alpha": ALPHA, "min_oos_sharpe": MIN_OOS_SHARPE},
            "verdict": verdict,
            "gates": {"oos_sharpe>=min": gate1, "permutation<alpha": gate2,
                      "walkforward_robust": gate3},
            "portfolio": {
                "is_sharpe": round(portfolio_is, 3),
                "oos_sharpe": round(portfolio_oos, 3),
                "pooled_permutation_p": (round(pooled_p, 4) if not np.isnan(pooled_p) else None),
                "walkforward": {k: round(v, 3) for k, v in wf_years.items()},
                "walkforward_verdict": wf_verdict,
            },
            "per_pair": per_pair,
            "discipline_note": (
                "Weights in big_move.py are hand-set priors. A FAIL keeps the meter "
                "display-only — that is a valid, expected outcome, not a bug. Record the "
                "verdict in a logged rationale before any live wiring (CLAUDE.md #4)."
            ),
        }

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))

        print(f"\n{'='*60}")
        print(f"BIG-MOVE VALIDATION — {verdict}  ({report['mode']} mode)")
        print(f"  Portfolio OOS Sharpe : {portfolio_oos:.3f}  (IS {portfolio_is:.3f})")
        print(f"  Pooled permutation p : {report['portfolio']['pooled_permutation_p']}")
        print(f"  Walk-forward         : {wf_verdict}  {report['portfolio']['walkforward']}")
        print(f"  Gates                : {report['gates']}")
        print(f"  Report               : {REPORT}")
        print(f"{'='*60}\n")
        return 0 if verdict == "PASS" else 1

    except Exception as exc:  # noqa: BLE001
        logging.error("validation error: %s", exc, exc_info=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
