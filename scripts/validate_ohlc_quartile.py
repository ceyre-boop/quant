#!/usr/bin/env python3
"""
scripts/validate_ohlc_quartile.py
=================================
GATE for the OHLC-quartile candle-structure hypothesis (sovereign/research/ohlc_quartile.py),
queued to the autonomous research factory 2026-06-15.

HYPOTHESIS: intrabar quartile levels (25/50/75/100% of sub-bars) on 4H + 5min predict
the direction of the NEXT 4H bar on ES/NQ. A SIMPLE model only — complexity overfits.

DISCIPLINE: Mechanism is NONE STATED (exploratory). Same class as the killed ES/NQ bias
engine (p=0.567). The expected base-case verdict is NOT_SIGNIFICANT, and producing that
cheaply IS the win. Runs in factory SHADOW mode — touches nothing live.

GATES (all must hold for PASS)
  1. OOS √n-weighted Sharpe >= MIN_OOS_SHARPE (default 0.30)
  2. Directional permutation p < ALPHA (default 0.05), >= min shuffles, both-sided
  3. Walk-forward verdict == ROBUST (every fold positive, min > 0.3)

Output: data/research/ohlc_quartile_validation.json   (shape matches the factory parser)
Exit:   0 = PASS, 1 = FAIL (valid outcome), 2 = ERROR

Reuses: scripts/holdout_validation_v014.py::{sharpe_ci, classify_decay},
        scripts/derive_hypothesis_pvalues.py::benjamini_hochberg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.research.ohlc_quartile import build_feature_matrix, FEATURE_COLS  # noqa: E402
from scripts.holdout_validation_v014 import sharpe_ci, classify_decay           # noqa: E402
from scripts.derive_hypothesis_pvalues import benjamini_hochberg                # noqa: E402

logging.basicConfig(level=logging.ERROR)
for lib in ("yfinance", "peewee", "urllib3", "requests", "sklearn"):
    logging.getLogger(lib).setLevel(logging.ERROR)

REPORT = ROOT / "data" / "research" / "ohlc_quartile_validation.json"

INSTRUMENTS = ["MES", "MNQ"]      # ES/NQ micros
DEFAULT_COST_PCT = 0.0001         # ~1 tick round-trip drag per predicted bar
ALPHA = 0.05
MIN_OOS_SHARPE = 0.30
N_FOLDS = 4
N_PERM = 10000


# ── Model (simple, guarded) ───────────────────────────────────────────────────

def _fit_predict(X_tr, y_tr, X_te):
    """Simple logistic regression. Falls back to a majority-class predictor if the
    training labels are single-class or sklearn is unavailable."""
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), int(round(float(np.mean(y_tr)))))
    try:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=200, C=1.0)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    except Exception:
        # numpy fallback: predict the training majority class (no edge — honest null)
        return np.full(len(X_te), int(round(float(np.mean(y_tr)))))


# ── Walk-forward OOS over a single instrument's feature matrix ─────────────────

def _walk_forward(feat: pd.DataFrame, cost_pct: float, n_folds: int):
    """Expanding-window walk-forward. Returns (oos_df, fold_sharpes) where oos_df has
    columns ret_net (strategy return per predicted bar) and ts."""
    n = len(feat)
    if n < (n_folds + 1) * 20:
        n_folds = max(2, n // 40)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    X = feat[FEATURE_COLS].to_numpy()
    y = feat["y"].to_numpy()
    ret = feat["ret"].to_numpy()
    idx = feat.index

    oos_ret, oos_ts, fold_sharpes = [], [], []
    for k in range(1, n_folds):
        tr_end = bounds[k]
        te_end = bounds[k + 1]
        if tr_end < 20 or te_end - tr_end < 5:
            continue
        preds = _fit_predict(X[:tr_end], y[:tr_end], X[tr_end:te_end])
        pos = np.where(preds == 1, 1.0, -1.0)              # long up-pred, short down-pred
        r = pos * ret[tr_end:te_end] - cost_pct
        oos_ret.extend(r.tolist())
        oos_ts.extend(idx[tr_end:te_end].tolist())
        fold_sharpes.append(_ann_sharpe(np.array(r), _years(idx[tr_end:te_end])))

    return pd.DataFrame({"ret_net": oos_ret}, index=pd.DatetimeIndex(oos_ts)), fold_sharpes


def _ann_sharpe(r: np.ndarray, years: float) -> float:
    r = np.asarray(r, float)
    if len(r) < 5 or r.std(ddof=1) == 0 or years <= 0:
        return 0.0
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(len(r) / years))


def _years(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 1.0
    return max((idx.max() - idx.min()).days / 365.25, 1e-6)


def _sqrtn_weighted(pairs: List[tuple]) -> float:
    pairs = [(s, n) for s, n in pairs if n > 0 and not np.isnan(s)]
    if not pairs:
        return 0.0
    w = [np.sqrt(n) for _, n in pairs]
    return float(sum(s * wi for (s, _), wi in zip(pairs, w)) / sum(w))


def _permutation_p(oos_ret: np.ndarray, years: float, n_perm: int, rng) -> float:
    """Both-sided directional permutation: flip the sign of each OOS return and
    recompute Sharpe. p = P(null Sharpe >= actual)."""
    r = np.asarray(oos_ret, float)
    if len(r) < 5:
        return float("nan")
    actual = _ann_sharpe(r, years)
    ge = 0
    for _ in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=len(r))
        if _ann_sharpe(r * flip, years) >= actual:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_5min(instrument: str) -> Optional[pd.DataFrame]:
    """Best-effort 5min loader: databento → yfinance fallback. None on failure."""
    try:
        from sovereign.futures.bar_feed import load_history
        from sovereign.es_nq.data_store import resample_5min
        for source, lookback in (("databento", "720d"), ("yf", "7d")):
            try:
                df1m = load_history(instrument, source=source, lookback=lookback)
                if df1m is not None and len(df1m) > 50:
                    return resample_5min(df1m)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _synthetic_5min(edge: float, rng, n_days: int = 400) -> pd.DataFrame:
    """Synthetic 1min→5min with a known directional edge (for --smoke). edge>0 makes
    quartile structure weakly predict the next 4H direction; edge=0 = pure noise."""
    idx = pd.date_range("2022-01-03 09:30", periods=n_days * 78, freq="5min", tz="UTC")
    n = len(idx)
    base = rng.normal(0, 0.0008, n)
    # Inject edge: a slow regime signal that also tilts the next-bar drift.
    regime = np.sin(np.arange(n) / 137.0)
    steps = base + edge * 0.0006 * regime
    close = 4000 * np.exp(np.cumsum(steps))
    o = np.empty(n); o[0] = close[0]; o[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 1.5, n))
    hi = np.maximum(o, close) + wig
    lo = np.minimum(o, close) - wig
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": close,
                         "Volume": rng.integers(1, 100, n)}, index=idx)


def _evaluate(instrument: str, df5m: pd.DataFrame, cost_pct: float, n_perm: int,
              n_folds: int, rng) -> Dict:
    feat = build_feature_matrix(df5m)
    if feat is None or len(feat) < 60:
        return {"instrument": instrument, "error": "insufficient feature rows"}
    oos, fold_sharpes = _walk_forward(feat, cost_pct, n_folds)
    if len(oos) < 10:
        return {"instrument": instrument, "error": "insufficient OOS rows"}

    yrs = _years(oos.index)
    oos_sharpe = _ann_sharpe(oos["ret_net"].to_numpy(), yrs)
    lo, hi, se = sharpe_ci(oos_sharpe, len(oos))
    perm_p = _permutation_p(oos["ret_net"].to_numpy(), yrs, n_perm, rng)
    acc = float((np.sign(oos["ret_net"]) > 0).mean())  # fraction of profitable predicted bars
    all_pos = all(s > 0 for s in fold_sharpes) if fold_sharpes else False
    wf_verdict = "ROBUST" if (all_pos and fold_sharpes and min(fold_sharpes) > 0.3) else "FRAGILE"
    return {
        "instrument": instrument,
        "oos_sharpe": round(oos_sharpe, 3),
        "oos_ci": [lo, hi],
        "oos_rows": int(len(oos)),
        "hit_rate": round(acc, 3),
        "p_value": perm_p,
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
        "walkforward": wf_verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OHLC-quartile hypothesis validation gate")
    ap.add_argument("--nperm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_PCT)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, no network")
    ap.add_argument("--smoke-edge", type=float, default=0.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    try:
        per_inst: List[Dict] = []
        for inst in INSTRUMENTS:
            df5m = (_synthetic_5min(args.smoke_edge, rng) if args.smoke
                    else _load_5min(inst))
            if df5m is None or len(df5m) < 200:
                per_inst.append({"instrument": inst, "error": "no/insufficient data"})
                continue
            per_inst.append(_evaluate(inst, df5m, args.cost, args.nperm, args.folds, rng))

        scored = [p for p in per_inst if "oos_sharpe" in p]
        benjamini_hochberg(scored, ALPHA)

        portfolio_oos = _sqrtn_weighted([(p["oos_sharpe"], p["oos_rows"]) for p in scored])
        ps = [p["p_value"] for p in scored
              if isinstance(p["p_value"], float) and not np.isnan(p["p_value"])]
        pooled_p = float(np.median(ps)) if ps else float("nan")
        wf_all_robust = bool(scored) and all(p["walkforward"] == "ROBUST" for p in scored)

        gate1 = portfolio_oos >= MIN_OOS_SHARPE
        gate2 = (not np.isnan(pooled_p)) and pooled_p < ALPHA and args.nperm >= 10000
        gate3 = wf_all_robust
        verdict = "PASS" if (gate1 and gate2 and gate3) else "FAIL"

        report = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "hypothesis": "OHLC intrabar quartile levels (4H+5min) predict next-4H direction on ES/NQ",
            "mode": "smoke" if args.smoke else "live",
            "mechanism": "NONE STATED — exploratory (see plan; expected base case NOT_SIGNIFICANT)",
            "params": {"nperm": args.nperm, "alpha": ALPHA, "cost_pct": args.cost,
                       "min_oos_sharpe": MIN_OOS_SHARPE, "model": "logistic_regression"},
            "verdict": verdict,
            "gates": {"oos_sharpe>=min": gate1, "permutation<alpha": gate2,
                      "walkforward_robust": gate3},
            "portfolio": {
                "oos_sharpe": round(portfolio_oos, 3),
                "pooled_permutation_p": (round(pooled_p, 4) if not np.isnan(pooled_p) else None),
                "walkforward_verdict": "ROBUST" if wf_all_robust else "FRAGILE",
            },
            "per_instrument": per_inst,
            "discipline_note": ("Shadow test only. A FAIL is the expected, valid outcome — the idea "
                                "got a fair cheap test instead of a live deployment. Never auto-deployed."),
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))

        print(f"\n{'='*60}")
        print(f"OHLC-QUARTILE VALIDATION — {verdict}  ({report['mode']} mode)")
        print(f"  Portfolio OOS Sharpe : {portfolio_oos:.3f}")
        print(f"  Pooled permutation p : {report['portfolio']['pooled_permutation_p']}")
        print(f"  Walk-forward         : {report['portfolio']['walkforward_verdict']}")
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
