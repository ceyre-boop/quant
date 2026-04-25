"""
Progressive Selectivity Retrainer
==================================
Wakes every 4 hours, reads accumulated trades from DuckDB, retrains XGBoost,
then asks: "does raising the confidence threshold make MORE money this month?"

Selectivity only earns the right to increase if it produces HIGHER monthly PnL
than the previous month. Win rate matters — but only when money grows too.

Saves:
  models/xgb_veto.json          — XGBoost model (auto-updated)
  models/threshold_history.json — history of threshold decisions + monthly PnL
  models/current_threshold.json — the active threshold the live system reads

Usage:
    python training/retrain_loop.py              # runs forever
    python training/retrain_loop.py --once       # single retrain cycle
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH     = ROOT / "data" / "harvest.db"
MODELS_DIR  = ROOT / "models"
LOG_PATH    = ROOT / "logs" / "retrain.log"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RETRAIN] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RETRAIN_INTERVAL_SEC = 4 * 3600   # 4 hours

# Threshold candidates to evaluate each cycle
THRESHOLD_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Minimum trade count to trust a monthly PnL reading
MIN_TRADES_FOR_EVAL = 200

FEATURE_COLS = [
    "stop_atr_mult", "tp_rr", "atr_period",
    "regime", "hurst", "atr_norm", "vol_pct",
    "month", "day_of_week", "direction",
    # ── Lagged sequence context (added 2026-04-22) ────────────────────────
    # These three features give XGBoost short-term 'memory' of recent
    # trade outcomes — equivalent to 80% of what a Transformer would do
    # without any new data infrastructure.
    "rolling_win_rate_regime",  # rolling win rate for this regime (last 20 trades)
    "recent_drawdown_streak",   # consecutive losses heading into this trade
    "bars_since_profitable",    # how long since last win (capped at 50)
]


# ── Sequence feature engineering ─────────────────────────────────────────────

def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject 3 lagged sequence features sorted by window_start.
    All features are computed using only PRIOR trades (shift(1))
    to prevent look-ahead bias.
    """
    df = df.sort_values("window_start").copy()

    # 1. Rolling win rate by regime — last 20 trades in the same regime
    df["rolling_win_rate_regime"] = (
        df.groupby("regime")["is_profitable"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
          .fillna(0.5)  # neutral prior when insufficient history
    )

    # 2. Consecutive loss streak heading into this trade
    streak, current = [], 0
    for v in df["is_profitable"].shift(1).fillna(1).values:
        current = current + 1 if v == 0 else 0
        streak.append(current)
    df["recent_drawdown_streak"] = streak

    # 3. Bars since last profitable trade (capped at 50 to limit outlier influence)
    bars, since = [], 0
    for v in df["is_profitable"].shift(1).fillna(1).values:
        since = 0 if v == 1 else min(since + 1, 50)
        bars.append(since)
    df["bars_since_profitable"] = bars

    return df


# ── Load data ─────────────────────────────────────────────────────────────────

def load_trades(min_rows: int = 1000) -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        log.warning("DB not found — harvester hasn't run yet")
        return None

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT {', '.join(FEATURE_COLS)},
               pnl, pnl_r, is_profitable,
               window_start, harvested_at,
               symbol, strategy
        FROM trades
    """).df()
    con.close()

    if len(df) < min_rows:
        log.info(f"Only {len(df)} trades — need {min_rows} to retrain")
        return None

    df["window_start"] = pd.to_datetime(df["window_start"])
    df["harvested_at"] = pd.to_datetime(df["harvested_at"])

    # Inject sequence context features (no look-ahead — uses shift(1))
    df = add_lagged_features(df)
    log.info("  Lagged sequence features injected: rolling_win_rate_regime, "
             "recent_drawdown_streak, bars_since_profitable")
    return df


# ── Train XGBoost ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, float]:
    """Train XGBoost to predict P(profitable). Returns (model, val_accuracy)."""
    X = df[FEATURE_COLS].fillna(0).astype(np.float32)
    y = df["is_profitable"].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,   # prevent overfitting to thin regimes
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_acc = (model.predict(X_val) == y_val).mean()
    log.info(f"  Model trained on {len(X_tr):,} trades — val accuracy: {val_acc*100:.1f}%")
    return model, val_acc


# ── Progressive threshold selection ──────────────────────────────────────────

def simulate_threshold(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    threshold: float,
    month_label: str,
) -> Tuple[float, int, float]:
    """
    Apply threshold to trades in month_label (YYYY-MM), return (total_pnl, n_trades, win_rate).
    A trade is 'taken' if model P(profitable) >= threshold.
    """
    mask = df["window_start"].dt.to_period("M").astype(str) == month_label
    subset = df[mask].copy()
    if len(subset) < 10:
        return 0.0, 0, 0.0

    X = subset[FEATURE_COLS].fillna(0).astype(np.float32)
    proba = model.predict_proba(X)[:, 1]  # P(profitable)
    taken = proba >= threshold

    taken_df = subset[taken]
    if len(taken_df) == 0:
        return 0.0, 0, 0.0

    total_pnl  = float(taken_df["pnl"].sum())
    n_trades   = len(taken_df)
    win_rate   = float(taken_df["is_profitable"].mean())
    return total_pnl, n_trades, win_rate


def select_threshold(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    current_threshold: float,
    history: list,
) -> Tuple[float, dict]:
    """
    Core rule: pick the threshold that maximises monthly PnL for the most recent
    complete month, subject to: PnL >= PnL of the month before that.

    This enforces progressive improvement — selectivity only stays if money grows.
    """
    periods = sorted(df["window_start"].dt.to_period("M").astype(str).unique())
    if len(periods) < 2:
        log.info("  Not enough monthly data to evaluate thresholds")
        return current_threshold, {}

    this_month = periods[-1]
    prev_month = periods[-2]

    results = {}
    log.info(f"  Evaluating thresholds on {this_month} vs baseline {prev_month}")

    # Baseline: what did prev month earn at current threshold?
    baseline_pnl, baseline_n, baseline_wr = simulate_threshold(
        df, model, current_threshold, prev_month
    )
    log.info(f"    Baseline [{prev_month}] threshold={current_threshold:.2f}  "
             f"PnL=${baseline_pnl:,.0f}  n={baseline_n}  WR={baseline_wr*100:.1f}%")

    best_threshold = current_threshold
    best_pnl = -np.inf

    for thr in THRESHOLD_CANDIDATES:
        pnl, n, wr = simulate_threshold(df, model, thr, this_month)
        if n < MIN_TRADES_FOR_EVAL:
            log.info(f"    thr={thr:.2f}  n={n} (too few trades, skip)")
            continue

        results[thr] = {"pnl": pnl, "n_trades": n, "win_rate": wr}
        log.info(f"    thr={thr:.2f}  PnL=${pnl:,.0f}  n={n}  WR={wr*100:.1f}%")

        # Must exceed baseline monthly PnL to be valid
        if pnl > baseline_pnl and pnl > best_pnl:
            best_pnl = pnl
            best_threshold = thr

    if best_threshold == current_threshold:
        log.info(f"  → No improvement found. Keeping threshold={current_threshold:.2f}")
    else:
        this_pnl, this_n, this_wr = results[best_threshold]["pnl"], results[best_threshold]["n_trades"], results[best_threshold]["win_rate"]
        log.info(
            f"  → UPGRADE threshold {current_threshold:.2f} → {best_threshold:.2f}  "
            f"PnL ${baseline_pnl:,.0f} → ${this_pnl:,.0f}  "
            f"WR {baseline_wr*100:.1f}% → {this_wr*100:.1f}%"
        )

    decision = {
        "ts":                str(datetime.utcnow()),
        "prev_threshold":    current_threshold,
        "new_threshold":     best_threshold,
        "baseline_month":    prev_month,
        "baseline_pnl":      baseline_pnl,
        "baseline_n":        baseline_n,
        "baseline_wr":       baseline_wr,
        "eval_month":        this_month,
        "eval_results":      results,
        "upgraded":          best_threshold != current_threshold,
    }
    return best_threshold, decision


# ── Persistence ───────────────────────────────────────────────────────────────

def load_threshold_state() -> Tuple[float, list]:
    threshold_file = MODELS_DIR / "current_threshold.json"
    history_file   = MODELS_DIR / "threshold_history.json"

    threshold = 0.50  # start permissive
    if threshold_file.exists():
        threshold = json.loads(threshold_file.read_text())["threshold"]

    history = []
    if history_file.exists():
        history = json.loads(history_file.read_text())

    return threshold, history


def save_threshold_state(threshold: float, decision: dict, history: list) -> None:
    history.append(decision)
    (MODELS_DIR / "current_threshold.json").write_text(
        json.dumps({"threshold": threshold, "updated_at": str(datetime.utcnow())}, indent=2)
    )
    (MODELS_DIR / "threshold_history.json").write_text(
        json.dumps(history[-200:], indent=2)  # keep last 200 decisions
    )


# ── Progression report ────────────────────────────────────────────────────────

def log_progression(df: pd.DataFrame, model: xgb.XGBClassifier, threshold: float) -> None:
    """Print month-by-month PnL with current threshold — shows the money trend."""
    periods = sorted(df["window_start"].dt.to_period("M").astype(str).unique())
    log.info("  ── Monthly Progression (current threshold) ──")
    prev_pnl = None
    for p in periods[-6:]:  # last 6 months
        pnl, n, wr = simulate_threshold(df, model, threshold, p)
        arrow = ""
        if prev_pnl is not None:
            arrow = "▲" if pnl > prev_pnl else "▼"
        log.info(f"    {p}  PnL=${pnl:>10,.0f}  n={n:4d}  WR={wr*100:5.1f}%  {arrow}")
        prev_pnl = pnl
    log.info("  ─────────────────────────────────────────────")


# ── Main retrain cycle ────────────────────────────────────────────────────────

def retrain_cycle() -> None:
    log.info("─── Retrain cycle start ───")

    df = load_trades()
    if df is None:
        log.info("No data yet — skipping")
        return

    log.info(f"  Loaded {len(df):,} trades from DB")

    current_threshold, history = load_threshold_state()
    log.info(f"  Current threshold: {current_threshold:.2f}")

    model, val_acc = train_model(df)

    # Save model
    model_path = MODELS_DIR / "xgb_veto.json"
    model.save_model(str(model_path))
    log.info(f"  Model saved → {model_path}")

    # Feature importance snapshot
    importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1]
    )
    log.info("  Feature importances: " + "  ".join(f"{f}={v:.3f}" for f, v in importance[:5]))

    # Select threshold
    new_threshold, decision = select_threshold(df, model, current_threshold, history)
    save_threshold_state(new_threshold, decision, history)

    # Progression report
    log_progression(df, model, new_threshold)

    log.info(f"─── Cycle complete. Active threshold: {new_threshold:.2f} ───")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle then exit")
    args = parser.parse_args()

    if args.once:
        retrain_cycle()
    else:
        log.info(f"Retrain loop starting — interval: {RETRAIN_INTERVAL_SEC//3600}h")
        while True:
            retrain_cycle()
            log.info(f"Sleeping {RETRAIN_INTERVAL_SEC//3600}h until next cycle...")
            time.sleep(RETRAIN_INTERVAL_SEC)
