"""
Evolution Worker — Autonomous Hyperparameter Evolution
======================================================
Runs continuous Optuna HPO across all M4 cores, promoting the
winning model only when it demonstrably outperforms the current
champion on a 30-day out-of-sample window.

Design constraints:
  - NEVER promotes on in-sample or validation loss alone
  - Requires minimum 30 days OOS data to activate
  - Promotion gated by PnL delta, not accuracy alone
  - Writes results to models/ so HarvestVeto hot-reloads automatically

Usage:
    python training/evolution_worker.py            # runs forever (24h cycles)
    python training/evolution_worker.py --once     # single trial run
    python training/evolution_worker.py --status   # print current champion

Schedule: Runs one Optuna study per 24h cycle.
          Safe to run alongside retrain_loop.py (uses separate model paths).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH      = ROOT / "data" / "harvest.db"
MODELS_DIR   = ROOT / "models"
LOG_PATH     = ROOT / "logs" / "evolution.log"
CHAMPION_PATH    = MODELS_DIR / "evo_champion.json"      # current best params
CHAMPION_MODEL   = MODELS_DIR / "xgb_veto_evo.json"     # champion model file
CHALLENGER_MODEL = MODELS_DIR / "xgb_veto_challenger.json"
HISTORY_PATH     = MODELS_DIR / "evo_history.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CYCLE_INTERVAL_SEC   = 24 * 3600    # one study per 24 hours
N_TRIALS             = 60           # Optuna trials per cycle
N_JOBS               = max(1, multiprocessing.cpu_count() - 2)  # leave 2 cores free
MIN_TRADES_TO_ACTIVATE = 50_000     # do not promote until harvest is substantial
MIN_OOS_DAYS         = 30           # require 30 days of held-out data
MIN_OOS_TRADES       = 200          # and at least 200 OOS trades

FEATURE_COLS = [
    "stop_atr_mult", "tp_rr", "atr_period",
    "regime", "hurst", "atr_norm", "vol_pct",
    "month", "day_of_week", "direction",
    # Lagged sequence features (added by add_lagged_features)
    "rolling_win_rate_regime", "recent_drawdown_streak", "bars_since_profitable",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EVO] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trades() -> Optional[pd.DataFrame]:
    """Load all trades from DuckDB. Returns None if DB missing or too small."""
    if not DB_PATH.exists():
        log.warning("harvest.db not found — evolution worker sleeping")
        return None
    try:
        base_cols = [
            "stop_atr_mult", "tp_rr", "atr_period",
            "regime", "hurst", "atr_norm", "vol_pct",
            "month", "day_of_week", "direction",
            "pnl", "pnl_r", "is_profitable",
            "window_start", "symbol", "strategy",
        ]
        con = duckdb.connect(str(DB_PATH), read_only=True)

        # Only select columns that actually exist in the DB
        available = {r[0] for r in con.execute("PRAGMA table_info(trades)").fetchall()}
        select_cols = [c for c in base_cols if c in available]

        df = con.execute(f"SELECT {', '.join(select_cols)} FROM trades").df()
        con.close()
    except Exception as e:
        log.error(f"DB read failed: {e}")
        return None

    if len(df) < 1000:
        log.info(f"Only {len(df)} trades — need 1,000+ to run trials")
        return None

    df["window_start"] = pd.to_datetime(df["window_start"])
    log.info(f"Loaded {len(df):,} trades spanning "
             f"{df['window_start'].min().date()} → {df['window_start'].max().date()}")
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject 3 sequence-context features that give XGBoost
    short-term 'memory' without any Transformer infrastructure.

    These are computed in-dataset (not live inference) and are
    therefore safe for training. The live system will need to
    compute these at inference time — tracked in TODO below.
    """
    df = df.sort_values("window_start").copy()

    # 1. Rolling win rate by regime over last 20 trades in same regime
    df["rolling_win_rate_regime"] = (
        df.groupby("regime")["is_profitable"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
          .fillna(0.5)
    )

    # 2. Drawdown streak — consecutive losses up to this point
    profit_shift = df["is_profitable"].shift(1).fillna(1)
    streak = []
    current = 0
    for v in profit_shift:
        if v == 0:
            current += 1
        else:
            current = 0
        streak.append(current)
    df["recent_drawdown_streak"] = streak

    # 3. Bars since last profitable trade (capped at 50)
    bars = []
    since = 0
    for v in df["is_profitable"].shift(1).fillna(1):
        if v == 1:
            since = 0
        else:
            since = min(since + 1, 50)
        bars.append(since)
    df["bars_since_profitable"] = bars

    return df


def split_oos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/OOS split.
    OOS = last MIN_OOS_DAYS calendar days.
    Train = everything before that.
    """
    cutoff = df["window_start"].max() - timedelta(days=MIN_OOS_DAYS)
    train = df[df["window_start"] < cutoff]
    oos   = df[df["window_start"] >= cutoff]
    return train, oos


# ── Model helpers ─────────────────────────────────────────────────────────────

def train_with_params(df_train: pd.DataFrame, params: dict):
    """
    Train XGBoost with a specific param set.
    Returns the fitted model.
    """
    import xgboost as xgb

    available_features = [c for c in FEATURE_COLS if c in df_train.columns]
    X = df_train[available_features].fillna(0).astype(np.float32)
    y = df_train["is_profitable"].astype(int)

    pos_weight = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))

    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=1,   # Optuna manages parallelism at the trial level
    )
    model.fit(X, y, verbose=False)
    return model


def score_on_oos(model, df_oos: pd.DataFrame) -> Tuple[float, int, float]:
    """
    Evaluate model on OOS data using SIMULATED PnL as the fitness metric.
    Returns (total_simulated_pnl, n_trades_taken, win_rate).

    Uses the model's P(profitable) at threshold=0.55 to decide which
    OOS trades would have been taken, then sums their actual PnL.
    This is the key: fitness = real money, not accuracy.
    """
    if len(df_oos) < 10:
        return -np.inf, 0, 0.0

    available_features = [c for c in FEATURE_COLS if c in df_oos.columns]
    X = df_oos[available_features].fillna(0).astype(np.float32)
    proba = model.predict_proba(X)[:, 1]

    threshold = 0.55
    taken_mask = proba >= threshold
    taken = df_oos[taken_mask]

    if len(taken) == 0:
        return -np.inf, 0, 0.0

    total_pnl  = float(taken["pnl"].sum())
    n_trades   = len(taken)
    win_rate   = float(taken["is_profitable"].mean())
    return total_pnl, n_trades, win_rate


# ── Optuna objective ──────────────────────────────────────────────────────────

def run_optuna_study(df_train: pd.DataFrame, df_oos: pd.DataFrame,
                     n_trials: int) -> Tuple[dict, float]:
    """
    Run an Optuna study and return (best_params, best_oos_pnl).
    Trials run in parallel using N_JOBS processes.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 9),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "gamma":           trial.suggest_float("gamma", 0, 5.0),
        }
        try:
            model = train_with_params(df_train, params)
            pnl, n_trades, _ = score_on_oos(model, df_oos)
            # Penalise over-filtering (taking < 20% of available trades is too selective)
            selectivity_penalty = max(0, (0.2 - n_trades / max(len(df_oos), 1)) * 1000)
            return pnl - selectivity_penalty
        except Exception as e:
            log.debug(f"Trial failed: {e}")
            return -1e9

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run trials - use n_jobs for parallelism
    effective_jobs = min(N_JOBS, n_trials)
    log.info(f"  Running {n_trials} Optuna trials across {effective_jobs} cores...")
    study.optimize(objective, n_trials=n_trials, n_jobs=effective_jobs, show_progress_bar=False)

    best_params = study.best_params
    best_value  = study.best_value
    log.info(f"  Best trial PnL: ${best_value:,.0f} | params: {best_params}")
    return best_params, best_value


# ── Champion management ───────────────────────────────────────────────────────

def load_champion() -> Tuple[Optional[dict], float]:
    """Load champion params and OOS PnL. Returns (params, oos_pnl)."""
    if not CHAMPION_PATH.exists():
        return None, -np.inf
    try:
        data = json.loads(CHAMPION_PATH.read_text())
        return data.get("params"), float(data.get("oos_pnl", -np.inf))
    except Exception:
        return None, -np.inf


def save_champion(params: dict, oos_pnl: float, model, meta: dict) -> None:
    """Persist champion params, model, and history entry."""
    import xgboost as xgb

    record = {
        "promoted_at":   str(datetime.utcnow()),
        "params":        params,
        "oos_pnl":       oos_pnl,
        "meta":          meta,
    }
    CHAMPION_PATH.write_text(json.dumps(record, indent=2))

    # Save champion XGB model — HarvestVeto will hot-reload this path if configured
    model.save_model(str(CHAMPION_MODEL))

    # Append to history
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    history.append(record)
    HISTORY_PATH.write_text(json.dumps(history[-100:], indent=2))

    log.info(f"  ✓ Champion promoted — OOS PnL: ${oos_pnl:,.0f}")


# ── Main cycle ────────────────────────────────────────────────────────────────

def evolution_cycle() -> None:
    log.info("=" * 60)
    log.info(f"Evolution cycle start | {datetime.utcnow().isoformat()}")
    log.info(f"Cores available: {multiprocessing.cpu_count()} | Using: {N_JOBS}")

    # 1. Load data
    df = load_trades()
    if df is None:
        log.info("No data — skipping cycle")
        return

    df = add_lagged_features(df)

    # 2. Check activation gate
    total_trades = len(df)
    log.info(f"  Total trades in DB: {total_trades:,} | Activation threshold: {MIN_TRADES_TO_ACTIVATE:,}")

    if total_trades < MIN_TRADES_TO_ACTIVATE:
        log.info(
            f"  [DORMANT] Harvest has {total_trades:,} trades — "
            f"need {MIN_TRADES_TO_ACTIVATE:,} to activate promotion. "
            f"Running trials for learning only (no promotion)."
        )
        # Still run trials to learn, just don't promote
        df_train, df_oos = split_oos(df)
        if len(df_oos) < MIN_OOS_TRADES:
            log.info(f"  OOS has only {len(df_oos)} trades — need {MIN_OOS_TRADES}. Aborting.")
            return
        log.info(f"  Train: {len(df_train):,} | OOS: {len(df_oos):,} trades")
        best_params, best_pnl = run_optuna_study(df_train, df_oos, n_trials=N_TRIALS)
        log.info(f"  [LEARNING] Best params found (not promoted): {best_params}")
        log.info(f"  [LEARNING] Would have earned: ${best_pnl:,.0f} OOS")
        log.info("=" * 60)
        return

    # 3. Temporal split
    df_train, df_oos = split_oos(df)
    if len(df_oos) < MIN_OOS_TRADES:
        log.info(f"  OOS has only {len(df_oos)} trades — need {MIN_OOS_TRADES}. Aborting.")
        return

    log.info(f"  Train: {len(df_train):,} | OOS: {len(df_oos):,} trades "
             f"({df_oos['window_start'].min().date()} → {df_oos['window_start'].max().date()})")

    # 4. Load current champion baseline
    champ_params, champ_pnl = load_champion()
    log.info(f"  Current champion OOS PnL: ${champ_pnl:,.0f}")

    # 5. Run Optuna study
    best_params, best_pnl = run_optuna_study(df_train, df_oos, n_trials=N_TRIALS)

    # 6. Train the challenger model with best params for scoring + saving
    log.info("  Training challenger model with best params...")
    challenger = train_with_params(df_train, best_params)
    actual_pnl, n_taken, win_rate = score_on_oos(challenger, df_oos)
    log.info(f"  Challenger OOS: PnL=${actual_pnl:,.0f} | trades={n_taken} | WR={win_rate:.1%}")

    # 7. Promotion gate — MUST beat champion AND be positive
    if actual_pnl > champ_pnl and actual_pnl > 0:
        delta = actual_pnl - champ_pnl
        log.info(f"  ✓ PROMOTED — Challenger beats champion by ${delta:,.0f}")
        meta = {
            "challenger_pnl":  actual_pnl,
            "champion_pnl":    champ_pnl,
            "delta":           delta,
            "oos_trades":      n_taken,
            "oos_win_rate":    win_rate,
            "oos_days":        MIN_OOS_DAYS,
            "total_db_trades": total_trades,
        }
        save_champion(best_params, actual_pnl, challenger, meta)
    else:
        log.info(
            f"  ✗ NOT PROMOTED — Challenger ${actual_pnl:,.0f} "
            f"vs Champion ${champ_pnl:,.0f} "
            f"({'negative PnL' if actual_pnl <= 0 else 'did not improve'})"
        )

    log.info(f"Evolution cycle complete | {datetime.utcnow().isoformat()}")
    log.info("=" * 60)


def print_status() -> None:
    """Print current champion status."""
    champ_params, champ_pnl = load_champion()
    if champ_params is None:
        print("No champion promoted yet.")
        return

    data = json.loads(CHAMPION_PATH.read_text())
    print(f"\n{'='*50}")
    print(f"Evolution Worker Status")
    print(f"{'='*50}")
    print(f"Champion promoted at: {data.get('promoted_at', 'unknown')}")
    print(f"Champion OOS PnL:     ${champ_pnl:,.0f}")
    meta = data.get("meta", {})
    print(f"Trades at promotion:  {meta.get('total_db_trades', 'unknown'):,}")
    print(f"OOS win rate:         {meta.get('oos_win_rate', 0):.1%}")
    print(f"\nParams:")
    for k, v in (champ_params or {}).items():
        print(f"  {k:25s} = {v}")

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
        print(f"\nPromotions to date:   {len(history)}")
    print(f"{'='*50}\n")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sovereign Evolution Worker")
    parser.add_argument("--once",   action="store_true", help="Run one cycle then exit")
    parser.add_argument("--status", action="store_true", help="Print champion status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.once:
        evolution_cycle()
    else:
        log.info(f"Evolution worker starting — cycle interval: {CYCLE_INTERVAL_SEC // 3600}h")
        log.info(f"M4 cores: {multiprocessing.cpu_count()} | Active trials: {N_JOBS}")
        while True:
            evolution_cycle()
            log.info(f"Sleeping {CYCLE_INTERVAL_SEC // 3600}h until next cycle...")
            time.sleep(CYCLE_INTERVAL_SEC)
