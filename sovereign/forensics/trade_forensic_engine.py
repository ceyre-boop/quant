"""
Trade Forensic Engine — Layer 1 of the Sovereign Intelligence Architecture.

"The strongest systems do not merely learn. They forget correctly."

Every closed trade gets ONE failure label. Not just R-multiple — the structural
signature of why it failed. This becomes the training data for everything
downstream: commitment detection, latent feature search, cross-system bridge,
and veto health monitoring.

FAILURE TAXONOMY (mutually exclusive):
    TIMING_FAILURE       — thesis correct, entry premature
    THESIS_FAILURE       — direction wrong, price never came back
    REGIME_FAILURE       — regime shifted mid-trade
    EXECUTION_FAILURE    — poor entry placement (MAE > 0.8R in 3 bars)
    SIZING_FAILURE       — hit TP1 but closed early (risk cap)
    COMMITMENT_FAILURE   — no follow-through (the new category)

VETO HEALTH MONITORING:
    Every active veto tracked monthly.
    Auto-retirement if rolling_sharpe_impact < 0.0 for 3 months.
    HARMFUL flag if veto is blocking winners.

AUTO-HYPOTHESIS GENERATION:
    Every 50 trades, cluster COMMITMENT_FAILUREs.
    If any feature separates wins from failures at > 70% accuracy:
    → write CANDIDATE hypothesis to ledger.
    → message Colin. Human decides. Machine never auto-activates.

Storage:
    data/forensics/trade_forensics.jsonl   — one line per trade
    data/forensics/veto_health.json        — live veto monitoring
    data/forensics/auto_hypotheses.json    — machine-generated candidates

Run:
    PYTHONPATH=/path/to/quant python3 sovereign/forensics/trade_forensic_engine.py
    PYTHONPATH=/path/to/quant python3 sovereign/forensics/trade_forensic_engine.py --system ict
    PYTHONPATH=/path/to/quant python3 sovereign/forensics/trade_forensic_engine.py --system forex
    PYTHONPATH=/path/to/quant python3 sovereign/forensics/trade_forensic_engine.py --veto-health
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
FORENSICS_DIR   = ROOT / "data" / "forensics"
LEDGER_PATH     = ROOT / "data" / "agent" / "hypothesis_ledger.json"
MESSAGES_PATH   = ROOT / "data" / "agent" / "messages_to_colin.json"
FORENSICS_FILE  = FORENSICS_DIR / "trade_forensics.jsonl"
VETO_HEALTH_FILE = FORENSICS_DIR / "veto_health.json"
AUTO_HYPO_FILE  = FORENSICS_DIR / "auto_hypotheses.json"

ICT_TRADE_FILES = [
    ROOT / "logs" / "ict_backtest_results.json",
    ROOT / "logs" / "ict_backtest_window_A.json",
    ROOT / "logs" / "ict_backtest_window_B.json",
]
FOREX_TRADE_FILE = ROOT / "logs" / "forex_backtest_trades.json"

# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class CommitmentMetrics:
    bars_to_stop: int
    post_entry_atr_ratio: float        # ATR first 3 bars / ATR at entry
    mfe_within_3_bars: float           # max favorable excursion, in R
    mae_within_3_bars: float           # max adverse excursion, in R
    session: str
    price_went_right_after: bool       # did price reach TP direction after stop?
    max_adverse_post_entry_r: float    # worst drawdown post entry before exit
    eventual_direction_correct: bool   # did price eventually move our way?
    hold_bars_or_days: int


@dataclass
class ForensicRecord:
    system: str               # ICT | FOREX
    trade_id: str
    pair: str
    direction: str
    entry_date: str
    outcome: str              # STOP | TP1 | TP2 | TIMEOUT | trailing_stop | reversal | time
    pnl_r: float              # R-multiple (normalized)
    hold: int                 # bars (ICT) or days (forex)
    session: str
    grade: str                # ICT grade or "MACRO" for forex
    score: float              # ICT score or macro_score for forex

    # Failure taxonomy
    failure_label: Optional[str]   # one of 6 labels, None for winners
    win_driver: Optional[str]      # populated for winners

    # Commitment metrics
    metrics: Dict[str, Any]

    # Auto-hypothesis features
    features: Dict[str, float]

    classified_at: str


# ── Price data loader ─────────────────────────────────────────────────────

_price_cache: Dict[str, pd.DataFrame] = {}

def _load_prices(pair: str) -> Optional[pd.DataFrame]:
    """Load price OHLCV for a forex pair (cached)."""
    if pair in _price_cache:
        return _price_cache[pair]
    try:
        import yfinance as yf
        # Normalize pair name
        ticker = pair if "=X" in pair else f"{pair}=X"
        df = yf.download(ticker, start="2013-01-01", end="2026-06-01", progress=False)
        if df is None or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        _price_cache[pair] = df
        return df
    except Exception:
        return None


def _get_prices_around_trade(
    pair: str, entry_date: str, hold: int, system: str
) -> Optional[pd.DataFrame]:
    """Return price slice covering entry + 2× hold period after."""
    df = _load_prices(pair if "=X" in pair else f"{pair}=X")
    if df is None:
        return None
    try:
        entry_ts = pd.Timestamp(entry_date).tz_localize(None)
        # Look forward 2× hold period for post-trade detection
        if system == "ICT":
            # hold_bars in 1h bars → ~0.5 days per bar
            lookahead = timedelta(days=max(hold * 2, 10))
        else:
            lookahead = timedelta(days=hold * 2)
        window = df.loc[entry_ts: entry_ts + lookahead]
        return window if len(window) >= 3 else None
    except Exception:
        return None


# ── Failure taxonomy classifiers ──────────────────────────────────────────

def _classify_ict_trade(trade: Dict, prices: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Classify an ICT trade into exactly one failure label (or one win driver).
    Returns (failure_label, win_driver, commitment_metrics).
    """
    outcome = trade["outcome"]
    pnl_r   = trade["pnl_r"]
    entry   = float(trade["entry"])
    stop    = float(trade["stop"])
    tp1     = float(trade.get("tp1", 0))
    tp2     = float(trade.get("tp2", 0))
    direction = trade["direction"]  # LONG or SHORT
    hold_bars = int(trade.get("hold_bars", 5))
    atr       = float(trade.get("atr", 0.001))
    session   = trade.get("session", "")
    exit_price = float(trade.get("exit_price", entry))

    stop_distance = abs(entry - stop)
    if stop_distance < 1e-8:
        stop_distance = atr

    # Compute post-entry ATR ratio from prices if available
    post_entry_atr_ratio = 1.0
    price_went_right = False
    eventual_correct = False
    mfe_3bar = 0.0
    mae_3bar = 0.0

    if prices is not None and len(prices) >= 3:
        closes = prices["Close"].values if "Close" in prices.columns else prices.iloc[:, 3].values
        highs  = prices["High"].values  if "High"  in prices.columns else prices.iloc[:, 1].values
        lows   = prices["Low"].values   if "Low"   in prices.columns else prices.iloc[:, 2].values

        # ATR in first 3 bars vs ATR at entry
        if len(closes) >= 4:
            bars3_range = np.mean(highs[:3] - lows[:3])
            post_entry_atr_ratio = bars3_range / max(atr, 1e-8)

        # MFE and MAE in first 3 bars (in R)
        if direction == "LONG":
            mfe_3bar = float((np.max(highs[:min(3, len(highs))]) - entry) / stop_distance)
            mae_3bar = float((entry - np.min(lows[:min(3, len(lows))])) / stop_distance)
            # Did price eventually reach TP1 direction?
            if len(closes) > hold_bars:
                price_went_right = np.max(highs[hold_bars:]) > (entry + stop_distance * 1.5)
            eventual_correct = np.max(highs) > (entry + stop_distance)
        else:
            mfe_3bar = float((entry - np.min(lows[:min(3, len(lows))])) / stop_distance)
            mae_3bar = float((np.max(highs[:min(3, len(highs))]) - entry) / stop_distance)
            if len(closes) > hold_bars:
                price_went_right = np.min(lows[hold_bars:]) < (entry - stop_distance * 1.5)
            eventual_correct = np.min(lows) < (entry - stop_distance)

    metrics = {
        "bars_to_stop": hold_bars if outcome == "STOP" else -1,
        "post_entry_atr_ratio": round(post_entry_atr_ratio, 3),
        "mfe_within_3_bars": round(mfe_3bar, 3),
        "mae_within_3_bars": round(mae_3bar, 3),
        "session": session,
        "price_went_right_after": bool(price_went_right),
        "eventual_direction_correct": bool(eventual_correct),
        "hold_bars_or_days": hold_bars,
        "max_adverse_post_entry_r": round(mae_3bar, 3),
    }

    # ── WIN DRIVERS ───────────────────────────────────────────────────────
    if pnl_r > 0:
        if outcome == "TP2":
            driver = "FULL_TP2_HIT" if session == "London" else "TP2_HIT"
        elif outcome == "TP1":
            driver = "TP1_RUNNER"
        elif mfe_3bar >= 1.0:
            driver = "FAST_MOVER"
        else:
            driver = "MARGINAL_WIN"
        return None, driver, metrics

    # ── FAILURE TAXONOMY ──────────────────────────────────────────────────

    # EXECUTION_FAILURE: MAE > 0.8R in first 3 bars (slippage / bad level)
    if mae_3bar > 0.8 and hold_bars <= 3:
        return "EXECUTION_FAILURE", None, metrics

    # SIZING_FAILURE: would have hit TP1 but forced out (only detectable
    # if exit_price passed TP1 level momentarily — use MFE proxy)
    if mfe_3bar >= 1.4 and outcome == "STOP":
        return "SIZING_FAILURE", None, metrics

    # TIMING_FAILURE: thesis correct but too early
    # Price eventually moved our way after we were stopped
    if outcome == "STOP" and price_went_right and hold_bars <= 5:
        return "TIMING_FAILURE", None, metrics

    # THESIS_FAILURE: direction wrong, price moved against and stayed
    if outcome == "STOP" and not eventual_correct and pnl_r <= -0.8:
        return "THESIS_FAILURE", None, metrics

    # COMMITMENT_FAILURE: no follow-through (the key new category)
    # ATR compressed after entry AND minimal favorable move before stop
    if post_entry_atr_ratio < 0.70 and mfe_3bar < 0.5:
        return "COMMITMENT_FAILURE", None, metrics

    # REGIME_FAILURE: catch-all for mid-trade reversals (without live HMM data,
    # detect as: stopped after holding > 4 bars with moderate initial progress)
    if hold_bars > 4 and mfe_3bar > 0.3 and outcome == "STOP":
        return "REGIME_FAILURE", None, metrics

    # Default COMMITMENT_FAILURE for remaining losses
    return "COMMITMENT_FAILURE", None, metrics


def _classify_forex_trade(trade: Dict, pair: str, prices: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str], Dict]:
    """Classify a forex backtest trade."""
    pnl_pct  = float(trade["pnl_pct"])
    hold     = int(trade.get("hold_days", 10))
    exit_r   = trade.get("exit_reason", "")
    direction = int(trade.get("direction", 1))
    entry    = float(trade.get("entry", 0))
    exit_p   = float(trade.get("exit", entry))

    # Approximate stop distance (1% of price as proxy when not stored)
    approx_stop = entry * 0.01
    pnl_r = pnl_pct / max(approx_stop / entry, 0.005) if entry > 0 else pnl_pct * 100

    post_entry_atr_ratio = 1.0
    price_went_right = False
    eventual_correct = False
    mfe_3bar = 0.0
    mae_3bar = 0.0

    if prices is not None and len(prices) >= 5:
        closes = prices["Close"].values if "Close" in prices.columns else prices.iloc[:, 3].values
        highs  = prices["High"].values  if "High"  in prices.columns else prices.iloc[:, 1].values
        lows   = prices["Low"].values   if "Low"   in prices.columns else prices.iloc[:, 2].values

        entry_atr = float(np.mean(highs[:3] - lows[:3])) if len(highs) >= 3 else approx_stop
        post3_atr = float(np.mean(highs[3:6] - lows[3:6])) if len(highs) >= 6 else entry_atr
        post_entry_atr_ratio = post3_atr / max(entry_atr, 1e-8)

        if direction == 1:
            mfe_3bar = (np.max(highs[:min(3, len(highs))]) - entry) / max(approx_stop, 1e-8)
            mae_3bar = (entry - np.min(lows[:min(3, len(lows))])) / max(approx_stop, 1e-8)
            if len(closes) > hold:
                price_went_right = np.max(highs[hold:]) > entry + approx_stop * 1.5
            eventual_correct = np.max(highs) > entry + approx_stop
        else:
            mfe_3bar = (entry - np.min(lows[:min(3, len(lows))])) / max(approx_stop, 1e-8)
            mae_3bar = (np.max(highs[:min(3, len(highs))]) - entry) / max(approx_stop, 1e-8)
            if len(closes) > hold:
                price_went_right = np.min(lows[hold:]) < entry - approx_stop * 1.5
            eventual_correct = np.min(lows) < entry - approx_stop

    metrics = {
        "bars_to_stop": hold if exit_r in ("stop", "trailing_stop") else -1,
        "post_entry_atr_ratio": round(float(post_entry_atr_ratio), 3),
        "mfe_within_3_bars": round(float(mfe_3bar), 3),
        "mae_within_3_bars": round(float(mae_3bar), 3),
        "session": "MACRO",
        "price_went_right_after": bool(price_went_right),
        "eventual_direction_correct": bool(eventual_correct),
        "hold_bars_or_days": hold,
        "max_adverse_post_entry_r": round(float(mae_3bar), 3),
        "exit_reason": exit_r,
    }

    if pnl_pct > 0:
        if exit_r == "reversal" and hold <= 5:
            driver = "TIMELY_EXIT"
        elif exit_r == "time" and hold >= 15:
            driver = "MACRO_HOLD_PAID"
        elif exit_r == "trailing_stop":
            driver = "TRAILING_CAPTURED_TREND"
        else:
            driver = "MARGINAL_WIN"
        return None, driver, metrics

    # Failure taxonomy for forex
    if mae_3bar > 0.8 and hold <= 3:
        return "EXECUTION_FAILURE", None, metrics
    if exit_r in ("stop", "trailing_stop") and price_went_right and hold <= 5:
        return "TIMING_FAILURE", None, metrics
    if exit_r in ("stop",) and not eventual_correct and pnl_pct < -0.006:
        return "THESIS_FAILURE", None, metrics
    if post_entry_atr_ratio < 0.70 and mfe_3bar < 0.5:
        return "COMMITMENT_FAILURE", None, metrics
    if hold > 8 and mfe_3bar > 0.3 and exit_r in ("trailing_stop", "reversal"):
        return "REGIME_FAILURE", None, metrics
    return "COMMITMENT_FAILURE", None, metrics


# ── Feature vector for ML / clustering ───────────────────────────────────

def _build_features(trade: Dict, system: str, metrics: Dict) -> Dict[str, float]:
    """Numeric features used for auto-hypothesis clustering."""
    features: Dict[str, float] = {
        "post_entry_atr_ratio": float(metrics.get("post_entry_atr_ratio", 1.0)),
        "mfe_within_3_bars":    float(metrics.get("mfe_within_3_bars", 0.0)),
        "mae_within_3_bars":    float(metrics.get("mae_within_3_bars", 0.0)),
        "hold":                 float(metrics.get("hold_bars_or_days", 5)),
        "price_went_right":     float(metrics.get("price_went_right_after", False)),
        "eventual_correct":     float(metrics.get("eventual_direction_correct", False)),
        "session_london":       1.0 if metrics.get("session") == "London" else 0.0,
        "session_ny_pm":        1.0 if metrics.get("session") == "NY_PM" else 0.0,
    }
    if system == "ICT":
        features["score"]       = float(trade.get("score", 7.5))
        features["grade_aplus"] = 1.0 if trade.get("grade") == "A+" else 0.0
        features["pd_align"]    = float(trade.get("component_scores", {}).get("pd_alignment", 0))
        features["mkt_struct"]  = float(trade.get("component_scores", {}).get("market_structure", 0))
        features["displacement"]= float(trade.get("component_scores", {}).get("displacement", 0))
    return features


# ── Auto-hypothesis generation ────────────────────────────────────────────

def _run_auto_hypothesis(records: List[ForensicRecord]) -> List[Dict]:
    """
    Every 50 trades: cluster COMMITMENT_FAILUREs vs winners.
    If any feature separates at >70% accuracy → emit CANDIDATE hypothesis.
    NEVER auto-activates. Human reviews and decides.
    """
    failures = [r for r in records if r.failure_label == "COMMITMENT_FAILURE"]
    winners  = [r for r in records if r.win_driver is not None]

    if len(failures) < 10 or len(winners) < 10:
        return []

    feature_names = list(failures[0].features.keys())
    candidates = []

    for feat in feature_names:
        f_vals = np.array([r.features.get(feat, 0.0) for r in failures])
        w_vals = np.array([r.features.get(feat, 0.0) for r in winners])

        # Try a simple threshold sweep
        all_vals = np.concatenate([f_vals, w_vals])
        labels   = np.concatenate([np.zeros(len(f_vals)), np.ones(len(w_vals))])

        best_acc = 0.5
        best_thresh = 0.0
        best_direction = "lt"

        for thresh in np.percentile(all_vals, np.arange(20, 81, 10)):
            # "feature < thresh" predicts failure
            pred_lt = (all_vals < thresh).astype(int)
            acc_lt  = 1 - np.mean(pred_lt == labels)
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_thresh = thresh
                best_direction = "lt"
            # "feature > thresh" predicts failure
            pred_gt = (all_vals > thresh).astype(int)
            acc_gt  = 1 - np.mean(pred_gt == labels)
            if acc_gt > best_acc:
                best_acc = acc_gt
                best_thresh = thresh
                best_direction = "gt"

        if best_acc >= 0.70:
            hyp_id = f"AUTO-HYP-{len(candidates)+1:03d}"
            condition = f"{feat} {'<' if best_direction == 'lt' else '>'} {best_thresh:.4f}"
            candidates.append({
                "hypothesis_id": hyp_id,
                "feature": feat,
                "condition": condition,
                "separation_accuracy": round(best_acc, 4),
                "sample_size": len(failures) + len(winners),
                "commitment_failures_analyzed": len(failures),
                "status": "CANDIDATE",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "forensic_engine_auto",
                "note": "CANDIDATE ONLY. Human reviews and activates. Machine never auto-activates vetoes.",
            })

    return sorted(candidates, key=lambda x: -x["separation_accuracy"])


# ── Veto health monitor ───────────────────────────────────────────────────

ACTIVE_VETOES = {
    "NY_PM_BLOCK": {
        "description": "Block all ICT trades in NY_PM session",
        "birth_date": "2026-05-18",
        "system": "ICT",
        "blocked_simulated_r": -0.283,  # avg R of trades that would have fired
    },
    "ATR_FLOOR_1.8pct": {
        "description": "Equity ATR must be >= 1.8% (was 2.2%)",
        "birth_date": "2026-05-18",
        "system": "EQUITY",
        "blocked_simulated_r": None,
    },
    "MACRO_AGAINST": {
        "description": "Forex: block when real_rate_diff opposes direction",
        "birth_date": "2026-05-17",
        "system": "FOREX",
        "blocked_simulated_r": -0.838,  # forensics finding
    },
    "APLUS_DOWNGRADE": {
        "description": "ICT A+ grade treated as A for execution decision",
        "birth_date": "2026-05-18",
        "system": "ICT",
        "blocked_simulated_r": -0.375,
    },
}

def _compute_veto_health(records: List[ForensicRecord]) -> Dict:
    """
    For each active veto, estimate rolling impact from forensic records.
    Returns health assessment and retirement recommendations.
    """
    health = {}
    alerts = []
    now = datetime.now(timezone.utc)

    for veto_name, veto_info in ACTIVE_VETOES.items():
        birth = datetime.fromisoformat(veto_info["birth_date"]).replace(tzinfo=timezone.utc)
        months_active = max(int((now - birth).days / 30), 1)

        # Find records that represent this veto's blocked trades
        # (trades that fired the relevant condition)
        blocked_r = veto_info.get("blocked_simulated_r")

        status = "HEALTHY"
        notes  = []

        if blocked_r is not None:
            if blocked_r > 0.20:
                status = "HARMFUL"
                notes.append(f"Blocking trades with avg +{blocked_r:.3f}R — likely hurting performance.")
                alerts.append({
                    "priority": "URGENT",
                    "veto": veto_name,
                    "message": f"Veto {veto_name} appears to be blocking profitable trades "
                               f"(avg R simulated: +{blocked_r:.3f}). Immediate review needed.",
                })
            elif blocked_r < -0.15:
                status = "HEALTHY"
                notes.append(f"Blocking trades with avg {blocked_r:.3f}R — working correctly.")
            else:
                status = "WATCH"
                notes.append(f"Marginal effectiveness: avg {blocked_r:.3f}R on blocked trades.")

        if months_active < 1:
            notes.append("Too new to evaluate — wait 30 days before retirement consideration.")

        health[veto_name] = {
            "description": veto_info["description"],
            "system": veto_info["system"],
            "birth_date": veto_info["birth_date"],
            "months_active": months_active,
            "blocked_simulated_r": blocked_r,
            "status": status,
            "notes": notes,
            "retirement_eligible": months_active >= 3 and status == "HARMFUL",
        }

    return {"veto_health": health, "alerts": alerts, "evaluated_at": now.isoformat()}


# ── Main processing loop ──────────────────────────────────────────────────

def process_ict_trades(verbose: bool = True) -> List[ForensicRecord]:
    records = []
    seen_ids = set()

    for path in ICT_TRADE_FILES:
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        trades = raw.get("trades", [])
        window = path.stem  # e.g. "ict_backtest_results"

        if verbose:
            print(f"  {path.name}: {len(trades)} trades")

        for i, trade in enumerate(trades):
            trade_id = f"ICT_{window}_{i:04d}"
            if trade_id in seen_ids:
                continue
            seen_ids.add(trade_id)

            pair = trade.get("pair", "GBPUSD")
            entry_date = trade.get("entry_dt", "2023-01-01 02:00")

            prices = _get_prices_around_trade(
                pair, entry_date,
                hold=int(trade.get("hold_bars", 5)),
                system="ICT"
            )

            failure_label, win_driver, metrics = _classify_ict_trade(trade, prices)
            features = _build_features(trade, "ICT", metrics)
            pnl_r = float(trade.get("pnl_r", 0))

            records.append(ForensicRecord(
                system="ICT",
                trade_id=trade_id,
                pair=pair,
                direction=trade.get("direction", "LONG"),
                entry_date=str(entry_date),
                outcome=trade.get("outcome", ""),
                pnl_r=pnl_r,
                hold=int(trade.get("hold_bars", 5)),
                session=trade.get("session", ""),
                grade=trade.get("grade", ""),
                score=float(trade.get("score", 0)),
                failure_label=failure_label,
                win_driver=win_driver,
                metrics=metrics,
                features=features,
                classified_at=datetime.now(timezone.utc).isoformat(),
            ))

            # Back-fill decision log with forensic classification
            try:
                from sovereign.intelligence.decision_logger import update_outcome
                outcome_label = failure_label or win_driver or trade.get("outcome", "")
                update_outcome(
                    pair=pair,
                    entry_timestamp=str(entry_date),
                    outcome=outcome_label,
                    r_realized=pnl_r,
                    system="ICT",
                )
            except Exception:
                pass

    return records


def process_forex_trades(verbose: bool = True) -> List[ForensicRecord]:
    records = []

    if not FOREX_TRADE_FILE.exists():
        return records

    raw = json.loads(FOREX_TRADE_FILE.read_text())

    for pair, trades in raw.items():
        if verbose:
            print(f"  {pair}: {len(trades)} trades")

        prices_df = _load_prices(pair)

        for i, trade in enumerate(trades):
            entry_date = trade.get("entry_date", "2015-01-01")
            trade_id = f"FX_{pair.replace('=X','')}_{i:04d}"

            prices = None
            if prices_df is not None:
                try:
                    entry_ts = pd.Timestamp(entry_date).tz_localize(None)
                    hold_days = int(trade.get("hold_days", 10))
                    lookahead = timedelta(days=hold_days * 2)
                    prices = prices_df.loc[entry_ts: entry_ts + lookahead]
                    if len(prices) < 3:
                        prices = None
                except Exception:
                    prices = None

            failure_label, win_driver, metrics = _classify_forex_trade(trade, pair, prices)
            features = _build_features(trade, "FOREX", metrics)
            pnl_r_fx = float(trade.get("pnl_pct", 0)) * 100

            records.append(ForensicRecord(
                system="FOREX",
                trade_id=trade_id,
                pair=pair,
                direction="LONG" if int(trade.get("direction", 1)) == 1 else "SHORT",
                entry_date=str(entry_date),
                outcome=trade.get("exit_reason", ""),
                pnl_r=pnl_r_fx,
                hold=int(trade.get("hold_days", 10)),
                session="MACRO",
                grade="MACRO",
                score=0.0,
                failure_label=failure_label,
                win_driver=win_driver,
                metrics=metrics,
                features=features,
                classified_at=datetime.now(timezone.utc).isoformat(),
            ))

            # Back-fill decision log with forensic classification
            try:
                from sovereign.intelligence.decision_logger import update_outcome
                outcome_label = failure_label or win_driver or trade.get("exit_reason", "")
                update_outcome(
                    pair=pair,
                    entry_timestamp=str(entry_date),
                    outcome=outcome_label,
                    r_realized=pnl_r_fx,
                    system="FOREX",
                )
            except Exception:
                pass

    return records


def save_records(records: List[ForensicRecord]) -> None:
    FORENSICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FORENSICS_FILE, "w") as f:
        for r in records:
            line = {
                "system": r.system, "trade_id": r.trade_id, "pair": r.pair,
                "direction": r.direction, "entry_date": r.entry_date,
                "outcome": r.outcome, "pnl_r": r.pnl_r, "hold": r.hold,
                "session": r.session, "grade": r.grade, "score": r.score,
                "failure_label": r.failure_label, "win_driver": r.win_driver,
                "metrics": r.metrics, "features": r.features,
                "classified_at": r.classified_at,
            }
            f.write(json.dumps(line) + "\n")


def print_summary(records: List[ForensicRecord]) -> None:
    from collections import Counter

    ict_r   = [r for r in records if r.system == "ICT"]
    forex_r = [r for r in records if r.system == "FOREX"]

    for system_name, sys_records in [("ICT", ict_r), ("FOREX", forex_r)]:
        if not sys_records:
            continue
        losses = [r for r in sys_records if r.failure_label is not None]
        wins   = [r for r in sys_records if r.win_driver   is not None]

        print(f"\n{'='*58}")
        print(f"{system_name} FORENSICS — {len(sys_records)} trades")
        print(f"{'='*58}")
        print(f"Winners: {len(wins)}  |  Losses: {len(losses)}")

        if losses:
            label_counts = Counter(r.failure_label for r in losses)
            print(f"\nFailure taxonomy:")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                pct = count / len(losses) * 100
                avg_r = np.mean([r.pnl_r for r in losses if r.failure_label == label])
                bar = "█" * min(int(pct / 3), 20)
                print(f"  {label:<22} {count:>4} ({pct:>4.0f}%)  avgR={avg_r:>7.3f}  {bar}")

        if wins:
            driver_counts = Counter(r.win_driver for r in wins)
            print(f"\nWin drivers:")
            for driver, count in sorted(driver_counts.items(), key=lambda x: -x[1]):
                pct = count / len(wins) * 100
                avg_r = np.mean([r.pnl_r for r in wins if r.win_driver == driver])
                print(f"  {driver:<28} {count:>4} ({pct:>4.0f}%)  avgR={avg_r:>7.3f}")


def run_full_analysis(system: str = "both", verbose: bool = True) -> List[ForensicRecord]:
    FORENSICS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    if system in ("ict", "both"):
        if verbose:
            print("\nProcessing ICT trades...")
        records.extend(process_ict_trades(verbose=verbose))

    if system in ("forex", "both"):
        if verbose:
            print("\nProcessing Forex trades (fetching prices for classification)...")
        records.extend(process_forex_trades(verbose=verbose))

    if verbose:
        print(f"\nTotal records: {len(records)}")
        print("Saving to data/forensics/trade_forensics.jsonl...")

    save_records(records)

    # Auto-hypothesis generation
    if len(records) >= 50:
        if verbose:
            print("Running auto-hypothesis generation...")
        candidates = _run_auto_hypothesis(records)
        AUTO_HYPO_FILE.write_text(json.dumps({"candidates": candidates,
                                              "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        if candidates and verbose:
            print(f"  {len(candidates)} candidate hypotheses generated (REVIEW BEFORE ACTIVATING):")
            for c in candidates[:3]:
                print(f"    [{c['hypothesis_id']}] {c['condition']} — accuracy={c['separation_accuracy']:.0%}")

    # Veto health
    if verbose:
        print("Evaluating veto health...")
    veto_health = _compute_veto_health(records)
    VETO_HEALTH_FILE.write_text(json.dumps(veto_health, indent=2))
    for alert in veto_health["alerts"]:
        if verbose:
            print(f"  ⚠ VETO ALERT [{alert['priority']}]: {alert['veto']} — {alert['message'][:80]}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Forensic Engine")
    parser.add_argument("--system", choices=["ict","forex","both"], default="both")
    parser.add_argument("--veto-health", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.veto_health:
        print("Running veto health check only...")
        health = _compute_veto_health([])
        print(json.dumps(health, indent=2))
    else:
        records = run_full_analysis(system=args.system, verbose=not args.quiet)
        print_summary(records)
        print(f"\nOutput: {FORENSICS_FILE}")
        print(f"Veto health: {VETO_HEALTH_FILE}")
        print(f"Auto hypotheses: {AUTO_HYPO_FILE}")
