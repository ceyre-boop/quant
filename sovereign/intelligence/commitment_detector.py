"""
Commitment Detector — Layer 2 of the Sovereign Intelligence Architecture.

Answers one question before every entry: Has the market actually started moving?

A signal says "this is the direction." The commitment detector says "yes but has
price committed to that move yet?" The forensic engine classified 80% of forex losses
and 7% of ICT losses as COMMITMENT_FAILURE. This pre-entry gate is designed to catch
those before they cost R.

COMMITMENT SCORE (0.0 to 1.0) from 5 components:
    1. Volume confirmation  (0–0.25): volume expanding in signal direction
    2. Multi-pair alignment (0–0.25): correlated pairs moving the same way
    3. ATR expansion        (0–0.20): volatility waking up (market starting to move)
    4. Session quality      (0–0.15): London=0.15, NY_AM=0.10, NY_PM=0.00
    5. Failed auction       (0–0.15): price tested against and was rejected

LABELS:
    COMMITTED   (>0.65): enter at full size
    DEVELOPING  (0.40–0.65): enter at half size, re-check next bar
    UNCOMMITTED (<0.40): skip this setup

VALIDATION:
    After computing scores on all 1,160 forensic records, we measure:
    - Information Coefficient (IC) vs actual outcome
    - Separation accuracy: COMMITTED vs COMMITMENT_FAILURE
    - Threshold calibration: what score cutoff best predicts failure?

Integration:
    sovereign/orchestrator.py (forex): gate before signal execution
    ict/pipeline.py (ICT): gate after grade assignment, before stage 6
    data/forensics/commitment_log.jsonl: every check logged

Run:
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/commitment_detector.py
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/commitment_detector.py --validate
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/commitment_detector.py --live GBPUSD LONG
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
COMMITMENT_LOG  = ROOT / "data" / "forensics" / "commitment_log.jsonl"
FORENSICS_FILE  = ROOT / "data" / "forensics" / "trade_forensics.jsonl"
VALIDATION_FILE = ROOT / "data" / "forensics" / "commitment_validation.json"

# Correlated pairs used for multi-pair confirmation
# Validated commitment thresholds per system (from forensic validation 2026-05-19)
# ICT: market_structure score >= 1.5 is the dominant predictor of commitment failure
#   London+A+mkt<1.5: Sharpe 3.314, WR 53%, avgR +1.372 (vs 1.864/38%/+0.745 unfiltered)
# FOREX: daily market data IC = -0.028 (too weak to gate on) — system gates already sufficient
ICT_MARKET_STRUCTURE_VETO = 1.5   # mkt_struct >= this → UNCOMMITTED for ICT
ICT_DISPLACEMENT_FLOOR    = 0.15  # displacement < this → DEVELOPING for ICT

PAIR_CORRELATES: Dict[str, List[str]] = {
    "GBPUSD":  ["EURUSD", "GBPJPY"],
    "GBPUSD=X": ["EURUSD=X", "GBPJPY=X"],
    "EURUSD":  ["GBPUSD", "AUDNZD"],
    "EURUSD=X": ["GBPUSD=X", "AUDNZD=X"],
    "USDJPY":  ["GBPJPY"],
    "USDJPY=X": ["GBPJPY=X"],
    "AUDUSD":  ["AUDNZD", "GBPUSD"],
    "AUDUSD=X": ["AUDNZD=X", "GBPUSD=X"],
    "GBPJPY":  ["GBPUSD", "USDJPY"],
    "GBPJPY=X": ["GBPUSD=X", "USDJPY=X"],
    "AUDNZD":  ["AUDUSD"],
    "AUDNZD=X": ["AUDUSD=X"],
    "USDCAD":  ["AUDUSD"],
    "USDCAD=X": ["AUDUSD=X"],
}


@dataclass
class CommitmentState:
    score: float
    label: str               # COMMITTED | DEVELOPING | UNCOMMITTED
    components: Dict[str, float]
    size_multiplier: float   # 1.0, 0.5, or 0.0
    reason: str
    indicator_note: str = ""  # INDICATOR_AGREE / INDICATOR_CONFLICT / "" if unavailable

    def to_dict(self) -> dict:
        return asdict(self)


# ── Price data cache ──────────────────────────────────────────────────────

_price_cache: Dict[str, pd.DataFrame] = {}

def _get_prices(pair: str, start: str = "2014-01-01") -> Optional[pd.DataFrame]:
    key = pair
    if key in _price_cache:
        return _price_cache[key]
    try:
        import yfinance as yf
        ticker = pair if "=X" in pair else f"{pair}=X"
        df = yf.download(ticker, start=start, end="2026-06-01", progress=False)
        if df is None or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        _price_cache[key] = df
        return df
    except Exception:
        return None


def _slice_at(df: pd.DataFrame, date_str: str, lookback: int = 25) -> Optional[pd.DataFrame]:
    """Return price slice ending at date, with lookback bars for context."""
    try:
        ts = pd.Timestamp(date_str).tz_localize(None)
        hist = df.loc[:ts].tail(lookback + 1)
        return hist if len(hist) >= 5 else None
    except Exception:
        return None


# ── Component scorers ─────────────────────────────────────────────────────

def _score_volume(df_slice: pd.DataFrame, direction: int) -> float:
    """
    0–0.25: Is volume expanding in signal direction?
    Compares most recent bar volume to 20-bar avg.
    Also checks that the bar's price movement aligns with direction.
    """
    if df_slice is None or "Volume" not in df_slice.columns:
        return 0.10   # neutral when no volume data

    vols = df_slice["Volume"].values
    if len(vols) < 5 or vols[-1] == 0:
        return 0.10

    avg_vol = float(np.mean(vols[-20:]))
    if avg_vol < 1:
        return 0.10

    vol_ratio = float(vols[-1]) / avg_vol

    # Check bar direction matches signal
    closes = df_slice["Close"].values if "Close" in df_slice.columns else df_slice.iloc[:, 3].values
    bar_direction = 1 if closes[-1] > closes[-2] else -1

    if bar_direction != direction:
        return 0.0   # volume on wrong-direction bar: no commitment

    # Score: 0.25 at 1.3× average, 0.25 max at 2×+
    raw = min((vol_ratio - 1.0) / 1.0, 1.0) if vol_ratio > 1.0 else 0.0
    return round(min(raw * 0.25, 0.25), 4)


def _score_multi_pair(pair: str, direction: int, date_str: str) -> float:
    """
    0–0.25: Are correlated pairs moving the same direction?
    Checks the most recent bar of each correlated pair.
    """
    correlates = PAIR_CORRELATES.get(pair, PAIR_CORRELATES.get(f"{pair}=X", []))
    if not correlates:
        return 0.125   # no correlates known: neutral

    aligned = 0
    checked = 0
    for corr_pair in correlates:
        df = _get_prices(corr_pair)
        if df is None:
            continue
        sl = _slice_at(df, date_str, lookback=3)
        if sl is None or len(sl) < 2:
            continue
        closes = sl["Close"].values if "Close" in sl.columns else sl.iloc[:, 3].values
        corr_dir = 1 if closes[-1] > closes[-2] else -1
        # For USD pairs: same direction means aligned carry
        # For crosses: depends on structure — simplified check
        if corr_dir == direction:
            aligned += 1
        checked += 1

    if checked == 0:
        return 0.125
    return round((aligned / checked) * 0.25, 4)


def _score_atr_expansion(df_slice: pd.DataFrame) -> float:
    """
    0–0.20: Is ATR expanding (market waking up)?
    Compare 5-bar ATR to 20-bar ATR.
    """
    if df_slice is None or len(df_slice) < 10:
        return 0.10   # neutral

    high  = df_slice["High"].values  if "High"  in df_slice.columns else df_slice.iloc[:, 1].values
    low   = df_slice["Low"].values   if "Low"   in df_slice.columns else df_slice.iloc[:, 2].values
    close = df_slice["Close"].values if "Close" in df_slice.columns else df_slice.iloc[:, 3].values

    # True range
    n = len(close)
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))

    if len(tr) < 10:
        return 0.10

    atr_5  = float(np.mean(tr[-5:]))
    atr_20 = float(np.mean(tr[-20:] if len(tr) >= 20 else tr))

    if atr_20 < 1e-8:
        return 0.10

    ratio = atr_5 / atr_20
    if ratio >= 1.15:
        return 0.20
    elif ratio >= 1.05:
        return 0.12
    elif ratio >= 0.95:
        return 0.08
    else:
        return 0.0   # contracting — bad sign


def _score_session(session: str) -> float:
    """
    0–0.15: Session quality.
    London = 0.15 (primary edge session per forensics)
    NY_AM  = 0.10 (reasonable participation)
    NY_PM  = 0.00 (confirmed anti-edge)
    Other  = 0.05
    """
    s = (session or "").upper()
    if "LONDON" in s:
        return 0.15
    if "NY_AM" in s or "NY_OPEN" in s or "NY_AM" in s:
        return 0.10
    if "NY_PM" in s or "NY PM" in s:
        return 0.00
    if "MACRO" in s or s == "":
        return 0.10   # macro signals: no session filter, moderate default
    return 0.05


def _score_failed_auction(df_slice: pd.DataFrame, direction: int) -> float:
    """
    0–0.15: Has price tested against the signal direction and been rejected?
    This is ICT's 'manipulation' phase completing — a genuine sweep and rejection.
    Detection: in last 3 bars, price moved against direction by >0.3×ATR
    then closed back in the signal direction.
    """
    if df_slice is None or len(df_slice) < 4:
        return 0.0

    high  = df_slice["High"].values[-4:]
    low   = df_slice["Low"].values[-4:]
    close = df_slice["Close"].values[-4:]

    if len(close) < 3:
        return 0.0

    # Compute ATR proxy
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    atr = float(np.mean(tr)) if len(tr) > 0 else 0.001

    # For LONG: look for price dip below recent low then close above midpoint
    # For SHORT: look for price spike above recent high then close below midpoint
    recent_close = close[-1]
    prev_close   = close[-2]

    if direction == 1:   # LONG
        wicked_low = np.min(low[-3:])
        swept = wicked_low < prev_close - 0.3 * atr
        rejected = recent_close > (wicked_low + 0.5 * (prev_close - wicked_low))
        return 0.15 if (swept and rejected) else 0.0
    else:                # SHORT
        wicked_high = np.max(high[-3:])
        swept = wicked_high > prev_close + 0.3 * atr
        rejected = recent_close < (wicked_high - 0.5 * (wicked_high - prev_close))
        return 0.15 if (swept and rejected) else 0.0


# ── Indicator consensus component ────────────────────────────────────────

def _score_indicator_consensus(
    pair: str,
    direction_int: int,
    df: pd.DataFrame,
    date_str: str,
) -> tuple[float, str]:
    """
    6th commitment component: do 30 independent indicators agree with our direction?
    Returns (score_delta, note):
      +0.00 to +0.20 if consensus agrees (scaled by conviction)
      -0.15          if consensus actively opposes direction
       0.00          if FLAT/NEUTRAL or any error
    """
    try:
        from sovereign.intelligence.indicator_consensus import score_indicator_consensus
        sl = _slice_at(df, date_str, lookback=90)
        if sl is None or len(sl) < 60:
            return 0.0, ""

        clean_pair = pair.replace("=X", "")
        consensus = score_indicator_consensus(clean_pair, sl)
        dir_label = "LONG" if direction_int == 1 else "SHORT"

        if consensus.direction in ("FLAT", "NEUTRAL"):
            return 0.0, f"indicator_consensus={consensus.direction} ({consensus.bullish_count}B/{consensus.bearish_count}b)"

        if consensus.direction == dir_label:
            boost = round(consensus.conviction * 0.20, 4)
            green_note = f" | {len(consensus.matching_green_long) + len(consensus.matching_green_short)} green combos" if consensus.matching_green_long or consensus.matching_green_short else ""
            return boost, (
                f"INDICATOR_AGREE: {consensus.bullish_count if direction_int==1 else consensus.bearish_count}/30 "
                f"agree {dir_label} | conv={consensus.conviction:.0%}{green_note} | boost=+{boost:.3f}"
            )
        else:
            opp_count = consensus.bearish_count if direction_int == 1 else consensus.bullish_count
            return -0.15, (
                f"INDICATOR_CONFLICT: system says {dir_label} but "
                f"{opp_count}/30 indicators say {consensus.direction} | penalty=-0.150"
            )
    except Exception:
        return 0.0, ""


# ── Main detector ─────────────────────────────────────────────────────────

class CommitmentDetector:
    """
    Pre-entry gate: score market commitment before executing any signal.

    Usage:
        detector = CommitmentDetector()
        state = detector.compute("GBPUSD=X", 1, "2026-05-19 09:00", session="London")
        if state.label == "UNCOMMITTED":
            skip_trade()
        elif state.label == "DEVELOPING":
            signal.size_multiplier = 0.50
    """

    COMMITTED_THRESHOLD   = 0.65
    DEVELOPING_THRESHOLD  = 0.40

    def __init__(self, log: bool = True):
        self._log = log

    def compute_ict(
        self,
        component_scores: Dict[str, float],
        session: str,
        grade: str,
        score: float,
    ) -> CommitmentState:
        """
        ICT-specific commitment scoring using component scores (pre-entry by definition).

        Validated against 310 ICT forensic records:
          market_structure >= 1.5 → 87.5% accuracy predicting commitment failure
          London + Grade A + mkt_struct < 1.5 → Sharpe 3.314 (vs 1.864 unfiltered)

        These are component scores already computed before the trade decision.
        No additional price data fetch needed.
        """
        mkt_struct   = float(component_scores.get("market_structure", 0))
        displacement = float(component_scores.get("displacement", 0))

        # Hard veto: high market structure = trading against committed opposing move
        if mkt_struct >= ICT_MARKET_STRUCTURE_VETO:
            return CommitmentState(
                score=0.15,
                label="UNCOMMITTED",
                components={"market_structure": mkt_struct, "displacement": displacement,
                            "session": _score_session(session), "grade": 1.0 if grade == "A" else 0.5},
                size_multiplier=0.0,
                reason=f"mkt_struct={mkt_struct:.2f} >= {ICT_MARKET_STRUCTURE_VETO} — strong opposing structure",
            )

        # Build score from components
        c_session    = _score_session(session)
        c_mkt_struct = 0.30 if mkt_struct < 1.0 else (0.20 if mkt_struct < 1.3 else 0.05)
        c_displace   = 0.25 if displacement > 1.8 else (0.15 if displacement > ICT_DISPLACEMENT_FLOOR else 0.0)
        c_grade      = 0.20 if grade == "A" else (0.10 if grade == "A+" else 0.0)
        c_score_band = 0.10 if 7.0 <= score <= 9.0 else 0.0   # sweet spot: not too low, not too high

        total = min(c_session + c_mkt_struct + c_displace + c_grade + c_score_band, 1.0)

        if total >= self.COMMITTED_THRESHOLD:
            label, size = "COMMITTED", 1.0
            reason = f"ICT score={total:.2f} — structure+displacement+session aligned"
        elif total >= self.DEVELOPING_THRESHOLD:
            label, size = "DEVELOPING", 0.75
            reason = f"ICT score={total:.2f} — partial alignment, 75% size"
        else:
            label, size = "UNCOMMITTED", 0.0
            reason = f"ICT score={total:.2f} — insufficient commitment"

        return CommitmentState(
            score=round(total, 4),
            label=label,
            components={
                "session": c_session, "market_structure": c_mkt_struct,
                "displacement": c_displace, "grade": c_grade, "score_band": c_score_band,
            },
            size_multiplier=size,
            reason=reason,
        )

    def compute(
        self,
        pair: str,
        direction: int,   # 1=LONG -1=SHORT
        date_str: str,
        session: str = "",
        lookback: int = 25,
    ) -> CommitmentState:
        df = _get_prices(pair)
        df_slice = _slice_at(df, date_str, lookback=lookback) if df is not None else None

        direction_int = 1 if str(direction).upper() in ("1", "LONG") else -1

        # Score each component
        c_volume     = _score_volume(df_slice, direction_int)
        c_multi_pair = _score_multi_pair(pair, direction_int, date_str)
        c_atr        = _score_atr_expansion(df_slice)
        c_session    = _score_session(session)
        c_auction    = _score_failed_auction(df_slice, direction_int)

        # 6th component: 30-indicator consensus (graceful — returns 0 if library not built)
        c_indicator, indicator_note = (
            _score_indicator_consensus(pair, direction_int, df, date_str)
            if df is not None else (0.0, "")
        )

        base_total = c_volume + c_multi_pair + c_atr + c_session + c_auction
        total = round(min(base_total + c_indicator, 1.0), 4)

        if total >= self.COMMITTED_THRESHOLD:
            label = "COMMITTED"
            size  = 1.0
            reason = f"score={total:.2f} — all components aligned"
        elif total >= self.DEVELOPING_THRESHOLD:
            label = "DEVELOPING"
            size  = 0.50
            reason = f"score={total:.2f} — partial commitment, half size"
        else:
            label = "UNCOMMITTED"
            size  = 0.0
            reason = f"score={total:.2f} — market not moving, skip"

        state = CommitmentState(
            score=total,
            label=label,
            components={
                "volume":         c_volume,
                "multi_pair":     c_multi_pair,
                "atr_expand":     c_atr,
                "session":        c_session,
                "failed_auct":    c_auction,
                "indicator_cons": c_indicator,
            },
            size_multiplier=size,
            reason=reason,
            indicator_note=indicator_note,
        )

        if self._log:
            _write_log(state, pair, direction_int, date_str, session)

        return state

    def compute_historical(
        self,
        pair: str,
        direction: int,
        date_str: str,
        session: str = "",
    ) -> CommitmentState:
        """Same as compute() but never writes to log (for bulk validation)."""
        original = self._log
        self._log = False
        state = self.compute(pair, direction, date_str, session)
        self._log = original
        return state


def _write_log(state: CommitmentState, pair: str, direction: int,
               date_str: str, session: str) -> None:
    COMMITMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "pair": pair, "direction": direction, "date": date_str, "session": session,
        **state.to_dict(),
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(COMMITMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Historical validation ─────────────────────────────────────────────────

def validate_against_forensics(max_records: int = 400) -> Dict:
    """
    Compute commitment scores on all forensic records and measure:
    - IC (information coefficient): correlation(score, outcome)
    - Separation: COMMITTED vs COMMITMENT_FAILURE
    - Optimal threshold calibration
    - Expected Sharpe impact of applying the gate
    """
    print("Loading forensic records...")
    records = []
    with open(FORENSICS_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  {len(records)} total records")
    if len(records) > max_records:
        # Sample evenly: all commitment failures + proportional wins
        failures = [r for r in records if r.get("failure_label") == "COMMITMENT_FAILURE"]
        winners  = [r for r in records if r.get("win_driver") is not None]
        others   = [r for r in records if r not in failures and r not in winners]
        import random
        random.seed(42)
        sample_wins  = random.sample(winners, min(len(winners), max_records // 2))
        sample_fail  = random.sample(failures, min(len(failures), max_records // 2))
        records = sample_wins + sample_fail
        print(f"  Sampled {len(records)} ({len(sample_wins)} wins + {len(sample_fail)} commitment failures)")

    detector = CommitmentDetector(log=False)

    scores  = []
    outcomes = []   # 1 = win, 0 = commitment failure
    labels  = []
    pnl_rs  = []
    failure_labels = []

    print("Computing commitment scores (this fetches price data — may take a moment)...")
    done = 0
    for rec in records:
        pair      = rec["pair"]
        direction = 1 if rec["direction"] == "LONG" else -1
        date_str  = rec["entry_date"]
        session   = rec.get("session", "")
        pnl_r     = float(rec.get("pnl_r", 0))
        failure   = rec.get("failure_label")
        win_drv   = rec.get("win_driver")

        state = detector.compute_historical(pair, direction, date_str, session)
        scores.append(state.score)
        outcomes.append(1 if win_drv is not None else 0)
        labels.append(state.label)
        pnl_rs.append(pnl_r)
        failure_labels.append(failure or "WIN")

        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(records)}...")

    scores_arr   = np.array(scores)
    outcomes_arr = np.array(outcomes)
    pnl_arr      = np.array(pnl_rs)

    # IC: rank correlation between score and outcome
    from scipy.stats import spearmanr, ks_2samp
    ic, ic_pval = spearmanr(scores_arr, outcomes_arr)

    # KS test: committed vs commitment-failure score distributions
    commit_scores = [s for s, f in zip(scores, failure_labels)
                     if f == "COMMITMENT_FAILURE"]
    win_scores    = [s for s, f in zip(scores, failure_labels) if f == "WIN"]

    if commit_scores and win_scores:
        ks_stat, ks_pval = ks_2samp(commit_scores, win_scores)
    else:
        ks_stat, ks_pval = 0.0, 1.0

    # Threshold sweep: what gate cutoff maximises Sharpe?
    best_sharpe = -999
    best_thresh = 0.40
    results_by_thresh = []

    for thresh in np.arange(0.20, 0.75, 0.05):
        kept_idx = [i for i, s in enumerate(scores) if s >= thresh]
        blocked  = len(scores) - len(kept_idx)
        if not kept_idx:
            continue

        kept_pnl = pnl_arr[kept_idx]
        kept_sharpe = (float(np.mean(kept_pnl)) /
                       max(float(np.std(kept_pnl)), 1e-8)) * np.sqrt(252 / 10)
        kept_wr  = float(np.mean([outcomes_arr[i] for i in kept_idx]))

        results_by_thresh.append({
            "threshold": round(thresh, 2),
            "trades_kept": len(kept_idx),
            "trades_blocked": blocked,
            "win_rate": round(kept_wr, 4),
            "avg_pnl_r": round(float(np.mean(kept_pnl)), 4),
            "sharpe_proxy": round(kept_sharpe, 4),
        })

        if kept_sharpe > best_sharpe and len(kept_idx) >= 20:
            best_sharpe = kept_sharpe
            best_thresh = thresh

    # Label distribution
    from collections import Counter
    label_counts    = Counter(labels)
    failure_by_label = Counter(f for f, l in zip(failure_labels, labels)
                                if f == "COMMITMENT_FAILURE")

    # Separation: of all UNCOMMITTED, what % are commitment failures?
    uncommitted_idx = [i for i, l in enumerate(labels) if l == "UNCOMMITTED"]
    if uncommitted_idx:
        uncommitted_failures = sum(1 for i in uncommitted_idx
                                   if failure_labels[i] == "COMMITMENT_FAILURE")
        uncommitted_precision = uncommitted_failures / len(uncommitted_idx)
    else:
        uncommitted_precision = 0.0

    result = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "records_scored": len(records),
        "ic": round(float(ic), 4),
        "ic_pvalue": round(float(ic_pval), 4),
        "ic_interpretation": (
            "STRONG (>0.15)" if abs(ic) > 0.15 else
            "MODERATE (0.05-0.15)" if abs(ic) > 0.05 else
            "WEAK (<0.05)"
        ),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pval), 4),
        "committed_avg_score":   round(float(np.mean(win_scores)), 3)    if win_scores    else 0,
        "commitment_fail_avg_score": round(float(np.mean(commit_scores)), 3) if commit_scores else 0,
        "score_separation": round(
            float(np.mean(win_scores)) - float(np.mean(commit_scores)), 3
        ) if win_scores and commit_scores else 0,
        "label_distribution": dict(label_counts),
        "uncommitted_precision": round(uncommitted_precision, 4),
        "best_threshold": round(best_thresh, 2),
        "best_sharpe_at_threshold": round(best_sharpe, 4),
        "threshold_sweep": results_by_thresh,
        "recommendation": (
            "WIRE IT IN — IC strong, use threshold {:.2f}".format(best_thresh)
            if abs(ic) > 0.10 and uncommitted_precision > 0.55 else
            "MONITOR — IC moderate, validate on 100 more live trades"
            if abs(ic) > 0.05 else
            "DO NOT WIRE — IC too weak (<0.05), needs more signal components"
        ),
    }

    VALIDATION_FILE.write_text(json.dumps(result, indent=2))
    return result


def print_validation_report(result: Dict) -> None:
    print(f"\n{'='*60}")
    print(f"COMMITMENT DETECTOR VALIDATION")
    print(f"{'='*60}")
    print(f"Records scored:     {result['records_scored']}")
    print(f"IC:                 {result['ic']:+.4f}  ({result['ic_interpretation']})")
    print(f"IC p-value:         {result['ic_pvalue']:.4f}")
    print(f"KS statistic:       {result['ks_statistic']:.4f}  p={result['ks_pvalue']:.4f}")
    print()
    print(f"Score distribution:")
    print(f"  Winners avg:             {result['committed_avg_score']:.3f}")
    print(f"  Commitment failures avg: {result['commitment_fail_avg_score']:.3f}")
    print(f"  Separation:              {result['score_separation']:+.3f}")
    print()
    print(f"Label distribution: {result['label_distribution']}")
    print(f"Uncommitted precision: {result['uncommitted_precision']*100:.0f}% are commitment failures")
    print()
    print(f"Best threshold:     {result['best_threshold']:.2f}")
    print(f"Best Sharpe proxy:  {result['best_sharpe_at_threshold']:.4f}")
    print()
    print(f"Recommendation: {result['recommendation']}")
    print()
    print(f"Threshold sweep (top 5 by Sharpe):")
    top5 = sorted(result['threshold_sweep'], key=lambda x: -x['sharpe_proxy'])[:5]
    print(f"  {'Thresh':>7} {'Kept':>6} {'Blocked':>8} {'WR':>7} {'AvgR':>8} {'Sharpe':>8}")
    for row in top5:
        print(f"  {row['threshold']:>7.2f} {row['trades_kept']:>6} {row['trades_blocked']:>8}"
              f" {row['win_rate']*100:>6.1f}% {row['avg_pnl_r']:>8.3f} {row['sharpe_proxy']:>8.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Run historical validation against forensic records")
    parser.add_argument("--live", nargs=3, metavar=("PAIR", "DIRECTION", "DATE"),
                        help="Score a single live check: --live GBPUSD=X LONG 2026-05-19")
    parser.add_argument("--session", default="", help="Session name for --live check")
    parser.add_argument("--max-records", type=int, default=400)
    args = parser.parse_args()

    if args.live:
        pair, direction, date = args.live
        dir_int = 1 if direction.upper() == "LONG" else -1
        detector = CommitmentDetector(log=True)
        state = detector.compute(pair, dir_int, date, session=args.session)
        print(f"\nCommitment check: {pair} {direction} @ {date}")
        print(f"  Score:      {state.score:.4f}")
        print(f"  Label:      {state.label}")
        print(f"  Size mult:  {state.size_multiplier}×")
        print(f"  Reason:     {state.reason}")
        print(f"  Components: {state.components}")
    elif args.validate:
        result = validate_against_forensics(max_records=args.max_records)
        print_validation_report(result)
        print(f"\nFull results saved: {VALIDATION_FILE}")
    else:
        # Default: run validation
        result = validate_against_forensics(max_records=args.max_records)
        print_validation_report(result)
        print(f"\nFull results saved: {VALIDATION_FILE}")
