"""
Latent Feature Search — Layer 3 of the Sovereign Intelligence Architecture.

Searches for the third feature that separates forex COMMITMENT_FAILURE trades
from winners. Momentum is not it (distributions overlap). Rate differential and
macro alignment are not it (both groups are highly aligned). Something else
is preventing the market from following through.

CANDIDATE FEATURES:
    1. Commitment score at entry  — from Layer 2 (market already moving?)
    2. Cross-pair divergence      — are correlated pairs contradicting the signal?
    3. Time of month              — first 5d / mid-month / last 5d structural flows
    4. VIX term structure slope   — flat curve = uncertainty = poor follow-through
    5. Short-term momentum (5d)   — has price already started moving this week?
    6. Catalyst proximity         — [SKIPPED: CB calendar only covers 2025+]

METHOD PER FEATURE:
    - IC (rank correlation vs actual R-multiple)
    - KS test (distribution separation, wins vs failures)
    - Threshold accuracy (best simple cutoff)
    - Expected Sharpe if used as gate

VERDICT:
    IC > 0.15         → STRONG, recommend wiring
    IC 0.08–0.15      → MODERATE, validate on live data first
    IC < 0.08         → WEAK, not actionable

Run:
    PYTHONPATH=/path/to/quant python3 sovereign/forensics/latent_feature_search.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
FORENSICS_FILE = ROOT / "data" / "forensics" / "trade_forensics.jsonl"
RESULTS_FILE   = ROOT / "data" / "forensics" / "latent_feature_results.json"

# ── Price and market data cache ───────────────────────────────────────────

_cache: Dict[str, pd.DataFrame] = {}

def _prices(ticker: str, start: str = "2014-01-01") -> Optional[pd.DataFrame]:
    if ticker in _cache:
        return _cache[ticker]
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end="2026-06-01", progress=False)
        if df is None or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        _cache[ticker] = df
        return df
    except Exception:
        return None


def _close_at(ticker: str, date_str: str, lookback: int = 10) -> Optional[np.ndarray]:
    df = _prices(ticker)
    if df is None:
        return None
    try:
        ts = pd.Timestamp(date_str).tz_localize(None)
        sl = df.loc[:ts].tail(lookback)
        if len(sl) < 3:
            return None
        col = "Close" if "Close" in sl.columns else sl.columns[0]
        return sl[col].values.astype(float)
    except Exception:
        return None


# ── Feature 1: Commitment score from Layer 2 ─────────────────────────────

def feat_commitment_score(pair: str, direction: int, date_str: str) -> Optional[float]:
    """Use Layer 2 commitment detector on the entry bar."""
    try:
        from sovereign.intelligence.commitment_detector import CommitmentDetector
        state = CommitmentDetector(log=False).compute(
            pair=pair, direction=direction, date_str=date_str, session="MACRO"
        )
        return state.score
    except Exception:
        return None


# ── Feature 2: Cross-pair divergence ─────────────────────────────────────

CORR_PAIRS = {
    "EURUSD=X": ["GBPUSD=X", "AUDNZD=X"],
    "GBPUSD=X": ["EURUSD=X", "GBPJPY=X"],
    "USDJPY=X": ["GBPJPY=X"],
    "AUDUSD=X": ["AUDNZD=X", "GBPUSD=X"],
    "GBPJPY=X": ["GBPUSD=X", "USDJPY=X"],
    "AUDNZD=X": ["AUDUSD=X"],
    "USDCAD=X": ["AUDUSD=X"],
}

def feat_cross_pair_divergence(pair: str, direction: int, date_str: str) -> Optional[float]:
    """
    Fraction of correlated pairs moving AGAINST the signal direction.
    High divergence = market hasn't decided → poor follow-through expected.
    Returns 0.0 (none diverge) to 1.0 (all diverge).
    """
    correlates = CORR_PAIRS.get(pair, [])
    if not correlates:
        return None

    against = 0
    checked = 0
    for cp in correlates:
        closes = _close_at(cp, date_str, lookback=4)
        if closes is None or len(closes) < 2:
            continue
        cp_dir = 1 if closes[-1] > closes[-2] else -1
        # Adjust for USD quote: EURUSD long vs USDJPY long move opposite
        # Simplification: check if correlated pair's recent move aligns
        if cp_dir != direction:
            against += 1
        checked += 1

    return (against / checked) if checked > 0 else None


# ── Feature 3: Time of month ─────────────────────────────────────────────

def feat_time_of_month(date_str: str) -> str:
    """
    Returns: EARLY (day 1-5), MID (day 6-20), LATE (day 21+)
    BMS signals fire on first business day — so EARLY captures month-start flows.
    """
    try:
        day = pd.Timestamp(date_str).day
        if day <= 5:
            return "EARLY"
        elif day <= 20:
            return "MID"
        else:
            return "LATE"
    except Exception:
        return "MID"


# ── Feature 4: VIX term structure slope ──────────────────────────────────

_vix_df: Optional[pd.DataFrame] = None
_vix3m_df: Optional[pd.DataFrame] = None

def _load_vix():
    global _vix_df, _vix3m_df
    if _vix_df is None:
        _vix_df  = _prices("^VIX")
    if _vix3m_df is None:
        _vix3m_df = _prices("^VIX3M")

def feat_vix_slope(date_str: str) -> Optional[float]:
    """
    VIX3M - VIX (term structure slope).
    Positive (normal): market expects MORE vol later → calm now, risk-on.
    Negative (inverted): immediate fear → poor follow-through for macro trades.
    Near-zero (flat): uncertainty, no clear regime.
    """
    _load_vix()
    if _vix_df is None or _vix3m_df is None:
        return None
    try:
        ts = pd.Timestamp(date_str).tz_localize(None)
        vix_close  = _vix_df.loc[:ts].tail(1)
        vix3m_close = _vix3m_df.loc[:ts].tail(1)
        if vix_close.empty or vix3m_close.empty:
            return None
        v  = float(vix_close["Close"].iloc[-1] if "Close" in vix_close.columns else vix_close.iloc[-1, 0])
        v3 = float(vix3m_close["Close"].iloc[-1] if "Close" in vix3m_close.columns else vix3m_close.iloc[-1, 0])
        return round(v3 - v, 3)
    except Exception:
        return None


# ── Feature 5: Short-term momentum (5-day) ───────────────────────────────

def feat_momentum_5d(pair: str, direction: int, date_str: str) -> Optional[float]:
    """
    5-day price momentum aligned with signal direction.
    Positive = price already moving our way this week (commitment present).
    Negative = price moving against us short-term (entry into counter-move).
    """
    closes = _close_at(pair, date_str, lookback=8)
    if closes is None or len(closes) < 6:
        return None
    mom = (closes[-1] - closes[-6]) / (closes[-6] + 1e-10)
    return round(mom * direction, 6)  # positive = aligned with direction


# ── Statistics ────────────────────────────────────────────────────────────

def compute_ic(values: List[float], outcomes: List[float]) -> Tuple[float, float]:
    """Spearman rank IC and p-value."""
    from scipy.stats import spearmanr
    if len(values) < 10:
        return 0.0, 1.0
    ic, pval = spearmanr(values, outcomes)
    return float(ic), float(pval)


def compute_ks(vals_a: List[float], vals_b: List[float]) -> Tuple[float, float]:
    from scipy.stats import ks_2samp
    if len(vals_a) < 5 or len(vals_b) < 5:
        return 0.0, 1.0
    stat, pval = ks_2samp(vals_a, vals_b)
    return float(stat), float(pval)


def best_threshold_accuracy(win_vals: List[float], fail_vals: List[float]) -> Tuple[float, float, str]:
    """Find threshold and direction with highest classification accuracy."""
    if not win_vals or not fail_vals:
        return 0.5, 0.0, "n/a"
    all_vals  = np.array(win_vals + fail_vals)
    all_labels = np.array([1.0] * len(win_vals) + [0.0] * len(fail_vals))

    best_acc, best_thresh, best_dir = 0.5, 0.0, "gt"
    for pct in range(10, 91, 10):
        thresh = float(np.percentile(all_vals, pct))
        for direction in ["gt", "lt"]:
            pred = (all_vals >= thresh if direction == "gt" else all_vals < thresh).astype(float)
            acc  = float(np.mean(pred == all_labels))
            if acc > best_acc:
                best_acc, best_thresh, best_dir = acc, thresh, direction
    return best_acc, best_thresh, best_dir


def sharpe_if_gated(
    records: List[Dict], feature_vals: List[float],
    threshold: float, direction: str
) -> Dict:
    """Simulate Sharpe if we veto trades where feature signal fires."""
    kept, vetoed = [], []
    for rec, val in zip(records, feature_vals):
        fires = val >= threshold if direction == "gt" else val < threshold
        if fires:
            vetoed.append(rec)
        else:
            kept.append(rec)

    if not kept:
        return {}

    kept_r  = np.array([r["pnl_r"] for r in kept])
    vetoed_r = np.array([r["pnl_r"] for r in vetoed]) if vetoed else np.array([0.0])
    base_r   = np.array([r["pnl_r"] for r in records])

    def sharpe(pnls):
        return float(np.mean(pnls) / (np.std(pnls) + 1e-9)) * np.sqrt(252 / 10)

    kept_wins = sum(1 for r in kept if r["win_driver"])
    return {
        "trades_kept":    len(kept),
        "trades_vetoed":  len(vetoed),
        "kept_win_rate":  round(kept_wins / len(kept), 4),
        "kept_avg_r":     round(float(np.mean(kept_r)), 4),
        "kept_sharpe":    round(sharpe(kept_r), 4),
        "base_sharpe":    round(sharpe(base_r), 4),
        "sharpe_delta":   round(sharpe(kept_r) - sharpe(base_r), 4),
    }


# ── Main search ───────────────────────────────────────────────────────────

def run_search(verbose: bool = True) -> Dict:
    if verbose:
        print("Loading forex forensic records...")

    all_records = [json.loads(l) for l in open(FORENSICS_FILE)]
    forex = [r for r in all_records if r["system"] == "FOREX"]
    commit_fail = [r for r in forex if r["failure_label"] == "COMMITMENT_FAILURE"]
    winners     = [r for r in forex if r["win_driver"] is not None]

    # Work with a combined set: all commitment failures + all winners
    study = commit_fail + winners

    if verbose:
        print(f"  Commitment failures: {len(commit_fail)}")
        print(f"  Winners: {len(winners)}")
        print(f"  Study set: {len(study)}")

    feature_defs = [
        ("commitment_score",      "Layer 2 commitment detector score",       True),
        ("cross_pair_divergence", "Fraction of correlated pairs diverging",   True),
        ("time_of_month_early",   "Entry in first 5 days of month (binary)",  False),
        ("vix_slope",             "VIX3M minus VIX (term structure slope)",   True),
        ("momentum_5d_aligned",   "5-day momentum aligned with direction",    True),
    ]

    all_feature_values: Dict[str, List] = {f[0]: [] for f in feature_defs}
    outcomes_r  = []
    outcomes_bin = []  # 1=win 0=commitment_failure
    valid_idx   = []

    if verbose:
        print(f"\nComputing features (fetching market data)...")

    for i, rec in enumerate(study):
        pair      = rec["pair"]
        direction = 1 if rec["direction"] == "LONG" else -1
        date_str  = rec["entry_date"]

        # Feature 1: commitment score
        cs = feat_commitment_score(pair, direction, date_str)

        # Feature 2: cross-pair divergence
        div = feat_cross_pair_divergence(pair, direction, date_str)

        # Feature 3: time of month
        tom = feat_time_of_month(date_str)
        early = 1.0 if tom == "EARLY" else 0.0

        # Feature 4: VIX slope
        vs = feat_vix_slope(date_str)

        # Feature 5: 5-day momentum
        m5 = feat_momentum_5d(pair, direction, date_str)

        # Only keep record if all features computed
        if any(v is None for v in [cs, div, vs, m5]):
            continue

        all_feature_values["commitment_score"].append(cs)
        all_feature_values["cross_pair_divergence"].append(div)
        all_feature_values["time_of_month_early"].append(early)
        all_feature_values["vix_slope"].append(vs)
        all_feature_values["momentum_5d_aligned"].append(m5)

        outcomes_r.append(float(rec["pnl_r"]))
        outcomes_bin.append(1.0 if rec["win_driver"] else 0.0)
        valid_idx.append(i)

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(study)} computed...")

    valid_records = [study[i] for i in valid_idx]
    commit_valid  = [r for r in valid_records if r["failure_label"] == "COMMITMENT_FAILURE"]
    win_valid     = [r for r in valid_records if r["win_driver"] is not None]

    if verbose:
        print(f"\n  Valid records: {len(valid_records)} "
              f"({len(win_valid)} wins / {len(commit_valid)} failures)")

    results = []

    if verbose:
        print(f"\n{'='*64}")
        print(f"LATENT FEATURE SEARCH RESULTS")
        print(f"{'='*64}")
        print(f"{'Feature':<28} {'IC':>6} {'p-val':>7} {'KS':>6} {'Acc':>6} {'ΔSharpe':>9}")
        print(f"{'-'*64}")

    for feat_name, description, _ in feature_defs:
        vals = all_feature_values[feat_name]
        if not vals:
            continue

        ic,   ic_p  = compute_ic(vals, outcomes_r)
        ks,   ks_p  = compute_ks(
            [v for v, r in zip(vals, valid_records) if r["win_driver"]],
            [v for v, r in zip(vals, valid_records) if r["failure_label"] == "COMMITMENT_FAILURE"]
        )

        win_vals  = [v for v, r in zip(vals, valid_records) if r["win_driver"]]
        fail_vals = [v for v, r in zip(vals, valid_records) if r["failure_label"] == "COMMITMENT_FAILURE"]

        acc, thresh, direction = best_threshold_accuracy(win_vals, fail_vals)
        gate_sim = sharpe_if_gated(valid_records, vals, thresh, direction)

        ic_label = ("★★ STRONG"   if abs(ic) > 0.15 else
                    "★ MODERATE"  if abs(ic) > 0.08 else
                    "  WEAK")

        if verbose:
            dsharpe = gate_sim.get("sharpe_delta", 0)
            print(f"{feat_name:<28} {ic:>+6.3f} {ic_p:>7.4f} {ks:>6.3f} {acc:>5.0%} {dsharpe:>+9.4f}  {ic_label}")

        results.append({
            "feature": feat_name,
            "description": description,
            "ic": round(ic, 4),
            "ic_pvalue": round(ic_p, 4),
            "ic_strength": ic_label.strip(),
            "ks_statistic": round(ks, 4),
            "ks_pvalue": round(ks_p, 4),
            "threshold_accuracy": round(acc, 4),
            "best_threshold": round(thresh, 6),
            "best_direction": direction,
            "gate_simulation": gate_sim,
            "win_avg": round(float(np.mean(win_vals)),  4) if win_vals  else None,
            "fail_avg": round(float(np.mean(fail_vals)), 4) if fail_vals else None,
            "separation": round(float(np.mean(win_vals)) - float(np.mean(fail_vals)), 4)
                          if win_vals and fail_vals else None,
        })

    results.sort(key=lambda x: -abs(x["ic"]))

    # Winner
    winner = results[0] if results else None
    verdict = ""
    if winner:
        ic = abs(winner["ic"])
        if ic > 0.15:
            verdict = f"WIRE IT IN — {winner['feature']} IC={winner['ic']:+.3f}"
        elif ic > 0.08:
            verdict = f"PROMISING — validate {winner['feature']} on 100+ live trades"
        else:
            verdict = "NO STRONG LATENT FEATURE FOUND — failures may be irreducible noise"

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "study_size": len(valid_records),
        "commitment_failures_analyzed": len(commit_valid),
        "winners_analyzed": len(win_valid),
        "results": results,
        "winner": winner,
        "verdict": verdict,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2))

    if verbose:
        print(f"\nVERDICT: {verdict}")
        if winner:
            print(f"\nTop feature: {winner['feature']}")
            print(f"  IC={winner['ic']:+.4f}  KS={winner['ks_statistic']:.4f}")
            print(f"  Win avg: {winner['win_avg']} | Failure avg: {winner['fail_avg']}")
            print(f"  Separation: {winner['separation']:+.4f}")
            gs = winner.get("gate_simulation", {})
            if gs:
                print(f"  Gate simulation: kept={gs['trades_kept']} vetoed={gs['trades_vetoed']}")
                print(f"    WR: {gs['kept_win_rate']*100:.0f}%  AvgR: {gs['kept_avg_r']:+.3f}  ΔSharpe: {gs['sharpe_delta']:+.4f}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Limit records for speed testing")
    args = parser.parse_args()
    result = run_search(verbose=True)
    print(f"\nFull results saved: {RESULTS_FILE}")
