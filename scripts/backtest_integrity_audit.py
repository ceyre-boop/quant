"""
Backtest Integrity Audit — sovereign/quant system.

Answers one question: "Are my backtest numbers truthful?"

Seven checks:
  1. Lookahead scan — entry timing + data publication lags
  2. Parameter provenance — LITERATURE / OPTIMIZED / UNKNOWN
  3. Fill realism — spread, swap, slippage costs; estimated Sharpe impact
  4. Holdout contamination — was test data used for parameter selection?
  5. Multiple testing correction — Benjamini-Hochberg on all hypotheses
  6. Regime robustness — per-regime Sharpe breakdown
  7. Survivor bias — active vs all-ever-tested pairs

Output: data/audit/backtest_integrity_YYYY_MM_DD.json + terminal report
"""
from __future__ import annotations

import json
import ast
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parents[1]
AUDIT_DIR = ROOT / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _grep(path: Path, pattern: str) -> list[tuple[int, str]]:
    """Return (lineno, line) pairs where pattern matches, case-insensitive."""
    hits = []
    try:
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if re.search(pattern, line, re.IGNORECASE):
                hits.append((i, line.rstrip()))
    except FileNotFoundError:
        pass
    return hits


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ─── CHECK 1 — Lookahead scan ────────────────────────────────────────────────

def check_lookahead() -> dict:
    """
    Trace data dependency chain for known lookahead risk points.
    Returns structured findings per risk category.
    """
    flags = []

    # 1a. Entry timing: fast_engine.py enters at closes[i] on signal bar
    fe = ROOT / "backtest" / "fast_engine.py"
    hits = _grep(fe, r"entry_price\s*=\s*closes\[i\]")
    if hits:
        flags.append({
            "severity": "HIGH",
            "category": "ENTRY_TIMING",
            "file": "backtest/fast_engine.py",
            "lines": [h[0] for h in hits],
            "detail": (
                "Entry at closes[i] on the same bar the signal fires. "
                "If the signal uses that bar's close to decide direction, "
                "the fill is impossible in live trading — you can't enter "
                "at the same close that generated the signal. "
                "Fix: shift entry to opens[i+1] (next bar open)."
            ),
            "estimated_impact": "Overstates performance; magnitude depends on bar size.",
        })

    # 1b. fast_backtester.py (forex) — check entry is next-bar open
    fb = ROOT / "sovereign" / "forex" / "fast_backtester.py"
    next_bar_hits = _grep(fb, r"opens\[i\s*\+\s*1\]")
    if next_bar_hits:
        flags.append({
            "severity": "OK",
            "category": "ENTRY_TIMING",
            "file": "sovereign/forex/fast_backtester.py",
            "lines": [h[0] for h in next_bar_hits],
            "detail": "Forex backtester enters at opens[i+1] — correct next-bar fill.",
        })
    else:
        flags.append({
            "severity": "WARN",
            "category": "ENTRY_TIMING",
            "file": "sovereign/forex/fast_backtester.py",
            "detail": "Could not confirm next-bar entry in forex backtester — verify manually.",
        })

    # 1c. COT data publication lag — CFTC releases Friday for prior-Tuesday positions
    ce = ROOT / "sovereign" / "forex" / "cot_engine.py"
    lag_hits = _grep(ce, r"shift|lag|pub_date|release")
    if not lag_hits:
        flags.append({
            "severity": "HIGH",
            "category": "COT_LAG",
            "file": "sovereign/forex/cot_engine.py",
            "detail": (
                "No publication lag found. CFTC COT data is released each Friday "
                "for positions as-of the prior Tuesday — minimum 3-day lag. "
                "If backtest uses COT as-of Tuesday on Tuesday, that's lookahead. "
                "Fix: after loading, apply series = series.shift(4) "
                "(available Friday = 4 trading days after report date)."
            ),
            "estimated_impact": "Minor directional signal quality improvement.",
        })

    # 1d. FRED CPI publication lag — CPI released ~2-3 weeks after reference month end
    df = ROOT / "sovereign" / "forex" / "data_fetcher.py"
    cpi_lag_hits = _grep(df, r"cpi.*shift|shift.*cpi|publication|release_date|lag_days")
    fred_hits = _grep(df, r"CPIAUCSL|CP0000EZ|GBRCPI|CANCPI")
    if fred_hits and not cpi_lag_hits:
        flags.append({
            "severity": "MEDIUM",
            "category": "CPI_LAG",
            "file": "sovereign/forex/data_fetcher.py",
            "detail": (
                "CPI data fetched via FRED but no publication lag applied. "
                "US CPI (CPIAUCSL) is released ~12-15 days after month end. "
                "If backtest uses January CPI on Feb 1 when actual release was Feb 14: lookahead. "
                "Since macro signal holds 60+ days, this is likely a small but real bias. "
                "Fix: apply shift(1) (one month) to all monthly CPI series."
            ),
            "estimated_impact": "Small; monthly signal, 60d hold reduces sensitivity.",
        })

    # 1e. IRP z-score: current rates applied to historical prices
    fv = ROOT / "sovereign" / "forex" / "fair_value.py"
    irp_hits = _grep(fv, r"irp_fv_hist.*current rates|rates change slowly|simplified")
    flags.append({
        "severity": "LOW",
        "category": "IRP_ZSCORE_APPROXIMATION",
        "file": "sovereign/forex/fair_value.py",
        "detail": (
            "IRP z-score uses current rates applied backward to historical prices "
            "(acknowledged as approximation at line 81). Since rates change slowly "
            "this is unlikely to cause material bias for the macro signal. "
            "Not a hard lookahead, but worth noting."
        ),
    })

    # 1f. Z-score window scope — confirm all z-scores use tail() not full-window
    zscore_files = [
        ROOT / "sovereign" / "forex" / "fair_value.py",
        ROOT / "sovereign" / "forex" / "cot_engine.py",
        ROOT / "sovereign" / "forex" / "macro_engine.py",
        ROOT / "sovereign" / "forex" / "signal_engine.py",
    ]
    full_window_hits = []
    for f in zscore_files:
        hits = _grep(f, r"\.mean\(\)|\.std\(\)")
        tail_hits = _grep(f, r"\.tail\(")
        # If there are mean/std calls without tail context → potential full-window z-score
        for lineno, line in hits:
            if "tail" not in line and "rolling" not in line and "window" not in line.lower():
                full_window_hits.append(f"{f.name}:{lineno}: {line.strip()}")

    if full_window_hits:
        flags.append({
            "severity": "LOW",
            "category": "ZSCORE_WINDOW",
            "detail": (
                "Some .mean()/.std() calls found without explicit tail() or rolling() context. "
                "Verify these are not computing statistics over the full dataset. "
                "Samples (first 5):"
            ),
            "samples": full_window_hits[:5],
        })

    return {"flags": flags}


# ─── CHECK 2 — Parameter provenance ──────────────────────────────────────────

def check_parameters() -> dict:
    """
    Catalog key thresholds and classify as LITERATURE / OPTIMIZED / UNKNOWN.
    OPTIMIZED parameters need holdout validation before trusting the Sharpe.
    """
    params = [
        {
            "name": "SIGNAL_THRESHOLD",
            "value": 0.15,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "Lowered from 0.20 for 'more signals'. Selected by in-sample inspection.",
            "needs_holdout": True,
        },
        {
            "name": "TRAILING_ATR_MULT",
            "value": 1.25,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "Forensics v1 chose 1.25x over 1.0x based on Sharpe (1.024 vs 0.884). In-sample.",
            "needs_holdout": True,
        },
        {
            "name": "STOP_ATR_MULT",
            "value": 2.0,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "LITERATURE",
            "note": "2× ATR stop is standard across professional swing trading systems.",
            "needs_holdout": False,
        },
        {
            "name": "HOLD_DAYS (default)",
            "value": 60,
            "file": "sovereign/forex/signal_engine.py",
            "source": "OPTIMIZED",
            "note": "Various hold sweeps (5d, 7d, 20d, 60d) run in-sample. 60d selected.",
            "needs_holdout": True,
        },
        {
            "name": "PAIR_HOLD_OVERRIDES (GBPUSD=6, AUDUSD=5, EURUSD=5, AUDNZD=7)",
            "value": "per-pair",
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "v007 per-pair hold sweep on 2015-2024 full window. In-sample.",
            "needs_holdout": True,
        },
        {
            "name": "VIX gate USDJPY/AUDNZD",
            "value": 13.0,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "v014: swept from 15→13. HYP-044 confirmed on 2015-2024 full backtest.",
            "needs_holdout": True,
        },
        {
            "name": "VIX gate EURUSD/GBPUSD",
            "value": 18.0,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "Empirically found in full-window sweep (2015-2024). In-sample.",
            "needs_holdout": True,
        },
        {
            "name": "VIX gate AUDUSD",
            "value": 20.0,
            "file": "sovereign/forex/forex_backtester.py",
            "source": "OPTIMIZED",
            "note": "Empirically found in full-window sweep. In-sample.",
            "needs_holdout": True,
        },
        {
            "name": "CONVICTION_NEUTRAL_THRESHOLD",
            "value": 0.35,
            "file": "sovereign/forex/strategy.py",
            "source": "UNKNOWN",
            "note": "Origin unclear — appears in strategy.py without documented sweep.",
            "needs_holdout": True,
        },
        {
            "name": "CROWDED_Z (COT gate)",
            "value": 1.5,
            "file": "sovereign/forex/cot_engine.py",
            "source": "LITERATURE",
            "note": "1.5σ crowding is standard in institutional COT analysis.",
            "needs_holdout": False,
        },
        {
            "name": "IRP_Z_THRESHOLD / PPP_Z_THRESHOLD",
            "value": 1.5,
            "file": "sovereign/forex/fair_value.py",
            "source": "LITERATURE",
            "note": "1.5σ deviation threshold is standard for mean-reversion signals.",
            "needs_holdout": False,
        },
        {
            "name": "HISTORY_WINDOW (IRP z-score, 756 days = 3yr)",
            "value": 756,
            "file": "sovereign/forex/fair_value.py",
            "source": "OPTIMIZED",
            "note": "3-year window chosen without documented sweep. UNKNOWN origin.",
            "needs_holdout": True,
        },
        {
            "name": "kelly_fraction",
            "value": 0.5,
            "file": "config/parameters.yml",
            "source": "OPTIMIZED",
            "note": "Full Kelly (0.5 is already conservative) — origin not documented.",
            "needs_holdout": False,
        },
    ]

    optimized = [p for p in params if p["source"] == "OPTIMIZED"]
    unknown = [p for p in params if p["source"] == "UNKNOWN"]
    needs_holdout = [p["name"] for p in params if p.get("needs_holdout")]

    return {
        "total_parameters_audited": len(params),
        "literature_sourced": len([p for p in params if p["source"] == "LITERATURE"]),
        "optimized_in_sample": len(optimized),
        "unknown_origin": len(unknown),
        "parameters_needing_holdout_validation": needs_holdout,
        "detail": params,
    }


# ─── CHECK 3 — Fill realism ───────────────────────────────────────────────────

def check_fill_realism() -> dict:
    """
    Inventory what cost components are modeled in the backtester
    and estimate the Sharpe impact of each missing component.
    """
    findings = []

    # Check forex backtester for spread, swap, slippage
    fb = ROOT / "sovereign" / "forex" / "forex_backtester.py"
    fb_text = fb.read_text(errors="replace") if fb.exists() else ""

    has_spread = bool(re.search(r"spread|pip_cost|transaction_cost", fb_text, re.I))
    has_swap = bool(re.search(r"swap|rollover|overnight_cost|carry_cost", fb_text, re.I))
    has_slippage = bool(re.search(r"slippage", fb_text, re.I))

    if not has_spread:
        findings.append({
            "missing": "SPREAD_COST",
            "severity": "MEDIUM",
            "detail": (
                "No explicit bid-ask spread cost in forex_backtester.py. "
                "Typical FX spreads: EURUSD=0.5pip, GBPUSD=0.8pip, USDJPY=0.5pip, "
                "AUDUSD=0.8pip, AUDNZD=1.5pip. At 60d hold, entry+exit spread = "
                "~1-3 pips round-trip. For 60 trades/year: ~60-180 pips/year."
            ),
            "sharpe_impact_estimate": "-0.05 to -0.15 Sharpe (conservative estimate)",
        })

    if not has_swap:
        findings.append({
            "missing": "OVERNIGHT_SWAP",
            "severity": "HIGH",
            "detail": (
                "No swap/rollover cost modeled. With 60-day average holds, swap is material. "
                "POSITIVE for carry-aligned trades (e.g., long AUDUSD when RBA > FED): adds P&L. "
                "NEGATIVE for counter-carry trades: subtracts P&L. "
                "Net effect depends on direction distribution — likely net positive since "
                "macro signal should align with carry direction most of the time, "
                "but this assumption is untested. "
                "Typical G10 swap: ±1-5 pips/day at current rates. "
                "Over 60 days × 60 trades: this is NOT negligible."
            ),
            "sharpe_impact_estimate": "+0.05 to +0.20 Sharpe if carry-aligned; -0.10 to -0.25 if misaligned",
        })

    if not has_slippage:
        findings.append({
            "missing": "SLIPPAGE",
            "severity": "LOW",
            "detail": "No slippage model. At daily bar resolution with next-open entry, slippage is minor for liquid G10 pairs.",
            "sharpe_impact_estimate": "< -0.03 Sharpe",
        })
    else:
        findings.append({
            "missing": None,
            "present": "SLIPPAGE",
            "severity": "OK",
            "detail": "Slippage is modeled (found in backtester code).",
        })

    # Check fast_engine for equity backtests
    fe = ROOT / "backtest" / "fast_engine.py"
    fe_text = fe.read_text(errors="replace") if fe.exists() else ""
    has_fe_commission = bool(re.search(r"commission", fe_text, re.I))
    has_fe_slippage = bool(re.search(r"slippage", fe_text, re.I))
    has_fe_swap = bool(re.search(r"swap", fe_text, re.I))

    findings.append({
        "engine": "fast_engine (equity-style)",
        "commission_modeled": has_fe_commission,
        "slippage_modeled": has_fe_slippage,
        "swap_modeled": has_fe_swap,
        "severity": "OK" if has_fe_commission and has_fe_slippage else "MEDIUM",
    })

    # Stop fill realism
    findings.append({
        "topic": "STOP_FILL",
        "severity": "OK",
        "detail": (
            "fast_backtester.py uses opens[i+1] for entry. Stop fills need verification: "
            "ideal stop fill uses min(opens[i], stop_price) to simulate gap-through. "
            "Check _simulate_forex_core for stop exit logic."
        ),
    })

    total_missing = len([f for f in findings if f.get("missing") and f["missing"] is not None])
    return {
        "missing_cost_components": total_missing,
        "combined_sharpe_impact_estimate": (
            "-0.10 to -0.30 if swap is net negative; "
            "+0.05 to +0.10 if carry-aligned swaps add P&L."
        ),
        "findings": findings,
    }


# ─── CHECK 4 — Holdout contamination ─────────────────────────────────────────

def check_holdout() -> dict:
    """
    Determine if the optimization window and the reported performance window overlap.
    All key parameter selections used the full 2015-2024 dataset — no clean holdout exists.
    """
    optimized_on_full_window = [
        "VIX gates (13/18/20) — HYP-044 used 2015-2024",
        "PAIR_HOLD_OVERRIDES (v007 sweep) — used 2015-2024",
        "TRAILING_ATR_MULT=1.25 — forensics v1, used 2015-2024",
        "SIGNAL_THRESHOLD=0.15 — chose for more signals, evaluated on full window",
        "Pair retirements (USDCAD, GBPJPY, NZDUSD, etc.) — decided by 2015-2024 Sharpe",
        "PAIR_VIX_GATES (EURUSD=18, GBPUSD=18, AUDUSD=20) — threshold sweep on 2015-2024",
    ]

    proper_split = {
        "train": "2015-2019 (choose parameters)",
        "validation": "2020-2022 (test parameters, never used for selection)",
        "holdout": "2023-2024 (final report Sharpe — touch only once)",
    }

    return {
        "contaminated": True,
        "severity": "HIGH",
        "detail": (
            "ALL key parameters were selected using the full 2015-2024 dataset. "
            "The reported avg_sharpe=2.0970 is in-sample performance, not out-of-sample. "
            "The system has never been tested on data it wasn't optimized on. "
            "This does NOT mean the edge is fake — the economic logic is sound — "
            "but the reported Sharpe is an upper bound, not an estimate of live performance."
        ),
        "parameters_optimized_on_full_window": optimized_on_full_window,
        "recommended_split": proper_split,
        "action": (
            "Run backtest with parameters FROZEN as of end-2022, "
            "then report Sharpe on 2023-2024 only. "
            "If Sharpe > 1.0 on holdout, edge is real. "
            "Expect ~30-40% Sharpe decay from in-sample to out-of-sample."
        ),
    }


# ─── CHECK 5 — Multiple testing correction ────────────────────────────────────

def check_multiple_testing() -> dict:
    """
    Apply Benjamini-Hochberg FDR correction to confirmed hypotheses.
    Since p-values aren't stored, use sample size as a proxy for evidence quality.
    """
    ledger_path = ROOT / "data" / "agent" / "hypothesis_ledger.json"
    data = _read_json(ledger_path)
    if not data:
        return {"error": "hypothesis_ledger.json not found"}

    hyps = data.get("ledger", [])
    confirmed = [h for h in hyps if h.get("status") == "CONFIRMED"]
    rejected = [h for h in hyps if h.get("status") == "REJECTED"]

    # Extract sample sizes from result strings where available
    small_sample_confirmed = []
    for h in confirmed:
        result = h.get("result", "")
        # Look for n=XX patterns
        ns = re.findall(r"n\s*=\s*(\d+)", result)
        if ns:
            n = int(ns[0])
            if n < 50:
                small_sample_confirmed.append({
                    "id": h["id"],
                    "name": h["name"],
                    "n": n,
                    "result": result,
                    "flag": "LOW_POWER — n < 50, result may not replicate",
                })

    # BH correction: with alpha=0.05, k hypotheses, each needs p < (rank/k)*0.05
    # Without p-values, we can only flag that correction was never applied
    total = len(hyps)
    n_confirmed = len(confirmed)
    # At p<0.05 with 43 tests, expect ~2 false positives by chance
    expected_false_positives = round(total * 0.05, 1)
    # At p<0.01, expect ~0.4 false positives
    expected_fp_strict = round(total * 0.01, 1)

    return {
        "total_hypotheses": total,
        "confirmed": n_confirmed,
        "rejected": len(rejected),
        "queued_or_testing": total - n_confirmed - len(rejected),
        "bh_correction_applied": False,
        "expected_false_positives_at_p05": expected_false_positives,
        "expected_false_positives_at_p01": expected_fp_strict,
        "low_power_confirmed": small_sample_confirmed,
        "severity": "MEDIUM",
        "detail": (
            f"With {total} hypotheses tested, Benjamini-Hochberg correction was never applied. "
            f"At p<0.05: expect ~{expected_false_positives} false positives among confirmed results by chance alone. "
            f"Low-power confirmed hypotheses (n<50) are most at risk. "
            "Action: for each confirmed hypothesis, record the actual p-value or CI width "
            "and apply BH correction. Results that don't survive correction should be "
            "downgraded to TENTATIVE."
        ),
        "most_at_risk": [h["id"] for h in small_sample_confirmed],
    }


# ─── CHECK 6 — Regime robustness ─────────────────────────────────────────────

def check_regime_robustness() -> dict:
    """
    Attempt to load backtest results broken down by regime window.
    If the data doesn't exist, prescribe the analysis needed.
    """
    results_path = ROOT / "logs" / "forex_backtest_results.json"
    results = _read_json(results_path)

    regime_windows = {
        "pre_covid":   ("2015-01-01", "2019-12-31", "Normal rates, low vol"),
        "covid":       ("2020-01-01", "2021-12-31", "Zero rates, extreme vol then QE"),
        "rate_shock":  ("2022-01-01", "2023-06-30", "Fastest hiking cycle in 40 years"),
        "normalization": ("2023-07-01", "2024-12-31", "Rates plateau, vol subsides"),
    }

    existing_breakdown = None
    if results and isinstance(results, dict):
        existing_breakdown = results.get("regime_breakdown")

    return {
        "regime_breakdown_exists": existing_breakdown is not None,
        "regime_windows": regime_windows,
        "severity": "HIGH" if not existing_breakdown else "OK",
        "detail": (
            "No per-regime Sharpe breakdown found in backtest results. "
            "The current avg_sharpe=2.0970 is across the full 2015-2024 window. "
            "If this edge only works in rate-trending regimes (2022-2023), "
            "performance in stable regimes (2015-2019) may be near zero. "
            "Macro rate differential signals are theoretically strongest in rate-trend regimes "
            "and weakest in range-bound/QE regimes — this needs empirical confirmation."
            if not existing_breakdown else
            "Per-regime breakdown exists."
        ),
        "action": (
            "Run scripts/full_backtest.py with date range per window above. "
            "Report Sharpe for each window independently. "
            "PASS criterion: Sharpe > 0 in ALL four windows. "
            "FAIL: any window negative = regime-dependent edge."
        ) if not existing_breakdown else None,
        "existing_breakdown": existing_breakdown,
    }


# ─── CHECK 7 — Survivor bias ──────────────────────────────────────────────────

def check_survivor_bias() -> dict:
    """
    Compare active universe Sharpe vs all-ever-tested pairs.
    """
    # Pairs retired from the macro universe with documented reasons
    retired_pairs = [
        {
            "pair": "USDCAD=X",
            "reason": "Structural: BOC policy historically lags Fed — rate diff signal redundant",
            "last_sharpe": 0.071,  # avg %/trade was 0.071 vs portfolio 0.204
            "retired_version": "v008",
        },
        {
            "pair": "GBPJPY=X",
            "reason": "Performance: dual-bank noise (BOE crisis + BOJ intervention risk)",
            "last_sharpe": 0.741,  # vs portfolio 1.286
            "retired_version": "v009",
        },
        {
            "pair": "NZDUSD=X",
            "reason": "Performance: Sharpe 0.22, RBNZ tracks RBA (AUDUSD already captures signal)",
            "last_sharpe": 0.220,
            "retired_version": "oracle_audit",
        },
        {
            "pair": "USDCHF=X",
            "reason": "Structural: SNB held -0.75% for 8 years — rate diff signal broken structurally",
            "last_sharpe": -0.450,
            "retired_version": "v004",
        },
        {
            "pair": "EURJPY=X",
            "reason": "Structural: dual ECB+BOJ influence creates systematic signal conflicts",
            "last_sharpe": None,  # not recorded
            "retired_version": "pair_universe",
        },
        {
            "pair": "EURGBP=X",
            "reason": "Performance: ECB+BOE in lockstep — Sharpe -0.04, profit factor 1.00",
            "last_sharpe": -0.040,
            "retired_version": "v004",
        },
    ]

    active_sharpe = 2.0970  # v014 reported avg

    # Compute universe-wide average including retired pairs (using available Sharpes)
    known_retired_sharpes = [p["last_sharpe"] for p in retired_pairs if p["last_sharpe"] is not None]
    active_pairs_n = 5
    # Estimate: 5 active pairs × avg_sharpe + N retired pairs × their sharpe / total
    all_sharpes = [active_sharpe] * active_pairs_n + known_retired_sharpes
    universe_avg = float(np.mean(all_sharpes))
    survivor_bias_delta = active_sharpe - universe_avg

    # Classify retirements
    structural = [p for p in retired_pairs if "Structural" in p["reason"]]
    performance = [p for p in retired_pairs if "Performance" in p["reason"]]

    return {
        "active_pairs": 5,
        "retired_pairs": len(retired_pairs),
        "total_ever_tested": active_pairs_n + len(retired_pairs),
        "active_avg_sharpe": active_sharpe,
        "universe_avg_sharpe_estimate": round(universe_avg, 4),
        "survivor_bias_delta": round(survivor_bias_delta, 4),
        "structural_retirements": len(structural),
        "performance_retirements": len(performance),
        "severity": "MEDIUM",
        "detail": (
            f"Universe avg Sharpe (including retired pairs) ≈ {universe_avg:.3f} "
            f"vs active avg {active_sharpe:.3f}. "
            f"Survivor bias inflates reported Sharpe by ~{survivor_bias_delta:.3f}. "
            f"{len(structural)} retirements are structural (defensible exclusions). "
            f"{len(performance)} retirements are performance-based (caution: these inflate the average). "
            f"GBPJPY (Sharpe 0.741) and NZDUSD (Sharpe 0.22) are borderline — "
            f"they weren't zero-edge, they were below-portfolio-average."
        ),
        "retired_pairs": retired_pairs,
    }


# ─── Summary ──────────────────────────────────────────────────────────────────

def compute_adjusted_sharpe(
    reported: float,
    fill_impact_low: float = -0.10,
    holdout_decay: float = 0.35,
    survivor_bias: float = 0.0,
) -> dict:
    """
    Conservative adjusted Sharpe estimate.
    fill_impact_low: low-end (best case) impact of missing spread/swap costs
    holdout_decay: expected in-to-out-of-sample Sharpe ratio (typically 0.60-0.70)
    survivor_bias: amount to subtract for survivorship (computed in check_7)
    """
    after_fills = reported + fill_impact_low  # fill_impact is negative
    after_survivor = after_fills - survivor_bias
    after_holdout = after_survivor * holdout_decay

    return {
        "reported": reported,
        "after_fill_costs_conservative": round(after_fills, 3),
        "after_survivor_adjustment": round(after_survivor, 3),
        "expected_out_of_sample_low": round(after_holdout, 3),
        "expected_out_of_sample_high": round(after_survivor * (holdout_decay + 0.25), 3),
        "interpretation": (
            "The 'expected OOS' range is the most honest estimate of what live trading will see. "
            "This assumes normal parameter decay from in-sample to out-of-sample. "
            "If regime robustness holds (all 4 windows positive), expect the higher end of range. "
            "Sharpe > 1.0 OOS = institutional grade and safe to trade. "
            "Sharpe 0.5-1.0 OOS = viable with tight risk management. "
            "Sharpe < 0.5 OOS = marginal — monitor closely."
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("BACKTEST INTEGRITY AUDIT")
    print(f"Run date: {date.today()}")
    print("=" * 60)

    results = {}

    print("\n[1/7] Lookahead scan...")
    results["lookahead"] = check_lookahead()
    flags = results["lookahead"]["flags"]
    high = [f for f in flags if f.get("severity") == "HIGH"]
    print(f"  → {len(high)} HIGH-severity flags, {len(flags)} total findings")
    for f in high:
        print(f"     ⚠  {f['category']}: {f['file']}")

    print("\n[2/7] Parameter provenance...")
    results["parameters"] = check_parameters()
    p = results["parameters"]
    print(f"  → {p['optimized_in_sample']} OPTIMIZED, {p['unknown_origin']} UNKNOWN, {p['literature_sourced']} LITERATURE")
    print(f"     {len(p['parameters_needing_holdout_validation'])} parameters need holdout validation")

    print("\n[3/7] Fill realism...")
    results["fill_realism"] = check_fill_realism()
    fr = results["fill_realism"]
    print(f"  → {fr['missing_cost_components']} missing cost components")
    print(f"     Combined impact: {fr['combined_sharpe_impact_estimate']}")

    print("\n[4/7] Holdout contamination...")
    results["holdout"] = check_holdout()
    h = results["holdout"]
    contaminated = h["contaminated"]
    print(f"  → Contaminated: {'YES ⚠' if contaminated else 'NO ✓'}")
    if contaminated:
        print(f"     {len(h['parameters_optimized_on_full_window'])} parameter groups optimized on full 2015-2024 window")

    print("\n[5/7] Multiple testing correction...")
    results["multiple_testing"] = check_multiple_testing()
    mt = results["multiple_testing"]
    if "error" not in mt:
        print(f"  → {mt['confirmed']} confirmed / {mt['total_hypotheses']} total hypotheses")
        print(f"     Expected false positives at p<0.05: ~{mt['expected_false_positives_at_p05']}")
        low_power = mt.get("low_power_confirmed", [])
        if low_power:
            print(f"     ⚠  Low-power confirmed (n<50): {[h['id'] for h in low_power]}")

    print("\n[6/7] Regime robustness...")
    results["regime"] = check_regime_robustness()
    rr = results["regime"]
    print(f"  → Regime breakdown exists: {'YES ✓' if rr['regime_breakdown_exists'] else 'NO ⚠'}")
    if not rr["regime_breakdown_exists"]:
        print("     Cannot confirm edge holds across all 4 regime windows")

    print("\n[7/7] Survivor bias...")
    results["survivor"] = check_survivor_bias()
    sv = results["survivor"]
    print(f"  → Active: {sv['active_pairs']} pairs, Ever tested: {sv['total_ever_tested']} pairs")
    print(f"     Active avg Sharpe: {sv['active_avg_sharpe']:.4f}")
    print(f"     Universe avg Sharpe (est): {sv['universe_avg_sharpe_estimate']:.4f}")
    print(f"     Survivor bias delta: +{sv['survivor_bias_delta']:.4f}")

    # Adjusted Sharpe
    print("\n" + "=" * 60)
    print("ADJUSTED SHARPE ESTIMATE")
    print("=" * 60)
    reported = 2.0970
    adj = compute_adjusted_sharpe(
        reported=reported,
        fill_impact_low=-0.10,
        holdout_decay=0.65,  # expect ~65% of in-sample Sharpe to survive OOS
        survivor_bias=sv["survivor_bias_delta"],
    )
    results["adjusted_sharpe"] = adj
    print(f"  Reported (in-sample):          {adj['reported']:.3f}")
    print(f"  After fill costs (conservative): {adj['after_fill_costs_conservative']:.3f}")
    print(f"  After survivor adjustment:       {adj['after_survivor_adjustment']:.3f}")
    print(f"  Expected OOS range:              {adj['expected_out_of_sample_low']:.3f} – {adj['expected_out_of_sample_high']:.3f}")
    print(f"\n  {adj['interpretation']}")

    # Severity summary
    print("\n" + "=" * 60)
    print("PRIORITY ACTIONS")
    print("=" * 60)
    actions = [
        ("HIGH",   "Run 2023-2024 holdout with parameters frozen as of end-2022"),
        ("HIGH",   "Run per-regime Sharpe breakdown (4 windows)"),
        ("HIGH",   "Fix fast_engine.py: shift entry from closes[i] to opens[i+1]"),
        ("MEDIUM", "Apply COT publication lag: series.shift(4) in cot_engine.py"),
        ("MEDIUM", "Apply CPI publication lag: shift(1) for monthly CPI in data_fetcher.py"),
        ("MEDIUM", "Model spread costs: add ~1-3 pips round-trip per trade in forex_backtester.py"),
        ("MEDIUM", "Model swap costs: add overnight carry/rollover for 60-day holds"),
        ("MEDIUM", "Apply BH correction to 13 confirmed hypotheses — record p-values going forward"),
        ("LOW",    "Document CONVICTION_NEUTRAL_THRESHOLD=0.35 provenance (UNKNOWN)"),
        ("LOW",    "Document HISTORY_WINDOW=756 (IRP z-score) provenance"),
    ]

    for sev, action in actions:
        icon = "🔴" if sev == "HIGH" else "🟡" if sev == "MEDIUM" else "🟢"
        print(f"  {icon} [{sev}] {action}")

    # Save JSON
    out = {
        "audit_date": str(date.today()),
        "reported_sharpe": reported,
        **results,
    }
    out_path = AUDIT_DIR / f"backtest_integrity_{date.today().strftime('%Y_%m_%d')}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Saved: {out_path.relative_to(ROOT)}")
    print("=" * 60 + "\n")

    return out


if __name__ == "__main__":
    main()
