"""HYP-090 pre-registration writer — Phase 0 of the MODERN adaptive study.

Writes the hash-locked prereg JSON and appends a PREREGISTERED ledger entry
BEFORE any data is read. Everything in DOC is LAW for the study: grid, arms,
definitional locks, verdict criteria. Post-lock discoveries become stamped
deviations, never silent edits.

Mechanics copied from research/political_alpha/preregister_hyp085.py (NOT
imported — module self-containment).

Usage:
    python3 research/modern/preregister_hyp090.py           # register
    python3 research/modern/preregister_hyp090.py --verify  # re-check lock
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.modern._lib import LEDGER_PATH, PREREG_PATH, canonical_hash

FROZEN_AT = "2026-07-11T00:00:00Z"

DOC = {
    "id": "HYP-090",
    "slug": "modern_adaptive_params",
    "family": ("MODERN-ADAPTIVE (self-contained; BH alpha=0.05 across the m=6 pre-registered "
               "runs {A1,A2} x windows {90,180,365} — declared here, applied at adjudication)"),
    "name": ("MODERN: daily adaptive parameter selection (trailing-window ranking + regime "
             "matching, full surface incl. pair selection) vs static v015"),
    "status": "PREREGISTERED",
    "frozen_at": FROZEN_AT,
    "prior_materials_banner": (
        "PRIOR MATERIALS — NON-EVIDENTIARY: the 2026-07-11 chat exploration and dispatch's "
        "'MODERN' description inform thesis text only. PRIOR FAMILY KILLS, disclosed and NOT "
        "double-counted in this study's n_trials: HYP-065 (Fed-cycle conditioning REJECTED), "
        "HYP-066 (regime-keyed exits NOT_SIGNIFICANT), HYP-067 (10,100-policy evolutionary "
        "search NOT_ROBUST, holdout -0.401), the 180-config exit sweep (0/180 FDR), the "
        "cross-pair regime router (NOT_SUPPORTED). Arithmetic prior: ~53 trades/yr portfolio "
        "gives ~13 trades per 90d window; argmax over thousands of variants on 13 trades "
        "manufactures ~3 sigma of selection bias daily. No study data was read before this lock."
    ),
    "governing_plan": "plans/TICK-023.md (approved 2026-07-11; master Plans/glistening-juggling-clover.md)",
    "thesis": (
        "H0: daily adaptive parameter selection over trailing windows (with or without regime "
        "matching) does NOT beat the static v015 configuration out-of-sample on 2015-2026 daily "
        "forex after selection-bias correction. H1: adaptivity adds risk-adjusted return that "
        "survives the selection-noise floor, deflation at search size, costs, and per-year "
        "robustness."
    ),
    "grid": {
        "signal_threshold": [0.10, 0.15, 0.20, 0.25],
        "hold_days": [30, 45, 60, 90],
        "stop_atr_mult": [1.5, 2.0, 2.5],
        "trailing_atr_mult": [1.0, 1.25, 1.5, 2.0],
        "vix_gate": ["on", "off"],
        "config_385": ("v015-exact incumbent: theta=0.15, hold=60, stop=2.0, PER-PAIR trailing "
                       "overrides (GBPUSD 2.0, AUDUSD 1.0, EURUSD 1.25, USDJPY default 1.25), "
                       "gate=on — included so the selector can hold the incumbent; excluding it "
                       "would bias the comparison against static"),
        "n_configs": 385,
        "pair_subsets": "all 15 non-empty subsets of {EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X}",
        "n_variants": 5775,
        "windows_calendar_days": [90, 180, 365],
        "n_selection_cells": 17325,
    },
    "arms": {
        "A0_static": ("v015-exact (config 385, all 4 pairs) daily M2M series over the replay "
                      "span. ABORT GATE, not a verdict: canonical trade-based reconcile "
                      "weighted_portfolio_sharpe(2015-2024) must be within 0.6886 +/- 0.01 or "
                      "the study HALTS (SystemExit) and escalates — the band is never re-tuned."),
        "A1_recent_winner": ("at each replay day t: argmax over the 5,775 variants of the "
                             "trailing-W annualized daily-M2M Sharpe using data <= t only; the "
                             "selected variant's day-(t+1) return is taken. Ties broken by "
                             "canonical config-manifest order (deterministic)."),
        "A2_regime_matched": ("regime vector at day s = trailing-W means of [vix_close, vix_z, "
                              "spy_bull, mean-4-pair rate_diff_mom, mean-4-pair ATR%-trailing-"
                              "252d-percentile], each standardized by EXPANDING trailing 252d "
                              "mean/std (min 60 obs; full-sample standardization is look-ahead "
                              "and forbidden). Map rows: (regime vector at s -> every variant's "
                              "trailing-W Sharpe at s), s <= t-1. Selection at t: k-NN (k=25, "
                              "Euclidean); score variants by mean Sharpe across the k matched "
                              "rows; argmax; apply t+1. Cold start (< k map rows): hold config "
                              "385 all-pairs."),
        "A3_placebo": ("identical machinery, uniform-random valid variant each day; 500 seeds "
                       "derived via crc32 mixing (cross-process stable); the selection-noise "
                       "floor. Envelope = 95th percentile of the 500 placebo replay Sharpes "
                       "per window."),
    },
    "definitional_locks": {
        "daily_m2m_return": ("per (config, pair, day s): direction x (close_s - close_{s-1}) / "
                             "entry_price inside a trade (entry day: close_s - open_entry), 0 "
                             "when flat; increments must sum EXACTLY (1e-12) to the kernel's "
                             "per-trade pnl_pct before costs. The kernel's final still-open "
                             "position is emitted as flagged open-tail rows (excluded from "
                             "trade-based stats, included in daily series)."),
        "causal_costs": ("spread+slippage fraction charged on ENTRY day (known at entry); swap "
                         "accrued DAILY at 1.4 x annual_rate/365 (exit-day allocation would leak "
                         "final hold length into trailing windows). Costed totals reconcile to "
                         "ForexBacktester._apply_costs within the documented swap-rounding "
                         "tolerance (hold%5 != 0 => <= 0.4 swap-day difference)."),
        "portfolio_return": ("variant (config c, subset S): equal-notional mean of the |S| pair "
                             "series; flat pairs contribute 0."),
        "window_score": ("annualized (sqrt 252) Sharpe of the variant's daily M2M series over "
                         "the trailing W CALENDAR days ending at t inclusive; degenerate window "
                         "(std 0 or < 2 obs) scores 0."),
        "sharpe_conventions": ("trade-based sqrt-n-weighted Sharpe is used ONLY for the A0 "
                               "reconcile gate; ALL arm comparisons use annualized daily-M2M "
                               "Sharpe on the common replay span."),
        "handover": ("zero-cost daily config handover is an IDEALIZATION and is stated as such; "
                     "a switching-cost variant charges a round-trip spread fraction per pair "
                     "whose implied position changes on a config switch, using "
                     "ForexBacktester SPREAD_COST/slippage constants. The costed variant is "
                     "verdict criterion 5, not decoration."),
        "spans": ("precompute 2015-01-01 -> 2026-06-30; replay 2016-07-01 -> 2026-06-30 (18mo "
                  "warmup covers the 365d window + 252d standardization); reconcile window "
                  "2015-01-01 -> 2024-12-31. All market inputs frozen to parquet with sha256 "
                  "manifest before any selection runs."),
    },
    "primary": {
        "statistic": ("annualized daily-M2M Sharpe difference, adaptive run R vs A0, on the "
                      "common replay span, per run R in {A1,A2} x {90,180,365}"),
        "null": ("stationary block bootstrap (Politis-Romano, mean block L=5 per the locked "
                 "block-length derivation), paired-day resampling preserving cross-series "
                 "correlation, 10,000 draws, seed 42"),
        "sidedness": "ONE-SIDED — H1 predicts Sharpe(R) > Sharpe(A0)",
        "n": 10000,
        "seed": 42,
        "p_formula": "(n_le + 1) / (N + 1) where n_le = #{bootstrap diffs <= 0}",
    },
    "selection_bias_correction": {
        "dsr_primary_n_trials": 5775,
        "dsr_note": ("primary deflated_sharpe_ratio at n_trials=5,775 — the full per-day "
                     "selection surface incl. pair subsets (the degenerate failure mode is the "
                     "machinery latching onto one lucky variant, a 5,775-way search)"),
        "dsr_disclosed_secondaries": [6, 17325],
        "placebo_floor": "A1/A2 must exceed the 95th-pctile A3 envelope for their window",
    },
    "verdict_criteria": {
        "CONFIRMED": ("exists run R in {A1,A2} x {90,180,365} passing ALL: (1) block-bootstrap "
                      "one-sided p < 0.05 vs A0 AND survives Benjamini-Hochberg across the 6 "
                      "run p-values (alpha 0.05); (2) Sharpe(R) > 95th percentile of the "
                      "500-seed A3 envelope for its window; (3) deflated Sharpe > 0 at "
                      "n_trials=5,775; (4) per-year non-degrade: every full calendar year "
                      "2017-2025, Sharpe_R(y) >= Sharpe_A0(y) - 0.05; (5) the switching-costed "
                      "R still beats A0 (point estimate); and the reconcile band held."),
        "NOT_ROBUST": ("some run passes (1) but fails (2), (3) or (5) — the in-sample-inflation "
                       "signature — or passes (1) while failing (4) (regime-concentrated)"),
        "NOT_SIGNIFICANT": "no run passes (1)",
        "ABORT": ("gate-zero or reconcile failure -> NO verdict, halt, escalate to operator; "
                  "the band and the locks are never re-tuned after data"),
    },
    "validation_protocol": {
        "multiple_testing": {"family": "BH alpha=0.05, m=6 declared runs", "note": "declared here at lock time"},
        "no_model_training": ("no fitted models; k-NN matching is instance lookup, not "
                              "training; RISK_CONSTITUTION Art. 6 untouched"),
        "data_substrate": ("yfinance daily OHLCV (auto_adjust) for 4 pairs + SPY + ^VIX and "
                           "FRED rate differentials, ALL frozen to "
                           "data/research/modern/spot_cache/*.parquet with sha256 manifest at "
                           "P1; the study runs only from frozen inputs (yfinance/FRED drift is "
                           "a documented repo issue)"),
        "look_ahead": ("truncation-invariance test: selections at sampled t identical when all "
                       "data > t is deleted and features rebuilt; expanding standardization "
                       "only; trailing percentiles only; selection at t applies from t+1"),
        "isolation": ("research/modern/ AST whitelist: sovereign.forex.{fast_backtester, "
                      "forex_backtester, exit_machine, data_fetcher, signal_engine}, "
                      "sovereign.discovery, sovereign.reporting; forbidden roots ict/ict_engine/"
                      "config/audit/scripts/layer*/execution/entry_engine/orchestrator; no "
                      "OANDA/launchd; no config/parameters.yml writes (this study must never "
                      "resemble monthly_reopt.py's live re-optimizer); writes only under "
                      "data/research/modern/ + this prereg + ledger entries"),
    },
    "prior_expectation": "NOT_ROBUST",
    "verdict": None,
    "universe": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
    "hash_method": ("sha256 of json.dumps(doc, sort_keys=True, separators=(',',':')) where doc "
                    "= this object MINUS the hash_lock field"),
}


def register() -> None:
    if PREREG_PATH.exists():
        sys.exit(f"REFUSED: {PREREG_PATH} already exists — preregs are never overwritten.")

    doc = dict(DOC)
    doc["hash_lock"] = canonical_hash(doc)
    PREREG_PATH.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"prereg written: {PREREG_PATH}")
    print(f"hash_lock:      {doc['hash_lock']}")

    ledger = json.loads(LEDGER_PATH.read_text())
    assert isinstance(ledger, list), "ledger must be a JSON array"
    if any(e.get("id") == "HYP-090" for e in ledger):
        sys.exit("REFUSED: HYP-090 already present in the ledger.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = LEDGER_PATH.with_suffix(f".bak-{stamp}.json")
    shutil.copy2(LEDGER_PATH, backup)
    print(f"ledger backup:  {backup}")

    ledger.append({
        "id": "HYP-090",
        "name": DOC["name"],
        "status": "PREREGISTERED",
        "date_registered": FROZEN_AT[:10],
        "family": "MODERN-ADAPTIVE (BH m=6 declared runs)",
        "hash_lock": doc["hash_lock"],
        "prereg_file": "data/research/preregister/HYP-090_modern_adaptive_params.json",
        "mechanism": ("if parameter-level regime structure exists, recently-best / "
                      "regime-matched parameters should outperform the static incumbent "
                      "out-of-sample; prior kills (HYP-065/066/067, exit sweep, regime router) "
                      "say it does not at daily resolution"),
        "methodology_note": ("self-contained walk-forward replay in research/modern/ per "
                             "plans/TICK-023.md; precompute-then-replay over 5,775 variants; "
                             "placebo selection floor + DSR at search size; TICK-023"),
        "prior_expectation": "NOT_ROBUST",
        "result": None,
        "p_value": None,
        "bh_survives": None,
        "oos_sharpe": None,
        "is_sharpe": None,
        "standalone": True,
        "auto_generated": False,
        "source": "manual",
    })

    with tempfile.NamedTemporaryFile(
        "w", dir=LEDGER_PATH.parent, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(json.dumps(ledger, indent=2) + "\n")
    Path(tmp.name).replace(LEDGER_PATH)
    print(f"ledger entry appended: HYP-090 PREREGISTERED ({len(ledger)} entries total)")


def verify() -> None:
    doc = json.loads(PREREG_PATH.read_text())
    stored = doc.get("hash_lock")
    recomputed = canonical_hash(doc)
    if stored != recomputed:
        sys.exit(f"HASH MISMATCH: stored {stored} != recomputed {recomputed}")
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((e for e in ledger if e.get("id") == "HYP-090"), None)
    if entry is None:
        sys.exit("LEDGER: HYP-090 entry missing")
    if entry.get("status") != "PREREGISTERED":
        sys.exit(f"LEDGER: unexpected status {entry.get('status')!r}")
    if entry.get("hash_lock") != stored:
        sys.exit("LEDGER: hash_lock does not match prereg file")
    print(f"VERIFIED: HYP-090 hash-lock intact ({stored}) and ledger entry PREREGISTERED.")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else register()
