"""HYP-091 pre-registration writer — Phase 0 of the TSMOM diversification study.

Writes the hash-locked prereg JSON to data/research/preregister/ and appends a
PREREGISTERED entry to the canonical hypothesis ledger, BEFORE any price data is
loaded. This is the spec-sanctioned Phase 0 write outside the module tree
(mirrors research/political_alpha/preregister_hyp085.py); everything else this
study writes stays under data/research/tsmom_hyp091/.

Hash + ledger mechanics COPIED from research/political_alpha/preregister_hyp085.py
(_canonical_hash :141, register :149) — NOT imported (isolation).

Relationship to HYP-089: a parallel session committed an HYP-089 TSMOM quick-look
(658733f, NOT_SIGNIFICANT) that (a) correlated against a FRED rate-differential
SIGN PROXY rather than the actual v015 carry return series, (b) modeled NO
financing (spread-only), (c) rebalanced DAILY, and (d) never registered in the
canonical ledger. HYP-091 is not a re-test of the same instrument: it pre-registers
the corrected measurement — monthly (Moskowitz) rebalance, correlation vs the ACTUAL
v015 returns, and correct rate-differential-derived financing — the terms that
determine whether TSMOM diversifies the REAL carry book. The HYP-089 look is
declared here as a NON-EVIDENTIARY prior; its thresholds do not inform this lock.

Usage:
    python3 research/tsmom_hyp091/preregister_tsmom.py           # register
    python3 research/tsmom_hyp091/preregister_tsmom.py --verify  # re-check lock
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = ROOT / "data" / "research" / "preregister" / "HYP-091_tsmom_carry_diversification.json"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"

HYP_ID = "HYP-091"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


DOC = {
    "id": HYP_ID,
    "slug": "tsmom_carry_diversification",
    "family": "NONE (standalone; single pre-stated primary + permutation gauntlet — no BH family of variants)",
    "name": "TSMOM diversification: does 12-month time-series momentum add a real diversifier to the v015 carry book?",
    "status": "PREREGISTERED",
    "frozen_at": None,  # set at register() time, then hashed
    "ticket": "TICK-027",
    "plan": "Plans/here-s-what-the-literature-sleepy-tarjan.md",
    "prior_materials_banner": (
        "PRIOR MATERIALS — NON-EVIDENTIARY: the 2026-07-12 literature summary (Moskowitz/Ooi/Pedersen "
        "2012; AQR Century of Evidence; Koijen carry; Hutchinson 2022 FX decay) informs thesis text "
        "and locked definitions ONLY. The parallel-session HYP-089 TSMOM quick-look (commit 658733f) "
        "is a declared PRIOR LOOK, not evidence: it used a carry-sign proxy, no financing, and daily "
        "rebalance; its numbers and its 0.30 Sharpe bar do NOT inform any threshold here. No price, "
        "financing, or v015 return data was read before this lock; all thresholds come from this spec."
    ),
    "relationship_to_prior": {
        "prior_id": "HYP-089 (parallel session, committed 658733f, NOT_SIGNIFICANT, NOT ledger-registered)",
        "why_not_a_re_test": (
            "HYP-089 measured a different, weaker instrument: carry proxied by the FRED rate-diff SIGN "
            "(not actual v015 returns), zero financing (spread-only), daily rebalance, 3x cap. HYP-091 "
            "corrects all three axes that determine the diversification answer. Fixing an invalid "
            "measurement is not parameter-fishing; the two are declared here for full transparency."
        ),
    },
    "thesis": (
        "H0: 12-month time-series momentum on the 4 v015 pairs adds no usable diversification to the "
        "live v015 carry book, in-sample 2016-2024 net of realistic costs — its standalone out-of-sample "
        "Sharpe is <= 0 OR its return stream is too correlated with v015 carry to diversify. "
        "H1: TSMOM has positive OOS Sharpe AND low correlation with v015 carry, i.e. a genuine diversifier."
    ),
    "universe": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
    "data_substrate": (
        "yfinance daily OHLCV (auto_adjust=True), SAME series as v015 via the "
        "sovereign/forex/forex_backtester.py::_download_price convention; warm-up pull from 2013-06-01 so "
        "the first 252-day signal is available at 2016-mid; v015 carry returns from the git-tracked "
        "data/proof/backtest_trades_v015_2015_2024.csv (field risk_adjusted_pnl_pct, aggregated by "
        "exit-month); FRED policy-rate differentials via sovereign/forex/data_fetcher.py::get_pair_differentials."
    ),
    "signal": {
        "rule": "sign(C_t / C_{t-252} - 1) — trailing 252-trading-day spot return, raw (not rf-subtracted)",
        "rebalance": "monthly (last trading day of each month); signal held constant through the following month",
        "warmup": "first usable signal ~2016-mid (252 prior trading days required); OOS thus ~24 months",
        "excess_variant": "rf-subtracted (cumulative DGS3MO) is a DESCRIPTIVE robustness check only, never adjudicated",
    },
    "sizing": {
        "ex_ante_vol": "sigma_daily = daily_log_return.ewm(com=60).std() at the rebalance close (info <= t; no look-ahead)",
        "annualize": "sigma_ann = sigma_daily * sqrt(252)",
        "weight": "w = sign * min(target_vol_ann / sigma_ann, LEV_CAP)",
        "target_vol_ann": 0.10,
        "LEV_CAP": 2.0,
        "portfolio": "equal-risk average across the 4 pairs: R_m = mean_i(r_i_net)",
        "cap_note": (
            "without the cap, Sharpe is invariant to target_vol (gross pnl, spread turnover, financing all "
            "scale linearly -> constant cancels); the cap is the only thing making target_vol load-bearing. "
            "A with/without-cap pair is reported as descriptive robustness."
        ),
    },
    "costs": {
        "spread": "|w_m - w_{m-1}| * (SPREAD_COST[pair] + 2*SLIPPAGE_PER_SIDE) at each monthly rebalance (forex_backtester.py:43-51)",
        "financing_PRIMARY": (
            "CORRECT rate-differential-derived model (operator decision 2026-07-12; NOT the Colin-gated "
            "SWAP_RATES_ANNUAL, which TICK-024 proves is ~10x too small + one sign flip). Anchored "
            "differential-tracking swap: financing_side(t) = oanda_side_now + s*(diff(t) - diff_now), "
            "s=+1 long / -1 short, diff(t) = FRED policy-rate differential (get_pair_differentials, base-quote), "
            "diff_now = diff at the calibration date, oanda_side_now = 2026 OANDA snapshot from "
            "data/research/swap_calibration.json (TICK-024). Reproduces the OANDA snapshot AND the trade-227 "
            "anchor at t=now; varies correctly across 2015-2024 (captures the 2022 USDJPY carry blowout). "
            "Accrued daily: swap_day = w * financing_side(t)/365 per calendar day held. Touches no gated table."
        ),
        "financing_robustness": (
            "DESCRIPTIVE legs, not adjudicated: (i) the broken SWAP_RATES_ANNUAL model — the apples-to-apples "
            "cross-check since v015's CSV was costed with it; (ii) price-only (no financing). The spread "
            "between leg (i) and the primary is reported so the financing-regime mismatch's effect on the "
            "correlation is visible."
        ),
        "net": "r_i_net = w * (C_end/C_start - 1) + sum_days(swap_day) - spread_cost",
    },
    "primary_test": {
        "correlation_frequency": "MONTHLY (locked)",
        "carry_series": (
            "ACTUAL v015 returns: sum of risk_adjusted_pnl_pct by EXIT-month from "
            "data/proof/backtest_trades_v015_2015_2024.csv (the field v015's own Sharpe uses) — NOT a "
            "rate-diff sign proxy. Correlate against the TSMOM monthly portfolio return R_m over the overlap."
        ),
        "correlation_caveat": (
            "the primary correlates correct-financing TSMOM against the v015 CSV, which was costed with the "
            "BROKEN swap model — a financing-regime mismatch. Robustness leg (i) (broken-model TSMOM) is the "
            "mismatch-free comparison and is reported alongside."
        ),
        "sharpe_method": "sovereign/reporting/equity_curve.py::_sharpe (v015-comparable; annualizes by sqrt(n/years))",
        "IS_window": "2016-mid..2022",
        "OOS_window": "2023-01-01..2024-12-31 (aligns with data/risk/oos_trades_2023_2024.json)",
    },
    "null_hypothesis": (
        "The diversification thesis FAILS if EITHER: (a) standalone OOS (2023-2024) Sharpe <= 0, OR "
        "(b) monthly return correlation with the ACTUAL v015 carry series > 0.5."
    ),
    "success_criteria": (
        "VALID_EDGE CANDIDATE (no live capital) requires ALL of: OOS Sharpe > 0 AND monthly corr with v015 <= 0.5 "
        "(null not triggered) AND directional-permutation p < 0.05 AND deflated-Sharpe prob > 0.95 AND "
        "bh_survives AND holdout (2023-2024) Sharpe > 0. Promotion to CONFIRMED / any live sizing is a SEPARATE "
        "human step (RISK_CONSTITUTION Art. 6, scripts/approve_edge.py) — explicitly OUT OF SCOPE here."
    ),
    "failure_criteria": (
        "null triggered (OOS Sharpe <= 0 OR corr > 0.5) OR any gauntlet leg fails -> NOT_SIGNIFICANT (the prior). "
        "Sealed on the pre-stated conjunction; no parameter search, no re-run with different knobs."
    ),
    "deliverables": (
        "per-calendar-year standalone Sharpe table (2016-2024, DESCRIPTIVE — regime-concentration read, expect "
        "2022 to dominate); IS/OOS Sharpe; monthly corr vs v015 (+ rolling-60d, 2022 window, CI); 50/50 combined "
        "portfolio Sharpe vs max(carry, tsmom) (CONFIRMATORY, not adjudicated); financing robustness legs; "
        "directional-permutation + DSR + BH + holdout verdict."
    ),
    "statistics": {
        "permutation": "directional monthly-sign timing permutation (preserve long/short ratio), N>=10000, seed 42, p=(n_ge+1)/(N+1)",
        "deflated_sharpe": "sovereign/discovery/gate.py::deflated_sharpe_ratio; n_trials=1 (family of one) => no deflation, BH == raw p<0.05; real guards are the permutation test + this hash-lock",
        "power_caveat": (
            "~104 monthly obs full / ~24 OOS / ~12 per year. Permutation significance is exact at small n but "
            "POWER IS LOW: a true Sharpe ~0.5 over 24 OOS months may not reach p<0.05. Per-year Sharpes are "
            "DESCRIPTIVE only (SE ~ +-0.5). NOT_SIGNIFICANT is about as consistent with low power as with no edge."
        ),
    },
    "prior_expectation": "NOT_SIGNIFICANT",
    "isolation": (
        "research/tsmom_hyp091/ writes only under data/research/tsmom_hyp091/ (+ this Phase-0 prereg/ledger "
        "write); reads sovereign read-only (Sharpe/DSR utils, price loader, differentials, v015 CSV); does NOT "
        "touch the parallel session's research/tsmom/, nor any execution-path / live-parameter file. AST-checked "
        "by test_isolation.py."
    ),
    "verdict": None,
    "hash_method": (
        "sha256 of json.dumps(doc, sort_keys=True, separators=(',',':')) where doc = this object MINUS the hash_lock field"
    ),
}


def _canonical_hash(doc: dict) -> str:
    # Copied from research/political_alpha/preregister_hyp085.py:141 (NOT imported).
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def register() -> None:
    if PREREG_PATH.exists():
        sys.exit(f"REFUSED: {PREREG_PATH} already exists — preregs are never overwritten.")

    ledger = json.loads(LEDGER_PATH.read_text())
    assert isinstance(ledger, list), "ledger must be a JSON array"
    if any(e.get("id") == HYP_ID for e in ledger):
        sys.exit(f"REFUSED: {HYP_ID} already present in the ledger.")

    doc = dict(DOC)
    doc["frozen_at"] = _now_iso()
    doc["hash_lock"] = _canonical_hash(doc)

    PREREG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREREG_PATH.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"prereg written: {PREREG_PATH}")
    print(f"frozen_at:      {doc['frozen_at']}")
    print(f"hash_lock:      {doc['hash_lock']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = LEDGER_PATH.with_suffix(f".bak-{stamp}.json")
    shutil.copy2(LEDGER_PATH, backup)
    print(f"ledger backup:  {backup}")

    ledger.append({
        "id": HYP_ID,
        "name": doc["name"],
        "status": "PREREGISTERED",
        "date_registered": doc["frozen_at"][:10],
        "family": "NONE",
        "hash_lock": doc["hash_lock"],
        "prereg_file": "data/research/preregister/HYP-091_tsmom_carry_diversification.json",
        "mechanism": (
            "time-series momentum (sign of trailing 12-month return, inverse-vol sized, monthly rebalance) "
            "as a potential diversifier of the v015 carry book; near-zero carry/momentum correlation is the "
            "documented property that would make combining them rational (Moskowitz 2012, Koijen 2018)"
        ),
        "methodology_note": (
            "isolated study research/tsmom_hyp091/; corrects the parallel HYP-089 quick-look (proxy corr + "
            "no financing + daily rebalance) with monthly rebalance, correlation vs ACTUAL v015 returns, and "
            "correct rate-differential financing; TICK-027"
        ),
        "relationship_to_prior": doc["relationship_to_prior"],
        "prior_expectation": "NOT_SIGNIFICANT",
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
    print(f"ledger entry appended: {HYP_ID} PREREGISTERED ({len(ledger)} entries total)")


def verify() -> None:
    doc = json.loads(PREREG_PATH.read_text())
    stored = doc.get("hash_lock")
    recomputed = _canonical_hash(doc)
    if stored != recomputed:
        sys.exit(f"HASH MISMATCH: stored {stored} != recomputed {recomputed}")
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((e for e in ledger if e.get("id") == HYP_ID), None)
    if entry is None:
        sys.exit(f"LEDGER: {HYP_ID} entry missing")
    if entry.get("status") != "PREREGISTERED":
        sys.exit(f"LEDGER: unexpected status {entry.get('status')!r}")
    if entry.get("hash_lock") != stored:
        sys.exit("LEDGER: hash_lock does not match prereg file")
    print(f"VERIFIED: {HYP_ID} hash-lock intact ({stored}) and ledger entry PREREGISTERED.")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else register()
