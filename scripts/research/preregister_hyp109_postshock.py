#!/usr/bin/env python3
"""HYP-109 — post-shock abstention. Pre-registration. Sealed BEFORE any data is read.

The surviving finding this tests: magnitude is conditionable, direction is not.
Two pre-registered short-horizon results survived (HYP-093 gap-size fade, HYP-095
vol-regime reversion); both are magnitude effects with no directional content;
seven G10 forex directional hypotheses, the NQ discovery track, and 5,040 ORB
configs are all null on sign. This registers the one form of that finding a $2k
retail trader can act on with no leverage, no options, no speed, and no single
instrument or regime: go flat for five sessions after a top-decile |return| day.

The golden-rule filter: a candidate qualifies only if it holds for any trader at
any level. If it breaks when you change the trader, it is fitted, not real.

Discipline (mirrors scripts/research/preregister_positioning.py):
- Every parameter below is FROZEN here. None is swept. None is revisited after
  the result. Wanting k=3 later is a new hypothesis with a new id.
- Hash method: sha256 of json.dumps(doc minus hash_lock, sort_keys=True,
  separators=(',',':')). The test script asserts this hash at gate zero before
  computing a single number, and again after.
- prior_expectation is NOT_SIGNIFICANT. The most likely failure is test (b):
  the short-term-reversal literature predicts positive drift after large down
  days, which would make this a directional effect in disguise. That outcome is
  KILL_DIRECTIONAL and it is the most informative one available, because it
  would sharpen the surviving finding itself.
- n_trials is declared against the HONEST mined count: data/research/
  yield_frontier/mined_n.json _total = 1543, plus this test = 1544. The
  HYP-093-era figure of 809 must not be reused; you cannot spend it twice.

Usage:
  .venv313/bin/python scripts/research/preregister_hyp109_postshock.py --write
  .venv313/bin/python scripts/research/preregister_hyp109_postshock.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG_DIR = ROOT / "data" / "research" / "preregister"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
MINED_N = ROOT / "data" / "research" / "yield_frontier" / "mined_n.json"

HYP_ID = "HYP-109"
SLUG = "postshock_abstention"
FROZEN_AT = "2026-09-02T00:00:00Z"

INSTRUMENTS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]
COMPANION = "UVXY"


def _mined_total() -> int:
    """Honest multiplicity count. Fail loud rather than default to a stale number."""
    n = json.loads(MINED_N.read_text())["_total"]
    if not isinstance(n, int) or n < 1543:
        raise SystemExit(f"mined_n._total looks wrong ({n!r}); refusing to sign")
    return n


def build_doc() -> dict:
    n_trials = _mined_total() + 1
    return {
        "id": HYP_ID,
        "slug": SLUG,
        "name": "Post-shock abstention: flat for five sessions after a top-decile |return| day",
        "status": "PREREGISTERED",
        "frozen_at": FROZEN_AT,
        "family": "MAGNITUDE-NOT-DIRECTION-2026-09",
        "family_note": ("Single-member family. The three candidates proposed in this session "
                        "(vol-managed sizing, post-shock abstention, ATR-scaled exits) were "
                        "proposed, not tested; only this one is registered and run."),
        "phase": "one_hypothesis_one_test",

        "thesis": (
            "After a top-decile absolute-return day in a liquid ETF, the following five "
            "sessions carry elevated realized volatility and zero directional drift. A long "
            "holder who goes flat for those five sessions therefore improves risk-adjusted "
            "return without sacrificing raw return."
        ),
        "mechanism": (
            "Volatility clusters; sign does not. A shock day forecasts the SIZE of the next "
            "week's moves but carries no information about their direction. For a directionless "
            "long holder the days after a shock are pure variance with no compensating drift — "
            "abstaining removes variance and, if the drift is truly zero, costs no return."
        ),
        "surviving_finding_tested": (
            "magnitude conditionable, direction null — confirmed across seven G10 forex "
            "directional nulls, the NQ discovery track (0 VALID_EDGE, cluster0_long was beta), "
            "MEGASCAN v2 (5,040 ORB configs, 0 FWER survivors), and on the exit side "
            "(HYP-059/060 CONFIRMED: time exits beat trailing stops). The two survivors, "
            "HYP-093 and HYP-095, are both magnitude-conditioned and both fail the golden rule "
            "as implementations (HTB microcap borrow; single instrument). This is the "
            "golden-rule-clean form."
        ),
        "golden_rule": {
            "rule": ("must hold for any trader at any level: no dependence on account size, "
                     "leverage, execution speed, colocation, one instrument, one venue, one "
                     "regime, or anything a $2k retail trader lacks"),
            "how_satisfied": ("ten liquid retail ETFs, daily bars, no leverage, no options, "
                              "no intraday timing, 12.5 years spanning 2018 / 2020 / 2022; "
                              "the action is to hold cash"),
        },

        "instrument_set": INSTRUMENTS,
        "companion_instrument": {
            "symbol": COMPANION,
            "role": "DESCRIPTIVE ONLY — a number and a 95% CI, no p-value, NO verdict weight",
            "question": ("after a SPY shock, does UVXY's next-5-session return exceed its "
                         "unconditional (roll-decayed) baseline? i.e. does an instrument that "
                         "pays for magnitude without direction actually pay after the signal?"),
            "precedent": "HYP-085 Test 1 — descriptive by design; a lead, not a result",
        },

        "data": {
            "source": "data/cache/daily_universe/<SYM>.parquet",
            "columns": ["date", "open", "high", "low", "close", "volume"],
            "window": ["2014-01-02", "2026-07-16"],
            "warmup_sessions": 252,
            "effective_window": ["2015-01-02", "2026-07-16"],
            "return_definition": "close-to-close log return",
        },

        "event_definition": {
            "shock": "abs(r_t) >= p90 of the trailing 252-session distribution of abs(r)",
            "percentile_window": "sessions t-252 .. t-1 ONLY — t itself excluded, no look-ahead",
            "per_instrument": True,
            "percentile": 0.90,
        },
        "action": {
            "rule": "flat for sessions t+1 .. t+k inclusive",
            "k": 5,
            "overlap": "union — an overlapping shock extends the flat window",
            "reentry": "open of the first session after the flat window",
            "baseline": "buy-and-hold, equal-weight across the ten instruments",
        },
        "costs": {
            "round_trip_bp": 2.0,
            "charged_on": "every flat episode (exit + re-entry)",
            "note": "conservative; SPY-class quoted spreads are ~0.5bp",
        },

        "statistics": {
            "bootstrap": {"type": "stationary_block", "L": 5, "draws": 10000, "seed": 42,
                          "note": "mirrors the HYP-093 prereg"},
            "cpcv": {"n_groups": 6, "test_groups": 2, "n_splits": 15, "embargo_sessions": 5,
                     "impl": "sovereign/discovery/cpcv.py::combinatorial_purged_splits"},
            "dsr": {"impl": "sovereign/discovery/gate.py::deflated_sharpe_ratio",
                    "n_trials": n_trials,
                    "n_trials_note": f"mined_n._total {n_trials - 1} + this test"},
            "a_magnitude": {
                "stat": "median RV(t+1..t+5 | shock) / median RV(t+1..t+5 | non-shock), pooled",
                "rv": "std of daily r over the five sessions",
                "pass": "ratio > 1.0 AND one-sided bootstrap p < 0.05",
            },
            "b_direction_null": {
                "stat": "mean cumr(t+1..t+5 | shock) - mean cumr(t+1..t+5 | non-shock), pooled",
                "holds": "95% block-bootstrap CI contains zero",
                "fails": "95% CI excludes zero -> KILL_DIRECTIONAL",
            },
            "c_tradeability": {
                "stat": "dSharpe = Sharpe(abstain, net of costs) - Sharpe(buy-and-hold), annualised",
                "per_fold": "computed on each of the 15 purged CPCV test folds",
                "full_sample": "DSR prob at n_trials above",
                "pass": "dSharpe > 0 in >= 12 of 15 folds AND full-sample DSR prob >= 0.95",
                "null": "dSharpe > 0 in <= 7 of 15 folds",
                "inconclusive": "8..11 of 15 folds",
            },
            "d_raw_return": {
                "stat": "delta pct/day = abstain - buy-and-hold",
                "floor": 0.0005,
                "floor_note": "constitutional 0.05%/day",
            },
        },

        "verdict_ladder": {
            "CONFIRMED": "a passes AND b holds AND c passes AND d >= 0",
            "VALID_BUT_BELOW_FLOOR": "a passes AND b holds AND c passes AND d < 0 (Sharpe bought by giving up return)",
            "KILL_DIRECTIONAL": "b fails — drift exists after shocks; directional in disguise; record the sign as a lead for a SEPARATE prereg; NULL for this candidate",
            "NULL": "a fails, OR c null",
            "INCONCLUSIVE": "c inconclusive, OR any instrument with < 30 shock events, OR a fold error — not a pass, not a re-run trigger",
        },
        "success_criteria": "CONFIRMED per the ladder",
        "failure_criteria": "anything else; the hypothesis is then dead and is not re-run",
        "abort": {
            "min_shock_events_per_instrument": 30,
            "no_rerun": "one run, one verdict; no parameter is changed after the result is seen",
            "no_scan": "no search across parameterisations for the one that passes",
        },
        "frozen_parameters": {
            "percentile": 0.90, "k": 5, "trailing_sessions": 252, "round_trip_bp": 2.0,
            "block_L": 5, "seed": 42, "draws": 10000, "cpcv": "6/2", "n_trials": n_trials,
        },
        "prior_expectation": "NOT_SIGNIFICANT",
        "most_likely_failure": (
            "b fails: short-term reversal after large down days is a well-documented effect, "
            "in which case post-shock is directional in disguise -> KILL_DIRECTIONAL"
        ),
        "verdict": None,
        "hash_method": ("sha256(json.dumps(doc, sort_keys=True, separators=(',',':')).encode()) "
                        "where doc = this object MINUS the hash_lock field"),
    }


def _canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write() -> int:
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    path = PREREG_DIR / f"{HYP_ID}.json"
    if path.exists():
        print(f"refusing to overwrite existing prereg {path}")
        return 1
    ledger = json.loads(LEDGER.read_text())
    if any(e.get("id") == HYP_ID for e in ledger):
        print(f"refusing: {HYP_ID} already in the ledger")
        return 1

    doc = build_doc()
    doc["hash_lock"] = _canonical_hash(doc)
    path.write_text(json.dumps(doc, indent=2))
    print(f"signed {path.name}  {doc['hash_lock'][:16]}")

    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    ledger.append({
        "id": HYP_ID,
        "name": doc["name"],
        "status": "PREREGISTERED",
        "date_tested": None,
        "result": None,
        "verdict": None,
        "methodology_note": ("Pre-registered 2026-09-02 before any data read. Golden-rule-clean "
                             "form of 'magnitude conditionable, direction null'. One run, one "
                             "verdict. n_trials declared at the honest mined count "
                             f"({doc['frozen_parameters']['n_trials']})."),
        "hash_lock": doc["hash_lock"],
        "prereg_file": str(path.relative_to(ROOT)),
        "p_value": None,
        "bh_survives": None,
        "oos_sharpe": None,
        "is_sharpe": None,
        "prior_expectation": "NOT_SIGNIFICANT",
        "source": "operator_session_2026-09-02",
        "auto_generated": False,
    })
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    json.dump(ledger, tmp, indent=2)
    tmp.close()
    Path(tmp.name).replace(LEDGER)
    print(f"ledger: +1 PREREGISTERED (backup {backup.name})")
    return 0


def verify() -> int:
    path = PREREG_DIR / f"{HYP_ID}.json"
    doc = json.loads(path.read_text())
    good = doc.get("hash_lock") == _canonical_hash(doc)
    print(f"{'OK  ' if good else 'FAIL'} {path.name} {doc.get('hash_lock', '')[:16]}")
    ledger = json.loads(LEDGER.read_text())
    entry = next((e for e in ledger if e.get("id") == HYP_ID), None)
    match = entry is not None and entry.get("hash_lock") == doc.get("hash_lock")
    print("ledger hash_lock matches prereg file:", match)
    print("ledger status:", entry.get("status") if entry else None)
    if not (good and match):
        raise SystemExit("PREREGISTRATION VERIFY FAILED — do not proceed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return write()
    if a.verify:
        return verify()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
