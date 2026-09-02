#!/usr/bin/env python3
"""HYP-110 — overnight partition on ten liquid ETFs. Pre-registration. Sealed BEFORE
any open/close partition is computed.

Session 2026-09-02 mapped where a durable edge can exist for a trader with no
speed, size, or information advantage (research/TAXONOMY_2026-09-02_where_edges_
can_exist.md). Bucket (i) information is exhausted for direction. Bucket (iii)
sizing died twice today: the HYP-109 abstention overlay is noise on the delta
(Sharpe −0.21, DSR 0.000, 5/10 instruments, ex-2020 negative) and the pre-declared
regime test came back STORY_ONLY with the point estimate running against
vol-targeting. Bucket (ii) structure holds the one standalone survivor in the
whole ledger that nobody followed up — OVERNIGHT-QQQ VALID_EDGE, 5.49 bp/day
overnight vs 0.09 intraday, rejected only as a carry diversifier. This registers
it on its own merits, golden-rule-clean: ten ETFs, no leverage, no options, no
speed — the action is to own the instrument from the close to the open and not
from the open to the close.

Discipline (mirrors preregister_hyp109_postshock.py):
- Every parameter is FROZEN here. None is swept. None is revisited after the
  result. Wanting a different cost, window, or instrument list later is a new
  hypothesis with a new id.
- Hash: sha256 of json.dumps(doc minus hash_lock, sort_keys=True,
  separators=(',',':')). The test script asserts it at gate zero and after.
- Lesson from HYP-109 applied: every significance statistic is on the DELTA
  (strategy minus incumbent) or on the jointly-resampled ΔSharpe — never on the
  strategy series alone. Golden rule and ex-2020 are pre-declared components of
  CONFIRMED, not post-hoc reads.
- n_trials: mined_n._total 1543 + HYP-109 + today's regime test = 1545.

Usage:
  .venv313/bin/python scripts/research/preregister_hyp110_overnight.py --write
  .venv313/bin/python scripts/research/preregister_hyp110_overnight.py --verify
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

HYP_ID = "HYP-110"
SLUG = "overnight_partition_etf10"
FROZEN_AT = "2026-09-02T00:00:00Z"
INSTRUMENTS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "XLF", "XLE"]
TRIALS_SPENT_THIS_SESSION = 2   # HYP-109 (adjudicated) + the pre-declared regime test


def _mined_total() -> int:
    n = json.loads(MINED_N.read_text())["_total"]
    if not isinstance(n, int) or n < 1543:
        raise SystemExit(f"mined_n._total looks wrong ({n!r}); refusing to sign")
    return n


def build_doc() -> dict:
    n_trials = _mined_total() + TRIALS_SPENT_THIS_SESSION
    return {
        "id": HYP_ID,
        "slug": SLUG,
        "name": "Overnight partition: own ten liquid ETFs close-to-open only, flat open-to-close",
        "status": "PREREGISTERED",
        "frozen_at": FROZEN_AT,
        "family": "STRUCTURE-OVERNIGHT-2026-09",
        "family_note": ("Single-member family. Lineage: OVERNIGHT-QQQ VALID_EDGE (2026-06-07), "
                        "rejected only as a carry diversifier, never tested standalone. "
                        "Candidates not registered today: SPY-only variant, earnings-catalyst "
                        "momentum, continuous vol-targeting (prior weakened by the regime test)."),
        "phase": "one_hypothesis_one_test",
        "taxonomy_bucket": "(ii) execution / structure — a timing partition, not a race",

        "thesis": (
            "In liquid ETFs the risk premium accrues overnight (close to next open); the "
            "intraday session (open to close) carries no positive drift. Owning the instruments "
            "only from close to open therefore keeps the return and sheds the variance of the "
            "intraday leg, improving risk-adjusted return relative to buy-and-hold, net of costs."
        ),
        "mechanism": (
            "Known market structure, not a story: information arrives while the market is closed "
            "(earnings after the close, macro before the open, overseas sessions) and is priced "
            "at the open; intraday flow is dominated by liquidity provision and rebalancing with "
            "no net directional compensation. The overnight premium is compensation for holding "
            "gap risk — it is real AND it is where the crashes live, which is why the fold, "
            "ex-2020 and per-instrument components are part of the verdict."
        ),
        "golden_rule": {
            "rule": ("must hold for any trader at any level: no dependence on account size, "
                     "leverage, execution speed, colocation, one instrument, one venue, one regime"),
            "how_satisfied": ("ten retail ETFs, fractional shares, market-on-close entry and "
                              "market-on-open exit (auction orders, available at every retail "
                              "broker, no speed required), 11.5 years spanning 2018 / 2020 / 2022; "
                              "per-instrument consistency is a pre-declared verdict component"),
        },

        "instrument_set": INSTRUMENTS,
        "data": {
            "source": "data/cache/daily_universe/<SYM>.parquet",
            "columns": ["date", "open", "high", "low", "close", "volume"],
            "window": ["2014-01-02", "2026-07-16"],
            "warmup_sessions": 252,
            "warmup_note": ("no indicator needs warm-up; 252 sessions are dropped ONLY so the "
                            "incumbent is numerically identical to HYP-109's "
                            "(+131.7%, Sharpe 0.496, maxDD −33.2%, +0.0290 %/day)"),
            "effective_window": ["2015-01-02", "2026-07-16"],
            "returns": {
                "overnight": "ln(open_t / close_{t-1})",
                "intraday": "ln(close_t / open_t)",
                "close_to_close": "overnight + intraday (the incumbent's return)",
            },
            "data_quality_abort": ("if any instrument has open == close on more than 1% of "
                                   "sessions in the effective window (stale-open artefact), "
                                   "verdict is INCONCLUSIVE — data, not hypothesis"),
        },

        "strategy": {
            "rule": "for every instrument, every session: buy at close_{t-1}, sell at open_t; flat open_t..close_t",
            "portfolio": "equal-weight across the ten, rebalanced daily (same as the incumbent)",
            "incumbent": "buy-and-hold, equal-weight, close-to-close — identical series to HYP-109",
            "delta_series": "overnight_net − incumbent  (= −intraday − cost, by construction)",
        },
        "costs": {
            "round_trip_bp_per_instrument_day": 1.0,
            "charged_on": "every session, every instrument (one round trip per instrument-day)",
            "note": ("SPY/QQQ quoted spread ≈0.2–0.5 bp, XLF/EEM ≈2 bp; opening/closing auction "
                     "fills pay no spread. 1.0 bp round trip is the frozen figure. A break-even "
                     "cost is REPORTED descriptively; it does not alter the verdict."),
        },

        "statistics": {
            "bootstrap": {"type": "stationary_block", "L": 5, "draws": 10000, "seed": 42,
                          "resampling": "joint — the same session indices are applied to every series"},
            "cpcv": {"n_groups": 6, "test_groups": 2, "n_splits": 15, "embargo_sessions": 1,
                     "impl": "sovereign/discovery/cpcv.py::combinatorial_purged_splits"},
            "dsr": {"impl": "sovereign/discovery/gate.py::deflated_sharpe_ratio",
                    "n_trials": n_trials,
                    "n_trials_note": f"mined_n._total {n_trials - TRIALS_SPENT_THIS_SESSION} + HYP-109 + regime test",
                    "applied_to": "Sharpe of the DELTA series, n_obs = its length"},
            "a_partition": {
                "stat": "mean(overnight EW) − mean(intraday EW), %/day, gross",
                "pass": "95% joint block-bootstrap CI excludes zero from above",
                "fail": "CI includes zero or lies below -> KILL_STRUCTURE",
            },
            "b_delta": {
                "stat_1": "dSharpe = Sharpe(overnight_net) − Sharpe(incumbent), annualised, jointly resampled",
                "pass_1": "95% CI excludes zero from above",
                "stat_2": "DSR prob of Sharpe(delta series) at n_trials above",
                "pass_2": ">= 0.95",
            },
            "c_folds": {
                "stat": "dSharpe on each of the 15 purged CPCV test folds",
                "pass": ">= 12 of 15 positive", "null": "<= 7 of 15", "inconclusive": "8..11",
            },
            "g_golden_rule": {
                "stat": "per-instrument dSharpe = Sharpe(overnight_net_i) − Sharpe(hold_i)",
                "pass": ">= 7 of 10 positive", "null": "<= 5 of 10", "inconclusive": "6 of 10",
            },
            "x_ex2020": {"stat": "dSharpe with calendar 2020 removed", "pass": "> 0"},
            "d_raw_return": {
                "stat_1": "delta %/day = overnight_net − incumbent",
                "stat_2": "overnight_net %/day vs constitutional floor 0.0005",
                "note": "the incumbent itself (+0.0290 %/day) is below the floor",
            },
            "descriptive_only": [
                "break-even round-trip cost (bp) at which dSharpe = 0",
                "per-instrument and per-year tables",
                "overnight vs intraday mean and Sharpe per instrument",
            ],
        },

        "verdict_ladder": {
            "CONFIRMED": "a AND b1 AND b2 AND c>=12 AND g>=7 AND x AND delta%/day >= 0 AND overnight_net >= floor",
            "VALID_BUT_BELOW_FLOOR": "a AND b1 AND b2 AND c>=12 AND g>=7 AND x, but delta%/day < 0 OR overnight_net < floor",
            "KILL_STRUCTURE": "a fails — the premium is not overnight; the mechanism is absent on this set",
            "NULL": "a passes AND (b1 fails OR c<=7 OR g<=5)",
            "INCONCLUSIVE": "anything else (c 8..11, g == 6, b2 fails alone, x fails alone, data abort, fold error) — not a pass, not a re-run trigger",
        },
        "success_criteria": "CONFIRMED per the ladder",
        "failure_criteria": "anything else; dead; not re-run",
        "abort": {
            "no_rerun": "one run, one verdict; no parameter is changed after the result is seen",
            "no_scan": "no search across costs, windows, instrument subsets or partitions for the one that passes",
        },
        "frozen_parameters": {
            "round_trip_bp": 1.0, "warmup_sessions": 252, "block_L": 5, "seed": 42,
            "draws": 10000, "cpcv": "6/2", "embargo_sessions": 1, "n_trials": n_trials,
            "floor_per_day": 0.0005,
        },
        "prior_expectation": "NOT_SIGNIFICANT",
        "most_likely_failure": (
            "b2: the effect is real and positive but not large enough to clear a 1545-trial DSR "
            "hurdle on a 2,900-day delta series; or c/g: the premium is concentrated in the "
            "equity-index ETFs and absent in TLT/GLD, so it fails the golden-rule component "
            "even if it passes in aggregate"
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
        "id": HYP_ID, "name": doc["name"], "status": "PREREGISTERED",
        "date_tested": None, "result": None, "verdict": None,
        "methodology_note": ("Pre-registered 2026-09-02 before any open/close partition was computed. "
                             "Bucket (ii) of the edge taxonomy; lineage OVERNIGHT-QQQ VALID_EDGE. "
                             "All significance on the delta / jointly-resampled dSharpe. "
                             f"n_trials {doc['frozen_parameters']['n_trials']}."),
        "hash_lock": doc["hash_lock"], "prereg_file": str(path.relative_to(ROOT)),
        "p_value": None, "bh_survives": None, "oos_sharpe": None, "is_sharpe": None,
        "prior_expectation": "NOT_SIGNIFICANT", "source": "operator_session_2026-09-02",
        "auto_generated": False,
    })
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    json.dump(ledger, tmp, indent=2); tmp.close()
    Path(tmp.name).replace(LEDGER)
    print(f"ledger: +1 PREREGISTERED (backup {backup.name})")
    return 0


def verify() -> int:
    path = PREREG_DIR / f"{HYP_ID}.json"
    doc = json.loads(path.read_text())
    good = doc.get("hash_lock") == _canonical_hash(doc)
    print(f"{'OK  ' if good else 'FAIL'} {path.name} {doc.get('hash_lock', '')[:16]}")
    entry = next((e for e in json.loads(LEDGER.read_text()) if e.get("id") == HYP_ID), None)
    match = entry is not None and entry.get("hash_lock") == doc.get("hash_lock")
    print("ledger hash_lock matches prereg file:", match)
    print("ledger status:", entry.get("status") if entry else None)
    if not (good and match):
        raise SystemExit("PREREGISTRATION VERIFY FAILED — do not proceed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.write:
        return write()
    if a.verify:
        return verify()
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
