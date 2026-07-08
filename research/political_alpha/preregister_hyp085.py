"""HYP-085 pre-registration writer — Phase 0 of the political-alpha study.

Writes the hash-locked prereg JSON to data/research/preregister/ and appends a
PREREGISTERED entry to the canonical hypothesis ledger, BEFORE any event data
is collected. This is the spec-sanctioned Phase 0 write outside the module
tree (Political-Alpha-Claude-Code-Spec.md §7 Phase 0); everything else this
study writes stays inside research/political_alpha/.

Hash + ledger mechanics COPIED from scripts/research/preregister_positioning.py
(_canonical_hash :221, append_ledger :264) — NOT imported (isolation, NN#1).

Usage:
    python3 research/political_alpha/preregister_hyp085.py           # register
    python3 research/political_alpha/preregister_hyp085.py --verify  # re-check lock
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
PREREG_PATH = ROOT / "data" / "research" / "preregister" / "HYP-085_political_alpha_trump_events.json"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"

FROZEN_AT = "2026-07-08T20:42:00Z"

DOC = {
    "id": "HYP-085",
    "slug": "political_alpha_trump_events",
    "family": "NONE (standalone; single pre-stated bootstrap test — no BH family)",
    "name": "Political-alpha: Trump statements produce abnormal moves in tagged instruments (event study)",
    "status": "PREREGISTERED",
    "frozen_at": FROZEN_AT,
    "prior_materials_banner": (
        "PRIOR MATERIALS — NON-EVIDENTIARY: the 2026-07-08 chat exploration and the vault docs "
        "Political-Alpha-Hypothesis-2026-07.md / Political-Alpha-Claude-Code-Spec.md inform thesis "
        "text and locked definitions ONLY. No event or positioning data was read before this lock; "
        "all thresholds come from the locked spec, never from results."
    ),
    "governing_spec": (
        "~/Obsidian/Obsidian/Trading/Research/Political-Alpha-Claude-Code-Spec.md "
        "(GREENLIGHT, locked 2026-07-08; §1–§2 are the locked hypothesis and definitions)"
    ),
    "thesis": (
        "H0: Trump's public language produces no statistically detectable abnormal market moves — "
        "sector/currency moves around his statements are indistinguishable from random baseline at "
        "p < 0.05. H1: there exists a detectable, non-random pattern: specific language signals "
        "abnormal moves in targeted forex pairs and sectors, consistent with advance positioning."
    ),
    "observation_definition": (
        "One observation per (qualifying statement × tagged instrument). Qualifying statement = "
        "primary-source Trump statement (Truth Social @realDonaldTrump, White House releases, "
        "C-SPAN-timestamped pressers, Federal Register EOs), 2025-01-20 → present, explicitly "
        "naming a sector/currency/country AND announcing a policy action with direct price "
        "implications; vague sentiment excluded. Deterministic regex classifier committed with "
        "Phase 1; minimum separation 5 trading days per instrument (keep first, log drops)."
    ),
    "primary": {
        "statistic": (
            "big-move exceedance RATE across evaluable event rows, where big_move = "
            "|r_T0| > 2·sigma60(T0) OR |r_T1| > 2·sigma60(T1); daily log returns; sigma60 = "
            "trailing 60-day rolling SD (ddof=1) shifted one day (never includes the tested day); "
            "T0 mapping: fx instruments = first bar date >= statement UTC date; us_etf = same ET "
            "date if statement lands before 16:00 ET on a session day, else next session"
        ),
        "null": (
            "10,000 statement-level placebo sets: one random eligible timestamp (pinned 12:00 ET) "
            "per real evaluable statement, applied to all that statement's instrument rows via the "
            "identical mapping. Eligible dates: inside 2025-01-20 → run date; >=60 prior returns "
            "and valid P0/T0/T1 on every tagged instrument; not within ±5 trading days of any real "
            "event T0 on those instruments; T0/T1 not in the pre-listed 2025-26 FOMC/CPI/NFP "
            "calendar (committed as module data with Phase 1)"
        ),
        "sidedness": "ONE-SIDED — H1 predicts MORE exceedances than the placebo null",
        "n": 10000,
        "seed": 42,
        "p_formula": "(n_ge + 1) / (N + 1)",
    },
    "secondaries": (
        "Descriptive only, never adjudicated: (1) normality of pooled standardized pre-window "
        "returns (QQ plot + Shapiro–Wilk) and direction-aligned skew; (2) manipulation-signal "
        "count = post big_move AND pre-announcement positioning moved directionally, where "
        "pre_rr25_move = FXE rr25(D0-1) − rr25(D0-3) (EOD, strictly pre-statement, never widened) "
        "and the volume leg is the put/call volume ratio shift on the same ~30d expiry; condition "
        "is rr25 OR volume directional; FXE is the positioning proxy for all forex rows, native "
        "chains probed per ETF and recorded unavailable rather than synthesized; (3) optional "
        "Quiver STOCK-Act cross-check (30–45d disclosure lag) if the free endpoint returns rows."
    ),
    "success_criteria": (
        "Bootstrap p < 0.05 on the primary exceedance statistic → H0 rejected → CANDIDATE RESULT "
        "ONLY. Promotion to an edge requires the standard discovery gauntlet (permutation, "
        "deflated Sharpe, BH, CPCV) as a separate, later, ledgered step; nothing here touches "
        "live parameters or training (RISK_CONSTITUTION Art. 6)."
    ),
    "failure_criteria": (
        "p >= 0.05 → NOT_SIGNIFICANT (the prior). Fewer than 30 qualifying catalog events → "
        "shortfall stated everywhere, definitions never loosened; fewer than 30 evaluable rows → "
        "UNDERPOWERED, never a directional claim."
    ),
    "validation_protocol": {
        "multiple_testing": {
            "family": "NONE",
            "note": (
                "single pre-stated primary test; the richer gauntlet (BH, permutation, CAR) is "
                "deliberately excluded from this study per spec §10 to keep the pre-registration "
                "un-p-hackable"
            ),
        },
        "no_model_training": "event study only; no fitted parameters beyond the pre-registered thresholds",
        "data_substrate": (
            "yfinance daily OHLCV (auto_adjust) from 2023-06-01 for the 10-instrument universe; "
            "ThetaData FXE EOD option chains via local ThetaTerminal v3 for rr25 (Black-76 "
            "bisection IV, 25-delta interp in delta space, expiry nearest 30d in [20,45], "
            "min 5 strikes); all public data, all responses cached; hourly bars NOT used — "
            "daily mapping per spec §6 default"
        ),
        "isolation": (
            "research/political_alpha/ imports nothing from sovereign/, ict/, ict-engine/, "
            "config/, audit/, scripts/ — AST-enforced by its own test_isolation.py"
        ),
    },
    "prior_expectation": "NOT_SIGNIFICANT",
    "verdict": None,
    "universe": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "DX-Y.NYB",
        "XLE", "SLX", "XLF", "KWEB", "GLD",
    ],
    "hash_method": (
        "sha256 of json.dumps(doc, sort_keys=True, separators=(',',':')) "
        "where doc = this object MINUS the hash_lock field"
    ),
}


def _canonical_hash(doc: dict) -> str:
    # Copied from scripts/research/preregister_positioning.py:221 (NOT imported).
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def register() -> None:
    if PREREG_PATH.exists():
        sys.exit(f"REFUSED: {PREREG_PATH} already exists — preregs are never overwritten.")

    doc = dict(DOC)
    doc["hash_lock"] = _canonical_hash(doc)
    PREREG_PATH.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"prereg written: {PREREG_PATH}")
    print(f"hash_lock:      {doc['hash_lock']}")

    ledger = json.loads(LEDGER_PATH.read_text())
    assert isinstance(ledger, list), "ledger must be a JSON array"
    if any(e.get("id") == "HYP-085" for e in ledger):
        sys.exit("REFUSED: HYP-085 already present in the ledger.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = LEDGER_PATH.with_suffix(f".bak-{stamp}.json")
    shutil.copy2(LEDGER_PATH, backup)
    print(f"ledger backup:  {backup}")

    ledger.append({
        "id": "HYP-085",
        "name": doc["name"],
        "status": "PREREGISTERED",
        "date_registered": FROZEN_AT[:10],
        "family": "NONE",
        "hash_lock": doc["hash_lock"],
        "prereg_file": "data/research/preregister/HYP-085_political_alpha_trump_events.json",
        "mechanism": (
            "policy statements from a single actor move the named sector/currency; advance "
            "positioning, if any, shows up as pre-announcement FXE rr25 / options-volume drift"
        ),
        "methodology_note": (
            "self-contained event study in research/political_alpha/ per the vault spec "
            "Political-Alpha-Claude-Code-Spec.md; one bootstrap primary, descriptive secondaries; "
            "TICK-020"
        ),
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
    print(f"ledger entry appended: HYP-085 PREREGISTERED ({len(ledger)} entries total)")


def verify() -> None:
    doc = json.loads(PREREG_PATH.read_text())
    stored = doc.get("hash_lock")
    recomputed = _canonical_hash(doc)
    if stored != recomputed:
        sys.exit(f"HASH MISMATCH: stored {stored} != recomputed {recomputed}")
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((e for e in ledger if e.get("id") == "HYP-085"), None)
    if entry is None:
        sys.exit("LEDGER: HYP-085 entry missing")
    if entry.get("status") != "PREREGISTERED":
        sys.exit(f"LEDGER: unexpected status {entry.get('status')!r}")
    if entry.get("hash_lock") != stored:
        sys.exit("LEDGER: hash_lock does not match prereg file")
    print(f"VERIFIED: HYP-085 hash-lock intact ({stored}) and ledger entry PREREGISTERED.")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else register()
