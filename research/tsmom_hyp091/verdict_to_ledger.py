"""Phase 4 — write the adjudicated verdict back to the canonical hypothesis ledger.

Backs the ledger up first, preserves the Phase-0 hash_lock, and updates only the
HYP-091 entry's result fields. Idempotent-ish: refuses to overwrite a non-PREREGISTERED
status unless --force. Run AFTER run_study.py has produced results.json.

    python3 research/tsmom_hyp091/verdict_to_ledger.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from research.tsmom_hyp091._lib import OUT_DIR, ROOT

LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
RESULTS_PATH = OUT_DIR / "results.json"
HYP_ID = "HYP-091"


def main() -> None:
    if not RESULTS_PATH.exists():
        sys.exit(f"REFUSED: {RESULTS_PATH} missing — run run_study.py first.")
    res = json.loads(RESULTS_PATH.read_text())
    g = res["gauntlet"]
    prim = res["correlation"]["primary"]

    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((e for e in ledger if e.get("id") == HYP_ID), None)
    if entry is None:
        sys.exit(f"REFUSED: {HYP_ID} not in ledger — Phase 0 must run first.")
    if entry.get("status") not in ("PREREGISTERED",) and "--force" not in sys.argv:
        sys.exit(f"REFUSED: {HYP_ID} status is {entry.get('status')!r}, not PREREGISTERED "
                 f"(pass --force to re-adjudicate).")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(LEDGER_PATH, LEDGER_PATH.with_suffix(f".bak-{stamp}.json"))

    entry["status"] = g["verdict"]
    entry["verdict"] = g["verdict"]
    entry["result"] = res["verdict_reasons"]
    entry["date_tested"] = stamp[:8]
    entry["p_value"] = g["permutation_p"]
    entry["pvalue_source"] = "directional_sign_permutation"
    entry["bh_survives"] = g["bh_survives"]
    entry["oos_sharpe"] = g["oos_sharpe"]
    entry["is_sharpe"] = g["is_sharpe"]
    entry["full_sharpe"] = g["full_sharpe"]
    entry["carry_corr_monthly"] = prim["corr_full"]
    entry["null_triggered"] = g["null_triggered"]
    entry["deployment"] = "OUT_OF_SCOPE (research pass; RISK_CONSTITUTION Art. 6)"
    # hash_lock preserved as-is (never recomputed here).

    with tempfile.NamedTemporaryFile("w", dir=LEDGER_PATH.parent, suffix=".tmp", delete=False) as tmp:
        tmp.write(json.dumps(ledger, indent=2) + "\n")
    Path(tmp.name).replace(LEDGER_PATH)
    print(f"ledger updated: {HYP_ID} -> {g['verdict']} (p={g['permutation_p']}, OOS Sharpe={g['oos_sharpe']:+.3f}, "
          f"corr={prim['corr_full']}); hash_lock preserved.")


if __name__ == "__main__":
    main()
