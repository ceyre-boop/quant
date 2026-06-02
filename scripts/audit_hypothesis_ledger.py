"""
Hypothesis Ledger Audit
=======================
Classifies every hypothesis entry by its methodology quality and flags
those that need re-testing through the canonical costed IS/OOS runner.

Classifications:
  HAS_OOS_PVALUE      — has a p_value from permutation test on OOS window
  IN_SAMPLE_ONLY      — has a Sharpe result but no OOS p-value
  METHODOLOGY_INVALID — explicitly used batch/fast backtester (no costs)
  NEEDS_RETEST        — CONFIRMED/PARTIAL but no OOS validation

Writes:
  data/audit/ledger_audit.md      — human-readable report
  Updates hypothesis_ledger.json  — marks NEEDS_RETEST entries
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
AUDIT_DIR   = ROOT / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Hypotheses that used the broken pipeline (batch backtester, no costs, wrong annualization).
# These are flagged METHODOLOGY_INVALID regardless of their stored status.
KNOWN_BROKEN_METHODOLOGY = {
    "HYP-027", "HYP-028", "HYP-030", "HYP-031", "HYP-034",
    "HYP-035", "HYP-036", "HYP-037", "HYP-038", "HYP-039",
    "HYP-040", "HYP-041", "HYP-042", "HYP-043", "HYP-044",
}

# Statuses that mean "we claimed this works"
CLAIMED_STATUSES = {
    "CONFIRMED", "PARTIAL_CONFIRMED", "CONFIRMED_DIVERSIFICATION",
    "PARTIAL", "DEPLOYED", "PARTIAL — needs formal promotion test",
}


def classify(h: dict) -> str:
    hyp_id = h.get("id", "")
    methodology = h.get("methodology", "")
    p_value     = h.get("p_value")
    status      = h.get("status", "")

    if methodology == "canonical_costed_is_oos":
        return "HAS_OOS_PVALUE"

    if hyp_id in KNOWN_BROKEN_METHODOLOGY:
        return "METHODOLOGY_INVALID"

    if isinstance(p_value, (int, float)):
        return "HAS_OOS_PVALUE"

    if status in CLAIMED_STATUSES:
        return "NEEDS_RETEST"

    return "IN_SAMPLE_ONLY"


def main() -> None:
    if not LEDGER_PATH.exists():
        print("ERROR: hypothesis_ledger.json not found")
        sys.exit(1)

    ledger = json.loads(LEDGER_PATH.read_text())
    hypotheses = ledger.get("hypotheses", [])

    counts = {
        "HAS_OOS_PVALUE": [],
        "IN_SAMPLE_ONLY": [],
        "METHODOLOGY_INVALID": [],
        "NEEDS_RETEST": [],
    }

    updated = False
    for h in hypotheses:
        c = classify(h)
        counts[c].append(h)
        # Mark METHODOLOGY_INVALID entries for re-testing if they were claimed confirmed
        if c == "METHODOLOGY_INVALID" and h.get("status") in CLAIMED_STATUSES:
            h["status"] = "NEEDS_RETEST"
            h["methodology_note"] = (
                "Original test used batch/fast backtester (no costs, wrong annualization, "
                "no OOS holdout). Re-run via: PYTHONPATH=. python3 scripts/run_hypothesis.py "
                f"--id {h['id']} --name '{h.get('name', '')}'"
            )
            updated = True

    if updated:
        LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=float))

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  HYPOTHESIS LEDGER AUDIT")
    print(f"  Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Total hypotheses: {len(hypotheses)}")
    print(f"{'═'*64}\n")

    print(f"  ✓ HAS_OOS_PVALUE      ({len(counts['HAS_OOS_PVALUE'])}): validated via canonical runner")
    for h in counts["HAS_OOS_PVALUE"]:
        p = h.get("p_value", "?")
        print(f"      {h['id']:10s} p={p}  status={h.get('status')}")

    print(f"\n  ⚠ NEEDS_RETEST        ({len(counts['NEEDS_RETEST'])}): claimed confirmed, no OOS validation")
    for h in counts["NEEDS_RETEST"]:
        print(f"      {h['id']:10s} status={h.get('status')}  name={h.get('name','')[:50]}")

    print(f"\n  ✗ METHODOLOGY_INVALID ({len(counts['METHODOLOGY_INVALID'])}): used broken pipeline")
    for h in counts["METHODOLOGY_INVALID"]:
        orig_status = h.get("status")
        print(f"      {h['id']:10s} was={orig_status}  name={h.get('name','')[:50]}")

    print(f"\n  · IN_SAMPLE_ONLY      ({len(counts['IN_SAMPLE_ONLY'])}): rejected/inconclusive, no retest needed")
    for h in counts["IN_SAMPLE_ONLY"]:
        print(f"      {h['id']:10s} status={h.get('status')}")

    # ── Write markdown report ─────────────────────────────────────────────
    lines = [
        "# Hypothesis Ledger Audit",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total: {len(hypotheses)} hypotheses",
        "",
        "## Action Required",
        "",
        "### NEEDS_RETEST — run through canonical runner",
        "```",
    ]
    for h in counts["NEEDS_RETEST"] + counts["METHODOLOGY_INVALID"]:
        if h.get("status") in CLAIMED_STATUSES or h.get("status") == "NEEDS_RETEST":
            lines.append(
                f"PYTHONPATH=. python3 scripts/run_hypothesis.py "
                f"--id {h['id']} --name \"{h.get('name', '')}\" --perms 500"
            )
    lines += [
        "```",
        "",
        "## Status Summary",
        "",
        f"| Classification | Count | Action |",
        f"|---------------|-------|--------|",
        f"| HAS_OOS_PVALUE | {len(counts['HAS_OOS_PVALUE'])} | ✓ None |",
        f"| NEEDS_RETEST | {len(counts['NEEDS_RETEST'])} | Re-run via canonical runner |",
        f"| METHODOLOGY_INVALID | {len(counts['METHODOLOGY_INVALID'])} | Re-run if claimed confirmed |",
        f"| IN_SAMPLE_ONLY | {len(counts['IN_SAMPLE_ONLY'])} | None (rejected) |",
        "",
        "## Why methodology_invalid?",
        "",
        "The batch backtester (`ForexBatchBacktester`) and fast backtester (`ForexFastBacktester.run()`)",
        "do **not** apply `_apply_costs()`. All Sharpe numbers from these paths are pre-cost,",
        "daily-annualized, and in-sample only. The canonical runner (`scripts/run_hypothesis.py`)",
        "uses `ForexBacktester` with costs, √(n/years) annualization, and mandatory IS/OOS split.",
    ]

    report_path = AUDIT_DIR / "ledger_audit.md"
    report_path.write_text("\n".join(lines))
    print(f"\n  Report saved → {report_path}")
    if updated:
        print(f"  Ledger updated — {sum(1 for h in counts['METHODOLOGY_INVALID'] if h.get('status') == 'NEEDS_RETEST')} entries marked NEEDS_RETEST")
    print()


if __name__ == "__main__":
    main()
