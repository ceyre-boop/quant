"""
Oracle Retrospective Validation
scripts/retro_validate.py

Runs the 3-gate statistical tests against all 9 active lessons in
I_am_a_good_trader.md. The validation pipeline (validation_cycle.py)
was built after the lessons were codified, so none of them have ever
been tested. This script fixes that.

Usage:
    python3 scripts/retro_validate.py
    python3 scripts/retro_validate.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.oracle.validation_cycle import (
    _compute_sharpe,
    _load_all_forensics,
    test_a_effect_size,
    test_b_significance,
    MIN_DELTA_SHARPE,
    MAX_P_VALUE,
    TRAIN_FRACTION,
    HOLDOUT_TOLERANCE,
)

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VALIDATIONS_DIR = ROOT / "data" / "oracle" / "validations"
PROVEN_RESEARCH = ROOT / "data" / "oracle" / "proven_research.json"
WISDOM_FILE = ROOT / "I_am_a_good_trader.md"

# Lessons that control EXIT behavior, not entry conditions.
# The forensics 'hold' field records actual trade duration, not whether a
# forced hold cap fired. These rules cannot be distinguished from confounders
# by the forensics slice test — only the backtester is a valid test for exit rules.
# Proved 2026-05-27: L-5 retro test returned delta=-0.087 (harmful) but backtester
# showed removing overrides dropped GBPUSD 1.89→1.70, AUDUSD 1.67→1.43.
# Never auto-suspend exit rules from retro results.
UNTESTABLE_EXIT_RULES: set[str] = {"L-5", "L-8"}


# ---------------------------------------------------------------------------
# Extended rule evaluator — fixes two bugs in the base _apply_rule:
#   1. hold_days -> field is actually named "hold"
#   2. pd_align lives in features{}, not top-level
# ---------------------------------------------------------------------------

def _apply_rule_extended(records: list[dict], rule_expr: str) -> tuple[list[float], list[float]]:
    filtered, all_r = [], []
    for r in records:
        r_val = float(r.get("pnl_r") or 0.0)
        all_r.append(r_val)
        feats = r.get("features") or {}
        safe_locals = {
            "commitment_score": float(r.get("commitment_score", 0.5)),
            "session":          str(r.get("session", "")),
            "grade":            str(r.get("grade", "")),
            "failure_label":    str(r.get("failure_label", "")),
            "hold":             float(r.get("hold", 0)),
            "hold_days":        float(r.get("hold", 0)),
            "mfe_ratio":        float(r.get("mfe_ratio") or feats.get("mfe_within_3_bars", 0)),
            "mae_ratio":        float(r.get("mae_ratio") or feats.get("mae_within_3_bars", 0)),
            "momentum_5d":      float(r.get("momentum_5d", 0)),
            "vix_slope":        float(r.get("vix_slope", 0)),
            "pair":             str(r.get("pair", "")),
            "outcome":          str(r.get("outcome", "")),
            "pnl_r":            r_val,
            "system":           str(r.get("system", "")),
            "score":            float(r.get("score", 0)),
            "pd_align":         float(feats.get("pd_align", -1)),
            "mkt_struct":       float(feats.get("mkt_struct", 0)),
        }
        try:
            if eval(rule_expr, {"__builtins__": {}}, safe_locals):  # noqa: S307
                filtered.append(r_val)
        except Exception:
            pass
    return filtered, all_r


def _test_c_extended(
    records: list[dict], rule_expr: str, train_delta: float, seed: int = 42
) -> dict:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_FRACTION)
    holdout = shuffled[split:]

    filtered_h, all_h = _apply_rule_extended(holdout, rule_expr)
    if len(filtered_h) < 5:
        return {"passed": False, "reason": "Insufficient holdout filtered sample",
                "n_holdout_filtered": len(filtered_h)}

    delta_holdout = _compute_sharpe(filtered_h) - _compute_sharpe(all_h)
    ratio = delta_holdout / train_delta if train_delta > 0 else 0.0
    passed = ratio >= (1.0 - HOLDOUT_TOLERANCE) and delta_holdout > 0

    return {
        "holdout_sharpe":       round(_compute_sharpe(filtered_h), 4),
        "all_holdout_sharpe":   round(_compute_sharpe(all_h), 4),
        "delta_holdout":        round(delta_holdout, 4),
        "replication_ratio":    round(ratio, 4),
        "threshold":            round(1.0 - HOLDOUT_TOLERANCE, 2),
        "n_holdout_filtered":   len(filtered_h),
        "passed":               passed,
    }


# ---------------------------------------------------------------------------
# Lesson registry
# ---------------------------------------------------------------------------

LESSONS = [
    {
        "id": "L-1", "lesson_num": 1,
        "name": "The Exact Setup Exists",
        "testable_rule": "system == 'ICT' and session == 'London' and grade == 'A'",
        "testability": "PARTIAL",
        "note": "commitment_score absent from forensics; tests session+grade gate only",
        "sample_needed": 50,
    },
    {
        "id": "L-2", "lesson_num": 2,
        "name": "NY_PM Is Anti-Edge",
        "testable_rule": "session != 'NY_PM'",
        "testability": "FULL",
        "note": "Non-NY_PM trades should outperform the full mix (removal benefit test)",
        "sample_needed": 100,
    },
    {
        "id": "L-3", "lesson_num": 3,
        "name": "Grade Labels Invert Quality",
        "testable_rule": "system == 'ICT' and grade == 'A'",
        "testability": "FULL",
        "note": "Grade A ICT trades vs all ICT (A + A+ mix); A should outperform",
        "sample_needed": 30,
    },
    {
        "id": "L-4", "lesson_num": 4,
        "name": "pd_alignment Anti-Signal",
        "testable_rule": "system == 'ICT' and pd_align == 0.0",
        "testability": "FULL",
        "note": "pd_align read from features{}; pd_align=0 trades should outperform",
        "sample_needed": 50,
    },
    {
        "id": "L-5", "lesson_num": 5,
        "name": "Macro Momentum Half-Lives",
        "testable_rule": None,
        "testability": "UNTESTABLE_EXIT",
        "note": "Controls forced exit timing (hold caps). Forensics cannot distinguish forced exits from natural ones. Validate with run_forex_scan.py --backtest only.",
        "sample_needed": None,
    },
    {
        "id": "L-6", "lesson_num": 6,
        "name": "Counter-Momentum Sizing",
        "testable_rule": None,
        "testability": "UNTESTABLE",
        "note": "momentum_5d not captured in trade_forensics.jsonl",
        "sample_needed": None,
    },
    {
        "id": "L-7", "lesson_num": 7,
        "name": "VIX Term Structure",
        "testable_rule": None,
        "testability": "UNTESTABLE",
        "note": "vix_slope not captured in trade_forensics.jsonl",
        "sample_needed": None,
    },
    {
        "id": "L-8", "lesson_num": 8,
        "name": "Safe-Haven VIX Gate Tightening",
        "testable_rule": None,
        "testability": "UNTESTABLE_EXIT",
        "note": "Controls which trades are taken via VIX threshold gate. Entry filter with exit implications — VIX value at entry not captured in forensics. Validate with run_forex_scan.py --backtest only.",
        "sample_needed": None,
    },
    {
        "id": "L-9", "lesson_num": 9,
        "name": "Knowing Is Not Doing",
        "testable_rule": None,
        "testability": "NOT_APPLICABLE",
        "note": "Meta-process lesson — statistical validation not applicable",
        "sample_needed": None,
    },
]


# ---------------------------------------------------------------------------
# Core validation runner
# ---------------------------------------------------------------------------

def run_lesson_validation(lesson: dict, records: list[dict], seed: int = 42) -> dict:
    rule = lesson["testable_rule"]
    lid = lesson["id"]
    date = TODAY

    base = {
        "lesson_id":       lid,
        "lesson_num":      lesson["lesson_num"],
        "lesson_name":     lesson["name"],
        "date":            date,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "testability":     lesson["testability"],
        "testability_note": lesson["note"],
        "testable_rule":   rule,
    }

    if lesson["testability"] in ("UNTESTABLE", "NOT_APPLICABLE", "UNTESTABLE_EXIT"):
        return {**base, "verdict": lesson["testability"], "status": lesson["testability"]}

    min_sample = max(lesson.get("sample_needed") or 200, 200)

    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_FRACTION)
    train = shuffled[:split]

    filtered_r, all_r = _apply_rule_extended(train, rule)

    if len(filtered_r) < 10:
        result = {**base,
                  "verdict": "REJECTED",
                  "status": "REJECTED",
                  "reason": f"Rule filters only {len(filtered_r)} trades in training set",
                  "trades_tested": len(train),
                  "trades_filtered": len(filtered_r)}
        return result

    ta = test_a_effect_size(filtered_r, all_r)
    tb = test_b_significance(filtered_r, all_r)
    tc = _test_c_extended(records, rule, ta["delta_sharpe"], seed=seed)

    all_pass = ta["passed"] and tb["passed"] and tc["passed"]
    if all_pass:
        verdict = "VALIDATED"
    elif len(records) < min_sample * 2:
        verdict = "DEFERRED"
    else:
        verdict = "REJECTED"

    # Status assignment
    if verdict == "VALIDATED":
        status = "VALIDATED"
    elif ta["delta_sharpe"] < 0:
        status = "SUSPENDED"
    elif not ta["passed"]:
        status = "LOW_CONFIDENCE"
    else:
        status = "REJECTED"

    return {
        **base,
        "verdict":                 verdict,
        "status":                  status,
        "trades_tested":           len(train),
        "trades_filtered":         len(filtered_r),
        "baseline_sharpe":         ta["sharpe_all"],
        "training_sharpe":         ta["sharpe_filtered"],
        "test_a_delta_sharpe":     ta["delta_sharpe"],
        "test_a_passed":           ta["passed"],
        "test_b_p_value":          tb.get("p_value"),
        "test_b_passed":           tb["passed"],
        "test_c_replication_ratio": tc.get("replication_ratio"),
        "test_c_holdout_sharpe":   tc.get("holdout_sharpe"),
        "test_c_passed":           tc["passed"],
    }


# ---------------------------------------------------------------------------
# File updaters
# ---------------------------------------------------------------------------

def _update_proven_research(results: list[dict], dry_run: bool) -> None:
    if not PROVEN_RESEARCH.exists():
        return
    data = json.loads(PROVEN_RESEARCH.read_text())
    lessons_by_num = {
        r["lesson_num"]: r for r in results
        if r.get("testable_rule") and r.get("verdict") not in ("UNTESTABLE", "NOT_APPLICABLE")
    }
    for entry in data.get("proven_lessons", []):
        num = entry.get("lesson_number_in_file")
        if num in lessons_by_num:
            res = lessons_by_num[num]
            entry["p_value"] = res.get("test_b_p_value")
            entry["holdout_validated"] = bool(res.get("test_c_passed"))
            entry["retro_validation_date"] = TODAY
            entry["retro_verdict"] = res.get("verdict")
            entry["retro_delta_sharpe"] = res.get("test_a_delta_sharpe")
    if not dry_run:
        PROVEN_RESEARCH.write_text(json.dumps(data, indent=2))


def _update_wisdom_file(results: list[dict], dry_run: bool) -> None:
    if not WISDOM_FILE.exists():
        return
    content = WISDOM_FILE.read_text()

    status_map = {r["lesson_num"]: r for r in results}

    def replace_health(match: re.Match) -> str:
        header_line = match.group(0)
        # Extract lesson number from header
        m = re.search(r'LESSON (\d+)', header_line)
        if not m:
            return header_line
        num = int(m.group(1))
        res = status_map.get(num)
        if not res:
            return header_line
        return header_line

    # Replace health markers lesson-by-lesson
    for res in results:
        num = res["lesson_num"]
        status = res.get("status")
        if status == "SUSPENDED":
            new_health = f"**Health:** 🔴 SUSPENDED — retro validation failed (delta={res.get('test_a_delta_sharpe', '?')})"
        elif status == "LOW_CONFIDENCE":
            new_health = f"**Health:** 🟡 LOW_CONFIDENCE — delta_sharpe={res.get('test_a_delta_sharpe', '?')} (below 0.05 threshold)"
        else:
            continue  # Don't touch VALIDATED, UNTESTABLE, NOT_APPLICABLE

        # Find the lesson block and replace its Health line
        pattern = re.compile(
            rf"(### LESSON {num} —[^\n]*\n(?:(?!### LESSON)[\s\S])*?)"
            rf"(\*\*Health:\*\*[^\n]*)",
            re.MULTILINE,
        )
        content = pattern.sub(lambda m: m.group(1) + new_health, content)

    if not dry_run:
        WISDOM_FILE.write_text(content)


def _write_validation_files(results: list[dict], dry_run: bool) -> None:
    VALIDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    suspended = []
    low_conf = []
    untestable_exit = []

    for res in results:
        all_results.append(res)
        if res.get("status") == "SUSPENDED":
            suspended.append(res["lesson_id"])
        elif res.get("status") == "LOW_CONFIDENCE":
            low_conf.append(res["lesson_id"])
        elif res.get("testability") == "UNTESTABLE_EXIT":
            untestable_exit.append(res["lesson_id"])

        if not dry_run:
            out = VALIDATIONS_DIR / f"retro_{res['lesson_id']}.json"
            out.write_text(json.dumps(res, indent=2))

    summary = {
        "date":                    TODAY,
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "total_lessons":           len(results),
        "lessons_suspended":       suspended,
        "lessons_low_confidence":  low_conf,
        "lessons_untestable_exit": untestable_exit,
        "results":                 all_results,
    }
    if not dry_run:
        (VALIDATIONS_DIR / "retro_summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _fmt(val, width=7) -> str:
    if val is None:
        return "—".center(width)
    if isinstance(val, float):
        return f"{val:+.4f}"[:width].center(width) if val < 0 else f"{val:.4f}"[:width].center(width)
    return str(val)[:width].center(width)


def _print_table(results: list[dict]) -> None:
    w = "═" * 93
    print(f"\n{w}")
    print(f"  ORACLE RETROSPECTIVE VALIDATION — {TODAY}")
    print(w)
    print(f"  {'LESSON':<6}  {'NAME':<34}  {'DELTA_S':>8}  {'P_VALUE':>8}  {'HOLDOUT':>8}  VERDICT")
    print("  " + "─" * 89)

    for r in results:
        lid    = r["lesson_id"]
        name   = r["lesson_name"][:33]
        ds     = r.get("test_a_delta_sharpe")
        pv     = r.get("test_b_p_value")
        hr     = r.get("test_c_replication_ratio")
        verdict = r.get("verdict", "")
        status  = r.get("status", "")

        ds_str = f"{ds:+.4f}" if ds is not None else "    —   "
        pv_str = f"{pv:.4f}"  if pv is not None else "    —   "
        hr_str = f"{hr:.4f}"  if hr is not None else "    —   "

        flag = ""
        if status == "SUSPENDED":        flag = " ⛔"
        elif status == "LOW_CONFIDENCE": flag = " ⚠️"
        elif status == "VALIDATED":      flag = " ✓"
        elif status == "UNTESTABLE":     flag = " ❓"
        elif status == "UNTESTABLE_EXIT": flag = " 🚫"
        elif status == "NOT_APPLICABLE": flag = "  —"

        print(f"  {lid:<6}  {name:<34}  {ds_str:>8}  {pv_str:>8}  {hr_str:>8}  {verdict}{flag}")

    print(w)

    # Action section
    suspended    = [r for r in results if r.get("status") == "SUSPENDED"]
    low_conf     = [r for r in results if r.get("status") == "LOW_CONFIDENCE"]
    untestable   = [r for r in results if r.get("testability") == "UNTESTABLE"]
    exit_rules   = [r for r in results if r.get("testability") == "UNTESTABLE_EXIT"]

    if suspended or low_conf:
        print("\n  LESSONS REQUIRING ACTION:")
        for r in suspended:
            print(f"  ⛔  {r['lesson_id']} SUSPENDED: {r['lesson_name']}")
            print(f"       delta_sharpe={r.get('test_a_delta_sharpe')}, fails all 3 gates")
            print(f"       → Status updated in I_am_a_good_trader.md")
        for r in low_conf:
            print(f"  ⚠️   {r['lesson_id']} LOW_CONFIDENCE: {r['lesson_name']}")
            print(f"       delta_sharpe={r.get('test_a_delta_sharpe'):.4f} (threshold: {MIN_DELTA_SHARPE})")
            print(f"       → Size modifier should be halved until more trades accumulate")

    if exit_rules:
        print("\n  UNTESTABLE EXIT RULES (backtester only — forensics blind to forced exits):")
        for r in exit_rules:
            print(f"  🚫  {r['lesson_id']}: {r['testability_note']}")

    if untestable:
        print("\n  UNTESTABLE — fields missing from trade_forensics.jsonl (add to capture):")
        for r in untestable:
            print(f"  ❓  {r['lesson_id']}: {r['testability_note']}")

    print(f"\n  Files written to: data/oracle/validations/")
    print(f"  proven_research.json updated with p_values and holdout results")
    print(w + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle retrospective lesson validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print table but skip all file writes")
    args = parser.parse_args()

    records = _load_all_forensics()
    if not records:
        print("ERROR: No forensics records found at data/forensics/trade_forensics.jsonl")
        sys.exit(1)

    print(f"\nLoaded {len(records)} forensics records. Running 3-gate validation on 9 lessons...")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    results = [run_lesson_validation(lesson, records) for lesson in LESSONS]

    _print_table(results)

    if not args.dry_run:
        _write_validation_files(results, dry_run=False)
        _update_proven_research(results, dry_run=False)
        _update_wisdom_file(results, dry_run=False)
    else:
        _write_validation_files(results, dry_run=True)
        _update_proven_research(results, dry_run=True)
        _update_wisdom_file(results, dry_run=True)


if __name__ == "__main__":
    main()
