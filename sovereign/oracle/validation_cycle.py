"""
Oracle Learning Cycle — Phase 3: TEST
sovereign/oracle/validation_cycle.py

Takes Oracle's candidate lesson and runs three statistical tests.
All three must pass for VALIDATED verdict. Cost: $0.00.

Tests:
  A — Effect size: delta Sharpe > 0.05
  B — Significance: two-sample t-test, p < 0.05
  C — Holdout replication: effect within 20% of training effect

Output: data/oracle/validations/YYYY_MM_DD.json
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parents[2]
FORENSICS_FILE  = ROOT / "data" / "forensics" / "trade_forensics.jsonl"
VALIDATED_DIR   = ROOT / "data" / "oracle" / "validations"
VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLE      = 200     # absolute minimum for any test
TRAIN_FRACTION  = 0.70    # 70/30 split
MIN_DELTA_SHARPE = 0.05   # Test A threshold
MAX_P_VALUE      = 0.05   # Test B threshold
HOLDOUT_TOLERANCE = 0.20  # Test C: effect must replicate within 20%


def _load_all_forensics() -> list[dict]:
    if not FORENSICS_FILE.exists():
        return []
    records = []
    with open(FORENSICS_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _compute_sharpe(r_multiples: list[float]) -> float:
    if len(r_multiples) < 5:
        return 0.0
    import statistics
    mean = statistics.mean(r_multiples)
    std  = statistics.stdev(r_multiples)
    return mean / std if std > 0 else 0.0


def _apply_rule(records: list[dict], rule_expr: str) -> tuple[list[float], list[float]]:
    """
    Apply Oracle's testable_rule to split records into filtered vs rest.
    rule_expr is a Python-like condition string evaluated against each record.
    Returns (filtered_r_multiples, all_r_multiples).
    """
    filtered = []
    all_r    = []

    for r in records:
        r_val = r.get("pnl_r") or r.get("r_multiple") or 0.0
        all_r.append(float(r_val))
        try:
            # Safe eval with record fields as locals
            safe_locals = {
                "commitment_score":   float(r.get("commitment_score") if r.get("commitment_score") is not None else 0.5),
                "session":            str(r.get("session", "")),
                "grade":              str(r.get("grade", "")),
                "failure_label":      str(r.get("failure_label", "")),
                "hold_days":          float(r.get("hold_days", 0)),
                "mfe_ratio":          float(r.get("mfe_ratio", 0)),
                "mae_ratio":          float(r.get("mae_ratio", 0)),
                "momentum_5d":        float(r.get("momentum_5d", 0)),
                "vix_slope":          float(r.get("vix_slope", 0)),
                "pair":               str(r.get("pair", "")),
                "outcome":            str(r.get("outcome", "")),
                "pnl_r":              float(r_val),
            }
            if eval(rule_expr, {"__builtins__": {}}, safe_locals):  # noqa: S307
                filtered.append(float(r_val))
        except Exception:
            pass

    return filtered, all_r


def test_a_effect_size(filtered_r: list[float], all_r: list[float]) -> dict:
    """Test A: Delta Sharpe must be > MIN_DELTA_SHARPE."""
    sharpe_filtered = _compute_sharpe(filtered_r)
    sharpe_all      = _compute_sharpe(all_r)
    delta           = sharpe_filtered - sharpe_all
    return {
        "sharpe_filtered": round(sharpe_filtered, 4),
        "sharpe_all":      round(sharpe_all, 4),
        "delta_sharpe":    round(delta, 4),
        "threshold":       MIN_DELTA_SHARPE,
        "passed":          delta > MIN_DELTA_SHARPE,
    }


def test_b_significance(filtered_r: list[float], all_r: list[float]) -> dict:
    """Test B: Two-sample t-test, p < MAX_P_VALUE."""
    if not HAS_SCIPY or len(filtered_r) < 10 or len(all_r) < 10:
        return {"passed": False, "p_value": None, "reason": "Insufficient sample or scipy not available"}
    rest_r = [r for r in all_r if r not in filtered_r]
    if len(rest_r) < 10:
        return {"passed": False, "p_value": None, "reason": "Not enough non-filtered trades for comparison"}
    _, p_val = sp_stats.ttest_ind(filtered_r, rest_r, equal_var=False)
    return {
        "p_value":   round(float(p_val), 6),
        "threshold": MAX_P_VALUE,
        "n_filtered": len(filtered_r),
        "n_rest":     len(rest_r),
        "passed":    float(p_val) < MAX_P_VALUE,
    }


def test_c_holdout(
    records: list[dict], rule_expr: str,
    train_sharpe: float, seed: int = 42
) -> dict:
    """Test C: Holdout replication — effect within 20% of training effect baseline."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_FRACTION)
    holdout = shuffled[split:]

    filtered_h, all_h = _apply_rule(holdout, rule_expr)
    if len(filtered_h) < 5:
        return {"passed": False, "reason": "Insufficient holdout filtered sample"}

    holdout_sharpe = _compute_sharpe(filtered_h)
    all_h_sharpe   = _compute_sharpe(all_h)
    delta_holdout  = holdout_sharpe - all_h_sharpe

    # Replication: holdout delta should be at least 80% of training delta
    replication_ratio = delta_holdout / train_sharpe if train_sharpe > 0 else 0.0
    passed = replication_ratio >= (1.0 - HOLDOUT_TOLERANCE) and delta_holdout > 0

    return {
        "holdout_sharpe":    round(holdout_sharpe, 4),
        "all_holdout_sharpe": round(all_h_sharpe, 4),
        "delta_holdout":     round(delta_holdout, 4),
        "replication_ratio": round(replication_ratio, 4),
        "threshold":         1.0 - HOLDOUT_TOLERANCE,
        "n_holdout_filtered": len(filtered_h),
        "passed": passed,
    }


def run_validation(
    reflection: dict,
    date: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Run all three statistical tests on Oracle's candidate lesson.
    Returns validation result dict. Saves to data/oracle/validations/YYYY_MM_DD.json.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    candidate = reflection.get("reflection", {}).get("candidate_lesson", {})
    rule_expr = candidate.get("testable_rule", "")
    sample_needed = int(candidate.get("sample_needed", MIN_SAMPLE))
    min_sample = max(sample_needed, MIN_SAMPLE)

    records = _load_all_forensics()

    if len(records) < min_sample:
        result = {
            "date": date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_lesson": candidate.get("lesson_text", ""),
            "testable_rule": rule_expr,
            "verdict": "DEFERRED",
            "reason": f"Only {len(records)} forensic records. Need {min_sample}.",
            "trades_available": len(records),
            "sample_needed": min_sample,
        }
        out = VALIDATED_DIR / f"{date}.json"
        out.write_text(json.dumps(result, indent=2))
        return result

    # 70/30 train/holdout split
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_FRACTION)
    train = shuffled[:split]

    filtered_r, all_r = _apply_rule(train, rule_expr)

    if len(filtered_r) < 10:
        result = {
            "date": date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_lesson": candidate.get("lesson_text", ""),
            "testable_rule": rule_expr,
            "verdict": "REJECTED",
            "reason": f"Rule filters only {len(filtered_r)} trades — insufficient sample",
            "trades_available": len(records),
        }
        out = VALIDATED_DIR / f"{date}.json"
        out.write_text(json.dumps(result, indent=2))
        return result

    ta = test_a_effect_size(filtered_r, all_r)
    tb = test_b_significance(filtered_r, all_r)
    tc = test_c_holdout(records, rule_expr, ta["delta_sharpe"], seed=seed)

    all_pass = ta["passed"] and tb["passed"] and tc["passed"]
    verdict = "VALIDATED" if all_pass else ("DEFERRED" if len(records) < min_sample * 2 else "REJECTED")

    result = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_lesson": candidate.get("lesson_text", ""),
        "testable_rule": rule_expr,
        "expected_impact": candidate.get("expected_impact", ""),
        "mechanism": candidate.get("mechanism", candidate.get("lesson_text", "")),
        "verdict": verdict,
        "trades_tested": len(train),
        "trades_filtered": len(filtered_r),
        "training_sharpe": ta["sharpe_filtered"],
        "baseline_sharpe": ta["sharpe_all"],
        "holdout_sharpe": tc.get("holdout_sharpe"),
        "test_a_delta_sharpe": ta["delta_sharpe"],
        "test_a_passed": ta["passed"],
        "test_b_p_value": tb.get("p_value"),
        "test_b_passed": tb["passed"],
        "test_c_replication_ratio": tc.get("replication_ratio"),
        "test_c_passed": tc["passed"],
    }

    out = VALIDATED_DIR / f"{date}.json"
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oracle validation cycle")
    parser.add_argument("--reflection", help="Path to reflection JSON file")
    parser.add_argument("--date", help="Date (YYYY-MM-DD)")
    parser.add_argument("--rule", help="Override testable_rule directly (for testing)")
    args = parser.parse_args()

    if args.rule:
        fake_reflection = {"reflection": {"candidate_lesson": {
            "lesson_text": "Manual test", "testable_rule": args.rule,
            "expected_impact": "unknown", "sample_needed": 100
        }}}
        result = run_validation(fake_reflection, date=args.date)
    elif args.reflection:
        data = json.loads(Path(args.reflection).read_text())
        result = run_validation(data, date=args.date)
    else:
        print("Provide --reflection FILE or --rule EXPR")
        raise SystemExit(1)

    print(f"Verdict: {result['verdict']}")
    print(f"  Test A (delta Sharpe): {result.get('test_a_delta_sharpe')} {'✓' if result.get('test_a_passed') else '✗'}")
    print(f"  Test B (p-value):      {result.get('test_b_p_value')} {'✓' if result.get('test_b_passed') else '✗'}")
    print(f"  Test C (holdout):      {result.get('test_c_replication_ratio')} {'✓' if result.get('test_c_passed') else '✗'}")
