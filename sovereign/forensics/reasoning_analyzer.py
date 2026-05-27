"""
Reasoning Pattern Analyzer — monthly batch analysis.
sovereign/forensics/reasoning_analyzer.py

Reads ALL decision logs with outcomes filled.
Answers: which reasoning chains produce the best and worst R?

Method:
  1. Load all closed decision_log entries
  2. Extract numeric feature vector per trade
  3. Run sklearn DecisionTreeRegressor (depth=3) — stays interpretable
  4. Run per-feature threshold sweep for individual signals
  5. Classify chains into BEST / WORST / SURPRISING
  6. Save monthly report to data/oracle/reasoning_analysis/YYYY_MM.json

Called monthly by oracle_cycle.run_monthly_monitor().
Direct: python3 -m sovereign.forensics.reasoning_analyzer
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DECISION_LOG_DIR  = ROOT / "data" / "decision_logs"
ANALYSIS_DIR      = ROOT / "data" / "oracle" / "reasoning_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLE = 10   # minimum trades to report any pattern
BEST_R_THRESHOLD  = 0.30   # avg R above this = BEST chain
WORST_R_THRESHOLD = -0.20  # avg R below this = WORST chain


# ─── Feature extraction ───────────────────────────────────────────────────────

def _parse_library_similarity(library_match: Optional[str]) -> float:
    """Extract similarity score from 'PATTERN_NAME at 0.XX' string."""
    if not library_match:
        return 0.0
    m = re.search(r"at\s+([\d.]+)", library_match)
    return float(m.group(1)) if m else 0.0


def _extract_features(rec: dict) -> dict:
    """
    Build numeric feature vector from a decision log record.
    All floats — decision tree needs no encoding.
    """
    why = (rec.get("why_this_trade") or "").lower()
    why_size = (rec.get("why_this_size") or "").lower()
    layers = [s.lower() for s in (rec.get("signal_layers_active") or [])]
    confirmations = rec.get("confirmations") or []

    return {
        # ── reasoning-chain boolean flags (from why_this_trade text) ──
        "has_rate_divergence":   float("rate_div" in why or "irp" in why or "rate diff" in why),
        "has_post_cb_drift":     float("post_cb" in why or "cb drift" in why or "central bank" in why),
        "has_london_session":    float("london" in why or "london" in " ".join(layers)),
        "has_ny_session":        float("ny_am" in why or "ny_pm" in why or "new york" in why),
        "has_library_match":     float(bool(rec.get("library_match"))),
        "has_fvg":               float("fvg" in why or "fvg_tap" in " ".join(layers)),
        "has_sweep":             float("sweep" in why or "sweep" in " ".join(layers)),
        "has_compression":       float("compress" in why or "adr" in why),
        "has_macro_signal":      float("macro" in why or "rate" in why),
        "has_commitment":        float("commit" in why or "uncommit" in why),

        # ── numeric fields ──────────────────────────────────────────────
        "commitment_score":      float(rec.get("commitment_score") or 0.5),
        "vix_at_entry":          float(rec.get("vix_at_entry") or 0.0),
        "rate_diff_z":           float(rec.get("rate_differential_zscore") or 0.0),
        "bars_since_signal":     float(rec.get("bars_since_signal") or 0),
        "risk_pct":              float(rec.get("risk_pct") or 0.0),
        "library_similarity":    _parse_library_similarity(rec.get("library_match")),
        "n_confirmations":       float(len(confirmations)),
        "score":                 float(rec.get("score") or 0.0),

        # ── derived ─────────────────────────────────────────────────────
        "is_ict":                float((rec.get("system") or "") == "ICT"),
        "is_grade_a":            float((rec.get("grade") or "") in ("A", "A+")),
        "is_grade_aplus":        float((rec.get("grade") or "") == "A+"),
        "freshness_mult":        _parse_freshness(why_size),
        "kelly_frac":            _parse_kelly(why_size),
    }


def _parse_freshness(why_size: str) -> float:
    m = re.search(r"freshness\s+([\d.]+)", why_size)
    return float(m.group(1)) if m else 1.0


def _parse_kelly(why_size: str) -> float:
    m = re.search(r"kelly\s+([\d.]+)", why_size)
    return float(m.group(1)) if m else 0.25


# ─── Data loader ─────────────────────────────────────────────────────────────

def load_all_closed_records() -> list[dict]:
    """Load every closed decision log entry across all months."""
    records = []
    if not DECISION_LOG_DIR.exists():
        return records
    for log_file in sorted(DECISION_LOG_DIR.glob("decisions_*.jsonl")):
        try:
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("outcome") and rec.get("r_realized") is not None:
                    records.append(rec)
        except Exception:
            continue
    return records


# ─── Decision tree analysis ───────────────────────────────────────────────────

def _run_decision_tree(features: list[dict], targets: list[float]) -> list[dict]:
    """
    Fit a depth-3 DecisionTreeRegressor and extract leaf-level rules.
    Returns list of {condition, avg_r, n, quality} dicts.
    """
    try:
        from sklearn.tree import DecisionTreeRegressor, export_text
        import numpy as _np
    except ImportError:
        return []

    if len(features) < MIN_SAMPLE:
        return []

    feature_names = list(features[0].keys())
    X = _np.array([[f[k] for k in feature_names] for f in features], dtype=float)
    y = _np.array(targets, dtype=float)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(3, len(features) // 10))
    tree.fit(X, y)

    # Walk the tree to extract leaf rules in plain English
    tree_text = export_text(tree, feature_names=feature_names, decimals=2)
    leaves = _extract_leaf_rules(tree, tree_text, feature_names, X, y)
    return leaves


def _extract_leaf_rules(tree, tree_text: str, feature_names: list, X, y) -> list[dict]:
    """
    For each leaf node: collect samples, compute avg R, build condition string.
    Uses the sklearn decision_path to map samples to leaves.
    """
    from sklearn.tree import _tree
    import numpy as _np

    t = tree.tree_
    node_indicator = tree.decision_path(X)
    leaf_ids = tree.apply(X)

    results = []
    for leaf_id in _np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        n = int(mask.sum())
        avg_r = float(_np.mean(y[mask]))
        if n < 3:
            continue

        # Build condition string by tracing path to this leaf
        condition_parts = []
        node_id = leaf_id
        while node_id != 0:
            parent = _np.where(
                (t.children_left == node_id) | (t.children_right == node_id)
            )[0]
            if len(parent) == 0:
                break
            parent = parent[0]
            feat = feature_names[t.feature[parent]]
            thresh = round(t.threshold[parent], 2)
            if t.children_left[parent] == node_id:
                condition_parts.append(f"{feat} ≤ {thresh}")
            else:
                condition_parts.append(f"{feat} > {thresh}")
            node_id = parent
        condition_parts.reverse()

        results.append({
            "condition": " AND ".join(condition_parts) or "all trades",
            "avg_r": round(avg_r, 3),
            "n": n,
            "quality": "BEST" if avg_r >= BEST_R_THRESHOLD else
                       "WORST" if avg_r <= WORST_R_THRESHOLD else "NEUTRAL",
        })

    return sorted(results, key=lambda x: x["avg_r"], reverse=True)


# ─── Per-feature threshold sweep ─────────────────────────────────────────────

def _threshold_sweep(features: list[dict], targets: list[float]) -> list[dict]:
    """
    For each numeric feature, find the threshold that best separates
    high-R from low-R trades. Returns top findings sorted by effect size.
    """
    import numpy as _np

    numeric_features = [
        "commitment_score", "vix_at_entry", "rate_diff_z",
        "bars_since_signal", "library_similarity", "n_confirmations",
        "freshness_mult", "score",
    ]

    y = _np.array(targets)
    findings = []

    for feat in numeric_features:
        vals = _np.array([f.get(feat, 0.0) for f in features])
        unique_vals = _np.unique(vals)
        if len(unique_vals) < 3:
            continue

        # Sweep candidate thresholds
        best_delta = 0.0
        best_thresh = None
        best_above = best_below = None

        for pct in [25, 33, 50, 67, 75]:
            thresh = float(_np.percentile(vals, pct))
            above = y[vals > thresh]
            below = y[vals <= thresh]
            if len(above) < 5 or len(below) < 5:
                continue
            delta = abs(float(_np.mean(above)) - float(_np.mean(below)))
            if delta > best_delta:
                best_delta = delta
                best_thresh = thresh
                best_above = above
                best_below = below

        if best_thresh is None or best_delta < 0.10:
            continue

        avg_above = round(float(_np.mean(best_above)), 3)
        avg_below = round(float(_np.mean(best_below)), 3)
        n_above = len(best_above)
        n_below = len(best_below)

        # Direction: which side is better?
        if avg_above >= avg_below:
            direction = f"{feat} > {round(best_thresh, 2)}"
            good_r, bad_r = avg_above, avg_below
            good_n, bad_n = n_above, n_below
        else:
            direction = f"{feat} ≤ {round(best_thresh, 2)}"
            good_r, bad_r = avg_below, avg_above
            good_n, bad_n = n_below, n_above

        findings.append({
            "feature": feat,
            "threshold": round(best_thresh, 2),
            "best_side": direction,
            "good_avg_r": good_r,
            "good_n": good_n,
            "bad_avg_r": bad_r,
            "bad_n": bad_n,
            "delta_r": round(good_r - bad_r, 3),
        })

    return sorted(findings, key=lambda x: -x["delta_r"])


# ─── Report builder ───────────────────────────────────────────────────────────

@dataclass
class ReasoningReport:
    month: str
    generated_at: str
    n_trades_analyzed: int
    system_breakdown: dict
    best_chains: list
    worst_chains: list
    neutral_chains: list
    feature_signals: list
    surprising_findings: list
    oracle_context_summary: str   # compact string for Oracle's daily prompt


def _build_surprising(tree_leaves: list, feature_signals: list) -> list[str]:
    """
    Flag results that contradict intuition or conventional wisdom.
    Heuristic: anything where the 'good' side seems backwards.
    """
    surprises = []
    for sig in feature_signals:
        feat = sig["feature"]
        good_side = sig["best_side"]
        delta = sig["delta_r"]

        # Surprising: high commitment_score is NOT better than low
        if feat == "commitment_score" and "≤" in good_side and delta > 0.15:
            surprises.append(
                f"Lower commitment_score (≤{sig['threshold']}) produces better R "
                f"(avg {sig['good_avg_r']:+.3f} vs {sig['bad_avg_r']:+.3f}) — "
                "high commitment score may reflect late entries after institutional move"
            )
        # Surprising: more confirmations is worse
        if feat == "n_confirmations" and "≤" in good_side and delta > 0.10:
            surprises.append(
                f"Fewer confirmations (≤{sig['threshold']}) produces better R "
                f"(avg {sig['good_avg_r']:+.3f} vs {sig['bad_avg_r']:+.3f}) — "
                "overconfirmation may mean the move is already done"
            )
        # Surprising: high VIX better than low
        if feat == "vix_at_entry" and ">" in good_side and delta > 0.10:
            surprises.append(
                f"High VIX at entry (>{sig['threshold']}) produces better R "
                f"(avg {sig['good_avg_r']:+.3f} vs {sig['bad_avg_r']:+.3f}) — "
                "elevated VIX may amplify rate differential signal"
            )
    return surprises


def _build_oracle_summary(report_dict: dict) -> str:
    """
    Compact 200-word summary Oracle loads as standing context in its daily prompt.
    """
    n = report_dict["n_trades_analyzed"]
    if n < MIN_SAMPLE:
        return (
            f"Reasoning analysis: only {n} closed trades — insufficient for pattern analysis. "
            "Accumulate at least 10 closed trades before monthly analysis is meaningful."
        )

    best = report_dict["best_chains"][:2]
    worst = report_dict["worst_chains"][:2]
    feats = report_dict["feature_signals"][:3]
    surprises = report_dict["surprising_findings"][:2]

    lines = [f"Monthly reasoning analysis ({n} trades, {report_dict['month']}):"]

    if best:
        lines.append("BEST chains:")
        for b in best:
            lines.append(f"  [{b['n']}tr, avg R={b['avg_r']:+.3f}] {b['condition']}")

    if worst:
        lines.append("WORST chains:")
        for w in worst:
            lines.append(f"  [{w['n']}tr, avg R={w['avg_r']:+.3f}] {w['condition']}")

    if feats:
        lines.append("Strongest single-feature signals:")
        for f in feats:
            lines.append(
                f"  {f['best_side']} → avg R={f['good_avg_r']:+.3f} (n={f['good_n']}) "
                f"vs {f['bad_avg_r']:+.3f} (n={f['bad_n']}), delta={f['delta_r']:+.3f}"
            )

    if surprises:
        lines.append("Surprising:")
        for s in surprises:
            lines.append(f"  {s[:120]}")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_analysis(month: Optional[str] = None) -> ReasoningReport:
    """
    Load all closed decision logs, run tree + sweep, save monthly report.
    Returns the ReasoningReport dataclass.
    """
    month = month or datetime.now(timezone.utc).strftime("%Y_%m")

    records = load_all_closed_records()
    n = len(records)

    print(f"\nReasoning Analyzer — {month}")
    print(f"  Closed trades found: {n}")

    system_breakdown = {}
    for r in records:
        s = r.get("system", "UNKNOWN")
        system_breakdown[s] = system_breakdown.get(s, 0) + 1

    if n < MIN_SAMPLE:
        report = ReasoningReport(
            month=month,
            generated_at=datetime.now(timezone.utc).isoformat(),
            n_trades_analyzed=n,
            system_breakdown=system_breakdown,
            best_chains=[],
            worst_chains=[],
            neutral_chains=[],
            feature_signals=[],
            surprising_findings=[],
            oracle_context_summary=(
                f"Reasoning analysis: only {n} closed trades — "
                f"need {MIN_SAMPLE} minimum. Accumulating."
            ),
        )
    else:
        features = [_extract_features(r) for r in records]
        targets  = [float(r["r_realized"]) for r in records]

        print(f"  Running decision tree (depth=3) ...")
        tree_leaves = _run_decision_tree(features, targets)
        best    = [l for l in tree_leaves if l["quality"] == "BEST"]
        worst   = [l for l in tree_leaves if l["quality"] == "WORST"]
        neutral = [l for l in tree_leaves if l["quality"] == "NEUTRAL"]

        print(f"  Running feature threshold sweep ...")
        feature_signals = _threshold_sweep(features, targets)

        surprising = _build_surprising(tree_leaves, feature_signals)

        report_dict = {
            "month": month,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_trades_analyzed": n,
            "system_breakdown": system_breakdown,
            "best_chains": best,
            "worst_chains": worst,
            "neutral_chains": neutral,
            "feature_signals": feature_signals,
            "surprising_findings": surprising,
        }

        oracle_summary = _build_oracle_summary(report_dict)
        report_dict["oracle_context_summary"] = oracle_summary

        report = ReasoningReport(**report_dict)

        # Print
        print(f"\n{'='*60}")
        print("BEST REASONING CHAINS (avg R > +{:.2f}):".format(BEST_R_THRESHOLD))
        for i, b in enumerate(best, 1):
            print(f"  {i}. [{b['n']}tr, avg R={b['avg_r']:+.3f}] {b['condition']}")
        if not best:
            print("  None yet — more trades needed")

        print(f"\nWORST REASONING CHAINS (avg R < {WORST_R_THRESHOLD:+.2f}):")
        for i, w in enumerate(worst, 1):
            print(f"  {i}. [{w['n']}tr, avg R={w['avg_r']:+.3f}] {w['condition']}")
        if not worst:
            print("  None yet — more trades needed")

        print(f"\nSTRONGEST SINGLE-FEATURE SIGNALS (top 5):")
        for sig in feature_signals[:5]:
            print(
                f"  {sig['best_side']:45s} "
                f"good={sig['good_avg_r']:+.3f}(n={sig['good_n']}) "
                f"bad={sig['bad_avg_r']:+.3f}(n={sig['bad_n']}) "
                f"Δ={sig['delta_r']:+.3f}"
            )

        if surprising:
            print(f"\nSURPRISING FINDINGS:")
            for s in surprising:
                print(f"  ⚡ {s}")

        print(f"{'='*60}")

    # Save
    out_path = ANALYSIS_DIR / f"{month}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2))
    try:
        display = out_path.relative_to(ROOT)
    except ValueError:
        display = out_path
    print(f"\n  Saved: {display}")

    return report


def load_latest_report() -> Optional[str]:
    """
    Return the oracle_context_summary from the most recent monthly report.
    Called by reflect_cycle to inject into Oracle's daily prompt.
    """
    if not ANALYSIS_DIR.exists():
        return None
    reports = sorted(ANALYSIS_DIR.glob("*.json"))
    if not reports:
        return None
    try:
        data = json.loads(reports[-1].read_text())
        return data.get("oracle_context_summary")
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monthly reasoning pattern analyzer")
    parser.add_argument("--month", help="Month (YYYY_MM), default: current month")
    args = parser.parse_args()
    run_analysis(month=args.month)
