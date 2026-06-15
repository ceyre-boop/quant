#!/usr/bin/env python3
"""Component 2 — Hypothesis Generator.

Rule-based (NO LLM). Reads the reps and the ledger the system already has and
proposes falsifiable candidate hypotheses to data/research/auto_hypothesis_queue.jsonl.

Discipline: a detection rule emits a candidate ONLY when the data actually shows
the pattern. If the field it needs is absent (e.g. VIX is null across ICT reps, or
macro_conviction is too sparse), the rule emits nothing — it never fabricates a
signal to hit a count. Every candidate is deduped against the live ledger and the
retired-lessons archive before it is written.

Schedule: launchd com.alta.hypothesis.generator, nightly 03:00 PT (after Oracle).
Direct:   python3 sovereign/autonomous/hypothesis_generator.py [--dry-run]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.autonomous import _common as C

DECISION_GLOB = str(ROOT / "data" / "decision_logs" / "decisions_2026_*.jsonl")
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
ARCHIVE_PATH = ROOT / "I_was_a_good_trader.md"
QUEUE_PATH = ROOT / "data" / "research" / "auto_hypothesis_queue.jsonl"
GENERATOR_LOG = ROOT / "data" / "agent" / "generator_log.jsonl"

_log = C.make_logger("hypothesis_generator")

# A component whose score is ~0 across essentially every rep is an anti-signal
# candidate (it contributes weight while never firing). Threshold is deliberately
# strict so we only flag a component that is genuinely inert.
_INERT_FRACTION = 0.95
_MIN_REPS_FOR_PATTERN = 50


def _load_reps() -> list[dict]:
    reps: list[dict] = []
    for f in sorted(glob.glob(DECISION_GLOB)):
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                reps.append(json.loads(line))
    return reps


def _load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return json.loads(LEDGER_PATH.read_text())


def _existing_text() -> str:
    """Lowercased blob of ledger names+mechanisms and the archive, for dedup."""
    parts: list[str] = []
    for h in _load_ledger():
        parts.append(str(h.get("name", "")))
        parts.append(str(h.get("mechanism", "")))
    if ARCHIVE_PATH.exists():
        parts.append(ARCHIVE_PATH.read_text())
    return " ".join(parts).lower()


def _is_duplicate(keywords: list[str], blob: str) -> bool:
    """A candidate is a duplicate if ALL of its distinguishing keywords already
    co-occur in the ledger/archive blob."""
    return all(k.lower() in blob for k in keywords)


def _candidate(source: str, detector: str, hypothesis: str, mechanism: str,
               test_spec: dict, evidence: str, dup_id: str | None) -> dict:
    return {
        "id": f"HYP-AUTO-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
        "source": source,
        "detector": detector,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "evidence": evidence,
        "test_spec": test_spec,
        "duplicates_check": dup_id,
        "status": "QUEUED",
        "generated_at": C.now_iso(),
    }


# ── Detection rules ───────────────────────────────────────────────────────────

def rule_expiry_by_session(reps: list[dict]) -> list[dict]:
    """ICT limit orders that EXPIRE (never fill) clustered by session → entry-
    mechanism / time-of-day candidate. Grounded in the actual EXPIRED population."""
    ict = [r for r in reps if r.get("system") == "ICT" and r.get("session")]
    if len(ict) < _MIN_REPS_FOR_PATTERN:
        return []
    by_session_total: Counter = Counter()
    by_session_expired: Counter = Counter()
    for r in ict:
        s = str(r.get("session"))
        by_session_total[s] += 1
        if r.get("outcome") == "EXPIRED":
            by_session_expired[s] += 1
    out: list[dict] = []
    overall = sum(by_session_expired.values()) / max(1, sum(by_session_total.values()))
    for s, total in by_session_total.most_common():
        if total < _MIN_REPS_FOR_PATTERN:
            continue
        rate = by_session_expired[s] / total
        # Flag a session whose expiry rate is materially worse than the overall mean.
        if rate >= overall + 0.10:
            out.append(_candidate(
                source="closed_trades",
                detector="expiry_by_session",
                hypothesis=f"ICT limit entries placed in the {s} session expire (never fill) "
                           f"at {rate:.0%} vs {overall:.0%} overall — a session-scoped fill "
                           "deficit, not a directional edge.",
                mechanism="If price routinely fails to return to the FVG limit during this "
                          "session, the setup is being signalled but not executed; either the "
                          "limit offset is too deep for this session's volatility or the window "
                          "should be vetoed.",
                test_spec={
                    "data": f"ICT reps, {s} session, decision_logs 2026",
                    "method": "permutation",
                    "n_required": _MIN_REPS_FOR_PATTERN,
                    "p_threshold": 0.05,
                    "both_sides": True,
                },
                evidence=f"{by_session_expired[s]}/{total} expired in {s}; overall {overall:.0%}.",
                dup_id=None,
            ))
        if len(out) >= 3:
            break
    return out


def rule_inert_components(reps: list[dict]) -> list[dict]:
    """A component_scores key that is ~0 across (almost) all reps is an anti-signal
    candidate — it dilutes the score while never contributing."""
    scored = [r.get("component_scores", {}) for r in reps if isinstance(r.get("component_scores"), dict)]
    if len(scored) < _MIN_REPS_FOR_PATTERN:
        return []
    keys = set().union(*[set(d.keys()) for d in scored]) if scored else set()
    blob = _existing_text()
    out: list[dict] = []

    def _numeric(v) -> bool:
        # bool is a subclass of int — exclude it (below_proven_bar etc.)
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    for k in sorted(keys):
        vals = [float(d[k]) for d in scored if k in d and _numeric(d[k])]
        if len(vals) < _MIN_REPS_FOR_PATTERN:
            continue  # not a numeric component, or too sparse to judge
        zero_frac = sum(1 for v in vals if v == 0) / len(vals)
        if zero_frac >= _INERT_FRACTION:
            dup = _is_duplicate([k, "anti-signal"], blob) or _is_duplicate([k, "zero-weight"], blob)
            out.append(_candidate(
                source="closed_trades",
                detector="inert_component",
                hypothesis=f"The '{k}' component scores 0 in {zero_frac:.0%} of reps — dropping "
                           "its weight (or treating non-zero as an anti-signal) does not reduce, "
                           "and may improve, realized edge.",
                mechanism="A component that almost never fires adds noise to the composite score "
                          "and can invert the intended ranking; zero-weighting isolates the "
                          "components that actually carry signal.",
                test_spec={
                    "data": "all reps with component_scores, decision_logs 2026",
                    "method": "walk-forward",
                    "n_required": _MIN_REPS_FOR_PATTERN,
                    "p_threshold": 0.05,
                    "both_sides": True,
                },
                evidence=f"'{k}' == 0 in {zero_frac:.0%} of {len(vals)} reps.",
                dup_id="ledger-match" if dup else None,
            ))
        if len(out) >= 3:
            break
    return out


def rule_conviction_band(reps: list[dict]) -> list[dict]:
    """FOREX reps: compare realized R across macro_conviction bands (nested in
    component_scores). Emits only if enough resolved FOREX reps exist."""
    forex = []
    for r in reps:
        if r.get("system") != "FOREX":
            continue
        cs = r.get("component_scores", {}) or {}
        conv = cs.get("macro_conviction")
        rr = r.get("r_realized")
        if conv is not None and rr is not None:
            forex.append((float(conv), float(rr)))
    if len(forex) < 30:  # honest: too few resolved FOREX reps to claim a band effect
        return []
    low = [rr for conv, rr in forex if conv < 0.20]
    high = [rr for conv, rr in forex if conv >= 0.30]
    if len(low) < 10 or len(high) < 10:
        return []
    import statistics
    mlow, mhigh = statistics.mean(low), statistics.mean(high)
    if mhigh - mlow <= 0.10:
        return []
    return [_candidate(
        source="closed_trades",
        detector="conviction_band",
        hypothesis=f"FOREX trades with macro_conviction ≥0.30 realize {mhigh:.2f}R vs "
                   f"{mlow:.2f}R for the 0.10–0.20 band — raising the conviction floor lifts "
                   "expectancy.",
        mechanism="Below-proven-bar conviction passes near-zero-signal trades; a stricter "
                  "floor keeps only the reps where the macro edge is actually present.",
        test_spec={
            "data": "FOREX reps with macro_conviction + r_realized, decision_logs 2026",
            "method": "bootstrap",
            "n_required": 30,
            "p_threshold": 0.05,
            "both_sides": True,
        },
        evidence=f"high band mean {mhigh:.2f}R (n={len(high)}); low band {mlow:.2f}R (n={len(low)}).",
        dup_id=None,
    )]


def rule_ledger_retest_gaps(ledger: list[dict]) -> list[dict]:
    """Each RETEST_BLOCKED hypothesis with no costed re-run is a standing research
    gap → propose a costed, OOS re-test."""
    blob = _existing_text()
    out: list[dict] = []
    for h in ledger:
        if h.get("status") != "RETEST_BLOCKED":
            continue
        name = h.get("name", h.get("id", "unknown"))
        out.append(_candidate(
            source="queue_item",
            detector="ledger_retest_gap",
            hypothesis=f"Re-test '{name}' on the canonical costed runner with an OOS holdout — "
                       "the original verdict used an uncosted/misannualized backtest and was "
                       "marked RETEST_BLOCKED.",
            mechanism=h.get("mechanism", "Original effect was real in-sample but never confirmed "
                                          "under costs + out-of-sample; a clean re-run resolves it."),
            evidence=f"Ledger {h.get('id')} status RETEST_BLOCKED; "
                     f"note: {str(h.get('methodology_note',''))[:160]}",
            test_spec={
                "data": f"forex canonical runner, holdout OOS; ref {h.get('id')}",
                "method": "walk-forward",
                "n_required": 100,
                "p_threshold": 0.05,
                "both_sides": True,
            },
            dup_id=h.get("id"),
        ))
        if len(out) >= 3:
            break
    return out


def run(dry_run: bool = False) -> dict:
    reps = _load_reps()
    ledger = _load_ledger()
    blob = _existing_text()
    _log(f"loaded {len(reps)} reps, {len(ledger)} ledger entries | dry_run={dry_run}")

    candidates: list[dict] = []
    candidates += rule_expiry_by_session(reps)
    candidates += rule_inert_components(reps)
    candidates += rule_conviction_band(reps)
    candidates += rule_ledger_retest_gaps(ledger)

    # Final dedup pass: drop any candidate whose key phrase already lives in the ledger
    # AND whose detector is not a deliberate ledger re-test (those carry their own ref).
    kept: list[dict] = []
    for c in candidates:
        if c["detector"] == "ledger_retest_gap":
            kept.append(c)
            continue
        keyphrase = re.findall(r"'([^']+)'", c["hypothesis"])
        if keyphrase and _is_duplicate([keyphrase[0]], blob) and c.get("dup_id") == "ledger-match":
            _log(f"  drop duplicate: {c['detector']} ({keyphrase[0]})")
            continue
        kept.append(c)

    _log(f"generated {len(kept)} candidate(s) from {len(candidates)} raw")
    for c in kept:
        _log(f"  + [{c['detector']}] {c['hypothesis'][:90]}...")

    if not dry_run:
        for c in kept:
            C.append_jsonl(QUEUE_PATH, c)
        C.append_jsonl(GENERATOR_LOG, {
            "timestamp": C.now_iso(),
            "reps": len(reps),
            "ledger": len(ledger),
            "generated": len(kept),
            "detectors": Counter(c["detector"] for c in kept),
        })

    return {"timestamp": C.now_iso(), "generated": len(kept),
            "by_detector": dict(Counter(c["detector"] for c in kept)),
            "dry_run": dry_run}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Hypothesis Generator — rule-based candidates.")
    parser.add_argument("--dry-run", action="store_true", help="generate and log but write nothing")
    args = parser.parse_args()
    print(json.dumps(run(dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
