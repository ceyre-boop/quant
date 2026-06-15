#!/usr/bin/env python3
"""Component 3 — Autonomous Research Factory.

Processes QUEUED hypotheses from data/research/auto_hypothesis_queue.jsonl: pre-registers
the test, runs the mapped validator as a subprocess, applies the methodology gate, and
emits a verdict (VALID_EDGE | NOT_SIGNIFICANT | INVALID | BLOCKED_NO_VALIDATOR | ERROR).

GATED. With config/autonomous.yml::live = false (default) the factory is DRY-RUN: verdicts
go to the shadow file data/research/auto_hypothesis_results.jsonl and factory_log.jsonl, the
curated ledger is NOT touched, and nothing is sent to Colin. Flip `live: true` to activate
direct ledger writes (append-only, backed-up, source:auto_factory) and HIGH-priority notify
on VALID_EDGE. It NEVER deploys to live trading — approval is always human (approve_edge.py).

Validators are bespoke (embedded params), so routing is an explicit registry, not a generic
runner. A candidate with no registered validator is BLOCKED_NO_VALIDATOR — honest, not INVALID.

Schedule: launchd com.alta.research.factory, every 4h weekdays (dry-run until flag flipped).
Direct:   python3 sovereign/autonomous/research_factory.py [--validate <name>] [--live]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.autonomous import _common as C

QUEUE_PATH = ROOT / "data" / "research" / "auto_hypothesis_queue.jsonl"
RESULTS_SHADOW = ROOT / "data" / "research" / "auto_hypothesis_results.jsonl"
PREREGISTER_DIR = ROOT / "data" / "research" / "preregister"
PROCESSED_DIR = ROOT / "data" / "research" / "processed"
LEDGER_PATH = ROOT / "data" / "agent" / "hypothesis_ledger.json"
FACTORY_LOG = ROOT / "data" / "agent" / "factory_log.jsonl"

_log = C.make_logger("research_factory")

# Registry of runnable validators. Each declares the methodology it provides so the
# gate can be honest, plus where it writes its result JSON. `min_nperm` is re-checked
# against the actual result to catch a validator that silently ran fewer shuffles.
VALIDATOR_REGISTRY = {
    "big_move": {
        "script": "scripts/validate_big_move.py",
        "args": ["--nperm", "10000", "--seed", "42"],
        "result": "data/research/big_move_validation.json",
        "methodology": {"permutation": True, "walk_forward": True, "both_sides": True},
        "parser": "big_move",
        "timeout": 600,
    },
}


def _route(candidate: dict) -> str | None:
    """Pick a registered validator for a candidate, or None if none fits. Routing is
    keyword-based on hypothesis text + test_spec; deliberately conservative — we would
    rather BLOCK than run the wrong test."""
    text = (candidate.get("hypothesis", "") + " " +
            json.dumps(candidate.get("test_spec", {}))).lower()
    if any(k in text for k in ("big-move", "big move", "directional day", "institutional move")):
        return "big_move"
    return None


# ── Result parsers (one per registered validator) ─────────────────────────────

def _parse_big_move(result_path: Path) -> dict:
    data = json.loads(result_path.read_text())
    gates = data.get("gates", {})
    return {
        "gates_all_pass": bool(gates) and all(gates.values()),
        "nperm": int(data.get("params", {}).get("nperm", 0)),
        "perm_p": data.get("portfolio", {}).get("pooled_permutation_p"),
        "oos_sharpe": data.get("portfolio", {}).get("oos_sharpe"),
        "walkforward": data.get("portfolio", {}).get("walkforward_verdict"),
    }


_PARSERS = {"big_move": _parse_big_move}


# ── Methodology gate (constraint #2) ──────────────────────────────────────────

def _methodology_ok(spec: dict, parsed: dict, cfg: dict) -> tuple[bool, str]:
    """Valid only with permutation (≥ min shuffles), walk-forward, and both-sides."""
    m = cfg.get("methodology", {})
    min_perm = m.get("min_permutations", 10000)
    decl = spec.get("methodology", {})
    if not decl.get("permutation"):
        return False, "no permutation test"
    if not decl.get("walk_forward") and m.get("require_walk_forward", True):
        return False, "no walk-forward"
    if not decl.get("both_sides") and m.get("require_both_sides", True):
        return False, "single-sided test"
    if parsed.get("nperm", 0) < min_perm:
        return False, f"permutations {parsed.get('nperm')} < required {min_perm}"
    return True, "complete"


def _verdict(methodology_ok: bool, gates_all_pass: bool) -> str:
    if not methodology_ok:
        return "INVALID"
    return "VALID_EDGE" if gates_all_pass else "NOT_SIGNIFICANT"


# ── Pre-registration ──────────────────────────────────────────────────────────

def _preregister(hyp_id: str, validator: str, spec: dict) -> Path:
    """Freeze the test spec to disk BEFORE running. The factory executes against this
    frozen file — no parameter tuning mid-run (constraint #2)."""
    PREREGISTER_DIR.mkdir(parents=True, exist_ok=True)
    path = PREREGISTER_DIR / f"{hyp_id}.json"
    C.atomic_write_json(path, {
        "hypothesis_id": hyp_id,
        "validator": validator,
        "frozen_at": C.now_iso(),
        "script": spec["script"],
        "args": spec["args"],
        "methodology": spec["methodology"],
    })
    return path


# ── Run one registered validator end-to-end ───────────────────────────────────

def _run_validator(validator: str) -> dict:
    """Execute the validator subprocess and parse its result. Returns a dict with
    {ran, parsed, error}. Exit code 2 is a run error (not a verdict)."""
    spec = VALIDATOR_REGISTRY[validator]
    script = ROOT / spec["script"]
    if not script.exists():
        return {"ran": False, "error": f"validator script missing: {script}"}
    cmd = ["/opt/homebrew/bin/python3", str(script), *spec["args"]]
    _log(f"  running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              timeout=spec.get("timeout", 600))
    except subprocess.TimeoutExpired:
        return {"ran": False, "error": "validator timed out"}
    if proc.returncode == 2:
        return {"ran": False, "error": f"validator errored (exit 2): {proc.stderr[-300:]}"}
    result_path = ROOT / spec["result"]
    if not result_path.exists():
        return {"ran": False, "error": f"no result file at {result_path}"}
    parsed = _PARSERS[spec["parser"]](result_path)
    return {"ran": True, "parsed": parsed, "result_path": str(result_path)}


# ── Ledger write (live only) — append-only, backed up, tagged ─────────────────

def _append_to_ledger(entry: dict) -> str:
    """Back up the curated ledger, then append ONE auto-tagged entry. Never modifies
    or deletes an existing entry. Atomic. Returns the backup path."""
    ledger = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists() else []
    bak = LEDGER_PATH.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER_PATH, bak)
    ledger.append(entry)
    C.atomic_write_json(LEDGER_PATH, ledger)
    return str(bak)


def _ledger_entry(candidate: dict, validator: str, verdict: str, parsed: dict) -> dict:
    return {
        "id": candidate["id"],
        "name": candidate["hypothesis"][:120],
        "status": verdict,
        "date_tested": C.now_iso()[:10],
        "source": "auto_factory",          # distinguishes autonomous entries
        "validator": validator,
        "mechanism": candidate.get("mechanism", ""),
        "result": {k: parsed.get(k) for k in ("perm_p", "oos_sharpe", "walkforward", "nperm")},
        "auto_generated": True,
    }


# ── Processing ────────────────────────────────────────────────────────────────

def _process_one(candidate: dict, cfg: dict, live: bool) -> dict:
    """Process a single queued candidate end-to-end. Returns a result record."""
    hyp_id = candidate.get("id", "unknown")
    validator = _route(candidate)
    base = {"hypothesis_id": hyp_id, "detector": candidate.get("detector"),
            "timestamp": C.now_iso(), "live": live}

    if validator is None:
        _log(f"  {hyp_id}: BLOCKED_NO_VALIDATOR (no registered validator matches)")
        return {**base, "verdict": "BLOCKED_NO_VALIDATOR",
                "note": "no registered validator; needs one built before it can be tested"}

    spec = VALIDATOR_REGISTRY[validator]
    prereg = _preregister(hyp_id, validator, spec)
    _log(f"  {hyp_id}: pre-registered → {prereg.name}")
    run = _run_validator(validator)
    if not run["ran"]:
        _log(f"  {hyp_id}: ERROR — {run['error']}")
        return {**base, "validator": validator, "verdict": "ERROR", "note": run["error"]}

    parsed = run["parsed"]
    ok, reason = _methodology_ok(spec, parsed, cfg)
    verdict = _verdict(ok, parsed["gates_all_pass"])
    _log(f"  {hyp_id}: {verdict} (methodology {reason}; gates_pass={parsed['gates_all_pass']}, "
         f"perm_p={parsed['perm_p']}, oos_sharpe={parsed['oos_sharpe']})")

    record = {**base, "validator": validator, "verdict": verdict,
              "methodology": reason, "parsed": parsed, "result_path": run["result_path"]}

    if live:
        if verdict in ("VALID_EDGE", "NOT_SIGNIFICANT", "INVALID"):
            bak = _append_to_ledger(_ledger_entry(candidate, validator, verdict, parsed))
            record["ledger_backup"] = bak
            _log(f"  {hyp_id}: appended to ledger (backup {Path(bak).name})")
        if verdict == "VALID_EDGE":
            C.write_message("HIGH",
                            f"Research factory found VALID_EDGE: {candidate['hypothesis'][:100]} "
                            "— needs human review (approve_edge.py). NOT auto-deployed.",
                            source="research_factory", tag="EDGE")
    return record


def run(dry_run: bool = True, validate_name: str | None = None) -> dict:
    cfg = C.load_config()
    live = (not dry_run) and C.is_live()
    factory_cap = cfg.get("budget", {}).get("factory_daily_usd_cap", 5.0)
    spent = C.daily_spend_usd()
    if spent > factory_cap:
        _log(f"BUDGET STOP — daily spend ${spent} > factory cap ${factory_cap}")
        return {"status": "BUDGET_EXCEEDED", "spent": spent}

    _log(f"factory start — live={live} (config.live={C.is_live()}, dry_run={dry_run}) "
         f"| spent today ${spent}")

    results: list[dict] = []

    # Direct mode: validate one registered hypothesis end-to-end (operator + proof path).
    if validate_name:
        if validate_name not in VALIDATOR_REGISTRY:
            raise ValueError(f"unknown validator '{validate_name}'; "
                             f"registered: {list(VALIDATOR_REGISTRY)}")
        spec = VALIDATOR_REGISTRY[validate_name]
        synthetic = {
            "id": f"REG-{validate_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            "hypothesis": f"Registered validator '{validate_name}' edge check (big move classifier).",
            "test_spec": {"validator": validate_name},
            "mechanism": "Standing system-validation hypothesis run directly via the factory.",
            "detector": "registered_validator",
        }
        results.append(_process_one(synthetic, cfg, live))
    else:
        # Queue mode: process QUEUED candidates.
        if not QUEUE_PATH.exists():
            _log("no queue file — nothing to process")
            return {"status": "EMPTY_QUEUE", "processed": 0}
        queued = [json.loads(l) for l in QUEUE_PATH.read_text().splitlines() if l.strip()]
        pending = [c for c in queued if c.get("status") == "QUEUED"]
        _log(f"{len(pending)} QUEUED candidate(s)")
        for c in pending:
            results.append(_process_one(c, cfg, live))

    # Persist verdicts: shadow file always; queue consumption only when live.
    for r in results:
        C.append_jsonl(RESULTS_SHADOW, r)
        C.append_jsonl(FACTORY_LOG, r)

    summary = {
        "timestamp": C.now_iso(),
        "live": live,
        "processed": len(results),
        "verdicts": {v: sum(1 for r in results if r["verdict"] == v)
                     for v in sorted({r["verdict"] for r in results})},
    }
    _log(f"factory done — {summary['verdicts']}")
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Research Factory — run validators on queued hypotheses.")
    parser.add_argument("--validate", metavar="NAME", help="run one registered validator end-to-end")
    parser.add_argument("--live", action="store_true",
                        help="attempt live actions (still requires config.live=true)")
    args = parser.parse_args()
    print(json.dumps(run(dry_run=not args.live, validate_name=args.validate), indent=2))


if __name__ == "__main__":
    main()
