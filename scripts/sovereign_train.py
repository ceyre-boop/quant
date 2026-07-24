#!/usr/bin/env python3
"""Sovereign self-play training runner (spec: research/SELF_PLAY_TRAINING_ARCHITECTURE.md §7).

    python3 scripts/sovereign_train.py --watch

Runs the Phase 0-5 self-play pipeline. Training a trading policy is model training,
gated behind TICK-024 + a CONFIRMED HYP-071-net ledger stamp (RISK_CONSTITUTION
Art. 6). The IGNITION GATE (sovereign/training/gate.py) is evaluated and printed
LOUDLY at startup. While the gate is CLOSED (default), every run is SCAFFOLD/DRY:
the full pipeline structure executes, a checkpoint + a training_log entry are
written, but NO production policy is trained and NO parameter update is written.

This is a simulation/training loop. It NEVER places a live trade and never touches
the MT5 / OANDA execution bridge.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.training import gate as gate_mod            # noqa: E402
from sovereign.training import policy_rollout              # noqa: E402
from sovereign.training import policy_updater              # noqa: E402
from sovereign.training import director as director_mod    # noqa: E402
from sovereign.training import snapshots as snapshots_mod  # noqa: E402
from sovereign.training.value_scorer import ValueScorer, GrossReturnError  # noqa: E402

CONFIG = ROOT / "config" / "training.yml"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _stamp(watch: bool, msg: str) -> None:
    if watch:
        print(f"[{_now()}] {msg}", flush=True)


def _load_cfg() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"training config not found: {CONFIG}")
    return yaml.safe_load(CONFIG.read_text()) or {}


def run(watch: bool = False) -> dict:
    cfg = _load_cfg()
    started = datetime.now(timezone.utc)

    if watch:
        print(f"SOVEREIGN TRAINING RUN — {started.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ── IGNITION GATE — evaluated and printed loudly BEFORE any work ──────────
    status = gate_mod.evaluate_gate(CONFIG)
    print(gate_mod.render_gate_banner(status), flush=True)
    gate_open = status.open

    # ── PHASE 0: Data load ────────────────────────────────────────────────────
    _stamp(watch, "PHASE 0: Loading data window (DRY placeholder)..." if not gate_open
           else "PHASE 0: Loading 252 days across 4 pairs...")
    pairs = cfg.get("rollout", {}).get("pairs", [])
    _stamp(watch, f"PHASE 0: Done. Pairs: {', '.join(pairs)}")

    # ── PHASE 1: Policy rollout ───────────────────────────────────────────────
    _stamp(watch, "PHASE 1: Rolling out current policy...")
    rollout = policy_rollout.rollout_policy(CONFIG, gate_open=gate_open)
    _stamp(watch, f"PHASE 1: {len(rollout)} simulated trades "
                  f"({'DRY' if rollout.dry else 'LIVE'}). "
                  f"Distribution: {rollout.pair_distribution}")

    # ── PHASE 2: Value scoring (NET-GUARDED) ──────────────────────────────────
    _stamp(watch, "PHASE 2: Scoring positions via HYP-071 board...")
    scorer = ValueScorer(CONFIG)
    scoring_skipped = False
    net_r = np.array([t["gross_return_r"] for t in rollout.trades])  # dry: gross
    try:
        scorer.load_board()
        exp_r = np.array([t["expected_r"] for t in rollout.trades])
        value_scores = scorer.score_batch(net_r, exp_r)
        _stamp(watch, f"PHASE 2: Done. mean={value_scores.mean():+.3f} "
                      f"p25={np.percentile(value_scores,25):+.3f} "
                      f"p75={np.percentile(value_scores,75):+.3f}")
    except GrossReturnError as exc:
        # Expected while the board is gross. The guard fired — pipeline continues
        # DRY with placeholder scores that are NEVER used to train.
        scoring_skipped = True
        _stamp(watch, f"PHASE 2: NET-RETURN GUARD FIRED — scoring refused.")
        _stamp(watch, f"           {exc}")
        rng = np.random.default_rng(0)
        value_scores = rng.normal(0.0, 0.3, size=len(rollout))
        _stamp(watch, "PHASE 2: proceeding DRY with placeholder scores (NOT for training).")

    # ── PHASE 3: Policy update ────────────────────────────────────────────────
    _stamp(watch, "PHASE 3: Training next policy on top-quartile trades...")
    update = policy_updater.refit_policy(value_scores, net_r, gate_open=gate_open, config_path=CONFIG)
    _stamp(watch, f"PHASE 3: {update.note} "
                  f"(top n={update.n_top}, bottom n={update.n_bottom}, "
                  f"threshold={update.threshold:+.3f})")
    if update.placebo is not None:
        _stamp(watch, f"PHASE 3: placebo margin={update.placebo.margin:+.3f} "
                      f"(min {update.placebo.margin_min:.3f}) eligible={update.placebo.eligible}")

    # ── PHASE 4: Director review (human-gated) ────────────────────────────────
    _stamp(watch, "PHASE 4: Director review...")
    baseline = cfg.get("policy_params", {})
    # DRY: nothing was refit, so proposed == baseline (no parameter movement).
    proposed = dict(baseline)
    report = director_mod.review(baseline, proposed, regime_fraction=0.60,
                                  placebo=update.placebo, config_path=CONFIG)
    if watch:
        print(director_mod.render_report(report), flush=True)

    # Enforce: never auto-approve, never commit while gate closed, and never reach
    # the human-approval step at all if the mandatory placebo control failed
    # (report.all_pass already folds in placebo_ok — see director.py).
    committed = False
    if gate_open and report.all_pass and not cfg.get("director", {}).get("auto_approve", False):
        _stamp(watch, "PHASE 4: [Waiting for human confirmation → Enter to commit, Ctrl-C to abort]")
        # Human gate would block here in an interactive live run.

    # ── PHASE 5: Checkpoint ───────────────────────────────────────────────────
    _stamp(watch, "PHASE 5: Writing checkpoint + training_log entry...")
    entry = _write_checkpoint_and_log(cfg, status, rollout, update, report,
                                       scoring_skipped, committed, started,
                                       baseline, proposed)
    _stamp(watch, f"PHASE 5: Done. verdict={entry['verdict']} mode={entry['mode']}")
    return entry


def _write_checkpoint_and_log(cfg, status, rollout, update, report,
                              scoring_skipped, committed, started,
                              baseline, proposed) -> dict:
    ckpt_dir = ROOT / cfg.get("paths", {}).get("checkpoint_dir", "data/training")
    log_path = ROOT / cfg.get("paths", {}).get("training_log", "logs/training_log.jsonl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ts = started.strftime("%Y%m%dT%H%M%SZ")
    verdict = "SCAFFOLD_DRY" if not status.open else ("COMMITTED" if committed else "PENDING_HUMAN")
    entry = {
        "timestamp": started.isoformat(),
        "mode": status.mode,
        "gate_open": status.open,
        "gate_checks": status.checks,
        "gate_reasons": status.reasons,
        "verdict": verdict,
        "rollout_trades": len(rollout),
        "rollout_dry": rollout.dry,
        "scoring_net_guard_fired": scoring_skipped,
        "policy_update": {
            "dry": update.dry, "n_top": update.n_top,
            "n_bottom": update.n_bottom, "threshold": update.threshold,
        },
        "placebo_control": (
            {
                "eligible": update.placebo.eligible,
                "real_metric": update.placebo.real_metric,
                "placebo_metric": update.placebo.placebo_metric,
                "margin": update.placebo.margin,
                "margin_min": update.placebo.margin_min,
                "significant": update.placebo.significant,
                "composition_ok": update.placebo.composition_ok,
                "n_folds": update.placebo.n_folds,
                "seed": update.placebo.seed,
                "reason": update.placebo.reason,
            } if update.placebo is not None else None
        ),
        "director": {
            "all_pass": report.all_pass,
            "recommendation": report.recommendation,
            "flags": report.flags,
            "placebo_ok": report.placebo_ok,
            "placebo_margin": report.placebo_margin,
            "params_before": {d.name: d.old for d in report.diffs},
            "params_after": {d.name: d.new for d in report.diffs},
        },
        "elo_estimate": None,   # populated only on a real committed cycle
        "committed": committed,
    }

    # Checkpoint file (large artifacts would live under the gitignored data/training/).
    ckpt_path = ckpt_dir / f"cycle_{ts}.json"
    ckpt_path.write_text(json.dumps(entry, indent=2))

    # Append-only training log.
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

    # Snapshot the policy-param state for undo/restore (data/training/snapshots/,
    # see sovereign/training/snapshots.py). While the gate is closed baseline ==
    # proposed by construction, so this records a legitimate no-op state.
    snapshots_mod.record_cycle(
        params_before=baseline,
        params_after=proposed,
        cycle_ref=ckpt_path.name,
        committed=committed,
        timestamp=ts,
    )
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="Sovereign self-play training runner")
    ap.add_argument("--watch", action="store_true", help="live phased stdout")
    args = ap.parse_args()
    entry = run(watch=args.watch)
    if not args.watch:
        print(json.dumps({"verdict": entry["verdict"], "mode": entry["mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
