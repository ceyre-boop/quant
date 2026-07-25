#!/usr/bin/env python3
"""Daily Intelligence Pipeline (DIP) — phased orchestrator.

RECONCILIATION NOTE (read this): there was no `scripts/daily_intelligence_pipeline.py` before this
file. The DIP's compute half already lives in `scripts/dip_daily.sh` (harvest → XGBoost retrain), and
the briefing half in `scripts/morning_market_briefing.py::build()` (collectors → synthesize →
daily_briefing.json + the A2/A3 contracts). This script is the phased Python entry the AlphaZero report
and the Ollama work-order assume: it SEQUENCES those existing pieces into two explicit phases so the
synthesizer is called in PHASE 2, not at Phase-1 data-fetch time. It does not duplicate them.

PHASES
  Phase 1 (--phase 1): fetch data + write raw collector JSONs (market_state, lead_lag, volume_profile,
           news, event_calendar) to data/briefing/. NO SYNTHESIS in Phase 1.
  Phase 2 (--phase 2):
     2a/2b  feature assembly + XGBoost retrain — delegated to the existing dip_daily.sh compute half.
            OFF by default (heavy); enable with --with-retrain. Reconciliation: these steps are the
            harvest+retrain that dip_daily.sh already owns; we do not reimplement them here. NOTE:
            dip_daily.sh is ALSO scheduled standalone at 02:30 ET via com.alta.dip_daily.plist — do
            not pass --with-retrain on the scheduled dip_peak run or harvest+retrain runs twice a
            day for no benefit. Manual re-runs may pass it.
     2c     SYNTHESIS — read the Phase-1 data and produce data/agent/daily_briefing.json via
            morning_market_briefing.build() (Ollama-first three-tier chain; deterministic fallback if
            all model tiers return None). NEVER blocks Phase 2 on a synthesis failure.
     2d     hypothesis batch — sovereign.autonomous.hypothesis_generator.run(), which injects the
            fresh daily_briefing as CONTEXT-ONLY into the candidate batch.
  Phase 3 (--phase 3): DIFFUSION — append today's regime + hypothesis-batch summary into the Obsidian
           brain (data/agent/dip_phase3.json checkpoint). See phase3() docstring for the honest
           reconciliation of what this diffuses vs. what the original work order specified — the
           per-hypothesis Obsidian note writer and similarity-matched wikilinking it asked for do not
           exist (no Library pattern-similarity matcher is built anywhere in this repo); this phase
           diffuses the regime snapshot and batch counts, which are real and available today.

RECONCILIATION SUMMARY (see NEXT.md 2026-07-25 entry for the full spec-to-reality map):
  - data/agent/dip_phase{1,2,3}.json are now written (spec-named, alongside the pre-existing
    data/_dip_pipeline_phase{1,2}_checkpoint.json which stay for backward compatibility).
  - There is no data/ml/ feature-matrix / VADER-sentiment / walk-forward-window layer. The spec's
    "recursive walk-forward XGBoost training" is satisfied by the ALREADY-LIVE, differently-shaped
    continuous_harvester.py + training/retrain_loop.py pair (data/harvest.db, models/xgb_veto.json,
    models/threshold_history.json) via dip_daily.sh. The news-sentiment feature layer (VADER) is
    genuinely not built anywhere — it is not silently faked here.

DISCIPLINE: research/context loop only. No order_send, no MT5/OANDA bridge, no frozen execution-path
file (forex_exit_manager / decide_exit / carry_engine) touched. Fail-loud: each phase writes a
checkpoint on success and an error file on failure.

Usage:
    python3 scripts/daily_intelligence_pipeline.py --phase 1
    python3 scripts/daily_intelligence_pipeline.py --phase 2 [--with-retrain] [--dry-run-hypotheses]
    python3 scripts/daily_intelligence_pipeline.py --phase 3
    python3 scripts/daily_intelligence_pipeline.py --phase all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BRIEF_RAW_DIR = ROOT / "data" / "briefing"
DAILY_BRIEFING_JSON = ROOT / "data" / "agent" / "daily_briefing.json"
CKPT_DIR = ROOT / "data"
AGENT_DIR = ROOT / "data" / "agent"
OBSIDIAN_DIP_LOG = Path.home() / "Obsidian" / "Obsidian" / "Trading" / "System" / "DIP-Daily-Log.md"
GATE_SCAN_PATH = AGENT_DIR / "petrules_gate_scan.json"
GENERATOR_LOG = AGENT_DIR / "generator_log.jsonl"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _spec_checkpoint(phase_num: int, payload: dict) -> None:
    """Write the spec-named checkpoint the DIP work order's Definition of Done checks for
    (data/agent/dip_phaseN.json), alongside the pre-existing data/_dip_pipeline_phaseN_checkpoint.json
    which stays for backward compatibility with anything already reading it."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    p = AGENT_DIR / f"dip_phase{phase_num}.json"
    p.write_text(json.dumps({"date": _today(), "phase": phase_num, "completed_at": _now(), **payload},
                             indent=2, default=str))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint(phase: str, payload: dict) -> None:
    p = CKPT_DIR / f"_dip_pipeline_{phase}_checkpoint.json"
    p.write_text(json.dumps({"phase": phase, "ts": _now(), **payload}, indent=2, default=str))
    err = CKPT_DIR / f"_dip_pipeline_{phase}_error.json"
    if err.exists():
        err.unlink()


def _error(phase: str, msg: str) -> None:
    (CKPT_DIR / f"_dip_pipeline_{phase}_error.json").write_text(
        json.dumps({"phase": phase, "ts": _now(), "error": msg}, indent=2))


def _already_ran_today(phase_num: int) -> bool:
    """Sleep-safety: a phase that already checkpointed today (spec-named file) skips a re-run."""
    p = AGENT_DIR / f"dip_phase{phase_num}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("date") == _today()
    except Exception:
        return False


# ─── Phase 1 — data fetch (NO synthesis) ─────────────────────────────────────────────────────────
def phase1() -> dict:
    """Fetch the five collectors and write their raw JSON. No synthesis here — that is Phase 2's job."""
    if _already_ran_today(1):
        print("[DIP] phase 1 already completed today. Skip.")
        return {"skipped": True}

    from sovereign.briefing import market_data, lead_lag, volume_profile, news_feed, event_calendar

    BRIEF_RAW_DIR.mkdir(parents=True, exist_ok=True)
    collectors = {
        "market_state": market_data.collect,
        "lead_lag_regime": lead_lag.classify,
        "volume_profile": volume_profile.build_all,
        "news_feed": news_feed.fetch,
        "event_calendar": event_calendar.build,
    }
    written, errors = {}, {}
    for name, fn in collectors.items():
        try:
            data = fn()
            (BRIEF_RAW_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
            written[name] = "ok"
        except Exception as e:  # a single collector failing must not sink the phase
            errors[name] = str(e)
    result = {"written": written, "errors": errors, "synthesis_called": False}
    _checkpoint("phase1", result)

    regime = None
    try:
        regime = json.loads((BRIEF_RAW_DIR / "lead_lag_regime.json").read_text()).get("regime")
    except Exception:
        pass
    macro_events_today = []
    try:
        macro_events_today = json.loads((BRIEF_RAW_DIR / "event_calendar.json").read_text())
    except Exception:
        pass
    gate_scan_fresh = False
    try:
        gate_scan_fresh = json.loads(GATE_SCAN_PATH.read_text()).get("scanned_at", "").startswith(_today())
    except Exception:
        pass
    if errors:
        (AGENT_DIR / "dip_phase1_error.json").write_text(json.dumps(
            {"date": _today(), "ts": _now(), "errors": errors}, indent=2))
    else:
        err = AGENT_DIR / "dip_phase1_error.json"
        if err.exists():
            err.unlink()
    _spec_checkpoint(1, {"regime": regime, "macro_events_today": macro_events_today,
                          "gate_scan_fresh": gate_scan_fresh, "written": written, "errors": errors})

    print(f"[DIP] phase 1 — wrote {len(written)}/{len(collectors)} raw collector JSONs to data/briefing/ "
          f"(no synthesis){' — errors: ' + ','.join(errors) if errors else ''}")
    return result


# ─── Phase 2 — assemble + synthesize + hypotheses ────────────────────────────────────────────────
def _phase2_retrain() -> dict:
    """2a/2b — delegate feature assembly + XGBoost retrain to the existing dip_daily.sh compute half."""
    sh = ROOT / "scripts" / "dip_daily.sh"
    if not sh.exists():
        return {"ran": False, "note": "scripts/dip_daily.sh absent — retrain skipped"}
    try:
        r = subprocess.run(["bash", str(sh)], cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
        return {"ran": True, "returncode": r.returncode, "tail": (r.stdout or "")[-400:]}
    except Exception as e:
        return {"ran": False, "error": str(e)}


def phase2(with_retrain: bool = False, dry_run_hypotheses: bool = False) -> dict:
    """2a/2b (optional retrain) → 2c synthesis → 2d hypothesis batch. Never blocks on synthesis failure."""
    if _already_ran_today(2):
        print("[DIP] phase 2 already completed today. Skip.")
        return {"skipped": True}

    result: dict = {}

    # Training gate (CLAUDE.md Art. 6 / RISK_CONSTITUTION.md): training may only run against a
    # ledger with at least one CONFIRMED verdict. dip_daily.sh / retrain_loop.py do not implement
    # this gate themselves (they are the pre-existing, unmodified compute half — freeze discipline
    # keeps them untouched), so the orchestrator enforces it before delegating.
    gate_open = False
    try:
        ledger = json.loads((AGENT_DIR / "hypothesis_ledger.json").read_text())
        gate_open = any(str(e.get("status", "")).upper() == "CONFIRMED" for e in ledger)
    except Exception as e:
        result["training_gate_error"] = str(e)

    # 2a/2b — feature assembly + XGBoost (delegated, optional).
    if with_retrain and gate_open:
        print("[DIP] phase 2a/2b — feature assembly + XGBoost retrain (dip_daily.sh)")
        result["retrain"] = _phase2_retrain()
    elif with_retrain and not gate_open:
        print("[DIP] phase 2a/2b — BLOCKED: no CONFIRMED ledger entry, training gate closed")
        result["retrain"] = {"ran": False, "note": "training gate closed — no CONFIRMED ledger entry"}
    else:
        result["retrain"] = {"ran": False, "note": "skipped (pass --with-retrain to run harvest+retrain)"}
    result["training_gate_open"] = gate_open

    # 2c — SYNTHESIS. build() reads fresh collectors, runs the Ollama-first three-tier chain, and
    # writes data/agent/daily_briefing.json (with a deterministic narrative if every model tier
    # returns None). It never raises on a synthesis failure, so Phase 2 is never blocked by it.
    print("[DIP] phase 2c — synthesis → data/agent/daily_briefing.json")
    try:
        from scripts import morning_market_briefing as mmb
        briefing = mmb.build()
        result["synthesis"] = {
            "ok": True,
            "synthesis_source": briefing.get("synthesis_source"),
            "directional_bias": briefing.get("directional_bias"),
            "confidence": briefing.get("confidence"),
        }
        print(f"[DIP]   synthesis_source={briefing.get('synthesis_source')} "
              f"bias={briefing.get('directional_bias')} conf={briefing.get('confidence')}")
    except Exception as e:
        # Even a hard failure here must not abort the phase — record it and continue.
        result["synthesis"] = {"ok": False, "error": str(e)}
        _error("phase2_synthesis", str(e))
        print(f"[DIP]   synthesis step errored (continuing): {e}")

    # 2d — hypothesis batch, which injects the fresh daily_briefing as context-only.
    print(f"[DIP] phase 2d — hypothesis batch (dry_run={dry_run_hypotheses})")
    try:
        from sovereign.autonomous import hypothesis_generator as hg
        result["hypotheses"] = hg.run(dry_run=dry_run_hypotheses)
    except Exception as e:
        result["hypotheses"] = {"ok": False, "error": str(e)}
        print(f"[DIP]   hypothesis batch errored (continuing): {e}")

    _checkpoint("phase2", result)

    skipped = ["feature_matrix (data/ml/ layer not built)", "dip_xgb_trainer (not built — see module docstring)"]
    if not with_retrain or not gate_open:
        skipped.append("retrain")
    _spec_checkpoint(2, {
        "training_gate_open": gate_open,
        "retrain": result.get("retrain"),
        "synthesis": result.get("synthesis"),
        "hypotheses_run": (result.get("hypotheses") or {}).get("generated"),
        "oracle_cycles": 0,
        "skipped": skipped,
    })
    return result


# ─── Phase 3 — diffusion ──────────────────────────────────────────────────────────────────────────
def phase3() -> dict:
    """Diffusion. HONEST RECONCILIATION vs. the original work order's Phase 3:

    - 3a Obsidian sync: the spec asked for per-hypothesis notes with similarity-matched wikilinks
      to Library patterns. No pattern-similarity matcher exists anywhere in this repo, so that piece
      is NOT built (not faked). What IS built and diffused here: an append-only dated section in
      ~/Obsidian/Obsidian/Trading/System/DIP-Daily-Log.md with today's regime reading and the
      hypothesis batch count/detector breakdown — both real, both available from files Phase 1/2
      already wrote (data/briefing/lead_lag_regime.json, data/agent/generator_log.jsonl).
    - 3b Ledger stamp: the spec assumed a per-hypothesis jsonl ledger with a `last_batch_run` field.
      The real ledger (data/agent/hypothesis_ledger.json) holds only ADJUDICATED verdicts, not batch
      candidates — writing a synthetic `last_batch_run` field into it would mix batch bookkeeping
      into the adjudication record. RE-SCOPED: generator_log.jsonl already append-stamps every batch
      run (timestamp, reps, ledger count, detectors) — that IS the batch stamp. This phase reads the
      latest entry rather than writing a second, competing stamp.
    - 3c Calibration append: only meaningful once petrules_gate_scan.json is fresh with a real
      tier3_plus count. As of this build the gate scan has never run live on Colin's Mac
      (scanned_at error placeholder) — so this step honestly opens zero calibration rows today,
      exactly as the spec's own "if tier3_plus > 0 and fresh" condition dictates.
    """
    if _already_ran_today(3):
        print("[DIP] phase 3 already completed today. Skip.")
        return {"skipped": True}

    result: dict = {"obsidian_notes_written": 0, "ledger_entries_stamped": 0, "calibration_rows_opened": 0}
    try:
        regime = None
        try:
            regime = json.loads((BRIEF_RAW_DIR / "lead_lag_regime.json").read_text()).get("regime")
        except Exception:
            pass

        latest_batch = None
        if GENERATOR_LOG.exists():
            lines = [l for l in GENERATOR_LOG.read_text().splitlines() if l.strip()]
            if lines:
                latest_batch = json.loads(lines[-1])

        tier3_plus = 0
        gate_fresh = False
        try:
            gs = json.loads(GATE_SCAN_PATH.read_text())
            gate_fresh = str(gs.get("scanned_at", "")).startswith(_today())
            tier3_plus = gs.get("tier3_plus", 0) if gate_fresh else 0
        except Exception:
            pass

        OBSIDIAN_DIP_LOG.parent.mkdir(parents=True, exist_ok=True)
        section = (
            f"\n## {_today()} DIP Diffusion\n"
            f"Regime: {regime}\n"
            f"Hypothesis batch: {(latest_batch or {}).get('generated', 0)} generated "
            f"({(latest_batch or {}).get('detectors', {})})\n"
            f"Gate scan: {'fresh, ' + str(tier3_plus) + ' Tier3+' if gate_fresh else 'not fresh — skipped'}\n"
        )
        with OBSIDIAN_DIP_LOG.open("a") as f:
            f.write(section)
        result["obsidian_notes_written"] = 1

        result["ledger_entries_stamped"] = 1 if latest_batch else 0
        result["ledger_stamp_source"] = "data/agent/generator_log.jsonl (see phase3() docstring — " \
            "no second stamp written to hypothesis_ledger.json)"

        if gate_fresh and tier3_plus > 0:
            calib = AGENT_DIR / "gate_calibration.jsonl"
            with calib.open("a") as f:
                f.write(json.dumps({"date": _today(), "tier3_plus": tier3_plus,
                                     "outcome_logged": False, "opened_at": _now()}) + "\n")
            result["calibration_rows_opened"] = 1

        _spec_checkpoint(3, result)
        print(f"[DIP] phase 3 — diffused regime={regime} batch={((latest_batch or {}).get('generated', 0))} "
              f"to Obsidian; calibration_rows_opened={result['calibration_rows_opened']}")
    except Exception as e:
        (AGENT_DIR / "dip_phase3_error.json").write_text(json.dumps(
            {"date": _today(), "ts": _now(), "error": str(e)}, indent=2))
        print(f"[DIP] phase 3 errored: {e}")
        raise
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily Intelligence Pipeline — phased orchestrator")
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], required=True)
    ap.add_argument("--with-retrain", action="store_true", help="Phase 2: run harvest+XGBoost retrain (heavy)")
    ap.add_argument("--dry-run-hypotheses", action="store_true", help="Phase 2: generate hypotheses without writing the queue")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1()
    if args.phase in ("2", "all"):
        phase2(with_retrain=args.with_retrain, dry_run_hypotheses=args.dry_run_hypotheses)
    if args.phase in ("3", "all"):
        phase3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
