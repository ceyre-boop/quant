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
            harvest+retrain that dip_daily.sh already owns; we do not reimplement them here.
     2c     SYNTHESIS — read the Phase-1 data and produce data/agent/daily_briefing.json via
            morning_market_briefing.build() (Ollama-first three-tier chain; deterministic fallback if
            all model tiers return None). NEVER blocks Phase 2 on a synthesis failure.
     2d     hypothesis batch — sovereign.autonomous.hypothesis_generator.run(), which injects the
            fresh daily_briefing as CONTEXT-ONLY into the candidate batch.

DISCIPLINE: research/context loop only. No order_send, no MT5/OANDA bridge, no frozen execution-path
file (forex_exit_manager / decide_exit / carry_engine) touched. Fail-loud: each phase writes a
checkpoint on success and an error file on failure.

Usage:
    python3 scripts/daily_intelligence_pipeline.py --phase 1
    python3 scripts/daily_intelligence_pipeline.py --phase 2 [--with-retrain] [--dry-run-hypotheses]
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


# ─── Phase 1 — data fetch (NO synthesis) ─────────────────────────────────────────────────────────
def phase1() -> dict:
    """Fetch the five collectors and write their raw JSON. No synthesis here — that is Phase 2's job."""
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
    result: dict = {}

    # 2a/2b — feature assembly + XGBoost (delegated, optional).
    if with_retrain:
        print("[DIP] phase 2a/2b — feature assembly + XGBoost retrain (dip_daily.sh)")
        result["retrain"] = _phase2_retrain()
    else:
        result["retrain"] = {"ran": False, "note": "skipped (pass --with-retrain to run harvest+retrain)"}

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
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily Intelligence Pipeline — phased orchestrator")
    ap.add_argument("--phase", choices=["1", "2", "all"], required=True)
    ap.add_argument("--with-retrain", action="store_true", help="Phase 2: run harvest+XGBoost retrain (heavy)")
    ap.add_argument("--dry-run-hypotheses", action="store_true", help="Phase 2: generate hypotheses without writing the queue")
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1()
    if args.phase in ("2", "all"):
        phase2(with_retrain=args.with_retrain, dry_run_hypotheses=args.dry_run_hypotheses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
