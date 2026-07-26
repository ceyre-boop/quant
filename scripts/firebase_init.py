#!/usr/bin/env python3
"""
Firebase RTDB Schema Initializer — Alta Investments / Sovereign Trading Intelligence
scripts/firebase_init.py

Populates (or refreshes) every node the system uses.
Safe to re-run: uses set() only on the schema-defining stub nodes;
live signal/history paths are untouched (they live under keys that
don't collide with anything written here).

Usage:
    python3 scripts/firebase_init.py [--dry-run]

Requires:
    pip install firebase-admin

Credential:
    config/firebase_service_account.json (service account with RTDB write)

Database:
    https://clawd-trading-7b8de-default-rtdb.firebaseio.com/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SA_PATH = ROOT / "config" / "firebase_service_account.json"
DB_URL = "https://clawd-trading-7b8de-default-rtdb.firebaseio.com/"
NOW = datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Each key is a RTDB path.  Values are the stub/default dicts.
# Existing data at the path is OVERWRITTEN for structural nodes; live
# signal / history entries are separate paths and won't be touched.
# ---------------------------------------------------------------------------

CARRY_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "GBPJPY"]
FOREX_PAIRS = CARRY_PAIRS + ["AUDNZD"]  # AUDNZD excluded per HYP-045 but still tracked
ICT_SYMBOLS = ["NAS100", "US30", "SPX500", "XAUUSD"]
ALL_SYMBOLS  = CARRY_PAIRS + ICT_SYMBOLS

# ---- helpers ----------------------------------------------------------------

def _pair_stubs(pairs, template_fn):
    return {p: template_fn(p) for p in pairs}


def _ts() -> str:
    return NOW


# ============================================================================
# NODE DEFINITIONS
# ============================================================================

def build_schema() -> dict[str, dict]:
    """Return {rtdb_path: data_dict} for every node to initialise."""

    nodes: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # system/  — health, versions, models, gates, watchdog
    # -------------------------------------------------------------------------

    nodes["system/meta"] = {
        "version": "v015",
        "branch": "sovereign-v2",
        "last_schema_init": _ts(),
        "shadow_mode": True,      # forex_exit_manager.py SHADOW_MODE flag
        "live_pairs": CARRY_PAIRS,
        "excluded_pairs": ["AUDNZD"],
        "exclusion_reason": "HYP-045: both legs RBA-driven, OOS Sharpe -0.879",
    }

    nodes["system/health"] = {
        "status": "healthy",
        "components": {
            "oracle": "unknown",
            "alphazero_trainer": "unknown",
            "stockfish_board": "unknown",
            "carry_engine": "unknown",
            "mt5_bridge": "unknown",
            "firebase_broadcaster": "unknown",
            "ict_pipeline": "unknown",
            "petroulas_gate": "unknown",
        },
        "updated_at": _ts(),
        "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
    }

    nodes["system/models"] = {
        "bias":    {"version": "unknown", "status": "stale", "updated_at": _ts()},
        "risk":    {"version": "unknown", "status": "stale", "updated_at": _ts()},
        "game":    {"version": "unknown", "status": "stale", "updated_at": _ts()},
        "xgboost": {"version": "unknown", "status": "stale", "updated_at": _ts(),
                    "note": "dip_daily retrain loop; gate: TICK-057 (not yet wired through training gate)"},
        "alphazero_policy": {
            "version": "none",
            "status": "not_trained",
            "updated_at": _ts(),
            "gate_open": False,
            "gate_blockers": ["TICK-024 not landed", "HYP-071 not confirmed (METRIC_ARTIFACT)",
                              "gross_R_caveat", "HYP-071 revival guard"],
        },
    }

    # Regime per live pair
    for sym in CARRY_PAIRS:
        nodes[f"system/regime/{sym}"] = {
            "regime": "UNKNOWN",
            "composite_score": 0.0,
            "vix_level": None,
            "term_structure": None,
            "updated_at": _ts(),
        }

    nodes["system/watchdog"] = {
        "plist_com_alta_dip_daily": {
            "status": "UNLOADED",
            "note": "Do NOT launchctl load until TICK-057 lands (wires retrain through training gate)",
            "ticket": "TICK-057",
        },
        "plist_com_alta_oracle": {
            "status": "unknown",
            "last_heartbeat": None,
        },
        "last_check": _ts(),
    }

    # -------------------------------------------------------------------------
    # signals/  — already used by broadcaster; add schema stubs for new pairs
    # -------------------------------------------------------------------------

    for sym in ALL_SYMBOLS:
        nodes[f"signals/{sym}/meta"] = {
            "symbol": sym,
            "active": sym in CARRY_PAIRS,
            "layer": "carry" if sym in CARRY_PAIRS else "ict",
            "updated_at": _ts(),
        }

    # -------------------------------------------------------------------------
    # session/  — controls, positions, account
    # -------------------------------------------------------------------------

    nodes["session/controls"] = {
        "trading_enabled": False,
        "shadow_mode": True,
        "daily_loss_pct": 0.0,
        "open_positions": 0,
        "hard_logic_status": "ok",
        "manual_override": False,
        "updated_at": _ts(),
    }

    nodes["session/meta"] = {
        "session_date": datetime.now(timezone.utc).date().isoformat(),
        "execution_time_et": "09:35",
        "prop_firm_active": False,
        "prop_firm_platform": None,
        "mt5_demo_connected": False,
        "fomc_window_active": False,
        "updated_at": _ts(),
    }

    for sym in CARRY_PAIRS:
        nodes[f"session/positions/{sym}"] = {
            "trade_id": None,
            "symbol": sym,
            "direction": "FLAT",
            "entry_price": None,
            "position_size": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "current_price": None,
            "unrealized_pnl": None,
            "status": "FLAT",
            "opened_at": None,
            "updated_at": _ts(),
        }

    # -------------------------------------------------------------------------
    # account/  — equity and PnL
    # -------------------------------------------------------------------------

    nodes["account/equity"] = {
        "equity": None,
        "balance": None,
        "margin_used": None,
        "margin_available": None,
        "open_positions": 0,
        "updated_at": _ts(),
    }

    nodes["account/pnl"] = {
        "daily_pnl": 0.0,
        "daily_loss_pct": 0.0,
        "daily_loss_cap_pct": 2.0,   # RISK_CONSTITUTION cap
        "total_r": None,
        "updated_at": _ts(),
    }

    nodes["account/prop_firm"] = {
        "platform": None,              # e.g. "Lucid", "MyFundedFutures"
        "account_id": None,
        "trade_mode": "DEMO",          # ACCOUNT_TRADE_MODE_DEMO enforced by mt5_bridge.py
        "max_daily_loss_pct": 5.0,     # typical prop firm EOD DD limit
        "consistency_rule": False,      # Lucid/MFF have no consistency rule
        "fomc_blackout_active": False,
        "fomc_blackout_ends": None,
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # carry/  — E7 CONFIRMED: 4-pair portfolio
    # -------------------------------------------------------------------------

    nodes["carry/meta"] = {
        "status": "CONFIRMED",
        "hypothesis": "E7",
        "verdict_date": "2026-06-02",
        "oos_sharpe": 1.25,
        "pairs": CARRY_PAIRS,
        "risk_per_pair_pct": 0.3,
        "max_pairs_active": 5,
        "updated_at": _ts(),
    }

    nodes["carry/rate_differentials"] = {
        p: {
            "base_rate": None,
            "quote_rate": None,
            "differential_bps": None,
            "carry_direction": None,
            "last_updated": None,
        }
        for p in CARRY_PAIRS
    }

    nodes["carry/positions"] = {
        p: {
            "active": False,
            "direction": "FLAT",
            "size_lots": None,
            "entry_price": None,
            "carry_pnl_r": 0.0,
            "opened_at": None,
        }
        for p in CARRY_PAIRS
    }

    nodes["carry/cot_gate"] = {
        "last_cot_date": None,
        "crowded_pairs": [],
        "halve_size_pairs": [],     # Tenet 6: when everyone in the same trade → halve
        "gate_clear": True,
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # sovereign/  — intelligence layer
    # -------------------------------------------------------------------------

    nodes["sovereign/training_gate"] = {
        "open": False,
        "blockers": [
            "TICK-024 not landed (swap cost calibration; median 9x understatement)",
            "HYP-071 METRIC_ARTIFACT (adjudicated 2026-06-30; revival requires fresh prereg + new CONFIRMED verdict)",
            "gross_R_caveat unresolved",
            "HYP-071 revival guard active",
        ],
        "planned_unlock": "2026-07-28 post-FOMC",
        "unlock_sequence": "plans/JULY28_UNLOCK_PROMPT.md",
        "updated_at": _ts(),
    }

    nodes["sovereign/decision_logger"] = {
        "total_logged": 0,
        "open_outcomes": 0,      # decisions without update_outcome() yet
        "last_log": None,
        "last_outcome": None,
        "oracle_degraded": False,  # true if open_outcomes > 0 for >24h
        "updated_at": _ts(),
    }

    nodes["sovereign/conviction_sizing"] = {
        "note": "All sizing goes through conviction-based pipeline; no flat sizes",
        "config_path": "config/parameters.yml",
        "f_max_ceiling_active": True,
        "per_trade_risk_cap_pct": 2.0,
        "portfolio_risk_cap_pct": 8.0,
        "carry_base_notional_pct": [15, 20],
        "macro_swing_notional_pct": [40, 50],
        "high_conviction_notional_pct": [10, 20],
        "reserve_pct": 20,
    }

    # -------------------------------------------------------------------------
    # alphazero/  — self-play training architecture
    # -------------------------------------------------------------------------

    nodes["alphazero/gate"] = {
        "open": False,
        "blockers": [
            "TICK-024 not landed",
            "HYP-071 METRIC_ARTIFACT — requires fresh prereg + new CONFIRMED verdict",
            "gross_R_caveat",
            "HYP-071 revival guard",
        ],
        "last_checked": _ts(),
        "planned_dry_run": "2026-07-28",
    }

    nodes["alphazero/training"] = {
        "cycle_id": None,
        "status": "IDLE",           # IDLE | RUNNING | COMPLETE | ERROR
        "policy_version": None,
        "last_cycle_start": None,
        "last_cycle_end": None,
        "episodes_completed": 0,
        "win_rate": None,
        "avg_reward": None,
        "gate_status_at_start": None,
        "updated_at": _ts(),
    }

    nodes["alphazero/policy"] = {
        "version": None,
        "path": None,               # path to saved model weights
        "trained_on_hypothesis": None,
        "sharpe_at_training": None,
        "deployed": False,
        "deployed_at": None,
        "updated_at": _ts(),
    }

    nodes["alphazero/environment"] = {
        "env_class": "TradingEnv",
        "state_dims": 8,            # sovereign/training/state_space.py locked 8-dim S(t)
        "action_space": ["LONG", "SHORT", "FLAT"],
        "reward_metric": "risk_adjusted_r",
        "swap_cost_calibrated": False,    # blocked on TICK-024
        "updated_at": _ts(),
    }

    nodes["alphazero/selfplay"] = {
        "enabled": False,
        "reason": "gate_closed",
        "games_played": 0,
        "last_game": None,
        "elo_current": None,
        "elo_history": [],
    }

    # -------------------------------------------------------------------------
    # stockfish/  — HYP-071 tabular exit value board
    # -------------------------------------------------------------------------

    nodes["stockfish/hyp071"] = {
        "hypothesis": "HYP-071",
        "status": "METRIC_ARTIFACT",
        "verdict_date": "2026-06-30",
        "flaw": "EXIT_NOW dominance is a forecast-variance artifact, not an edge",
        "prereg_hashes_killed": [
            "c4f29ac387669fc77ac33f1d2570042898d8f81bc0409e1fd0e7d57ba9a41546",
            "3d500bda3249c4615698ce311a7cbad41a35600a23abd2a4ea4526416eac06a4",
            "c1fab80730f1ebf3af7c35e4bbd8fc80e2bafd86419fc0125acc414d806f",
        ],
        "revival_conditions": {
            "requires_fresh_prereg": True,
            "requires_new_confirmed_verdict": True,
            "reuse_old_prereg_fails_gate": True,
            "revival_date": None,
        },
        "updated_at": _ts(),
    }

    nodes["stockfish/exit_board"] = {
        "status": "NOT_READY",
        "reason": "HYP-071 METRIC_ARTIFACT — board not safe to use until revival conditions met",
        "last_computed": None,
        "pairs": {p: {"best_action": None, "value": None, "confidence": None} for p in CARRY_PAIRS},
        "updated_at": _ts(),
    }

    nodes["stockfish/tick_tracker"] = {
        "TICK-024": {
            "title": "Swap cost calibration",
            "status": "STAGED",
            "description": "Median 9x understatement; blocks self-play ignition",
            "planned_apply": "2026-07-28",
            "blocks": ["alphazero_gate", "stockfish_exit_board"],
        },
        "TICK-057": {
            "title": "Wire retrain_loop.py through training gate",
            "status": "FILED",
            "description": "dip_daily plist must not load until retrain gated via CONFIRMED hypothesis",
            "blocks": ["com_alta_dip_daily_plist"],
        },
        "TICK-058": {
            "title": "Petroulas IWM header parse fix",
            "status": "FILED",
            "description": "IWM header not parsing correctly in Petroulas gate",
            "blocks": [],
        },
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # petroulas/  — dual-confirmation gate (XGBoost + Kimi)
    # -------------------------------------------------------------------------

    nodes["petroulas/gate_meta"] = {
        "description": "Dual-confirmation: XGBoost confidence AND Kimi petroulas_worthy=True",
        "normal_size_pct": [1.0, 2.0],
        "conviction_size_pct": [3.0, 5.0],
        "kimi_thresholds": {
            "magnitude_min": 7,
            "conviction_min": 7,
            "composite_stress_min": 6.0,
        },
        "note": "Thresholds hardcoded in imbalance_engine/petroulas_gate.py (NOT in config/)",
        "updated_at": _ts(),
    }

    nodes["petroulas/decisions"] = {
        "last_decision": {
            "symbol": None,
            "approved": None,
            "position_size_pct": None,
            "fault_quality": None,
            "xgb_confidence": None,
            "kimi_magnitude": None,
            "kimi_conviction": None,
            "kimi_petroulas_worthy": None,
            "stress_score": None,
            "thesis_id": None,
            "timestamp": None,
        },
        "approved_count": 0,
        "rejected_count": 0,
        "updated_at": _ts(),
    }

    nodes["petroulas/active_theses"] = {
        "_note": "Live Petroulas theses with falsification tests; keyed by thesis_id",
    }

    # -------------------------------------------------------------------------
    # oracle/  — reflect_cycle, lessons, health
    # -------------------------------------------------------------------------

    nodes["oracle/meta"] = {
        "cycle_file_pattern": "data/oracle/reflections/YYYY_MM_DD.json",
        "wisdom_file": "I_am_a_good_trader.md",
        "cost_per_cycle_cents": 8,
        "reflects_on_last_n_days": 7,
        "updated_at": _ts(),
    }

    nodes["oracle/latest_reflection"] = {
        "date": None,
        "candidate_lesson": None,
        "system_health_note": None,
        "retirement_flag": None,
        "updated_at": _ts(),
    }

    nodes["oracle/health"] = {
        "decision_logger_closed_loop": True,
        "open_outcomes_count": 0,
        "last_reflect_date": None,
        "last_reflect_ok": None,
        "degraded": False,
        "degraded_reason": None,
        "updated_at": _ts(),
    }

    nodes["oracle/lessons_summary"] = {
        "active_lessons": 0,
        "retired_lessons": 0,
        "last_sync": None,
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # mesolimbic/  — Cursus Honorum, Elo, dopamine pathway
    # -------------------------------------------------------------------------

    nodes["mesolimbic/meta"] = {
        "description": "Cursus Honorum — trading knowledge game, Elo as dopamine pathway",
        "question_bank_path": "games/cursus_honorum/question_bank.json",
        "question_count": 100,
        "categories": [
            "EXPECTANCY", "SIZING", "CARRY", "KELLY",
            "SHARPE", "EXITS", "RECOVERY", "TRUST",
        ],
        "updated_at": _ts(),
    }

    nodes["mesolimbic/elo"] = {
        "colin": {
            "current": 1200,           # starting Elo
            "peak": 1200,
            "games_played": 0,
            "last_game": None,
        },
        "alphazero": {
            "current": 1548,           # from alphazero_run_01.json (100/100, run 2026-07-26)
            "peak": 1548,
            "run_file": "data/cursus_honorum/alphazero_run_01.json",
            "run_date": "2026-07-26",
            "score": "100/100",
            "note": "System strong on EXITS/SHARPE (mechanically encoded). Weak spots = no config encoding for Petroulas Kimi thresholds, combat veto deadbands, CONVICTION_NEUTRAL_THRESHOLD",
        },
        "updated_at": _ts(),
    }

    nodes["mesolimbic/category_scores"] = {
        "colin": {cat: {"score": None, "last_tested": None} for cat in
                  ["EXPECTANCY", "SIZING", "CARRY", "KELLY", "SHARPE", "EXITS", "RECOVERY", "TRUST"]},
        "alphazero": {
            # From alphazero_run_01.json — system strengths vs colin weaknesses (inverted)
            "EXITS":      {"score": 1.00, "note": "system strength (mechanically encoded)"},
            "SHARPE":     {"score": 1.00, "note": "system strength (mechanically encoded)"},
            "CARRY":      {"score": 0.95, "note": "carry_engine.py well-encoded"},
            "EXPECTANCY": {"score": 0.93, "note": "ledger facts well-encoded"},
            "KELLY":      {"score": 0.90, "note": "risk constitution encoded"},
            "SIZING":     {"score": 0.88, "note": "conviction pipeline encoded"},
            "RECOVERY":   {"score": 0.85, "note": "Art.3 DD ladder encoded"},
            "TRUST":      {"score": 0.80, "note": "tenet 6 / COT gate encoded"},
        },
        "updated_at": _ts(),
    }

    nodes["mesolimbic/sessions"] = {
        "total_sessions": 0,
        "last_session": None,
        "streak_days": 0,
        "updated_at": _ts(),
    }

    nodes["mesolimbic/system_heatmap"] = {
        "run_file": "data/cursus_honorum/alphazero_run_01.json",
        "summary_file": "data/cursus_honorum/alphazero_run_01_summary.md",
        "score": "100/100",
        "elo": 1548,
        "abstentions": 1,    # LLM_BOUNDARY: petroulas_worthy (Kimi judgment)
        "abstention_note": "Kimi's petroulas_worthy judgment is an LLM boundary — no coded formula to test",
        "key_insight": "System knowledge gaps live in source code, not config files",
        "updated_at": "2026-07-26",
    }

    # -------------------------------------------------------------------------
    # propfirm/  — MT5 demo, FOMC blackout, challenge state
    # -------------------------------------------------------------------------

    nodes["propfirm/meta"] = {
        "platforms_approved": ["Lucid", "MyFundedFutures"],
        "consistency_rule": False,
        "drawdown_type": "EOD",       # NOT intraday
        "demo_hard_guard": True,
        "demo_guard_note": "mt5_bridge.py refuses order_send unless ACCOUNT_TRADE_MODE_DEMO",
        "updated_at": _ts(),
    }

    nodes["propfirm/challenge"] = {
        "active": False,
        "platform": None,
        "account_id": None,
        "phase": None,                 # "challenge" | "verification" | "funded"
        "start_date": None,
        "max_daily_loss_pct": 5.0,
        "max_total_loss_pct": 10.0,
        "profit_target_pct": 8.0,
        "current_balance": None,
        "peak_balance": None,
        "current_drawdown_pct": 0.0,
        "updated_at": _ts(),
    }

    nodes["propfirm/mt5"] = {
        "connected": False,
        "demo_verified": False,
        "account_trade_mode": None,    # must be ACCOUNT_TRADE_MODE_DEMO
        "last_ping": None,
        "fomc_setup_deadline": "2026-07-29T14:00:00-04:00",  # before Wednesday 2pm FOMC
        "updated_at": _ts(),
    }

    nodes["propfirm/fomc_blackout"] = {
        "active": False,
        "window_start": None,
        "window_end": None,
        "hypothesis": "HYP-061",
        "cb_blackout_gate": "CONFIRMED",
        "veto_range_days": [3, 14],     # 3-14d pre BOE/FED
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # live_state/  — PresentState node (already exists; add schema structure)
    # -------------------------------------------------------------------------

    nodes["live_state/schema"] = {
        "version": "v015",
        "last_schema_init": _ts(),
        "pairs": CARRY_PAIRS,
        "note": "Populated by sovereign/present_state.py at 09:35 ET daily",
    }

    nodes["live_state/ignition"] = {
        "gate_open": False,
        "shadow_mode": True,
        "next_check": "2026-07-28 post-FOMC",
        "unlock_plan": "plans/JULY28_UNLOCK_PROMPT.md",
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # library/  — 63-pattern historical analog match
    # -------------------------------------------------------------------------

    nodes["library/meta"] = {
        "patterns": 63,
        "volumes": 10,
        "current_read_date": "2026-05",
        "volumes_converging": 10,
        "updated_at": _ts(),
    }

    nodes["library/current_match"] = {
        "pattern": "ASIAN_CURRENCY_CONTAGION",
        "similarity": 0.927,
        "kelly_cap_pct": 2.0,
        "ptj_level": "SEVERE",
        "defense_mode": True,
        "caution_note": "Pattern match >= 0.90: verify WHICH features drive match before accepting defense mode. Pattern match ≠ causal match.",
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # session_closing/  — 25-task protocol state
    # -------------------------------------------------------------------------

    nodes["session_closing/protocol"] = {
        "version": "alta-session-close v1.0",
        "levels": ["Janitor", "Technician", "Analyst", "Manager", "CEO"],
        "tasks_total": 25,
        "skill_path": "alta-session-close",
        "trigger_phrases": [
            "clean kitchen", "that's a wrap", "closing out",
            "we're done for today", "end of session", "/lastrunoftheday",
            "before I go", "session close", "go home", "wrap up", "done for the day",
        ],
        "updated_at": _ts(),
    }

    nodes["session_closing/last_session"] = {
        "date": "2026-07-25",
        "stamp": "SESSION CLOSE 2026-07-25 18:45 UTC",
        "results": {
            "Janitor": "5/6*",
            "Technician": "6/7**",
            "Analyst": "4/6***",
            "Manager": "3/4****",
            "CEO": "3/3",
        },
        "updated_at": _ts(),
    }

    # -------------------------------------------------------------------------
    # ict/  — pipeline state (shadow, isolation)
    # -------------------------------------------------------------------------

    nodes["ict/meta"] = {
        "isolation_rule": "ict/ and ict-engine/ MUST NOT import from sovereign/",
        "isolation_test": "tests/ -k test_pipeline_does_not_import_sovereign",
        "permutation_p": 0.52,
        "verdict": "NOT_PROVEN",
        "use": "TP/SL reference layer only; unvalidated pattern edge",
        "updated_at": _ts(),
    }

    nodes["ict/pipeline_health"] = {
        "baseline_tests": "4 failed / 23 passed",
        "baseline_date": "2026-07-21",
        "known_failures": [
            "TestScoreAndGrade x2",
            "TestRiskEngineGate x2",
        ],
        "note": "Do NOT treat 21/21 as target — stale baseline. 4 pre-existing failures tracked in plans/restoration-ledger.md",
        "updated_at": _ts(),
    }

    return nodes


# ============================================================================
# RUNNER
# ============================================================================

def main(dry_run: bool = False) -> None:
    print("=" * 70)
    print("Alta Investments — Firebase RTDB Schema Initializer")
    print(f"Target:  {DB_URL}")
    print(f"Mode:    {'DRY RUN (no writes)' if dry_run else 'LIVE WRITE'}")
    print(f"Time:    {NOW}")
    print("=" * 70)

    if not SA_PATH.exists():
        print(f"\nERROR: service account not found at {SA_PATH}")
        sys.exit(1)

    schema = build_schema()
    print(f"\nSchema compiled: {len(schema)} nodes\n")

    if dry_run:
        for path, data in sorted(schema.items()):
            print(f"  [DRY] /{path}  ({len(json.dumps(data))} bytes)")
        print("\nDry run complete — no writes.")
        return

    # --- initialise firebase-admin ---
    try:
        import firebase_admin
        from firebase_admin import credentials, db
    except ImportError:
        print("ERROR: firebase-admin not installed. Run:")
        print("  pip install firebase-admin --break-system-packages")
        sys.exit(1)

    try:
        cred = credentials.Certificate(str(SA_PATH))
        try:
            app = firebase_admin.initialize_app(cred, {"databaseURL": DB_URL})
        except ValueError:
            # already initialised (e.g. re-run in same interpreter)
            app = firebase_admin.get_app()
    except Exception as e:
        print(f"ERROR initialising firebase-admin: {e}")
        sys.exit(1)

    # --- write ---
    ok = 0
    fail = 0
    for path, data in sorted(schema.items()):
        try:
            ref = db.reference(path)
            ref.set(data)
            print(f"  ✓  /{path}")
            ok += 1
        except Exception as e:
            print(f"  ✗  /{path}  ERROR: {e}")
            fail += 1

    print(f"\nDone: {ok} written, {fail} failed.")

    # Write a manifest so we know what was initialised
    manifest_path = ROOT / "data" / "agent" / "firebase_schema_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "initialized_at": NOW,
        "db_url": DB_URL,
        "nodes_written": ok,
        "nodes_failed": fail,
        "paths": sorted(schema.keys()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written → {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise Firebase RTDB schema for Alta Investments")
    parser.add_argument("--dry-run", action="store_true", help="Print schema without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
