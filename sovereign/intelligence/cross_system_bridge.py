"""
Cross-System Bridge — Layer 4 of the Sovereign Intelligence Architecture.

Two-way intelligence sharing between the Quant (forex/equity) and ICT systems.
Both systems trade toward the same goal. When one detects danger or opportunity,
the other should know immediately.

QUANT → ICT  (macro environment signals):
    When forex macro uncertainty is elevated:
        TIGHTEN_THRESHOLDS → ICT only takes best setups, widens stops
    When Library convergence is extreme (CRITICAL regime):
        HALT_NEW_POSITIONS → ICT pauses until regime normalises

ICT → QUANT  (execution reality signals):
    When ICT stop clustering is detected (3+ stops in rolling window):
        REDUCE_CONVICTION → Forex sizes down 50% for 48h
    When commitment failures cluster (market not following through):
        REDUCE_CONVICTION → Confirms macro uncertainty from execution side

STATE FILE: data/forensics/cross_system_state.json
    Written by either system, read by both.
    Staleness check: if older than 6h, treat as neutral.

SIGNAL LOG: data/forensics/cross_system_signals.jsonl
    Append-only audit trail of every signal emitted.

Usage:
    bridge = CrossSystemBridge()

    # At ICT pipeline entry:
    state = bridge.read_state()
    if state.ict_mode == 'TIGHTEN':
        min_grade = 'A'  # already enforced, but now extra tight

    # After ICT stop:
    bridge.record_ict_stop(pair='GBPUSD', session='London')

    # In forex orchestrator, after signal generation:
    bridge.update_quant_state(library_insight, commitment_scores)
    state = bridge.read_state()
    if state.quant_signal == 'REDUCE_CONVICTION':
        signal.size_multiplier *= 0.50

Run:
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/cross_system_bridge.py
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/cross_system_bridge.py --status
    PYTHONPATH=/path/to/quant python3 sovereign/intelligence/cross_system_bridge.py --update
"""
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE  = ROOT / "data" / "forensics" / "cross_system_state.json"
SIGNAL_LOG  = ROOT / "data" / "forensics" / "cross_system_signals.jsonl"
VETO_LEDGER = ROOT / "data" / "ledger" / "ict_veto_ledger_2026_05.jsonl"
ICT_TRADES  = ROOT / "data" / "ledger" / "ict_paper_trades.json"

# Thresholds
LIBRARY_TIGHTEN_THRESHOLD   = 0.85   # library convergence → tighten ICT
LIBRARY_HALT_THRESHOLD      = 0.95   # extreme convergence → halt new ICT positions
MACRO_UNCERTAINTY_THRESHOLD = 0.70   # forex uncertainty → tighten ICT
ICT_STOP_CLUSTER_N          = 3      # N stops in window → reduce forex conviction
ICT_STOP_WINDOW_HOURS       = 24     # rolling window for stop clustering
COMMITMENT_FAIL_CLUSTER_N   = 3      # N commitment failures → reduce conviction
STATE_STALENESS_HOURS       = 6      # hours before state is treated as neutral


# ── State dataclass ───────────────────────────────────────────────────────

@dataclass
class BridgeState:
    # Quant → ICT direction
    ict_mode: str = "NORMAL"
    # NORMAL | TIGHTEN | HALT_NEW
    ict_mode_reason: str = ""
    ict_mode_since: str = ""

    # ICT → Quant direction
    quant_signal: str = "NORMAL"
    # NORMAL | REDUCE_CONVICTION
    quant_signal_reason: str = ""
    quant_signal_since: str = ""
    quant_signal_expires: str = ""   # ISO timestamp when signal expires

    # Live metrics driving the signals
    library_convergence: float = 0.0
    library_threat_score: float = 0.0
    library_primary_regime: str = ""
    macro_uncertainty: float = 0.0
    commitment_score_avg: float = 0.5

    ict_stops_24h: int = 0
    ict_commitment_failures_24h: int = 0

    # Meta
    last_updated: str = ""
    updated_by: str = ""   # QUANT | ICT | MANUAL

    def is_stale(self) -> bool:
        if not self.last_updated:
            return True
        try:
            updated = datetime.fromisoformat(self.last_updated)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - updated) > timedelta(hours=STATE_STALENESS_HOURS)
        except Exception:
            return True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BridgeState":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ── Bridge class ──────────────────────────────────────────────────────────

class CrossSystemBridge:

    def __init__(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Read / Write state ────────────────────────────────────────────────

    def read_state(self) -> BridgeState:
        if not STATE_FILE.exists():
            return BridgeState()
        try:
            raw = json.loads(STATE_FILE.read_text())
            state = BridgeState.from_dict(raw)
            if state.is_stale():
                return BridgeState()   # stale state = neutral, don't act on old data
            return state
        except Exception:
            return BridgeState()

    def _write_state(self, state: BridgeState) -> None:
        state.last_updated = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))

    def _log_signal(self, signal_type: str, direction: str,
                    value: str, reason: str, metrics: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": signal_type,
            "direction": direction,
            "value": value,
            "reason": reason,
            "metrics": metrics,
        }
        with open(SIGNAL_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Quant → ICT: compute from forex/library state ─────────────────────

    def update_quant_state(
        self,
        library_insight: Optional[Any] = None,
        commitment_scores: Optional[List[float]] = None,
        macro_uncertainty: float = 0.0,
    ) -> BridgeState:
        """
        Called by the forex orchestrator or scheduler after computing macro state.
        Updates the QUANT → ICT signal based on Library and commitment scores.
        """
        state = self.read_state()
        now   = datetime.now(timezone.utc).isoformat()

        # Extract library metrics
        lib_convergence = 0.0
        lib_threat      = 0.0
        lib_regime      = ""
        if library_insight is not None:
            try:
                lib_convergence = sum(
                    1 for vm in library_insight.volume_matches
                    if vm.similarity >= 0.60
                ) / max(len(library_insight.volume_matches), 1)
                lib_threat  = float(library_insight.threat_score)
                lib_regime  = str(library_insight.primary_regime or "")
            except Exception:
                pass

        # Commitment scores across forex pairs
        commit_avg = float(sum(commitment_scores) / len(commitment_scores)) \
                     if commitment_scores else 0.5

        # Determine ICT mode
        prev_mode = state.ict_mode
        if lib_convergence >= LIBRARY_HALT_THRESHOLD or lib_threat >= 0.95:
            new_mode = "HALT_NEW"
            mode_reason = (f"Library CRITICAL: {lib_convergence:.0%} volumes converging, "
                           f"threat={lib_threat:.2f}, regime={lib_regime}")
        elif (lib_convergence >= LIBRARY_TIGHTEN_THRESHOLD or
              macro_uncertainty >= MACRO_UNCERTAINTY_THRESHOLD or
              commit_avg < 0.40):
            new_mode = "TIGHTEN"
            mode_reason = (f"Elevated uncertainty: lib_conv={lib_convergence:.0%}, "
                           f"macro_uncertainty={macro_uncertainty:.2f}, "
                           f"commit_avg={commit_avg:.2f}")
        else:
            new_mode = "NORMAL"
            mode_reason = (f"All clear: lib_conv={lib_convergence:.0%}, "
                           f"threat={lib_threat:.2f}, commit_avg={commit_avg:.2f}")

        # Update state
        state.library_convergence    = round(lib_convergence, 4)
        state.library_threat_score   = round(lib_threat, 4)
        state.library_primary_regime = lib_regime
        state.macro_uncertainty      = round(macro_uncertainty, 4)
        state.commitment_score_avg   = round(commit_avg, 4)
        state.ict_mode               = new_mode
        state.ict_mode_reason        = mode_reason
        state.updated_by             = "QUANT"

        if new_mode != prev_mode:
            state.ict_mode_since = now
            self._log_signal(
                signal_type="QUANT_TO_ICT",
                direction="QUANT→ICT",
                value=new_mode,
                reason=mode_reason,
                metrics={"lib_conv": lib_convergence, "lib_threat": lib_threat,
                         "macro_unc": macro_uncertainty, "commit_avg": commit_avg},
            )

        self._write_state(state)
        return state

    # ── ICT → Quant: compute from ICT execution reality ──────────────────

    def record_ict_stop(self, pair: str = "", session: str = "",
                        reason: str = "STOP") -> BridgeState:
        """
        Called each time an ICT trade hits its stop.
        Checks if stops are clustering → emits REDUCE_CONVICTION if so.
        """
        state = self.read_state()
        now   = datetime.now(timezone.utc)

        # Count stops in rolling 24h window from veto ledger
        stops_24h = self._count_ict_stops(window_hours=ICT_STOP_WINDOW_HOURS)
        state.ict_stops_24h = stops_24h

        prev_signal = state.quant_signal

        if stops_24h >= ICT_STOP_CLUSTER_N:
            new_signal   = "REDUCE_CONVICTION"
            signal_reason = (f"ICT stop cluster: {stops_24h} stops in last "
                             f"{ICT_STOP_WINDOW_HOURS}h — market not following through")
            expires = (now + timedelta(hours=48)).isoformat()
        elif state.quant_signal == "REDUCE_CONVICTION":
            # Check if existing signal has expired
            try:
                exp = datetime.fromisoformat(state.quant_signal_expires)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if now > exp:
                    new_signal    = "NORMAL"
                    signal_reason = "REDUCE_CONVICTION expired"
                    expires       = ""
                else:
                    new_signal    = state.quant_signal
                    signal_reason = state.quant_signal_reason
                    expires       = state.quant_signal_expires
            except Exception:
                new_signal = "NORMAL"
                signal_reason = ""
                expires = ""
        else:
            new_signal    = "NORMAL"
            signal_reason = f"ICT stops in 24h: {stops_24h} (threshold: {ICT_STOP_CLUSTER_N})"
            expires       = ""

        state.quant_signal         = new_signal
        state.quant_signal_reason  = signal_reason
        state.quant_signal_expires = expires
        state.updated_by           = "ICT"

        if new_signal != prev_signal:
            state.quant_signal_since = now.isoformat()
            self._log_signal(
                signal_type="ICT_TO_QUANT",
                direction="ICT→QUANT",
                value=new_signal,
                reason=signal_reason,
                metrics={"stops_24h": stops_24h, "pair": pair, "session": session},
            )

        self._write_state(state)
        return state

    def record_commitment_failure(self, pair: str = "") -> BridgeState:
        """
        Called when the commitment detector vetoes an ICT trade
        (market not following through on multiple setups in a day).
        """
        state  = self.read_state()
        fails  = self._count_commitment_failures(window_hours=ICT_STOP_WINDOW_HOURS)
        state.ict_commitment_failures_24h = fails

        prev_signal = state.quant_signal
        if fails >= COMMITMENT_FAIL_CLUSTER_N and state.quant_signal == "NORMAL":
            state.quant_signal         = "REDUCE_CONVICTION"
            state.quant_signal_reason  = (f"ICT commitment failures clustering: "
                                          f"{fails} in last {ICT_STOP_WINDOW_HOURS}h")
            state.quant_signal_expires = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
            state.quant_signal_since   = datetime.now(timezone.utc).isoformat()
            state.updated_by           = "ICT"

            self._log_signal(
                signal_type="ICT_TO_QUANT",
                direction="ICT→QUANT",
                value="REDUCE_CONVICTION",
                reason=state.quant_signal_reason,
                metrics={"commitment_failures_24h": fails, "pair": pair},
            )
            self._write_state(state)

        return state

    # ── ICT pipeline integration ──────────────────────────────────────────

    def get_ict_thresholds(self) -> dict:
        """
        Called at start of each ICT pipeline cycle.
        Returns adjusted thresholds based on current bridge state.
        Normal defaults match existing pipeline config.
        """
        state = self.read_state()

        if state.ict_mode == "HALT_NEW":
            return {
                "active": False,
                "min_grade": "A",
                "min_score": 999.0,   # effectively blocks all new entries
                "stop_buffer_mult": 1.0,
                "max_daily_trades": 0,
                "size_multiplier": 0.0,
                "mode": "HALT_NEW",
                "reason": state.ict_mode_reason,
            }
        elif state.ict_mode == "TIGHTEN":
            return {
                "active": True,
                "min_grade": "A",
                "min_score": 8.0,     # tighter than normal 7.0
                "stop_buffer_mult": 1.5,
                "max_daily_trades": 1,
                "size_multiplier": 0.75,
                "mode": "TIGHTEN",
                "reason": state.ict_mode_reason,
            }
        else:  # NORMAL
            return {
                "active": True,
                "min_grade": "A",
                "min_score": 7.0,
                "stop_buffer_mult": 1.0,
                "max_daily_trades": 3,
                "size_multiplier": 1.0,
                "mode": "NORMAL",
                "reason": "All clear",
            }

    # ── Quant pipeline integration ────────────────────────────────────────

    def get_quant_size_multiplier(self) -> float:
        """
        Called before executing a forex/equity position.
        Returns size multiplier based on current bridge state.
        """
        state = self.read_state()
        if state.quant_signal == "REDUCE_CONVICTION":
            return 0.50
        return 1.0

    # ── Data readers ──────────────────────────────────────────────────────

    def _count_ict_stops(self, window_hours: int = 24) -> int:
        """Count real ICT stops from paper trades + veto ledger in rolling window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        count = 0

        # From paper trades (closed with STOP outcome)
        try:
            trades = json.loads(ICT_TRADES.read_text())
            for t in trades.get("closed", []):
                if t.get("outcome") == "STOP":
                    try:
                        ts = datetime.fromisoformat(t.get("closed_at", ""))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts > cutoff:
                            count += 1
                    except Exception:
                        pass
        except Exception:
            pass

        return count

    def _count_commitment_failures(self, window_hours: int = 24) -> int:
        """Count COMMITMENT_DETECTOR vetos from veto ledger in rolling window."""
        if not VETO_LEDGER.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        count  = 0
        try:
            with open(VETO_LEDGER) as f:
                for line in f:
                    try:
                        v = json.loads(line)
                        reason = v.get("veto_reason", "")
                        if "COMMITMENT_DETECTOR" in reason:
                            ts_str = v.get("timestamp", "")
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts > cutoff:
                                count += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return count

    # ── Full update cycle ─────────────────────────────────────────────────

    def run_full_update(self, verbose: bool = True) -> BridgeState:
        """
        Run a complete bridge update: query Library, compute commitment,
        check ICT stops, and write state.
        Called by agent_scheduler.py every 2 hours.
        """
        import numpy as np

        if verbose:
            print("Cross-System Bridge — full update cycle")

        # 1. Query Library with current market state
        library_insight = None
        try:
            import yfinance as yf
            import pandas as pd
            spy = yf.download("SPY", period="300d", progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            vix = yf.download("^VIX", period="300d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            spy_arr = spy["Close"].values[-252:].astype(float)
            vix_arr = vix["Close"].values[-252:].astype(float)

            from sovereign.risk.alexandrian_library import AlexandrianLibrary
            lib = AlexandrianLibrary()
            library_insight = lib.query(spy_arr, vix_arr)
            if verbose:
                print(f"  Library: {library_insight.primary_regime} "
                      f"threat={library_insight.threat_score:.2f} "
                      f"size={library_insight.size_modifier:.2f}")
        except Exception as e:
            if verbose:
                print(f"  Library unavailable: {e}")

        # 2. Compute commitment scores across all forex pairs
        commitment_scores = []
        try:
            from sovereign.intelligence.commitment_detector import CommitmentDetector
            from sovereign.forex.pair_universe import ALL_PAIRS
            detector = CommitmentDetector(log=False)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for pair in ALL_PAIRS:
                # Use neutral direction for macro assessment
                state_long  = detector.compute(pair, 1,  today, session="MACRO")
                state_short = detector.compute(pair, -1, today, session="MACRO")
                commitment_scores.append(max(state_long.score, state_short.score))
            if verbose:
                print(f"  Commitment avg: {sum(commitment_scores)/len(commitment_scores):.3f} "
                      f"across {len(commitment_scores)} pairs")
        except Exception as e:
            if verbose:
                print(f"  Commitment scores unavailable: {e}")

        # 3. Update quant state
        state = self.update_quant_state(
            library_insight=library_insight,
            commitment_scores=commitment_scores if commitment_scores else None,
            macro_uncertainty=float(library_insight.threat_score) if library_insight else 0.0,
        )

        # 4. Update ICT stop clustering
        stops_24h = self._count_ict_stops()
        fails_24h = self._count_commitment_failures()
        state.ict_stops_24h = stops_24h
        state.ict_commitment_failures_24h = fails_24h

        if stops_24h >= ICT_STOP_CLUSTER_N or fails_24h >= COMMITMENT_FAIL_CLUSTER_N:
            state.quant_signal = "REDUCE_CONVICTION"
            state.quant_signal_reason = (
                f"ICT clustering: {stops_24h} stops + {fails_24h} commitment failures in 24h"
            )
            state.quant_signal_expires = (
                datetime.now(timezone.utc) + timedelta(hours=48)
            ).isoformat()
        elif state.quant_signal == "REDUCE_CONVICTION":
            try:
                exp = datetime.fromisoformat(state.quant_signal_expires)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > exp:
                    state.quant_signal = "NORMAL"
                    state.quant_signal_reason = "Expired"
                    state.quant_signal_expires = ""
            except Exception:
                state.quant_signal = "NORMAL"

        self._write_state(state)

        if verbose:
            print(f"\n  ICT mode:     {state.ict_mode} — {state.ict_mode_reason[:60]}")
            print(f"  Quant signal: {state.quant_signal} — {state.quant_signal_reason[:60]}")
            print(f"  Stops 24h: {stops_24h} | Commitment fails 24h: {fails_24h}")
            print(f"\n  State written to: {STATE_FILE}")

        return state

    def print_status(self) -> None:
        state = self.read_state()
        stale = state.is_stale()

        print(f"\n{'='*58}")
        print(f"CROSS-SYSTEM BRIDGE STATUS{'  [STALE]' if stale else ''}")
        print(f"{'='*58}")
        print(f"Last updated: {state.last_updated[:19] if state.last_updated else 'never'}"
              f" (by {state.updated_by or 'unknown'})")
        print()
        print(f"QUANT → ICT:")
        print(f"  Mode:        {state.ict_mode}")
        print(f"  Reason:      {state.ict_mode_reason[:70]}")
        print(f"  Since:       {state.ict_mode_since[:19] if state.ict_mode_since else 'unknown'}")
        thresholds = CrossSystemBridge().get_ict_thresholds()
        print(f"  Effect:      min_score={thresholds['min_score']} "
              f"size_mult={thresholds['size_multiplier']}× "
              f"max_trades={thresholds['max_daily_trades']}")
        print()
        print(f"ICT → QUANT:")
        print(f"  Signal:      {state.quant_signal}")
        print(f"  Reason:      {state.quant_signal_reason[:70]}")
        mult = CrossSystemBridge().get_quant_size_multiplier()
        print(f"  Effect:      size_multiplier={mult}×")
        if state.quant_signal_expires:
            print(f"  Expires:     {state.quant_signal_expires[:19]}")
        print()
        print(f"Live metrics:")
        print(f"  Library convergence: {state.library_convergence:.0%}")
        print(f"  Library threat:      {state.library_threat_score:.3f}")
        print(f"  Library regime:      {state.library_primary_regime}")
        print(f"  Commitment avg:      {state.commitment_score_avg:.3f}")
        print(f"  ICT stops 24h:       {state.ict_stops_24h}")
        print(f"  Commitment fails 24h:{state.ict_commitment_failures_24h}")


# ── Wire into ICT pipeline ────────────────────────────────────────────────

_bridge_instance: Optional[CrossSystemBridge] = None

def get_bridge() -> CrossSystemBridge:
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CrossSystemBridge()
    return _bridge_instance


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-System Bridge")
    parser.add_argument("--status", action="store_true", help="Show current state")
    parser.add_argument("--update", action="store_true", help="Run full update cycle")
    parser.add_argument("--simulate-stop", action="store_true",
                        help="Simulate an ICT stop being recorded")
    args = parser.parse_args()

    bridge = CrossSystemBridge()

    if args.status:
        bridge.print_status()
    elif args.simulate_stop:
        print("Simulating ICT stop...")
        state = bridge.record_ict_stop(pair="GBPUSD", session="London", reason="TEST")
        bridge.print_status()
    elif args.update:
        state = bridge.run_full_update(verbose=True)
    else:
        # Default: run full update and show status
        state = bridge.run_full_update(verbose=True)
        bridge.print_status()
