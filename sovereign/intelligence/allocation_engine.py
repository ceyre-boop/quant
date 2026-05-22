"""
Sovereign Capital Allocation Engine
sovereign/intelligence/allocation_engine.py

Converts five regime classifiers into continuous allocation weights (0.0-1.0)
for each of the three trading systems. Replaces the binary HALT/NORMAL bridge
signals with a proportional dimmer that scales deployment based on regime quality.

Each system has a different sensitivity to regime:
  Equity — most sensitive to macro threat (Library kills equity in CRITICAL)
  ICT    — most sensitive to session quality and commitment (fragile, precision)
  Forex  — least sensitive (macro signal IS the edge; some regimes actually help)

Inputs (all existing infrastructure):
  1. Alexandrian Library — threat score + regime label
  2. VIX slope (VIX3M - VIX) — carry regime health
  3. Cross-system bridge — stop clustering, commitment failures
  4. System-level drawdown — from live paper account state

Output: data/forensics/allocation_state.json
  {
    "equity_weight": 0.0-1.0,
    "ict_weight":    0.0-1.0,
    "forex_weight":  0.0-1.0,
    "regime_tag":    string,
    "confidence":    0.0-1.0,
    "reason":        string,
    "components":    {...}   ← each modifier's contribution
  }

All three systems read this file at execution start.
Staleness: >4h → treat as neutral weights (0.75 each, don't block, don't boost).

Run:
    python3 sovereign/intelligence/allocation_engine.py
    python3 sovereign/intelligence/allocation_engine.py --status
    python3 sovereign/intelligence/allocation_engine.py --update
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

ROOT           = Path(__file__).resolve().parents[2]
STATE_FILE     = ROOT / "data" / "forensics" / "allocation_state.json"
BRIDGE_FILE    = ROOT / "data" / "forensics" / "cross_system_state.json"
CHALLENGE_FILE = ROOT / "data" / "propfirm" / "active_challenge.json"

STALENESS_HOURS = 4


# ── Regime → base allocation table ───────────────────────────────────────────
# Each system's base weight given Library threat level.
# These are the "clean slate" weights before modifiers.
# Equity and ICT are more fragile. Forex is more structural.

THREAT_ALLOCATIONS = {
    # threat_max: (equity, ict, forex)
    0.30: (1.00, 1.00, 1.00),   # NORMAL — full deployment
    0.50: (0.85, 0.80, 0.90),   # ELEVATED — light reduction
    0.70: (0.65, 0.50, 0.80),   # WARNING — meaningful reduction
    0.85: (0.40, 0.25, 0.65),   # DANGER — significant reduction
    0.95: (0.20, 0.00, 0.40),   # TIGHTEN — ICT halted, others minimal
    1.01: (0.00, 0.00, 0.25),   # CRITICAL — equity/ICT halted, forex minimal
}


@dataclass
class AllocationState:
    # Core allocation weights
    equity_weight: float = 1.0
    ict_weight:    float = 1.0
    forex_weight:  float = 1.0

    # Regime context
    regime_tag:    str   = "UNKNOWN"
    confidence:    float = 0.5
    reason:        str   = ""

    # Component breakdown
    library_threat:        float = 0.0
    library_regime:        str   = ""
    vix_slope:             float = 0.0
    ict_commitment_fails:  int   = 0
    ict_stops_24h:         int   = 0
    equity_buffer_pct:     float = 1.0   # prop challenge buffer as fraction

    # Meta
    last_updated:  str   = ""
    updated_by:    str   = "ALLOCATOR"

    def is_stale(self) -> bool:
        if not self.last_updated:
            return True
        try:
            ts = datetime.fromisoformat(self.last_updated)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts) > timedelta(hours=STALENESS_HOURS)
        except Exception:
            return True

    def to_dict(self) -> dict:
        return asdict(self)


def _neutral() -> AllocationState:
    """Return neutral weights when state is stale or unavailable."""
    return AllocationState(
        equity_weight=0.75, ict_weight=0.75, forex_weight=0.75,
        regime_tag="NEUTRAL_STALE", confidence=0.3,
        reason="Allocation state stale — using conservative neutral weights",
        last_updated=datetime.now(timezone.utc).isoformat(),
    )


def _base_weights(threat: float) -> tuple[float, float, float]:
    """Look up base allocation weights for a given threat score."""
    for max_threat, weights in THREAT_ALLOCATIONS.items():
        if threat <= max_threat:
            return weights
    return (0.0, 0.0, 0.25)


def _interpolate_weights(threat: float) -> tuple[float, float, float]:
    """Smooth interpolation between threat breakpoints."""
    thresholds = list(THREAT_ALLOCATIONS.keys())
    values     = list(THREAT_ALLOCATIONS.values())

    for i in range(len(thresholds)):
        if threat <= thresholds[i]:
            if i == 0:
                return values[0]
            # Interpolate between i-1 and i
            lo_t, hi_t = thresholds[i-1], thresholds[i]
            lo_w, hi_w = values[i-1], values[i]
            alpha = (threat - lo_t) / (hi_t - lo_t)
            return tuple(round(lo_w[j] + alpha * (hi_w[j] - lo_w[j]), 4) for j in range(3))

    return (0.0, 0.0, 0.25)


def _load_bridge_state() -> dict:
    if BRIDGE_FILE.exists():
        try:
            return json.loads(BRIDGE_FILE.read_text())
        except Exception:
            pass
    return {}


def _load_challenge_buffer() -> float:
    """Returns prop challenge buffer as fraction of starting drawdown (1.0 = full buffer)."""
    if CHALLENGE_FILE.exists():
        try:
            ch = json.loads(CHALLENGE_FILE.read_text())
            balance      = float(ch.get("balance", 100_000))
            floor        = float(ch.get("drawdown_floor", 92_000))
            account_size = float(ch.get("account_size", 100_000))
            max_dd       = float(ch.get("max_dd", 0.08))
            starting_dd  = account_size * max_dd
            buffer       = balance - floor
            return max(0.0, min(1.0, buffer / starting_dd))
        except Exception:
            pass
    return 1.0


def _get_vix_slope() -> Optional[float]:
    """VIX3M - VIX. Positive = contango. Negative = backwardation."""
    try:
        import yfinance as yf
        vix  = yf.download("^VIX",  period="5d", progress=False)
        vix3 = yf.download("^VIX3M", period="5d", progress=False)
        if hasattr(vix.columns,  "get_level_values"): vix.columns  = vix.columns.get_level_values(0)
        if hasattr(vix3.columns, "get_level_values"): vix3.columns = vix3.columns.get_level_values(0)
        v  = float(vix["Close"].dropna().iloc[-1])
        v3 = float(vix3["Close"].dropna().iloc[-1])
        return round(v3 - v, 3)
    except Exception:
        return None


def compute_allocation(verbose: bool = True) -> AllocationState:
    """
    Compute current allocation weights from all regime classifiers.
    Called by agent_scheduler every 2h and at system startup.
    """

    # ── 1. Load Library state from bridge (already queried) ──────────────────
    bridge = _load_bridge_state()
    threat  = float(bridge.get("library_threat_score", 0.5))
    regime  = str(bridge.get("library_primary_regime", "UNKNOWN"))
    commits = int(bridge.get("ict_commitment_failures_24h", 0))
    stops   = int(bridge.get("ict_stops_24h", 0))

    # ── 2. Base weights from Library threat ──────────────────────────────────
    eq_w, ict_w, fx_w = _interpolate_weights(threat)
    components = {
        "library_threat":    {"value": threat, "eq": eq_w, "ict": ict_w, "fx": fx_w},
    }

    # ── 3. VIX slope modifier ─────────────────────────────────────────────────
    vix_slope = _get_vix_slope()
    vix_mod_fx  = 0.0
    vix_mod_ict = 0.0
    if vix_slope is not None:
        if vix_slope > 5.0:       # steep contango — carry fragile
            vix_mod_fx = -0.15; vix_mod_ict = -0.10
        elif vix_slope > 3.0:     # elevated contango — caution
            vix_mod_fx = -0.10; vix_mod_ict = -0.05
        elif vix_slope < -1.0:    # backwardation — vol spike imminent
            vix_mod_fx = -0.20; vix_mod_ict = -0.25
        elif vix_slope < 0.0:     # slight inversion — watch closely
            vix_mod_fx = -0.10; vix_mod_ict = -0.15
        components["vix_slope"] = {
            "value": vix_slope, "fx_mod": vix_mod_fx, "ict_mod": vix_mod_ict
        }
    fx_w  = max(0.0, min(1.0, fx_w  + vix_mod_fx))
    ict_w = max(0.0, min(1.0, ict_w + vix_mod_ict))

    # ── 4. ICT execution reality modifiers ───────────────────────────────────
    ict_cluster_mod_fx  = 0.0
    ict_cluster_mod_ict = 0.0
    if commits >= 3:        # commitment failures clustering → market choppy
        ict_cluster_mod_fx  = -0.25
        ict_cluster_mod_ict = -0.50
    elif commits >= 1:
        ict_cluster_mod_fx  = -0.10
        ict_cluster_mod_ict = -0.20
    if stops >= 3:          # stop clustering → directional bias failing
        ict_cluster_mod_fx  = min(ict_cluster_mod_fx  - 0.20, 0.0)
        ict_cluster_mod_ict = min(ict_cluster_mod_ict - 0.30, 0.0)
    components["ict_clustering"] = {
        "commits_24h": commits, "stops_24h": stops,
        "fx_mod": ict_cluster_mod_fx, "ict_mod": ict_cluster_mod_ict
    }
    fx_w  = max(0.0, min(1.0, fx_w  + ict_cluster_mod_fx))
    ict_w = max(0.0, min(1.0, ict_w + ict_cluster_mod_ict))

    # ── 5. Prop challenge buffer modifier (equity only) ──────────────────────
    buffer_pct = _load_challenge_buffer()
    eq_buffer_mod = 0.0
    if buffer_pct < 0.25:    # very close to floor — protect the challenge
        eq_buffer_mod = -0.50
    elif buffer_pct < 0.40:  # caution territory
        eq_buffer_mod = -0.25
    elif buffer_pct < 0.60:
        eq_buffer_mod = -0.10
    components["prop_challenge_buffer"] = {
        "buffer_pct": round(buffer_pct, 3), "eq_mod": eq_buffer_mod
    }
    eq_w = max(0.0, min(1.0, eq_w + eq_buffer_mod))

    # ── 6. Regime tag ─────────────────────────────────────────────────────────
    if threat > 0.95:
        regime_tag = f"CRITICAL_{regime}"
    elif threat > 0.85:
        regime_tag = f"DANGER_{regime}"
    elif threat > 0.70:
        regime_tag = f"WARNING_{regime}"
    elif threat > 0.50:
        regime_tag = "ELEVATED_UNCERTAINTY"
    else:
        regime_tag = "NORMAL"

    if commits >= 3 or stops >= 3:
        regime_tag += "_EXECUTION_CHOPPY"

    # ── 7. Confidence: how certain is this read? ─────────────────────────────
    # High when library threat is clear, low when in transition zones
    confidence_map = {
        (0.00, 0.20): 0.90,
        (0.20, 0.40): 0.75,
        (0.40, 0.60): 0.55,   # transition — less certain
        (0.60, 0.80): 0.70,
        (0.80, 1.01): 0.90,
    }
    confidence = 0.60
    for (lo, hi), conf in confidence_map.items():
        if lo <= threat < hi:
            confidence = conf
            break

    # Build reason string
    reason_parts = [f"Library threat={threat:.2f} ({regime})"]
    if vix_slope is not None:
        reason_parts.append(f"VIX slope={vix_slope:+.1f}")
    if commits > 0 or stops > 0:
        reason_parts.append(f"ICT: {commits} commit_fails + {stops} stops (24h)")
    if eq_buffer_mod < 0:
        reason_parts.append(f"Prop buffer={buffer_pct*100:.0f}% of DD")
    reason = " | ".join(reason_parts)

    state = AllocationState(
        equity_weight=round(eq_w, 3),
        ict_weight=round(ict_w, 3),
        forex_weight=round(fx_w, 3),
        regime_tag=regime_tag,
        confidence=round(confidence, 3),
        reason=reason,
        library_threat=threat,
        library_regime=regime,
        vix_slope=vix_slope or 0.0,
        ict_commitment_fails=commits,
        ict_stops_24h=stops,
        equity_buffer_pct=round(buffer_pct, 3),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))

    if verbose:
        print(f"\n{'='*56}")
        print(f"SOVEREIGN ALLOCATION ENGINE")
        print(f"{'='*56}")
        print(f"Regime:  {regime_tag}")
        print(f"Confidence: {confidence:.0%}")
        print()
        print(f"  EQUITY  {eq_w*100:5.1f}%  {'█' * int(eq_w*20)}")
        print(f"  ICT     {ict_w*100:5.1f}%  {'█' * int(ict_w*20)}")
        print(f"  FOREX   {fx_w*100:5.1f}%  {'█' * int(fx_w*20)}")
        print()
        print(f"  {reason}")
        print()
        print(f"  Saved → {STATE_FILE.name}")

    return state


def read_allocation() -> AllocationState:
    """Read current allocation. Returns neutral weights if stale."""
    if not STATE_FILE.exists():
        return _neutral()
    try:
        raw = json.loads(STATE_FILE.read_text())
        state = AllocationState(**{k: v for k, v in raw.items()
                                   if k in AllocationState.__dataclass_fields__})
        if state.is_stale():
            return _neutral()
        return state
    except Exception:
        return _neutral()


def print_status() -> None:
    state = read_allocation()
    stale = state.is_stale()
    print(f"\nAllocation state {'[STALE — using neutral 75%]' if stale else ''}")
    print(f"  EQUITY  {state.equity_weight*100:5.1f}%")
    print(f"  ICT     {state.ict_weight*100:5.1f}%")
    print(f"  FOREX   {state.forex_weight*100:5.1f}%")
    print(f"  Regime: {state.regime_tag}")
    print(f"  Updated: {state.last_updated[:16]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        compute_allocation(verbose=True)
