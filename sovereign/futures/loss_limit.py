"""Hard daily loss limit for the Track-2 futures sandbox.

Sandbox-local (no forex/ICT imports). Locks the auto-execution path when the session's realized loss
hits the limit; unlock is manual (delete the lock file or press 'u' in the monitor). This is the
non-negotiable safety floor — paper or not, the machine does not get to keep losing at machine speed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = ROOT / "data" / "futures" / ".session_lock"
TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"

POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}      # $ per index point per micro contract
DEFAULT_LIMIT_USD = 500.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def session_pnl_usd(bridge=None, instrument: str = "MES") -> float:
    """Realized session P&L in dollars. Prefer IB account RealizedPnL (authoritative during a live
    session); fall back to summing today's CLOSED logged trades (entry/exit × point value × size)."""
    if bridge is not None:
        try:
            s = bridge.account_summary()
            rp = s.get("RealizedPnL")
            if rp is not None:
                return round(float(rp), 2)
        except Exception:
            pass
    # Account-total realized P&L across all instruments today (matches IB's account RealizedPnL);
    # point value is per the trade's OWN instrument. `instrument` arg is only a default fallback.
    pnl = 0.0
    if TRADE_LOG.exists():
        today = _today()
        for line in TRADE_LOG.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not str(r.get("ts", "")).startswith(today):
                continue
            entry, exit_, size = r.get("entry"), r.get("exit"), r.get("size_contracts") or 0
            if entry is None or exit_ is None or not size:
                continue
            pv = POINT_VALUE.get(r.get("instrument"), POINT_VALUE.get(instrument, 5.0))
            mult = 1 if r.get("direction") == "LONG" else -1
            pnl += (float(exit_) - float(entry)) * mult * pv * float(size)
    return round(pnl, 2)


def is_locked() -> bool:
    return LOCK_FILE.exists()


def lock_reason() -> str:
    try:
        return json.loads(LOCK_FILE.read_text()).get("reason", "")
    except Exception:
        return ""


def lock(reason: str) -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps({"locked_at": _now(), "reason": reason}, indent=2))


def unlock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def check_and_lock(pnl_usd: float, limit_usd: float = DEFAULT_LIMIT_USD) -> bool:
    """Return True if the auto path should be blocked (already locked, or now locked). Locks when
    realized P&L <= -limit. Only ever reduces activity — never unlocks."""
    if is_locked():
        return True
    if pnl_usd <= -abs(limit_usd):
        lock(f"daily loss limit: session P&L ${pnl_usd:.2f} <= -${abs(limit_usd):.0f}")
        return True
    return False
