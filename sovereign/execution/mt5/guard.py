"""Demo-vs-live guard + live-unlock gate (spec §7) — the load-bearing invariant.

The bridge is physically incapable of routing to a live account without an explicit,
logged unlock. This module is pure logic over an account-info object, so it is fully
unit-testable without a terminal.

Rules (spec §7):
  - account_info() is None                → hard abort (cannot verify DEMO → cannot route)
  - trade_mode == DEMO                     → allowed
  - trade_mode != DEMO, no unlock          → hard abort
  - trade_mode != DEMO, ONLY env OR file   → hard abort (BOTH required)
  - trade_mode != DEMO, env AND file       → live allowed (dead by default; logged upstream)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from . import ACCOUNT_TRADE_MODE_DEMO, TRADE_MODE_NAMES

LIVE_ENV_FLAG = "ALTA_MT5_ALLOW_LIVE"
DEFAULT_UNLOCK_PATH = Path("data/execution/mt5_LIVE_UNLOCK.json")
REQUIRED_UNLOCK_FIELDS = {"rationale", "operator_signature", "authorized_at"}


class GuardError(RuntimeError):
    """Base for guard aborts."""


class NoConnectionError(GuardError):
    """account_info() returned None — cannot verify DEMO."""


class LiveAccountError(GuardError):
    """Connected account is not DEMO and no valid unlock is present."""


def _attr(account: Any, name: str) -> Any:
    """Read a field from a MetaTrader5 namedtuple-ish object OR a plain dict/mock."""
    if isinstance(account, dict):
        return account.get(name)
    return getattr(account, name, None)


def load_unlock(unlock_path: Path | str = DEFAULT_UNLOCK_PATH) -> Optional[dict]:
    """Return a well-formed unlock dict, or None if absent/malformed.

    A malformed or partial unlock file is treated as ABSENT — it must never
    accidentally enable live routing. All REQUIRED_UNLOCK_FIELDS must be present
    and non-empty.
    """
    p = Path(unlock_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    for field in REQUIRED_UNLOCK_FIELDS:
        if not data.get(field):
            return None
    return data


def live_routing_permitted(
    *,
    env: Optional[dict] = None,
    unlock_path: Path | str = DEFAULT_UNLOCK_PATH,
) -> bool:
    """True only if BOTH the env flag is set AND a well-formed unlock file exists.

    Both are required (spec §7). Absent either, live routing is unreachable.
    """
    env = os.environ if env is None else env
    env_ok = env.get(LIVE_ENV_FLAG) == "1"
    file_ok = load_unlock(unlock_path) is not None
    return env_ok and file_ok


def assert_routable(
    account: Any,
    *,
    env: Optional[dict] = None,
    unlock_path: Path | str = DEFAULT_UNLOCK_PATH,
) -> str:
    """Assert the connected account may be routed to. Returns the trade-mode name.

    Raises NoConnectionError if there is no account, LiveAccountError if the account
    is not DEMO and there is no valid unlock. This is called at stage time AND again
    immediately before every order_send (TOCTOU-safe, spec §7).
    """
    if account is None:
        raise NoConnectionError(
            "REFUSING: account_info() is None — cannot verify DEMO, cannot route. "
            "Is the MT5 terminal running and logged in?"
        )

    trade_mode = _attr(account, "trade_mode")
    login = _attr(account, "login")
    server = _attr(account, "server")
    mode_name = TRADE_MODE_NAMES.get(trade_mode, f"UNKNOWN({trade_mode})")

    if trade_mode == ACCOUNT_TRADE_MODE_DEMO:
        return mode_name

    # Non-DEMO account. Only permitted with BOTH env flag AND a well-formed unlock file.
    if live_routing_permitted(env=env, unlock_path=unlock_path):
        # Dead by default. Reaching here means Colin explicitly created the unlock
        # file and set the env flag — a logged, deliberate act (spec §7).
        return mode_name

    raise LiveAccountError(
        f"REFUSING: account {login} on {server} is trade_mode={trade_mode} "
        f"({mode_name}), not DEMO. Live routing requires BOTH env {LIVE_ENV_FLAG}=1 "
        f"AND a well-formed unlock file at {Path(unlock_path)} with a logged rationale. "
        f"Neither should exist for demo trading."
    )
