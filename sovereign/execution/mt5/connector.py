"""Connector seam (spec §8a) — isolates the Windows-only MetaTrader5 package.

The guard / contract / idempotency / approval logic is identical regardless of how
we reach the terminal (Windows VM, Wine prefix, socket-EA). Only this layer differs,
so it sits behind a `Connector` interface and the platform choice is swappable.

BOTH the bridge and the FOMC-window logger use the SAME connector — a single MT5
connection surface, one place where "am I really talking to a demo terminal" is
answered.

No silent mocking (CLAUDE.md): MT5Connector fails LOUD with exact remediation when
the MetaTrader5 package cannot be imported (e.g. on macOS/Darwin). MockConnector is
for tests ONLY and never masquerades as a real fill.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class ConnectorError(RuntimeError):
    """Raised when the terminal/package/connection is unavailable. Always loud."""


@dataclass(frozen=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    time_msc: int  # broker epoch ms

    @property
    def spread_points(self) -> Optional[float]:
        # In price terms; caller converts to points using symbol digits if needed.
        return None if self.bid is None or self.ask is None else (self.ask - self.bid)


class Connector(Protocol):
    """Minimal surface both the bridge and the FOMC logger depend on."""

    def initialize(self) -> None: ...
    def account_info(self) -> Any: ...
    def symbol_info(self, symbol: str) -> Any: ...
    def symbol_tick(self, symbol: str) -> Optional[Tick]: ...
    def positions_get(self, symbol: Optional[str] = None) -> list: ...
    def order_send(self, request: dict) -> Any: ...
    def last_error(self) -> Any: ...
    def shutdown(self) -> None: ...


# --------------------------------------------------------------------------- #
# Real connector — the Windows-only surface.                                  #
# --------------------------------------------------------------------------- #

_REMEDIATION = (
    "MetaTrader5 package is not importable on this machine.\n"
    f"  platform: {platform.system()} ({platform.machine()})\n"
    "The MetaTrader5 pip package binds to the terminal via a Windows-only mechanism.\n"
    "Run the bridge on the connector host chosen in specs/mt5_bridge.md §8a — Option A:\n"
    "  1. Windows 11 (ARM) VM via UTM on Apple Silicon (or Parallels).\n"
    "  2. Install the MetaTrader 5 terminal; log into The5%ers DEMO/practice server.\n"
    "  3. Inside the VM's Python:  pip install MetaTrader5\n"
    "  4. Enable 'Algo Trading' in the terminal; add target symbols to Market Watch.\n"
    "Then re-run:  python mt5_bridge.py --selftest"
)


class MT5Connector:
    """Thin adapter over the real MetaTrader5 package. Lazy import; fails loud."""

    def __init__(self, *, login: Optional[int] = None, password: Optional[str] = None,
                 server: Optional[str] = None, terminal_path: Optional[str] = None):
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._mt5 = None  # bound in initialize()

    def _import_mt5(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as e:  # ImportError on Darwin, others on partial installs
            raise ConnectorError(f"{_REMEDIATION}\n  underlying error: {e!r}")
        return mt5

    def initialize(self) -> None:
        mt5 = self._import_mt5()
        kwargs: dict[str, Any] = {}
        if self._terminal_path:
            kwargs["path"] = self._terminal_path
        if self._login is not None:
            kwargs.update(login=self._login, password=self._password, server=self._server)
        ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
        if not ok:
            err = mt5.last_error()
            raise ConnectorError(
                f"MetaTrader5.initialize() failed: {err}. Is the terminal running "
                f"and logged in? (server={self._server})"
            )
        self._mt5 = mt5

    def _require(self):
        if self._mt5 is None:
            raise ConnectorError("connector not initialized — call initialize() first")
        return self._mt5

    def account_info(self) -> Any:
        return self._require().account_info()

    def symbol_info(self, symbol: str) -> Any:
        return self._require().symbol_info(symbol)

    def symbol_tick(self, symbol: str) -> Optional[Tick]:
        mt5 = self._require()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ConnectorError(
                f"symbol '{symbol}' not found. Add it to Market Watch in the terminal."
            )
        if not info.visible:
            # Attempt to enable; do not guess a different symbol name.
            if not mt5.symbol_select(symbol, True):
                raise ConnectorError(
                    f"symbol '{symbol}' is not in Market Watch and could not be enabled."
                )
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            return None
        return Tick(symbol=symbol, bid=t.bid, ask=t.ask, time_msc=getattr(t, "time_msc", 0))

    def positions_get(self, symbol: Optional[str] = None) -> list:
        mt5 = self._require()
        res = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return list(res) if res else []

    def order_send(self, request: dict) -> Any:
        return self._require().order_send(request)

    def last_error(self) -> Any:
        return self._require().last_error()

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None


# --------------------------------------------------------------------------- #
# Mock connector — TESTS ONLY. Never used by the CLI. Never fakes a real fill  #
# in production: the CLI constructs MT5Connector, which fails loud off-Windows.#
# --------------------------------------------------------------------------- #

@dataclass
class MockAccount:
    login: int = 1234567
    server: str = "The5ers-Demo"
    trade_mode: int = 0  # DEMO
    balance: float = 100_000.0
    currency: str = "USD"


@dataclass
class MockConnector:
    """Deterministic connector for unit tests. Records order_send calls."""

    account: Optional[MockAccount] = field(default_factory=MockAccount)
    ticks: dict = field(default_factory=dict)      # symbol -> Tick
    positions: list = field(default_factory=list)
    send_result: Any = None
    sent_requests: list = field(default_factory=list)
    initialized: bool = False

    def initialize(self) -> None:
        self.initialized = True

    def account_info(self) -> Any:
        return self.account

    def symbol_info(self, symbol: str) -> Any:
        return {"name": symbol, "visible": True}

    def symbol_tick(self, symbol: str) -> Optional[Tick]:
        return self.ticks.get(symbol)

    def positions_get(self, symbol: Optional[str] = None) -> list:
        if symbol is None:
            return list(self.positions)
        return [p for p in self.positions if getattr(p, "symbol", None) == symbol
                or (isinstance(p, dict) and p.get("symbol") == symbol)]

    def order_send(self, request: dict) -> Any:
        self.sent_requests.append(request)
        return self.send_result

    def last_error(self) -> Any:
        return (0, "ok")

    def shutdown(self) -> None:
        self.initialized = False
