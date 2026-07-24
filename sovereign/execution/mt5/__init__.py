"""MT5 execution bridge — DEMO-only order routing for The5%ers Step 3.

NEW, isolated infrastructure (TICK-056, spec: specs/mt5_bridge.md). This package
consumes a decoupled `order_intent` JSON contract and MUST NOT import anything from
the frozen execution path (forex_exit_manager, decide_exit, execution/harness.py,
carry_engine, ict/pipeline.py). Isolation is enforced by tests/test_mt5_bridge.py.

The demo-vs-live guard (guard.py) is the load-bearing invariant: the bridge is
physically incapable of routing to a live account without an explicit, logged unlock.
"""

# Trade-mode constants (mirror MetaTrader5.ACCOUNT_TRADE_MODE_*; spec §7).
# We define them here so guard logic is testable without the Windows-only package.
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2

TRADE_MODE_NAMES = {
    ACCOUNT_TRADE_MODE_DEMO: "DEMO",
    ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
    ACCOUNT_TRADE_MODE_REAL: "REAL",
}

__all__ = [
    "ACCOUNT_TRADE_MODE_DEMO",
    "ACCOUNT_TRADE_MODE_CONTEST",
    "ACCOUNT_TRADE_MODE_REAL",
    "TRADE_MODE_NAMES",
]
