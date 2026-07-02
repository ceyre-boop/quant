"""Shadow-window audit harness (Track A). Read-only observers of the L2 shadow run.

Nothing in this package may be imported by live/backtest execution code. The
analyzer imports the shared decide_exit READ-ONLY (parity philosophy: the audit
replays the same kernel the live manager and the backtester call).
"""
