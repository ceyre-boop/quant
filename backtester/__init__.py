"""Production backtesting engine for ~/quant.

Bias-free, universal, massively parallel. Built 2026-07-17 to eliminate the
biases catalogued in research/gapper/BACKTEST_BIAS_AUDIT.md — chiefly the
exact-trigger stop-fill assumption that flattered the gapper fade by ~19pt/yr.

Public surface:
    data.get_minute_bars / get_daily_bars   — unified data layer
    engine.run                              — event-level backtest
    audit.audit_run                         — bias checklist (auto after run)
    mc.run_mc                               — block-bootstrap Monte Carlo
    scanner.scan                            — parallel strategy search + FWER
"""
__version__ = "1.0.0"
