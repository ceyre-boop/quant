"""Alta ES/NQ Systematic Daily Intelligence Engine.

Sandbox-local research package. Import policy (enforced by tests/unit/test_es_nq_isolation.py):
  ALLOWED:   sovereign.futures.* (plumbing: bar_feed, ib_bridge patterns),
             sovereign.utils.kill_switch, config/es_nq_params.yml (via .config)
  FORBIDDEN: sovereign.forex, sovereign.intelligence, sovereign.oracle, ict, ict-engine,
             layer1, layer2 — es_nq shares ZERO features with forex/ICT (brief rule #8).

Oracle reads this package's FILES (data/es_nq/session_log.jsonl); this package never
imports oracle. All thresholds live in config/es_nq_params.yml — never hardcoded.

Components:
  daily_bias_engine — 5-input pre-market directional bias (pre-registered weights)
  structure_gate    — AMD sweep + VWAP-reclaim confirmation layer
  session_sizing    — adaptive 3-trade ladder (probe/press|pullback/runner)
  session_logger    — data/es_nq/session_log.jsonl (Oracle harvest source)
  backtest          — session-by-session replay over cached parquets
  live_scanner      — shared intraday logic for the paper runner (time injected)
  data_store        — Databento GLBX.MDP3 pulls + parquet caches + daily table
"""
