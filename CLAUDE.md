# SOVEREIGN / QUANT — CLAUDE.md
# Last updated: 2026-05-26 | Repo: ceyre-boop/quant.git

---

## NON-NEGOTIABLES (read these even in compressed context)

1. **ICT/sovereign isolation.** `ict/` and `ict-engine/` must never import from `sovereign/`.
   Enforced by test: `python3 -m pytest tests/ -k test_pipeline_does_not_import_sovereign`
   Cross-layer logic goes through `ict-engine/orchestrator.py` — never `ict/pipeline.py`.

2. **Close the Oracle loop.** Every trade decision wired to decision_logger must receive
   an `update_outcome()` call when the trade closes. Oracle (`sovereign/oracle/reflect_cycle.py`)
   cannot learn without closed-loop outcomes.

3. **Conviction sizing only.** No flat position sizes. All sizing goes through the
   conviction-based sizing pipeline. Check `config/parameters.yml` for thresholds.

4. **No live parameter changes without logging.** Any change to `config/parameters.yml`,
   `config/ict_params.yml`, or risk limits requires a logged rationale before applying.

5. **Read TRADING_PHILOSOPHY.md before any architectural decision.**
   Every component must serve one of the six tenets or it should not exist.

---

## DECISION_LOGGER PATTERN

When wiring a new decision point, always capture:
- commitment score, rate differential, library match, bars since signal
- Use: `sovereign/intelligence/decision_logger.py`
- Wire through `ict-engine/orchestrator.py` (isolation-safe), NOT `ict/pipeline.py`

---

## ORACLE FEEDBACK LOOP

```python
# Entry
decision_logger.log(context)      # capture full entry reasoning

# Exit (REQUIRED — oracle cannot learn without this)
decision_logger.update_outcome(trade_id, outcome)
```

Oracle reads logs via `sovereign/oracle/reflect_cycle.py`.
Skipping `update_outcome()` is silent data loss — oracle degrades without it.

---

## TEST COMMANDS

```bash
python3 -m pytest tests/ -q                  # full suite
python3 -m pytest tests/ -k "not sklearn"    # skip missing-dependency tests
python3 -m pytest tests/unit/test_*.py -v    # unit tests only

# ICT pipeline suite (must stay 21/21)
python3 -m pytest tests/ -k ict -v

# v014 holdout validation
python3 scripts/holdout_validation_v014.py
python3 scripts/run_replay_validation.py
```

---

## COMMIT STYLE

- Imperative mood, present tense: "Add commitment gate to ICT pipeline"
- Prefix with system: `[ICT]`, `[FOREX]`, `[ORACLE]`, `[RISK]`, `[INFRA]`
- Reference hypothesis when applicable: "[FOREX] Wire HYP-044 VIX gate for USDJPY/AUDNZD"

---

## ARCHITECTURE QUICK-REFERENCE

| Layer | Path | Boundary |
|-------|------|---------|
| ICT detection | `ict/`, `ict-engine/` | MUST NOT import sovereign/ |
| Sovereign intelligence | `sovereign/` | Full access |
| Oracle | `sovereign/oracle/` | Reads decision logs, writes lessons |
| Cross-layer bridge | `ict-engine/orchestrator.py` | Only safe ICT→sovereign entry point |
| Decision logger | `sovereign/intelligence/decision_logger.py` | All decision contexts |
| Config | `config/parameters.yml`, `config/ict_params.yml` | Never hardcode thresholds |

Current live version: Forex v015 — anchored as logs/research/v003 (tracked snapshot), HYP-045 4-pair portfolio (AUDNZD excluded).
HYP-045 CONFIRMED 2026-06-02: AUDNZD exclusion → OOS costed Sharpe **1.08** (CI [0.84, 1.32], n=103);
decay 1.61 (ROBUST); p=0.002. v014 baseline was 0.76 (5-pair, AUDNZD OOS Sharpe -0.879 — active drag).
Prior headline 2.097 was uncosted and annualized as if trading daily (these strategies trade 4–14×/yr).
Evidence status (2026-06-02):
- **Forex macro edge: PROVEN real** — permutation test p<0.001. BUT **regime-fragile**: rolling
  walk-forward 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09 (only pays in rate-trending regimes).
- **ICT pattern edge: NOT PROVEN** — permutation p=0.52; fails BH. Treat ICT as unvalidated.
- **AUDNZD removed**: both legs RBA-driven (RBNZ tracks RBA ≤1 quarter) — no independent rate differential.
- HYP-044 VIX gate: rolled back (REJECTED_OOS, p=0.50, delta≈0).
- **Overnight-QQQ (2026-06-07): real STANDALONE edge (VALID_EDGE, net Sharpe 0.574) but REJECTED as a
  carry diversifier** — long-short robustly re-couples with carry in the COVID crash (ρ=0.42, BH p=0.007,
  sig>calm); correlated return-stacking, not a crisis-independent second edge. Do not re-explore as a
  diversifier. (Measured vs DBV proxy, died 2023-03; real v015 overlap thin: n=39, ρ=0.036.)
- Next: v007 per-pair hold implementation (GBPUSD 6d/2.0×, AUDUSD/EURUSD 5d, GBPJPY 5d) toward the 1.5 Sharpe target.
Full system state: `~/.claude/memory/alta-investments-architecture.md`
