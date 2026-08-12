# RISK_RESEARCH.md — Alta Dynamic Risk Engine

The formal contract for the cascading survival layer. (This document was authored alongside the
engine; §1–8 are reserved for the broader risk research narrative. §9 is the architecture, §10 the
build order — the two sections the engine was built from.)

---

## §9 — Architecture (the cascading survival layer)

ONE unified engine is the **sole sizing authority**. `DecisionChain` and the forex scan call
`risk_engine.decide()`; nothing sizes around it. It sizes and constrains — it never generates
signals or executes.

**Compose-only-reduces.** Every operation can only make the system safer:

- **MODULATORS** (volatility L3, drawdown L4, regime L5) output a factor in `[0,1]` and **compound
  multiplicatively** onto the base. Stacking only shrinks.
- **CEILINGS** (Kelly L2, portfolio/CVaR L6, prop L7) output an absolute `risk_pct` and bind via
  **`min()`**.
- **HARD GATES** (L0) force the whole decision to **0** (halt).

```
desired = base × vol_f × dd_f × regime_f
capped  = min(desired, kelly_ceil, portfolio_ceil, prop_ceil)
final   = 0  if any hard gate fires  else  capped
```

**Invariant (enforced by test):** `final ≤ base` AND `final ≤ every ceiling`, always. A bug or bad
estimate in any single layer can only ever make the system SAFER.

**Layers** (each a pure function `(signal, state, config) → float`):

| Layer | File | Kind | Source reused |
|---|---|---|---|
| L0 gates | `layers/gates.py` | halt→0 | daily/DD/health/threat/mc + internal guard (PropRiskManager 2%/5%) |
| L1 base | `layers/base_size.py` | base risk_pct | grade levels (mirror `ict/micro_risk._GRADE_RISK`) |
| L2 Kelly | `layers/kelly.py` | ceiling | `kelly_engine.fractional_kelly` + `hoeffding_win_rate` |
| L3 volatility | `layers/volatility.py` | modulator | ATR/EWMA vol (state) |
| L4 drawdown | `layers/drawdown.py` | modulator | monotonic taper table |
| L5 regime | `layers/regime.py` | modulator | `AlexandrianLibrary` threat_score |
| L6 portfolio | `layers/portfolio.py` | ceiling | correlation heat + empirical CVaR (`monte_carlo_prop.load_pool`) |
| L7 prop | `layers/prop.py` | ceiling | worst-case simultaneous-stop survival math |

All tunable numbers live in `sovereign/risk/config/risk_config.yaml` (zero magic numbers). Every
decision is appended to `data/risk/risk_decisions.jsonl` for Oracle auditing.

**L7 (the survival math):** the max `risk_pct` such that if THIS stop is hit AND every open position
stops out simultaneously (correlated worst case), equity stays above both the daily-loss floor and
the max-drawdown floor (8%/8% FunderPro default, configurable static/trailing) with a safety buffer.
Returns 0 if already at the edge.

---

## §10 — Priority / build order (survival-ordered)

Partial completion still ships a real engine.

1. **Commit 1 [P0]** — state + orchestrator (compound/min) + L1 base + L0 gates + L7 prop + audit +
   `test_invariant`/`test_gates`/`test_prop`. ► A working engine that **cannot blow the prop account**.
2. **Commit 2 [P1]** — L4 drawdown + L2 Kelly + tests.
3. **Commit 3 [P2]** — L3 volatility + L5 regime + tests.
4. **Commit 4 [P3]** — L6 portfolio/CVaR + correlation heat + test.
5. **Commit 5 [P1]** — Monte-Carlo integration "killer" test + `DecisionChain`/forex wiring.

**Killer test result (10,000 bootstrapped real-v015 paths × 120 trades):**
`engine P(breach 8% DD) = 0.00%` vs `naive flat-1% fixed-fractional = 37.7%`. Binding-constraint
distribution: ~55% halt at the 5% internal guard (before the 8% floor), ~45% normal base/modulated.
The engine reduces ruin by **stopping** in deepening drawdown — the survival design working.

---

*Engine never executes. Sizing/assessment only. OANDA stays dry-run.*
