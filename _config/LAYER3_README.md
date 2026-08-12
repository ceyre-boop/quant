# Layer 3: Reference Material (Factory)

**Layer 3 holds the factory — stable rules, conventions, and constraints that persist across all runs.**

This directory is NOT edited per-run. These files are:
- **Configured once** during workspace setup
- **Reused across every hypothesis** and research stage
- **Internalized as constraints** by the model (not as inputs/outputs)
- **Sacred** — changes require dated amendments and ratification

## Contents

- **`trading_philosophy.md`** — The six tenets that govern all component decisions. Read before designing new research.
- **`risk_constitution.md`** — Hard capital preservation rules (ratified). No override. Binding.
- **`gate_functions.md`** — Specification of statistical gates (Sharpe threshold, permutation test, OOS degradation). Mechanical.
- **`sizing_model.md`** — Capital allocation and position sizing rules. All live trades use this.
- **`hypothesis_ledger_schema.md`** — Structure of the hypothesis record (verdict, evidence stage, date). Machine-readable.
- **`decision_logger_schema.md`** — Schema for every entry/exit decision (commitment score, market state, size). Oracle reads this.
- **`option_data_catalog.md`** — Data vendor capabilities and caveats (Polygon free tier, Alpaca SIP, delay/coverage gaps).

## Usage Pattern

When you write a Stage CONTEXT.md:
1. Reference these files in the "Inputs → Layer 3" section
2. The model loads them as **constraints**, not as task inputs
3. The model internalizes the rules and checks all work against them
4. You never repeat the rules in a stage — you reference them

Example:
```markdown
### Layer 3 (Reference Material)
- `_config/risk_constitution.md` — Per-trade size caps, drawdown breakers
- `_config/gate_functions.md` — Sharpe gate ≥0.30, permutation p<0.05
```

The model then:
- Reads those files
- Applies them as constraints during execution
- Does not re-explain them in its reasoning (it assumes they're memorized)
- Flags violations explicitly if work diverges from them

## Amendment Process

If a rule in Layer 3 changes:
1. Update the file with a **dated amendment block**
2. Commit with the rationale in the commit message
3. Update all affected Stage CONTEXT.md files to reference the amendment date
4. DO NOT edit stage outputs retroactively — they reflect the rules that were in effect when they ran

Example amendment:
```markdown
## Amendments

- 2026-08-12 — Raised Sharpe gate from 0.25 to 0.30 (permutation test p<0.05 on 100+ trades)
- 2026-07-07 — Initial version (ratified)
```

---

**Golden rule: Layer 3 is read-only for any individual research run. Changes are global and deliberate.**
