# HYP-093 References (Layer 3)

This folder contains Layer 3 reference material — stable rules that persist across all runs of this hypothesis.

These files are **read-only** for this hypothesis stage. They are configured once during workspace setup and reused across all hypothesis stages.

## Links to Layer 3 Factory

- `../../_config/trading_philosophy.md` — Six tenets governing all component decisions
- `../../_config/risk_constitution.md` — Capital preservation rules (Articles 1–6, ratified)
- `../../_config/gate_functions.md` — Statistical gates (Sharpe, permutation, OOS degradation)
- `../../_config/sizing_model.md` — Conviction-based sizing, carry heat checks
- `../../shared/hypothesis_ledger_schema.md` — Ledger structure for recording verdicts
- `../../shared/decision_logger_schema.md` — Trade entry/exit logging (Oracle reads this)

## Usage

When reading the CONTEXT.md for this hypothesis, it references these files by path. Example:

```markdown
### Layer 3 (Reference Material)
- `_config/gate_functions.md` — Sharpe ≥0.30 DSR-adjusted, permutation p<0.05
```

The model loads the referenced files and internalizes them as constraints during execution.

## Non-Negotiable Rule

**Do not edit files in this directory.** Edits to Layer 3 files affect all hypotheses and must be coordinated as amendments in the Layer 3 files themselves (with dated commit messages and ratification trails).

If you discover a Layer 3 rule is wrong or insufficient:
1. Document the issue in `../../_config/LAYER3_README.md` as an amendment proposal
2. Propose the change to Colin with rationale
3. Update Layer 3 file + all affected hypothesis CONTEXT.md files in the same commit

Example commit message:
```
[Layer3-Amendment] Gate 1 Sharpe threshold: 0.25 → 0.30

Reason: 0.25 was aspirational; 0.30 maps to permutation p<0.05 at n=100 trades.
Affects: All hypotheses in mechanism_validation and hypothesis_testing stages.
Ratification: Colin, 2026-08-12.
```
