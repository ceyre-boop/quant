# Operator Skill Scorecard — Human ↔ System Mapping
**Alta Investments · 2026-07-25 · from the trading-education game (baseline assessment)**

The game measures the *operator's* skill the same way the Elo harness measures the policy's.
Same philosophy, other side of the desk. Each weakness below maps to the system component that
exists to compensate for it — the machine is strongest exactly where the human is weakest,
which is not a coincidence; it's the design.

## Baseline — 2026-07-25

**Rank EDGE HUNTER · +126 Elo · 60% accuracy**

| Skill | Score | Maps to system component | Implication |
|---|---|---|---|
| Expectancy | 83% | `PATTERN_FRAMEWORK.md` (expectancy math, §VI) | Operator can audit the machine's edge claims — keep him in the verdict loop |
| Sizing | 71% | Conviction sizing pipeline, W6 F2+F3, quarter-Kelly caps | Good instincts, but caps stay welded — 71% is not 100% |
| Carry | 50% | v015 carry engine + COT panel (`harvest_2026-07-25/cot/`) + regime gate | The operator's edge understanding is HALF of what the system trades. The nuanced half (when NOT to run it, crowding, regime exits) is exactly what the regime gate + COT features encode. Don't override the gate on intuition |
| Recovery | 43% | RISK_CONSTITUTION Art. 3 ladder + TICK-044 (daily loss halt that never fires) + kill switch | Rules exist because in-the-moment application is the human failure mode. 43% says: the breakers must be MACHINE-enforced, never operator-discretionary — which is also why the Art. 3 observability gap (no position ledger) is a RED item |
| Sharpe | 33% | Deflated Sharpe gauntlet, permutation tests, vol-targeting risk engine | The operator under-weights risk-adjustment; the gauntlet exists so no un-risk-adjusted number ever reaches a decision |
| **Exits** | **0%** | **`exit_machine.py` (frozen) + HYP-071 value-function program** | **The single largest human-machine gap. The frozen exit engine IS the compensation — this score is the empirical argument for why exits stay rules-based and frozen, and why the exit value function keeps getting rebuilt (071 → net recompute → fresh prereg) despite repeated kills** |

## Standing conclusions

1. **Division of labor is now measured, not assumed.** Human: expectancy judgment, edge adjudication, final authorization. Machine: exits, risk-adjustment, breaker enforcement, carry nuance.
2. **The two 🔴 scores (Exits 0%, Sharpe 33%) are precisely the two places the system must never defer to the operator in the moment.** The constitution already says this; the game just proved it empirically.
3. **Training routing:** the adaptive feed escalates on SHARPE and EXITS. Next build: live Claude question feed + post-answer explanation layer (Option B), difficulty keyed off this heatmap.
4. Re-assess after each meaningful block of study; append dated blocks here and in the Obsidian brain (`00-BRAIN/Trading-Skill-Scorecard.md`). The delta between assessments is the operator's own Elo curve, running alongside the machine's.

*Personal/longitudinal copy: Obsidian `00-BRAIN/Trading-Skill-Scorecard.md`.*
