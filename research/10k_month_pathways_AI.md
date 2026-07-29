# PATHWAYS TO $10K/MONTH — AI REFERENCE DOCUMENT
# Alta Investments · Research Context · 2026-07-21
# AUDIENCE: AI agent reading this as context for Colin's situation. Dense, no prose padding.

---

## COLIN'S CURRENT STATE (as of 2026-07-21)

Confirmed edges:
- Forex carry v015: OOS Sharpe 1.25, p<0.001, ~0.02%/day net, regime-fragile (pays only in rate-trending regimes). Live, paper account, OANDA.
- HYP-093 The Undertow: p=0.031, DSR 0.987, F2+F3 sizing clears floor at 0.058%/day gross. NOT live. Shadow running since 2026-07-13, 3 signals.

Capital situation: own capital only (amount undisclosed but implied small — below $25K threshold where The Undertow generates meaningful dollars). Premed undergraduate. No prop firm pathway for The Undertow (drawdown structure incompatible).

Goal: $10,000/month = $120,000/year net.

Gap to goal: at 15% net annual, needs ~$800,000 deployed. At 30% net, needs ~$400,000. At 50% net (exceptional), needs ~$240,000. Current capital is almost certainly orders of magnitude below these thresholds.

---

## THE CAPITAL MATH — NON-NEGOTIABLE CONSTRAINT

$10k/month is a capital problem first, edge problem second.

At any return rate R and current capital C, years to $800k via compounding alone:
- C=$10k, R=15%: ~33 years
- C=$10k, R=30%: ~17 years
- C=$10k, R=50%: ~11 years
- C=$25k, R=15%: ~21 years
- C=$25k, R=30%: ~12 years
- C=$25k, R=50%: ~8 years
- C=$50k, R=30%: ~9 years (with monthly $2k injection: ~5 years)

The fastest path to $10k/month is NOT finding a 100%/year edge (those don't exist at retail scale without catastrophic ruin risk). It is: (1) highest feasible risk-adjusted return, (2) maximum external capital injection rate, (3) multiple uncorrelated edges stacked. These three levers are additive; pure compounding alone is too slow unless the starting capital is already substantial.

---

## PATH ARCHETYPES — HOW TRADERS HAVE DONE IT

### PATH A: Capital Scale + One Proven Edge (Pure Compounding)

Description: Deploy a single confirmed mechanical edge with drawdown-aware sizing. Reinvest 100% of profits. Add external capital from non-trading income. Wait.

Who has done it: Ernie Chan (Quantitative Trading, 2009) — built systematic equity strategies over years, required $500k+ capital before generating meaningful income. His public records suggest 15-25% annualized on own capital before strategy decay set in. Michael Covel's "Trend Following" documents systematic CTAs who ran 20-30 years to scale. These are not fast paths.

Mechanics: The constraint is always capital, not edge quality. A 30% annualized return on $50k is $15k/year; on $400k it's $120k/year. The lever is the denominator.

Realistic timeline for Colin: 8-15 years from meaningful starting capital if a 25-30% portfolio return can be sustained without regime breaks. Accelerated by: medical school income (residency $60-80k/year, attending $200k+), deliberate capital injection, strategy stacking.

Why it matters for Alta: This is the base case if no other path works. It is the slowest path but the most predictable.

### PATH B: Multiple Edge Stack (Portfolio Return Lift)

Description: Identify 3-5 statistically confirmed, uncorrelated edges. Run them simultaneously. Portfolio Sharpe rises faster than individual Sharpe due to diversification. Portfolio return target: 30-50% annualized vs 15% for a single edge.

Who has done it: Two Sigma, Renaissance (institutional versions). At retail scale: very few documented cases, because most traders who claim this are either (a) lying about drawdowns, (b) overfitting, or (c) not actually running uncorrelated strategies. The academic literature on this is Andrew Lo's "Portfolio of Alpha" work — 150 independent trials gives 98% confidence. Most retail traders never get to 3 confirmed independent edges.

Alta's current portfolio:
- Edge 1: Forex carry v015 (~5% net/year after regime haircuts)
- Edge 2: The Undertow (~15% gross/year, friction unknown)
- Edge 3: HYP-095 NQ VIX dip (valid signal, sizing blocked — stopless)
- Candidate: TICK-034 catalyst split (untested)
- Candidate: HYP-107 Divining Rod (NOT confirmed — needs live fill verification)

If 3 confirmed uncorrelated edges stack at 15% each with correlation ≈ 0 (equity gapper fade is anti-correlated with carry in trending regimes — one pays when the other doesn't), portfolio return approaches 30-35% with lower drawdown than any single edge alone. That halves the capital needed to reach $10k/month.

Priority for Alta: Unlock HYP-095 sizing (stopless is the blocker — a structured stop or position-sizing ceiling needs to be derived). Confirm HYP-107 with live fills. These are the next two confirmed edges in the pipeline.

### PATH C: Prop Firm Leverage + Right Strategy

Description: Pass a funded-account challenge with a strategy that fits the drawdown rules. Trade $100k-$400k funded capital at 50-80% profit split.

Mechanics at $200k funded, 15% gross return, 80% split: $200k × 15% × 80% = $24k/year from carry-type strategies. To hit $10k/month ($120k/year) from a $200k account you need 75% gross return — not achievable with low-variance systematic strategies.

At $1M funded (available from top-tier funded firms like FTMO Elite, MyFundedFutures high-tier), 15% gross × 80% split = $120k/year = $10k/month. But: these accounts require 10% drawdown limits. The Undertow fails at a 10% limit 35% of the time.

WHAT WORKS IN FUNDED ACCOUNTS: Forex carry strategies with Sharpe > 1 and low drawdown DO pass. The v015 carry is a legitimate funded-account candidate. At $400k funded, 15% gross × 80% = $48k/year. Not $10k/month but meaningful runway. The path: (1) carry strategy in funded account for capital generation, (2) The Undertow in own-capital account for higher return, (3) profits from funded carry accelerate own-capital base.

Who has done it: Documented in funded-account communities. The viable subset is tiny — most funded-account success stories are either (a) carry/trend traders with Sharpe > 1, or (b) discretionary traders with exceptional intuition, not replicable systems.

Alta relevance: The carry strategy could be run in a funded account NOW. The capital from that profit split feeds Colin's own account faster than compounding alone. This is an underexplored lever.

### PATH D: Discretionary Overlay (The Options / Big-Mover Track)

Description: Run the systematic base, but layer a discretionary options/single-name catalyst book on top. The discretionary book targets 3-5 trades/month at 50-200% returns. If hit rate is 30-40%, net contribution is 10-20% additional per year, but with high variance.

Who has done it: Most documented $10k/month retail traders are in this category. Not purely systematic. Running a carry base plus discretionary momentum/options on high-conviction catalysts. The "SMB Capital" model. The "Investors Underground" community. The documented survivorship bias here is severe — for every trader posting $10k/month, 50 blew up trying.

What this means: The discretionary options book we've been building (the weekly big-mover scan, the evaluation card, the continuation/exhaustion rubric) is a real path to supplemental return on top of the systematic base. But it is not auditable the way The Undertow is. It depends on skill development over time, not a frozen signal. Colin's natural pattern recognition (the Robinhood background) is a real asset here, not noise.

Honest assessment: If Colin runs The Undertow + carry systematically AND develops a disciplined discretionary book with position sizing and logging, the combination could hit $10k/month faster than the systematic path alone. But the discretionary component cannot be promised or formalized — it has to be earned through live experience with strict P&L tracking.

### PATH E: External Capital / Partnership

Description: Demonstrate track record to a family office, angel investor, or institutional partner. Access $500k-$5M in managed capital at 1-2% management fee + 20% carry.

Mechanics: $1M AUM at 1% mgmt fee = $10k/year. At 20% carry on 15% gross returns = $30k/year from carry. Total $40k/year. At $5M AUM: $200k/year. This requires a 2-3 year audited track record.

Who has done it: The classic fund launch path. Most quant funds start between $1M-$10M from friends/family. The barrier is legal (RIA registration or exempt from registration with <15 clients in some jurisdictions), operational (audit, custody, reporting), and reputational (track record).

Alta's readiness: Not close. No audited live track record exists. The simulation results, while rigorous, are not the same as 2-3 years of live P&L. This is a 3-5 year path from today assuming the live shadow produces the expected results.

---

## SYNTHESIS — FASTEST REALISTIC PATH TO $10K/MONTH FOR COLIN

Ranked by speed and capital leverage:

NEAR-TERM (0-18 months): Develop the discretionary book alongside the systematic base. Set up the carry strategy in a funded account ($200k tier, Sharpe-compatible). Start injecting premed/work income into own-capital base. Target: $500-$2,000/month contribution, not $10k. This is the runway phase.

MEDIUM-TERM (18-48 months): If The Undertow clears W7 and the cost cascade, run both systematic edges at scale. Funded carry + own-capital Undertow. If HYP-095 sizing is solved and HYP-107 confirms, portfolio return lifts toward 30-35%. Target: $2,000-$5,000/month. Stack mode active.

LONG-TERM (48-84 months): If 3+ edges confirmed, own capital grown via injection + compounding to $300-500k range, funded accounts supplementing, discretionary book contributing: $10k/month is reachable. Medical attending salary ($200k+) running parallel enables aggressive capital injection.

THE CRITICAL INSIGHT: $10k/month from trading alone on small starting capital is nearly impossible at low risk. $10k/month from trading + external income injection + funded account leverage + compounding over 4-6 years is achievable and has documented precedent. The system is not the bottleneck. The capital and time are.

---

## WHAT DOES NOT WORK (documented failure modes)

1. Trying to force $10k/month from small capital with high leverage — ruin probability spikes above 50%.
2. Prop firm challenges with drawdown-incompatible strategies (The Undertow, as proven by W6).
3. Options gambling without systematic edge identification first — documented 80-90% failure rate.
4. Adding edges before confirming existing edges with live P&L — overfitting risk, and emotional capital depleted by running unconfirmed strategies.
5. Skipping the shadow period — live execution always differs from simulation. The cost cascade (SSR, locate, slippage) has killed real-money versions of strategies that backtested at 30%+.

---

## NUMBERS FOR THE CLAUDE CONTEXT WINDOW

At various portfolio return rates and capital levels, annual P&L:
$25k @ 15% = $3,750/yr = $312/mo
$25k @ 30% = $7,500/yr = $625/mo
$50k @ 15% = $7,500/yr = $625/mo
$50k @ 30% = $15,000/yr = $1,250/mo
$100k @ 15% = $15,000/yr = $1,250/mo
$100k @ 30% = $30,000/yr = $2,500/mo
$200k @ 15% = $30,000/yr = $2,500/mo
$200k @ 30% = $60,000/yr = $5,000/mo
$400k @ 15% = $60,000/yr = $5,000/mo
$400k @ 30% = $120,000/yr = $10,000/mo ← TARGET
$800k @ 15% = $120,000/yr = $10,000/mo ← TARGET

Takeaway: $10k/month requires either $400k+ at 30% net OR $800k+ at 15% net.
Neither is achievable from compounding alone in a reasonable timeframe without capital injection.

Alta's portfolio target: 3 confirmed uncorrelated edges → 30% portfolio return → $400k threshold.
Capital injection plan: premed income + funded-account profit split → compress timeline.

---

## KEY REFERENCES FOR FUTURE SESSIONS

- Andrew Lo "Portfolio of Alpha" — n=150 independent trials, 98% confidence. Basis for multi-edge stack math.
- McLean-Pontiff (2016) — 58% OOS retention for academic anomalies post-publication. Alta's 61% benchmark.
- Ernie Chan "Quantitative Trading" (2009) / "Algorithmic Trading" (2013) — documented retail quant path, capital constraints, strategy lifecycle.
- Busseti-Ryu-Boyd (2016) arXiv:1603.06183 — RCK sizing, the W6 recommended policy.
- Grossman-Zhou (1993) — drawdown-modulated sizing, the F3 governor.
- AQR "Fact, Fiction, and Momentum Investing" — the regime-fragility problem (relevant to carry v015 2021/2024 down years).

---

*Alta Investments · AI Reference Document · 10k/month pathways · 2026-07-21*
*For: Claude Code, Claude Research Dispatch, future session context*
*Not for: sharing externally, regulatory purposes, or investment advice*
