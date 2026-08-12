# Financial Freedom — The Alta Investments Roadmap
## Colin Eyre · Alta Investments · Updated 2026-07-22
### Goal: $10,000/month net income from trading

---

## The Math First (Non-Negotiable)

$10k/month = $120k/year.

| Capital | Return needed | Achievable? |
|---|---|---|
| $50k | 240% | No — ruin territory |
| $100k | 120% | No — not at retail scale |
| $200k | 60% | Extremely unlikely systemically |
| $400k | **30%** | **Yes — proven at institutional scale** |
| $800k | **15%** | **Yes — Alta carry is at 1.25 Sharpe** |

The bottleneck is capital, not edge quality. The system already has the edge. The job now is building the capital base while the system compounds.

---

## The Three Levers (All Must Run Simultaneously)

**Lever 1 — Systematic edges compounding.**
Every confirmed edge running 24/7, no intervention. Carry v015 + The Undertow + whatever gets confirmed next. No edge runs until it clears the gauntlet. No exceptions.

**Lever 2 — External capital injection.**
Premed → residency ($60-80k/yr) → attending ($200k+). Every dollar not spent on living goes into the trading account. This is the timeline compressor. Without injection, compounding alone takes 15-30 years. With injection, 4-7 years.

**Lever 3 — Funded account leverage.**
Run the carry strategy in a prop firm ($100k-$400k funded capital at 80% profit split). This generates $300-$1,600/month immediately, before own capital is large enough to matter. That income goes back into the own-capital base.

---

## The Steps — Sequenced

### STEP 1 — Data and Observability (NOW — should have been done first)
**Status: INCOMPLETE**

The dashboard is showing $0. The shadow is running but nothing writes the P&L out. This step was supposed to be done before anything else. It is the foundation that makes every other step verifiable.

**What needs to happen:**
- [ ] Writer script: shadow signals + paper P&L → `prop_account_balance.json` (1 day build)
- [ ] Carry shadow visible on dashboard with real daily P&L
- [ ] Undertow shadow visible on dashboard — 0 signals is correct, but it should show confirmed 0, not blank
- [ ] Every number on every dashboard is real or explicitly labeled PAPER

**Why it was Step 1:** You can't watch something pass a funded account if you can't see it trading. You can't trust the system if you can't verify it. Data first, always.

---

### STEP 2 — Simulated Live (NOW → until FOMC and regime flip)
**Status: NOT STARTED**

Both edges should be trading simulated live right now on paper. Not waiting for a good regime. Not waiting for anything. The system runs every day, takes trades when signals fire, logs every result, and the dashboard shows the live forward P&L.

**What "simulated live" means:**
- Real entry prices (market prices at signal time, not backtest prices)
- Real fills modeled with realistic slippage
- Real sizing at F2+F3 policy (4% notional for Undertow, 0.3% risk per pair for carry)
- Real P&L tracking with drawdown monitoring
- Dashboard updates every session

**Why you want this even in a bad regime:**
- You watch the system correctly go flat on bad days. That IS the system working.
- The 3/4 carry pairs NARROWING right now is real information. Watching the strategy stay patient teaches you more than any backtest.
- The Undertow fires maybe 3 times a month. When it fires you need to see it live, not discover it the next day.
- The W7 gate requires live forward data. You can't clear W7 without it running.

**Specific deliverable:** Two paper accounts visible on the dashboard. Carry paper account. Undertow paper account. Both updating daily. Both showing drawdown, P&L, trades taken, and trades vetoed with the veto reason.

---

### STEP 3 — Funded Account Challenge (AFTER July 29 FOMC + regime flip)
**Status: WAITING**

Firm: The5%ers. Account: $100K High Stakes. Cost: ~$129.

**Gate conditions before spending $129:**
- [ ] MT5 Python bridge built and 1-week demo validated
- [ ] FOMC July 28-29 passed with neutral/hawkish language
- [ ] At least 2/4 carry pairs showing WIDENING on live regime map
- [ ] Simulated live (Step 2) running cleanly with no execution errors

**Expected income:** $300-$600/month on $100k account. Goes directly into own-capital base.

**Scaling path:** The5%ers scales to $200k → $400k → $1M on proven performance. At $400k funded + 5% carry + 80% split = $16k/year = $1,333/month from the funded account alone.

---

### STEP 4 — Stack the Edges (Ongoing)
**Status: IN PROGRESS**

| Edge | Status | Next Gate |
|---|---|---|
| Forex carry v015 | CONFIRMED, paper live | W7 live shadow |
| The Undertow (HYP-093) | CONFIRMED, F2+F3 sized | W7 live shadow |
| HYP-095 NQ VIX dip | VALID signal, sizing blocked | Solve stopless problem |
| Petrules Gate | SPEC written, not built | Phase 0 data audit |
| ICT daily pipeline | DISPATCH written | Dispatch build |

Target portfolio with 3 confirmed uncorrelated edges at 15% each: ~30% portfolio return. That cuts the capital needed to $400k instead of $800k.

---

### STEP 5 — Capital Injection Engine (Parallel, Always)
**Status: ONGOING**

Every dollar of non-trading income goes into the account. No exceptions once the system is verified trustworthy (that's Step 1 and 2).

Timeline with injection:

| Scenario | Monthly injection | Years to $400k |
|---|---|---|
| Premed/undergrad | $0-500 | 15+ years (compounding only) |
| Working + residency | $1,500-3,000 | 7-10 years |
| Attending + carry funded | $5,000-8,000 | 3-5 years |

The funded account (Step 3) is itself a capital injection mechanism. $300-600/month from the funded account reinvested into own capital compounds the base without requiring outside income.

---

### STEP 6 — Discretionary Overlay (When System Is Proven, Not Before)
**Status: LEARNING PHASE**

The weekly big-mover scans, the options watch, the intraday mover cards — these are building the pattern recognition that becomes a real discretionary book. Not tradeable with real money until:
- Steps 1-3 are running cleanly
- At least 6 months of logged discretionary paper trades with honest P&L
- Win rate and expectancy measured on live paper, not recalled from memory

The Robinhood background is a genuine asset. The continuation/exhaustion rubric is real pattern recognition. This becomes Lever 3 eventually — but only after the systematic base is verified and the paper book has a measured track record.

---

## The Honest Timeline

| Phase | Timeframe | Monthly income target |
|---|---|---|
| Data + simulation live | Now → Aug 2026 | $0 (learning phase) |
| Funded account challenge | Aug-Sep 2026 | $300-600/month (funded carry) |
| Two edges live + funded | Sep 2026-2027 | $500-1,500/month |
| Three edges + capital building | 2027-2028 | $1,500-4,000/month |
| Full stack + capital injected | 2029-2031 | $5,000-10,000/month |

This is honest. It is not fast. It is achievable. The system is built correctly. The capital and time are what remain.

---

## The One Thing That Kills This

Abandoning the process during a bad regime.

2021 was a bad year. 2024 was a bad year. In both years the correct action was to stay patient and let the system veto trades. The traders who blew up in those years were the ones who decided the strategy was broken and started overriding it.

The regime map, the philosophy gate, the veto ledger — these exist specifically so that when the bad year comes, the system says WAIT and you believe it because you can see exactly why.

---

*Alta Investments · Financial Freedom Roadmap · 2026-07-22*
*"The system is not the bottleneck. The capital and time are."*
