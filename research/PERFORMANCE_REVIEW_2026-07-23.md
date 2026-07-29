# Performance Review — AlphaZero & Stockfish Halves
## Alta Investments · 2026-07-23 · For: Colin

---

## The Headline

Both halves ran today for the first time as a connected system. Neither is fake. Neither
is lying to you. Both have real findings and real limitations. Here is what you need to
know about each.

---

## AlphaZero Half — The Briefing Synthesizer

### What fired today

The system ran but the Opus brain did not light up. The API key is valid — the account
ran out of credits. What you got instead is the deterministic fallback: the same schema,
real data, but the directional call is NEUTRAL/0 because it refuses to manufacture a
call it didn't actually make. That is the correct behavior. When you top up the credits,
every field stays the same except `synthesis_source` flips from `"deterministic_fallback"`
to `"claude-opus-4-8"` and you get a real probability-weighted directional read.

What the fallback DID produce, from real live data:

**Regime today: ROTATION_WARN.** NQ is weak relative to ES. Correlation is high (0.887)
but NQ is lagging, which means the market is rotating out of tech-heavy names toward
broader indices. Tech-led longs are suspect in this regime. This is a real read, not a
fabricated one — it comes from the lead-lag spread being negative.

**Macro backdrop (from FRED, live):**
GDP growing at 2.1%. Inflation not fully defeated — CPI at 3.23%, core PCE at 2.89%
against a 2% Fed target. Fed funds at 3.63%, 10-year at 4.63%, yield curve NORMAL (not
inverted). VIX at 17.05 — elevated caution but not fear. Consumer sentiment at 44.8 —
that number is low. Historically, readings below 50 indicate the consumer is worried.
It does not crash markets by itself but it matters for the narrative around FOMC.

**Calendar pressure:** FOMC in 6 days. PCE in 8. NFP in 15. The system correctly
flagged all three with size-reduction guidance. This is the synthesizer doing what it
was designed to do — making you aware of event risk before it lands.

**Big move call today (unvalidated — display only):**
LONG USDJPY, probability 73%, confidence 42%, expected range ~0.54%, NY afternoon
session. Drivers: event risk is the dominant factor (FOMC proximity tightening
USD/JPY differential expectations), momentum minor, compression minor. This is labeled
UNVALIDATED — run `scripts/validate_big_move.py` to check against the ledger before
using it. At 42% confidence, size accordingly.

**Forex regime: NO_TRADE_TODAY.** USDJPY is the closest to a signal at 100% proximity
but the carry pairs are not in a trading regime. This is consistent with all four pairs
showing NARROWING differentials in the funded account report.

### The scorecard — 14 calls so far

The system has been running for 14 sessions. Hit rate on directional calls: 0.786.
Regime accuracy: 0.314.

Read those numbers carefully before getting excited or worried about either.

The directional hit rate of 0.786 sounds good. It might be good. But 14 calls is not
enough data to know. You need 30-60 before the mean has any statistical meaning, and
probably 100+ before you would trust it enough to let it size positions. Right now it
is a number being tracked, not a number to act on.

The regime accuracy of 0.314 is the more interesting one. The regime call has three
states — CONCENTRATION, BREADTH, ROTATION_WARN — so a random baseline is 33%. At
0.314 the regime caller is essentially performing at random so far. This either means
it is too early to tell (14 calls, three-class problem), or the regime classification
is genuinely hard and the current features are not sufficient. Watch this number over
the next 30 sessions. If it stays near 33%, the regime calls are not adding information.
If it climbs toward 50%+, there is real signal in the lead-lag regime read.

### What it needs

1. **API credits topped up.** Until then, the directional brain sits idle.
2. **30 more sessions.** The scorecard needs time. Do not let anyone — including this
   system — make sizing decisions based on 14 observations.
3. **No gates, no vetoes.** It is context-only until `provenance.verified` flips true,
   and that flip requires the scorecard to prove calibration over real observations.
   The system is correctly enforcing this on itself. Leave it alone.

### Grade: B (for infrastructure, not results)

The infrastructure is working correctly. The data is real. The fallback is honest. The
scorecard is accumulating. The grade is B rather than A only because the Opus brain
hasn't fired yet and because 14 observations is too early to grade the actual calls.
Check back at n=50.

---

## Stockfish Half — HYP-071 Exit Value Function

### What happened

The harness ran the full locked protocol. Every prerequisite cleared. The pre-registration
hashes verified. The reconciliation gate hit 0.6886 exactly — the same Sharpe the ledger
shows, no drift, no misconfiguration. 459 trades re-traced at 100% parity. The table
was computed.

The result contradicted the pre-registered expectation of NOT_SIGNIFICANT. That is
worth pausing on. The system was designed to fail this test — the researchers wrote
"NOT_SIGNIFICANT — likely the 4th confirmation of the data-ceiling thesis" before
looking at a single result. That kind of pre-registration honesty is rare. And the table
found structure anyway. That does not mean it is real. It means it warrants the careful
reading that follows.

### What it found

Nine cells where the current static exit rules say HOLD_AND_TRAIL but the value table
says EXIT_NOW. Every one of them is CPCV-stable (sign-consistent across cross-validation
splits) and forward-consistent (the same action is optimal in both the 2023-24 and
2025-26 windows). The pattern is coherent:

**High-ATR positions should exit sooner.** In all volatility environments and across
early, mid, and late hold periods, when ATR is in the top tercile, the table says exit.
This makes economic sense: in high-volatility environments, trailing stops get hit for
worse prices, and the probability-weighted continuation is worse than taking profit now.

**Late-hold positions at any ATR should exit.** Even in mid and low volatility, when
the position has been held more than two-thirds of the expected window and the excursion
is underwater or modest (not extended), exit is better than holding.

This pattern is exactly what you would expect from first principles. When the market is
volatile and you are late in the hold window, the expected value of holding another day
is negative once downside deviation is properly penalized.

### Why it is PROVISIONAL and not CONFIRMED

One caveat is load-bearing above all others.

These nine cells are all carry-aligned positions. The table was computed on GROSS
returns — before financing costs. For carry-aligned positions, you are earning positive
carry income every day you hold. That income is a reason to hold longer. If you fold in
correctly modelled financing costs, some of these "EXIT_NOW" calls might flip back to
"HOLD_AND_TRAIL" because the carry income tips the expected value back in favor of
holding.

And here is the problem: the carry cost model in this repo is known to be wrong. TICK-024
documents it — the financing is mis-modelled by approximately 10× on some pairs, with a
sign flip on EUR shorts. Until that is fixed and the table is recomputed on net returns,
the count of surviving EXIT_NOW divergences is not reliable.

The other two concerns are real but smaller. The regime-window robustness check —
which asks whether the result is the same if you use a 252-day ATR window instead of
60-day — came in at 0.854, below the 0.90 bar. The table is somewhat sensitive to how
you define the volatility regime. That is not disqualifying but it is a flag. And the
forward window is thin — only 246 observations across 2025-26, giving 23 common cells
to compare across the two windows. That is enough to be meaningful but not enough to
be decisive.

### What to do

The next step is mechanically clear: recompute the identical table using the corrected
financing model from TICK-024 and count how many of the nine cells survive. That is a
single computation run, not a new experiment — it uses the same locked harness, the same
pre-registration, the same protocol. If all nine survive net-of-carry, the result is
strong and warrants a CONFIRMED stamp. If several collapse, the data-ceiling thesis is
confirmed and the search for exit structure ends here.

**Nothing in the live system was touched.** The rule changes the table implies are staged
in `research/HYP-071_STAGED_EXIT_RULE_PROPOSAL.md` but not applied. The exit machine
stays frozen until July 28 and until you manually stamp the ledger. That is correct.

### Grade: B+ (for the finding, not for certainty)

The methodology is airtight. The harness ran clean. The finding is coherent and
economically sensible. The grade is B+ rather than A because one known cost-model flaw
stands between PROVISIONAL and CONFIRMED, and that flaw could erase the result. The
finding is interesting enough to pursue. It is not interesting enough to trade on yet.

---

## Combined Assessment

| | AlphaZero | Stockfish |
|---|---|---|
| Is it working? | Infrastructure yes, brain offline (credits) | Yes, computed and validated |
| Is the finding real? | Too early to know (n=14) | Provisional — TICK-024 recompute required |
| What does it block on? | API credits + 30+ more sessions | Net-of-carry recompute + Colin ledger stamp |
| Safe to use in sizing? | Context only, not yet | Frozen, not yet |
| When does this change? | n=50, directional hit rate > 55% stable | After net recompute + July 28 |

Both halves are behaving correctly. Neither is over-claiming. Neither is lying. The
AlphaZero half needs time and credits. The Stockfish half needs one more computation
and your adjudication. Both of those are within reach this week.

---

## Action List

1. **Top up API credits.** Every day without Opus is a day the directional brain sits
   idle. This is the cheapest fix with the highest immediate return.

2. **Let the scorecard run.** Do not change anything about how the synthesizer works
   until you have 50 observations. Resist the urge to tweak it early.

3. **Recompute HYP-071 net-of-carry** (after TICK-024 financing fix). One run of the
   same harness. This is the decider.

4. **July 28:** Confirm L2 shadow went live. If clean, it is the natural moment to:
   - Flip the A2 carry multiplier switch (one line of code)
   - Adjudicate HYP-071 if the net recompute has results
   - Push sovereign-v2 to master so the dashboard update shows on the live URL

---

*Alta Investments · Performance Review · 2026-07-23*
*"The infrastructure is honest. The findings are provisional. That is exactly right."*
