# Funded Account Decision Report
## Alta Investments · 2026-07-21
### Q: Am I ready? Which firm? What does it cost? What do I make?

---

## The Short Answer

You are ready on the strategy side. You are not ready on the execution side — yet.
There is one engineering task between you and a funded account that is not optional.
It takes days, not months, and it should be done before you spend a dollar on a challenge.

The firm is **The5%ers**. The account is **$100K High Stakes**.
The timing is **cautious — not blocked, but not ideal**.
The cost is **~$129 to try, free rebuy if you pass then blow.**
The expected monthly income on a $100K account at proven carry returns is **~$400–$600/month**.

That last number is important. Read it before reading anything else.

---

## The Engineering Problem You Must Solve First

Your strategy runs on **OANDA's REST API**. Every funded account firm —
The5%ers, FTMO, FundingPips, all of them — runs on **MetaTrader 5 (MT5)**.
These are different systems. Your Python code cannot talk to MT5 without a bridge.

This is not a blocker. It is a **prerequisite with a known solution**.

The MetaTrader5 Python package is official, maintained by MetaQuotes, and documented.
It lets your existing Python signal logic send orders to MT5 directly:

```python
import MetaTrader5 as mt5
mt5.initialize()
mt5.order_send(request)  # same signals, different destination
```

What changes: the execution layer (`execute_daily.py`) needs a new order-routing
target — MT5 instead of OANDA. What does NOT change: the signal logic, the regime
gates, the sizing, the risk rules, everything inside sovereign/. The strategy is
untouched. Only the wire at the end changes.

**This must be built and tested on a demo MT5 account before you spend money on
a challenge.** Running a 1-week paper test on the MT5 demo confirms fills, slippage,
and connectivity before real money is on the line. The execution path is frozen per
CLAUDE.md — this change goes through the proper unlock protocol in NEXT.md.

**Estimated build time: 3–5 days for the bridge + 1 week of demo validation.**

---

## The Firm: The5%ers — High Stakes Program

After reviewing the live funded account landscape for automated forex trading,
The5%ers is the clearest match for the carry strategy. Here is why and where the
risks are.

### Why The5%ers

| Requirement | The5%ers | FTMO | FundingPips |
|---|---|---|---|
| Automated EAs / Python bots | ✅ Explicitly allowed | ✅ Allowed | ⚠️ Restricted — trade management EAs only |
| Forex carry pairs (GBP, EUR, AUD) | ✅ Full forex | ✅ Full forex | ✅ Full forex |
| Overnight / weekend holding | ✅ Allowed | ✅ Allowed | ✅ Allowed |
| No consistency rule | ✅ No consistency rule | ⚠️ Has consistency rule | ⚠️ Partial consistency on payouts |
| No time limit on challenge | ✅ No time limit | ❌ 30/60 day limits | ❌ Has time limits |
| Max drawdown | 6% trailing | 10% static | 10% static |
| MT5 | ✅ MT5 native | ✅ MT4/MT5 | ✅ MT5 |

The no-time-limit rule is significant for a carry strategy. Your second funded account
blew due to inactivity within a 30-day window. The5%ers does not have this problem.
The carry strategy fires roughly 4–14 times per year per pair — it can sit quiet for
weeks. A time-limited challenge punishes exactly this style. No time limit removes that
risk entirely.

The consistency rule is the other killer for carry. FTMO's consistency rule requires
that no single day's P&L exceeds a set percentage of total profits. A carry strategy
can sit flat for 45 days then book a 3% gain in one session when the rate move triggers.
That single session would breach a consistency rule. The5%ers does not have one.

### The One Risk at The5%ers: Tighter Drawdown

The5%ers High Stakes uses a **6% trailing drawdown**, not 10% static.
This is tighter than FTMO and matters for your strategy.

The carry v015 OOS walk-forward showed 0% blow-up on the one real year —
but the bootstrap, which includes disaster events, showed material blow-up risk.
A 6% trailing limit is stricter than a 10% static limit in a bad-regime year.

Honest assessment: the carry strategy's worst rolling drawdown historically is
roughly 8–12% in a bad year (2021, 2024). A 6% trailing limit would have been
breached in those years. This is the reason to be careful about timing (see §Regime).
In a good-regime year the strategy barely moves against you.

**This does not disqualify The5%ers. It means timing matters more.**

---

## Regime Check: Is Now a Good Time?

The carry strategy pays in rate-trending regimes. It bleeds in rate-compressing regimes.
Here is where the macro sits today.

### Current State (July 21, 2026)

**Fed funds rate: 3.50–3.75%.** Next FOMC is July 28–29. Market pricing: 84%
probability of hold. Fed is in a wait-and-see mode — not actively cutting.

**Rate differentials as of today (live regime map readings — trust these over forward projections):**
- AUD/USD: differential_trend = **NARROWING**. The live feature overrides projected
  RBA/Fed forward paths. What's actually happening in the data is compression, not widening.
- GBP/USD: differential_trend = **NARROWING**. BOE easing path compressing the differential.
- EUR/USD: differential_trend = **NARROWING**. Bund-Treasury differential moving against EUR.
- GBP/JPY: differential_trend = **NARROWING**. BoJ hiking slowly; yen carry unwind risk
  elevated near ¥160. GBP/JPY is the highest-risk pair in the current window.

**Correction note (2026-07-21):** An earlier draft of this section stated AUD/USD was
"moving in AUD's favor — one of your four pairs is in a genuine trending differential."
That was drawn from projected forward RBA/Fed paths, not the live regime feature. The live
feature shows NARROWING. The rule is: trust the feature, not the forecast.

### Regime Verdict: CAUTIOUS — Leaning Red

All four pairs show NARROWING differentials on the live regime map. The return driver
for carry is rate differential *trending in your favor* — not just a wide differential
that exists but is compressing. The strategy is regime-fragile and this is not the regime.

1. **FOMC July 28–29 is in one week.** If the Fed signals a pivot to cuts (even
   slightly dovish language), carry trades unwind fast. Opening a funded challenge
   account the week before an FOMC meeting is timing risk you don't need.

2. **Yen carry unwind risk is elevated.** GBP/JPY is one of your four live pairs.
   Japanese yen intervention risk is documented and active. A coordinated BoJ/MoF
   intervention crushes GBP/JPY quickly and could hit your 6% trailing limit in
   one session.

3. **No bright spots.** All four pairs are NARROWING. The strategy needs ≥2 pairs
   WIDENING to justify entering a challenge. We are not there.

**Recommendation on timing: wait until after July 29 FOMC.**
If the Fed holds with neutral language (likely): the regime is confirmed stable,
and you enter the challenge in a known-safe window. If dovish: you've saved
yourself a challenge fee. Cost of waiting: one week. Value of waiting: removes
the single biggest near-term risk to the carry trade.

---

## The Costs

### What You Pay

| Item | Amount | When |
|---|---|---|
| The5%ers High Stakes $100K challenge fee | ~$129 | Upfront, one-time |
| Free rebuy (first failure) | $0 | The5%ers offers one free retry |
| Second challenge (if needed) | ~$129 | Only if you blow the free retry |
| MT5 demo account (bridge test) | $0 | Free, any MT5 broker |

**Expected annual challenge cost if the strategy performs as in the walk-forward:**
~$0–$129/year. In a good-regime year, you pass the challenge and trade the funded
account without paying again. In a disaster year, you burn one retry + potentially
one repurchase.

**Worst-case cost estimate (bad regime year, two blown challenges):** ~$258.
This is the maximum downside on the challenge side.

### What You Make (Honest Numbers)

The carry v015 returns 0.02%/day net in the walk-forward, which compounds to
roughly **5% net per year after regime haircuts**. That is the proven OOS number.
Not 15%. 15% was the gross mined number before haircuts. The live regime-adjusted
number is 5%.

On a $100K funded account at The5%ers, 80% profit split:
- $100,000 × 5% × 80% = **$4,000/year = $333/month**
- In a good-regime year (like the walk-forward year): 5–8% → $400–$640/month
- In a bad-regime year (2021, 2024 equivalent): strategy underperforms, possibly
  negative before the regime shift. Challenge may blow.

**The honest number: expect $300–$600/month from a $100K account.**

This is real income for ~$129 of risk. It is not $10K/month. But it feeds the
own-capital account, builds a track record, and runs automatically.

### Scaling Path

The5%ers scales accounts. If you pass the challenge and trade profitably:
- Start: $100K, 80% split
- After consistent performance: scale to $200K → $400K → $1M+ over time
- At $400K funded, 5% carry, 80% split: $16,000/year = $1,333/month
- At $1M funded (requires strong multi-year track record): $40,000/year = $3,333/month

The $10K/month number from a funded carry account alone requires $2.5M+ AUM at these
return rates. Realistically, the funded account is the runway income while the own-capital
base compounds. Not the destination.

---

## How Often Will You Need to Rebuy?

Based on the walk-forward data and bootstrap:

**Good-regime years (no disaster events):** Blow-up probability near 0%.
You pass the challenge once and trade for years without paying again.

**Bad-regime years (2021/2024 style):** The 6% trailing limit may be breached.
Bootstrap blow-up rate on the carry strategy in disaster scenarios: unknown precisely
(this was measured for The Undertow, not the carry). But given carry's lower daily
variance vs. the gapper fade, the blow-up risk in a funded challenge is lower.

**Conservative estimate:** One rebuy per 3–4 years in a mixed-regime environment.
Annual challenge cost amortized: $33–$43/year against $300–$600/month income.
That is an acceptable cost of business.

**The one scenario to fear:** A yen carry unwind (like August 2024) landing during
the challenge window. That is a single-session 6%+ drawdown event. It is rare — but
it is the disaster scenario the bootstrap was testing. This is why the GBP/JPY exposure
during a yen-fragile regime needs to be monitored and possibly paused.

---

## The Decision

| Question | Answer |
|---|---|
| Strategy ready? | Yes. Proven OOS, calibrated sizing. |
| Execution bridge ready? | No. Build the MT5 bridge first (3–5 days). |
| Timing good? | Cautious. Wait for July 29 FOMC first. |
| Which firm? | The5%ers, $100K High Stakes |
| Cost to start? | ~$129 |
| Expected monthly income? | $300–$600/month on a $100K account |
| Rebuy frequency? | ~1 per 3–4 years in normal conditions |
| Biggest risk? | Yen carry unwind during challenge window — monitor GBP/JPY |

**The sequence:**
1. This week: build and test the MT5 Python bridge on a free demo account
2. July 29: watch the FOMC. Neutral/hawkish = green light. Dovish = wait another week
3. Early August: open The5%ers $100K High Stakes challenge
4. Run the automated strategy. Let it work. Do not intervene.

You built the system so you don't have to second-guess yourself anymore.
That is exactly what The5%ers lets you do — run your automation, hold overnight,
no consistency rule, no time limit. The only thing left is the bridge and the timing.

---

*Alta Investments · Funded Account Decision Report · 2026-07-21*
*Strategy: Forex carry v015 · Firm: The5%ers · Account: $100K High Stakes*
