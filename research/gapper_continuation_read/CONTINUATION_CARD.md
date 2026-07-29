# Gapper Continuation-vs-Exhaustion Read — daily decision card (paper)

**What this is.** A discretionary, structured, *loggable* read of an already-moving gapper:
is it **still going** or **done**? Not a prediction of which stock moves — a **current-state
evaluation** of one that already moved, precise enough that the next hour is constrained. Paper
only, no money. The point of the log is to find out whether the read is real *before* any
automation or capital (training gate: measure first).

**The baselines this read must beat (be honest with yourself every morning):**
- **Base rate is FADE.** HYP-093 (sealed): >=50%-by-10:30 gappers fade, median ≈ −6.5% same-day;
  ~70.8% close red (n=3,685). A "still going" call fights a ~70% headwind — it has to earn it.
- **A mechanized version of this card already FAILED.** HYP-092: a mechanized continuation checklist
  did NOT separate (p=0.594, well-powered). So the *only* thing being tested here is whether YOUR
  discretionary read adds information a mechanical rule can't. If it doesn't beat the fade base rate,
  it isn't skill.
- **Mechanistic backing (why it's still worth testing):** the fade concentrates in no-news / soft-news
  attention pumps and **weakens or flips on hard catalysts** (Bali 2011 overreaction; W1 ranks
  "catalyst-reliability split" the #1 test-first prediction). Structure is a proxy for catalyst
  reliability: real catalysts hold structure and continue; pumps climax and fade.

**Highest-value framing:** the read may pay more as a **fade veto** ("don't short this one, it has
gas") than as a continuation-long — because avoiding the fade's 30–100% squeeze tail is worth more
than catching a continuer. Log both directions so we can tell which.

---

## Daily protocol
1. **~10:30–11:00 ET**, pull the day's top movers (any watchlist / Robinhood / Finviz gainers —
   the *list is not the edge*, the read is). Take the ones meeting the frozen filter: **up ≥50% intraday,
   ≥ $2, ≥500K volume by 10:30, exclude M&A/buyout**.
2. Pick **3–5 names** (enough to build a sample; few enough to read carefully). For EACH, at the
   moment you look, record **P_read** (current price) and make the call below. Write the **reason**.
3. **At close**, record the close and mark the outcome. Do NOT revise the morning read.
4. One row per name per day. Two–three weeks ≈ 30–75 rows → enough to see if the read separates.

## The read (score each; the call is yours, the score keeps you consistent)
Mark each tell **+1 (going) / 0 / −1 (done)**. The read is the *reason*, not the sum — but log both.

| Tell | Still going (+1) | Done (−1) |
|------|------------------|-----------|
| **VWAP** | holding above, reclaims dips | lost VWAP, rejecting from below |
| **Prior intraday swing low** | higher lows intact, above it | broke it, lower lows |
| **Volume shape** | steady/rising on pushes, not climaxing | blow-off spike then dry-up |
| **Rejection** | no clean lower-high; buyers defend | clean lower-high rejection held |
| **Range position** | strong vs its own morning range/HOD | fading off HOD, giving back the move |
| **Halt behavior** *(log always)* | resumes and holds/extends | halted up then bleeds on resume |

**Call:** `GOING` / `DONE` / `UNSURE` (UNSURE is allowed and informative — log it).

## Catalyst-reliability dimension (the mechanistically most powerful variable — always log)
- **Catalyst type:** `HARD` (earnings, FDA/PDUFA, contract, guidance) / `SOFT` (PR, minor news) /
  `NONE` (no news — pure attention) / `DILUTION-RISK` (shelf/ATM/cash-burner — fades harder).
- Prediction to check: your `GOING` calls should cluster in `HARD`; `DONE` in `NONE/SOFT/DILUTION`.

## Log schema (one JSONL/CSV row per name — maps to decision_logger so it can feed Oracle later)
```
date, ticker, read_time_et, P_read, pct_up_at_read, vol_by_1030,
call (GOING|DONE|UNSURE), score (−6..+6), reason (free text — the actual read),
catalyst (HARD|SOFT|NONE|DILUTION-RISK), halted_today (bool),
close, ret_read_to_close (= close/P_read − 1),   # signed: + = continued, − = faded
outcome (CONTINUED if ret≥0 | FADED if ret<0 | flag |ret|<2% as FLAT),
fade_pnl_if_shorted (= −ret_read_to_close)        # the fade-veto view
```
`ret_read_to_close` is the one number that scores both framings: continuation = its sign,
fade-veto = whether your `DONE`/`GOING` call would have kept you out of a bad short.

## What 2–3 weeks buys us (then, and only then, a prereg)
1. **Does the read separate?** mean `ret_read_to_close` for `GOING` vs `DONE` — is the gap real and
   larger than the mechanized HYP-092 card managed (which was ~zero)?
2. **Beats the base rate?** do `DONE` calls fade ≥ the −6.5% base rate (read improves fade *entry
   selection*), and do `GOING` calls actually escape the fade (read as a veto)?
3. **Catalyst split holds?** GOING↔HARD, DONE↔NONE — confirms the mechanism, not just the tape.
4. If yes → hash-locked prereg (catalyst-reliability split, discretionary-read variant) on the
   MINING/new data through the standard gate; then scanner + decision_logger + Oracle wiring. If no →
   sealed, logged, no capital. Either way the log is the evidence.

*Relationship to the live program:* this is a NEW signal/filter (per TICK-033's constitution, new
signals need new hypotheses on new data) — it does NOT touch the frozen HYP-093 fade or its execution
program. Its natural payoff, if real, is as the **catalyst-reliability veto** on the fade that W1
flagged as highest-value. Seeds a future hypothesis; not itself evidence yet.
