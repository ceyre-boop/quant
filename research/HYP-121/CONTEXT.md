# HYP-121 — Malhotra (2018) microstructure guidance: source intake

**Layer 2: Stage-Specific Hypothesis Definition**

*Created: 2026-09-28*

---

## Status: SOURCE_INTAKE — this is not yet a hypothesis

No prereg. No hash lock. Nothing here is a claim that an edge exists.
This file exists so a future session does not re-litigate the source from scratch.

**Do not promote to `hypothesis_testing` until the Open Question below is closed.**

---

## The source

Malhotra, Yogesh. "Guidance to a Goldman Sachs alumnus Hedge Fund with $400 Billion–$500
Billion AUM: Alpha Trading Strategies Analysis, Maximizing Alpha for Hedge Funds, and High
Frequency Econometrics for Analyzing Price Impact of Trades, Liquidity, and Market
Microstructure." 26 Dec 2018. SSRN 3306817. DOI 10.2139/ssrn.3306817.

- 8 pages · self-posted to SSRN · **not peer reviewed**
- 1,319 downloads / 2,530 abstract views / **0 citations** as of 2026-09-28
- Author: Global Risk Management Network LLC

## Retrieval status — INCOMPLETE, and this matters

**The 8-page body was not extracted.** SSRN returned 429 to automated fetch and served a
Cloudflare interstitial to the browser path; the PDF opened outside the controllable tab
group. What follows is derived from the abstract, keyword list and reference list only.

Per `CLAUDE.md` "No silent mocking" — this is recorded rather than papered over. Any session
that promotes this hypothesis **must first obtain and read the full 8 pages** and replace this
section with what the body actually says.

## What the abstract claims

An engagement report. Recommendations stated to rest on three inputs:

1. Alpha trading strategies analysis across 400 trading strategies
2. Post-crisis strategies for maximizing hedge fund alpha
3. High-frequency econometrics for price impact of trades, liquidity, market microstructure

Framing claim: liquidity and volatility drove structural microstructure shifts post-2008 that
impair hedge-fund portfolio alpha.

## Prior expectation: LOW

Stated up front so it cannot be quietly revised upward later.

1. **No testable specification is visible.** The abstract describes recommendations, not a
   rule. Nothing in it names an entry, an exit, a horizon, or a universe. A document with no
   falsifiable claim cannot produce a verdict — `TRADING_PHILOSOPHY.md` Tenet 1.
2. **Zero citations in seven years.** Weak signal, but not nothing for a 2018 paper on a
   heavily-worked topic.
3. **The headline AUM figure does not describe any hedge fund.** No hedge fund has run
   $400–500B; the largest peaked near $160B. The number is plausibly firm-wide or notional.
   Not disqualifying, but it is the kind of framing that warrants reading the body closely
   rather than trusting the summary.
4. **This repo has already been into this exact territory.** See below — that is the
   strongest reason for a low prior.

## The overlap that actually decides this

**HYP-119** (`16ecff37`, adjudicated 2026-09-14) tested liquidity-cascade state on liquid
names: next-30-min move **2.7×** (CI [2.57, 2.87]) — and **riding it lost −0.28%/trade in
both directions**. Verdict **MAGNITUDE_ONLY**.

**HYP-120** (`894aafbf`) reduced to EWMA on daily vol / trailing RV on minute range, **no
direction** (hit rate 51.5% = drift).

Both say the same thing: in this repo, microstructure and liquidity state have repeatedly
produced **real magnitude signal and zero tradeable direction**. Malhotra's paper is squarely
in that territory. The default prior for anything derived from it is therefore
**MAGNITUDE_ONLY, not an edge** — and the burden is on the extraction to show otherwise.

`research/EDGE_LEDGER.md`: "There is currently no proven retail-clean equity/ETF edge in this
repo." Nothing here changes that line.

## Where the real content is

> **2026-09-29:** the Dufour–Engle lead below was picked up as its own folder, `research/HYP-122/` (paper read in full, scoped, prereg drafted). Malhotra's body is still unread; this folder's Open Question is unchanged.

The paper's substantive value is its reference list, not its recommendations. Of the 14
references, the one that carries actual testable machinery:

- **Dufour, A. & Engle, R.F. (2000), "Time and the Price Impact of a Trade," *Journal of
  Finance*** — the genuine, heavily-cited result. Price impact of a trade is a function of
  the time between trades; impact rises as trade intervals shorten. This is falsifiable,
  well-specified, and does not depend on Malhotra's paper at all.
- Aldridge, I. (2009), *A Practical Guide to Algorithmic Strategies and Trading Systems*
- Boehmer, Broussard & Kallunki (2002), *Using SAS in Financial Research*
- Brocklebank & Dickey (2003), *SAS for Forecasting Time Series*
- (10 further references not enumerated on the landing page)

Two of the four visible references are SAS methods manuals — consistent with the paper being
a methods survey rather than an empirical result.

---

## Open Question — the gate on this folder

**Does the 8-page body contain a single falsifiable claim with a named universe, horizon and
decision rule?**

- **No** → close as `GRAVEYARD / NOT_TESTABLE`. Keep the Dufour–Engle lead as its own
  hypothesis on its own merits. Do not carry Malhotra's framing into it.
- **Yes** → write the prereg against **that claim**, seal it, and run once. The prereg must
  state `prior_expectation: MAGNITUDE_ONLY` given HYP-119/120.

## Process to close it

1. Obtain the full PDF (SSRN direct, or the ResearchGate mirror at publication 330264026).
2. Read all 8 pages. Record in `output/extraction.md`: every named method, every stated
   decision rule, every empirical number and whether a source is given for it.
3. Adjudicate the Open Question in `output/verdict.md`. One paragraph, explicit.
4. Update the ledger entry. If NOT_TESTABLE, say so and stop — a null is recorded with the
   same care as a pass.

**Do not** build infrastructure against this before step 3. Building is unrestricted;
ignition is not (`RISK_CONSTITUTION.md` Art. 6).

## Success criteria for this stage

- [ ] Full 8-page text obtained and read
- [ ] `output/extraction.md` written — methods, rules, numbers, sourcing
- [ ] Open Question adjudicated in `output/verdict.md`
- [ ] Ledger entry updated with the verdict
- [ ] If TESTABLE: prereg drafted, hash-locked, `prior_expectation: MAGNITUDE_ONLY`

## Inputs

**Layer 3:** `_config/trading_philosophy.md` (Tenet 1), `_config/gate_functions.md`,
`_config/risk_constitution.md` (Art. 6), `shared/hypothesis_ledger_schema.md`

**Layer 4 / cross-reference:** `data/research/hyp119/result.json`,
`data/research/hyp120/result.json`, `research/EDGE_LEDGER.md`,
`research/HYPOTHESIS_LESSONS.md`
