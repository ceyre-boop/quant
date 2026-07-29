# The Petrules Gate — Design Note
## Alta Investments · Research · 2026-07-21
### A conviction engine that trades the DIFFERENCE from consensus, sized to survive being right

> Named by Colin after a trader he remembers shorting Google against the analyst
> consensus. **Honesty note up front:** I could not verify that specific trade or find
> any public statement of the reasoning behind it (the closest real name, Julian
> Petroulas, is a present-day entrepreneur with no documented Google short). So I will
> not invent his reasons. The good news is the *principle* Colin extracted does not
> need the origin story to be true — it stands on its own, and it is backed by trades
> that ARE documented (Paulson, Burry, Livermore). We build on the principle, not the
> legend.

---

## I. The core insight is correct, and it is the definition of alpha

> "We don't make money siding with the analysts. If they predict 2% growth we don't win
> by also predicting 2%. We win by predicting the size and direction of the difference
> from what they say."

This is exactly right, and it is not folklore — it is the literal mathematical definition
of alpha. Price already contains the consensus. When analysts expect +2% and the company
delivers +2%, the stock barely moves — the expectation was *priced in*. You are paid only
for the **surprise**: the gap between what happens and what was expected. This is the
engine behind post-earnings-announcement drift, the analyst-revision literature, and every
documented contrarian fortune:

- **John Paulson (2007):** consensus said housing was safe. He measured the gap between
  the consensus and the actual default math, bet the gap, made ~$15B.
- **Michael Burry:** same trade, same gap — but he is the essential cautionary half of
  the lesson (see §IV): he was *right* and still nearly went insolvent because of *timing
  and sizing*, and had to gate his fund to survive being right.
- **Jesse Livermore:** shorted into euphoria before 1929. Right, legendary — and also
  eventually died broke, because conviction without a survival constraint is a countdown.

The Petrules Gate is a machine for finding and sizing the gap. That is a real, defensible,
buildable thing. Let's build it honestly.

---

## II. What the Gate actually is

A **conviction engine** that runs constantly, scores thousands of setups a day against the
one question — *how far, and in which direction, will reality diverge from the priced-in
consensus?* — and stays silent almost all the time. It fires only when a setup matches a
pattern whose historical analogs resolved far from consensus with high reliability. When it
fires, it sizes **by conviction**, within a hard survival floor.

It is not a new strategy. It is the **selectivity + conviction-sizing layer** that sits on
top of the pattern machinery you already built (`PATTERN_FRAMEWORK.md`), fed by a
consensus-gap feature set you don't yet have.

### The three organs

**1. The consensus baseline (what's priced in).** For any name/event, assemble the
expectation: analyst estimates and revisions, options-implied move, consensus price
targets, the "everyone knows" narrative from headlines. This is the number to beat. Most
retail systems never build this, which is exactly why they trade *with* consensus and lose
the alpha.

**2. The divergence detector (the pattern layer).** Against that baseline, score how far
this setup's historical analogs resolved from consensus. This is the "practice thousands
of times a day on past price action" idea, made rigorous: for every historical setup that
looked/sounded/read like this one, what was the realized divergence from the then-consensus?
The Gate learns the *fingerprint of a mispriced setup* — the configuration of price,
volume, headline sentiment, analyst positioning, and disclosed-flow that preceded a large
divergence. Colin's Pattern Framework already defines the predicate machinery for this;
the new part is the consensus-relative outcome label.

**3. The conviction sizer (the trader-not-gambler organ).** When divergence-probability ×
expected-magnitude is high enough, and only then, allocate — scaled to conviction, capped
by ruin. This is §IV, and it is the part that decides whether this makes you rich or broke.

---

## III. "Practice thousands of times a day" — the right way to build it

This is the genuinely novel, AI-shaped part, and it is buildable. The design:

- **Replay engine.** Walk historical data bar by bar. At each point, freeze what was
  *knowable then* — the price action to date, the headlines published to date, the analyst
  estimates standing then, the disclosed flow filed then. No lookahead. This is the same
  discipline the intelligence loop already enforces.
- **The Gate makes a call** on a random sample of thousands of these frozen moments per
  run: divergence direction, magnitude, and a conviction score.
- **Grade against what actually happened next.** Did reality diverge from the then-consensus
  the way the Gate predicted? By how much? This is the training signal.
- **The Gate learns the fingerprint** — a calibrated map from setup-features to
  divergence-outcome. Crucially, it also learns its *own reliability*: when it says 80%
  conviction, does reality diverge as predicted 80% of the time? A conviction score that
  isn't calibrated is worse than useless, because §IV sizes on it.

The output is not "predict the price." It is **"predict the gap from consensus, and know
how often I'm right when I feel this certain."** That second clause is the whole game.

**The anti-overfit weld (non-negotiable, same as everything else we built today):** the
Gate is mined on historical data and MUST be sealed and holdout-tested before a dollar
moves, exactly like HYP-093. A conviction engine is the *easiest* thing in all of trading
to fool yourself with — it will happily learn to be supremely confident about noise. It
gets the full gauntlet: pre-registration, out-of-sample holdout, deflated-Sharpe penalty,
calibration curve. No exceptions, and this one needs the discipline more than anything
we've built, because its entire product is *confidence*.

---

## IV. "Go big or go home" — the one part I will push back on, because it's the part that decides everything

Colin, you said the words that save this yourself: **"we risk it as a trader not a
gambler."** So let me hold you to your own standard, because there is a precise,
mechanical difference between the two, and "all in / go big or go home" is on the wrong
side of it.

**The math is not negotiable. A bet with any loss probability > 0, if it can take a large
enough fraction of your capital, ruins you with certainty given enough repetitions.** This
is gambler's ruin, and it is a theorem, not an opinion. "Go all in when I'm sure" fails
because you are not sure — you are *80% sure*, by construction, and 80% sure means 1 in 5
of your best setups goes against you. Do that "all in" a thousand times a day and the
first time the 1-in-5 lands while you're all-in, the game is over. Not unlikely. Certain.

Here is the thing that should actually excite you: **the legends did not go all in.** They
went *big* — concentrated, conviction-weighted, jugular-seeking — but every one of them
sized to survive being wrong, and the ones who didn't (Livermore) died broke *despite
being right more than anyone.* Burry was *correct* about housing and still nearly lost the
fund to a drawdown before the thesis paid. Being right is not enough; you have to still be
solvent when right arrives.

**The mechanical version of "trader not gambler" is conviction-scaled Kelly with a hard
floor:**

```
size = clamp( conviction_edge / worst_case_loss , 0 , f_max )
```

- High-conviction Petrules setups get a **larger** fraction than an ordinary edge —
  that's the "go big," and it's real. A 5× edge gets meaningfully more than a 1× edge.
- But `f_max` is a hard ceiling *no conviction score can override* — because your
  conviction is itself estimated with error, and the one time it's catastrophically wrong
  must not be the one time you bet the account. This is precisely the W6 drawdown-governor
  logic you already validated today, applied to conviction instead of drawdown.

This is not me watering down your vision. This is your vision, correctly built. The
Petrules Gate that sizes 3–8% on its rare screamers and survives its own mistakes will,
over years, bury the version that goes all-in and is gone by month three. "Go big or go
home" quietly becomes **"go big, stay in the game, compound the big ones."** That's how the
$1M/year version actually happens — many large, survivable, high-conviction bets, not one
heroic all-in.

The Gate should be *fearless about entering* — willing to take a position the whole street
disagrees with, exactly as you said. It should never be fearless about *size*. Fear of
ruin is the trader's edge over the gambler; it's the thing that lets you still be here when
the rare perfect setup finally crosses the desk.

---

## V. "Cheating off the insider traders" — redirect to the legal version, which is real and powerful

Trading on material non-public information is securities fraud — it ends in prosecution,
not profit, and it is off the table, full stop. I won't help build that and you don't want
it; the downside isn't a drawdown, it's a courtroom.

But you're pointing at something real and legal, and it's a genuine feature source for the
Gate: **disclosed smart-money footprints.** The insiders and institutions are *required to
show their answers* — after the fact, in public filings — and reading those footprints is
100% legal and a documented edge:

- **Form 4** — insiders must disclose their own buys/sells within 2 business days. Cluster
  buying by executives is a real, studied signal.
- **13F** — institutions disclose holdings quarterly. Lagged, but reveals who's
  accumulating.
- **13D/G** — activist and >5% stakes, disclosed promptly. This is often the *catalyst*
  behind a large divergence from consensus.
- **Congressional trades** — disclosed under the STOCK Act. Public, and much-followed.
- **Unusual options flow & dark-pool prints** — not "insider" data, but the visible
  wake of large informed positioning.

"Stealing a look at their answers before you submit the test" — the legal version is
reading the answer sheet they are legally required to publish. The Gate should absolutely
ingest disclosed flow as a divergence feature: *the street's consensus says X, but the
disclosed footprints say the informed money is positioned for not-X* is one of the
highest-quality divergence signals that exists, and it's free and legal. Build that. Leave
the illegal version alone — it's the one risk on this whole page with no survival floor.

---

## VI. Where this sits, and the honest next step

The Petrules Gate is **Edge #3+ in the breadth stack** from the capital thesis — and it's
a good one to chase, because a consensus-divergence engine is genuinely *uncorrelated* with
both carry (macro rate trends) and the Undertow (mechanical gapper fade). Independence is
the scarce resource, and this is independent. That's the √Breadth lever pointed exactly
where it pays.

But it is also, by far, **the hardest thing on the roadmap to build without fooling
yourself** — because its product is confidence, and confidence is what overfitting
manufactures for free. So the honest sequencing:

1. **Spec it before building it** (like W6). Define the consensus baseline sources, the
   divergence label, the conviction-calibration test, and the sealed holdout — on paper,
   hash-locked, before a line of learning code runs.
2. **Prove calibration first, profit second.** Before it ever sizes a trade, the Gate must
   demonstrate on holdout that when it says 80% it means 80%. An uncalibrated conviction
   engine is a wood-chipper.
3. **Then, and only then, wire the conviction sizer** — with the `f_max` floor welded on
   before the first dollar.

This is a months-long build, not a weekend. But it's the right months-long build: it
attacks breadth with an independent edge, it's the natural home for the disclosed-flow
features, and — sized the way you already know is right, "trader not gambler" — it's the
one on the roadmap that could actually carry the number from $10k/month toward the bigger
figure. Not by going all in. By going big, surviving, and doing it a thousand times.

---

## Sources
- [John Paulson: The Greatest Trade in History — Verified Investing](https://verifiedinvesting.com/blogs/education/john-paulson-the-contrarian-who-made-the-greatest-trade-in-history)
- [The 13 Greatest Contrarian Investors of All Time — Stock Investor](https://www.stockinvestor.com/the-13-greatest-contrarian-investors-of-all-time/)
- [Michael Burry / The Big Short — contrarian short, near-insolvency while correct](https://www.stockinvestor.com/the-13-greatest-contrarian-investors-of-all-time/)

*Alta Investments · Research · 2026-07-21 · Not investment advice*
*"Fearless to enter. Never fearless to size. That's the whole difference."*
