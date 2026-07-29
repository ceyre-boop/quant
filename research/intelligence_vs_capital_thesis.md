# Where Intelligence Actually Compresses The Timeline
## Alta Investments · Research · 2026-07-21
### A rigorous answer to: "AI + my system beats the clumsiness of the market"

You're right about part of this and wrong about part of it, and the split matters
more than either half alone. This document draws the line precisely, because getting
it wrong in one direction wastes years and getting it wrong in the other direction
ends the account.

---

## I. The one equation this whole question reduces to

Grinold's Fundamental Law of Active Management:

```
IR  =  IC  ×  √Breadth
```

- **IR** — information ratio, your risk-adjusted return. This is the thing you want big.
- **IC** — information coefficient, the correlation between your forecast and reality.
  Your *skill*. Ranges 0 (no skill) to 1 (perfect). Real quant edges live at IC ≈
  0.02–0.06. Renaissance is rumoured ~0.05–0.10.
- **Breadth** — the number of *independent* bets you make per year.

Every claim about "superintelligence beating the market" is really a claim about one of
these two terms. So let's ask, honestly, which ones AI moves and by how much.

### AI moves IC — but not to 1

Better forecasting raises IC. This is real and you've already done it: the Undertow's
edge is a higher-IC read of a specific inefficiency than a human eyeballing charts. But
IC is bounded hard by **how much signal exists in the data**, not by how smart the
reader is. A parabolic gapper's fade is ~66% likely — that is the *ceiling* of what any
intelligence can extract from that pattern, because the other 34% is genuinely driven by
information nobody had at 10:30. Superintelligence does not raise 66% to 95%; the 34% is
not stupidity, it is irreducible uncertainty. **IC has a data-imposed ceiling that no
amount of compute removes.**

### AI moves Breadth — this is the real lever, and it's the one you're underusing

Breadth is where the market's "clumsiness" actually lives, and it is where your system is
a genuine weapon. Here is why, and here is the catch.

---

## II. Why the market stays clumsy — and why that doesn't mean one big score

The inefficiencies you can exploit **survive precisely because they are too small for big
money to bother with.** This is not incidental; it is the mechanism. The academic
capacity literature is blunt about it: cost and market impact grow *faster than linearly*
with size, so a strategy that nets +15%/yr at $1M nets far less at $100M and is often
negative at $1B. The big, clumsy edge you're imagining does not exist — if it did, a
multi-billion-dollar fund would have flattened it, because they *can* deploy the capital
to do so. What's left for you is a field of **small pools**, each capacity-limited, each
invisible to institutions because harvesting them doesn't move their needle.

The Undertow is exactly this: real edge, ~$900k capacity ceiling. Beyond that the locates
dry up and the fills degrade. That ceiling is not a flaw in the strategy — it is the
reason the edge still exists at all.

**So the intelligence system does not find you one $10M/year edge. It finds you twenty
$50k/year edges that nobody else can be bothered to pick up.** That is a completely
different game, and it is one where AI genuinely changes what a single person can do.

---

## III. The bottleneck AI actually removes: breadth throughput

Read the one line from the capacity research that matters most for you:

> "For retail traders specifically, the main constraints aren't RAM and GPUs — they're
> time: you have a job, family, sleep, and can't baby-sit intraday signals or rebalance
> 20 positions daily."

**The retail constraint has always been time, not intelligence.** Most retail traders
never assemble even three confirmed independent edges — not because they aren't smart
enough, but because each edge takes months of manual mining, validation, and monitoring,
and a human can only run a handful at once. Grinold says IR scales with √Breadth, but a
human's Breadth is capped by attention. That cap is the wall almost everyone hits.

**Your system removes that specific wall.** Today, in one day, the intelligence loop
resolved 8 folklore claims and built + stress-tested a full sizing simulator. A human
analyst does maybe one of those a month. If that velocity is real and repeatable, you are
not "using AI to trade" — you are running an **edge-discovery factory** whose output is
validated, capacity-limited edges, at a rate no solo human has ever matched. *That* is the
"results like never seen before." Not a magic return number — a discovery-and-validation
throughput that turns the √Breadth term from a trickle into a stream.

This is the honest, defensible version of your thesis. It is genuinely exciting and it is
mathematically real.

---

## IV. The wall that stays standing: capital, and it is not intelligence-shaped

Here is the part the pathways doc got right and you're pushing against — and I have to
hold this line, because the failure mode on the other side of it is ruin.

**$10k/month is $120k/year. That requires either $400k at 30% net or $800k at 15% net.**
No term in Grinold's law is capital. IR is a *rate*. Intelligence raises the rate; it does
not manufacture the base the rate applies to. A 30% edge on $10k is $3k/year no matter how
superintelligent the 30% is. **The denominator is a separate problem, and it is the
binding one.**

And the three things that cap the rate itself are all hard:

1. **√Breadth is diminishing.** Going from 1 edge to 4 doesn't 4× your IR — it 2× it
   (√4 = 2). From 4 to 16 is another 2×. The returns to breadth are real but sublinear;
   you cannot brute-force your way to a 100% portfolio.
2. **Independence is scarce.** The √ only works if bets are *uncorrelated*. Most "new"
   edges are the same old beta wearing a hat — your own Discovery-Ledger found 28
   candidates yielding zero *independent* edges. Finding genuinely uncorrelated edges is
   far harder than finding edges. This is the real scarce resource, and it's where the
   intelligence should point.
3. **Ruin is the silent term.** The one place "superintelligence beats the market"
   thinking actually kills people is leverage. Your own W6 result is the proof: the
   Undertow at the *right* size makes ~15%/yr; at a size that would chase $10k/month on
   small capital it blows up 35–57% of the time. Intelligence that ignores the ruin
   constraint isn't intelligence, it's a faster way to zero. The market's clumsiness does
   not protect you from a −60% halt gap-through; nothing does except sizing.

---

## V. So what actually compresses 4–6 years — concretely

Three levers, in order of how much time they buy and how much they depend on the system
you've built:

### 1. Turn the edge-discovery engine into a measured factory (the √Breadth play)
Stop treating edges as one-offs. Make edge-discovery a pipeline with a throughput metric:
**validated independent edges per quarter.** You have 1.5 (carry + Undertow, partially
correlated). The target isn't "a better edge" — it's *N independent edges* so the
portfolio IR rises by √N and the drawdown of any one stops mattering. This is the thing
your system does that no human competitor can, and it's currently un-productionised. The
intelligence loop should be pointed here, hard.

### 2. Aggregate capacity across many small edges (the deployable-capital play)
One edge caps at $900k. Ten uncorrelated edges each capped at $200k is $2M of deployable
capacity at a portfolio return that's *higher and smoother* than any single one. The
intelligence system's job is to keep the pipeline full so that capacity, not ideas, is
what you run out of. This directly attacks the capital wall from the return side.

### 3. Sell the byproduct, not just the trades (the capital-injection play — the real accelerator)
This is the lever the pathways doc missed entirely. **The binding constraint is capital,
and compounding is too slow to build it — but trading is not the only thing your
intelligence system produces.** It produces validated methodology, reusable tooling
(the give-me-the-numbers skill, the claim harness, the sizing simulator), research other
traders would pay for, and a documented track record of *rigor* that is itself rare. That
output can generate income *now*, injected straight into the capital base, compressing the
one timeline that compounding can't. A person who can build what you built today has an
income stream that isn't gated on 4 years of P&L. That is how the capital wall actually
falls faster — not by out-trading it, by out-*producing* it and injecting the proceeds.

---

## VI. The honest synthesis

Your instinct is half right, and it's the important half:

- **Right:** the market is clumsy, the clumsiness is real and harvestable, and your
  system can harvest it at a *breadth* no solo human ever has. The novel thing — the
  "never seen before" — is your edge-discovery-and-validation velocity. That is a genuine
  superpower and you should lean into it completely.
- **Wrong, and dangerously so if unchecked:** intelligence does not repeal the capital
  math or the ruin constraint. There is no size of brain that turns $10k into $10k/month
  fast at low risk. Anyone (or any AI) who tells you otherwise is selling the exact
  failure mode your own doc lists first.

The reframe that makes this energizing instead of deflating: **the bottleneck moved.** It
used to be "can you find even one real edge" — a wall most people die against. You're past
it. The new bottleneck is *breadth throughput × capacity aggregation × capital injection*,
and every one of those three is a thing your intelligence system is genuinely built to
attack. That's not a 4–6 year grind by default. How fast it goes depends on how many
independent edges the factory can validate and how much of its byproduct you can turn into
injectable capital — both of which are, for once, actually intelligence-shaped problems.

Point the superintelligence at breadth and byproduct. Not at leverage. That's the whole
game.

---

## Sources
- [The Capacity of Trading Strategies — AEA](https://www.aeaweb.org/conference/2016/retrieve.php?pdfid=21020&tk=BGQnasd4)
- [Fundamental Law of Active Portfolio Management — AnalystPrep](https://analystprep.com/study-notes/cfa-level-2/state-and-interpret-the-fundamental-law-of-active-portfolio-management-including-its-component-terms-transfer-coefficient-information-coefficient-breadth-and-active-risk-aggressiveness/)
- [Grinold (1989): The Fundamental Law of Active Management — Blank Capital Research](https://blankcapitalresearch.com/learn/grinold-fundamental-law-active-management)
- [The Fundamental Law of Active Management: Redux — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0927539817300543)
- [The Most Realistic Quant System for Retail Traders — Medium](https://medium.com/jin-system-architect/the-most-realistic-quant-system-for-retail-traders-isnt-the-sharpest-knife-it-s-the-one-that-826da1071acc)

*Alta Investments · Research · 2026-07-21 · Not investment advice*
