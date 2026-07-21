---
name: give-me-the-numbers
description: Turn a CONFIRMED trading edge into the numbers that decide whether to trade it — annual return distribution, dollar P&L by account size, and a funded-account pass/blow-up overlay. Use when the user says "give me the numbers", "what would this make me", "how does it trade at different sizes", "run the dollar table", "would this pass a funded account", or asks what a validated strategy is worth in real money. Requires a sealed/validated per-event return set and a sizing rule — it resamples reality, it does not invent an edge.
---

# Give Me The Numbers

Takes a **confirmed** edge and answers the only question that matters after the
statistics are settled: *what does this actually make me, and what would it do to a
funded account?*

## When to use this skill

The user has a strategy whose edge is already proven (passed a holdout / gauntlet) and a
sizing policy, and now wants the money view:

- "give me the numbers" / "what would this make me at $25K?"
- "how does it trade at different account sizes?"
- "would this pass a funded challenge?"
- "run the dollar table on HYP-093 / the F2+F3 policy"

Do NOT use this to evaluate an *unproven* or *mined* edge. The engine resamples the
per-event returns it is given — feed it a selection-biased mined set and it will faithfully
report a selection-biased fantasy. The input must be a sealed, out-of-sample event set.

## What it produces

Three tables, all from a bootstrap over the sealed event returns:

1. **Annual return distribution** — p5 / p25 / p50 / p75 / p95, mean, probability of a
   profitable year, and drawdown (median and p95). The **median** is the headline (returns
   are right-skewed; the mean oversells).
2. **Dollars per year by account size** — linear in account, at every percentile.
3. **Funded-account overlay** — for standard evaluation rule sets (target vs max
   drawdown), the probability of PASSING versus BLOWING UP the eval account. The blow-up
   column is usually the real finding: a strategy whose drawdown makes it compound on own
   capital often trips funded-account limits, which is why a funded vehicle may not exist
   for it.

## Honesty disciplines baked in (do not remove)

- **Block bootstrap by default** (5-event blocks) to preserve serial clustering, which
  inflates drawdown and is the honest stress. `--iid` exists but is labelled OPTIMISTIC.
- **Pessimistic disaster mixture** (halt / gap-through / buy-in at 0.5%/event by default,
  scaled by per-tier worst-case) layered on every path.
- **Median, not mean**, as the typical-year headline.
- **It never invents an edge.** Output is only as good as the sealed input.

## How to run it

```bash
python3 skills/give-me-the-numbers/run_numbers.py \
    --events <path to sealed events json> \
    --size-t10 0.0799 --size-t20 0.0673 --locate 0.5 --dd-governor 0.15 \
    --accounts 5000,10000,25000,50000,100000,200000 \
    --label "The Undertow (HYP-093) F2+F3"
```

Flat sizing instead of per-tier: `--size 0.04`.

### Input formats (either works)
- W6 / gauntlet: `{"events": [{"ret_event": -0.12, "tier": "T10"}, ...]}`
- Bare returns: `{"returns": [-0.12, 0.03, ...]}`

### The canonical example (The Undertow, HYP-093, W6 F2+F3 policy)

```bash
python3 skills/give-me-the-numbers/run_numbers.py \
    --events data/research/yield_frontier/optimization/W6_inputs/hyp093_events.json \
    --size-t10 0.0799 --size-t20 0.0673 --locate 0.5 --dd-governor 0.15 \
    --label "The Undertow (HYP-093) F2+F3"
```

This reproduces the 2026-07-21 result: ~+11% median year, 75% of years profitable,
p95 drawdown ~14%, and a ~35% blow-up rate under a standard 10% funded drawdown limit —
the quantitative form of TICK-032's "no funded vehicle" verdict.

## After running

Report the median year, the dollar row for the account sizes the user cares about, and —
if the blow-up rate is high — say plainly that the drawdown that makes the edge work is
the same drawdown a funded account forbids, so it is an own-capital strategy. Do not lead
with the mean or the p95; lead with the median and the drawdown.
