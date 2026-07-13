# W2 — Options Microstructure on Gappers (specialist agent brief, 2026-07-13)
Stamp: research input to TICK-033; NOT evidence. Verdict changes the HYP-096 design.

## GO/NO-GO PRIOR: NO-GO on naive long puts (~0.10 survival prior)
The killer is structural, not frictional: **put-call parity transfers the borrow cost into the put
premium** (Atmaz-Basak JFE 2019; Battalio-Schultz JF 2006: synthetic shorts no cheaper even in the
dot-com bubble; Muravyev-Pearson-Pollet JF 2025: borrow fees erase exactly this overpricing class).
Defined risk fixes the SIZING rider, but the position re-pays the locate cost, then adds:
- entry IV 200-600% (GME near-dated printed ~800%); ATM 5td put at 300% IV = ~17% of spot premium
  vs a -6.5% median fade -> median trade LOSES in every modeled scenario (-12% to -39%)
- spreads 20-50% of premium round-trip on hot smallcap chains; displayed size 10-50 contracts
- chain coverage only ~20-35% of our universe (fresh-IPO movers ~0%; optionable = biased
  fallen-angel subsample); monthly-only expiries (nearest ~2.5wk) — no same-day puts exist
- chains go dark during LULD halts — untradeable exactly when the fade accelerates
Revealed preference: the entire professional fade ecosystem trades stock+locates, not puts.

## The salvageable designs (descending prior) — HYP-096 redirection
1. **SHORT CALL VERTICALS** — same fade exposure, defined risk, COLLECTS the 300%+ IV instead of
   paying it (how vol pros express this). Leading candidate.
2. Put DEBIT spreads (sells back most of the inflated vol).
3. Whatever design: restrict to optionable subset AND gate on a FEASIBILITY MEASUREMENT first —
   historical chain-existence rate + 10:35 NBBO snapshots on our actual signal sample (ThetaData
   terminal serves this). If coverage <=25%, the rest may be moot.

## Prereg modeling constants (from measured sources)
Chain existence 20-35%; entry half-spread 15-40% of premium (sens to 50%); entry IV 200-600%
(base 350%); IV decay -10..-25 vol-pts/day, floor ~100%; nearest DTE uniform 0-22td; no exits
during halts. Listing criteria: 7M float / 2,000 holders / 2.4M 12-mo volume / $3-for-3-days
(ISE Options 4 §3(b), Cboe 4.3). Coverage measured 2026-07: <$20 stocks 47% optionable,
<$300M caps 28%, intersection lower.
