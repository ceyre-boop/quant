# Session Summary — 2026-07-17

A winning day by the only measure that matters: two confirmed edge families
intact, a bug caught before it cost a dollar, and a new live shadow running.

## The confirmed strategies (5 names, 3 independent edges)

| # | Name | Hypothesis | Status | How it trades / why it works |
|---|------|-----------|--------|------------------------------|
| ① | **Sovereign Carry** | HYP-045 / v015 | LIVE, real (regime-fragile) | Long high-rate / short low-rate FX, ~60-day holds, conviction-sized. Paid the interest-rate differential; permutation p<0.001. Only pays in rate-trending regimes. The only strategy trading live. |
| ② | **The Undertow** | HYP-093 | CONFIRMED anchor | Short a parabolic gapper at 10:30, cover at close. Blow-off moves exhaust and fade. p=0.031, DSR 0.99. Borrow-capped (72% of names have no options; locate hardest on best signals). |
| ③ | **The Updraft** | HYP-105 | ~~CONFIRMED~~ **RETRACTED** | Long the gapper 09:31→10:30. **Look-ahead bug — see below.** |
| ④ | **The Divining Rod** | HYP-106 | ~~CONFIRMED~~ **RETRACTED** | Runner filter on The Updraft. **Same look-ahead bug.** |
| ⑤ | **Storm Dip** | HYP-095 | Real but tiny | Long the NQ/Nasdaq dip when VIX is elevated; fear overshoots mean-revert. DSR 0.999, below profit floor. Diversifier, not a standalone earner. |

Net: **2 solid independent families** (Sovereign Carry FX, The Undertow gapper
fade) plus Storm Dip as a small diversifier. The gapper long-side is a real
*direction* but not yet a tradeable strategy (see HYP-107).

## The retraction — HYP-105 / HYP-106 (③ ④)

**Cause: look-ahead in universe construction.** The event universe was defined
by `gain_1030 ≥ 100%` — the price **at 10:30** — but the long strategy **enters
at 09:31**, an hour before that condition is knowable. The 234 events were 255
winners hand-picked from 1,475 candidates *by their 10:30 outcome*; the 1,220
stocks that gapped but didn't moon were never tested. Buying at 09:31 the stocks
we already knew had mooned by 10:30 is circular. The "+50–67% median, Sharpe 3.6,
10:1 tail" was survivorship, and the realistic-fill model "survived" only because
a fake +50% dwarfs any spread. Both verdicts are void.

**This is the system working.** A confident, bug-built edge caught before any
capital rode it is the best possible outcome of the research method.

## The honest result — HYP-107

De-biased: universe re-selected using ONLY 09:31-available info (overnight gap),
including all the non-runners, clean 70/30 date split.
- Blindly buying morning gappers **loses** (median −0.3%) — the fade from the
  other side.
- The filter (moderate overnight gap ≤0.577 + low first-minute volume ≤5.854)
  **still holds out of sample**: holdout gross **median +5.4%, win 70%, tail
  4.4, permutation p=0.0005** (n=57).

Real, positive-skew, mechanically sensible (moderate quiet gaps continue,
climax gaps fade) — but ~10× smaller than the look-ahead fantasy. At +5% gross,
09:31 microcap spreads (1–15%) and LULD halts may eat most of it. **Not
tradeable on backtest evidence — needs live fills to confirm.**

## Live shadow now running

`research/gapper/hyp107_shadow.py` (+ `com.alta.hyp107_shadow.plist`,
tracked-not-loaded, operator promotes) logs hypothetical 09:31→10:30 trades on
every ≥30% gapper that passes the frozen filter — **no capital, no orders**,
tagged `shadow_hyp107`. Records realized 09:31 spread vs the backtest. Target 40
events. Tracking: `data/research/gapper/hyp107_shadow/hyp107_tracking.json`.
Joins the HYP-100 (stop overlay) and HYP-103 (EV config) forward shadows.

## Standing priority order
1. **The Undertow shadow trades** (HYP-093) — the confirmed anchor; keep the
   sealed shadow accumulating toward its N≥40 gate (HYP-100/103 clocks).
2. **HYP-107 shadow** — observe 40 events; does the real +5% survive live 09:31
   fills? Median-return + realized-spread vs backtest is the go/no-go.
3. **Funded-account scaling** — only after the above; borrow/locate (TICK-037 IB
   snapshots) is the binding constraint for the short fade, not signal.

## Ledger movement today
HYP-099 (regime filter, NOT_SIG) · HYP-100/103 (prereg, forward-pending) ·
HYP-104 (down-gap short, overfit NOT_CONFIRMED) · HYP-105/106 (REFUTED,
look-ahead) · HYP-107 (real, execution-unresolved). Plus two megascans
(84,236 hypotheses) — nothing beat the anchors out of sample. Also: production
`backtester/` engine rebuilt bias-free; the corrected honest gapper-fade number
is ~+10–18% / Sharpe ~2, not the biased +24.4% / 3.4.
