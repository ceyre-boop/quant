# HYP-110 — overnight partition on ten liquid ETFs

**VERDICT: INCONCLUSIVE (pre-declared data abort).** Dead. Not re-run.

Sealed `d981bf1d43170fe0` before any open/close partition was computed (commit
`a5b3e6b`); hash verified at gate zero and after. Ledger: `ADJUDICATED INCONCLUSIVE`.

## Why INCONCLUSIVE and not a substantive verdict

The sealed ladder puts the data-quality abort first: any instrument with
open == close on >1% of sessions → INCONCLUSIVE. Three tripped it — EFA 1.35%,
EEM 1.59%, XLF 1.62% (yfinance stale-open prints). That rule was frozen before
the run and it is applied as written.

## What the numbers said anyway (descriptive — no verdict weight, no re-run)

Every substantive component failed, and not narrowly:

| | result | sealed bar |
|---|---|---|
| (a) partition, mean(on) − mean(id) gross | +0.0116%/day, CI **[−0.0165, +0.0400]** — includes 0 | would be KILL_STRUCTURE |
| (b1) ΔSharpe overnight_net − incumbent | **−0.242**, CI [−0.62, +0.15] | fail |
| (b2) DSR on delta @1545 | prob **0.000** | fail |
| (c) folds | **4/15** positive, mean −0.26 | null |
| (g) golden rule | **3/10** instruments | null |
| (x) ex-2020 | −0.419 | fail |
| (d) raw | overnight_net +0.0103%/day vs incumbent +0.0290 | −0.0187%/day |
| break-even cost | **0.02 bp** — it loses at essentially zero cost | descriptive |

Incumbent reproduced exactly: +131.7%, Sharpe 0.496, maxDD −33.2%.
Overnight net: +34.8%, Sharpe 0.254, maxDD −24.6%.

**The premium on this set is not overnight.** Gross overnight +0.0203%/day vs
intraday +0.0087%/day — both positive, the split roughly 70/30, and the overnight
leg carries most of the variance (2020, 2022 gap risk). Per instrument the
partition goes both ways: IWM/GLD/XLE are overnight-heavy, TLT/EFA/EEM are
intraday-heavy (EFA overnight is *negative*). The OVERNIGHT-QQQ lineage does not
generalise: QQQ here is on +0.043 / id +0.023, not 5.49 bp / 0.09 bp — the earlier
figure was a different window, and this window says the intraday leg on QQQ is
worth keeping.

Had the abort not fired, the verdict would have been KILL_STRUCTURE by (a). The
abort makes the formal verdict INCONCLUSIVE; the substance is a kill. Either way
the hypothesis is dead and is not re-run on cleaner opens — that would be a new
id, and the descriptive read above gives no reason to register one.

## Constraints honoured

One hypothesis. One run. No parameter changed after the result. Reported as
what the sealed ladder says, with the substantive read labelled descriptive.
