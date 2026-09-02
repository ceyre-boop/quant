# HYP-112 — post-shock ATM straddle vs matched control (ten ETFs, 2020-01 → 2026-07)

**VERDICT: INCONCLUSIVE** by the sealed data abort (XLE: 12 priced events < 30). Substantively
the **strongest null of the day** — every component fails with CIs entirely on the wrong side.
Sealed `f2d552448a236cc7` before any event chain was read; hash verified before and after;
ledger `ADJUDICATED`. One run. 74 chains returned ThetaTerminal-side 475 errors (Java
ArrayIndexOutOfBounds on specific DIA/other expirations) and were dropped as unquoted per the
sealed rule; 28.7% of events lost to unquoted strikes overall.

| | shock straddle | control straddle (t−10) |
|---|---|---|
| return on premium, 5-session hold | **−22.1%** | −12.9% |
| implied move (straddle/spot) | 1.75% | 1.37% |
| realized \|5-session move\| | 2.77% | 2.38% |
| realized / implied (median) | 1.37 | 1.47 |

| component | result |
|---|---|
| (b1) delta Sharpe | **−1.05**, date-block CI [−1.86, −0.21] |
| (b2) DSR @1548 | 0.000 |
| (c) folds | **0/15** |
| (g) instruments | 2/10 (EFA, XLE) |
| (x) ex-2020 | −1.12 |
| per year | negative delta every year 2020–2026 |
| (d) on spot | −0.845%/event vs +0.25% floor |

1,025 events priced on both legs; DTE at entry median 8 days; premium/spot median 2.9%.

## What it means

**Magnitude is priced — over-priced.** After a shock, implied vol rises 28% (1.37% → 1.75%
implied move) while realized rises 16% (2.38% → 2.77%). The market reprices the clustering
HYP-109 measured, and then some: the realized/implied ratio is *lower* after a shock than
on an ordinary day. Buying the post-shock straddle at the ask loses 22% of premium in five
sessions against a 13% loss for the unconditional straddle — a −9-point delta with a CI that
never touches zero, negative in every year and 13 of 15 folds... in fact 15 of 15.

The chain of the day, closed: HYP-109 (a) said next-week RV is 1.36× after a shock. UVXY
said the retail vol product doesn't pay for it (roll). HYP-112 says the *direct* instrument
doesn't pay for it either — because the option market already knows. There is no retail
instrument that pays for post-shock magnitude. **Magnitude is conditionable but not
monetisable** at this resolution with these instruments.

## What survives

Only the reversal: the unconditional next-session fade (HYP-111a incumbent negated), one
regime, not improvable by size (HYP-113). And — worth stating — the straddle result is an
argument *for* the fade: an over-priced vol surface after a shock is the same thing as
liquidity being expensive, which is what a next-day fade is paid for providing.

## Constraints honoured

One run. No DTE, strike-offset, hold or control-offset change after the result. INCONCLUSIVE
is the sealed word; the substantive read is labelled as such.
