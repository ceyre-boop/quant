# HYP-122 — extraction: Dufour & Engle, "Time and the Price Impact of a Trade"

**Read in full 2026-09-29**: all 47 pages. That covers the text, Tables I–VII, Appendices I–III, both figures and all 19 footnotes.
Version read: UCSD Discussion Paper 99-15, "Current Draft: June 1999". It thanks the JF Editor and the referee, so it is near-final.
The published version is *J. Finance* 55(6), Dec 2000, pp. 2467–2498, and table numbers may differ slightly from it.
Source URL and sha256 are in `../references/README.md`.

Every number below is quoted from the paper, with its location. Nothing here is a claim about current markets.

---

## 1. The claim, in one sentence

When the time between consecutive trades is **short**, three things happen:
- a signed trade moves the mid-quote **more** (larger price impact);
- prices adjust to the trade's information **faster**;
- trade signs are **more positively autocorrelated**.

The authors read this as *active markets have more informed traders, so they are less liquid*. That is Easley & O'Hara (1992), and it contradicts the Admati & Pfleiderer (1988) pooling story.

## 2. Data

- **Source:** TORQ (NYSE), Nov 1 1990 – Jan 31 1991, 62 trading days. Nov 23 1990 dropped.
- **Stocks:** the 18 most-traded on day 1 of the sample (Appendix I): BA, CAL, CL, CPC, DI, FDX, FNM, FPL, GE, GLX, HAN, IBM, MO, NI, POM, SLB, T, XON.
- **Size:** from 5,674 trades (CAL) to 71,080 (GE). Mean duration runs from 20.1 s (GE) to 230.6 s (CAL). Mean spread is about $0.15–0.25 (Table I).
- **Cleaning** (§II.A):
  - NYSE quotes only.
  - Same-venue, same-timestamp, same-price prints merged into one trade.
  - The prevailing quote is the last quote **≥ 5 s before** the trade (the Lee–Ready 1991 fix for 1990 reporting lag).
  - Opening trade and overnight change dropped; trades after 16:00 dropped.
- **Trade sign:** mid-quote rule only; trades at the mid are set to 0. Result: 44.8% buys, 35.8% sells, 21.5% unclassified. Footnote 15 notes the tick test as an option but does not use it for the main results.
- **Return:** r_t = 100·(ln q_{t+1} − ln q_t), the mid-quote change *following* trade t. Between 72% and 96% of r_t are **zero**, because the specialist usually doesn't revise the quote.
- **Duration:** T_t = seconds since the previous trade, **+1 s**, so that ln T ≥ 0.

## 3. The model (eq. 9, the estimated system)

A Hasbrouck (1991) bivariate VAR in trade time, 5 lags, estimated by OLS with White HC standard errors.
The trade's impact coefficients interact with ln(duration):

```
r_t  = Σ_{i=1..5} a_i r_{t-i} + λ_open D_t x_t + Σ_{i=0..5} [γ_i + δ_i ln T_{t-i}] x_{t-i} + v1_t
x_t  = Σ_{i=1..5} c_i r_{t-i} + λ_open D_{t-1} x_{t-1} + Σ_{i=1..5} [γ_i + δ_i ln T_{t-i}] x_{t-i} + v2_t
```

- The hypothesis is **δ < 0**: shorter duration means a larger coefficient on the signed trade.
- **Time of day:** with 9 diurnal dummies, lagged dummies are jointly insignificant in 16 of 18 stocks. They were dropped except for a first-30-minutes dummy.
- **Duration process:** diurnally adjusted with a piecewise-linear spline (nodes 9:30, 10, 11, 12, 13, 14, 15, 15:30, 16). It is then fit with a **Weibull ACD(2,1)** (Table V):
  - FNM: ζ = 0.9445, θ = 0.896.
  - IBM: ζ = 0.9457, θ = 0.885.
  - θ < 1 means durations are over-dispersed.
  - The ACD is used **only** to simulate durations for impulse responses. The VAR itself treats duration as strongly exogenous.

## 4. Results, with their strength

| Result | Evidence | Location |
|---|---|---|
| Price impact rises as duration falls | δ_0 < 0 and significant for **13/18** stocks. Σδ = 0 rejected for 13/18, and the sum is negative in **17** of those. | §III.B, Table IV |
| Trade autocorrelation rises as duration falls | δ jointly ≠ 0 for 11/18. Σδ < 0 for 16/18 and significant for 11. Strongest in T and GE, the most active names. | §III.A, Table III |
| Time of day doesn't explain it | Lagged diurnal dummies significant in only 2/18 per equation. The open dummy matters in a few. | §III, Tables III–IV |
| Fast markets adjust faster | FNM, buy shock at 10:00 on 1991-01-17 (fastest) vs 12:30 on 1990-12-24 (slowest): cumulative +0.0689% at 6 min 15 s, **>3×** the slow case. Converges in about 4 min vs >23 min. | §III.C, Fig. 1b |
| Predictability is tiny | Adding duration gives "only small enhancements to the R²". Example: FNM return equation adj. R² = 0.063. | §III.B, Table II |
| **Robustness: weak** | After adding **trade size and spread** to the impact equation (eq. 11), duration is significant for only **8/18** stocks and keeps its negative sign in **6** of them. Volume is significant in 7/18 and spread in 18/18. The authors say directly that *"dynamics of volume and spread predominantly characterize the price impact of trades and… the net effect of time duration is only marginal."* | §III.D, Table VI |
| Duration isn't exogenous | Short durations follow large returns and large trades (ACD residual regression). The paper's exogeneity assumption is therefore violated in-sample. The authors defer this to future work. | §III.E, Table VII |
| No buy/sell asymmetry | "No conclusive evidence of asymmetric price impact of buyer and seller initiated transactions." | fn 16 |
| Off-NYSE trades carry less information | Regional and NASDAQ prints move NYSE quotes less. Trades at the mid (tick-test classified) have smaller effects. | fn 16 |

## 5. What the paper does NOT claim

- **No trading rule, no forecast of direction beyond the trade's own sign, no costs, no P&L.**
- The direction of the impact comes from the *observed trade sign*. The paper says prices move toward the side of a trade that has **already happened**. It does not say which side the next trade will be on, beyond the positive sign autocorrelation. The γ coefficients (e.g. FNM trade eq. γ1 = 0.31) carry the autocorrelation, and short durations raise it further.
- **One regime:** a 3-month window of NYSE floor-specialist trading, with 1/8-dollar ticks and 1990 reporting lags. There is no out-of-sample period.
- The authors' own framing of the practical use is **liquidity measurement and execution timing**, not alpha. In their words, the results "could be used to design optimal trading strategies" in the sense of *how much and how fast prices respond* to trades.

## 6. What changed between 1991 and today (testing needs to handle these)

| 1990 TORQ | 2016–2025 SIP (Alpaca) | Consequence for any test |
|---|---|---|
| Specialist sets quotes | Fragmented, HFT market making across 16 or more venues | "Quote revision after a trade" is now an NBBO update. Most revisions are no longer the specialist's learning. |
| Durations in seconds (mean 20–230 s) | Durations in µs–ms. Sweeps print many trades at one timestamp. | The "+1 s" floor and ln(seconds) no longer make sense. The duration unit and the rule for merging same-timestamp prints must be pre-specified. |
| 5-s quote lag (Lee–Ready) | Timestamps sync to µs/ns | Sign against the prevailing NBBO at trade time with **no** lag (Holden–Jacobsen practice). |
| Tick $0.125 | Tick $0.01; spreads ~1 tick on liquid names | The per-trade impact scale is far smaller, so the cost bar is dominated by the spread. |
| No odd-lot prints | Odd lots reported (condition `I`), a large share of prints | The primary sample must exclude or include them by pre-specified rule. |

Alpaca SIP was checked 2026-09-29 on the existing key. IBM `trades` and `quotes` return tick rows for **2016-06-01** (ms timestamps) and **2025-06-02** (ns timestamps). The fields are:
- trades: `c` conditions, `x` exchange, `z` tape, `s` size, `p` price;
- quotes: NBBO `bp/ap/bs/as/bx/ax`.

So the data exists, at $0.
