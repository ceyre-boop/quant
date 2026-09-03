# FX Carry Trade v1 (adv)

**Status: RESEARCH SPEC — not live, not yet sealed as a strategy.** Saved 2026-09-03 at Colin's
request as the named construction that survived the carry program (HYP-115 → HYP-118). The one
component that has not had its own sealed test is marked. Nothing here is a recommendation to
allocate; it is what the ledger supports.

## What it is

Cross-sectional carry with characteristic ranking and unlevered risk parity. No forecasting model, no
fitted parameters, no leverage, no volatility targeting.

| element | rule | evidence |
|---|---|---|
| universe | G10 (45 pairs) and EM×funding (MXN ZAR BRL KRW INR × USD JPY EUR, 15 positions); run as two sleeves | HYP-117 (G10), HYP-118 (EM) |
| score, monthly, at t−1 | z(carry) + z(mom12) + z(value) − 0.5·z(rvol3), z-scored across positions within the month | HYP-117 IC 0.15 (1990–2005 unseen), HYP-118 IC 0.10 (EM unseen) |
| selection | top-5 by \|score\| per sleeve, sign = sign(score) | sealed in both |
| weights | **risk parity: ∝ 1/rvol3, gross = 1 per sleeve** | HYP-118 M6 raw — **descriptive, not the sealed claim → HYP-119** |
| leverage | **none.** No vol targeting. Cap 1× notional per sleeve | HYP-118: vol-targeting levered into 1999/2011/2017; worst month −35% vs −23% unlevered |
| rebalance | monthly, first session; the hold horizon is irrelevant for carry (map §holding) | carry_map |
| costs modelled | swap haircut 30%, 3 bp/leg G10, 6 bp/leg EM | frozen in both preregs |
| crash brake | none in v1 (the VIX>30 ×0.5 rule was part of the failed managed line) | HYP-118 |
| data | FRED only: DEX* spot, IRSTCI01/IR3TIB01 rates, OECD CPI; `research/carry_model/{data_fred,panel,em,m6}.py` | free |

## What to expect (honest, from the unseen samples)

| sleeve | ann | Sharpe | maxDD | worst month |
|---|---|---|---|---|
| G10 ranking top-5, 1990–2005 holdout | +6.8% | 0.80 | −16% | — |
| G10 factor, 2006–2026 | +1.9% | 0.23 | −33% | −16% |
| EM ranking + risk parity, 1997–2026 | +5.6% | 0.63 | −24% | −23% |

Planning numbers: **~+5%/yr, Sharpe ~0.6, a −25% year about once a decade.** On $15k: ~$750/yr,
a −$3,750 year sometime. The premium has decayed across eras (G10 +6% → +1.7%); anchor on the low
number.

## What is proven and what is not

- **Proven (two sealed unseen samples):** the ranking has skill (IC 0.10–0.15, 25–13 of years positive,
  p < 0.001). It picks *which* carry to hold.
- **Not proven:** that the ranking's Sharpe beats plain carry (CI spans 0 in both samples); that any
  timing rule (Fed, drawdown, spread, VIX, vol) improves it out of sample (HYP-117 M3 negative; HYP-118
  managed worse). It cannot pick *when* carry pays.
- **Pending its own seal — HYP-119:** "unlevered risk parity beats equal weight on both sleeves"
  (Sharpe and maxDD deltas with CIs, both universes, n_trials 1643).

## Why the drawdown is what it is

Carry is a crash-risk premium: monthly mean ≈ 1/15 of monthly σ, skew −1.35, kurtosis 7. A Sharpe-0.6
process with these tails produces a −25% drawdown over a decade as a matter of arithmetic. The
drawdown is the risk being paid for; the average year is the payment. Constructions that shrink the
drawdown (timing, vol targeting) have so far shrunk the payment or levered into the crash.

## Before it could go live

1. Seal and run HYP-119 (risk parity claim).
2. Measure OANDA's actual per-pair swap for two weeks with 1 unit — the interbank differential minus
   the broker's take is the whole P&L at retail (TICK-024).
3. Logged `param_change_log` rationale, execution-freeze unlock in NEXT.md, kill switch armed,
   `decision_logger.update_outcome()` on every close.
4. Colin's explicit go, in writing, with the planning numbers above acknowledged.

Lineage: `research/carry_map/REPORT.md` (map), `MODEL_DESIGN.md` (framing),
`data/research/hyp117/VERDICT.md`, `data/research/hyp118/VERDICT.md`.
