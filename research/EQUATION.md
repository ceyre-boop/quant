# THE EQUATION — what the market pays, written as one line with every symbol defined

**Status: v1, 2026-10-10. The top of the ledger.** Every term below names the file that measures it
and the current value, or says "unmeasured" out loud. Nothing on this page is a claim of edge; the
claims live in `research/EDGE_LEDGER.md`. This page is the thing Navier–Stokes had and trading
doesn't: the equation written down first, with everyone agreeing what each variable means, so that
"solve it" becomes "measure each term, then find which one you can move."

Rules for this page: a term is either (a) measured, with a file path and value, (b) a pure
definition, or (c) marked **UNMEASURED**. Values are re-stated here only when they change in their
source; the source wins on any disagreement. Corrections are appended, never overwritten.

---

## 1. The equation

Account growth over a period is the sum, over every trade taken, of size times realised return
minus the cost of taking it:

```
G  =  Σ_i  q_i · R_i  −  C_i                                              (1)
```

In expectation per year, with the controllable terms separated:

```
E[G]  ≈  f · ( w·W − (1−w)·L − c ) · q̄                                     (2)
          └─┘   └──────────────────┘   └┘
        frequency      edge per trade   size
```

Subject to the constraint that isn't in the mean at all:

```
survive( sequence of R_i , q )                                            (3)
```

Equation (2) is what you optimise. Equation (3) is what kills you while you do it. The literature
calls (2) "expectancy" and (3) "risk of ruin"; both are old. What is not old is writing every
symbol against this repo's own measurement.

---

## 2. The symbols

### Per trade (equation 1)

| symbol | definition | set by | measured where | current value (v015 carry) |
|---|---|---|---|---|
| `R_i` | realised fractional return of trade i, `(p_exit − p_entry)·side / p_entry`, **after** costs | signal × exit × sequence | `data/proof/backtest_trades_v015_2015_2024.csv` col `pnl_pct` | n=411, 2015–24 |
| `q_i` | size of trade i as a fraction of equity at risk | sizing | `sovereign/risk/layers/base_size.py::base_size`, `sovereign/risk/kelly_engine.py`; conviction inputs via `sovereign/intelligence/decision_logger.py::log` | cap 0.75%/trade (RISK_CONSTITUTION Art. 1); backtest used 1.0% (`MAX_RISK_PER_TRADE_PCT` in `data/proof/v015_manifest.json`) — **the backtest and the constitution disagree by 1.33×** |
| `C_i` | cost of trade i: spread + slippage×2 + commission + financing(hold) | broker + order type + hold | `sovereign/forex/forex_backtester.py::_apply_costs` (L524) | see §3 |

### Per year (equation 2)

| symbol | definition | set by | measured where | current value (v015 carry) |
|---|---|---|---|---|
| `f` | trades per year | **signal** (threshold, gates, universe) | `data/proof/backtest_equity_v015_2015_2024.json` `stats.n_trades / stats.years` | **41.4/yr** portfolio (411 / 9.92), ≈10/pair/yr. (CLAUDE.md's "4–14×/yr" is the per-pair live cadence; use 41 for the book.) |
| `w` | hit rate, `P(R_i > 0)` | signal × exit | same file, `stats.win_rate` | **0.487** (IS+OOS 2015–24) · 0.527 (OOS 2023–24) · 0.482 (fresh 2025–26, n=56) |
| `W` | mean win, `E[R_i | R_i>0]` | **exit** (hold, trail, stop) | computed from `backtest_trades_v015_2015_2024.csv` | **+1.229%** (2015–24) · +1.489% (OOS) |
| `L` | mean loss, `E[−R_i | R_i≤0]` | **exit** | same | **0.634%** (2015–24) · 0.606% (OOS) |
| `c` | mean cost per trade as a fraction | **broker** | §3 | already inside `R_i` above; ≈ 0.024% spread+slip, financing **mis-modelled** (§3) |
| `q̄` | mean size as fraction of equity | **sizing**, bounded by (3) | `RISK_CONSTITUTION.md` Art. 1–3 | 0.75% per trade, 2.5% carry heat, breakers 3.5/5/6.5% |
| `hold` | mean days in trade (drives financing and f) | exit | trades csv col `hold_days` | **6.2 d** mean (max `HOLD_DAYS=60`; trail exits first) |

### The uncontrollable (equation 3)

| symbol | definition | measured where | current value |
|---|---|---|---|
| sequence | the order in which the `R_i` arrive | `data/proof/backtest_equity_*.json` `points` (equity path) | maxDD **−9.1%** (2015–24), −3.9% (OOS), −5.4% (fresh) |
| regime | the slow variable that changes `w`, `W`, `L` together | walk-forward, `CLAUDE.md` "Current live state" | yearly Sharpe 2021 −0.13 / 2022 +0.51 / 2023 +1.26 / 2024 −0.09 — **carry pays in rate-trending years only** |

---

## 3. The cost term, opened up

`c = spread + 2·slippage + commission + financing(hold)`

| component | modelled value | file | truth status |
|---|---|---|---|
| spread (round trip) | 1.0–1.4 pips by pair | `forex_backtester.py::SPREAD_COST` L44 | modelled, not measured (execution_tracker doesn't record it) |
| slippage per side | 0.5 pips | `SLIPPAGE_PER_SIDE` L51, overlaid by `_calibrated_slippage(pair)` when `cost_calibrator.py` has real fills | **live-calibrated when available** (Gap 1) |
| commission | 0 | OANDA practice is spread-only | correct for this broker; wrong for any other |
| financing | `SWAP_RATES_ANNUAL` L66 | **BROKEN: understates OANDA by ~9× (5.5–11.9×) on all 4 live pairs, sign-flipped on EURUSD SHORT** — `research/TICK-024_cost_measurement.md` | superseded by `sovereign/forex/swap_model.py::ratediff_financing_rate` (per-date FRED differential), applied only after the execution-freeze unlock |

This is the term Colin means by "every entry and exit is different due to slippage and broker."
It is not noise; it is a per-broker function and it has already moved the live anchor once
(0.6886 → 0.6452 when the swap leg was re-based). **Any strategy compared across brokers must be
compared with this row filled in for each.**

---

## 4. Plug the numbers in

Edge per trade, costs already inside `R`, from `backtest_trades_v015_*.csv`:

| window | f /yr | w | W | L | w·W − (1−w)·L | f × edge (at 1× notional) | measured CAGR | implied q̄ (notional/equity) |
|---|---|---|---|---|---|---|---|---|
| 2015–24 (IS+OOS) | 41.4 | 0.487 | 1.229% | 0.634% | **+0.272%** | 11.3% | **8.7%** | 0.77 |
| 2023–24 (OOS) | 57.0 | 0.527 | 1.489% | 0.606% | **+0.499%** | 28.4% | **20.9%** | 0.74 |
| 2025–26 (fresh, n=56) | 39.7 | 0.482 | 0.954% | 0.768% | **+0.062%** | 2.5% | **2.2%** | 0.89 |

Read the last two columns together: `f × edge` is what the signal and exit produce per unit of
notional; CAGR is what the account produced; the ratio is `q̄`, the average notional carried per
unit of equity, and it comes out at 0.74–0.89 in every window. **The equation reproduces the
backtest in all three windows with one consistent sizing term.** That is the check that the symbols
are right.

Everything that differs between the three rows is in `w, W, L` moving together with regime: the OOS
window had wins 50% larger than the fresh window and the same hit rate. That is the sequence term
(3), not a bug, and it is why "do not deploy v015 expecting 1.25" is on the manifest.

---

## 5. What the equation says about "maximise x"

Read (2) left to right with the current values:

- **`f` is small and set by the signal.** 41 trades/yr across four pairs. Doubling `f` by loosening
  the threshold was tried (0.35 → 0.10, un-logged, 2 AM) and inflated proximity without edge
  (`project_threshold_lowering`). `f` only goes up honestly with a *second* edge, and HYP-089/091
  (TSMOM), HYP-044 (VIX), HYP-090 (adaptive), overnight-QQQ, and the 2026-09 program all failed to
  supply one. **Closed doors are recorded in `research/HYPOTHESIS_LESSONS.md`.**
- **`w·W − (1−w)·L` is at its measured peak for this signal.** 180 exit configs swept
  (`project_exit_config_sweep`): v015 is the global optimum, 0/180 beat it in both regimes. HYP-066
  (regime-keyed exits) and HYP-067 (GA exit policy) both killed. The exit term is not where x comes
  from.
- **`c` is the one term that is currently *wrong*, not merely small.** The financing leg is 9× under
  and sign-flipped on one pair. Fixing it (TICK-024, staged) re-prices every historical trade; the
  honest `R_i` is not yet known. **This is the first thing to fix, because every other term is
  measured against it.**
- **`q̄` is the only lever with headroom** — and (3) bounds it. At Calmar ~1.07 and maxDD −9%
  unlevered, levering to 10%/yr leaves <1% headroom under a 10% prop cap
  (`project_carry_propfirm_fit`). Sharpe 1.25 at this size was +0.4% over two years.
  "Sizing is the lever, not the edge" (MAGNUM_OPUS Law) is equation (2) with the numbers in.

So the honest statement of the problem is not "find the strategy." It is:

```
maximise   f · edge · q̄
subject to survive(sequence, q̄)
where      edge is fixed by the market (carry pays for crash risk, ~Sharpe 0.6 long-run),
           f is fixed by how often the market offers it,
           c must be re-measured per broker before anything else is trusted,
           and q̄ is bounded by the drawdown you can actually sit through.
```

That is a sizing-and-survival problem with a cost-measurement prerequisite. It is solvable in the
sense Colin means: each symbol has a definition, a file, and a number, and the remaining work is
moving the one that moves.

---

## 6. The Stockfish question, answered by the equation

A Stockfish-style button needs a value function `V(state)` and a search over actions. Equation (2)
*is* the value function once you condition each term on state: `w(s), W(s), L(s), c(s)`. The repo
already has the conditioning variables (regime, VIX gate, rate differential, commitment score,
library match — all captured by `decision_logger.log`). What it does not have, and what every killed
hypothesis says it cannot yet have, is evidence that conditioning on `s` moves `w·W − (1−w)·L` out
of sample (HYP-066, HYP-090, HYP-117 fitted models all negative). Until one conditioning variable
survives a sealed test, `V(s) = V` is the best estimate, and a button that trades `V` is just v015
with automatic sizing. **The button is built. It is the thing on the freeze.**

---

## 7. Open items this page creates

| # | item | moves which term | owner / gate |
|---|---|---|---|
| 1 | Apply TICK-024 financing model, re-run `scripts/prove.py` on all three windows, re-state §2–§4 | `c`, therefore all of `R` | execution-freeze unlock in `NEXT.md` |
| 2 | Reconcile backtest `MAX_RISK_PER_TRADE_PCT=0.01` with Art. 1 0.75% — pick one, log it | `q̄` | `param_change_log` rationale |
| 3 | Record spread per fill in `execution_tracker` so spread stops being modelled | `c` | infra, freeze-safe |
| 4 | Write `f, w, W, L, c, hold` per pair into `data/proof/` as a machine twin of this page | all | `scripts/prove.py` extension |
| 5 | Re-evaluate the fresh window at ~100 trades (manifest says UNDETERMINED at 45) | `w, W, L` | time |

---

Lineage: `research/MAGNUM_OPUS.md` (laws), `research/EDGE_LEDGER.md` (claims),
`research/strategies/FX_CARRY_V1_ADV.md` (the carry construction, Sharpe ~0.6 long-run),
`data/proof/v015_manifest.json` (frozen params, windows, hashes),
`research/TICK-024_cost_measurement.md` (financing truth). Written 2026-10-10 from Colin's framing:
nobody solved Navier–Stokes before it was written down with every variable defined.
