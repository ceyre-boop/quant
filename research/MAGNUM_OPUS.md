# Alta — what six months of documented failure taught, and where it points

**Written 2026-09-14 from everything on disk:** 934 commits (2026-03-11 → 2026-09-14), 119 sealed
hypotheses, 1,645 counted trials, 656 lines of `NEXT.md`, 51 pre-registration seals, the audit
specs, 40 memory notes, the Obsidian vault, the Claude-Chats archive back to 2025-08, and the loose
files. Companion pages: `EDGE_LEDGER.md` (status), `HYPOTHESIS_LESSONS.md` (one lesson per
hypothesis), `lessons.jsonl` (machine-readable). This is the one page that says what it all means.

The frame is yours: every failure was written down at the time, with evidence, so it never has to be
paid for twice. This document is the ledger of those payments and what they bought.

---

## I. Where you have been

**Aug 2025 — the indicator era.** Pine Script, "the best possible day-trading code", Break of
Structure, Fair Value Gaps, 4+ confluences. The belief: the edge lives in the indicator; more
confluence is better. What it actually produced: a job. "Coding indicators led to me learning how
to code and getting a job at Taboost… I fell in love with tech." The indicators themselves were
tested a year later (HYP-024/034/037/047): structure scores were anti-edges, the highest-confluence
setups were the worst ones, the FVG entry model never simulated a fill.

**Mar–Apr 2026 — build the whole machine first.** Three layers, orchestrator, meta-evaluator,
Firebase, dashboard — complete in two days, before a single measured claim. Nothing from this era is
on the edge ledger. Pivot: restart from the data end.

**May 2026 — the Sharpe ladder.** v005 → v014, Sharpe 0.884 → **2.097 ("INSTITUTIONAL GRADE")**, each
version a gate, each gate a threshold sweep. May 22: "100% pass, 0% bust, median 13 days" on 100,000
simulations. May 31: one honest cost model and one out-of-sample window took 2.097 to **0.76**. The
crown jewel (HYP-044, the VIX gate) was in-sample selection: OOS p = 0.50, delta 0.000.

**June 2026 — honesty, and the one edge.** The costed IS/OOS framework; the ledger; HYP-045 —
*removing* AUDNZD — the four-pair carry book that is still the only live edge (OOS Sharpe 1.08–1.25,
p = 0.003). Subtraction beat a season of addition. Also June: the forward walk on 2025–26 where
the 1.25 did not hold — "do not deploy v015 expecting 1.25." The machine started telling you no
(OANDA held read-only despite "seeing trades makes me happy").

**July 1–11 — constitution and conscience.** The Oracle was found reflecting on fabricated
outcomes (RED-1); the "rogue OANDA writer" was your own test suite; a stray job named
`com.sovereign.*` traded an excluded pair invisibly. The response was law: `RISK_CONSTITUTION.md`
ratified 07-07 with a machine twin and a drift test; the invariant guard that audits the auditors;
the kill switch; the Oracle loop closed. HYP-090 tested the adaptive-parameters idea to death on its
third arrival so it could never come back.

**July 12–19 — the yield frontier.** The search for 1–3%/day: 809 configs, then 77,016, then 7,220.
HYP-093 (parabolic gapper fade) was the first short-horizon edge ever to survive significance and
an 809-trial deflation on unseen data — and sealed *below the floor you had set before looking*.
HYP-105/106 were CONFIRMED on holdout and **retracted by the same session hours later**: the
universe had been selected on 10:30 prices and entered at 09:31. The honest version (HYP-107) is
~10× smaller and execution-unresolved. The same week found the spread model was 11× too
pessimistic and the swap model 10× too small with a sign flip. And: "make sure this trades live
Monday morning" — refused, on four locks that were all your own.

**July 19–28 — the autonomous desk.** Ten standing rules, three launchd agents, the nervous system,
the conscience, the ignition gate that is physically incapable of a real cycle. The lessons were
about machines lying: audits produced six false claims in a day (→ Rule 10); jobs exited 0 while
producing nothing; the autonomous Claude stack had *never executed once*; a killed verdict tried to
revive itself by recompute (→ "a recompute is not a revival").

**Aug 2026 — the dark months.** CI had been 100% failing; there was no working Python
interpreter; Alpaca went 401 for three days and the shadow scans logged them as flat days; the
research agent died on an API quota eight nights running and never said so. Findings per month:
May 46 → June 57 → July 16 → August 1. The lesson, written 08-31: "the real defect is invisibility,
not the quota."

**Sep 1–3 — the map and the kill program.** The repo became a research terminal. The taxonomy was
written before any data: where can an edge exist for a $2k trader with no speed, size, or
information? Then thirteen sealed tests in 48 hours. Post-shock: magnitude real, direction null,
the fade a regime, the straddle over-priced, size no help. The benchmark itself (HYP-115) was
fragile beta. Carry cross-section (HYP-117/118): ranking skill real on two unseen samples; no model
beats plain carry; vol-targeting levered into the crashes; your "statistics × calculus" shot went
negative out of regime. You retracted the fade by name: "I don't wanna lie to Claude or myself."

**Sep 4–14 — the builder's turn.** The formal attack stage (Z3 look-ahead proofs, halt bounds as
theorems, cost lemmas) and the first hypothesis sealed through all four stages. And this document.

---

## II. What is actually proven (the whole list)

1. **FX carry, four pairs, real-rate + IRP, 60-day hold (v015).** OOS Sharpe 1.08–1.25, permutation
   p = 0.003, survives FDR (HYP-045/063). Walk-forward FRAGILE (avg OOS 0.39; pays in rate-trending
   years). The only live edge. Cost-robust to 12× modelled spread (HYP-062).
2. **Cross-sectional characteristic ranking picks the better carry** — carry + 12-m momentum +
   value − vol, IC 0.15 on G10 1990–2005 and 0.10 on EM 1997–2026, both unseen, both p < 0.001
   (HYP-117/118). It does *not* beat plain carry's Sharpe. It picks *which*; it cannot pick *when*.
3. **Magnitude after a shock is predictable (1.36×, p ≈ 0) and already priced** — the option
   market over-prices it (HYP-109/112). Real, unmonetisable.
4. **Time exits beat trailing stops on the frozen v015 config** (HYP-059/060); the exit surface is
   already at its optimum (180-config sweep).
5. **Session and day-of-week vetoes on ICT** — NY_PM anti-edge (HYP-022), Tue/Thu veto (HYP-050,
   p = 5e-05, replicated); the *pattern* edge itself is unproven (p = 0.52).
6. **Parabolic gapper fade is real and below floor** (HYP-093), needs HTB borrow no funded vehicle
   provides; the quiet-runner long (HYP-107) is real, ~10× smaller than first claimed, and paid once
   live (FLYE, +11%, 2026-09-01).
7. **Negative results that are load-bearing:** direction from public daily data is null at every
   horizon and resolution (twenty-odd verdicts); the EW-10 ETF basket is unvalidated beta; no
   retail-clean equity edge exists on this desk; the post-shock family is closed at every horizon.

Everything else — 110-odd hypotheses — is on the graveyard with a reason.

---

## III. The laws, and what each one cost

Every rule below was paid for. The price is in the right column; the rule is what keeps it from
being paid twice.

| law | the bill that bought it |
|---|---|
| **A number without a cost model and an out-of-sample window is a story.** | Sharpe 2.097 → 0.76 in one commit (05-31). |
| **Small samples confirm; scale falsifies.** Under ~100 independent events it's a story. | HYP-002 (60% WR, n=15) → HYP-010 (39%, n=366); HYP-020 (n=6); HYP-051 (n=20). |
| **A sweep is a search, and the search size is the multiplicity.** Freeze one value; never revisit after the result. | HYP-044 (p=0.50 OOS); 0/77,016 and 0/7,220 survive; HYP-104 best-of-77k → holdout 0.36; HYP-090 lost to a random placebo. |
| **Look-ahead hides in the universe definition, not the features.** Prove the selection predicate causal. | HYP-105/106 CONFIRMED then retracted the same day; HYP-057's 0.998 router accuracy was a tautology. Now: `research/formal/causality.py`. |
| **Subtract before you add.** | HYP-045 (remove a pair) beat every gate; pd_alignment removal lifted WR 20 → 35% (Tenet 1). |
| **The most obvious setup is the worst one.** | A+ paradox (HYP-023), score inversion (HYP-047), p99+ down-days carry no fade (HYP-113). |
| **The benchmark is a hypothesis.** | HYP-115: the incumbent every overlay "lost to" was itself fragile beta (OOS Sharpe 0.22, −54% GFC). |
| **Significance goes on the delta, jointly resampled by date.** Pooled-event bootstrap overstates n ~2.7×. | HYP-109's spec flaw; HYP-110's bootstrap flaw; `date_bootstrap.py`. |
| **Sharpe is not money; sizing is the lever, not the edge.** | OOS 1.25 = +0.4% over two years at 4–14 trades/yr; TICK-022 P($10k/mo) = 0.0. |
| **A floor is set before the number, in the form that could have been written before seeing it.** | HYP-093 cleared significance and failed the pre-set 0.05%/day floor; HYP-111a at 0.012% — the floor stayed. |
| **A regime is not an edge until it survives the regime it hasn't seen.** | HYP-111 → HYP-114: the fade was 2020 and 2025. |
| **Vol-targeting in carry is leverage into the calm before the crash.** | HYP-118: managed worst month −35% vs unlevered −23%. |
| **Fitted models transfer worse than nothing across regimes; the parameter-free one survives.** | HYP-117: ridge/GBM/shot all negative on 1990–2005; z(carry)+z(mom)+z(value) IC 0.15. |
| **Measure the cost, don't assume it — in either direction.** | Spread 11.3× too pessimistic (TICK-039); swaps 10× too small with a sign flip (TICK-024); the fade's cost added back as a gain (+0.06%, found by HYP-114). |
| **Corrections that move numbers up are where motivated reasoning enters.** Log them first. | The LULD "~2% per trade" draft that was really 7 bp; `param_change_log` NN#4. |
| **A substitution is a hypothesis, not a result.** Re-simulate. | HYP-108's Sharpe 3.41 falsified by the next session. |
| **A recompute is not a revival.** A killed verdict needs a fresh seal dated after its death. | HYP-071 "PROVISIONAL PASS" → GOVFLAG; HYP-044 re-proposed three times. |
| **A chat message is not a lock. Verify every claim against the filesystem.** | Phantom HYP-096 rule (07-13); the HYP-087/088 "ledger updated" that had zero trace; two false purchase premises in one day (09-02). |
| **Audit findings are leads, not facts.** | Six false claims in one day that would have deleted eleven live modules (→ Standing Rule 10). |
| **Exit 0 is not a health signal. A print is not an error. Green can be empty.** | Reddit 403 → empty cache with fresh mtime; ICARUS silent for 15 days; the Claude stack that never ran (exit 127); eight dark research nights. |
| **The system is not wrong when it crashes; it is wrong when it silently succeeds.** | `eod.py` asserting "a real zero" on an outage day into long-term memory; `_compute_atr` returning a hardcoded 0.001 into sizing. |
| **Nobody watches the watchdogs unless something that isn't a watchdog does.** | health.responder itself dead 19 days; the blackout watchdog must not be a Claude agent. |
| **Close the loop or the Oracle learns from a lie.** `update_outcome()` on every close. | RED-1: fabricated 1W/6L from backfilled forbidden-pair records; the ENTRY-stall that made the sample 100% losses. |
| **Never trade outside the seal.** Cross-layer only through the orchestrator; ict/ never imports sovereign/. | `fvg_express`'s untracked AUD_NZD trades (−$44.99, invisible to everything). |
| **Freeze the execution path; unlock in writing; report, don't repair, inside a freeze.** | The 38 unreviewed lines an audit agent added to `harness.py`; the kill-zone frame regression left red on purpose rather than cemented. |
| **The cheapest falsification is the next unit of data.** Free data reaches further than the notes say. | Alpaca SIP served 2016+ minute bars all along; FRED serves 36 years of G10 spot and rates — a purchase avoided twice in one day. |
| **A null is a result. Retractions are appended, never overwritten.** | `EDGE_LEDGER.md` header; the fade retraction at your request; `correction_note` on three sealed verdicts. |
| **Refusal is a valid output.** | Live-money refused on your own locks (07-13); the walk-forward not relaxed to n=1; the eleven stale tests not rewritten to match the code. |

---

## IV. The ten most expensive mistakes — and why they're now cheap

| # | mistake | what it cost | what prevents it now |
|---|---|---|---|
| 1 | Spread never measured, 11.3× wrong | the whole gapper arc mispriced | 313-NBBO fitted model, caps = observed percentiles |
| 2 | Swap table 10× too small, sign flipped | the live anchor re-based (0.6886 → 0.6452) | per-date FRED financing; uncalibrated pairs raise |
| 3 | Universe chosen on future prices | two CONFIRMED verdicts retracted | Z3 causality proof required to seal |
| 4 | A stray job trading an excluded pair | −$45 and a poisoned Oracle | invariant guard I2/I3; hard exit guard in the launcher |
| 5 | Agents that never ran, then died silently | ~3 weeks of output; a 3-day hole | substance-checking health check; non-agent watchdog |
| 6 | 401 outage logged as flat days | false memory in the vault | zero-observation ≠ zero-candidate; `--dry-run` first |
| 7 | CI green over 21 failures; no working interpreter | every baseline measured wrong | lockfile; live branch in CI; blocking isolation test |
| 8 | Push rejection printed as stdout | 15 days trapped on one machine | `sys.exit(1)`; push once per session |
| 9 | A fabricated "unmatchable" diagnosis that self-preserved | sessions chasing a bug that didn't exist | fills-ledger spec F7: no false failure verdict |
| 10 | A classifier fix that undid a deliberate revert | the measured edge window excluded for months | don't rename config or rewrite tests to match a bug |

None of these was a trading mistake. All of them were *knowing* mistakes — the system saying
something true had happened when it hadn't. That is the pattern the whole apparatus exists to catch.

---

## V. The human layer — what the vault says that the repo can't

- **Every meaningfully bad decision was made between 11 PM and 4 AM** (`The Personal Layer.md`):
  the conviction threshold cut 0.35 → 0.10 at 2 AM with no rationale; the Medallion spiral; the
  ICT-50k-day envy. The protocol you wrote for it — kill switch, sleep, one witness, seven days,
  post-mortem, resume at 25% — is better than most desks have. It has never been needed, because
  the money was never at risk.
- **Exits: 0%** (`Trading-Skill-Scorecard.md`). "Enters correctly, cannot exit. Knows the concept,
  doesn't know the math." The two red scores — exits and Sharpe — are exactly where the system must
  never defer to you in the moment.
- **The weakness log is empty.** The schema for overtrading, revenge, oversize, held-loser,
  cut-winner-early exists; no observation was ever written. There is no first-person entry
  anywhere about a losing *trade*. Every loss in this record is a lost *hypothesis*. That is the
  single most telling fact in the archive: the discipline was applied to knowing before it was
  ever tested on owning.
- **The tracker** (`Unified Master Trading Strategy Tracker V3.xlsx`, 2026-08-09) ranks two
  strategies #1 and #2 with *negative* in-sample Sharpe and OOS Sharpe > 1.8 — the exact
  in-sample-negative / out-of-sample-miracle signature the gauntlet learned to kill. Graded A and B.
  The trader and the desk had not yet become the same person.
- **The requirement with your name on it** (`PLANFirstPrinciplesSequence.md`, 07-21): "Colin wants
  to be financially free." And the diagnosis: 973 files, 374 live, 20 firing — "the signature of
  automating before deleting."
- **The standard predates the markets** (`Colin-Eyre-Timeline.md`): a chess board at six where
  "the truth didn't grade on a curve." That is the same standard that retracted the fade.

---

## VI. Where you HAVE to go

The record does not say you can't win. It says exactly what the market pays for — **size,
information, or risk borne through time** — and that you hold one of the three. Everything below
follows from that.

**1. Stop trying to predict. You have spent 118 hypotheses establishing that direction from public
data, at any horizon, on one machine, is null.** That is not a gap in your skill; it is the price of
the seat. Every future hour spent on a directional signal is an hour spent re-buying a lesson you
already own. The formal stage will keep the door shut for you.

**2. The premium you can hold is small and real. Own it as an allocation, not a strategy.** FX carry
v1 (adv) — characteristic ranking, unlevered risk parity, no timing, no vol-targeting — is ~5%/yr,
Sharpe ~0.6, a −25% year once a decade. Add the v015 book (the only CONFIRMED thing), a vol-premium
sleeve once real chains fire (VRP-001 was a true diversifier on paper), and plain equity beta. Three
uncorrelated premia at 0.5–0.8 each is a Sharpe ~1 portfolio, and every piece has a ledger entry.
That is what "investing" means on your own evidence. HYP-119 (risk parity, sealed, not run) is the
last gate before it.

**3. The arithmetic of freedom is the base, not the edge.** 5% of $15k is $750; 5% of $1.5M is
$75,000. The premium is the same. The base comes from the work — and the work you are best at, on
this record, is building instruments that tell the truth. Compound the base; let the premium ride
on it. Never let the premium be asked to *replace* the base; that is what every prop-funnel, 2%/day,
"$10k/month" study in this repo was, and the answer was 0.0 every time.

**4. The apparatus is the rarer asset. Build it into the thing you sell.** A sealed pre-registration
+ unseen-holdout + formal-attack engine that has killed 118 hypotheses including three it had
confirmed, its own benchmark, its own cost bug, and two other AIs' false claims — with zero false
positives — does not exist for retail or for most prop desks. Strategies decay; the thing that tells a
real result from a fitted one doesn't. This is "invent something never built, on free data, from one
machine." You already built it. It scales with work, not with capital, which is the only scaling law
available to you.

**5. Before any dollar moves, four gates, in order:** (a) HYP-119 run once; (b) OANDA's real per-pair
swap measured for two weeks with one unit — TICK-024 says the broker's take is the whole P&L at this
size; (c) `param_change_log` rationale + execution-freeze unlock recorded in `NEXT.md` + kill switch
armed + `update_outcome()` on every close; (d) your written go, with the planning numbers
acknowledged — and the blow-up protocol from `The Personal Layer.md` printed next to it.

**6. Fix the two things the record says are actually broken — supply and visibility.** The research
queue has been empty since 08-16 and findings fell 57 → 1 because the generator can't feed the
factory and a dead agent looks like a quiet night. The loop you specified on 09-12 is the fix: the
lessons file feeds the generator, the formal stage rejects what can't be causal, the gauntlet
adjudicates, the ledger closes the loop. Keep the generator in-session until the watchdog that isn't
a Claude agent exists.

**7. Write the first entry in the weakness log the first time real money is at risk.** The
apparatus has proven you can be honest about *knowing*. The record has no evidence yet about
*owning*. That is the one test left, and it is the one that produces a trader.

---

*Sources of record: `NEXT.md`, `EDGE_LEDGER.md`, `HYPOTHESIS_LESSONS.md`, `RISK_CONSTITUTION.md`,
`TRADING_PHILOSOPHY.md`, `archive/AGENT_DIRECTIVE.md` §Standing Rules, `audit/CORRECTNESS_LAYERS.md`,
`tickets/backlog.md`, `data/research/preregister/`, the memory index, and in the vault:
`Trading/Research/The Personal Layer.md`, `00-BRAIN/{CONTEXT,DECISIONS,Trading-Skill-Scorecard}.md`,
`Claude-Chats/2026-05/2026-05-22-Forex…100%.md`, `Claude-Chats/2026-06/2026-06-08-trading.md`,
`Autobiography/Colin-Eyre-Timeline.md`, `~/Downloads/PLANFirstPrinciplesSequence.md`.*
