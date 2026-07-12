# Yield Frontier Program — mine everything for %/day, then prove survivors clean
**Plan: immutable-wondering-alpaca · 2026-07-12 · Tickets: TICK-030 (M-phase), TICK-031 (G-phase) · HYP-093/094/095 reserved**

## Context

Colin's ask: "look at EVERYTHING and find a method in the madness... 2% or greater daily... try it with look back... then we go back and test the right way without any look-ahead." The requested workflow — deliberate dirty mining first, clean adjudication second — is exactly the repo's existing discovery→gauntlet architecture (discovery GENERATES, the gate DECIDES). This plan runs it across every universe with data on disk.

**The arithmetic, stated up front and carried in every report:** 2%/day compounds to ~147×/year; over 10-15 years the number exceeds all money on Earth. It is not a research target — it is a falsifiable ceiling the board will be measured against, and the written expectation is that honest top candidates land in the **0.05–0.5%/day-at-capacity** range (for scale: Medallion, history's best, ran ~0.15%/day net on capped capital; the shop's proven carry edge is ~0.02%/day). The program's real deliverable is the **honest yield frontier**: every mineable method ranked by net %/day at stated capacity, with tails and frictions first-class, and the best 2-3 proven or killed on untouched data. Where the confirmed frontier lands below the aspiration, the capital-scaling route (TICK-022 funnel math) is the documented remainder — that verdict is only credible after this program has actually looked everywhere.

Universes (operator-selected): **equities gappers** (1yr cached intraday, HYP-092 infra), **NQ futures** (2.97M 1-min bars, 2018-01→2026-06 — 8.4 years, the longest intraday record on disk), **SPY options premium** (1,642 EOD chains, quote dates 2022-01→2024-06-24). Crypto: declined (no infra).

## Architecture — two phases, hard-fenced

**M (MINING, look-back allowed, disclosed):** ~800-850 configurations across 16 families, every output stamped `MINING — not evidence`, every trial counted in an append-only `mined_n.json` that later feeds the deflated-Sharpe `n_trials` (multiplicity can't be laundered). 100% local cache — M-sessions make zero network calls.

**G (GAUNTLET, the "right way"):** operator picks top 2-3 board rows → hash-locked preregs (HYP-093/094/095) + PREREGISTERED ledger entries → only then is holdout data fetched/unfenced → verdicts via the existing machinery (≥10k permutations, deflated Sharpe at honest n_trials, BH across runs, stationary block bootstrap, per-year non-degrade, plus a locked TAIL condition so "mean survived, tail exploded" fails).

### Holdouts (locked at M0, before any mining)
| Universe | Mining window | Holdout | Fence |
|---|---|---|---|
| Equities | 2025-07→2026-06 (on disk) | 2024-07→2025-06 | **physically absent from disk**; fetched only by G1 after prereg hash-lock |
| NQ | 2018-01→2024-06 | 2024-07→2026-06 | `holdout_guard.load_nq()` truncates before miners see rows |
| Options | 2022-01→2023-09 | 2023-10→2024-06 (thin — prereg carries DATA_INSUFFICIENT clause) | filename-date filter in loader |

## File layout

`research/yield_frontier/`: `_lib.py` (stamp, mined-N counter; re-exports `research.modern._lib` seed/hash helpers), `frictions.py`, `holdout_guard.py`, `yield_board.py`, `m1_equities.py`, `m2_nq.py`, `m3_options.py`, `synth_report.py`, `preregister_yield.py`, `g1_fetch_equities_holdout.py`, `gauntlet_run.py`, `tests/` (7 test files — isolation AST-whitelist, holdout fences, friction known-values, look-ahead canary, VRP-collision, determinism, mined-N monotonic). Outputs only under `data/research/yield_frontier/`. Stats imported read-only from `sovereign/discovery/gate.py` + `research/modern/_lib.py` (precedent: research/modern; enforced by AST test). Not collected by the main suite (`pytest.ini` testpaths) — the 40-known-failure baseline is structurally untouched.

## Mined families (grids bounded, cells counted)

- **M1 equities (~272 cells):** F-EQ1 overnight continuation close→next-open/close, all 11k candidates, gap×range-location×volume×direction×exit (108 — the one wholly unmined surface); F-EQ2 parabolic-fade shorts with stop grid + M&A exclusion overlay (96); F-EQ3 halt-runner re-entry longs at next printed bar (36); F-EQ4 no-news recipe longs (16); F-EQ5 catalyst-conditioned longs (16).
- **M2 NQ (~348 cells):** F-NQ1 opening-range breakout grids (162); F-NQ2 first-hour momentum/fade (36); F-NQ3 overnight-gap fade/follow (60); F-NQ4 time-of-day segments (60); F-NQ5 VIX-regime session patterns (18); F-NQ6 Globex overnight w/ Nikkei/DAX condition (12). Fill conventions copied from `sovereign/es_nq/backtest.py` (stop-first, slippage, 15:55 flat); roll days skipped; killed bias-engine NOT re-mined (context row only).
- **M3 options (~185 cells):** F-OP1 short put spreads daily (81); F-OP2 iron condors (54); F-OP3 strangles (12); F-OP4 VIX overlay on top-10 cells (30); F-OP5 long-premium lottery counterweight (8). Unit test asserts no cell equals the gated VRP-001-v2 spec; VRP-001-v2 is never run.

## Frictions (coarse, applied even in mining; pessimistic scenario is the headline)

Equities shorts: HTB fee tiers 50→500% APR by gap size, locate fill-probability haircut (50%/75%), SSR one-bar delay proxy, stops fill at worse(stop, next printed bar open) — halt gap-through is real in the cached bars. Equities longs: VWAP+25bps fill, position ≤1% of by-10:30 dollar volume (= the capacity column). NQ: 1-tick slippage/side + $0.74 MNQ / $2.50 NQ RT. Options: mid ± k×half-spread (k∈{0.25,0.5,1.0}) from the chains' own bid/ask + $0.65/leg/side.

## Yield board (M4)

Every cell with n≥40 ranked on **net %/day at stated capacity**, columns: %/event mean+median, events/day, gross & net %/day, tail_p5/p1, ruin fraction P(event≤−25%), capacity ceiling $, per-year consistency, n, stamp. Context rows from the ledger: carry (PROVEN ~0.02%/day), ICT (UNPROVEN), bias engine (KILLED), overnight-QQQ, VRP-001 (dead), HYP-092 (sealed null). Report opens with the 2%/day arithmetic statement.

## Sessions

M0 scaffold+tests (2-3h) → M1 equities (compute <5min) → M2 NQ (<10min) → M3 options (<10min) → M4 board → **operator picks candidates** → G0 preregs+ledger (no data read) → G1 equities holdout fetch (Polygon 5/min, ~55min chunked via `--max-dates`; gate-zero refuses to run unless prereg hash verifies) → G2 gauntlet + verdicts to ledger (backup first). Each session: NEXT.md entry + push. M1-M4 can run in one long session if time allows; G-phase is separate by construction (operator gate between).

## Verification

Module tests green per session (`python3 -m pytest research/yield_frontier/tests/ -q`); look-ahead canary (a deliberate peek-next-bar family must show absurd yield in each miner — proves the evaluator would expose leaks); determinism (two runs → byte-identical board CSV via `seed_from`); holdout fence tests; main suite failure-count unchanged at M0 and G2; ICT isolation law green; prereg hashes verified pre/post any ledger write.

## Tripwires

Mining-reads-holdout (fences + physical absence + manifest-postdates-lock); multiplicity laundering (mined-N append-only, wired into DSR); small-n board gaming (n≥40 to rank); VRP prereg collision (unit test); rate-limit session overrun (chunking law); ledger corruption (timestamped backups); survivorship in stage-2 (active-proxy split reporting); tail-hiding (tail/ruin columns mandatory; gauntlet tail condition locked).

## Out of scope

Live wiring, model training (Article 6), crypto, VRP-001-v2 execution (separately gated on Colin's ack), any execution-path file, re-mining settled daily-resolution families (they enter as context rows only).
