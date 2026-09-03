# HYP-114 — deploying the post-shock fade: denominator, unseen years, wider universe, exit

**VERDICT: FAIL** (claim 1). Claim 2 FAIL. Claim 3 NO_DIFFERENCE.
Sealed `0efc2fdc9ed555f4` before any 2016–2019 or wider-universe session was read; hash verified before
and after; ledger `ADJUDICATED`. 8,013 sessions, 99.3% complete. One run.

| claim | result | bar |
|---|---|---|
| **1 — unseen 2016–2019, core ten** | **−0.027%/deployed-day**, CI [−0.108, +0.053], **4/10** instruments | CI > 0 and ≥ 7/10 → **FAIL** |
| 1 — full 2016–2026 | +0.048%, CI [−0.015, +0.112] — spans 0 | |
| 1 — account (10%/instrument, compounding) | CAGR **+1.3%**, maxDD −6.9%, calendar Sharpe 0.33, deployed 10% of the time | |
| **2 — 20 new ETFs, 2017–2026** | **+0.001%**, CI [−0.057, +0.061], **11/20**, ex-2020 −0.017% | **FAIL** |
| 2 — 30-ETF account vs 10-ETF | CAGR +3.0% / maxDD −19.4% vs +1.6% / −6.9% | more deployment, more drawdown, no more edge |
| **3 — 12:00 vs 15:55 exit** | delta −0.035%, CI [−0.086, +0.009] | NO_DIFFERENCE |

Per year, core ten: 2016 −.08 · 2017 +.16 · 2018 −.08 · 2019 −.03 · 2020 +.39 · 2021 +.01 · 2022 −.12 ·
2023 +.05 · 2024 −.02 · 2025 +.34 · 2026 −.05 (event-weighted, %/deployed-day).

## What it means

**The fade does not generalise.** It is absent in the four years no test had seen, absent in twenty
instruments no test had touched, and over the full decade its yield on deployed capital is not
distinguishable from zero. What HYP-111 found on 2020–2026 was two years — 2020 and 2025 — carrying
the rest. That is a regime, not an edge. The operator's prior (PASS) and mine (VALID_BUT_BELOW_FLOOR)
were both wrong; the data was harsher than either.

## Correction disclosed here, found by this run

HYP-111, HYP-111a and HYP-113 computed the fade as `−naive_net` where `naive_net = naive − cost`.
Negating it **added the 3 bp cost back as a gain**. Every fade number in those three verdicts is
overstated by 0.06%/event-day. This script computes the fade correctly (`−naive − cost`), which is why
its 2020–2026 core figure is +0.089% where HYP-111 printed +0.147%. Corrected HYP-111 secondary:
**+0.087%/event-day, CI [+0.004, +0.174], 8/10 instruments, ex-2020 +0.051%, 2020 +0.307% (CI
[−0.008, +0.643])** — its sealed components still pass, barely, and 3 of 7 years are negative.
Corrected expectancy: W 0.511, E = +0.098%/trade. The sealed verdicts are not rewritten; a
`correction_note` is attached to each ledger entry and each VERDICT.md. The forward log
(`fade_forward_log.py`) always subtracted the cost correctly.

## Status of the post-shock program after nine preregs

Closed at every resolution and every expression: direction (109), overnight (110), path (111/111a),
magnitude instrument (112), size (113), **fade out-of-regime and out-of-universe (114)**. The only
thing that was ever real is HYP-109(a): after a shock, next-week realized vol is 1.36× — and HYP-112
showed the option market prices that and more. There is nothing in the post-shock window a $2k
retail trader can be paid for. That is the map, and it is a paid-for one.
