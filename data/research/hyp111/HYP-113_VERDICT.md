# HYP-113 — post-shock fade: dose-response in shock size within p90+

**VERDICT: FLAT** (primary, all shocks) · **FLAT** (secondary, down-shocks). "Fade harder when
bigger" is killed cleanly. Sealed `f3cf34d66a29cdbe`; hash verified before and after; ledger
`ADJUDICATED`. Same 798 events and cached bars as HYP-111a; no new data. One run.

| bin (trailing-252 rank of \|r_t\|) | all shocks: mean fade, n | down-shocks | up-shocks |
|---|---|---|---|
| B1 [0.90, 0.95) | **+0.178%** (384), CI [+0.02, +0.40] | **+0.347%** (189), CI [+0.03, +0.75] | +0.014% (195) |
| B2 [0.95, 0.99) | +0.106% (303), CI [−0.03, +0.25] | +0.231% (160), CI [+0.03, +0.45] | −0.034% (143) |
| B3 [0.99, 1.0] | +0.203% (103), CI [−0.15, +0.58] | **−0.024%** (62), CI [−0.47, +0.43] | +0.545% (41), CI [+0.06, +0.94] |
| slope on rank | −0.4%/unit, CI [−3.9, +2.6] | −3.4%/unit, CI [−9.9, +1.8] | +2.8%/unit, CI [−0.9, +6.2] |

## What it means

- **No dose-response.** Neither the pooled nor the down-only slope is distinguishable from zero,
  and neither is monotone. The intuition is dead at this window.
- **The direction of the point estimate is the opposite of the intuition, on the side that
  matters.** After the *biggest* down days (p99+, 62 events) there is no next-day fade at all —
  those are the days the drop continues. The fade after down-shocks lives in the *ordinary*
  top-decile drops (B1, B2), not the extreme ones. "Fade the crowd harder when the drop was
  bigger" is exactly the trade that gets run over.
- Up-shocks show the mirror, descriptively: nothing in B1/B2, a fade after the very largest
  up-days (B3, 41 events, CI just clears zero). Descriptive; 41 events; no weight.
- Per instrument the bins scatter with no pattern; the 10/10 in HYP-111a was the unconditional
  fade, and it stays unconditional.

## What survived at the time (retracted by HYP-114 the next day — see CORRECTION below)

The unconditional next-session fade after a p90+ shock (HYP-111a's incumbent, negated):
+0.128%/event-day, CI excludes 0, all ten instruments, 2023-06 → 2026-07 — and it is **not
improvable by shock size**. Whether it exists in 2020–2022 remains the load-bearing gap.

## Constraints honoured

One run, no bin edge moved, no alternative size measure tried. Up-shock line was declared
descriptive and stays so.


---
**CORRECTION (2026-09-03, found by HYP-114):** every fade figure above was computed as `−naive_net`, which added the 3 bp cost back as a gain — overstated by 0.06%/event-day. Corrected HYP-111 fade: +0.087%/event-day, CI [+0.004, +0.174], 8/10, E = +0.098%/trade. HYP-114 then showed the fade absent on 2016–2019 and on 20 new ETFs. See `data/research/hyp114/VERDICT.md` and `research/EDGE_LEDGER.md`.
