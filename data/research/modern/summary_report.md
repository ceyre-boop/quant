# HYP-090 MODERN — Adjudication Report (TICK-023)

**VERDICT: NOT_SIGNIFICANT** · prereg hash 6dd9cc8565cf… verified pre/post · seed 42 · env {'python': '3.14.4', 'numpy': '2.4.4', 'platform': 'macOS-26.3.1-arm64-arm-64bit-Mach-O'} · runtime 62s

Registered prior: **NOT_ROBUST**. A0 static v015 daily-M2M Sharpe on the replay span: **0.9478**.

| run | Sharpe | costed | p vs A0 | BH | > placebo p95 | DSR@5775 | per-year | all |
|---|---|---|---|---|---|---|---|---|
| A1_W90 | 0.1672 | 0.0509 | 0.9987 | ✗ | ✗ (0.9251) | 1.5561 ✓ | ✗ | — |
| A1_W180 | 0.255 | 0.17 | 0.9968 | ✗ | ✗ (0.913) | 2.9496 ✓ | ✗ | — |
| A1_W365 | 0.2792 | 0.2158 | 0.9937 | ✗ | ✗ (0.9115) | 3.3349 ✓ | ✗ | — |
| A2_W90 | 0.231 | 0.1925 | 0.9982 | ✗ | ✗ (0.9251) | 2.5687 ✓ | ✗ | — |
| A2_W180 | 0.3811 | 0.3576 | 0.9854 | ✗ | ✗ (0.913) | 4.9516 ✓ | ✗ | — |
| A2_W365 | 0.4343 | 0.4193 | 0.9774 | ✗ | ✗ (0.9115) | 5.7967 ✓ | ✗ | — |

Criteria are the prereg's locked wording (verdict_criteria). The A3 placebo envelope is the selection-noise floor: beating A0 while not beating A3 is the in-sample-inflation signature, not an edge.

Prior family kills (disclosed, not double-counted): HYP-065, HYP-066, HYP-067, the 180-config exit sweep, the regime router.

Charts: `charts/equity_arms.png`, `charts/selection_timeline_*.png`, `charts/per_year.png`. Full numbers: `results.json`.