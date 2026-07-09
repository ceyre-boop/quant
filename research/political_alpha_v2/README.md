# Political-Alpha V2 (HYP-087 Track A + HYP-088 Track B)

Research-only, isolated. Successor to the NOT_SIGNIFICANT HYP-085 V1 event study.
Governing spec (LAW): `~/Obsidian/Obsidian/Trading/Research/Political-Alpha-V2-Claude-Code-Spec.md`

**Thesis.** V1 averaged a heterogeneous event population and found nothing (p=0.3637). V2
**conditions on event type before testing** and corrects for multiple comparisons. The
deliverable is a `cluster → instrument → direction → timing` lookup table, not a p-value.

**Isolation (HARD).** Imports nothing from `sovereign/ ict/ ict-engine/ config/ audit/
scripts/` or the V1 `political_alpha` module. Reads the V1 event catalog as input data only.
No OANDA, no launchd, no live params. AST-enforced: `tests/test_isolation.py`.

**A row that passes the statistical gate is a *candidate* regime signal — NOT a greenlight to
trade.** Going live requires a separate discovery gauntlet (permutation / deflated Sharpe /
BH / CPCV) per spec §9.

## Pipeline
| Phase | Script | Output |
|-------|--------|--------|
| 0 | `config/*.json` (LOCKED before any data pull) | pre-registration |
| 1 | `build_clusters.py` | `data/clustered_events.jsonl` + Bonferroni denom |
| 2 | `compute_cluster_returns.py` | `data/cluster_sar_matrix.jsonl` |
| 3 | `build_congressional_signal.py` | `data/congressional_signal.jsonl` |
| 4 | `run_statistical_tests.py` | `output/{cluster_playbook.md, cluster_sar_plots.png, congressional_signal_results.json, summary_report.md}` |

## Run
```bash
python3 -m pytest research/political_alpha_v2/tests/ -q     # isolation gate
python3 research/political_alpha_v2/build_clusters.py       # Phase 1
python3 research/political_alpha_v2/compute_cluster_returns.py   # Phase 2
python3 research/political_alpha_v2/build_congressional_signal.py # Phase 3
python3 research/political_alpha_v2/run_statistical_tests.py     # Phase 4
```

## Method (locked, spec §2)
- **SAR** = (R − μ)/σ; μ,σ from estimation window T-252..T-10 (mean-adjusted).
- **CSAR[0,+72h]** = SAR(T+0)+SAR(T+1)+SAR(T+2) — primary metric; **pre** = SAR(T-2)+SAR(T-1).
- **Bonferroni denom** = (clusters with ≥1 event) × 9 (full universe, conservative); locked at Phase 1.
- **Instrument universe** = XLE, XLF, XLV, XLI, KWEB, SLX, GLD, TLT, DX-Y.NYB (all yfinance).
- **personal_attack** = negative control.
