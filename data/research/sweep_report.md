# Micro-Edge Sweep Report
Generated: 2026-05-18 19:50 UTC
Sweep: 4,200 combinations in 25s

## Results
- Micro-edges found: 1800
- Portfolio edges (uncorrelated): 5
- Total risk deployed: 1.25%
- Estimated monthly win probability: 56.7%

## Top 20 Micro-Edges (by Sharpe)

| Pair | Hold | Threshold | Trailing | Stop | WR | AvgR | Sharpe |
|------|------|-----------|----------|------|----|------|--------|
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.10 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.15 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.20 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.25 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |
| GBPUSD=X | 6d | 0.25 | 2.0× | 2.0× | 57.8% | +0.324% | 1.454 |

## Portfolio Construction (Lo Framework)
- 5 uncorrelated edges
- 0.25% risk per edge
- 1.25% total risk
- 56.7% estimated monthly win probability

## Next Steps
1. Wire top edges into signal_engine as additional signal layers
2. Run signal_decay.py monthly to monitor edge degradation
3. Re-run sweep quarterly to discover new edges