# Research Journal Guide

Documenting your trading research using GitHub Issues.

## Why Journal?

1. **Memory** - Markets change, remember what you learned
2. **Accountability** - Track which ideas worked
3. **Sharing** - Others can learn from your research
4. **Debugging** - Find patterns in what fails

## Issue Templates

Use the provided templates:
- **Research Entry** - Observations and hypotheses
- **Model Training** - Track training runs

## Research Entry Format

```markdown
## Observation
NQ shows 68% win rate when VIX > 25 and regime = 2 (fear)

## Hypothesis
Fear regimes create oversold bounces that revert quickly.

## Evidence
- 23 trades since Jan 2025
- Win rate: 68% vs baseline 52%
- Avg win: $340 vs avg loss: $180
- Profit factor: 2.1

## Proposed Experiment
- [ ] Add VIX > 25 as feature
- [ ] Train new model with regime interaction
- [ ] Backtest 2020-2024 (multiple fear periods)
- [ ] Paper trade for 2 weeks

## Expected Outcome
Sharpe ratio improvement from 1.3 to 1.6+

## Related Issues
- #42 - Regime analysis
- #38 - VIX feature engineering
```

## Model Training Entry Format

```markdown
## Model Details
- **Model Type**: bias_engine
- **Version**: v7
- **Training Date**: 2026-03-11
- **Branch**: experiment/model/vix-regime-feature

## Dataset
- **Period**: 2024-01-01 to 2024-12-31
- **Symbols**: NQ, ES
- **Sample Size**: 8,432

## Hyperparameters
```yaml
learning_rate: 0.005
max_depth: 8
n_estimators: 500
early_stopping_rounds: 20
```

## Performance Metrics
| Metric | Value | vs v6 |
|--------|-------|-------|
| Sharpe Ratio | 1.58 | +0.12 |
| Win Rate | 56% | +2% |
| Max Drawdown | -8.2% | -1.3% |

## Feature Importance
1. vix_regime_interaction (NEW)
2. momentum_20d
3. liquidity_score
4. regime_classification
5. adr_pct

## Decision
- [x] Promote to production
- [ ] Needs more training
- [ ] Discard
- [ ] A/B test

## Artifacts
- Model: `models/bias_engine_v7_2026-03-11.pkl`
- Notebook: `notebooks/vix_regime_analysis.ipynb`
```

## Labeling System

| Label | Use For |
|-------|---------|
| `research` | Observations, hypotheses |
| `model` | ML model experiments |
| `strategy` | Trading logic changes |
| `risk` | Risk management research |
| `data` | Data quality/availability |
| `confirmed` | Hypothesis validated |
| `rejected` | Hypothesis disproven |
| `in-progress` | Active experiment |

## Research Workflow

### 1. Observe
Notice something in the data:
```markdown
Observation: NQ underperforms on FOMC days
```

### 2. Document
Create GitHub Issue with research template

### 3. Investigate
Query your data:
```python
from data.signal_archive import SignalArchive

archive = SignalArchive()
signals = archive.get_signals_for_range('2024-01-01', '2024-12-31')

# Filter FOMC days
fomc_signals = [s for s in signals if s.get('market_context', {}).get('is_fomc_day')]

# Analyze
# ... calculate stats ...
```

### 4. Hypothesize
```markdown
Hypothesis: Volatility expansion on FOMC days
leads to whipsaws that trigger stops.

Recommendation: Skip trading on FOMC days
or widen stops by 50%.
```

### 5. Test
Run experiment:
```bash
git checkout -b experiment/strategy/fomc-filter
# Implement filter
# Train model
# Backtest
```

### 6. Record Results
Update issue with results:
```markdown
## Results
- Without FOMC filter: Sharpe 1.32
- With FOMC filter: Sharpe 1.48
- Improvement: +12%

Decision: Merge to main
```

## Research Categories

### Market Microstructure
- Order flow patterns
- Liquidity dynamics
- Spread behavior

### Regime Analysis
- Volatility regimes
- Correlation regimes
- Trend vs mean-reversion

### Feature Engineering
- Technical indicators
- Alternative data
- Cross-asset features

### Risk Management
- Position sizing
- Stop loss optimization
- Drawdown control

## Periodic Review

**Monthly:** Review all `confirmed` and `rejected` issues
- What patterns emerge?
- What should we try next?

**Quarterly:** Archive old issues
- Close outdated research
- Summarize findings in wiki

## Example Research Thread

```
Issue #45: NQ momentum fade after 10am
  ↓
Issue #62: Test time-of-day feature
  ↓
Issue #78: Model v9 with time feature
  ↓
[MERGED] Time feature improves Sharpe 1.3→1.5
```

## Quick Research Queries

```python
# Best performing hours
from meta_evaluator.signal_analyzer import SignalAnalyzer

analyzer = SignalAnalyzer()
hourly = analyzer.analyze_by_hour(days_back=60)

for hour, stats in sorted(hourly.items()):
    if stats['win_rate'] > 0.6:
        print(f"{hour:02d}:00 - {stats['win_rate']:.0%} win rate")

# Best regime
regime_stats = analyzer.analyze_by_regime()
print(json.dumps(regime_stats, indent=2))
```
