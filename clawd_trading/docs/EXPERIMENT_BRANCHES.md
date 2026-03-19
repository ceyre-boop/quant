# Quant Trading System - Git Branch Strategy

This guide explains how to use Git branches for safe experimentation with your trading models.

## Branch Types

### 1. Main Branches

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production-ready code | Protected - requires PR |
| `develop` | Integration branch | Protected - requires PR |

### 2. Experiment Branches

Name convention: `experiment/{type}/{description}`

```
experiment/model/bias-feature-engineering-v2
experiment/strategy/new-exit-logic
experiment/risk/position-sizing-kelly
experiment/data/alternative-features
```

## Workflow

### Starting an Experiment

```bash
# 1. Start from latest develop
git checkout develop
git pull origin develop

# 2. Create experiment branch
git checkout -b experiment/model/xgboost-hyperopt

# 3. Make your changes
# ... edit code ...

# 4. Commit with descriptive message
git add .
git commit -m "experiment: tune XGBoost with Optuna

- Learning rate: 0.01 -> 0.005
- Max depth: 6 -> 8
- Added early stopping

Expected: Better Sharpe on regime 2-3"
```

### Running Experiment

1. **Push to GitHub**
   ```bash
   git push -u origin experiment/model/xgboost-hyperopt
   ```

2. **Trigger GitHub Actions**
   - Go to Actions tab
   - Run "Nightly Model Retraining" on your branch
   - Or push code to trigger CI

3. **Monitor Results**
   - Check Firebase at `/performance/weekly_metrics/`
   - Review artifacts in GitHub Actions
   - Compare metrics in `models/README.md`

### Evaluating Results

| Metric | Threshold | Action |
|--------|-----------|--------|
| Sharpe > 1.5 | Excellent | Consider merging |
| Sharpe 1.0-1.5 | Good | A/B test vs main |
| Sharpe < 1.0 | Poor | Abandon branch |
| Max DD > -15% | Too risky | Abandon branch |

### Merging or Abandoning

**If successful:**
```bash
# Create PR to develop
git checkout develop
git merge experiment/model/xgboost-hyperopt

# Delete experiment branch
git branch -d experiment/model/xgboost-hyperopt
```

**If failed:**
```bash
# Keep branch for reference (don't delete)
git checkout develop

# Tag for archive
git tag archive/xgboost-failed-$(date +%Y%m%d) experiment/model/xgboost-hyperopt
```

## Branch Naming Guide

```
experiment/{category}/{short-description}

Categories:
  model      - ML model changes
  strategy   - Trading logic changes
  risk       - Risk management changes
  data       - Data pipeline changes
  feature    - New features
  fix        - Bug fixes to test
```

## Example Experiments

### Experiment 1: Feature Engineering
```bash
git checkout -b experiment/model/momentum-features-v2
# Add new momentum indicators
# Train model
# Results: Sharpe 1.42 -> 1.58 ✓
```

### Experiment 2: Strategy Change
```bash
git checkout -b experiment/strategy/trailing-stop-exit
# Replace fixed exit with trailing stop
# Results: Win rate drops, abandon ✗
```

### Experiment 3: Risk Change
```bash
git checkout -b experiment/risk/dynamic-position-sizing
# Implement Kelly criterion sizing
# Results: Sharpe 1.3 -> 1.45 ✓
```

## CI/CD for Experiments

The `.github/workflows/nightly-retrain.yml` runs on any branch:

```yaml
on:
  push:
    branches:
      - 'experiment/**'
```

This means every push to an experiment branch triggers:
1. Model retraining
2. Performance evaluation
3. Metrics pushed to Firebase (tagged with branch name)

## Firebase Structure for Experiments

```
/experiments/{branch_name}/
  ├── metrics/         # Performance metrics
  ├── models/          # Trained model metadata
  └── status/          # Running/completed/failed
```

## Best Practices

1. **One change per branch** - Don't mix model + strategy changes
2. **Document expected outcome** in commit messages
3. **Set a time limit** - 1 week max for experiments
4. **Delete merged branches** - Keep repo clean
5. **Tag interesting failures** - For future reference

## Comparison Dashboard

Use the experiment comparison script:

```python
from meta_evaluator.experiment_tracker import compare_experiments

results = compare_experiments([
    'main',
    'experiment/model/xgboost-hyperopt',
    'experiment/strategy/new-exit-logic'
])
```
