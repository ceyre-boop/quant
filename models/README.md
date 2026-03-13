# Model Registry - Placeholder

This directory stores trained model artifacts.

## Structure

```
models/
  registry.json          # Model version registry
  bias_engine_v1.0.pkl   # XGBoost model v1.0
  bias_engine_v1.1.pkl   # Future versions...
```

## How to Update

1. Train new model:
   ```bash
   python -m layer1.train_model --output models/bias_engine_v1.1.pkl
   ```

2. Evaluate performance:
   ```bash
   python -m backtest.backtest_runner --evaluate-model models/bias_engine_v1.1.pkl
   ```

3. Update registry:
   ```bash
   python -m meta_evaluator.refit_scheduler --register-model v1.1
   ```

4. Commit:
   ```bash
   git add models/
   git commit -m "model: bias_engine_v1.1 - Sharpe: X.XX, Winrate: XX%"
   git push
   ```

## Model Performance Tracking

See `performance/weekly_metrics` in Firebase for live performance data.
