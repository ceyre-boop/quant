# Sovereign Core Gauntlet — preserved harness (archived)

These scripts ran the validation gauntlet that produced verdict **NO_EDGE** for the
Sovereign Core MoE (Clawd Trading). See `../sovereign_core_verdict.md` and ledger entry
**HYP-057**.

## Important: these target a DIFFERENT branch
They import `master-ml-archive` (`master`) code — `sovereign.orchestrator`, `train_core`,
`sovereign.router.regime_router`, etc. — which **does not exist on `sovereign-v2`**. They will
NOT run from this branch. To re-run, check the master code into a worktree:

```bash
git worktree add ../quant-sovereign-core master
cp .env ../quant-sovereign-core/.env          # Alpaca keys not needed (yfinance), but load_dotenv expects it
cd ../quant-sovereign-core
cp <this-dir>/*.py scripts/
/Users/taboost/quant/.venv/bin/python scripts/freeze_sovereign_dataset.py
/Users/taboost/quant/.venv/bin/python scripts/permutation_test_sovereign.py --mode signal --perms 1000
```

## Environment
- Python **3.9** via `~/quant/.venv` (system Python 3.14 has no numba wheel).
- Deps already in that venv: numba 0.60, xgboost 2.1.4, sklearn 1.6.1, yfinance 1.2.0.

## Reproducibility
- Data: clean yfinance daily bars, frozen to `data/cache/equity/` (dataset hash `ce8d80d3…`).
  yfinance is used instead of Alpaca free IEX because IEX is gappy/truncated (see HYP-057 note
  and the overnight-QQQ caveat).
- `permutation_test_sovereign.py --mode signal` → REAL Sharpe −0.183, p=0.164 (1000 perms, seed 7).
- `--mode live` reproduces the zero-trades-cold finding (EV gate blocks trade #1).
- `_diag_gates.py` produces the per-gate rejection tally (evidence for the router tautology and
  cold-start findings).

## Scripts
- `freeze_sovereign_dataset.py` — Stage 0: freeze clean yfinance OHLCV + manifest.
- `permutation_test_sovereign.py` — Stage 1: costed permutation kill-gate (`--mode signal|live`).
- `_diag_gates.py` — gate-by-gate rejection diagnostic.

**Do not revive the Sovereign Core on this harness.** Per the verdict, the only path back to the
MoE shell is a rebuild behind a *proven* edge (relabel router to forward returns, fix the
cold-start EV throttle, use a real data feed) — not a revival of the archived system.
