#!/usr/bin/env python3
"""
HYP-036: RMT Covariance at Portfolio Kelly Level — monthly rebalance.

HYP-035 tested RMT at per-trade correlation penalty → null result.
This tests the correct application layer: monthly Kelly fraction rebalancing
using RMT-cleaned covariance vs raw covariance.

Method:
  1. Compute 252-day rolling correlation matrix for 5 pairs monthly.
  2. Apply Marchenko-Pastur filter (q=T/N=252/5=50.4, λ_max≈1.302).
  3. Portfolio A: raw covariance → Kelly fractions.
  4. Portfolio B: RMT-cleaned covariance → Kelly fractions.
  5. Same entry signals (v013), different sizing only.
  6. Track: Sharpe, max drawdown, stress-period drawdown (COVID, 2022 rate shock).

Accept H1 if: drawdown materially lower during stress AND Sharpe maintained AND 2024 holdout replicates.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'AUDNZD=X', 'USDJPY=X']
TRAIN_START = '2018-01-01'
TRAIN_END   = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END   = '2025-06-01'
ROLLING_WINDOW = 252
REBALANCE_FREQ = 'ME'          # month-end rebalance
RISK_PCT       = 0.01          # 1% per pair at full Kelly
MAX_KELLY_FRAC = 0.03          # cap at 3% per pair

STRESS_WINDOWS = [
    ('COVID',     '2020-02-01', '2020-04-30'),
    ('Rate_Shock','2022-01-01', '2022-12-31'),
]


# ── Marchenko-Pastur filter ────────────────────────────────────────────────── #

def mp_lambda_max(q: float) -> float:
    """Upper edge of Marchenko-Pastur distribution."""
    return (1 + 1 / q ** 0.5) ** 2


def rmt_clean(corr_matrix: np.ndarray, q: float) -> np.ndarray:
    """
    Replace noise eigenvalues (below λ_max) with their mean.
    Returns cleaned correlation matrix with unit diagonal.
    """
    lam_max = mp_lambda_max(q)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Separate signal and noise eigenvalues
    signal_mask = eigenvalues > lam_max
    noise_eigenvalues = eigenvalues[~signal_mask]
    noise_mean = float(noise_eigenvalues.mean()) if noise_eigenvalues.size > 0 else 1.0

    cleaned_eigenvalues = np.where(signal_mask, eigenvalues, noise_mean)

    # Reconstruct
    cleaned = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T

    # Rescale to unit diagonal (correlation matrix invariant)
    d = np.sqrt(np.diag(cleaned))
    d[d == 0] = 1.0
    cleaned = cleaned / np.outer(d, d)
    np.fill_diagonal(cleaned, 1.0)

    n_signal = int(signal_mask.sum())
    n_noise  = int((~signal_mask).sum())
    return cleaned, n_signal, n_noise


# ── Kelly fraction from covariance ────────────────────────────────────────── #

def kelly_fractions(expected_returns: np.ndarray, cov: np.ndarray,
                    max_frac: float = MAX_KELLY_FRAC) -> np.ndarray:
    """
    Full-Kelly fractions: f = Σ^{-1} μ, clipped to max_frac.
    Uses pseudoinverse for numerical stability.
    """
    try:
        fracs = np.linalg.pinv(cov) @ expected_returns
    except np.linalg.LinAlgError:
        fracs = np.zeros(len(expected_returns))
    return np.clip(fracs, 0.0, max_frac)


# ── Load price data ────────────────────────────────────────────────────────── #

def load_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    prices = {}
    for pair in PAIRS:
        df = yf.download(pair, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        prices[pair] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    return pd.DataFrame(prices).dropna(how='all').ffill()


# ── Load v013 signals ─────────────────────────────────────────────────────── #

def load_signals(prices: pd.DataFrame) -> pd.DataFrame:
    from sovereign.forex.signal_engine import build_signal_frame
    signals = {}
    for pair in PAIRS:
        base, quote = pair.replace('=X', '').rstrip('X')[:2], pair.replace('=X', '')[-3:]
        try:
            df = build_signal_frame(pair, prices[[pair]].rename(columns={pair: 'Close'}),
                                    base, quote)
            signals[pair] = df['signal']
        except Exception as e:
            print(f'  Warning: signal for {pair} failed: {e}')
            signals[pair] = pd.Series(0, index=prices.index)
    return pd.DataFrame(signals).reindex(prices.index).fillna(0)


# ── Backtest engine ────────────────────────────────────────────────────────── #

def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame,
                 use_rmt: bool, label: str) -> pd.Series:
    """
    Monthly-rebalanced Kelly portfolio.
    Returns daily equity curve (starting at 1.0).
    """
    returns = prices.pct_change().fillna(0)
    rebalance_dates = returns.resample(REBALANCE_FREQ).last().index

    equity = pd.Series(1.0, index=returns.index)
    fracs  = np.ones(len(PAIRS)) * RISK_PCT  # equal-weight start
    q = ROLLING_WINDOW / len(PAIRS)

    for i, date in enumerate(returns.index[1:], 1):
        # Monthly rebalance
        if date in rebalance_dates and i >= ROLLING_WINDOW:
            window_ret = returns.iloc[i - ROLLING_WINDOW:i]
            cov_raw = window_ret.cov().values
            mu      = window_ret.mean().values * 252  # annualised

            if use_rmt:
                corr = window_ret.corr().values
                corr_clean, _, _ = rmt_clean(corr, q)
                std = window_ret.std().values
                cov_use = np.outer(std, std) * corr_clean
            else:
                cov_use = cov_raw

            fracs = kelly_fractions(mu, cov_use)

        # Apply signal mask: zero fraction when signal is 0
        sig = signals.loc[date].values
        active_fracs = fracs * (sig != 0).astype(float)

        # Portfolio daily return
        pair_rets = returns.loc[date].values
        port_ret  = float(np.dot(active_fracs, pair_rets))
        equity.iloc[i] = equity.iloc[i - 1] * (1 + port_ret)

    return equity


# ── Analytics ─────────────────────────────────────────────────────────────── #

def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def stress_drawdown(equity: pd.Series, start: str, end: str) -> float:
    window = equity.loc[start:end]
    if window.empty:
        return 0.0
    return max_drawdown(window)


# ── Main ──────────────────────────────────────────────────────────────────── #

def main():
    print('\n══ HYP-036: RMT Portfolio Kelly Backtest ══\n')
    print(f'Train:   {TRAIN_START} → {TRAIN_END}')
    print(f'Holdout: {HOLDOUT_START} → {HOLDOUT_END}')
    print(f'Pairs:   {", ".join(p.replace("=X","") for p in PAIRS)}')
    print(f'q = T/N = {ROLLING_WINDOW}/{len(PAIRS)} = {ROLLING_WINDOW/len(PAIRS):.1f}')
    print(f'λ_max (MP) = {mp_lambda_max(ROLLING_WINDOW/len(PAIRS)):.4f}\n')

    print('Loading prices…')
    train_prices  = load_prices(TRAIN_START, TRAIN_END)
    holdout_prices = load_prices(HOLDOUT_START, HOLDOUT_END)

    print('Loading signals…')
    train_signals  = load_signals(train_prices)
    holdout_signals = load_signals(holdout_prices)

    results = {}
    for split, prices, signals in [
        ('train',   train_prices,   train_signals),
        ('holdout', holdout_prices, holdout_signals),
    ]:
        print(f'\n── {split.upper()} ──')
        for use_rmt, label in [(False, 'Raw'), (True, 'RMT')]:
            eq = run_backtest(prices, signals, use_rmt=use_rmt, label=label)
            sh = sharpe(eq)
            dd = max_drawdown(eq)
            stress = {name: stress_drawdown(eq, s, e) for name, s, e in STRESS_WINDOWS}
            results[f'{split}_{label}'] = {
                'sharpe': sh, 'max_dd': dd, 'equity_end': float(eq.iloc[-1]),
                'stress': stress,
            }
            stress_str = '  '.join(f'{k}: {v:.1%}' for k, v in stress.items())
            print(f'  {label:4s}  Sharpe={sh:+.3f}  MaxDD={dd:.1%}  [{stress_str}]')

    print('\n══ VERDICT ══')
    train_delta  = results['train_RMT']['sharpe']  - results['train_Raw']['sharpe']
    holdout_delta = results['holdout_RMT']['sharpe'] - results['holdout_Raw']['sharpe']
    dd_improvement = {
        name: results['train_RMT']['stress'][name] - results['train_Raw']['stress'][name]
        for name, _, _ in STRESS_WINDOWS
    }

    print(f'  Sharpe delta (train):   {train_delta:+.4f}')
    print(f'  Sharpe delta (holdout): {holdout_delta:+.4f}')
    for name, delta in dd_improvement.items():
        print(f'  {name} drawdown improvement: {delta:+.1%}')

    h1_confirmed = (
        any(v < -0.15 for v in dd_improvement.values()) and
        abs(train_delta) < 0.05 and
        holdout_delta > -0.1
    )
    print(f'\n  H1 (RMT improves stress drawdown >15%, Sharpe maintained): {"CONFIRMED" if h1_confirmed else "REJECTED"}')

    # Save results
    output = {
        'hypothesis': 'HYP-036',
        'run_date': datetime.utcnow().isoformat(),
        'results': results,
        'verdict': 'H1_CONFIRMED' if h1_confirmed else 'H0_NOT_REJECTED',
        'train_sharpe_delta': train_delta,
        'holdout_sharpe_delta': holdout_delta,
        'stress_dd_improvement': dd_improvement,
    }
    out_path = Path('data/agent/hyp_036_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\n  Results saved to {out_path}')


if __name__ == '__main__':
    main()
