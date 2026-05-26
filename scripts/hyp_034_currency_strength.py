#!/usr/bin/env python3
"""
HYP-034: Currency Strength Composite
Does dynamic routing (strongest vs weakest G8 currency daily)
beat the fixed 5-pair v013 universe?

H0: No improvement vs fixed pair selection
H1: Dynamic routing improves portfolio Sharpe by > 0.20

Method: compute 8-currency daily strength (avg return vs all 7 others).
LONG strongest vs SHORT weakest each day. Same macro signal stack.
Compare Sharpe dynamic vs v013 fixed 2015-2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd

print('\n══ HYP-034: Currency Strength Composite ══\n')

try:
    import yfinance as yf

    # ── Load G8 currency proxies (USD base) ──────────────────────────────────
    # We use USD-denominated rates; USD strength is inverse of others
    CURRENCIES = ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'JPY', 'CHF']
    TICKERS = {
        'EUR': 'EURUSD=X', 'GBP': 'GBPUSD=X', 'AUD': 'AUDUSD=X',
        'NZD': 'NZDUSD=X', 'CAD': 'USDCAD=X', 'JPY': 'USDJPY=X', 'CHF': 'USDCHF=X',
    }
    # CAD, JPY, CHF are quoted USD/XXX so we flip their return sign

    print('Downloading FX data 2015-2024...')
    prices = {}
    for cur, ticker in TICKERS.items():
        df = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        prices[cur] = df['Close'].rename(cur)

    price_df = pd.DataFrame(prices).dropna(how='all')
    print(f'Data: {len(price_df)} days, {price_df.shape[1]} pairs')

    # Daily returns — flip USD-quoted pairs so positive = USD weaker
    rets = price_df.pct_change()
    # CAD, JPY, CHF: quote is USD/XXX, so positive return = USD stronger = those currencies weaker
    # We want: positive return = that currency is stronger vs USD
    for cur in ['CAD', 'JPY', 'CHF']:
        if cur in rets.columns:
            rets[cur] = -rets[cur]

    # Add USD as 8th currency: USD strength = negative avg of others
    rets['USD'] = -rets[CURRENCIES].mean(axis=1)

    ALL_CURRENCIES = CURRENCIES + ['USD']

    # ── Compute daily currency strength ─────────────────────────────────────
    # Strength of currency X = avg return of X vs all other 7
    # Since we have USD-denominated returns, this simplifies:
    # strength(X) = avg of (return of X/Y for all Y != X)
    # For EUR: strength = avg(EURUSD, EURGBP, EURAUD, ...)
    # Approximation using available pairs: use return rank method

    strength = pd.DataFrame(index=rets.index, columns=ALL_CURRENCIES, dtype=float)
    for cur in ALL_CURRENCIES:
        if cur == 'USD':
            strength['USD'] = rets['USD']
        else:
            strength[cur] = rets[cur]

    # Rank 1=weakest, 8=strongest each day
    ranked = strength.rank(axis=1)

    # ── Dynamic routing strategy ─────────────────────────────────────────────
    # Each day: LONG pair = (strongest_currency)/(weakest_currency)
    # We need this to map to an actual tradeable pair we have data for
    # Pair selection: if we can form it from our available tickers, use it;
    # otherwise approximate with closest available

    AVAILABLE_PAIRS = {
        ('EUR','USD'): 'EURUSD=X', ('USD','EUR'): 'EURUSD=X',
        ('GBP','USD'): 'GBPUSD=X', ('USD','GBP'): 'GBPUSD=X',
        ('AUD','USD'): 'AUDUSD=X', ('USD','AUD'): 'AUDUSD=X',
        ('NZD','USD'): 'NZDUSD=X', ('USD','NZD'): 'NZDUSD=X',
        ('AUD','NZD'): 'AUDNZD=X', ('NZD','AUD'): 'AUDNZD=X',
        ('USD','JPY'): 'USDJPY=X', ('JPY','USD'): 'USDJPY=X',
        ('USD','CAD'): 'USDCAD=X', ('CAD','USD'): 'USDCAD=X',
        ('GBP','JPY'): 'GBPJPY=X', ('JPY','GBP'): 'GBPJPY=X',
        ('EUR','JPY'): 'EURJPY=X', ('JPY','EUR'): 'EURJPY=X',
        ('AUD','JPY'): 'AUDJPY=X', ('JPY','AUD'): 'AUDJPY=X',
        ('GBP','AUD'): 'GBPAUD=X', ('AUD','GBP'): 'GBPAUD=X',
    }

    # Download cross pairs we might need
    extra_tickers = ['AUDNZD=X','GBPJPY=X','EURJPY=X','AUDJPY=X','GBPAUD=X']
    for ticker in extra_tickers:
        try:
            df = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            # Extract currency names from ticker
            base = ticker[:3]
            quote = ticker[3:6]
            prices[f'{base}/{quote}'] = df['Close']
        except Exception:
            pass

    # ── Backtest: dynamic strategy ───────────────────────────────────────────
    # Simplified: each day, identify strongest+weakest, compute hypothetical return
    # Use ATR-based hold (5 days avg, matching v013)
    HOLD_DAYS = 5
    RISK_PCT = 0.01  # 1% risk per trade
    STOP_MULT = 2.0  # 2 ATR stop

    def atr(close_series, n=14):
        r = close_series.pct_change().abs()
        return r.rolling(n).mean()

    # Dynamic strategy returns
    dynamic_rets = []
    fixed_v013_rets = []

    V013_PAIRS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDJPY=X']
    # Simplified fixed portfolio: equal weight, same hold
    pair_data = {}
    for ticker in V013_PAIRS:
        pair_name = ticker.replace('=X','')
        df = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        pair_data[pair_name] = df['Close']

    pair_df = pd.DataFrame(pair_data).dropna()

    # For each day, determine strongest and weakest currency from ranked
    common_idx = ranked.dropna().index.intersection(pair_df.index)
    common_idx = sorted(common_idx)

    print(f'\nBacktest period: {common_idx[0].date()} — {common_idx[-1].date()}')
    print(f'Trading days: {len(common_idx)}')

    # ── Dynamic strategy simulation ──────────────────────────────────────────
    dyn_equity = [1.0]
    fix_equity = [1.0]

    step = HOLD_DAYS

    for i in range(0, len(common_idx) - HOLD_DAYS, step):
        entry_date = common_idx[i]
        exit_date  = common_idx[min(i + HOLD_DAYS, len(common_idx)-1)]

        # Dynamic: strongest vs weakest at entry
        day_strength = strength.loc[entry_date]
        if day_strength.isna().all():
            continue
        strongest = day_strength.idxmax()
        weakest   = day_strength.idxmin()

        if strongest == weakest:
            continue

        # Find a tradeable representation
        # Try to compute implied return from available pairs
        # Approach: compute cross return as strongest_ret - weakest_ret
        s_ret = strength.loc[exit_date, strongest] if exit_date in strength.index else np.nan
        w_ret = strength.loc[exit_date, weakest]   if exit_date in strength.index else np.nan

        if np.isnan(s_ret) or np.isnan(w_ret):
            continue

        # Approximate: entry goes LONG strongest vs SHORT weakest
        # Return of position = return of strongest - return of weakest (over hold period)
        # Use cumulative return over hold period
        hold_slice = strength.loc[entry_date:exit_date]
        if len(hold_slice) < 2:
            continue

        # Cumulative return for each currency over hold
        cum_strong = (1 + hold_slice[strongest].fillna(0)).prod() - 1
        cum_weak   = (1 + hold_slice[weakest].fillna(0)).prod() - 1
        # Long strongest, short weakest
        trade_ret  = cum_strong - cum_weak  # symmetric

        # Apply fixed 1% risk, simple R calculation
        eq = dyn_equity[-1]
        dyn_equity.append(eq * (1 + RISK_PCT * trade_ret / abs(trade_ret + 1e-8) * min(abs(trade_ret) * 10, 3)))

        # Fixed: equal-weight all 5 v013 pairs
        fix_trade_rets = []
        for col in pair_df.columns:
            slice_p = pair_df[col].loc[entry_date:exit_date]
            if len(slice_p) < 2:
                continue
            r = slice_p.iloc[-1] / slice_p.iloc[0] - 1
            fix_trade_rets.append(r)

        if fix_trade_rets:
            avg_fix = np.mean(fix_trade_rets)
            eq_f = fix_equity[-1]
            fix_equity.append(eq_f * (1 + RISK_PCT * avg_fix / abs(avg_fix + 1e-8) * min(abs(avg_fix)*10, 3)))

    # ── Compute Sharpe ratios ─────────────────────────────────────────────────
    def sharpe(equity_curve):
        eq = np.array(equity_curve)
        rets = np.diff(eq) / eq[:-1]
        if rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252 / HOLD_DAYS))

    dyn_sharpe = sharpe(dyn_equity)
    fix_sharpe = sharpe(fix_equity)
    delta = dyn_sharpe - fix_sharpe

    print(f'\nResults:')
    print(f'  Dynamic (strongest vs weakest):  Sharpe = {dyn_sharpe:.4f}')
    print(f'  Fixed v013 (5-pair equal weight): Sharpe = {fix_sharpe:.4f}')
    print(f'  Delta:                            {delta:+.4f}')
    print(f'  H1 threshold:                     > +0.20')

    # Win rate of dynamic trades
    dyn_trade_rets = np.diff(dyn_equity) / np.array(dyn_equity[:-1])
    wr_dyn = float(np.mean(dyn_trade_rets > 0))
    fix_trade_rets2 = np.diff(fix_equity) / np.array(fix_equity[:-1])
    wr_fix = float(np.mean(fix_trade_rets2 > 0))

    print(f'\n  Dynamic WR:  {wr_dyn*100:.1f}%')
    print(f'  Fixed WR:    {wr_fix*100:.1f}%')

    if delta > 0.20:
        verdict = 'CONFIRMED — H1 accepted: dynamic routing adds > 0.20 Sharpe'
    elif delta > 0.05:
        verdict = 'PARTIAL — meaningful improvement but below H1 threshold'
    elif delta > -0.05:
        verdict = 'INCONCLUSIVE — marginal difference, within noise band'
    else:
        verdict = 'REJECTED — fixed universe outperforms dynamic routing'

    print(f'\nVERDICT: {verdict}')

    result = {
        'task': 'HYP-034',
        'dynamic_sharpe': round(dyn_sharpe, 4),
        'fixed_sharpe': round(fix_sharpe, 4),
        'delta': round(delta, 4),
        'dynamic_wr': round(wr_dyn, 4),
        'fixed_wr': round(wr_fix, 4),
        'n_trades': len(dyn_trade_rets),
        'verdict': verdict,
    }
    Path('data/agent/hyp_034_results.json').write_text(json.dumps(result, indent=2, default=float))
    print('\nSaved to data/agent/hyp_034_results.json')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
