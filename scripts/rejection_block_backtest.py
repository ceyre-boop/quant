"""
Rejection Block Backtester — NQ Futures
========================================
Method: ICT-derived intraday pattern from the rejection block video

Rules (as described):
1. LIQUIDITY SWEEP: Price wicks beyond a prior swing high/low (sweeps resting liquidity)
2. CONFIRMING CLOSE: Next candle closes in the opposite direction (bearish close for short, bullish for long)
3. PDA CONFLUENCE: Sweep occurs at or near a meaningful price level (order block, FVG, key level)
4. ENTRY: At open of candle after confirming close
5. STOP: Beyond the wick high/low of the rejection bar
6. TARGET: Fixed R:R (1.5R and 2R tested) OR nearest swing structure

Runs on 5-minute NQ bars. 30-day rolling windows tested across full history.
Compared against V3 carry-macro scenarios.

NOTE: This is research. Permutation test included. No edge assumed before data speaks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_PATH = "data/es_nq/nq_historical_5min.parquet"

# Liquidity sweep parameters
SWING_LOOKBACK = 20          # bars to define prior swing high/low
SWEEP_THRESH = 0.0003        # minimum wick beyond swing (0.03% of price)
MIN_WICK_RATIO = 0.4         # wick must be ≥40% of total candle range (fat wick requirement)

# PDA confluence: price within X% of a "level" (we use 20-bar highs/lows as proxy)
PDA_PROXIMITY = 0.002        # within 0.2% counts as confluence

# Trade parameters
RR_TARGET = 2.0              # risk:reward target
RR_ALT = 1.5                 # alternate conservative target
MAX_BARS_HOLD = 24           # max 2 hours on 5m bars before time-exit
SLIPPAGE_TICKS = 1           # 1 tick = $5/contract slippage per side
TICK_SIZE = 0.25
CONTRACT_MULTIPLIER = 20     # NQ = $20/point

# Session filter: NY session only (9:30 AM - 4:00 PM ET = 13:30-20:00 UTC)
SESSION_START = dtime(13, 30)
SESSION_END = dtime(20, 0)

# 30-day backtest window for "live" comparison
BACKTEST_DAYS = 30

# V3 comparison anchors (from session research)
V3_SCENARIOS = {
    "v015_baseline":   {"wr": 0.535, "mean_r_per_trade": 0.0021, "annual_pct": 5.0},
    "pessimistic_v3":  {"wr": 0.535, "mean_r_per_trade": 0.0036, "annual_pct": 8.0},
    "breakeven_v3":    {"wr": 0.535, "mean_r_per_trade": 0.0047, "annual_pct": 10.0},
    "conservative_v3": {"wr": 0.744, "mean_r_per_trade": 0.0061, "annual_pct": 13.0},
}


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_nq_data():
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index, utc=True)
    # Filter to regular/extended session only — drop overnight gaps
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df


def filter_session(df):
    """Keep only NY session bars."""
    times = df.index.time
    mask = (times >= SESSION_START) & (times < SESSION_END)
    return df[mask].copy()


# ─── FEATURE DETECTION ───────────────────────────────────────────────────────

def compute_swing_levels(df, lookback=SWING_LOOKBACK):
    """
    Rolling swing high and swing low over lookback bars.
    These represent 'resting liquidity' — the levels price will target.
    """
    df = df.copy()
    df['swing_high'] = df['high'].rolling(lookback).max().shift(1)
    df['swing_low'] = df['low'].rolling(lookback).min().shift(1)
    return df


def detect_rejection_blocks(df):
    """
    Scan every bar for a rejection block setup.
    Returns a list of trade signals with entry details.
    """
    df = compute_swing_levels(df)
    signals = []

    prices = df[['open', 'high', 'low', 'close', 'swing_high', 'swing_low']].values
    timestamps = df.index

    for i in range(SWING_LOOKBACK + 2, len(df) - 1):
        o, h, l, c = prices[i, 0], prices[i, 1], prices[i, 2], prices[i, 3]
        swing_h = prices[i, 4]
        swing_l = prices[i, 5]

        if pd.isna(swing_h) or pd.isna(swing_l):
            continue

        candle_range = h - l
        if candle_range < 1.0:  # skip doji/flat bars
            continue

        # ── BEARISH REJECTION BLOCK (short setup) ──
        # 1. Wick sweeps above prior swing high
        wick_above = h - max(o, c)  # upper wick
        body_size = abs(c - o)
        sweep_amount = h - swing_h

        if (sweep_amount > 0 and                              # swept the high
            sweep_amount / swing_h > SWEEP_THRESH and         # meaningful sweep
            wick_above / candle_range >= MIN_WICK_RATIO and   # fat wick
            c < o):                                           # bearish close (confirming)

            # PDA confluence: did we tap a meaningful overhead level?
            # Use swing_high itself as the PDA proxy (swept = tapped a level)
            pda_hit = sweep_amount / swing_h <= PDA_PROXIMITY * 10  # within 2% of level

            stop_price = h + TICK_SIZE  # stop above wick high
            entry_price = df['open'].iloc[i + 1]  # next bar open
            risk = stop_price - entry_price
            if risk <= 0:
                continue

            target_2r = entry_price - risk * RR_TARGET
            target_15r = entry_price - risk * RR_ALT

            signals.append({
                'timestamp': timestamps[i + 1],
                'direction': 'short',
                'entry': entry_price,
                'stop': stop_price,
                'target_2r': target_2r,
                'target_15r': target_15r,
                'risk_pts': risk,
                'wick_ratio': wick_above / candle_range,
                'sweep_pct': sweep_amount / swing_h,
                'pda_confluence': pda_hit,
                'bar_idx': i + 1,
            })

        # ── BULLISH REJECTION BLOCK (long setup) ──
        wick_below = min(o, c) - l  # lower wick
        sweep_amount_low = swing_l - l

        if (sweep_amount_low > 0 and                               # swept the low
            sweep_amount_low / swing_l > SWEEP_THRESH and          # meaningful sweep
            wick_below / candle_range >= MIN_WICK_RATIO and        # fat wick
            c > o):                                                # bullish close (confirming)

            pda_hit = sweep_amount_low / swing_l <= PDA_PROXIMITY * 10

            stop_price = l - TICK_SIZE
            entry_price = df['open'].iloc[i + 1]
            risk = entry_price - stop_price
            if risk <= 0:
                continue

            target_2r = entry_price + risk * RR_TARGET
            target_15r = entry_price + risk * RR_ALT

            signals.append({
                'timestamp': timestamps[i + 1],
                'direction': 'long',
                'entry': entry_price,
                'stop': stop_price,
                'target_2r': target_2r,
                'target_15r': target_15r,
                'risk_pts': risk,
                'wick_ratio': wick_below / candle_range,
                'sweep_pct': sweep_amount_low / swing_l,
                'pda_confluence': pda_hit,
                'bar_idx': i + 1,
            })

    return signals


# ─── TRADE SIMULATION ─────────────────────────────────────────────────────────

def simulate_trades(df, signals, rr_col='target_2r'):
    """
    Forward-simulate each signal through the subsequent bars.
    Returns trade results with R-multiples.
    """
    results = []
    prices = df[['high', 'low', 'close']].values
    timestamps = df.index

    for sig in signals:
        bar_idx = sig['bar_idx']
        if bar_idx >= len(df) - 1:
            continue

        entry = sig['entry']
        stop = sig['stop']
        target = sig[rr_col]
        direction = sig['direction']
        risk = sig['risk_pts']

        outcome = None
        bars_held = 0
        exit_price = None
        exit_reason = None

        for j in range(bar_idx, min(bar_idx + MAX_BARS_HOLD, len(df))):
            h = prices[j, 0]
            l = prices[j, 1]
            c = prices[j, 2]
            bars_held = j - bar_idx + 1

            if direction == 'short':
                # Check stop first (worst case)
                if h >= stop:
                    exit_price = stop
                    exit_reason = 'stop'
                    break
                # Check target
                if l <= target:
                    exit_price = target
                    exit_reason = 'target'
                    break
            else:  # long
                if l <= stop:
                    exit_price = stop
                    exit_reason = 'stop'
                    break
                if h >= target:
                    exit_price = target
                    exit_reason = 'target'
                    break

        # Time exit
        if exit_price is None:
            exit_price = prices[min(bar_idx + MAX_BARS_HOLD - 1, len(df) - 1), 2]
            exit_reason = 'time'

        # R-multiple (with slippage)
        slippage = SLIPPAGE_TICKS * TICK_SIZE * 2  # entry + exit
        if direction == 'short':
            gross_pts = entry - exit_price
        else:
            gross_pts = exit_price - entry

        net_pts = gross_pts - slippage
        r_multiple = net_pts / risk if risk > 0 else 0

        results.append({
            **sig,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'gross_pts': gross_pts,
            'net_pts': net_pts,
            'r_multiple': r_multiple,
            'win': r_multiple > 0,
            'date': sig['timestamp'].date(),
        })

    return pd.DataFrame(results)


# ─── ROLLING 30-DAY WINDOWS ──────────────────────────────────────────────────

def run_rolling_windows(df, session_df, window_days=30, step_days=30):
    """
    Run non-overlapping 30-day windows across the full history.
    Returns per-window stats.
    """
    all_signals = detect_rejection_blocks(session_df)
    if not all_signals:
        print("No signals detected.")
        return pd.DataFrame()

    sigs_df = pd.DataFrame(all_signals)
    sigs_df['date'] = pd.to_datetime(sigs_df['timestamp']).dt.date

    start_date = session_df.index[0].date()
    end_date = session_df.index[-1].date()

    window_results = []
    current = start_date

    while current + timedelta(days=window_days) <= end_date:
        window_end = current + timedelta(days=window_days)

        # Signals in this window
        mask = (sigs_df['date'] >= current) & (sigs_df['date'] < window_end)
        window_sigs = sigs_df[mask].to_dict('records')

        if len(window_sigs) < 5:
            current += timedelta(days=step_days)
            continue

        # Simulate
        trades = simulate_trades(session_df, window_sigs, rr_col='target_2r')

        if len(trades) == 0:
            current += timedelta(days=step_days)
            continue

        n = len(trades)
        wr = trades['win'].mean()
        mean_r = trades['r_multiple'].mean()
        total_r = trades['r_multiple'].sum()
        stop_rate = (trades['exit_reason'] == 'stop').mean()
        target_rate = (trades['exit_reason'] == 'target').mean()

        window_results.append({
            'window_start': current,
            'window_end': window_end,
            'n_trades': n,
            'win_rate': wr,
            'mean_r': mean_r,
            'total_r': total_r,
            'stop_rate': stop_rate,
            'target_rate': target_rate,
            'pda_pct': trades['pda_confluence'].mean() if 'pda_confluence' in trades.columns else 0,
        })

        current += timedelta(days=step_days)

    return pd.DataFrame(window_results)


# ─── PERMUTATION TEST ─────────────────────────────────────────────────────────

def permutation_test(trades_df, n_perms=1000, seed=42):
    """
    Test whether mean_r is better than random.
    H0: mean_r comes from a distribution with mean=0.
    Shuffle entry direction labels and recompute mean_r.
    """
    np.random.seed(seed)
    observed = trades_df['r_multiple'].mean()
    n = len(trades_df)

    perm_means = []
    r_vals = trades_df['r_multiple'].values
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=n)
        perm_means.append((np.abs(r_vals) * signs).mean())

    perm_means = np.array(perm_means)
    p_value = (perm_means >= observed).mean()
    return observed, p_value, perm_means


# ─── V3 COMPARISON ───────────────────────────────────────────────────────────

def compare_to_v3(rb_stats, all_trades):
    """
    Print honest side-by-side comparison.
    V3 is daily forex carry, RB is intraday NQ — apples vs oranges,
    but both are R-based systems so expectancy is comparable.
    """
    print("\n" + "="*70)
    print("REJECTION BLOCK vs V3 CARRY METHOD — HONEST COMPARISON")
    print("="*70)
    print("\nNOTE: Different instruments, time horizons, and edge types.")
    print("      Comparison is on expectancy (mean R/trade), not annualized %.")
    print("      V3 annual % assumes ~46 trades/year across 4 pairs, 60d holds.")
    print("      RB annual % assumes same trade frequency (NOT realistic — RB")
    print("      could trade 5-20x/day; scaling to annual requires position sizing.)")
    print()

    # RB stats
    if len(all_trades) > 0:
        rb_wr = all_trades['win'].mean()
        rb_mean_r = all_trades['r_multiple'].mean()
        rb_n = len(all_trades)
        rb_obs, rb_p, _ = permutation_test(all_trades)
    else:
        rb_wr = rb_mean_r = rb_n = rb_obs = rb_p = 0

    print(f"{'Metric':<35} {'RB (NQ intraday)':<22} {'V3 scenarios'}")
    print("-"*70)
    print(f"{'Sample size (trades)':<35} {rb_n:<22} {'~46-92/yr (4 pairs)'}")
    print(f"{'Win rate':<35} {rb_wr:.1%}{'':14} {'53.5–74.4%'}")
    print(f"{'Mean R/trade':<35} {rb_mean_r:+.4f}{'':14} {'+0.0021 to +0.0061'}")
    print(f"{'Permutation p-value':<35} {rb_p:.3f}{'':14} {'<0.001 (confirmed)'}")
    print(f"{'Edge status':<35} {'CONFIRMED' if rb_p < 0.05 else 'NOT CONFIRMED':<22} {'CONFIRMED (v015 OOS Sharpe 1.25)'}")
    print()

    print("V3 scenario breakdown:")
    for name, v3 in V3_SCENARIOS.items():
        label = name.replace('_', ' ').upper()
        print(f"  {label:<28} WR={v3['wr']:.1%}  mean_r={v3['mean_r_per_trade']:+.4f}  ~{v3['annual_pct']:.0f}%/yr")

    print()
    if rb_p < 0.05:
        print(f"✅ RB: Mean R {rb_mean_r:+.4f}, p={rb_p:.3f} — STATISTICALLY SIGNIFICANT")
    elif rb_p < 0.10:
        print(f"⚠️  RB: Mean R {rb_mean_r:+.4f}, p={rb_p:.3f} — MARGINAL (not significant at 0.05)")
    else:
        print(f"❌ RB: Mean R {rb_mean_r:+.4f}, p={rb_p:.3f} — NOT SIGNIFICANT (same as random)")

    print()
    print("WHAT THIS MEANS:")
    if rb_p < 0.05 and rb_mean_r > 0:
        print("  RB shows real edge. But it's intraday — needs prop firm sizing context,")
        print("  not the 0.25% effective risk of ALTA_METHOD. Different game entirely.")
        print("  To deploy: needs walk-forward, spread costs, session time gating.")
    else:
        print("  RB does not clear the statistical bar on this data.")
        print("  The video shows payout — that's not the same as a confirmed edge.")
        print("  Could be: small sample, selected examples, specific market regime.")
        print("  This does NOT mean it never works. It means we can't confirm it yet.")


# ─── DAILY EXECUTION CHECKLIST ────────────────────────────────────────────────

DAILY_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════════╗
║         REJECTION BLOCK — DAILY EXECUTION CHECKLIST                 ║
║         Based on: ICT-style intraday method (video reference)        ║
╚══════════════════════════════════════════════════════════════════════╝

PRE-MARKET (before 9:30 AM ET)
──────────────────────────────
□ 1. Mark prior day high/low on chart (these are your key liquidity levels)
□ 2. Mark Asian session high/low (globex range — resting liquidity)
□ 3. Mark any unfilled fair value gaps (FVGs) from prior session
□ 4. Note key HTF levels: weekly open, monthly open, HTF order blocks
□ 5. Check economic calendar — no entries within 15 min of high-impact news

SESSION OPEN (9:30 AM ET)
──────────────────────────
□ 6. Let first 15 minutes settle — no entries during opening drive chaos
□ 7. Watch for first major liquidity sweep (price wicks through prior high/low)
□ 8. Is sweep happening at a PDA (order block, FVG, key level)? YES = valid
□ 9. Did confirming candle close in opposite direction? YES = rejection block formed

ENTRY CRITERIA (all must be YES)
──────────────────────────────────
□ 10. Liquidity swept (wick beyond prior swing high/low): YES/NO
□ 11. Fat wick (≥40% of candle range): YES/NO
□ 12. Confirming close opposite direction: YES/NO
□ 13. PDA confluence at the level: YES/NO
□ 14. Not within 15 min of news event: YES/NO
□ 15. Within NY session (9:30 AM - 4:00 PM ET): YES/NO

→ If all YES: enter at open of next bar
→ Stop: 1 tick beyond rejection wick high/low
→ Target: nearest internal structure OR 2R

TRADE MANAGEMENT
─────────────────
□ 16. Stop placed as hard order (not mental) before entry fills
□ 17. Target identified before entry (nearest swing structure or 2R)
□ 18. Do NOT move stop after entry
□ 19. Do NOT add to position mid-trade
□ 20. Max 2-hour hold — exit at market if neither stop nor target hit

POST-SESSION
─────────────
□ 21. Log every signal (taken or passed): setup quality, gate results
□ 22. Log every trade: entry, stop, target, actual exit, R-multiple
□ 23. Note any gate bypasses (entries taken without all 6 criteria)
□ 24. Review: did RB form at the level you expected? What was the outcome?

WEEKLY REVIEW (feeds ML cycle)
────────────────────────────────
□ 25. Compliance rate: what % of entries passed all 6 gates?
□ 26. Performance split: compliant vs non-compliant entries
□ 27. Which PDA types produced the best RBs? (OB > FVG > fib > other)
□ 28. Which session times had the highest hit rate? (9:30-11 AM vs 1-3 PM)
□ 29. One proposed adjustment → preregister before changing any parameter

NOTE: This is ICT intraday (1m/5m), not ALTA_METHOD (60d macro carry).
      Different sizing, different risk per trade, different edge horizon.
      Cannot be mixed into the same position sizing framework.
"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading NQ 5-minute data...")
    df = load_nq_data()
    session_df = filter_session(df)
    print(f"Loaded: {len(df):,} raw bars, {len(session_df):,} session bars")
    print(f"Range: {df.index[0].date()} to {df.index[-1].date()}")

    # Focus on last 2 years for relevance
    cutoff = pd.Timestamp('2024-01-01', tz='UTC')
    recent_session = session_df[session_df.index >= cutoff].copy()
    print(f"Recent 2yr window: {len(recent_session):,} session bars\n")

    # Detect all signals
    print("Detecting rejection block signals...")
    all_signals = detect_rejection_blocks(recent_session)
    print(f"Signals found: {len(all_signals)}")

    if len(all_signals) == 0:
        print("No signals detected. Check parameters.")
        return

    # Signal breakdown
    sigs_df = pd.DataFrame(all_signals)
    print(f"  Long signals: {(sigs_df['direction']=='long').sum()}")
    print(f"  Short signals: {(sigs_df['direction']=='short').sum()}")
    print(f"  With PDA confluence: {sigs_df['pda_confluence'].sum()} ({sigs_df['pda_confluence'].mean():.1%})")
    print(f"  Mean wick ratio: {sigs_df['wick_ratio'].mean():.2f}")
    print(f"  Mean sweep %: {sigs_df['sweep_pct'].mean():.4f}")

    # Simulate all trades at 2R target
    print("\nSimulating trades (2R target)...")
    all_trades_2r = simulate_trades(recent_session, all_signals, rr_col='target_2r')
    print(f"Trades simulated: {len(all_trades_2r)}")

    # Simulate at 1.5R
    print("Simulating trades (1.5R target)...")
    all_trades_15r = simulate_trades(recent_session, all_signals, rr_col='target_15r')

    # ── FULL RESULTS ──
    print("\n" + "="*70)
    print("FULL BACKTEST RESULTS (2024-01-01 to 2026-06-09)")
    print("="*70)

    for label, trades in [("2R Target", all_trades_2r), ("1.5R Target", all_trades_15r)]:
        if len(trades) == 0:
            continue
        n = len(trades)
        wr = trades['win'].mean()
        mean_r = trades['r_multiple'].mean()
        total_r = trades['r_multiple'].sum()
        stop_r = (trades['exit_reason']=='stop').mean()
        tgt_r = (trades['exit_reason']=='target').mean()
        time_r = (trades['exit_reason']=='time').mean()

        obs, p, _ = permutation_test(trades)

        print(f"\n  [{label}]")
        print(f"  Trades: {n}")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Mean R: {mean_r:+.4f}")
        print(f"  Total R: {total_r:+.2f}")
        print(f"  Exit: stop={stop_r:.1%}  target={tgt_r:.1%}  time={time_r:.1%}")
        print(f"  Permutation p={p:.3f}  ({'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'})")
        print(f"  Avg bars held: {trades['bars_held'].mean():.1f}")

    # ── PDA FILTER EFFECT ──
    print("\n" + "="*50)
    print("PDA CONFLUENCE FILTER EFFECT (2R target)")
    print("="*50)
    for pda_flag, label in [(True, "With PDA"), (False, "Without PDA")]:
        subset = all_trades_2r[all_trades_2r['pda_confluence'] == pda_flag]
        if len(subset) > 10:
            print(f"  {label}: n={len(subset)}, WR={subset['win'].mean():.1%}, mean_r={subset['r_multiple'].mean():+.4f}")

    # ── DIRECTION SPLIT ──
    print("\n" + "="*50)
    print("DIRECTION SPLIT (2R target)")
    print("="*50)
    for direction in ['long', 'short']:
        subset = all_trades_2r[all_trades_2r['direction'] == direction]
        if len(subset) > 10:
            print(f"  {direction.upper()}: n={len(subset)}, WR={subset['win'].mean():.1%}, mean_r={subset['r_multiple'].mean():+.4f}")

    # ── RECENT 30-DAY WINDOW ──
    print("\n" + "="*50)
    print("MOST RECENT 30-DAY WINDOW")
    print("="*50)
    cutoff_30d = pd.Timestamp('2026-05-01', tz='UTC')
    recent_30d_sigs = [s for s in all_signals if s['timestamp'] >= cutoff_30d]
    if len(recent_30d_sigs) >= 5:
        trades_30d = simulate_trades(recent_session, recent_30d_sigs, rr_col='target_2r')
        if len(trades_30d) > 0:
            print(f"  Period: 2026-05-01 to 2026-06-09")
            print(f"  Trades: {len(trades_30d)}")
            print(f"  Win rate: {trades_30d['win'].mean():.1%}")
            print(f"  Mean R: {trades_30d['r_multiple'].mean():+.4f}")
            print(f"  Total R: {trades_30d['r_multiple'].sum():+.2f}")
    else:
        print(f"  Only {len(recent_30d_sigs)} signals — insufficient for 30-day stats")

    # ── ROLLING WINDOWS ──
    print("\n" + "="*50)
    print("ROLLING 30-DAY WINDOWS (yearly sample)")
    print("="*50)
    # Sample every 6 months to keep it fast
    yearly_sigs = [s for s in all_signals if s['timestamp'].year >= 2024]
    windows = run_rolling_windows(df, recent_session, window_days=30, step_days=30)
    if len(windows) > 0:
        print(f"  Windows tested: {len(windows)}")
        print(f"  Windows profitable (mean_r > 0): {(windows['mean_r'] > 0).sum()} / {len(windows)}")
        print(f"  Mean win rate across windows: {windows['win_rate'].mean():.1%}")
        print(f"  Mean R across windows: {windows['mean_r'].mean():+.4f}")
        print(f"  Best window: {windows['mean_r'].max():+.4f}R")
        print(f"  Worst window: {windows['mean_r'].min():+.4f}R")
        print(f"  Std dev of mean_r: {windows['mean_r'].std():.4f}")

    # ── V3 COMPARISON ──
    compare_to_v3(None, all_trades_2r)

    # ── DAILY CHECKLIST ──
    print(DAILY_CHECKLIST)

    # ── SAVE RESULTS ──
    out_path = "data/research/rejection_block_backtest_results.csv"
    all_trades_2r.to_csv(out_path, index=False)
    print(f"\nFull trade log saved: {out_path}")

    if len(windows) > 0:
        windows.to_csv("data/research/rejection_block_rolling_windows.csv", index=False)
        print("Rolling windows saved: data/research/rejection_block_rolling_windows.csv")


if __name__ == "__main__":
    main()
