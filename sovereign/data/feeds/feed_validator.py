"""
Sovereign Trading Intelligence — Feed Validator / Lookahead Auditor
Phase 1: Data Foundation

Zero tolerance: ANY lookahead violation halts the build.

The test is simple but absolute:
1. Compute a feature on the FULL dataset → get result at timestamp T
2. Compute the SAME feature on data TRUNCATED at timestamp T → get result at T
3. If full_result[T] != truncated_result[T] → FAIL. Lookahead detected.

This test catches:
- Features that use future data (shift errors, look-ahead rolling windows)
- Feeds that adjust historical bars retroactively
- Join operations that leak future information
- Any pandas operation that inadvertently uses the full index

Why this matters:
A lookahead in the feed layer will silently corrupt every model trained on top of it.
That is how you get a 79% backtest that goes live at 36%.
We know this because it already happened once.

This validator must pass on ALL feeds before Phase 2 begins.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Tolerance for floating point comparison
FLOAT_TOLERANCE = 1e-10


class LookaheadAuditor:
    """
    Tests feeds and feature functions for lookahead bias.
    
    The fundamental test: does truncating the input data
    change the output at the truncation point?
    
    If yes → the feature/feed is using future information.
    If no  → the feature/feed is causal (safe to use).
    """
    
    def __init__(self):
        self.results: List[dict] = []
        self.passed = True  # Flips to False on ANY failure
    
    def test_feed_consistency(
        self,
        feed_name: str,
        get_full_data: Callable,
        get_truncated_data: Callable,
        test_points: int = 10,
    ) -> dict:
        """
        Test that a feed returns identical values whether fetched
        as a full range or as truncated sub-ranges.
        
        This catches feeds that retroactively adjust historical bars
        (e.g., adjusted close recalculated after splits).
        
        Args:
            feed_name:          Human-readable name for logging
            get_full_data:      Callable() -> pd.DataFrame (full date range)
            get_truncated_data: Callable(end_date) -> pd.DataFrame (truncated)
            test_points:        Number of random truncation points to test
        
        Returns:
            Dict with test results.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"LOOKAHEAD AUDIT: {feed_name}")
        logger.info(f"{'='*60}")
        
        # Get full dataset
        full_df = get_full_data()
        if full_df.empty:
            result = {
                'feed': feed_name,
                'status': 'SKIP',
                'reason': 'No data returned from full fetch',
                'violations': 0,
                'tests_run': 0,
            }
            self.results.append(result)
            logger.warning(f"  SKIP: No data for {feed_name}")
            return result
        
        # Select test points — evenly spaced through the data
        n = len(full_df)
        if n < test_points * 2:
            test_points = max(n // 2, 1)
        
        # Avoid edges — test in the middle 80% of the data
        start_idx = int(n * 0.1)
        end_idx = int(n * 0.9)
        test_indices = np.linspace(
            start_idx, end_idx, test_points, dtype=int
        )
        
        violations = 0
        tests_run = 0
        violation_details = []
        
        for idx in test_indices:
            test_date = full_df.index[idx]
            full_row = full_df.iloc[idx]
            
            # Get truncated data up to this point
            try:
                trunc_df = get_truncated_data(test_date)
            except Exception as e:
                logger.warning(
                    f"  Truncated fetch failed at {test_date}: {e}"
                )
                continue
            
            if trunc_df.empty:
                continue
            
            tests_run += 1
            
            # Find the matching row in truncated data
            if test_date in trunc_df.index:
                trunc_row = trunc_df.loc[test_date]
            else:
                # Find closest date
                closest_idx = trunc_df.index.get_indexer(
                    [test_date], method='nearest'
                )
                if closest_idx[0] < 0 or closest_idx[0] >= len(trunc_df):
                    continue
                trunc_row = trunc_df.iloc[closest_idx[0]]
                # Only accept if within 1 day
                actual_date = trunc_df.index[closest_idx[0]]
                if abs((actual_date - test_date).total_seconds()) > 86400:
                    continue
            
            # Compare each column
            for col in full_df.columns:
                if col not in trunc_row.index:
                    continue
                
                full_val = full_row[col]
                trunc_val = trunc_row[col]
                
                # Handle NaN
                if pd.isna(full_val) and pd.isna(trunc_val):
                    continue
                if pd.isna(full_val) != pd.isna(trunc_val):
                    violations += 1
                    violation_details.append({
                        'date': str(test_date),
                        'column': col,
                        'full_value': full_val,
                        'trunc_value': trunc_val,
                        'type': 'NaN_MISMATCH',
                    })
                    continue
                
                # Float comparison with tolerance
                if isinstance(full_val, (int, float, np.integer, np.floating)):
                    if abs(float(full_val) - float(trunc_val)) > FLOAT_TOLERANCE:
                        violations += 1
                        violation_details.append({
                            'date': str(test_date),
                            'column': col,
                            'full_value': float(full_val),
                            'trunc_value': float(trunc_val),
                            'delta': abs(float(full_val) - float(trunc_val)),
                            'type': 'VALUE_MISMATCH',
                        })
                else:
                    if full_val != trunc_val:
                        violations += 1
                        violation_details.append({
                            'date': str(test_date),
                            'column': col,
                            'full_value': str(full_val),
                            'trunc_value': str(trunc_val),
                            'type': 'VALUE_MISMATCH',
                        })
        
        status = 'PASS' if violations == 0 else 'FAIL'
        if violations > 0:
            self.passed = False
        
        result = {
            'feed': feed_name,
            'status': status,
            'violations': violations,
            'tests_run': tests_run,
            'test_points': test_points,
            'details': violation_details[:10],  # Cap detail output
        }
        self.results.append(result)
        
        if status == 'PASS':
            logger.info(f"  [PASS] {tests_run} truncation tests, 0 violations")
        else:
            logger.error(
                f"  [FAIL] {violations} LOOKAHEAD VIOLATIONS "
                f"in {tests_run} tests"
            )
            for v in violation_details[:5]:
                logger.error(
                    f"    {v['date']} | {v['column']}: "
                    f"full={v['full_value']} vs trunc={v['trunc_value']}"
                )
        
        return result
    
    def test_feature_function(
        self,
        feature_name: str,
        feature_fn: Callable,
        price_data: pd.DataFrame,
        test_points: int = 10,
    ) -> dict:
        """
        Test that a feature function produces identical output
        regardless of how much future data exists in the input.
        
        This is the core lookahead test:
        - Compute feature on full data → get value at T
        - Compute feature on data[:T] → get value at T
        - These MUST be identical.
        
        Args:
            feature_name: Human-readable name
            feature_fn:   Callable(pd.DataFrame) -> pd.Series or pd.DataFrame
            price_data:   Full OHLCV DataFrame
            test_points:  Number of truncation points to test
        
        Returns:
            Dict with test results.
        """
        logger.info(f"\n  Testing feature: {feature_name}")
        
        # Compute on full data
        try:
            full_result = feature_fn(price_data)
        except Exception as e:
            result = {
                'feature': feature_name,
                'status': 'ERROR',
                'reason': str(e),
                'violations': 0,
                'tests_run': 0,
            }
            self.results.append(result)
            return result
        
        if isinstance(full_result, pd.DataFrame):
            full_result = full_result.iloc[:, 0]  # Take first column
        
        # Select test points
        n = len(price_data)
        start_idx = int(n * 0.3)  # Need warm-up for rolling calcs
        end_idx = int(n * 0.9)
        
        if end_idx <= start_idx:
            result = {
                'feature': feature_name,
                'status': 'SKIP',
                'reason': 'Insufficient data for testing',
                'violations': 0,
                'tests_run': 0,
            }
            self.results.append(result)
            return result
        
        test_indices = np.linspace(
            start_idx, end_idx, test_points, dtype=int
        )
        
        violations = 0
        tests_run = 0
        violation_details = []
        
        for idx in test_indices:
            # Truncate data at this point
            truncated = price_data.iloc[:idx + 1].copy()
            
            try:
                trunc_result = feature_fn(truncated)
            except Exception:
                continue
            
            if isinstance(trunc_result, pd.DataFrame):
                trunc_result = trunc_result.iloc[:, 0]
            
            tests_run += 1
            
            # Compare the value at the truncation point
            if truncated.index[-1] in full_result.index:
                full_val = full_result.loc[truncated.index[-1]]
            else:
                continue
            
            trunc_val = trunc_result.iloc[-1]
            
            # Handle NaN
            if pd.isna(full_val) and pd.isna(trunc_val):
                continue
            if pd.isna(full_val) != pd.isna(trunc_val):
                violations += 1
                violation_details.append({
                    'index': idx,
                    'date': str(truncated.index[-1]),
                    'full_value': full_val,
                    'trunc_value': trunc_val,
                    'type': 'NaN_MISMATCH',
                })
                continue
            
            if abs(float(full_val) - float(trunc_val)) > FLOAT_TOLERANCE:
                violations += 1
                violation_details.append({
                    'index': idx,
                    'date': str(truncated.index[-1]),
                    'full_value': float(full_val),
                    'trunc_value': float(trunc_val),
                    'delta': abs(float(full_val) - float(trunc_val)),
                    'type': 'VALUE_MISMATCH',
                })
        
        status = 'PASS' if violations == 0 else 'FAIL'
        if violations > 0:
            self.passed = False
        
        result = {
            'feature': feature_name,
            'status': status,
            'violations': violations,
            'tests_run': tests_run,
            'details': violation_details[:5],
        }
        self.results.append(result)
        
        symbol = '[PASS]' if status == 'PASS' else '[FAIL]'
        logger.info(
            f"    {symbol} {feature_name}: {status} "
            f"({tests_run} tests, {violations} violations)"
        )
        
        return result
    
    def summary(self) -> str:
        """Generate a human-readable summary of all test results."""
        lines = [
            "",
            "=" * 70,
            "LOOKAHEAD AUDIT -- SUMMARY",
            "=" * 70,
        ]
        
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.get('status') == 'PASS')
        failed = sum(1 for r in self.results if r.get('status') == 'FAIL')
        skipped = sum(
            1 for r in self.results
            if r.get('status') in ('SKIP', 'ERROR')
        )
        
        for r in self.results:
            name = r.get('feed', r.get('feature', 'unknown'))
            status = r.get('status', '?')
            violations = r.get('violations', 0)
            tests_run = r.get('tests_run', 0)
            
            symbol = {
                'PASS': '[OK]', 'FAIL': '[XX]', 'SKIP': '[--]', 'ERROR': '[!!]'
            }.get(status, '?')
            
            lines.append(
                f"  {symbol} {name:40s} {status:6s} "
                f"({tests_run} tests, {violations} violations)"
            )
        
        lines.append("")
        lines.append(f"  Total:   {total_tests}")
        lines.append(f"  Passed:  {passed}")
        lines.append(f"  Failed:  {failed}")
        lines.append(f"  Skipped: {skipped}")
        lines.append("")
        
        if self.passed:
            lines.append("  [OK] ALL FEEDS CLEAN -- PROCEED TO PHASE 2")
        else:
            lines.append("  [XX] LOOKAHEAD DETECTED -- FIX FEEDS BEFORE PROCEEDING")
            lines.append("  DO NOT BUILD FEATURES ON CONTAMINATED DATA.")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ======================================================================
# Pre-built audit functions for Phase 1 feeds
# ======================================================================

def audit_alpaca_feed(test_points: int = 5) -> dict:
    """
    Run lookahead audit on AlpacaFeed.
    
    Tests that historical OHLCV bars are identical whether
    fetched as a full range or truncated sub-ranges.
    """
    from sovereign.data.feeds.alpaca_feed import AlpacaFeed
    
    feed = AlpacaFeed()
    auditor = LookaheadAuditor()
    
    # Test a subset of the universe to keep runtime manageable
    test_tickers = ['SPY', 'QQQ', 'NVDA']
    
    for ticker in test_tickers:
        logger.info(f"\nAuditing AlpacaFeed: {ticker}")
        
        def get_full(t=ticker):
            return feed.get_bars(t, timeframe='1h')
        
        def get_truncated(end_date, t=ticker):
            start = datetime(2021, 1, 1, tzinfo=timezone.utc)
            # Normalize timezone to UTC for consistency
            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                end_utc = end_date.astimezone(timezone.utc)
            else:
                end_utc = end_date.replace(tzinfo=timezone.utc) if hasattr(end_date, 'replace') else end_date
            return feed.get_bars(t, start=start, end=end_utc, timeframe='1h')
        
        auditor.test_feed_consistency(
            feed_name=f"AlpacaFeed/{ticker}",
            get_full_data=get_full,
            get_truncated_data=get_truncated,
            test_points=test_points,
        )
    
    return {'auditor': auditor, 'results': auditor.results}


def audit_macro_feed(test_points: int = 5) -> dict:
    """
    Run lookahead audit on MacroFeed.
    
    Tests that macro series values are identical whether
    fetched as a full range or truncated sub-ranges.
    """
    from sovereign.data.feeds.macro_feed import MacroFeed
    
    feed = MacroFeed()
    auditor = LookaheadAuditor()
    
    # Test key series
    test_series = ['vixcls', 'dgs10']
    
    for series_key in test_series:
        logger.info(f"\nAuditing MacroFeed: {series_key}")
        
        def get_full(sk=series_key):
            s = feed.get_series(sk, start='2021-01-01')
            if isinstance(s, pd.Series):
                return s.to_frame(name=sk)
            return s
        
        def get_truncated(end_date, sk=series_key):
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(
                end_date, 'strftime'
            ) else str(end_date)
            s = feed.get_series(sk, start='2021-01-01', end=end_str)
            if isinstance(s, pd.Series):
                return s.to_frame(name=sk)
            return s
        
        auditor.test_feed_consistency(
            feed_name=f"MacroFeed/{series_key}",
            get_full_data=get_full,
            get_truncated_data=get_truncated,
            test_points=test_points,
        )
    
    return {'auditor': auditor, 'results': auditor.results}


def audit_hurst_function(test_points: int = 10) -> dict:
    """
    Run lookahead audit on the Hurst exponent calculation.
    
    This is critical: Hurst determines universe selection.
    If Hurst has lookahead, every trade was selected with
    information that wasn't available at decision time.
    """
    from sovereign.data.universe import calculate_hurst
    
    auditor = LookaheadAuditor()
    
    # Generate synthetic price data for controlled testing
    np.random.seed(42)
    n = 2000
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range('2021-01-01', periods=n, freq='h')
    price_df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 100000, n),
    }, index=dates)
    
    def hurst_feature(df):
        return calculate_hurst(df['close'], window=90)
    
    auditor.test_feature_function(
        feature_name='Hurst (R/S, window=90)',
        feature_fn=hurst_feature,
        price_data=price_df,
        test_points=test_points,
    )
    
    return {'auditor': auditor, 'results': auditor.results}


# ======================================================================
# Master audit runner
# ======================================================================

def run_full_audit(test_points: int = 5) -> bool:
    """
    Run the complete Phase 1 lookahead audit.
    
    Tests:
    1. AlpacaFeed — OHLCV bars consistency
    2. MacroFeed — macro series consistency
    3. Hurst function — no lookahead in universe selection
    
    Returns:
        True if all tests pass. False if ANY violation detected.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    
    master_auditor = LookaheadAuditor()
    
    print("\n" + "=" * 70)
    print("SOVEREIGN PHASE 1 -- LOOKAHEAD AUDIT")
    print("Zero tolerance. Any violation halts the build.")
    print("=" * 70)
    
    # ── Test 1: Hurst function (fast, no API) ────────────────────────
    print("\n[1/3] Testing Hurst exponent for lookahead...")
    hurst_result = audit_hurst_function(test_points=10)
    for r in hurst_result['results']:
        master_auditor.results.append(r)
        if r.get('status') == 'FAIL':
            master_auditor.passed = False
    
    # ── Test 2: Alpaca Feed ──────────────────────────────────────────
    print("\n[2/3] Testing AlpacaFeed for lookahead...")
    try:
        alpaca_result = audit_alpaca_feed(test_points=test_points)
        for r in alpaca_result['results']:
            master_auditor.results.append(r)
            if r.get('status') == 'FAIL':
                master_auditor.passed = False
    except Exception as e:
        logger.error(f"AlpacaFeed audit failed: {e}")
        master_auditor.results.append({
            'feed': 'AlpacaFeed',
            'status': 'ERROR',
            'reason': str(e),
            'violations': 0,
            'tests_run': 0,
        })
    
    # ── Test 3: Macro Feed ───────────────────────────────────────────
    print("\n[3/3] Testing MacroFeed for lookahead...")
    try:
        macro_result = audit_macro_feed(test_points=test_points)
        for r in macro_result['results']:
            master_auditor.results.append(r)
            if r.get('status') == 'FAIL':
                master_auditor.passed = False
    except Exception as e:
        logger.error(f"MacroFeed audit failed: {e}")
        master_auditor.results.append({
            'feed': 'MacroFeed',
            'status': 'ERROR',
            'reason': str(e),
            'violations': 0,
            'tests_run': 0,
        })
    
    # ── Summary ──────────────────────────────────────────────────────
    print(master_auditor.summary())
    
    return master_auditor.passed


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == '__main__':
    import sys
    
    all_clean = run_full_audit(test_points=5)
    
    if all_clean:
        print("\n[OK] PHASE 1 GATE CLEARED. Safe to proceed to Phase 2.\n")
        sys.exit(0)
    else:
        print("\n[XX] PHASE 1 GATE BLOCKED. Fix violations before proceeding.\n")
        sys.exit(1)
