"""
Sovereign Trading Intelligence -- Hurst Exponent Feature (Warp Drive 2.1)
Level 3 Optimization: Vectorized Sliding Window Matrix.
"""
import numpy as np
import pandas as pd
import logging
from numba import njit, prange
from config.loader import params

logger = logging.getLogger(__name__)

@njit(cache=True)
def _hurst_rs_matrix(matrix):
    """
    Calculates Hurst exponent for an entire matrix of windows at once.
    Matrix shape: (n_windows, window_size)
    """
    n_windows, window_size = matrix.shape
    log_matrix = np.log(matrix)
    
    # Vectorized Log Returns: (n_windows, window_size - 1)
    returns = np.diff(log_matrix, axis=1)
    n_ret = window_size - 1
    
    # Rescaled Range math across axis 1
    means = np.zeros(n_windows)
    for i in range(n_windows):
        means[i] = np.mean(returns[i])
        
    # Standard deviation per window
    stds = np.zeros(n_windows)
    for i in range(n_windows):
        stds[i] = np.std(returns[i])
        
    # Cumulative Max/Min deviations
    # We do this iteratively within Numba to save memory overhead
    results = np.full(n_windows, 0.5)
    log_n_ret = np.log(n_ret)
    
    for i in range(n_windows):
        if stds[i] == 0:
            continue
            
        ret_window = returns[i]
        curr_mean = means[i]
        
        # Inner loop for Rescaled Range
        max_dev = -1e10
        min_dev = 1e10
        cum_sum = 0.0
        for j in range(n_ret):
            cum_sum += (ret_window[j] - curr_mean)
            if cum_sum > max_dev: max_dev = cum_sum
            if cum_sum < min_dev: min_dev = cum_sum
            
        r = max_dev - min_dev
        if r > 0:
            results[i] = np.log(r / stds[i]) / log_n_ret
            
    return results

def compute_hurst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Hurst features with Level 3 Matrix Acceleration.
    """
    close = df['close'].values.astype(np.float64)
    p = params['regime']
    
    def _get_rolling_matrix(arr, window):
        # The 'Strisciando' Vectorization Trick
        shape = (arr.size - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    w_short = p['hurst_window_short']
    w_long = p['hurst_window_long']
    
    # 1. Expand into matrices
    mat_short = _get_rolling_matrix(close, w_short)
    mat_long = _get_rolling_matrix(close, w_long)
    
    # 2. Compute via JIT matrix engine
    h_short_vals = _hurst_rs_matrix(mat_short)
    h_long_vals = _hurst_rs_matrix(mat_long)
    
    # 3. Realign with index
    h_short = np.full(len(close), np.nan)
    h_long = np.full(len(close), np.nan)
    h_short[w_short-1:] = h_short_vals
    h_long[w_long-1:] = h_long_vals
    
    h_long_series = pd.Series(h_long, index=df.index)
    hurst_velocity = h_long_series.diff(10)

    return pd.DataFrame({
        'hurst_short':    h_short,
        'hurst_long':     h_long,
        'hurst_velocity': hurst_velocity,
    }, index=df.index)
