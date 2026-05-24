"""
sovereign/risk/market_memory.py
23-feature market fingerprint extractor used by AlexandrianLibrary.
"""
import numpy as np

FEATURE_NAMES = [
    'spy_30d_return', 'spy_60d_return', 'spy_90d_return',
    'spy_vol_30d', 'spy_vol_60d',
    'vix_level', 'vix_30d_change', 'vix_60d_change',
    'gold_30d_return', 'gold_60d_return', 'gold_spy_corr_60d',
    'spy_200sma_dist', 'spy_50sma_dist',
    'consecutive_down_weeks', 'consecutive_up_weeks',
    'drawdown_from_peak', 'spy_skew_30d', 'spy_kurt_30d',
    'hy_spread_proxy', 'vix_spy_beta',
    'momentum_score', 'reversal_score', 'regime_composite',
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(
    spy_prices: np.ndarray,
    vix_prices: np.ndarray = None,
    gold_prices: np.ndarray = None,
    hy_spread: np.ndarray = None,
    dxy_prices: np.ndarray = None,
) -> np.ndarray:
    s = np.array(spy_prices, dtype=float)
    n = len(s)
    v = np.array(vix_prices, dtype=float) if vix_prices is not None else np.full(n, 18.0)
    g = np.array(gold_prices, dtype=float) if gold_prices is not None else np.full(n, 180.0)

    def pct(arr, lookback):
        if len(arr) >= lookback + 1:
            return float(arr[-1] / arr[-(lookback + 1)] - 1)
        return 0.0

    def vol(arr, lookback):
        if len(arr) >= lookback + 1:
            r = np.diff(arr[-(lookback + 1):]) / arr[-(lookback + 1):-1]
            return float(np.std(r))
        return 0.0

    feats = [
        pct(s, 30),                   # spy_30d_return
        pct(s, 60),                   # spy_60d_return
        pct(s, 90),                   # spy_90d_return
        vol(s, 30),                   # spy_vol_30d
        vol(s, 60),                   # spy_vol_60d
        float(v[-1] / 100),           # vix_level (normalised)
        pct(v, 30),                   # vix_30d_change
        pct(v, 60),                   # vix_60d_change
        pct(g, 30),                   # gold_30d_return
        pct(g, 60),                   # gold_60d_return
    ]

    # gold_spy_corr_60d
    if n >= 61:
        sr = np.diff(np.log(s[-61:]))
        gr = np.diff(np.log(g[-61:]))
        corr = float(np.corrcoef(sr, gr)[0, 1]) if sr.std() > 0 and gr.std() > 0 else 0.0
    else:
        corr = 0.0
    feats.append(corr)

    # sma distances
    sma200 = float(s[-1] / np.mean(s[max(0, n - 200):]) - 1)
    sma50  = float(s[-1] / np.mean(s[max(0, n - 50):]) - 1)
    feats += [sma200, sma50]

    # consecutive down/up weeks (weekly closes = every 5th bar)
    weekly = s[::5]
    down = 0
    for i in range(len(weekly) - 1, 0, -1):
        if weekly[i] < weekly[i - 1]: down += 1
        else: break
    up = 0
    for i in range(len(weekly) - 1, 0, -1):
        if weekly[i] > weekly[i - 1]: up += 1
        else: break
    feats += [float(down), float(up)]

    # drawdown from peak
    peak = float(np.maximum.accumulate(s)[-1])
    feats.append(float((s[-1] - peak) / peak) if peak > 0 else 0.0)

    # skew / excess kurtosis (30d)
    if n >= 31:
        rets = np.diff(s[-31:]) / s[-31:-1]
        std = rets.std()
        skew = float(((rets - rets.mean()) ** 3).mean() / (std ** 3 + 1e-10))
        kurt = float(((rets - rets.mean()) ** 4).mean() / (std ** 4 + 1e-10)) - 3
    else:
        skew, kurt = 0.0, 0.0
    feats += [skew, kurt]

    # hy_spread proxy (VIX / 10 as rough credit proxy)
    feats.append(float(v[-1] / 10))

    # vix_spy_beta
    if n >= 31:
        sr30 = np.diff(s[-31:]) / s[-31:-1]
        vr30 = np.diff(v[-31:]) / (v[-31:-1] + 1e-6)
        cov = np.cov(sr30, vr30)
        beta = float(cov[0, 1] / (np.var(sr30) + 1e-10))
    else:
        beta = 0.0
    feats.append(beta)

    # composite scores
    momentum  = feats[0] + feats[1] + feats[2]
    reversal  = -(feats[0] + feats[1])
    regime    = feats[5] * 0.3 + abs(feats[13]) * 0.4 + float(down) * 0.3
    feats += [momentum, reversal, regime]

    arr = np.array(feats[:N_FEATURES], dtype=float)
    # Clip extremes
    arr = np.clip(arr, -5.0, 5.0)
    return arr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
