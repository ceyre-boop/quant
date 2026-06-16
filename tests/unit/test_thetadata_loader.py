"""ThetaDataLoader integration tests — run against the LIVE ThetaTerminal v3 (port 25503).

Marked `integration` and skipped automatically if the terminal isn't reachable, so the
offline suite stays green. These validate the real v3 transport/parse/cache, not the mock.
"""
import socket

import pandas as pd
import pytest

from sovereign.research.vrp.data_loader import OPTION_CHAIN_COLUMNS, ThetaDataLoader

REQUIRED = ["strike", "call_bid", "call_ask", "call_mid", "put_bid", "put_ask", "put_mid"]
DATE, EXPIRY = "2022-03-07", "2022-03-18"     # a known liquid SPY monthly window


def _terminal_up(host="127.0.0.1", port=25503) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _terminal_up(), reason="ThetaTerminal v3 not running on 25503")]


def test_option_chain_returns_expected_columns():
    ch = ThetaDataLoader().get_option_chain("SPY", DATE, EXPIRY)
    assert list(ch.columns) == OPTION_CHAIN_COLUMNS
    assert len(ch) > 50                                   # a real monthly chain has many strikes
    for col in REQUIRED:
        assert ch[col].notna().all(), f"required column {col} has nulls"
    # mid is the bid/ask midpoint
    assert ((ch["call_mid"] - (ch["call_bid"] + ch["call_ask"]) / 2).abs() < 1e-9).all()


def test_caching_avoids_duplicate_requests(tmp_path):
    ld = ThetaDataLoader(cache_dir=tmp_path)
    calls = {"n": 0}
    real_get = ld._get
    ld._get = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real_get(*a, **k))[1]

    a = ld.get_option_chain("SPY", DATE, EXPIRY)
    after_first = calls["n"]
    assert after_first >= 1 and (tmp_path / "SPY" / f"{DATE}_{EXPIRY}.parquet").exists()

    b = ld.get_option_chain("SPY", DATE, EXPIRY)             # should hit the parquet cache
    assert calls["n"] == after_first, "second call issued a network request instead of using cache"
    pd.testing.assert_frame_equal(a, b)


def test_chain_atm_consistent_with_yfinance_close():
    """Stock history is FREE-tier-gated (403), so we cross-validate the OPTION data against
    the free yfinance underlying via put-call parity: the ATM strike (min |call_mid-put_mid|)
    must sit near the actual SPY close."""
    import yfinance as yf
    ch = ThetaDataLoader().get_option_chain("SPY", DATE, EXPIRY)
    atm_strike = float(ch.iloc[(ch["call_mid"] - ch["put_mid"]).abs().idxmin()]["strike"])

    h = yf.Ticker("SPY").history(start="2022-03-07", end="2022-03-08", auto_adjust=False)
    spy_close = float(h["Close"].iloc[0])
    assert abs(atm_strike - spy_close) / spy_close < 0.03, (
        f"ATM strike {atm_strike} far from SPY close {spy_close}")


def test_underlying_close_documents_tier_limit():
    """get_underlying_close is intentionally not on the backtest path (stock = FREE tier)."""
    with pytest.raises(NotImplementedError):
        ThetaDataLoader().get_underlying_close("SPY", DATE)
