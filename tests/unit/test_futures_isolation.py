"""Futures sandbox isolation + scalp_strategy unit tests.

Isolation: the Track-2 futures sandbox must stay self-contained — no imports from
forex/ICT/intelligence/oracle (TRADING_PHILOSOPHY time-horizon doctrine + CLAUDE.md).
It may import sovereign.futures.* and config (shared, allowed).
"""
import inspect
import sys
from pathlib import Path

import pandas as pd

from sovereign.futures import scalp_strategy as strat
from sovereign.futures import bar_feed, config, telegram_gateway

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import futures_replay as fr  # noqa: E402

FORBIDDEN = [
    "from sovereign.forex", "import sovereign.forex",
    "from sovereign.intelligence", "import sovereign.intelligence",
    "from sovereign.oracle", "import sovereign.oracle",
    "from ict", "import ict",
    "from layer1", "import layer1",
    "from layer2", "import layer2",
]


def _assert_clean(module):
    src = inspect.getsource(module)
    for phrase in FORBIDDEN:
        assert phrase not in src, f"Isolation violated: '{phrase}' in {module.__name__}"


def test_scalp_strategy_isolated():
    _assert_clean(strat)


def test_bar_feed_isolated():
    _assert_clean(bar_feed)


def test_config_isolated():
    _assert_clean(config)


def test_telegram_gateway_isolated():
    _assert_clean(telegram_gateway)


def test_telegram_parse_reply():
    assert telegram_gateway.parse_reply("big now") == {"action": "enter", "size": "big", "timing": "now"}
    assert telegram_gateway.parse_reply("small wait") == {"action": "enter", "size": "small", "timing": "wait"}
    assert telegram_gateway.parse_reply("skip") == {"action": "skip"}
    assert telegram_gateway.parse_reply("garbage") is None


# ── scalp_strategy unit tests ─────────────────────────────────────────────────

def _bars(closes, vols=None):
    n = len(closes)
    vols = vols or [100] * n
    return pd.DataFrame({
        "Open": closes, "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes], "Close": closes, "Volume": vols,
    })


def test_compute_indicators_shapes():
    ind = strat.compute_indicators(_bars([100, 101, 102, 103, 104]))
    assert ind.last_price == 104
    assert ind.vwap > 0 and 0 <= ind.rsi <= 100


def test_round_turn_cost_is_real():
    # MES: 1 tick/side * 0.25 * 2 * $5 + $0.74 = $3.24
    assert config.round_turn_cost_usd("MES") == 3.24
    assert config.round_turn_cost_usd("MNQ") == 1.74


def test_compute_r_long_and_short():
    assert strat.compute_r(100, 99, 102, "LONG") == 2.0
    assert strat.compute_r(100, 101, 98, "SHORT") == 2.0


def test_target_from_rr():
    assert strat.target_from_rr("LONG", 100, 99, rr=1.0) == 101
    assert strat.target_from_rr("SHORT", 100, 101, rr=2.0) == 98


def test_kill_level_picks_soonest():
    # SHORT dies above -> min of candidate levels
    kl = strat.kill_level("SHORT", {"overnight_high": 105.0}, oracle_invalidation=103.0)
    assert kl == 103.0
    # LONG dies below -> max of candidate levels
    kl = strat.kill_level("LONG", {"overnight_low": 95.0}, oracle_invalidation=97.0)
    assert kl == 97.0


def test_live_session_bars_passthrough():
    """bar_feed.live_session_bars reuses the bridge's historical_bars with RTH 1-min."""
    class FakeBridge:
        def __init__(self): self.calls = []
        def historical_bars(self, contract, duration, bar_size, rth):
            self.calls.append((duration, bar_size, rth)); return "SENTINEL"
    b = FakeBridge()
    assert bar_feed.live_session_bars(b, object()) == "SENTINEL"
    assert b.calls == [("1 D", "1 min", True)]


def _synthetic_session(n=12):
    idx = pd.date_range("2026-06-05 13:30", periods=n, freq="1min", tz="UTC")
    px = [100 - i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "Open": px, "High": [p + 0.3 for p in px], "Low": [p - 0.3 for p in px],
        "Close": px, "Volume": [200] * n,
    }, index=idx)


def test_on_event_does_not_change_results():
    """The player's renderer hook must not alter what the batch report computes."""
    df = _synthetic_session()
    silent = fr.simulate_session(df, "2026-06-05", "SHORT", {}, "MES", "safe")
    events = []
    animated = fr.simulate_session(df, "2026-06-05", "SHORT", {}, "MES", "safe",
                                   on_event=lambda k, p: events.append(k))
    assert animated["trades"] == silent["trades"]      # identical engine output
    assert animated["net_usd"] == silent["net_usd"]
    assert events.count("bar") >= 1                      # renderer actually got bar events


def test_micro_signal_respects_bias_direction():
    from datetime import datetime, timezone
    now = datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)
    # curr above vwap+emas, rsi crossed up, big volume; prev below
    curr = strat.Indicators(last_price=101, vwap=100, rsi=55, curr_volume=1000,
                            avg_volume=100, ema_fast=99, ema_slow=98)
    prev = strat.Indicators(last_price=99, vwap=100, rsi=45, curr_volume=0,
                            avg_volume=0, ema_fast=0, ema_slow=0)
    assert strat.micro_signal("LONG", curr, prev, now=now,
                              last_entry_time=None, trades_taken=0) == "LONG"
    # opposite bias blocks it
    assert strat.micro_signal("SHORT", curr, prev, now=now,
                              last_entry_time=None, trades_taken=0) is None
