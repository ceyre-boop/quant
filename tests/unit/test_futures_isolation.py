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
from sovereign.futures import bar_feed, config, telegram_gateway, regime
from sovereign.futures import volume_profile as vpmod, cvd as cvdmod

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


def test_regime_isolated():
    _assert_clean(regime)


def test_volume_profile_isolated():
    _assert_clean(vpmod)


def test_cvd_isolated():
    _assert_clean(cvdmod)


# ── Increment 4 unit tests ────────────────────────────────────────────────────

def _vol_df(closes, vols):
    return pd.DataFrame({
        "Open": closes, "High": [c + 0.25 for c in closes], "Low": [c - 0.25 for c in closes],
        "Close": closes, "Volume": vols,
    })


def test_compute_profile_and_confluence():
    # heavy volume parked at 100 → POC ~100; entry at 100 should score on POC
    closes = [100.0] * 25 + [101.0, 99.0, 100.0, 102.0, 98.0]
    vols = [500] * 25 + [50, 50, 50, 50, 50]
    prof = vpmod.compute_profile(_vol_df(closes, vols))
    assert prof is not None and abs(prof["poc"] - 100.0) <= 1.0
    assert vpmod.confluence_score(prof["poc"], prof, tol_price=0.5) >= 1
    assert vpmod.confluence_score(99999, prof, tol_price=0.5) == 0


def test_cvd_fail_loud_then_signal():
    # <20 bars with volume>0 → None (fail loud)
    thin = _vol_df([100.0] * 10, [0] * 10)
    assert cvdmod.cvd_state(thin) is None
    # each bar closes ABOVE its open on real volume → positive delta → positive slope
    cl = [100.0 + i * 0.25 for i in range(30)]
    rising = pd.DataFrame({
        "Open": [c - 0.20 for c in cl], "High": [c + 0.05 for c in cl],
        "Low": [c - 0.25 for c in cl], "Close": cl, "Volume": [200] * 30,
    })
    st = cvdmod.cvd_state(rising)
    assert st is not None and st["slope"] > 0
    assert cvdmod.cvd_confirms("ORB", "LONG", st) is True
    assert cvdmod.cvd_confirms("ORB", "SHORT", st) is False
    assert cvdmod.cvd_confirms("ORB", "LONG", None) is None      # unknown when no state


def test_scale_in_never_grows_loser_and_runs():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "scripts"))
    import futures_replay as fr
    df = _synthetic_session(20)
    out = fr.simulate_session(df, "2026-06-05", "LONG", {}, "MES", "safe",
                              setups={"orb"}, scale_in=True)
    for t in out["trades"]:
        assert t["contracts"] >= 1
        assert "net_1lot_usd" in t                  # static counterfactual logged
        if t["net_usd"] < 0:
            assert t["contracts"] == 1              # losers never scaled


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


def test_vwap_bands_bracket_price():
    bands = strat.vwap_bands(_bars([100, 100, 100, 101, 99, 100, 100], vols=[100] * 7))
    assert bands is not None
    lower, upper, vwap, sigma = bands
    assert lower < vwap < upper and sigma >= 0


def test_vwap_mr_levels_stop_always_correct_side():
    # degenerate case from the live-IB proof: entry pierced below the lower band past the buffer.
    # bands = (lower, upper, vwap, sigma)
    bands = (7413.38, 7497.15, 7455.26, 41.89)
    s_long, t_long = strat.vwap_mr_levels("LONG", 7412.0, bands, "MES")
    assert s_long < 7412.0, "LONG stop must be strictly below entry"
    assert t_long == 7455.26
    s_short, t_short = strat.vwap_mr_levels("SHORT", 7498.0, bands, "MES")
    assert s_short > 7498.0, "SHORT stop must be strictly above entry"
    # normal case still sane
    s2, _ = strat.vwap_mr_levels("LONG", 7413.38, bands, "MES")
    assert s2 < 7413.38


def test_vwap_mr_signal_fades_extremes():
    from datetime import datetime, timezone
    now = datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc)
    bars = _bars([100] * 10 + [90], vols=[100] * 11)   # last bar stretched far below VWAP
    curr = strat.Indicators(90, 100, 30, 100, 100, 99, 99)
    assert strat.vwap_mr_signal(bars, curr, now=now, last_entry_time=None, trades_taken=0) == "LONG"
    bars_up = _bars([100] * 10 + [110], vols=[100] * 11)
    curr_up = strat.Indicators(110, 100, 70, 100, 100, 101, 101)
    assert strat.vwap_mr_signal(bars_up, curr_up, now=now, last_entry_time=None, trades_taken=0) == "SHORT"


def test_in_trade_window():
    from datetime import datetime, timezone
    # 13:45 UTC = 09:45 ET (EDT) → open window; 16:00 UTC = 12:00 ET → midday (blocked)
    assert strat.in_trade_window(datetime(2026, 6, 5, 13, 45, tzinfo=timezone.utc)) is True
    assert strat.in_trade_window(datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc)) is False


def test_regime_classify_and_router():
    import pandas as pd
    idx = pd.date_range("2026-06-05 13:30", periods=30, freq="1min", tz="UTC")
    # trending: price marches up, persistently above VWAP
    up = pd.DataFrame({"Open": range(100, 130), "High": [p + 0.5 for p in range(100, 130)],
                       "Low": [p - 0.5 for p in range(100, 130)], "Close": list(range(100, 130)),
                       "Volume": [100] * 30}, index=idx)
    reg = regime.classify_session(up, vix=15, adr_used_pct=0.9)
    assert reg["trend_state"] == "TRENDING"
    assert regime.setup_allowed("orb", reg)[0] is True
    assert regime.setup_allowed("vwap_mr", reg)[0] is False   # trending blocks MR


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
