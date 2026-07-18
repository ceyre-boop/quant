"""Fill-side conventions and the spread accounting identity."""
import pytest

from execution.quotes import Quote, fill_price, is_wide, round_trip


def q(bid, ask, ts="2026-07-16T13:31:00Z", sym="TEST"):
    return Quote(symbol=sym, ts_utc=ts, bid=bid, ask=ask,
                 bid_size=100, ask_size=100)


def test_tghl_real_quote_shape():
    """The measured counter-example: real microcap spread at 09:30:59 is sub-1%,
    not the 1-15% the backtest model's docstring assumes."""
    quote = q(1.37, 1.38)
    assert quote.mid == pytest.approx(1.375)
    assert quote.spread_pct == pytest.approx(0.00727, abs=1e-4)
    assert quote.spread_pct < 0.01


def test_long_pays_ask_receives_bid():
    quote = q(10.00, 10.10)
    entry, _ = fill_price(quote, "LONG", "ENTRY")
    exit_, _ = fill_price(quote, "LONG", "EXIT")
    assert entry == 10.10        # buy at the offer
    assert exit_ == 10.00        # sell at the bid


def test_short_receives_bid_pays_ask():
    quote = q(10.00, 10.10)
    entry, _ = fill_price(quote, "SHORT", "ENTRY")
    exit_, _ = fill_price(quote, "SHORT", "EXIT")
    assert entry == 10.00        # sell at the bid
    assert exit_ == 10.10        # buy back at the offer


def test_half_spread_is_symmetric():
    quote = q(10.00, 10.10)
    _, h = fill_price(quote, "LONG", "ENTRY")
    assert h == pytest.approx(quote.spread_pct / 2)


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_gross_minus_net_equals_spread_cost(side):
    """THE ACCOUNTING IDENTITY. If this fails the fill model is wrong."""
    eq = q(10.00, 10.10)
    xq = q(11.00, 11.10)
    r = round_trip(eq, xq, side)
    assert r["gross_return"] - r["net_return"] == pytest.approx(
        r["spread_cost"], abs=1e-4)


def test_long_profits_when_price_rises():
    r = round_trip(q(10.00, 10.10), q(11.00, 11.10), "LONG")
    assert r["gross_return"] > 0
    assert r["net_return"] < r["gross_return"]      # spread always costs


def test_short_profits_when_price_falls():
    r = round_trip(q(11.00, 11.10), q(10.00, 10.10), "SHORT")
    assert r["gross_return"] > 0
    assert r["net_return"] < r["gross_return"]


def test_crossed_and_locked_are_unusable():
    assert q(10.10, 10.00).is_crossed() is True
    assert q(10.10, 10.00).is_usable() is False
    assert q(10.00, 10.00).is_locked() is True
    assert q(10.00, 10.00).is_usable() is False
    assert q(10.00, 10.10).is_usable() is True


def test_zero_side_is_unusable():
    assert q(0.0, 10.10).is_usable() is False
    assert q(10.00, 0.0).is_usable() is False


def test_wide_quote_is_flagged_not_dropped():
    """Wide quotes are real. Dropping them would bias measured spread downward."""
    assert is_wide(q(1.00, 1.50)) is True       # 40% spread
    assert is_wide(q(1.37, 1.38)) is False


def test_invalid_side_or_leg_raises():
    with pytest.raises(ValueError):
        fill_price(q(1.0, 1.1), "SIDEWAYS", "ENTRY")
    with pytest.raises(ValueError):
        fill_price(q(1.0, 1.1), "LONG", "MIDDLE")


def test_age_seconds():
    quote = q(1.0, 1.1, ts="2026-07-16T13:31:00Z")
    from datetime import datetime, timezone
    at = datetime(2026, 7, 16, 13, 31, 5, tzinfo=timezone.utc)
    assert quote.age_seconds(at) == pytest.approx(5.0)
