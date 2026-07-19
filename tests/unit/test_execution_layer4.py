"""Layer 4 — signal -> harness -> risk wiring."""
from datetime import date
from unittest.mock import patch

import pytest

from execution import harness, risk
from execution.harness import FillRecord, apply_risk

DAY = date(2026, 6, 16)


def _state(dd=0.0):
    peak = 100_000.0
    return risk.AccountState(equity=peak * (1 - dd), peak_equity=peak)


def test_allowed_fill_is_annotated_not_altered():
    rec = FillRecord(ticker="AAPL", date=str(DAY), signal_type="LONG")
    out = apply_risk(rec, _state(), 0.005)
    assert out.signal_type == "LONG"
    assert out.risk_allowed is True and out.risk_action == "ALLOW"
    assert out.risk_breached == []


def test_blocked_fill_becomes_skip_risk_and_keeps_evidence():
    """A risk gate that discards its refusals leaves no proof it fired."""
    rec = FillRecord(ticker="AAPL", date=str(DAY), signal_type="LONG")
    out = apply_risk(rec, _state(dd=0.06), 0.005)      # 6% DD -> halt
    assert out.signal_type == "SKIP_RISK"
    assert out.risk_allowed is False
    assert out.risk_breached, "breached articles must be recorded on the row"
    assert "Article 3" in out.reason


def test_per_trade_breach_blocks_and_names_article():
    rec = FillRecord(ticker="AAPL", date=str(DAY), signal_type="LONG")
    out = apply_risk(rec, _state(), 0.02)
    assert out.signal_type == "SKIP_RISK"
    assert any("ART1_PER_TRADE" in b for b in out.risk_breached)


def test_flatten_zeroes_size_multiplier():
    rec = FillRecord(ticker="AAPL", date=str(DAY), signal_type="LONG")
    out = apply_risk(rec, _state(dd=0.07), 0.001)
    assert out.risk_action == "FLATTEN" and out.risk_size_mult == 0.0


def test_skip_risk_counts_as_a_skip():
    assert harness.is_skip("SKIP_RISK") is True


def test_signal_index_empty_without_dir():
    assert harness._signal_index(DAY, None) == {}


def test_signals_are_authoritative_when_present(tmp_path):
    """With a signal file, only GO tickers may fill."""
    from execution import signals as S
    from execution.scan import Candidate

    def _c(sym, gap):
        c = Candidate(symbol=sym, day=DAY, prev_close=2.0)
        c.overnight_gap, c.log_vol, c.vol_0930 = gap, 4.0, 10_000
        c.price_1025, c.cum_vol_1025, c.gain_1025 = 3.0, 600_000, 0.5
        c._window_bars = [{"t": "2026-06-16T14:25:00Z"}] * 10
        c.bars = []
        return c

    good, bad = _c("GOOD", 0.40), _c("BAD", 0.40)
    with patch.object(S.scan, "scan_universe", return_value=[good]), \
         patch.object(S.borrow, "load_locate", return_value=None):
        S.write_signals(DAY, S.build_signals(DAY, check_news=False), tmp_path)

    idx = harness._signal_index(DAY, tmp_path)
    assert ("GOOD", "HYP-107") in idx
    assert ("BAD", "HYP-107") not in idx


def test_fallback_preserved_when_no_signal_file(tmp_path):
    """Without a signal file the harness keeps its prior standalone behaviour."""
    with patch.object(harness.scan, "scan_universe", return_value=[]), \
         patch.object(harness.borrow, "load_locate", return_value=None):
        out = harness.run_session(DAY, out_dir=tmp_path, check_news=False,
                                  signals_dir=None)
    assert out == []


def test_harness_still_does_not_import_bias():
    """Layer 4 wiring must not have smuggled the predictor in."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(harness))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
            mods.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    assert not any(m.split(".")[-1] == "bias" for m in mods)
