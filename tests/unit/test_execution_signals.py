"""Layer 3 — GO/NO-GO emission contract."""
import json
from datetime import date
from unittest.mock import patch

import pytest

from execution import signals as S
from execution.scan import Candidate

DAY = date(2026, 6, 16)


def _c(sym, gap=0.40, lv=5.0, price=3.0, prev=2.0, vol=600_000, nbars=10):
    c = Candidate(symbol=sym, day=DAY, prev_close=prev)
    c.overnight_gap, c.log_vol, c.vol_0930 = gap, lv, int(10 ** lv)
    c.price_1025, c.cum_vol_1025 = price, vol
    c.gain_1025 = price / prev - 1
    c._window_bars = [{"t": f"2026-06-16T14:25:00Z"}] * nbars
    return c


def _build(cands, locate=None):
    with patch.object(S.scan, "scan_universe", return_value=cands), \
         patch.object(S.borrow, "load_locate", return_value=locate):
        return S.build_signals(DAY, check_news=False)


def test_no_go_rows_are_retained_with_reasons():
    """A zero-signal day must be auditable, not blank."""
    sigs = _build([_c("FAIL", gap=0.90)])           # gap above og_max
    no_gos = [s for s in sigs if s.decision == "NO_GO"]
    assert no_gos, "failures must be emitted, not dropped"
    assert all(s.reason for s in no_gos), "every NO_GO must carry a reason"
    assert any("og_max" in s.reason for s in no_gos)


def test_both_hypotheses_scored_for_every_candidate():
    sigs = _build([_c("AAA")])
    assert {s.hypothesis for s in sigs} == {"HYP-107", "HYP-093"}


def test_go_rows_are_ranked_contiguously():
    cands = [_c("AAA", gap=0.35, lv=4.0), _c("BBB", gap=0.40, lv=4.5),
             _c("CCC", gap=0.45, lv=5.0)]
    sigs = _build(cands)
    gos = sorted([s for s in sigs if s.hypothesis == "HYP-107" and s.decision == "GO"],
                 key=lambda s: s.rank)
    assert [s.rank for s in gos] == list(range(1, len(gos) + 1))


def test_rank_orders_by_headroom_not_magnitude():
    """More room inside the frozen band ranks higher. Descriptive, not conviction."""
    sigs = _build([_c("TIGHT", gap=0.55, lv=5.8), _c("ROOMY", gap=0.31, lv=4.0)])
    gos = {s.ticker: s for s in sigs
           if s.hypothesis == "HYP-107" and s.decision == "GO"}
    assert gos["ROOMY"].rank < gos["TIGHT"].rank


def test_short_without_borrow_is_no_go():
    """An unborrowable name is not a tradeable signal; recording it GO would
    overstate the opportunity set."""
    sigs = _build([_c("AAA", price=3.0, prev=2.0)], locate=None)
    s93 = [s for s in sigs if s.hypothesis == "HYP-093"][0]
    assert s93.decision == "NO_GO"
    assert s93.reason == "no_locate_snapshot"


def test_short_with_easy_borrow_is_go():
    sigs = _build([_c("AAA", price=3.0, prev=2.0)], locate={"AAA": "EASY"})
    s93 = [s for s in sigs if s.hypothesis == "HYP-093"][0]
    assert s93.decision == "GO"


def test_hard_borrow_is_no_go():
    sigs = _build([_c("AAA", price=3.0, prev=2.0)], locate={"AAA": "HARD"})
    s93 = [s for s in sigs if s.hypothesis == "HYP-093"][0]
    assert s93.decision == "NO_GO" and s93.reason == "tier_HARD"


def test_signal_ids_are_stable_and_unique():
    sigs = _build([_c("AAA"), _c("BBB")])
    ids = [s.signal_id for s in sigs]
    assert len(ids) == len(set(ids))
    assert all(s.signal_id == f"{DAY}:{s.ticker}:{s.hypothesis}" for s in sigs)


def test_frozen_hash_stamped_on_every_signal():
    from execution.config import FROZEN_HASH
    assert all(s.frozen_hash == FROZEN_HASH for s in _build([_c("AAA")]))


def test_written_file_keeps_no_go_and_counts(tmp_path):
    sigs = _build([_c("AAA"), _c("FAIL", gap=0.9)])
    p = S.write_signals(DAY, sigs, tmp_path)
    doc = json.loads(p.read_text())
    assert doc["n_go"] + doc["n_no_go"] == len(doc["signals"])
    assert doc["n_no_go"] > 0
    assert doc["frozen_hash"]


def test_go_list_helper_filters(tmp_path):
    sigs = _build([_c("AAA"), _c("FAIL", gap=0.9)])
    S.write_signals(DAY, sigs, tmp_path)
    assert all(s["decision"] == "GO" for s in S.go_list(DAY, tmp_path))


def test_empty_universe_writes_auditable_zero(tmp_path):
    p = S.write_signals(DAY, [], tmp_path)
    doc = json.loads(p.read_text())
    assert doc["n_go"] == 0 and doc["n_candidates"] == 0
    assert doc["signals"] == []
