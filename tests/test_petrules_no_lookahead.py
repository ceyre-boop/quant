"""
tests/test_petrules_no_lookahead.py

The Petrules Gate leakage audit — the deliverable that makes the whole Gate trustworthy.

The Phase-0 audit's biggest finding: revision-path features were a silent lookahead leak (a
value that LOOKS knowable at T but was published later). The replay engine's job is to make
that structurally impossible by refusing to carry any value without a publication timestamp
and gating every value with one strict rule:

    knowable at freeze  iff  published_ts  <  freeze_ts   (STRICT)

This test proves the guard has TEETH (it flags the known leak modes on real dates) and then
audits the actually-built sample: every present feature was knowable strictly before its
event's freeze, and every label is forward (published at/after freeze). Mirrors the style of
tests/test_feature_label_isolation.py.

No model/learning/calibration/sizer code is touched — this is pure plumbing validation.
"""
from __future__ import annotations

import importlib
import random
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = "research.petrules"

prov = importlib.import_module(f"{PKG}.provenance")
replay = importlib.import_module(f"{PKG}.replay_engine")
paid = importlib.import_module(f"{PKG}.paid_stubs")
sources = importlib.import_module(f"{PKG}.sources")


# ─── THE RULE, with teeth: knowable_at is strict-before ───────────────────────

def test_knowable_at_is_strict_before():
    freeze = datetime(2024, 5, 15)
    assert prov.knowable_at(datetime(2024, 5, 14), freeze) is True
    assert prov.knowable_at(datetime(2024, 5, 15), freeze) is False   # same instant → excluded
    assert prov.knowable_at(datetime(2024, 5, 16), freeze) is False   # after → excluded
    assert prov.knowable_at(None, freeze) is False                    # no ts → never knowable


def test_filing_on_freeze_day_is_rejected():
    """A disclosed filing dated the freeze day must NOT count as knowable (conservative)."""
    pv = prov.Provenanced(value={"cluster": True}, source="sec_edgar.form4",
                          published_ts=datetime(2024, 5, 15))
    assert not pv.knowable_at(datetime(2024, 5, 15))
    with pytest.raises(prov.LookaheadError):
        prov.assert_knowable("disclosed_form4_cluster", pv, datetime(2024, 5, 15))


def test_absent_value_is_skipped_not_trusted():
    """An ABSENT value (offline/paid-stub) carries no ts and passes the guard by being unused."""
    pv = prov.Provenanced(value=None, source="paid_stubs.options_implied_move", published_ts=None)
    assert pv.is_present is False
    prov.assert_knowable("options_implied_move", pv, datetime(2024, 5, 15))  # no raise


# ─── The 13F trap on REAL dates: filing date vs period-of-report ──────────────

def test_13f_period_of_report_would_leak_but_filing_date_does_not():
    """Real Berkshire 13F: period end 2026-03-31, FILED 2026-05-15. At a freeze of 2026-04-15,
    timestamping by period-of-report (the audit's classic trap) would call it knowable — a
    ~45-day hindsight leak. Timestamping by FILING date correctly excludes it."""
    sample = sources.edgar_13f_sample()
    assert sample, "13F fixture missing"
    filing_str, period_str = sample["Berkshire"]["sample"][0]  # ["2026-05-15", "2026-03-31"]
    filing = datetime.fromisoformat(filing_str)
    period = datetime.fromisoformat(period_str)
    freeze = datetime(2026, 4, 15)  # between period end and filing
    assert prov.knowable_at(period, freeze) is True,  "sanity: period end precedes freeze"
    assert prov.knowable_at(filing, freeze) is False, "filing date must NOT be knowable at freeze"
    # the builder must use the filing date → the leak is structurally impossible
    assert period < freeze < filing


# ─── Paid interfaces are STUBS, not silent fabricators ────────────────────────

@pytest.mark.parametrize("fn_name", [
    "options_implied_move", "options_skew_direction",
    "implied_move_change_30d", "consensus_revision_momentum",
])
def test_paid_interfaces_raise_not_implemented(fn_name):
    fn = getattr(paid, fn_name)
    with pytest.raises(NotImplementedError):
        fn("AAPL", datetime(2024, 5, 15))


# ─── Isolation: the package imports nothing from the live execution path ──────

def test_package_does_not_import_execution_path():
    banned = ("sovereign", "ict_engine", "forex_exit_manager", "decide_exit")
    src_dir = ROOT / "research" / "petrules"
    offenders = []
    for py in src_dir.glob("*.py"):
        text = py.read_text()
        for b in banned:
            if f"import {b}" in text or f"from {b}" in text or f"import ict" in text:
                offenders.append((py.name, b))
    assert not offenders, f"petrules must not import the execution path: {offenders}"


# ─── THE DATA-LEVEL AUDIT: every built example is leak-free ───────────────────

class TestBuiltSampleAudit:
    """Audit the actually-built (offline, real-fixture) sample of frozen events."""

    @pytest.fixture(scope="class")
    def events(self):
        evs = replay.build_sample()  # offline: AAPL + DKS real AV fixtures
        if not evs:
            pytest.skip("sample did not build (fixtures missing)")
        return evs

    def test_sample_is_nonempty_and_real(self, events):
        assert len(events) >= 50, f"expected a real multi-quarter sample, got {len(events)}"

    def test_every_present_feature_is_knowable_at_freeze(self, events):
        """The core no-lookahead assertion on a random sample of built examples."""
        rng = random.Random(1474)
        for fe in rng.sample(events, min(50, len(events))):
            for name, pv in fe.present_features().items():
                assert pv.published_ts < fe.freeze_ts, (
                    f"LOOKAHEAD in {fe.event_id}: feature '{name}' published "
                    f"{pv.published_ts} not strictly before freeze {fe.freeze_ts}"
                )

    def test_every_label_is_forward(self, events):
        """A label must be published AT/AFTER freeze — a same-bar label would fail here."""
        for fe in events:
            if fe.label.is_present:
                assert fe.label.published_ts >= fe.freeze_ts, (
                    f"{fe.event_id}: label published {fe.label.published_ts} BEFORE freeze "
                    f"{fe.freeze_ts} — not forward-looking"
                )

    def test_history_feature_never_includes_the_event_itself(self, events):
        """earnings_surprise_history must draw only on PRIOR quarters (strictly before freeze),
        never the current print — the self-referential tautology guard."""
        for fe in events:
            hist = fe.features["earnings_surprise_history"]
            if hist.is_present:
                assert hist.published_ts < fe.freeze_ts

    def test_build_time_audit_reraises_on_injected_leak(self, events):
        """Prove FrozenEvent.audit() itself has teeth: inject a post-freeze feature ts."""
        fe = events[0]
        leaked = prov.Provenanced(value={"x": 1}, source="test.injected",
                                  published_ts=fe.freeze_ts)  # == freeze → not strictly before
        fe.features["injected_leak"] = leaked
        with pytest.raises(prov.LookaheadError):
            fe.audit()
        del fe.features["injected_leak"]  # restore for other tests


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
