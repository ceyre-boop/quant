"""Positioning-family event-study tests — synthetic, seeded, offline.

If a test here fails after a protocol edit, the runner and the locked prereg have
drifted — fix the runner, never the prereg (hash-locked; gate zero enforces it).
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign.research.positioning import event_study as es
from sovereign.research.positioning import v015_replay as vr

ROOT = Path(__file__).resolve().parents[1]


def _idx(n, start="2019-01-01"):
    return pd.bdate_range(start, periods=n)


def _weekly(n, start=date(2019, 1, 4)):
    return [start + timedelta(weeks=i) for i in range(n)]


ENTER_072 = lambda v: 1 if v > 0.95 else (-1 if v < 0.05 else 0)   # noqa: E731
REARM_072 = lambda v: 0.10 <= v <= 0.90                            # noqa: E731


class TestCrossings:
    def test_hysteresis_two_events(self):
        vals = [0.5, 0.96, 0.97, 0.5, 0.96]
        out = es.detect_crossings(_weekly(5), vals, ENTER_072, REARM_072)
        assert len(out) == 2 and all(s == 1 for _, s, _ in out)

    def test_no_rearm_inside_band_gap(self):
        vals = [0.5, 0.96, 0.92, 0.96]     # 0.92 NOT in [0.10,0.90] → still disarmed
        out = es.detect_crossings(_weekly(4), vals, ENTER_072, REARM_072)
        assert len(out) == 1

    def test_low_side_mirror(self):
        vals = [0.5, 0.03, 0.04, 0.5, 0.02]
        out = es.detect_crossings(_weekly(5), vals, ENTER_072, REARM_072)
        assert [s for _, s, _ in out] == [-1, -1]

    def test_deoverlap_five_weeks_one_event(self):
        vals = [0.96, 0.97, 0.98, 0.99, 0.96]
        assert len(es.detect_crossings(_weekly(5), vals, ENTER_072, REARM_072)) == 1


class TestDirections:
    def test_usdjpy_inversion_fade(self):
        # crowd long JPY -> fade JPY -> USDJPY RISES: pair side must be +1
        crossings = [(date(2020, 6, 5), 1, 0.97)]
        jp = es.make_events("USDJPY", crossings, lambda s: -s, lambda v: v - 0.95)
        eu = es.make_events("EURUSD", crossings, lambda s: -s, lambda v: v - 0.95)
        assert jp[0].side == 1 and eu[0].side == -1

    def test_signed_return_sign(self):
        idx = _idx(30, "2021-03-01")
        rising = pd.Series(np.linspace(100, 110, 30), index=idx)
        ev_up = es.Event("USDJPY", date(2021, 3, 1), +1, 0.02)
        rets, kept, dropped = es.signed_returns([ev_up], {"USDJPY": rising}, 10)
        assert dropped == 0 and rets[0] > 0
        ev_dn = es.Event("EURUSD", date(2021, 3, 1), -1, 0.02)
        rets2, _, _ = es.signed_returns([ev_dn], {"EURUSD": rising}, 10)
        assert rets2[0] < 0


class TestPermutation:
    def test_null_p_roughly_uniform(self):
        rng = np.random.default_rng(7)
        idx = _idx(400, "2016-01-04")
        ps = []
        for sim in range(60):
            closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 400))), index=idx)
            weekly = [idx[i].date() for i in range(0, 350, 5)]
            ev_dates = rng.choice(len(weekly) - 10, size=6, replace=False)
            events = [es.Event("EURUSD", weekly[i], int(rng.choice([-1, 1])), 0.01) for i in ev_dates]
            elig = es.eligible_dates(closes, weekly, 20)
            r = es.pooled_primary_p({"EURUSD": events}, {"EURUSD": closes}, {"EURUSD": elig},
                                    20, np.random.default_rng(42), n_perm=299)
            ps.append(r.p)
        assert sum(1 for p in ps if p < 0.05) <= 9      # ~uniform: expect ~3/60, tolerate 9

    def test_injected_effect_detected(self):
        rng = np.random.default_rng(11)
        idx = _idx(400, "2016-01-04")
        base = rng.normal(0, 0.002, 400)
        weekly = [idx[i].date() for i in range(0, 350, 5)]
        ev_idx = [10, 25, 40, 55]
        for wi in ev_idx:
            pos = idx.get_indexer([pd.Timestamp(weekly[wi])])[0]
            base[pos + 1:pos + 21] += 0.002            # +2% drift over the window
        closes = pd.Series(100 * np.exp(np.cumsum(base)), index=idx)
        events = [es.Event("EURUSD", weekly[i], +1, 0.01) for i in ev_idx]
        elig = es.eligible_dates(closes, weekly, 20)
        r = es.pooled_primary_p({"EURUSD": events}, {"EURUSD": closes}, {"EURUSD": elig},
                                20, np.random.default_rng(42), n_perm=999)
        assert r.p < 0.02, r


class TestEx2020:
    def test_window_touching_2020_excluded(self):
        idx = pd.bdate_range("2019-11-01", "2020-03-01")
        closes = pd.Series(np.linspace(1.1, 1.2, len(idx)), index=idx)
        ev = es.Event("EURUSD", date(2019, 12, 20), 1, 0.01)   # window crosses into 2020
        full, _, _ = es.signed_returns([ev], {"EURUSD": closes}, 10)
        ex, _, _ = es.signed_returns([ev], {"EURUSD": closes}, 10, ex2020=True)
        assert len(full) == 1 and len(ex) == 0


class TestV015Replay:
    def test_reconcile_guard_halts_on_tamper(self, tmp_path, monkeypatch):
        df = vr.load_trades()
        tampered = df.copy()
        tampered["pnl_pct"] = -tampered["pnl_pct"]   # sign flip — Sharpe negates (scale tampers are Sharpe-invariant)
        bars = {p: pd.Series(np.linspace(1, 2, 2600),
                             index=pd.bdate_range("2015-01-01", periods=2600))
                for p in tampered["pair"].unique()}
        with pytest.raises(SystemExit, match="RECONCILE GUARD"):
            vr.reconcile_guard(tampered, bars)

    def test_fwd_max_drawdown(self):
        idx = _idx(60)
        eq = pd.Series(np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 99, 30)]), index=idx)
        dd = vr.fwd_max_drawdown(eq, idx[29], 20)
        assert dd is not None and dd < -0.05
        assert vr.fwd_max_drawdown(eq, idx[55], 20) is None   # incomplete window


class TestLedgerAnnotate:
    def test_annotate_appends_only(self, tmp_path, monkeypatch):
        import scripts.research.run_positioning_family as run
        ledger = [{"id": "HYP-072", "status": "PREREGISTERED", "hash_lock": "abc"},
                  {"id": "HYP-999", "status": "REJECTED"}]
        lp = tmp_path / "ledger.json"
        lp.write_text(json.dumps(ledger))
        monkeypatch.setattr(run, "LEDGER", lp)
        run._annotate_ledger("HYP-072", {"date": "2026-07-02", "by": "test", "note": "n"})
        out = json.loads(lp.read_text())
        e = next(x for x in out if x["id"] == "HYP-072")
        assert len(e["annotations"]) == 1 and e["status"] == "PREREGISTERED" and e["hash_lock"] == "abc"
        assert list(tmp_path.glob("*.bak-*.json"))
        with pytest.raises(AssertionError):
            run._annotate_ledger("HYP-999", {"date": "x", "by": "t", "note": "n"})
