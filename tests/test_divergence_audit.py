"""Shadow-divergence analyzer tests — SYNTHETIC log lines only.

Never reads data/exec/* (the real shadow log). Synthetic timelines are produced
by running the REAL forex_exit_manager.step_trade over synthetic bars and
serializing with run_daily's exact rounding — so the golden case exercises the
same kernel the audit replays.

If a test here fails after a spec edit, the spec and the analyzer drifted —
amend audit/divergence_spec.md (with a §10 entry) and the analyzer together.
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest

import audit.shadow_divergence as sd
from sovereign.execution.forex_exit_manager import (
    Action, MarketBar, cfg_for_pair, init_trade_state, step_trade,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "audit" / "divergence_spec.md"


def synth_records(closes, pair="EUR_USD", trade_id="T1", direction=1, entry=1.1000,
                  atr=0.005, start="2026-06-30", mode="SHADOW"):
    """Real step_trade over synthetic bars → run_daily-shaped records."""
    cfg = cfg_for_pair(pair)
    st = init_trade_state(trade_id, pair, direction, entry, atr, 60, start + "T00:00:00Z", cfg)
    recs, d = [], date.fromisoformat(start)
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        bar = MarketBar(pair=pair, date=d.isoformat(), close=c, atr_pct=atr, signal=direction,
                        hold_today=60)
        res = step_trade(st, bar, cfg)
        recs.append({
            "run_ts": d.isoformat() + "T12:31:00+00:00", "mode": mode, "trade_id": trade_id,
            "pair": pair, "direction": "LONG" if direction == 1 else "SHORT",
            "bar_date": d.isoformat(), "close": c, "atr_pct": round(atr, 6),
            "signal": direction, "hold_count": res.new_state.hold_count, "hold_limit": 60,
            "best_price": round(res.new_state.best_price, 5),
            "worst_price": round(res.new_state.worst_price, 5),
            "initial_stop": round(st.stop_price, 5),
            "decision": res.decision.name, "action": res.action.value,
            "would_amend_stop_to": round(res.amend_to, 5) if res.amend_to is not None else None,
            "trail_price": round(res.trail_price, 5), "reentry_signal": res.reentry_signal,
        })
        st = res.new_state
        if res.action == Action.CLOSE:
            break
        d += timedelta(days=1)
    return recs


def write_log(tmp_path, recs):
    p = tmp_path / "shadow.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


@pytest.fixture(scope="module")
def spec():
    s, sha, ver = sd.load_spec(SPEC)
    return s


# ── spec contract ────────────────────────────────────────────────────────────

def test_spec_single_yaml_fence_and_hash():
    s, sha, ver = sd.load_spec(SPEC)
    assert ver == s["spec_version"] >= 1
    assert len(sha) == 64
    for key in ("tol_price_abs", "tol_atr_quantum", "tol_safety", "l1_required_pass_rate",
                "l2_decision_match_min", "c5_allowed", "min_scored_weekdays",
                "live_records_allowed", "staleness_grace_min", "messages_cap",
                "close_delta_warn_rel"):
        assert key in s, f"spec missing {key}"
    doubled = SPEC.read_text() + "\n```yaml audit-spec\nx: 1\n```\n"
    two = SPEC.parent / "_two_fences_tmp.md"
    try:
        two.write_text(doubled)
        with pytest.raises(RuntimeError):
            sd.load_spec(two)
    finally:
        two.unlink()


# ── L1 ───────────────────────────────────────────────────────────────────────

def test_l1_golden_determinism(spec):
    recs = synth_records([1.101, 1.104, 1.108, 1.106, 1.111, 1.109, 1.102])
    r = sd.replay_l1(recs, spec, incidents={})
    assert not r.failed, r.failed
    assert r.passed + r.boundary == len(recs)
    assert not r.continuity


def test_l1_detects_tampered_decision(spec):
    recs = synth_records([1.101, 1.104, 1.108, 1.106])
    victim = next(r for r in recs if r["decision"] == "HOLD" and r["hold_count"] > 1)
    victim["decision"] = "TRAILING_ATR"
    victim["action"] = "CLOSE"
    r = sd.replay_l1(recs, spec, incidents={})
    assert len(r.failed) == 1 and r.failed[0]["bar_date"] == victim["bar_date"]


def test_l1_trail_tolerance_rounding_boundary(spec):
    recs = synth_records([1.101, 1.104])
    ok = sd.replay_l1(recs, spec, incidents={})
    assert not ok.failed
    recs[-1]["trail_price"] = round(recs[-1]["trail_price"] + 10 * spec["tol_price_abs"], 5)
    bad = sd.replay_l1(recs, spec, incidents={})
    assert any("trail" in p for f in bad.failed for p in f["problems"])


def test_l1_jpy_price_level_tolerance(spec):
    """spec v2: the atr round-6 quantization scales with price — a USDJPY-level
    timeline whose logged values carry full run_daily rounding must pass L1."""
    recs = synth_records([161.2, 161.9, 162.4, 162.1, 162.8], pair="USD_JPY",
                         entry=161.0, atr=0.006)
    r = sd.replay_l1(recs, spec, incidents={})
    assert not r.failed, r.failed
    # and the price-scaled tolerance is genuinely larger than the flat base
    assert sd._tol_price(spec, 161.0, 1.25) > 5 * float(spec["tol_price_abs"])


def test_incident_matches_run_date_not_only_bar_date(spec):
    """spec v2: a double-step performed on run-date D re-steps an EARLIER bar_date;
    the incident register must classify it C3 via run_ts."""
    recs = synth_records([1.101, 1.104, 1.108])
    recs[2]["hold_count"] = recs[1]["hold_count"] + 2
    run_day = recs[2]["run_ts"][:10]
    r = sd.replay_l1(recs, spec, incidents={run_day: "documented run-date incident"})
    assert any(c["class"] == "C3" for c in r.continuity)
    assert not any(c["class"] == "C5" for c in r.continuity)


def test_chaining_across_skip_duplicate(spec):
    recs = synth_records([1.101, 1.104, 1.108])
    dup = {"run_ts": recs[1]["run_ts"], "mode": "SHADOW", "trade_id": "T1", "pair": "EUR_USD",
           "bar_date": recs[1]["bar_date"], "action": "SKIP_DUPLICATE", "reason": "already stepped"}
    recs_with = recs[:2] + [dup] + recs[2:]
    r = sd.replay_l1(recs_with, spec, incidents={})
    assert not r.failed and not r.continuity


def test_double_step_c3_vs_c5(spec):
    recs = synth_records([1.101, 1.104, 1.108])
    recs[2]["hold_count"] = recs[1]["hold_count"] + 2  # jump
    day = recs[2]["bar_date"]
    c5 = sd.replay_l1(recs, spec, incidents={})
    assert any(c["class"] == "C5" for c in c5.continuity)
    c3 = sd.replay_l1(recs, spec, incidents={day: "documented incident"})
    assert any(c["class"] == "C3" for c in c3.continuity)
    assert not any(c["class"] == "C5" for c in c3.continuity)


def test_entry_state_recovery_both_branches():
    up = synth_records([1.105])   # close above entry → best=close, worst=entry
    dn = synth_records([1.095])   # close below entry → best=entry, worst=close
    for recs, expect in ((up, 1.1000), (dn, 1.1000)):
        first = recs[0]
        rc = round(first["close"], 5)
        entry = first["worst_price"] if first["best_price"] == rc else first["best_price"]
        assert entry == pytest.approx(expect, abs=1e-9)


# ── L2 ───────────────────────────────────────────────────────────────────────

class _FakeL2:
    """Run replay_l2's logic without yfinance by monkeypatching fetch."""


def _fake_bars(dates, closes):
    import pandas as pd
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame({"Open": closes, "High": [c * 1.001 for c in closes],
                       "Low": [c * 0.999 for c in closes], "Close": closes}, index=idx)
    return df


def test_l2_seeded_c1(spec, monkeypatch, tmp_path):
    recs = synth_records([1.101, 1.104, 1.108, 1.111])
    dates = [r["bar_date"] for r in recs]
    yf_closes = [float(r["close"]) for r in recs]
    yf_closes[2] = 1.020  # yfinance shows a crash OANDA didn't → stop hit on yf path only
    df = _fake_bars(dates, yf_closes)
    monkeypatch.setattr(sd, "fetch_yf_bars", lambda pair, s, e, o: (df, 0))
    r = sd.replay_l2(recs, spec, offline=False, incidents={})
    assert r.status == "OK" and r.scored >= 3
    assert any(m["class"] == "C1" and "substituting OANDA" in m["detail"] for m in r.mismatches), r.mismatches


def test_l2_constant_shift_alignment(spec, monkeypatch):
    recs = synth_records([1.101, 1.104, 1.108, 1.111, 1.115])
    shifted = [(date.fromisoformat(r["bar_date"]) + timedelta(days=1)).isoformat() for r in recs]
    df = _fake_bars(shifted, [float(r["close"]) for r in recs])
    monkeypatch.setattr(sd, "fetch_yf_bars", lambda pair, s, e, o: (df, 0))
    r = sd.replay_l2(recs, spec, offline=False, incidents={})
    assert r.shifts.get("EUR_USD") == 1
    assert not [m for m in r.mismatches if m["class"] == "C2"], r.mismatches


def test_l2_offline_skipped_loudly(spec, monkeypatch):
    recs = synth_records([1.101, 1.104])
    monkeypatch.setattr(sd, "fetch_yf_bars", lambda pair, s, e, o: (None, 0))
    r = sd.replay_l2(recs, spec, offline=True, incidents={})
    assert r.status == "SKIPPED_OFFLINE" and r.scored == 0


def test_post_close_tail_not_l2_scored(spec, monkeypatch):
    closes = [1.101, 1.104, 0.9, 1.101, 1.102]  # bar 3 = stop hit → CLOSE
    recs = synth_records(closes)
    close_idx = next(i for i, r in enumerate(recs) if r["action"] == "CLOSE")
    tail = synth_records(closes)[:close_idx + 1]
    assert recs[close_idx]["decision"] != "HOLD"
    dates = [r["bar_date"] for r in tail]
    df = _fake_bars(dates, [float(r["close"]) for r in tail])
    monkeypatch.setattr(sd, "fetch_yf_bars", lambda pair, s, e, o: (df, 0))
    r = sd.replay_l2(tail, spec, offline=False, incidents={})
    assert r.scored <= close_idx + 1  # nothing scored past the first CLOSE


def test_skip_records_c4_excluded(spec):
    recs = synth_records([1.101, 1.104])
    recs.append({"run_ts": "2026-07-01T12:31:00+00:00", "mode": "SHADOW", "trade_id": "T1",
                 "pair": "EUR_USD", "action": "SKIP", "reason": "no market data"})
    r = sd.replay_l1(recs, spec, incidents={})
    assert not r.failed  # SKIP ignored by L1
    gate = sd.evaluate_gates(r, sd.L2Result(), recs, spec)
    assert gate["gates"]["no_live_records"][0] == "PASS"


# ── gates / escalation ───────────────────────────────────────────────────────

def test_live_mode_record_urgent_and_gate_fail(spec):
    recs = synth_records([1.101, 1.104], mode="LIVE")
    gate = sd.evaluate_gates(sd.replay_l1(recs, spec, {}), sd.L2Result(), recs, spec)
    assert gate["gates"]["no_live_records"][0] == "FAIL"
    assert gate["overall"] == "NO-GO"


def test_gate_scorecard_pending_min_days(spec):
    recs = synth_records([1.101, 1.104, 1.108])
    l1 = sd.replay_l1(recs, spec, {})
    l2 = sd.L2Result(status="OK", scored=3, matched=3)
    gate = sd.evaluate_gates(l1, l2, recs, spec)
    assert gate["gates"]["coverage"][0] == "PENDING"
    assert gate["overall"] == "NOT-YET"


def test_escalation_schema_prefix_and_cap50(spec, monkeypatch, tmp_path):
    msgs = tmp_path / "messages.json"
    msgs.write_text(json.dumps({"messages": [
        {"text": f"old {i}", "priority": "FYI"} for i in range(60)]}))
    monkeypatch.setattr(sd, "MESSAGES_PATH", msgs)
    n = sd.escalate([("URGENT", "TEST_EVENT", "synthetic")], "2026-07-02", cap=50)
    doc = json.loads(msgs.read_text())
    assert n == 1
    assert doc["messages"][0]["text"].startswith("[AUDIT] TEST_EVENT")
    assert doc["messages"][0]["source"] == "shadow_audit"
    assert len(doc["messages"]) <= 50
    n2 = sd.escalate([("URGENT", "TEST_EVENT", "synthetic")], "2026-07-02", cap=50)
    assert n2 == 0  # deduped


def test_ambiguous_hold_today_cb_refresh_pass_with_note(spec):
    """A record whose decision only reproduces under the alternate hold_today passes with a note."""
    pair, cfg = "EUR_USD", cfg_for_pair("EUR_USD")
    st = init_trade_state("T9", pair, 1, 1.1, 0.005, 60, "2026-01-01T00:00:00Z", cfg)
    # walk hold_count to 20 with flat closes (no exits), then craft the cb_refresh bar
    for i in range(20):
        res = step_trade(st, MarketBar(pair, f"d{i}", 1.1, 0.005, 1, 60), cfg)
        st = res.new_state
    bar = MarketBar(pair, "2026-02-02", 1.1, 0.005, 1, 29)  # hold_today<30 → CB_REFRESH
    res = step_trade(st, bar, cfg)
    assert res.decision.name == "CB_REFRESH"
    rec = {"run_ts": "2026-02-02T12:31:00+00:00", "mode": "SHADOW", "trade_id": "T9", "pair": pair,
           "direction": "LONG", "bar_date": "2026-02-02", "close": 1.1, "atr_pct": 0.005,
           "signal": 1, "hold_count": res.new_state.hold_count, "hold_limit": 60,
           "best_price": round(res.new_state.best_price, 5),
           "worst_price": round(res.new_state.worst_price, 5),
           "initial_stop": round(st.stop_price, 5), "decision": "CB_REFRESH", "action": "CLOSE",
           "would_amend_stop_to": None, "trail_price": round(res.trail_price, 5),
           "reentry_signal": res.reentry_signal}
    r = sd.replay_l1([rec], spec, incidents={})
    assert not r.failed and r.ambiguous == 1


def test_parse_tolerates_one_trailing_partial(tmp_path):
    recs = synth_records([1.101, 1.104])
    p = write_log(tmp_path, recs)
    p.write_text(p.read_text() + '{"truncated": ')
    parsed, problems = sd.parse_shadow_log(p)
    assert len(parsed) == len(recs)
    assert problems and problems[0]["kind"] == "TRAILING_PARTIAL_LINE"
    lines = p.read_text().splitlines()
    lines.insert(1, "{corrupt}")
    p.write_text("\n".join(lines))
    _, problems2 = sd.parse_shadow_log(p)
    assert any(q["kind"] == "LOG_CORRUPT" for q in problems2)
