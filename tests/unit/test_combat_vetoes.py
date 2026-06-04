"""Combat-veto module tests.

Two jobs: (1) confirm the condition LOGIC is correct (the module faithfully implements the forensic
conditions), and (2) LOCK IN the survivorship finding — assert that blanket-skipping these conditions
is net-negative on the real forensic data, so the gate can never be silently re-enabled as a "+273R"
win. See COMBAT_VETOES_FINDING.md.
"""
import copy
import json
from pathlib import Path

from sovereign.forex import combat_vetoes as cv

ROOT = Path(__file__).resolve().parents[2]


def _enabled_cfg():
    c = copy.deepcopy(cv.load_config())
    c["enabled"] = True
    for r in c["rules"].values():
        r["enabled"] = True
    return c


# ── (1) condition logic ──────────────────────────────────────────────────────
def test_c001_macro_against_fires():
    cfg = _enabled_cfg()
    hits = cv.evaluate(real_rate_diff=-0.63, momentum_63d=0.05, atr_14d_pct=0.02, direction=1, cfg=cfg)
    assert any(h.rule_id == "C-001" for h in hits)           # long vs negative real rate diff


def test_c001_aligned_passes():
    cfg = _enabled_cfg()
    hits = cv.evaluate(real_rate_diff=+2.0, momentum_63d=0.05, atr_14d_pct=0.02, direction=1, cfg=cfg)
    assert not any(h.rule_id == "C-001" for h in hits)


def test_c005_weak_rate_fires():
    cfg = _enabled_cfg()
    hits = cv.evaluate(real_rate_diff=0.3, momentum_63d=0.05, atr_14d_pct=0.02, direction=1, cfg=cfg)
    assert any(h.rule_id == "C-005" for h in hits)


def test_c006_low_vol_fires():
    cfg = _enabled_cfg()
    hits = cv.evaluate(real_rate_diff=2.0, momentum_63d=0.05, atr_14d_pct=0.004, direction=1, cfg=cfg)
    assert any(h.rule_id == "C-006" for h in hits)


def test_c003_counter_momentum_fires():
    cfg = _enabled_cfg()
    hits = cv.evaluate(real_rate_diff=2.0, momentum_63d=-0.05, atr_14d_pct=0.02, direction=1, cfg=cfg)
    assert any(h.rule_id == "C-003" for h in hits)


def test_disabled_by_default():
    """Production config MUST keep the gate off (it is not a validated edge)."""
    assert cv.load_config()["enabled"] is False
    # With the shipped config, evaluate returns nothing regardless of inputs.
    assert cv.evaluate(-0.63, -0.05, 0.004, 1) == []


# ── (2) lock in the survivorship finding ─────────────────────────────────────
def test_blanket_vetoes_are_net_negative_on_real_data():
    """If anyone re-enables this thinking it recovers +273R: it does NOT. Net edge impact is negative.
    Recovered losses < forgone winners on the forensic set — proven, not asserted."""
    cfg = _enabled_cfg()
    trades = json.loads((ROOT / "data" / "research" / "trade_forensics.json").read_text())
    recovered = forgone = 0.0
    for t in trades:
        if cv.evaluate(t.get("real_rate_diff"), t.get("momentum_63d"),
                       t.get("atr_14d_pct"), int(t["direction"]), cfg):
            r = t["outcome_r"]
            if t.get("outcome") == "LOSS":
                recovered += -r          # avoided loss (positive)
            else:
                forgone += r             # forgone win (positive)
    assert forgone > recovered, (
        f"expected net-negative gate (forgone {forgone:.1f}R > recovered {recovered:.1f}R)")
