"""Parity harness (TICK-022, Phase 1 GATE).

Reproduces the three recorded MC artifacts BEFORE any new engine code is trusted:

  #1  logs/prop_challenge_sim.json        — ICT pool through PropFirmRules.mff
      (recorded run labeled window="combined" but its 102-trade pool matches
      today's window_B — pool drift documented; we pin window B and fingerprint
      n==102). Fully deterministic (random.Random(seed)) → EXACT match required.

  #2  data/agent/ftmo_swing_mc.json       — parametric "FTMO swing" MC.
      NOTE: the original script does NOT model real FTMO (it uses a TRAILING 10%
      DD and a 60-day cap; real FTMO Swing is STATIC 10% from initial with no
      time limit). Parity reproduces the artifact UNDER ITS OWN ASSUMPTIONS —
      the correct FTMO ruleset lives in rulesets.py and will diverge on purpose.
      numpy Generator stream → exact if numpy bit-stream matches; else Wilson-CI.

  #3  data/risk/prop_monte_carlo.json     — carry OOS bootstrap via
      sovereign.risk.monte_carlo_prop.run() with OUT monkeypatched into our
      parity dir (the live path must never be overwritten by research runs).

Run:  python3 research/prop_funnel/parity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.prop_funnel._lib import PARITY_DIR, ROOT, env_record, write_json

RECORDED_ICT = ROOT / "logs" / "prop_challenge_sim.json"
RECORDED_FTMO = ROOT / "data" / "agent" / "ftmo_swing_mc.json"
RECORDED_CARRY = ROOT / "data" / "risk" / "prop_monte_carlo.json"

# Statistical fallback when a recorded artifact predates the current numpy and
# the bit-stream differs: two-sided Wilson-style tolerance on a rate from n
# trials, doubled for slack. Every fallback is recorded as exact=False.
def _rate_tol(p: float, n: int) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 2 * 1.96 * (p * (1 - p) / n) ** 0.5


def _check(name: str, ours, recorded, tol: float = 0.0) -> dict:
    if ours is None or recorded is None:
        ok = ours == recorded
        delta = None
    else:
        delta = float(ours) - float(recorded)
        ok = abs(delta) <= tol if tol > 0 else ours == recorded
    return {"field": name, "ours": ours, "recorded": recorded,
            "delta": delta, "tol": tol, "ok": bool(ok)}


# ── Parity #1: ICT pool → PropFirmRules.mff ────────────────────────────────

def parity_1_ict_mff() -> dict:
    from sovereign.propfirm.challenge_simulator import _load_r_multiples, run_monte_carlo
    from sovereign.propfirm.rules_engine import PropFirmRules

    recorded = json.loads(RECORDED_ICT.read_text())
    r_values, _stats = _load_r_multiples("B")

    # Pool fingerprint: the recorded run used a 102-trade pool. If window_B has
    # drifted, parity is meaningless — fail loud, do not fuzz.
    if len(r_values) != recorded["n_trades_in_sequence"]:
        return {"name": "parity_1_ict_mff", "ok": False, "exact": False,
                "error": (f"pool drift: window_B has {len(r_values)} trades, recorded run used "
                          f"{recorded['n_trades_in_sequence']} — re-pin the pool before trusting parity")}

    def factory():
        r = PropFirmRules.mff(account_size=float(recorded.get("account_size", 100_000)))
        r.risk_per_trade_pct = float(recorded.get("risk_pct", 0.0075))
        return r

    mc = run_monte_carlo(r_values, factory,
                         n_simulations=int(recorded["n_simulations"]), seed=42)

    checks = [
        _check("pass_rate", mc["pass_rate"], recorded["pass_rate"]),
        _check("bust_rate", mc["bust_rate"], recorded["bust_rate"]),
        _check("timeout_rate", mc["timeout_rate"], recorded["timeout_rate"]),
        _check("median_days_to_pass", mc["median_days_to_pass"], recorded["median_days_to_pass"]),
        _check("p10_days_to_pass", mc["p10_days_to_pass"], recorded["p10_days_to_pass"]),
        _check("p90_days_to_pass", mc["p90_days_to_pass"], recorded["p90_days_to_pass"]),
    ]
    ok = all(c["ok"] for c in checks)
    return {"name": "parity_1_ict_mff", "ok": ok, "exact": ok, "checks": checks,
            "note": ("recorded run labeled window='combined' but pool matches today's window_B "
                     "(n==102); combined has since been regenerated to 43 trades — drift documented")}


# ── Parity #2: faithful re-implementation of scripts/ftmo_swing_mc.py ─────
# scripts/ is a forbidden import root, so the trial loop is transcribed
# verbatim (same RNG construction, same draw order) from ftmo_swing_mc.py.
# Constants mirror the ORIGINAL script (trailing DD + 60-day cap) — that is
# the artifact's own (non-FTMO) model, reproduced under its own assumptions.

_FTMO_INITIAL = 100_000.0
_FTMO_TARGET = 0.10
_FTMO_MAX_DD = 0.10          # trailing from equity high — NOT real FTMO
_FTMO_MIN_TDAYS = 4
_FTMO_MAX_DAYS = 60          # NOT real FTMO (no time limit)


def _ftmo_simulate_trial(seed_offset: int, p: dict) -> tuple[str, int]:
    rng = np.random.default_rng(42 + seed_offset)
    equity = _FTMO_INITIAL
    equity_high = _FTMO_INITIAL
    tdays: set[int] = set()
    day = 0
    while day < _FTMO_MAX_DAYS:
        day += 1
        n_today = rng.poisson(p["trades_per_day"])
        for _ in range(n_today):
            tdays.add(day)
            risk = equity * p["risk_per_trade"]
            if rng.random() < p["win_rate"]:
                pnl = risk * p["avg_win_r"]
            else:
                pnl = -risk * p["avg_loss_r"]
            equity += pnl
            equity_high = max(equity_high, equity)
            if (equity_high - equity) / equity_high >= _FTMO_MAX_DD:
                return "BUST", day
        if (equity - _FTMO_INITIAL) / _FTMO_INITIAL >= _FTMO_TARGET and len(tdays) >= _FTMO_MIN_TDAYS:
            return "PASS", day
    return "TIMEOUT", day


def parity_2_ftmo() -> dict:
    recorded = json.loads(RECORDED_FTMO.read_text())
    p = recorded["params"]
    n = int(recorded["n_trials"])

    results = [_ftmo_simulate_trial(i, p) for i in range(n)]
    outcomes = [r[0] for r in results]
    pass_days = [d for o, d in results if o == "PASS"]

    pass_rate = outcomes.count("PASS") / n * 100
    bust_rate = outcomes.count("BUST") / n * 100
    timeout_rate = outcomes.count("TIMEOUT") / n * 100
    median_days = round(float(np.median(pass_days)), 1) if pass_days else None

    rec = recorded["results"]
    tol_pp = _rate_tol(rec["pass_rate"] / 100, n) * 100
    checks = [
        _check("pass_rate_pct", round(pass_rate, 2), rec["pass_rate"], tol=tol_pp),
        _check("bust_rate_pct", round(bust_rate, 2), rec["bust_rate"], tol=tol_pp),
        _check("timeout_rate_pct", round(timeout_rate, 2), rec["timeout_rate"], tol=tol_pp),
        _check("median_days_to_pass", median_days, rec["median_days_to_pass"], tol=3.0),
    ]
    exact = all(c["delta"] in (0, 0.0, None) for c in checks)
    ok = all(c["ok"] for c in checks)
    return {"name": "parity_2_ftmo", "ok": ok, "exact": exact, "checks": checks,
            "note": ("reproduces the RECORDED artifact under its own assumptions "
                     "(trailing 10% DD + 60-day cap) — the original script does NOT model real "
                     "FTMO Swing (static DD, no time limit); do not quote its pass rate as FTMO")}


# ── Parity #3: carry OOS bootstrap via monte_carlo_prop, OUT redirected ────

def parity_3_carry_bootstrap() -> dict:
    import sovereign.risk.monte_carlo_prop as mcp

    recorded = json.loads(RECORDED_CARRY.read_text())

    # Clock pinning: run() derives trades/year from logs/forex_backtest_results.json,
    # which is gitignored and overwritten by every backtest run — it HAS drifted since
    # the recorded artifact (same drift class as parity #1's pool). The recorded payload
    # carries its own clock; parity means reproducing under the recorded clock.
    pool_pnls, pool_per_pair, tpy_now = mcp.load_pool()
    tpy_recorded = float(recorded["portfolio_trades_per_year"])
    original_load_pool = mcp.load_pool
    mcp.load_pool = lambda: (pool_pnls, pool_per_pair, tpy_recorded)

    original_out = mcp.OUT
    mcp.OUT = PARITY_DIR / "prop_monte_carlo_reproduced.json"   # write-safety: never the live path
    try:
        payload = mcp.run(
            account=float(recorded["account"]),
            floor_pct=float(recorded["floor_pct"]),
            target_pct=float(recorded["target_pct"]),
            n_sims=int(recorded["n_sims"]),
            seed=7,
        )
    finally:
        mcp.OUT = original_out
        mcp.load_pool = original_load_pool

    n = int(recorded["n_sims"])
    checks = [_check("pool_size", payload["pool_size"], recorded["pool_size"])]
    for h, rec_h in recorded["horizons"].items():
        ours_h = payload["horizons"].get(h, {})
        tol = _rate_tol(rec_h["p_pass"], n)
        checks.append(_check(f"h{h}.max_trades_in_window",
                             ours_h.get("max_trades_in_window"), rec_h["max_trades_in_window"]))
        checks.append(_check(f"h{h}.p_pass", ours_h.get("p_pass"), rec_h["p_pass"], tol=tol))
        checks.append(_check(f"h{h}.p_fail", ours_h.get("p_fail"), rec_h["p_fail"], tol=tol))
    exact = all((c["delta"] in (0, 0.0, None)) for c in checks)
    ok = all(c["ok"] for c in checks)
    return {"name": "parity_3_carry_bootstrap", "ok": ok, "exact": exact, "checks": checks,
            "note": (f"OUT monkeypatched into parity dir; live artifact untouched. Clock pinned to the "
                     f"recorded trades/year {tpy_recorded} — logs/forex_backtest_results.json has drifted "
                     f"(now implies {tpy_now:.1f}/yr), same drift class as parity #1's pool")}


def run_all_parity() -> dict:
    report = {
        "ticket": "TICK-022",
        "env": env_record(),
        "results": [parity_1_ict_mff(), parity_2_ftmo(), parity_3_carry_bootstrap()],
    }
    report["all_ok"] = all(r["ok"] for r in report["results"])
    report["all_exact"] = all(r.get("exact") for r in report["results"])
    write_json(PARITY_DIR / "parity_report.json", report)
    return report


if __name__ == "__main__":
    rep = run_all_parity()
    for r in rep["results"]:
        status = "EXACT" if r.get("exact") else ("OK(tol)" if r["ok"] else "FAIL")
        print(f"{r['name']:32s} {status}")
        if not r["ok"]:
            for c in r.get("checks", []):
                if not c["ok"]:
                    print(f"    {c['field']}: ours={c['ours']} recorded={c['recorded']} tol={c['tol']}")
            if "error" in r:
                print(f"    {r['error']}")
    print(f"\nALL {'GREEN' if rep['all_ok'] else 'RED'} — report: {PARITY_DIR / 'parity_report.json'}")
    sys.exit(0 if rep["all_ok"] else 1)
