#!/usr/bin/env python3
"""G0 — write hash-locked preregs HYP-093/094/095 + ledger PREREGISTERED entries.

Colin's riders (2026-07-12) are LAW in these specs:
 R1 costs first-class (pessimistic pre-declared borrow/locate/slippage schedules)
 R2 constitutional sizing (Art.1 0.75% worst-case; worst-case BEYOND the stop)
 R3 verdict thresholds pre-declared (DSR at full mined-N; constitutional floor)
Boundary constants (terciles, VIX cut) are frozen HERE from MINING data so the
holdout evaluation contains zero free parameters.
Run: python3 -m research.yield_frontier.preregister_yield [--verify]
"""
import json
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ._lib import REPO, canonical_hash, mined_total
from .holdout_guard import load_nq
from . import m1_equities as m1

PREREG_DIR = REPO / "data/research/preregister"
LEDGER = REPO / "data/agent/hypothesis_ledger.json"

STATS_BLOCK = {
    "primary": "stationary-block-bootstrap one-sided p(mean of DAILY constitutional net series > 0), L=5, 10000 draws, seed 42 (research.modern._lib.block_bootstrap_sharpe_diff_p vs zero arm)",
    "dsr": "deflated_sharpe_ratio(daily Sharpe, n_trials=MINED_N) must be >= 0.95 (sovereign.discovery.gate)",
    "family": "Benjamini-Hochberg across HYP-093/094/095 primary p-values, alpha=0.05, m=3",
    "mined_n": None,   # filled at write time from mined_n.json
    "daily_series": "net constitutional return for EVERY trading day in the holdout window (0.0 on no-event days) — the %/day claim is over calendar of opportunity, not event days only",
}

VERDICT_LADDER = ("CONFIRMED (all: primary p<0.05, BH survivor, DSR>=0.95, "
                  "constitutional floor met, tail condition met) / "
                  "VALID_BUT_BELOW_FLOOR (significance yes, floor no) / "
                  "NOT_ROBUST (tail/sizing premise violated) / "
                  "NOT_SIGNIFICANT / DATA_INSUFFICIENT (coverage/abort conditions)")


def mining_boundaries():
    """Freeze the boundary constants from MINING data (never from holdout)."""
    grouped = m1.load_grouped()
    cands = pd.read_csv(m1.GAPPER_DIR / "candidates.csv")
    dates = sorted(grouped)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    locs, dvols = [], []
    for _, r in cands.iterrows():
        bar = grouped.get(r["date"], {}).get(r["ticker"])
        if bar is None or r["date"] not in nxt:
            continue
        o, h, l, c, v = bar
        pc = r["prev_close_polygon"]
        if None in (o, h, l, c, v) or pc <= 0 or c / pc - 1 < 0.30:
            continue
        if grouped[nxt[r["date"]]].get(r["ticker"]) is None:
            continue
        locs.append((c - l) / (h - l) if h > l else np.nan)
        dvols.append(c * v)
    loc_q1, loc_q2 = np.nanpercentile(locs, [33.3, 66.7])
    dv_q1, dv_q2 = np.nanpercentile(dvols, [33.3, 66.7])
    daily = load_nq("daily").set_index("date")
    daily.index = pd.to_datetime(daily.index)
    aux = load_nq("aux").set_index("date")
    aux.index = pd.to_datetime(aux.index)
    sess = daily[~daily["roll_day"]]
    vix_prior = aux["vix"].shift(1).reindex(sess.index)
    v1, v2 = np.nanpercentile(vix_prior.dropna(), [33.3, 66.7])
    return {"eq1_loc_q1": round(float(loc_q1), 6), "eq1_loc_q2": round(float(loc_q2), 6),
            "eq1_dvol_q1": round(float(dv_q1), 2), "eq1_dvol_q2": round(float(dv_q2), 2),
            "nq_vix_t2": round(float(v2), 4)}


def build_preregs():
    b = mining_boundaries()
    n_mined = mined_total()
    stats = dict(STATS_BLOCK, mined_n=n_mined)
    common_equity_pipeline = {
        "holdout_window": {"start": "2024-07-01", "end": "2025-06-30"},
        "stage1": "Polygon grouped daily adjusted=true; ticker alpha len<=5 not 5-letter-WRU; prev close >=0.75; day high >= 1.20*prev close; day volume >= 500000",
        "stage2": "Alpaca SIP 5Min+1Day adjustment=split; slice = bars ET start in [09:30,10:25]; readable = >=8 bars AND last >=10:15; P=last slice close; qualifying = P>=1.30*prev_close AND P>=2.00 AND slice volume >= 500000",
        "fetch_rule": "G1 fetch runs ONLY after this prereg's hash is locked and its ledger entry is PREREGISTERED (gate-zero enforced in code)",
    }
    rider1_shorts = {
        "borrow_apr_schedule_pessimistic": {"gap_0.3_0.5": 1.00, "gap_0.5_1.0": 2.00,
                                            "gap_1.0_1.5": 4.00, "gap_1.5_plus": 6.00},
        "borrow_charge": "intraday short pays 1 day of schedule (covers locate fees); overnight short pays 2 days",
        "locate_availability": 0.50,
        "locate_model": "deterministic expectation: event weight x0.5; assumed random subset (undetectable selection disclosed)",
        "slippage_per_side": 0.005,
        "rider": "R1 — no verdict valid under default cost assumptions",
    }
    preregs = {}
    preregs["HYP-093"] = {
        "id": "HYP-093", "name": "Parabolic gapper fade (intraday short, +30% stop) — mined board row 1",
        "registered": "2026-07-12", "ticket": "TICK-031",
        "mined_source": "F-EQ2_fade_short thr0.5|stop0.3|close|mna_excl=True (net/day +3.78% MINED, median +5.6%/event, n=651)",
        "pipeline": common_equity_pipeline,
        "signal": "qualifying ticker-day AND gain_1030 >= 0.50 AND catalyst != MERGER_ACQ (Alpaca news [prev day 16:00 ET, day 10:30 ET], posthoc_scan keyword classifier verbatim)",
        "execution": {
            "entry": "short at open of first bar with ET start in [10:30,11:00)",
            "stop": "stop_px = entry*1.30; walk post bars: bar.o>=stop -> fill bar.o (gap-through); bar.h>=stop -> fill stop_px",
            "exit": "close of last bar ET start < 16:00 if not stopped",
            "event_return": "(entry - exit)/entry - 2*0.005 - APR(gain_1030)/365",
        },
        "costs": rider1_shorts,
        "sizing_constitutional": {
            "rider": "R2", "art1_worst_case_per_trade": 0.0075,
            "declared_worst_case": "2x stop distance = 60% adverse (halt gap-through multiplier)",
            "notional_per_event": 0.0125, "locate_weight": 0.50,
            "daily_net": "sum(event_returns)*0.0125*0.50 per day; 0 on no-event days",
        },
        "thresholds": {
            "rider": "R3", **stats,
            "constitutional_floor_pct_day": 0.0005,
            "tail_condition": "event p5 >= -0.60 AND no single event worse than -0.60 (else sizing premise broken -> NOT_ROBUST)",
        },
        "abort": ["stage-1 holdout cache incomplete -> partial-window truthfully or DATA_INSUFFICIENT",
                  "stage-2 usable coverage < 60% of candidates -> DATA_INSUFFICIENT"],
        "verdict_ladder": VERDICT_LADDER,
        "prior_materials": "Mining board is NON-EVIDENTIARY. Registered prior: NOT_SIGNIFICANT (operator's stated guess).",
    }
    preregs["HYP-094"] = {
        "id": "HYP-094", "name": "Overnight short of weak-closing moderate gappers — mined board row 2",
        "registered": "2026-07-12", "ticket": "TICK-031",
        "mined_source": "F-EQ1_overnight g30-50|locT1|volT2|short|next_close (net/day +3.53% MINED, n=237)",
        "pipeline": common_equity_pipeline,
        "signal": {"universe": "stage-1 candidate ticker-days (intraday qualification NOT required — mined family used grouped daily only)",
                   "close_gain": "[0.30, 0.50) vs prev grouped close",
                   "close_location": f"(c-l)/(h-l) <= {b['eq1_loc_q1']} (mining T1 boundary, FROZEN)",
                   "dollar_volume": f"c*v in ({b['eq1_dvol_q1']}, {b['eq1_dvol_q2']}] (mining T2 boundaries, FROZEN)"},
        "execution": {
            "entry": "short at day-D grouped close",
            "exit": "day-D+1 grouped close",
            "event_return": "(c_D - c_D1)/c_D - 2*0.005 - 2*APR(0.4)/365  [gap tier 0.3-0.5 -> APR 1.00]",
            "absent_next_day": "excluded and counted; absent rate > 5% of signals -> DATA_INSUFFICIENT",
        },
        "costs": rider1_shorts,
        "sizing_constitutional": {
            "rider": "R2", "art1_worst_case_per_trade": 0.0075,
            "declared_worst_case": "overnight gapper short has no functioning stop: worst-case = 100% adverse",
            "notional_per_event": 0.0075, "locate_weight": 0.50,
            "daily_net": "sum(event_returns)*0.0075*0.50 per day; 0 on no-event days",
        },
        "thresholds": {
            "rider": "R3", **stats,
            "constitutional_floor_pct_day": 0.0003,
            "tail_condition": "no single event worse than -1.00 (sizing premise) AND event p5 >= -0.50",
        },
        "abort": ["same as HYP-093"],
        "verdict_ladder": VERDICT_LADDER,
        "prior_materials": "Mining board NON-EVIDENTIARY. Registered prior: NOT_SIGNIFICANT.",
    }
    preregs["HYP-095"] = {
        "id": "HYP-095", "name": "NQ high-VIX prior-day-down session long — mined board row 3",
        "registered": "2026-07-12", "ticket": "TICK-031",
        "mined_source": "F-NQ5_vixregime vixT2|prior_dn|long (net/day +0.24% of notional MINED, n=265, 1 neg yr of 7)",
        "holdout_window": {"start": "2024-07-01", "end": "2026-06-09 (end of on-disk NQ data)"},
        "signal": f"non-roll RTH session with prior-day VIX close >= {b['nq_vix_t2']} (mining tercile-2 boundary, FROZEN) AND prior session rth open->close return < -0.002",
        "execution": {
            "entry": "long at session rth_open", "exit": "session rth_close",
            "event_return": "(rth_close/rth_open - 1) - 0.625/rth_open  [NQ mini: 1 tick/side + $2.50 RT]",
            "data": "nq_daily.parquet rows > 2024-06-30 (unfenced ONLY by gauntlet_run after gate-zero)",
        },
        "sizing_constitutional": {
            "rider": "R2", "art1_worst_case_per_trade": 0.0075,
            "declared_worst_case": "intraday index long, no stop in mined config: worst-case = 10% adverse session (2x worst mining-window session)",
            "notional_per_event": 0.075,
            "daily_net": "event_return*0.075 on signal days; 0 otherwise",
        },
        "thresholds": {
            "rider": "R3", **stats,
            "constitutional_floor_pct_day": 0.0002,
            "tail_condition": "no single session worse than -10% open->close (sizing premise)",
        },
        "abort": ["fewer than 40 signal sessions in holdout -> DATA_INSUFFICIENT"],
        "verdict_ladder": VERDICT_LADDER,
        "prior_materials": "Mining board NON-EVIDENTIARY. Registered prior: NOT_SIGNIFICANT.",
    }
    return preregs


def main():
    if "--verify" in sys.argv:
        ok = True
        for hyp in ("HYP-093", "HYP-094", "HYP-095"):
            fp = PREREG_DIR / f"{hyp}_yield_frontier.json"
            doc = json.loads(fp.read_text())
            lock = doc.pop("hash_lock")
            ok &= canonical_hash(doc) == lock
            print(hyp, "lock", "OK" if canonical_hash(doc) == lock else "BROKEN")
        sys.exit(0 if ok else 1)
    preregs = build_preregs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy(LEDGER, str(LEDGER) + f".bak-{stamp}")
    ledger = json.loads(LEDGER.read_text())
    assert not any(e.get("id") in preregs for e in ledger), "HYP-09[345] already in ledger"
    for hyp, doc in preregs.items():
        doc["hash_lock"] = canonical_hash(doc)
        fp = PREREG_DIR / f"{hyp}_yield_frontier.json"
        fp.write_text(json.dumps(doc, indent=2))
        ledger.append({
            "id": hyp, "name": doc["name"], "status": "PREREGISTERED",
            "date_registered": "2026-07-12",
            "prereg_file": str(fp.relative_to(REPO)), "hash_lock": doc["hash_lock"],
            "ticket": "TICK-031", "prior_expectation": "NOT_SIGNIFICANT",
            "methodology_note": f"Yield-frontier gauntlet; riders R1-R3 locked; DSR n_trials={doc['thresholds']['mined_n']}",
        })
        print(f"{hyp} locked {doc['hash_lock'][:8]} -> {fp.name}")
    LEDGER.write_text(json.dumps(ledger, indent=2))
    print(f"ledger: {len(ledger)} entries (backup .bak-{stamp})")


if __name__ == "__main__":
    main()
