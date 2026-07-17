#!/usr/bin/env python3
"""Prop-challenge MC sizing sweep on the CONFIRMED >=100% gapper edge alone.

(HYP-101 threshold relaxation found no preregistrable candidate — 2026-07-16
dirty scan: all relaxed thresholds fail the quality bar; 11:00 entry worse.)

Same engine as monte_carlo_prop_hyp100.py: 100k bootstrap paths per sizing,
$100k, +8% pass / -8% bust, 90 trading days, empirical day-P&L draws from the
forward sim (sim_annual_HYP093_forward.csv, 2% notional base, 25% stop).
Sweep: 1, 1.5, 2, 2.5, 3, 3.5, 4 % notional.

CAVEAT: assumes locate available on EVERY signal (see HYP-100 outputs).
"""
import csv
import json
import random
from pathlib import Path

SIM = Path(__file__).with_name("sim_annual_HYP093_forward.csv")
OUT = Path(__file__).with_name("monte_carlo_prop_HYP100_sweep.json")

N_PATHS = 100_000
START, PASS, BUST, WINDOW = 100_000.0, 0.08, -0.08, 90
SIM_NOTIONAL = 0.02
TRADING_DAYS_YEAR = 252
SEED = 42


def main():
    rng = random.Random(SEED)
    day_rets = [float(r["day_return"]) for r in csv.DictReader(open(SIM))]
    p_event = len(day_rets) / TRADING_DAYS_YEAR

    def run(notional):
        scale = notional / SIM_NOTIONAL
        n_pass = n_bust = 0
        for _ in range(N_PATHS):
            eq = START
            for _d in range(WINDOW):
                if rng.random() < p_event:
                    eq *= (1 + rng.choice(day_rets) * scale)
                if eq >= START * (1 + PASS):
                    n_pass += 1
                    break
                if eq <= START * (1 + BUST):
                    n_bust += 1
                    break
        return {"notional_pct": notional * 100,
                "p_pass": round(n_pass / N_PATHS, 4),
                "p_bust": round(n_bust / N_PATHS, 4),
                "p_time": round(1 - (n_pass + n_bust) / N_PATHS, 4),
                "meets_95_5": (n_pass / N_PATHS >= 0.95 and
                               n_bust / N_PATHS <= 0.05)}

    sweep = [run(s / 100) for s in (1, 1.5, 2, 2.5, 3, 3.5, 4)]
    winners = [s for s in sweep if s["meets_95_5"]]
    out = {
        "edge": ">=100% confirmed HYP-093 only (no relaxed threshold survived Step 1)",
        "n_paths": N_PATHS, "seed": SEED, "window_days": WINDOW,
        "account": START, "pass_threshold": PASS, "bust_threshold": BUST,
        "sweep": sweep,
        "lowest_sizing_meeting_95_5": winners[0] if winners else None,
        "CAVEAT": ("Assumes locate available on EVERY signal. Only 16/234 "
                   "events (6.8%) had a measurable options market; IB locate "
                   "series (TICK-037) just started. Real P(PASS) is lower."),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
