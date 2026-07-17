#!/usr/bin/env python3
"""Prop-challenge Monte Carlo on the HYP-093 forward-sim daily P&L distribution.

100k bootstrap paths per window. Day draw: with p = (event days / calendar
trading days) draw a random event-day return from the forward sim (scaled to
the tested sizing), else 0 (no-signal day). $100k account, +8% pass / -8% bust,
windows 30/60/90 trading days. Sensitivity: P(PASS 90d) at 1/2/3/4% notional.

CAVEAT (prominent, travels with every output): probabilities assume locate is
available on EVERY signal. Only 16/234 events (6.8%) had a measurable options
market; locate availability on the rest is unproven. Real P(PASS) is lower.
"""
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM = Path(__file__).with_name("sim_annual_HYP093_forward.csv")
OUT = Path(__file__).with_name("monte_carlo_prop_HYP100.json")
DASH = REPO / "data/gapper_mc.json"          # dashboard copy (synced)

N_PATHS = 100_000
START, PASS, BUST = 100_000.0, 0.08, -0.08
SIM_NOTIONAL = 0.02
TRADING_DAYS_YEAR = 252
SEED = 42


def main():
    rng = random.Random(SEED)
    day_rets = [float(r["day_return"]) for r in csv.DictReader(open(SIM))]
    n_event_days = len(day_rets)                     # 147
    p_event = n_event_days / TRADING_DAYS_YEAR       # chance a day has signals

    def run(window, notional):
        scale = notional / SIM_NOTIONAL
        p_pass = p_bust = 0
        for _ in range(N_PATHS):
            eq = START
            for _d in range(window):
                if rng.random() < p_event:
                    eq *= (1 + rng.choice(day_rets) * scale)
                if eq >= START * (1 + PASS):
                    p_pass += 1
                    break
                if eq <= START * (1 + BUST):
                    p_bust += 1
                    break
        return {"days": window, "notional": notional,
                "p_pass": round(p_pass / N_PATHS, 4),
                "p_bust": round(p_bust / N_PATHS, 4),
                "p_time": round(1 - (p_pass + p_bust) / N_PATHS, 4)}

    windows = [run(w, SIM_NOTIONAL) for w in (30, 60, 90)]
    sensitivity = [run(90, s) for s in (0.01, 0.02, 0.03, 0.04)]

    out = {
        "source_sim": "sim_annual_HYP093_forward.csv (234 events, 147 event days)",
        "n_paths": N_PATHS, "seed": SEED,
        "account": START, "pass_threshold": PASS, "bust_threshold": BUST,
        "p_event_day": round(p_event, 4),
        "windows": windows,
        "sensitivity_p_pass_90d_by_notional": sensitivity,
        "CAVEAT": ("Assumes locate available on EVERY signal. Only 16/234 "
                   "events (6.8%) had a measurable options market; locate on "
                   "the rest is unproven. Real P(PASS) is materially lower."),
    }
    OUT.write_text(json.dumps(out, indent=1))
    DASH.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
