"""Drift tripwire — ALERT ONLY. Never modifies a threshold.

WHAT THIS IS FOR
----------------
The HYP-107 filter's thresholds are frozen at the values that were preregistered
before its holdout was touched (`execution/config.py`, sha256-locked). If the live
win rate collapses relative to the sealed baseline, that is worth knowing.

WHAT THIS DELIBERATELY IS NOT
-----------------------------
It does not re-fit, re-tune, or Bayesian-update anything. HYP-090 tested adaptive
parameter selection across 17,325 cells:

    A0 static (do nothing)              Sharpe 0.9478
    A2_W365 (best adaptive arm)         Sharpe 0.4343
    A3 random-selection placebo, p95    Sharpe 0.9115

Every adaptive arm lost to *random selection* and to doing nothing. Its own report:
"beating A0 while not beating A3 is the in-sample-inflation signature, not an edge."
`research/yield_frontier/OPTIMIZATION_PROGRAM.md:12` records the pattern as BANNED.

So: this module observes and reports. Changing a frozen threshold requires a new
preregistration, not a running average.

HONEST POWER — READ THIS BEFORE TRUSTING AN ALL-CLEAR
------------------------------------------------------
A "2-sigma alert" sounds decisive and is not, at the sample sizes involved. Against
the sealed 70% baseline:

    n=20   alert fires only below 49.5% win rate
    n=50   below 57.0%
    n=100  below 60.8%
    n=200  below 63.5%

Two different questions get conflated here, so both are always reported:

    an OBSERVED 60% first trips the alert at        n = 84
    a TRUE 60% rate is RELIABLY caught (80% power) at  n ~ 177

The first ignores sampling noise — at n=84 an observed 60% is borderline and would
be missed about half the time. The second is what "reliably detect" actually means.
Quoting only the first overstates the instrument, and an earlier draft of this
module did exactly that.

At n=20 this catches COLLAPSE, NOT DRIFT. Every report states its own detectable
effect size, so silence is never mistaken for evidence of health.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILL_DIR = ROOT / "data" / "execution"
UTC = timezone.utc

#: Sealed HYP-107 holdout baseline (research/gapper/HYP-105-106-RETRACTION-and-honest-107.md).
#: These describe the preregistered result. They are reference values, not tunables.
BASELINE = {"win_rate": 0.70, "median_gross": 0.054, "n": 57}

SIGMA_THRESHOLD = 2.0


@dataclass
class DriftReport:
    n: int
    live_win_rate: float | None
    baseline_win_rate: float
    sigma: float | None
    z_score: float | None
    alert: bool
    detectable_below: float | None
    min_n_for_10pt_drop: int
    n_for_80pct_power: int
    note: str

    def to_json(self) -> dict:
        return asdict(self)


def sigma_at(n: int, p0: float = BASELINE["win_rate"]) -> float | None:
    """Standard error of a win rate estimated from n binary outcomes."""
    if n <= 0:
        return None
    return math.sqrt(p0 * (1 - p0) / n)


def detectable_below(n: int, p0: float = BASELINE["win_rate"]) -> float | None:
    """The win rate below which a 2-sigma alert would fire at this n."""
    s = sigma_at(n, p0)
    return None if s is None else p0 - SIGMA_THRESHOLD * s


def min_n_for_drop(drop: float = 0.10, p0: float = BASELINE["win_rate"]) -> int:
    """Smallest n at which an OBSERVED `drop`-sized rate would trip the alert.

    NOTE — this is the weaker of two questions, and they are easy to conflate:

      (a) "if the observed rate is exactly 60%, when does the alert fire?"  -> n=84
      (b) "if the TRUE rate is 60%, when will we reliably catch it?"        -> n≈177

    (a) is a threshold-crossing question and ignores sampling noise: at n=84 an
    observed 60% is borderline and would be missed roughly half the time. (b) is
    the statistical-power question and is what "reliably detect" actually means.
    Both are reported, because quoting only (a) overstates the instrument.
    """
    n = 1
    while n < 100_000:
        s = sigma_at(n, p0)
        if s is not None and SIGMA_THRESHOLD * s <= drop:
            return n
        n += 1
    return -1


def n_for_power(p1: float, p0: float = BASELINE["win_rate"],
                power: float = 0.80) -> int:
    """Sample size to detect a TRUE rate of `p1` with the given power.

    Standard one-sided two-proportion approximation:
        n = (z_alpha*sqrt(p0*q0) + z_beta*sqrt(p1*q1))^2 / (p0-p1)^2

    z_beta for 80% power is 0.8416. z_alpha is SIGMA_THRESHOLD, since that is the
    bar this tool actually applies.
    """
    if p1 >= p0:
        return -1
    z_beta = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)
    num = (SIGMA_THRESHOLD * math.sqrt(p0 * (1 - p0))
           + z_beta * math.sqrt(p1 * (1 - p1))) ** 2
    return math.ceil(num / (p0 - p1) ** 2)


def load_outcomes(fill_dir: Path | None = None,
                  hypothesis: str = "HYP-107") -> list[float]:
    """Net returns for filled signals of one hypothesis, oldest first."""
    fill_dir = fill_dir or FILL_DIR
    p = fill_dir / "fill_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("hypothesis") != hypothesis:
            continue
        if str(r.get("signal_type", "")).startswith("SKIP_"):
            continue
        if r.get("net_return") is None:
            continue
        out.append(float(r["net_return"]))
    return out


def assess(returns: list[float]) -> DriftReport:
    """Compare live win rate to the sealed baseline. Reports; never acts."""
    n = len(returns)
    p0 = BASELINE["win_rate"]
    min_n = min_n_for_drop(0.10, p0)
    power_n = n_for_power(p0 - 0.10, p0, 0.80)

    if n == 0:
        return DriftReport(
            n=0, live_win_rate=None, baseline_win_rate=p0, sigma=None, z_score=None,
            alert=False, detectable_below=None, min_n_for_10pt_drop=min_n,
            n_for_80pct_power=power_n,
            note="No filled outcomes yet. Absence of an alert is not evidence of health.")

    wins = sum(1 for r in returns if r > 0)
    live = wins / n
    s = sigma_at(n, p0)
    z = (live - p0) / s if s else None
    alert = bool(z is not None and z <= -SIGMA_THRESHOLD)
    floor = detectable_below(n, p0)

    note = (f"At n={n} this test only fires below {floor:.1%}. "
            f"An OBSERVED 10-point drop trips it at n>={min_n}; RELIABLY detecting a "
            f"TRUE 60% rate (80% power) needs n~{power_n}. "
            f"{'ALERT' if alert else 'No alert'} — "
            f"{'divergence exceeds 2 sigma' if alert else 'within 2 sigma, which at this n is a weak statement'}.")

    return DriftReport(n=n, live_win_rate=round(live, 4), baseline_win_rate=p0,
                       sigma=round(s, 4) if s else None,
                       z_score=round(z, 3) if z is not None else None,
                       alert=alert,
                       detectable_below=round(floor, 4) if floor else None,
                       min_n_for_10pt_drop=min_n, n_for_80pct_power=power_n,
                       note=note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drift tripwire (alert only, never re-tunes)")
    ap.add_argument("--fills", default=None)
    ap.add_argument("--hypothesis", default="HYP-107")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rets = load_outcomes(Path(args.fills) if args.fills else None, args.hypothesis)
    rep = assess(rets)

    if args.json:
        print(json.dumps(rep.to_json(), indent=2))
        return 1 if rep.alert else 0

    src = Path(args.fills) if args.fills else FILL_DIR
    print(f"DRIFT — {args.hypothesis}")
    print(f"  source            : {src / 'fill_log.jsonl'}")
    if args.fills:
        print("  ⚠ non-default source. A backfill or replay log is NOT live forward "
              "evidence; drift against a sealed baseline only means something when "
              "the fills are live.")
    print(f"  baseline win rate : {rep.baseline_win_rate:.1%} (sealed holdout, n={BASELINE['n']})")
    if rep.n:
        print(f"  live win rate     : {rep.live_win_rate:.1%}  (n={rep.n})")
        print(f"  z-score           : {rep.z_score}")
    else:
        print(f"  live win rate     : —  (n=0)")
    print(f"  alert             : {'YES' if rep.alert else 'no'}")
    print(f"\n  POWER: {rep.note}")
    print("\n  This tool never modifies a frozen threshold (HYP-090 tombstone).")
    return 1 if rep.alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
