#!/usr/bin/env python3
"""
Nightly review — the learning agent's end-of-session synthesis (<60s).

1. Session summary (trades / learning vs strict / W-L / gross+net $ / avg R).
2. Pattern extraction: rule-based comparison of entry reasoning vs exit attribution
   (e.g. "3/4 LOW cvd_quality -> losses"). Not ML — structured comparison.
3. MES/MNQ correlation note (same-bias divergence).
4. Appends session_learnings + recommended_posture_tomorrow to oracle_mornings.jsonl so
   tomorrow's morning note carries today's lessons. Reps compound.

Usage:  python3 scripts/futures_nightly_review.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sovereign.futures import review_common as rc

TRADE_LOG = ROOT / "data" / "futures" / "trade_log.jsonl"
ORACLE_LOG = ROOT / "data" / "futures" / "oracle_mornings.jsonl"
OUT = ROOT / "data" / "futures" / "nightly_reviews.jsonl"
MIN_BUCKET = 3          # need at least this many to call a pattern
DIVERGE_PP = 0.20       # bucket win-rate this far from overall = notable


def _patterns(closed: list[dict], overall_wr) -> list[str]:
    out = []
    if overall_wr is None:
        return out
    cuts = {
        "cvd_quality": lambda r: rc.reasoning_field(r, "cvd_quality"),
        "setup_type": lambda r: r.get("setup_type") or rc.reasoning_field(r, "setup_type"),
        "confidence": lambda r: rc.reasoning_field(r, "confidence"),
    }
    for cut_name, fn in cuts.items():
        for bucket, recs in rc.group_by(closed, fn).items():
            wr = rc.winrate(recs)
            if wr is None or len(recs) < MIN_BUCKET:
                continue
            w = sum(1 for r in recs if rc.is_win(r))
            if abs(wr - overall_wr) >= DIVERGE_PP:
                sign = "+" if wr > overall_wr else "-"
                out.append(f"{cut_name}={bucket}: {w}/{len(recs)} ({wr:.0%}) vs overall {overall_wr:.0%} "
                           f"[{sign}{abs(wr-overall_wr)*100:.0f}pp]")
    return out


def _correlation_note(closed: list[dict]) -> dict | None:
    mes = [r for r in closed if r.get("instrument") == "MES"]
    mnq = [r for r in closed if r.get("instrument") == "MNQ"]
    if not mes or not mnq:
        return None
    def _r(recs):
        rs = [r["r_realized"] for r in recs if isinstance(r.get("r_realized"), (int, float))]
        return round(sum(rs) / len(rs), 2) if rs else None
    mes_dir = mes[0].get("bias_direction"); mnq_dir = mnq[0].get("bias_direction")
    mes_r, mnq_r = _r(mes), _r(mnq)
    note = (f"MES {mes_dir} avg {mes_r}R vs MNQ {mnq_dir} avg {mnq_r}R. "
            + ("Same macro bias — " if mes_dir == mnq_dir else "Different bias — ")
            + ("divergent outcomes; investigate ORB-level / spread differences."
               if mes_r is not None and mnq_r is not None and abs(mes_r - mnq_r) > 0.5
               else "outcomes broadly aligned."))
    return {"mes_direction": mes_dir, "mnq_direction": mnq_dir, "mes_avg_r": mes_r,
            "mnq_avg_r": mnq_r, "correlation_note": note}


_DIRECTIONAL = {"LONG", "SHORT"}


def _killzone_agreement(day: str) -> dict | None:
    """Did each killzone Sonnet read agree with that day's daily Opus call (per instrument)?

    Reads oracle_mornings.jsonl for `day`. agree=True/False only when BOTH biases are directional
    (LONG/SHORT); NEUTRAL/NO_PREDICTION -> agree=None (no directional read to compare). Over ~2
    weeks this `agreements`/`disagreements` tally is the signal-vs-noise dataset for Option 2.
    """
    rows = [json.loads(l) for l in (ORACLE_LOG.read_text().splitlines() if ORACLE_LOG.exists() else [])
            if l.strip() and json.loads(l).get("date") == day]
    daily = {r["instrument"]: r.get("bias") for r in rows if r.get("synthesis_type") == "daily_opus"}
    kz = [r for r in rows if r.get("synthesis_type") == "killzone_sonnet"]
    if not kz:
        return None
    comparisons, agree_n, disagree_n = [], 0, 0
    for r in kz:
        inst, d_bias = r.get("instrument"), daily.get(r.get("instrument"))
        k_bias = r.get("bias")
        if d_bias in _DIRECTIONAL and k_bias in _DIRECTIONAL:
            ok = (d_bias == k_bias)
            agree_n += ok
            disagree_n += (not ok)
        else:
            ok = None
        comparisons.append({"killzone": r.get("killzone"), "instrument": inst,
                            "killzone_bias": k_bias, "daily_bias": d_bias, "agree": ok})
    return {"comparisons": comparisons, "agreements": agree_n, "disagreements": disagree_n,
            "daily_call_present": bool(daily)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(rc.ET).strftime("%Y-%m-%d"))
    ap.add_argument("--include-simulated", action="store_true",
                    help="Include data_quality=SIMULATED entries (off by default — they never train Oracle).")
    args = ap.parse_args()
    day = args.date

    trades = [t for t in rc.load_trades(TRADE_LOG, include_simulated=args.include_simulated)
              if rc.session_date(t) == day and t.get("size_contracts")]
    closed = [t for t in trades if t.get("exit") is not None]
    learning = sum(1 for t in trades if rc.reasoning_field(t, "learning_mode"))
    wins = sum(1 for t in closed if rc.is_win(t))
    losses = sum(1 for t in closed if rc.is_win(t) is False)
    pnls = [rc.trade_pnl_usd(t) for t in closed]
    pnls = [p for p in pnls if p is not None]
    gross = round(sum(max(p, 0) for p in pnls) + sum(min(p, 0) for p in pnls), 2)
    rs = [t["r_realized"] for t in closed if isinstance(t.get("r_realized"), (int, float))]
    avg_r = round(sum(rs) / len(rs), 3) if rs else None
    overall_wr = rc.winrate(closed)

    print(f"\n{'='*60}\n  FUTURES NIGHTLY REVIEW — {day}\n{'='*60}")
    print(f"  TRADES: {len(trades)} placed / {learning} learning-mode / {len(closed)} closed")
    print(f"  WIN/LOSS: {wins}W {losses}L  (win rate {overall_wr if overall_wr is not None else 'n/a'})")
    print(f"  NET P&L (costed): ${gross:+.2f}   AVG R: {avg_r if avg_r is not None else 'n/a'}")

    patterns = _patterns(closed, overall_wr)
    print("\n  PATTERNS:" + ("" if patterns else " none (need closed trades / divergence)"))
    for p in patterns:
        print(f"    - {p}")

    corr = _correlation_note(closed)
    if corr:
        print(f"\n  CORRELATION: {corr['correlation_note']}")

    kz_agree = _killzone_agreement(day)
    if kz_agree:
        print(f"\n  KILLZONE vs DAILY: {kz_agree['agreements']} agree / {kz_agree['disagreements']} disagree"
              + ("" if kz_agree["daily_call_present"] else "  (no daily Opus call logged today)"))
        for c in kz_agree["comparisons"]:
            mark = "=" if c["agree"] else ("x" if c["agree"] is False else "·")
            print(f"    {mark} {c['killzone']} {c['instrument']}: sonnet={c['killzone_bias']} daily={c['daily_bias']}")

    # ── compound into tomorrow's oracle morning note ──
    learnings = patterns or ([f"{len(closed)} closed trades, {overall_wr:.0%} win rate"] if overall_wr is not None
                             else ["No closed trades to learn from yet."])
    posture = ("Favor the setups/conditions that outperformed today; tighten where divergence was negative."
               if patterns else "Insufficient sample — keep taking reps, no posture change.")
    block = {
        "date": day, "type": "session_learnings",
        "trades": len(trades), "closed": len(closed), "win_rate": overall_wr,
        "net_pnl_usd": gross, "avg_r": avg_r,
        "session_learnings": learnings,
        "correlation": corr,
        "killzone_agreement": kz_agree,
        "recommended_posture_tomorrow": posture,
        "generated_at": datetime.now(rc.ET).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(block) + "\n")
    with ORACLE_LOG.open("a") as f:
        f.write(json.dumps(block) + "\n")
    print(f"\n  -> session_learnings appended to oracle_mornings.jsonl (surfaces in tomorrow's note)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
