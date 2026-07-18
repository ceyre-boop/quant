"""Build a HYP-107-passing event list from the archived gapper universe."""
import csv, json, math, random, sys
from datetime import date
from execution import alpaca
from execution.config import frozen
from execution.scan import bar_at, symbol_shape_ok

alpaca.load_env()
C = frozen("hyp107")
rows = list(csv.DictReader(open("data/research/gapper/per_candidate_enriched.csv")))
random.seed(20260718)
random.shuffle(rows)

out, checked = [], 0
for r in rows:
    if len(out) >= 60 or checked >= 320:
        break
    sym = r["ticker"].strip().upper()
    if not symbol_shape_ok(sym):
        continue
    d = date.fromisoformat(r["date"])
    checked += 1
    try:
        b = alpaca.minute_bars(sym, d)
        pc = float(r["prev_close"])
    except Exception:
        continue
    if not b or pc <= 0:
        continue
    b0930, b0931, b1030 = bar_at(b,9,30), bar_at(b,9,31), bar_at(b,10,30)
    if not (b0930 and b0931 and b1030):
        continue
    og = b0930["o"]/pc - 1
    lv = math.log10(b0930["v"] + 1)
    if og < C["gap_floor"] or og > C["og_max"] or lv > C["logvol_max"]:
        continue
    out.append({"ticker": sym, "date": str(d), "side": "LONG",
                "hypothesis": "HYP-107", "overnight_gap": round(og,4),
                "log_vol": round(lv,4)})
    print(f"  {len(out):3d}. {sym} {d} gap={og:.3f} logvol={lv:.3f}", flush=True)

json.dump(out, open("/tmp/hyp107_events.json","w"), indent=1)
print(f"\nchecked {checked} archived candidates -> {len(out)} HYP-107 filter passes")
