"""Phase 2 — per-cluster SAR matrix (spec §Phase-2).

For each kept event x each instrument in the event's cluster: pull yfinance OHLCV and
compute the SAR windows via _lib.sar_windows (estimation window T-252..T-10, mean-adjusted
mu/sigma; SAR_t=(r_t-mu)/sigma; csar_72h = SAR(T0)+SAR(T1)+SAR(T2); pre_csar_48h =
SAR(T-2)+SAR(T-1)). NO SILENT MOCKING (spec §8): missing/short data -> data_ok:false +
gap_reason; the row is emitted and skipped downstream, never backfilled.

Output: data/cluster_sar_matrix.jsonl
Run:    python3 research/political_alpha_v2/compute_cluster_returns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402


def main() -> int:
    events = _lib.read_jsonl(_lib.DATA_DIR / "clustered_events.jsonl")
    if not events:
        print("STOP: clustered_events.jsonl empty — run Phase 1 first.")
        return 2
    print(f"Phase 2 — SAR matrix for {len(events)} events over universe {_lib.UNIVERSE}")

    # fetch every universe instrument once (only those actually used are needed, but the
    # universe is small — fetch all so the coverage report is complete).
    used = sorted({i for e in events for i in e["instruments"]})
    px = {}
    for t in used:
        df = _lib.fetch_daily(t)
        px[t] = df
        print(f"  [yf] {t:10s}: {len(df)} bars"
              + ("" if not df.empty else "  !! EMPTY — rows for this instrument will be data_ok:false"))

    rows = []
    for e in events:
        for inst in e["instruments"]:
            w = _lib.sar_windows(e["timestamp_utc"], inst, px.get(inst))
            rows.append({
                "event_id": e["event_id"], "cluster": e["cluster"],
                "instrument": inst,
                "csar_72h": w["csar_72h"], "pre_csar_48h": w["pre_csar_48h"],
                "big_move": w["big_move"], "direction": w["direction"],
                "data_ok": w["data_ok"], "gap_reason": w["gap_reason"],
                "n_est_days": w["n_est_days"], "t0": w["t0"],
            })
    _lib.write_jsonl(_lib.DATA_DIR / "cluster_sar_matrix.jsonl", rows)

    ok = [r for r in rows if r["data_ok"]]
    gaps = pd.Series([r["gap_reason"] for r in rows if r["gap_reason"]]).value_counts().to_dict()
    print(f"\n  event x instrument rows: {len(rows)}   data_ok: {len(ok)} "
          f"({len(ok)/len(rows)*100:.1f}%)   gaps: {gaps if gaps else 'none'}")
    if ok:
        s = pd.Series([r["csar_72h"] for r in ok])
        print(f"  csar_72h (all ok): mean {s.mean():+.3f}  median {s.median():+.3f}  "
              f"|>2sigma| big_move: {sum(1 for r in ok if r['big_move'])}")
        print("\n  per-cluster data_ok CSAR mean:")
        df = pd.DataFrame(ok)
        for cl, g in df.groupby("cluster"):
            print(f"    {cl:20s}: n={len(g):3d}  mean CSAR {g['csar_72h'].mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
