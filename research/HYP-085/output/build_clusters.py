"""Phase 1 — language clustering + event classification (spec §Phase-1).

Loads the V1 event catalog (168 unique events), classifies each into <=1 cluster via the
LOCKED config/cluster_rules.json (priority-ordered keyword rules on statement_text),
applies a 5-trading-day minimum-separation de-dup per cluster, and locks the Bonferroni
denominator = (clusters with >=1 event) x 9.

De-dup note: "5-trading-day separation per cluster x instrument" — since every instrument
in a cluster shares that cluster's event set, de-dup is applied once per cluster at the
event level (identical kept-set for each of the cluster's instruments). Trading-day
separation is measured with np.busday_count (weekends excluded; holidays not — a
price-data-free proxy, since no return data exists at Phase 1). Every dropped event logged.

Outputs:
  data/clustered_events.jsonl   — one row per kept event
  data/dropped_events.jsonl     — de-dup drops + unclustered, with reasons
  data/bonferroni_lock.json     — the locked denominator + cluster counts

Run: python3 research/political_alpha_v2/build_clusters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lib  # noqa: E402

UNDERPOWERED_MIN = 5        # clusters with < 5 events flagged underpowered (spec §Phase-1)
DEDUP_TRADING_DAYS = 5      # minimum separation per cluster (spec §Phase-1 / V1)
TEXT_TRUNC = 400           # statement_text stored truncated for readability


def _busdays(a: str, b: str) -> int:
    """Business days between two UTC-date strings (weekends excluded)."""
    da = np.datetime64(pd.Timestamp(_lib.parse_ts(a).date()), "D")
    db = np.datetime64(pd.Timestamp(_lib.parse_ts(b).date()), "D")
    return int(abs(np.busday_count(da, db)))


def main() -> int:
    rules = _lib.load_cluster_rules()
    events = _lib.load_v1_events()
    if not events:
        print("STOP: V1 catalog empty/missing at", _lib.V1_EVENTS)
        return 2
    print(f"Phase 1 — clustering {len(events)} unique V1 events "
          f"against {len(rules['clusters'])} locked clusters")

    # 1) classify
    classified: dict[str, list[dict]] = {name: [] for name in rules["priority_order"]}
    unclustered: list[dict] = []
    for e in events:
        res = _lib.classify_event(e["statement_text"], rules)
        row = {**e, **res}
        if res["cluster"] is None:
            unclustered.append(row)
        else:
            classified[res["cluster"]].append(row)

    # 2) de-dup per cluster (5-trading-day separation), keep earliest
    kept: list[dict] = []
    dropped: list[dict] = []
    for name, rows in classified.items():
        rows.sort(key=lambda r: r["timestamp_utc"])
        last_kept_ts = None
        last_kept_id = None
        for r in rows:
            if last_kept_ts is not None and _busdays(last_kept_ts, r["timestamp_utc"]) < DEDUP_TRADING_DAYS:
                dropped.append({
                    "event_id": r["event_id"], "cluster": name,
                    "timestamp_utc": r["timestamp_utc"],
                    "drop_reason": f"within {DEDUP_TRADING_DAYS} trading days of kept "
                                   f"{last_kept_id} in cluster {name}",
                })
                continue
            kept.append(r)
            last_kept_ts, last_kept_id = r["timestamp_utc"], r["event_id"]

    for r in unclustered:
        dropped.append({"event_id": r["event_id"], "cluster": None,
                        "timestamp_utc": r["timestamp_utc"],
                        "drop_reason": "no cluster matched (unclustered — excluded, not forced)"})

    # 3) emit clustered_events.jsonl
    out_rows = []
    for r in sorted(kept, key=lambda x: x["timestamp_utc"]):
        out_rows.append({
            "event_id": r["event_id"],
            "cluster": r["cluster"],
            "confidence": r["confidence"],
            "matched_keywords": r["matched_keywords"],
            "instruments": r["instruments"],
            "timestamp_utc": r["timestamp_utc"],
            "statement_text": _lib_trunc(r["statement_text"]),
            "notes": "",
        })
    _lib.write_jsonl(_lib.DATA_DIR / "clustered_events.jsonl", out_rows)
    _lib.write_jsonl(_lib.DATA_DIR / "dropped_events.jsonl", dropped)

    # 4) cluster counts (on KEPT events) + underpowered flags
    counts = {name: sum(1 for r in kept if r["cluster"] == name) for name in rules["priority_order"]}
    n_with_events = sum(1 for n, c in counts.items() if c > 0)
    denom = n_with_events * len(rules["instrument_universe"])
    lock = {
        "locked_at": _lib.utc_now_iso(),
        "universe_size": len(rules["instrument_universe"]),
        "n_clusters_with_events": n_with_events,
        "bonferroni_denominator": denom,
        "cluster_counts_kept": counts,
        "underpowered_clusters": [n for n, c in counts.items() if 0 < c < UNDERPOWERED_MIN],
        "n_events_in": len(events),
        "n_kept": len(kept),
        "n_dropped_dedup": sum(1 for d in dropped if d["cluster"] is not None),
        "n_unclustered": sum(1 for d in dropped if d["cluster"] is None),
        "note": "Bonferroni denominator = (clusters with >=1 kept event) x 9 (full universe, "
                "conservative). LOCKED at Phase 1 — Phase 4 reads this exact number.",
    }
    _lib.write_json_pretty(_lib.DATA_DIR / "bonferroni_lock.json", lock)

    # 5) report
    print(f"\n  in: {len(events)}  kept: {len(kept)}  "
          f"dedup-dropped: {lock['n_dropped_dedup']}  unclustered: {lock['n_unclustered']}")
    print("\n  cluster (kept events):")
    for name in rules["priority_order"]:
        c = counts[name]
        flag = "  UNDERPOWERED (<5)" if 0 < c < UNDERPOWERED_MIN else ("  EMPTY" if c == 0 else "")
        print(f"    {name:20s}: {c:3d}{flag}")
    print(f"\n  clusters with events: {n_with_events}  ->  BONFERRONI DENOMINATOR = {denom} (LOCKED)")
    return 0


def _lib_trunc(text: str) -> str:
    import re
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t[:TEXT_TRUNC] + ("…" if len(t) > TEXT_TRUNC else "")


if __name__ == "__main__":
    sys.exit(main())
