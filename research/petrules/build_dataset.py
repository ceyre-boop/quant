"""Build the small proof-of-pipeline dataset and write it to research/petrules/output/.

Emits one row per FrozenEvent as JSONL (feature values + per-feature publication timestamps)
plus a flat CSV of the present scalar summaries. Every row re-runs the build-time audit, so
this script cannot emit a leaked example.

Usage: python3 -m research.petrules.build_dataset [TICKER ...]
       (no args → offline fixture sample: AAPL DKS)
Run from the repo root.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .replay_engine import build_sample

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def _serialize(fe) -> dict:
    return {
        "event_id": fe.event_id,
        "ticker": fe.ticker,
        "freeze_ts": fe.freeze_ts.isoformat(),
        "label": {
            "value": fe.label.value,
            "source": fe.label.source,
            "published_ts": fe.label.published_ts.isoformat() if fe.label.published_ts else None,
        },
        "features": {
            name: {
                "value": pv.value,
                "source": pv.source,
                "published_ts": pv.published_ts.isoformat() if pv.published_ts else None,
                "present": pv.is_present,
            }
            for name, pv in fe.features.items()
        },
        "meta": fe.meta,
    }


def main(argv):
    tickers = argv or None
    events = build_sample(tickers)
    jsonl = OUT / "sample_events.jsonl"
    with jsonl.open("w") as f:
        for fe in events:
            f.write(json.dumps(_serialize(fe)) + "\n")

    csv_path = OUT / "sample_events.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "ticker", "freeze_ts", "label_surprise", "label_ts",
                    "hist_prior_beats", "hist_mean_surprise", "n_present_features", "edgar_online"])
        for fe in events:
            hist = fe.features["earnings_surprise_history"].value or {}
            w.writerow([
                fe.event_id, fe.ticker, fe.freeze_ts.isoformat(),
                fe.label.value, fe.label.published_ts.isoformat() if fe.label.published_ts else None,
                hist.get("prior_beats"), hist.get("mean_surprise"),
                len(fe.present_features()), fe.meta.get("edgar_online"),
            ])

    print(f"built {len(events)} frozen events → {jsonl.name}, {csv_path.name}")
    if events:
        online = sum(1 for e in events if e.meta.get("edgar_online"))
        print(f"  EDGAR online for {online}/{len(events)} events "
              f"(offline → disclosed-flow features ABSENT, not fabricated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
