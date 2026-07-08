#!/usr/bin/env python3
"""Build the Layer-1 Phase-2 dataset (HYP-064): features_v1.parquet, labels_v1.parquet, load_report.json.

Phase 2 ONLY — NO model training, NO holdout (2024+) reads. Fail-loud: features that don't load
are reported (load_report.json + stdout), never zero-filled.

    python3 scripts/build_layer1_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sovereign.layer1.data_loader import LoadReport, TRAIN_END  # noqa: E402
from sovereign.layer1.feature_builder import build_panel        # noqa: E402
from sovereign.layer1.label_builder import build_labels         # noqa: E402

OUT = ROOT / "data" / "layer1"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = LoadReport()

    print("Building feature panel (network: yfinance / FRED / CFTC)...", flush=True)
    features, report, coverage = build_panel(report=report)
    print("Building forward-direction labels...", flush=True)
    labels, report = build_labels(report=report)

    rep = report.to_dict()
    rep["coverage"] = coverage

    if features.empty or labels.empty:
        (OUT / "load_report.json").write_text(json.dumps(rep, indent=2, default=str))
        print("FATAL: empty features or labels — nothing written except load_report.json.")
        for s in report.failures():
            print(f"   FAIL {s.source}: {s.reason}")
        return 1

    # Align features and labels on the common (date, pair) index. We KEEP the last-H rows per
    # pair whose forward label is NaN (no forward window): the features there are valid (computed
    # at t0), only the label is unknowable — preserving them makes the label's forward-looking
    # nature structurally provable in the parquet (the tail is NaN). Phase 3 drops NaN labels at
    # training time. This is also why no holdout is needed for labels: the forward window simply
    # runs out of (in-window) data and the row goes unlabelled.
    common = features.index.intersection(labels.index)
    features = features.loc[common].sort_index()
    labels = labels.loc[common].sort_index()

    # Holdout guard — belt and suspenders (nothing 2024+ should ever have been fetched).
    max_date = features.index.get_level_values(0).max()
    assert max_date <= pd.Timestamp(TRAIN_END), f"HOLDOUT LEAK: max date {max_date} > {TRAIN_END}"

    features.to_parquet(OUT / "features_v1.parquet")
    labels.to_parquet(OUT / "labels_v1.parquet")
    (OUT / "load_report.json").write_text(json.dumps(rep, indent=2, default=str))

    dates = features.index.get_level_values(0)
    print("=" * 72)
    print("LAYER-1 PHASE-2 DATASET BUILT")
    print(f"  features_v1.parquet : {features.shape[0]} rows x {features.shape[1]} features")
    print(f"  labels_v1.parquet   : {labels.shape[0]} rows x {labels.shape[1]} labels")
    print(f"  date range          : {dates.min().date()} .. {max_date.date()}  (holdout 2024+ untouched)")
    print(f"  pairs               : {sorted(set(features.index.get_level_values(1)))}")
    print(f"  features built      : {coverage['built']}/{coverage['declared']} declared")
    if coverage["missing"]:
        print(f"  !! FEATURES NOT BUILT ({len(coverage['missing'])}) — reported, NOT zero-filled:")
        for m in coverage["missing"]:
            print(f"       - {m}")
    fails = report.failures()
    if fails:
        print(f"  !! SOURCE LOAD FAILURES ({len(fails)}):")
        for s in fails:
            print(f"       - {s.source}: {s.reason}")
    print(f"  label balance (5d)  : {labels['fwd_direction_5d'].mean():.3f} positive")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
