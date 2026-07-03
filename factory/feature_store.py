"""D1 — point-in-time feature store over sentiment_board_state.

Every row carries BOTH timestamps: `date` (the as-of trading day the value describes)
and the provenance/release timestamp of its slowest-moving source column (COT release
Friday 15:30 ET; options iv_obs_date; macro ASOF). The board's release-date keying is
what makes `date` safe to train on — audited by scripts/audit_look_ahead.py (0
violations). Snapshots are parquet + a sha256 manifest: models reference DATA HASHES,
never "whatever the table said that day".
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experience.journal import board_row_hash  # shared row-hash (W1 board_ref uses the same fn)

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "factory" / "snapshots"

FEATURES_V1 = ["cot_net_pct", "cot_net_oi", "cot_net_pct_1y", "cot_net_z", "cot_flush_1w",
               "tff_lev_net_pct", "econ_surprise_z", "gdelt_tone", "gdelt_tone_5d",
               "vrp_signal", "vrp_pct", "rr25", "bf25", "atm_term_slope",
               "macro_curve", "macro_spread", "macro_inflation", "vix_level", "vix_momentum"]


def load_board(start: str, end: str, pairs: list[str] | None = None) -> pd.DataFrame:
    from sovereign.sentiment.store import connect
    con = connect(read_only=True)
    try:
        df = con.execute("SELECT * FROM sentiment_board_state WHERE date BETWEEN ? AND ? "
                         "ORDER BY date, pair", [start, end]).df()
        rel = con.execute("SELECT pair, publish_date, release_ts FROM sentiment_cot_weekly").df()
    finally:
        con.close()
    if pairs:
        df = df[df["pair"].isin(pairs)]
    # provenance: the release_ts of the latest COT row visible on each (date, pair)
    rel["publish_date"] = pd.to_datetime(rel["publish_date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["pair", "date"])
    out = []
    for pair, grp in df.groupby("pair"):
        r = rel[rel["pair"] == pair].sort_values("publish_date")
        merged = pd.merge_asof(grp, r[["publish_date", "release_ts"]].rename(
            columns={"publish_date": "date"}), on="date", direction="backward")
        out.append(merged.rename(columns={"release_ts": "cot_release_ts"}))
    return pd.concat(out, ignore_index=True) if out else df.assign(cot_release_ts=None)


def snapshot(start: str, end: str, pairs: list[str] | None = None,
             features: list[str] | None = None, tag: str = "v1") -> dict:
    """Write a hashed snapshot; returns the manifest (incl. the data sha256)."""
    feats = features or FEATURES_V1
    df = load_board(start, end, pairs)
    cols = ["date", "pair", "cot_release_ts"] + [c for c in feats if c in df.columns]
    data = df[cols].reset_index(drop=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    payload = data.to_csv(index=False).encode()
    sha = hashlib.sha256(payload).hexdigest()
    path = SNAP_DIR / f"board_{tag}_{sha[:12]}.parquet"
    data.to_parquet(path)
    manifest = {"sha256": sha, "path": str(path.relative_to(ROOT)), "tag": tag,
                "rows": int(len(data)), "start": start, "end": end,
                "pairs": sorted(data["pair"].unique().tolist()),
                "features": [c for c in feats if c in df.columns],
                "coverage": {c: float(data[c].notna().mean()) for c in feats if c in data.columns},
                "created": datetime.now(timezone.utc).isoformat(),
                "row_hash_fn": "experience.journal.board_row_hash"}
    (SNAP_DIR / f"board_{tag}_{sha[:12]}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["load_board", "snapshot", "board_row_hash", "FEATURES_V1"]
