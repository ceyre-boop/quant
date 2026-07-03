"""Decision journal (W1) — one row per decision by any engine, abstentions included.

Storage: data/experience/journal_YYYY_MM.jsonl (tracked). Append-only via idempotent
upsert keyed on decision_id (a re-run never duplicates). Coexists with
sovereign/intelligence/decision_logger.py — that remains the Oracle's entry record and
NON-NEGOTIABLE #2 surface; this journal is the memory organ's wider net (it also sees
exit-shadow steps and abstentions, which decision_logger never records).

Row schema (v1):
  ts               ISO UTC written time
  decision_ts      when the decision was made (bar/run time)
  decision_id      stable unique id (dedupe key)
  engine           carry | exit_shadow | predictive
  pair             OANDA-normalized (EUR_USD)
  board_ref        {date, pair, sha256} of the board row backing the decision (None if absent)
  thesis           {kind: structural_carry|hypothesis|abstention, id, falsification_predicates}
  action           ENTER | HOLD | AMEND_STOP | CLOSE | ABSTAIN
  size             units/risk_pct where known
  inferred         True when the row was reconstructed from absence-of-evidence (backfill)
  source           provenance string (which log/file produced it)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = ROOT / "data" / "experience"
RUBRIC = ROOT / "experience" / "ATTRIBUTION_RUBRIC.md"


def board_row_hash(row: dict) -> str:
    """Deterministic sha256 of a board row (sorted, compact). Shared with factory D1."""
    return hashlib.sha256(json.dumps(
        {k: (None if v != v else v) if isinstance(v, float) else v
         for k, v in sorted(row.items()) if k != "built_at"},
        sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def board_ref(d, pair: str, con=None) -> dict | None:
    """{date, pair, sha256} for the board row at (date,pair); None when absent."""
    from sovereign.sentiment.store import connect
    own = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute("SELECT * FROM sentiment_board_state WHERE date = ? AND pair = ?",
                         [str(d)[:10], pair.replace("_", "")]).df()
    finally:
        if own:
            con.close()
    if df.empty:
        return None
    return {"date": str(d)[:10], "pair": pair.replace("_", ""),
            "sha256": board_row_hash(df.iloc[0].to_dict())}


def _path(decision_ts: str) -> Path:
    return JOURNAL_DIR / f"journal_{decision_ts[:7].replace('-', '_')}.jsonl"


def read_all(month_glob: str = "journal_*.jsonl") -> list[dict]:
    rows = []
    for f in sorted(JOURNAL_DIR.glob(month_glob)):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def upsert(rows: list[dict]) -> int:
    """Idempotent append: rows whose decision_id already exists in their month file are skipped."""
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    by_file: dict[Path, list[dict]] = {}
    for r in rows:
        assert r.get("decision_id") and r.get("engine") and r.get("action"), r
        r.setdefault("ts", datetime.now(timezone.utc).isoformat())
        by_file.setdefault(_path(r["decision_ts"]), []).append(r)
    written = 0
    for path, batch in by_file.items():
        existing = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    existing.add(json.loads(line).get("decision_id"))
        with path.open("a") as fh:
            for r in batch:
                if r["decision_id"] in existing:
                    continue
                fh.write(json.dumps(r, default=str) + "\n")
                existing.add(r["decision_id"])
                written += 1
    return written
