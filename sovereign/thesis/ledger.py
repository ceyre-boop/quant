"""Append-only thesis ledger. Refuses duplicate ids and mutation of sealed fields."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import SEALED_FIELDS, canonical_hash, sealed_body, validate

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "thesis" / "thesis_ledger.jsonl"
PREREG_DIR = ROOT / "data" / "thesis" / "preregister"


class LedgerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_all(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def get(thesis_id: str, path: Path = LEDGER) -> dict[str, Any]:
    for r in read_all(path):
        if r["id"] == thesis_id:
            return r
    raise LedgerError(f"{thesis_id} not in ledger")


def _write_all(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, path)


def create(record: dict[str, Any], path: Path = LEDGER) -> dict[str, Any]:
    errs = validate(record)
    if errs:
        raise LedgerError("; ".join(errs))
    recs = read_all(path)
    if any(r["id"] == record["id"] for r in recs):
        raise LedgerError(f"{record['id']} already exists; ids are never reused")
    record = dict(record)
    record.setdefault("status", "DRAFT")
    record.setdefault("scorecards", [])
    record.setdefault("notes", [])
    record["created_at"] = _now()
    recs.append(record)
    _write_all(recs, path)
    return record


def update(record: dict[str, Any], path: Path = LEDGER) -> dict[str, Any]:
    """Replace a record. If it is sealed, every sealed field must be byte-identical."""
    recs = read_all(path)
    for i, r in enumerate(recs):
        if r["id"] == record["id"]:
            if r.get("hash_lock"):
                if canonical_hash(record) != r["hash_lock"]:
                    changed = [k for k in SEALED_FIELDS if r.get(k) != record.get(k)]
                    raise LedgerError(
                        f"{r['id']} is sealed; refusing to change {changed}. "
                        "Close it and open a new thesis instead."
                    )
                record = dict(record, hash_lock=r["hash_lock"], sealed_at=r["sealed_at"])
            record["updated_at"] = _now()
            recs[i] = record
            _write_all(recs, path)
            return record
    raise LedgerError(f"{record['id']} not in ledger")


def seal(thesis_id: str, path: Path = LEDGER, prereg_dir: Path = PREREG_DIR) -> dict[str, Any]:
    r = get(thesis_id, path)
    if r.get("hash_lock"):
        raise LedgerError(f"{thesis_id} already sealed ({r['hash_lock'][:8]})")
    errs = validate(r)
    if errs:
        raise LedgerError("; ".join(errs))
    h = canonical_hash(r)
    prereg_dir.mkdir(parents=True, exist_ok=True)
    pf = prereg_dir / f"{thesis_id}.json"
    if pf.exists():
        raise LedgerError(f"{pf} exists; refusing to overwrite a seal")
    doc = sealed_body(r)
    doc["hash_lock"] = h
    doc["sealed_at"] = _now()
    pf.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    r = dict(r, hash_lock=h, sealed_at=doc["sealed_at"], status="SEALED")
    recs = read_all(path)
    recs = [r if x["id"] == thesis_id else x for x in recs]
    _write_all(recs, path)
    return r


def verify(thesis_id: str, path: Path = LEDGER, prereg_dir: Path = PREREG_DIR) -> bool:
    r = get(thesis_id, path)
    pf = prereg_dir / f"{thesis_id}.json"
    if not r.get("hash_lock") or not pf.exists():
        return False
    doc = json.loads(pf.read_text())
    return canonical_hash(doc) == doc.get("hash_lock") == r["hash_lock"] == canonical_hash(r)


def set_status(thesis_id: str, status: str, note: str | None = None, path: Path = LEDGER) -> dict[str, Any]:
    r = get(thesis_id, path)
    r = dict(r, status=status)
    if note:
        r.setdefault("notes", []).append(f"{_now()} {note}")
    return update(r, path)


def add_scorecard(thesis_id: str, card: dict[str, Any], path: Path = LEDGER) -> dict[str, Any]:
    r = get(thesis_id, path)
    r = dict(r)
    r.setdefault("scorecards", []).append(card)
    if card.get("invalidated") and r.get("status") in ("SEALED", "LIVE"):
        r["status"] = "INVALIDATED"
    return update(r, path)
