"""Sealing / verifying helpers, exact idiom of scripts/research/preregister_hyp110_overnight.py."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG_DIR = ROOT / "data" / "research" / "preregister"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
MINED_N = ROOT / "data" / "research" / "yield_frontier" / "mined_n.json"


def mined_total() -> int:
    n = json.loads(MINED_N.read_text())["_total"]
    if not isinstance(n, int) or n < 1543:
        raise SystemExit(f"mined_n._total looks wrong ({n!r}); refusing to sign")
    return n


def canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "hash_lock"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_ledger(ledger: list) -> Path:
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    json.dump(ledger, tmp, indent=2); tmp.close()
    Path(tmp.name).replace(LEDGER)
    return backup


def write(doc: dict, note: str, source: str = "operator_session_2026-09-02") -> int:
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    path = PREREG_DIR / f"{doc['id']}.json"
    if path.exists():
        print(f"refusing to overwrite existing prereg {path}"); return 1
    ledger = json.loads(LEDGER.read_text())
    if any(e.get("id") == doc["id"] for e in ledger):
        print(f"refusing: {doc['id']} already in the ledger"); return 1
    doc["hash_lock"] = canonical_hash(doc)
    path.write_text(json.dumps(doc, indent=2))
    print(f"signed {path.name}  {doc['hash_lock'][:16]}")
    ledger.append({
        "id": doc["id"], "name": doc["name"], "status": "PREREGISTERED",
        "date_tested": None, "result": None, "verdict": None, "methodology_note": note,
        "hash_lock": doc["hash_lock"], "prereg_file": str(path.relative_to(ROOT)),
        "p_value": None, "bh_survives": None, "oos_sharpe": None, "is_sharpe": None,
        "prior_expectation": doc["prior_expectation"], "source": source, "auto_generated": False,
    })
    b = _write_ledger(ledger)
    print(f"ledger: +1 PREREGISTERED (backup {b.name})")
    return 0


def verify(hyp_id: str) -> dict:
    path = PREREG_DIR / f"{hyp_id}.json"
    doc = json.loads(path.read_text())
    good = doc.get("hash_lock") == canonical_hash(doc)
    entry = next((e for e in json.loads(LEDGER.read_text()) if e.get("id") == hyp_id), None)
    match = entry is not None and entry.get("hash_lock") == doc.get("hash_lock")
    print(f"{'OK  ' if good else 'FAIL'} {path.name} {doc.get('hash_lock', '')[:16]}   ledger match: {match}   status: {entry.get('status') if entry else None}")
    if not (good and match):
        raise SystemExit("PREREGISTRATION VERIFY FAILED — do not proceed")
    return doc


def gate_zero(hyp_id: str, label: str) -> dict:
    doc = verify(hyp_id)
    entry = next(e for e in json.loads(LEDGER.read_text()) if e.get("id") == hyp_id)
    if label == "start" and entry.get("status") != "PREREGISTERED":
        raise SystemExit(f"GATE ZERO FAILED: status is {entry.get('status')!r} — already adjudicated; does not run twice")
    return doc


def adjudicate(hyp_id: str, ledger_verdict: str, result: str, extra: dict) -> None:
    ledger = json.loads(LEDGER.read_text())
    for e in ledger:
        if e.get("id") == hyp_id:
            if e.get("status") != "PREREGISTERED":
                raise SystemExit("refusing: ledger entry is not PREREGISTERED")
            e.update({"status": "ADJUDICATED", "verdict": ledger_verdict, "result": result,
                      "date_tested": datetime.now(timezone.utc).strftime("%Y-%m-%d"), **extra})
    b = _write_ledger(ledger)
    print(f"ledger: {hyp_id} -> ADJUDICATED {ledger_verdict} (backup {b.name})")
