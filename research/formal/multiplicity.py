"""Trial-count accounting as a static property of a prereg document, not a memory."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MINED_N = ROOT / "data" / "research" / "yield_frontier" / "mined_n.json"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"


def count_declared(doc: dict) -> int:
    """Sum of every declared trial-bearing element in a prereg: claims, models, universes, cells."""
    n = 0
    n += len(doc.get("claims", {})) if isinstance(doc.get("claims"), dict) else 0
    n += len(doc.get("models_frozen", [])) if isinstance(doc.get("models_frozen"), list) else 0
    n += len(doc.get("universes", [])) if isinstance(doc.get("universes"), list) else 0
    n += int(doc.get("cells_compared", 0))
    return max(n, 1)


def expected_n_trials(doc: dict) -> tuple[int, int]:
    """(declared n_trials in the doc, floor implied by mined_n + ledger claims + this doc)."""
    mined = json.loads(MINED_N.read_text())["_total"]
    ledger = json.loads(LEDGER.read_text())
    prior_claims = sum(1 for e in ledger if str(e.get("id", "")).startswith("HYP-1") and int(str(e["id"]).split("-")[1][:3]) >= 109)
    declared = int(doc.get("n_trials") or doc.get("frozen_parameters", {}).get("n_trials") or 0)
    floor = mined + prior_claims + count_declared(doc)
    return declared, floor
