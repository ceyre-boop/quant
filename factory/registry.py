"""D4 — the registry: nothing anonymous. Every artifact records its data hash, feature
list, CV report path, and source HYP id. data/factory/registry.json, append-only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "factory" / "registry.json"


def register(model_id: str, hyp_id: str, data_sha256: str, features: list[str],
             cv_report_path: str, model_path: str, zoo_kind: str,
             min_confidence: float) -> dict:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    if any(e["model_id"] == model_id for e in reg):
        raise ValueError(f"model_id {model_id} already registered — registry is append-only")
    entry = {"model_id": model_id, "hyp_id": hyp_id, "data_sha256": data_sha256,
             "features": features, "cv_report": cv_report_path, "model_path": model_path,
             "zoo_kind": zoo_kind, "min_confidence": min_confidence,
             "registered": datetime.now(timezone.utc).isoformat()}
    reg.append(entry)
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    return entry


def lookup(model_id: str) -> dict | None:
    if not REGISTRY.exists():
        return None
    return next((e for e in json.loads(REGISTRY.read_text()) if e["model_id"] == model_id), None)
