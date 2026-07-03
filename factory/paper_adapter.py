"""D6 — paper adapter: hosts a REGISTERED model under constitution caps + abstention +
Track-T thesis predicates. Built and stub-tested; NOT enabled, NOT scheduled — enabling
it is a separate operator decision after the shadow-window gate.

Caps come from config/risk_constitution.yaml — **DRAFT VALUES** (unratified); every
decision line this adapter emits is stamped DRAFT-CAPS so nobody mistakes paper
discipline for a ratified constitution. It never touches a broker: it JOURNALS
would-be decisions (engine='predictive') through the experience organ.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from experience import journal
from factory.registry import lookup

ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION_YAML = ROOT / "config" / "risk_constitution.yaml"
ENABLED = False   # flipping this is an operator act, recorded in NEXT.md — not a code default


class PaperAdapter:
    def __init__(self, model_id: str):
        self.entry = lookup(model_id)
        if self.entry is None:
            raise SystemExit(f"paper adapter: model {model_id!r} is not in the registry — "
                             f"nothing anonymous runs here (D4).")
        cfg = yaml.safe_load(CONSTITUTION_YAML.read_text())
        assert cfg["meta"]["status"] == "DRAFT" or cfg["meta"]["ratified"], "constitution state unreadable"
        self.caps = {"per_trade_frac": cfg["article_1_per_trade"]["hard_cap_frac"],
                     "carry_heat_frac": cfg["article_2_carry_complex"]["carry_heat_cap_frac"],
                     "draft": not cfg["meta"]["ratified"]}

    def cap_stamp(self) -> str:
        return ("DRAFT-CAPS" if self.caps["draft"] else "RATIFIED-CAPS") + \
            f" per_trade={self.caps['per_trade_frac']} heat={self.caps['carry_heat_frac']}"

    def decide_and_journal(self, pair: str, decision: dict, predicates: dict | None,
                           board_date: str, hyp_id: str, dry_run: bool = True) -> dict:
        """One model decision -> journaled predictive row (ABSTAIN included). Never a broker call."""
        if not ENABLED and not dry_run:
            raise SystemExit("paper adapter is NOT enabled — enabling is an operator act "
                             "recorded in NEXT.md, never a default.")
        size = None
        if decision["decision"] != "ABSTAIN":
            size = {"risk_frac": min(self.caps["per_trade_frac"], 0.0075),
                    "cap_stamp": self.cap_stamp()}
        row = {"decision_ts": f"{board_date}T00:00:00+00:00",
               "decision_id": f"predictive:{self.entry['model_id']}:{pair}:{board_date}",
               "engine": "predictive", "pair": pair,
               "board_ref": journal.board_ref(board_date, pair),
               "thesis": {"kind": "hypothesis", "id": hyp_id,
                          "falsification_predicates": predicates},
               "action": "ABSTAIN" if decision["decision"] == "ABSTAIN" else "ENTER",
               "size": size,
               "detail": {"p": decision["p"], "confidence": decision["confidence"],
                          "direction": decision["decision"], "model": self.entry["model_id"],
                          "data_sha256": self.entry["data_sha256"], "cap_stamp": self.cap_stamp(),
                          "dry_run": dry_run},
               "inferred": False, "source": "factory/paper_adapter.py"}
        if not dry_run:
            journal.upsert([row])
        print(f"[paper_adapter {self.cap_stamp()}] {pair} {board_date}: "
              f"{decision['decision']} (p={decision['p']:.3f})"
              + (" [dry-run, not journaled]" if dry_run else ""))
        return row
