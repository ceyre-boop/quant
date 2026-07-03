"""D5 — ignition, HARD-gated. `python -m factory.train --hyp HYP-XXX` refuses unless the
ledger entry's status/verdict is CONFIRMED, printing RISK_CONSTITUTION Article 6
verbatim. Building the factory was unrestricted; igniting it is not.

First ignition is the operator's to witness — even a CONFIRMED entry only UNLOCKS the
command; nothing schedules it."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
CONSTITUTION = ROOT / "RISK_CONSTITUTION.md"


def article_6() -> str:
    text = CONSTITUTION.read_text()
    m = re.search(r"## Article 6[^\n]*\n(.*?)(?=\n## |\n---)", text, re.S)
    return ("## Article 6 — Unproven Edges\n" + m.group(1).strip()) if m else \
        "Article 6: No live capital is allocated to any edge without a confirmed, " \
        "pre-registered, out-of-sample entry in the hypothesis ledger."


def check_ignition(hyp_id: str, ledger_path: Path = LEDGER) -> dict:
    """Returns the CONFIRMED ledger entry or raises SystemExit with Article 6."""
    ledger = json.loads(ledger_path.read_text())
    entry = next((e for e in ledger if e.get("id") == hyp_id), None)
    if entry is None:
        raise SystemExit(f"IGNITION REFUSED — {hyp_id} does not exist in the hypothesis ledger.\n\n"
                         + article_6())
    status = entry.get("status") or entry.get("verdict")
    if status != "CONFIRMED":
        raise SystemExit(
            f"IGNITION REFUSED — {hyp_id} status is {status!r}, not CONFIRMED.\n"
            f"Training executes only against a confirmed, pre-registered, out-of-sample "
            f"ledger entry.\n\n{article_6()}")
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="factory.train")
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--zoo", default="logistic", choices=("logistic", "xgb", "mlp"))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31")
    a = ap.parse_args(argv)

    entry = check_ignition(a.hyp)          # the gate — everything below is unreachable until CONFIRMED
    print(f"IGNITION UNLOCKED for {a.hyp} (status CONFIRMED, hash "
          f"{str(entry.get('hash_lock', ''))[:12]}…) — proceeding to train {a.zoo}.")
    from factory.feature_store import snapshot
    manifest = snapshot(a.start, a.end)
    print(f"feature snapshot {manifest['sha256'][:12]}… rows={manifest['rows']}")
    # Label construction + fit + registry happen here at first REAL ignition — deliberately
    # not implemented further until a CONFIRMED hypothesis defines its own label. Nothing
    # trains speculatively (Article 6; the label IS the hypothesis).
    print("NOTE: label construction is hypothesis-defined; first real ignition implements it "
          "against the CONFIRMED entry's locked design, witnessed by the operator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
