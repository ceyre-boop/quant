#!/usr/bin/env python3
"""W3 weekly self-review — the machine reads its week and proposes; only Colin promotes.

Reads the week's journal + attributions → writes review/YYYY-WW.md and appends
candidate hypotheses to the ledger as status=PROPOSED (a new status, distinct from
PREREGISTERED: a proposal has no locked design, no hash, no protocol claim — it is an
idea the machine surfaced from lived experience, awaiting the operator's promotion
into a real pre-registration).

Scheduled Sundays 17:00 ET (com.alta.weekly_review). Self-play under fixed rules:
this job may propose players' moves, never rule changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import params  # noqa: E402
from experience import attribution as att  # noqa: E402
from experience import journal  # noqa: E402

REVIEW_DIR = ROOT / "review"
LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"


def week_bounds(any_day: date) -> tuple[date, date, str]:
    monday = any_day - timedelta(days=any_day.weekday())
    iso = any_day.isocalendar()
    return monday, monday + timedelta(days=6), f"{iso.year}-W{iso.week:02d}"


def propose_to_ledger(proposals: list[dict], dry_run: bool) -> list[str]:
    """Append PROPOSED entries (never modify existing; backup + atomic; dedupe by id)."""
    if dry_run or not proposals:
        return [p["id"] for p in proposals]
    ledger = json.loads(LEDGER.read_text())
    existing = {e.get("id") for e in ledger}
    backup = LEDGER.with_suffix(f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    shutil.copy2(LEDGER, backup)
    added = []
    for p in proposals:
        if p["id"] in existing:
            continue
        ledger.append(p)
        added.append(p["id"])
    tmp = tempfile.NamedTemporaryFile("w", dir=LEDGER.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(ledger, indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(LEDGER)
    return added


def build_review(week_of: date, dry_run: bool = False) -> Path:
    start, end, tag = week_bounds(week_of)
    w = (str(start), str(end))
    rows = [r for r in journal.read_all() if w[0] <= r["decision_ts"][:10] <= w[1]]
    atts = {a["decision_id"]: a for a in att.read_attributions()}
    week_atts = [atts[r["decision_id"]] for r in rows if r["decision_id"] in atts]

    acted = [r for r in rows if r["action"] in ("ENTER", "CLOSE", "AMEND_STOP")]
    abstained = [r for r in rows if r["action"] == "ABSTAIN"]
    days_acted = sorted({r["decision_ts"][:10] for r in acted})
    days_abstained = sorted({r["decision_ts"][:10] for r in abstained}
                            - set(days_acted))
    cls_counts = Counter(a["cls"] for a in week_atts)

    surprises, proposals = [], []
    n_ambig = cls_counts.get("AMBIGUOUS", 0)
    if week_atts and n_ambig / len(week_atts) > 0.5:
        surprises.append(f"{n_ambig}/{len(week_atts)} attributions AMBIGUOUS — the dominant "
                         f"cause is missing exit-mechanism/thesis inputs on closed carry records.")
        proposals.append({
            "id": f"PROP-{tag}-exit-reason-capture",
            "name": "Record the exit MECHANISM on outcome backfill so attributions stop being AMBIGUOUS",
            "status": "PROPOSED",
            "date_proposed": str(week_of),
            "source": "experience/weekly_review.py (machine-proposed; operator promotion required)",
            "observation": (f"Week {tag}: {n_ambig} of {len(week_atts)} closed decisions "
                            f"unclassifiable under ATTRIBUTION_RUBRIC v1 — the outcome matcher "
                            f"records WIN/LOSS+R but not HOW the trade exited, and carry entries "
                            f"carry no falsification predicates."),
            "proposal": ("At outcome-backfill time, join the OANDA fill's close reason (or the "
                         "shadow log's decision for the same trade/date) onto the decision "
                         "record; going forward, predictive entries must carry Track-T "
                         "predicates so the thesis-alive test is mechanical."),
            "promotion_path": "operator review -> preregister under standard protocol if testable",
            "result": None, "p_value": None, "auto_generated": True,
        })
    if cls_counts.get("luck_good", 0) > 0:
        surprises.append(f"{cls_counts['luck_good']} win(s) with a dead thesis (luck_good) — "
                         f"review whether a thesis-invalidation exit would have fired earlier.")
    if not week_atts:
        surprises.append("No closed decisions attributed this week (shadow window; broker "
                         "closes pending) — journal + abstention capture is the week's output.")

    REVIEW_DIR.mkdir(exist_ok=True)
    path = REVIEW_DIR / f"{tag}.md"
    lines = [
        f"# Weekly Self-Review — {tag} ({start} → {end})", "",
        f"_Generated by experience/weekly_review.py; rubric {att.rubric_hash()}; "
        f"machine PROPOSES, operator promotes._", "",
        "## Acted vs abstained",
        f"- Days with actions: {len(days_acted)} ({', '.join(days_acted) or '—'})",
        f"- Days abstain-only: {len(days_abstained)} ({', '.join(days_abstained) or '—'})",
        f"- Rows: {len(acted)} actions ({Counter(r['action'] for r in acted) or '{}'}), "
        f"{len(abstained)} abstentions ({sum(1 for r in abstained if r.get('inferred'))} inferred).",
        "",
        "## Board context on decision days",
        *(f"- {r['decision_ts'][:10]} {r['pair']} {r['action']}: board_ref="
          f"{(r.get('board_ref') or {}).get('sha256', 'absent')[:12]}"
          for r in acted[:12]),
        "",
        "## Attribution",
        f"- Classes: {dict(cls_counts) or 'none'}",
        *(f"- {a['decision_id']}: **{a['cls']}** {a.get('overlays') or ''} — {a['rationale']}"
          for a in week_atts[:10]),
        "",
        "## Calibration",
        "- No live probabilities exist yet (factory gated; zero models). This section "
        "activates when a registered model emits calibrated p's.",
        "",
        "## Surprises",
        *(f"- {s}" for s in (surprises or ["none"])),
        "",
        "## Machine proposals (status=PROPOSED in the ledger; promotion is yours)",
        *(f"- {p['id']}: {p['name']}" for p in (proposals or [])),
        *([] if proposals else ["- none this week"]),
    ]

    # ── Precedents (Alexandrian Library) — guarded additive section (TICK-005) ─────────────
    # Whole block in ONE try/except: a library/board-DB error must never take down the Sunday
    # review. Flag-gated (config/parameters.yml :: experience.precedents.review_enabled) —
    # off => no section, no citations. dry_run shows the section but writes no citations.
    precedents_lines: list[str] = []
    try:
        precedents_cfg = (params.get("experience") or {}).get("precedents", {})
        if precedents_cfg.get("review_enabled", False):
            from experience import citations as cit
            from experience import precedents as prec
            extremes = prec.week_board_extremes(w[0], w[1])
            ctx_tags = prec.week_context_tags(rows, week_atts, extremes)
            found = prec.find_precedents(ctx_tags, top_k=precedents_cfg.get("top_k", 3))
            if found:
                precedents_lines = [
                    "",
                    "## Precedents (Alexandrian Library)",
                    f"_Structured retrieval (method: structured_v1); context tags: "
                    f"{', '.join(sorted(ctx_tags)) or 'none'}._",
                    "",
                    *(f"- **{p['label']}** ({p['event_date']}, {p['source']}): {p['why']} → "
                      f"{p['what_followed']} (~{p['outcome_days']}d, severity={p['severity']}, "
                      f"matched={p['matched_tags']})"
                      for p in found),
                ]
                if not dry_run:
                    week_info = {"tag": tag, "end": end, "review_path": str(path)}
                    cit.append_citations([
                        cit.make_citation(week_info, p, [r["decision_id"] for r in acted],
                                          basis={"board_extremes": extremes})
                        for p in found
                    ])
    except Exception as e:
        precedents_lines = ["", "## Precedents (Alexandrian Library)",
                            f"_Precedent retrieval unavailable this week ({e})._"]
    lines.extend(precedents_lines)

    path.write_text("\n".join(lines) + "\n")
    added = propose_to_ledger(proposals, dry_run)
    print(f"[weekly_review] {path.name}: {len(rows)} rows, {len(week_atts)} attributions, "
          f"proposals added: {added or 'none'}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week-of", default=str(date.today()), help="any day inside the target week")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    build_review(date.fromisoformat(a.week_of), a.dry_run)
