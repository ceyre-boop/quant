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
import re
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

# ── TICK-006 forensics feed sources ─────────────────────────────────────────────────────
# Module-level so tests can monkeypatch per-feed (mirrors the journal.JOURNAL_DIR pattern).
# Functions below re-read these names from the module namespace at call time — never bind
# them as default-argument values, or a monkeypatch after import won't be seen.
ORACLE_REFLECTIONS_DIR = ROOT / "data" / "oracle" / "reflections"
VETO_LEDGER_DIR = ROOT / "data" / "ledger"
AUDIT_REPORTS_DIR = ROOT / "audit" / "reports"
LESSON_VELOCITY_PATH = ROOT / "data" / "oracle" / "lesson_velocity.json"
MARKET_BRIEFING_LATEST = ROOT / "data" / "oracle" / "market_briefings" / "latest.json"

# Ledger statuses that are still "in flight" — anything else counts as a terminal verdict
# for the propose-dedup gate in _feed_hypothesis_research.
_NON_TERMINAL_LEDGER_STATUSES = {"PREREGISTERED", "PROPOSED", "RETEST_BLOCKED"}
# Heterogeneous date fields seen across ledger entries (no single canonical "date" key).
_LEDGER_DATE_FIELDS = (
    "date", "date_registered", "date_confirmed", "date_decided", "date_formed",
    "date_tested", "date_updated", "date_proposed", "created_date", "confirmed_date",
    "closed", "added", "formed", "deployed",
)
_PROP_ID_RE = re.compile(r"^PROP-\d{4}-W\d{2}-(.+)$")


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


# ── TICK-006 forensics feeds ─────────────────────────────────────────────────────────────
# Six independent, guarded, additive reads. Each function is called from build_review inside
# its OWN try/except — a missing/corrupt source must degrade to "feed unavailable: <name>"
# and NEVER raise. Oracle-derived text (feeds 1, 5, 6) is REVIEW PROSE ONLY; none of it
# re-enters the proposals/ledger-write path.

def _feed_oracle_health(week_end: date) -> list[str]:
    """data/oracle/reflections/*.json -> latest system_health_note on/before week end.

    RED-1 (see NEXT.md / tickets/backlog.md): Oracle reflection reads a contaminated
    decision-log summary. Marked quarantined until TICK-011's source-exclusion fix lands;
    context only, never a proposal input.
    """
    candidates = []
    for p in ORACLE_REFLECTIONS_DIR.glob("*.json"):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d <= week_end:
            candidates.append((d, p))
    if not candidates:
        return ["### System health (Oracle)", "_No reflection on or before week end._"]
    latest_date, latest_path = max(candidates, key=lambda t: t[0])
    data = json.loads(latest_path.read_text())
    note = (data.get("reflection") or {}).get("system_health_note")
    if not note:
        return ["### System health (Oracle)",
                f"_{latest_date}: reflection present, no system_health_note field._"]
    return ["### System health (Oracle)",
            f"- {latest_date}: {note} [source quarantined: RED-1 open — context only]"]


def _feed_hypothesis_research(start: str, end: str) -> tuple[list[str], set[str]]:
    """data/agent/hypothesis_ledger.json -> this week's verdict Counter + INTERIM SEAL /
    BLOCKED tallies over entries/annotations dated in-window.

    Also returns the terminal-verdict key set (raw ledger id, plus the stable PROP-<tag>-
    <slug> suffix) so the proposal writer never re-proposes an idea that already has a
    terminal verdict somewhere in the ledger.
    """
    entries = json.loads(LEDGER.read_text())
    verdicts: Counter = Counter()
    interim_seal = 0
    blocked = 0
    terminal_keys: set[str] = set()
    for e in entries:
        status = e.get("status")
        eid = e.get("id") or ""
        if status and status not in _NON_TERMINAL_LEDGER_STATUSES:
            terminal_keys.add(eid)
            m = _PROP_ID_RE.match(eid)
            if m:
                terminal_keys.add(m.group(1))
        dates_in_window = [str(e.get(f))[:10] for f in _LEDGER_DATE_FIELDS
                           if isinstance(e.get(f), str)]
        if status and any(start <= d <= end for d in dates_in_window):
            verdicts[status] += 1
            if "BLOCKED" in status:
                blocked += 1
        for ann in (e.get("annotations") or []):
            adate = str(ann.get("date", ""))[:10]
            if start <= adate <= end:
                note = ann.get("note") or ""
                if isinstance(note, str) and note.startswith("INTERIM SEAL"):
                    interim_seal += 1
                if isinstance(note, str) and "BLOCKED" in note:
                    blocked += 1
    lines = [
        "### This week's research (hypothesis ledger)",
        f"- Verdicts this week: {dict(verdicts) or 'none'}",
        f"- INTERIM SEAL annotations: {interim_seal}",
        f"- BLOCKED: {blocked}",
    ]
    return lines, terminal_keys


def _feed_veto_ratio(start: date, end: date, n_acted: int, n_abstained: int) -> list[str]:
    """data/ledger/veto_ledger_{YYYY_MM}.jsonl (+ ict_veto_ledger_ variant) -> acted:
    abstained:vetoed ratio and the top veto stages fired this week."""
    months = sorted({start.strftime("%Y_%m"), end.strftime("%Y_%m")})
    start_s, end_s = str(start), str(end)

    def _count(prefix: str, stage_key: str) -> tuple[int, Counter]:
        total = 0
        stages: Counter = Counter()
        for m in months:
            p = VETO_LEDGER_DIR / f"{prefix}_{m}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ts = str(rec.get("timestamp", ""))[:10]
                if start_s <= ts <= end_s:
                    total += 1
                    stages[rec.get(stage_key, "UNKNOWN")] += 1
        return total, stages

    fx_total, fx_stages = _count("veto_ledger", "stage")
    ict_total, ict_stages = _count("ict_veto_ledger", "veto_stage")
    vetoed = fx_total + ict_total
    stages = fx_stages + ict_stages
    return [
        "### Acted : abstained : vetoed",
        f"- Ratio (acted:abstained:vetoed) = {n_acted}:{n_abstained}:{vetoed} "
        f"(forex veto={fx_total}, ict veto={ict_total})",
        f"- Top veto stages: {stages.most_common(3) or 'none'}",
    ]


def _feed_audit_parity(week_end: date) -> list[str]:
    """audit/reports/{date}.json (latest <= week end) -> one L1/L2 parity line."""
    candidates = []
    for p in AUDIT_REPORTS_DIR.glob("*.json"):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d <= week_end:
            candidates.append((d, p))
    if not candidates:
        return ["### Audit parity (shadow window)", "_No audit report on or before week end._"]
    latest_date, latest_path = max(candidates, key=lambda t: t[0])
    report = json.loads(latest_path.read_text())
    l1 = report.get("l1") or {}
    l2 = report.get("l2") or {}
    cont_viol = len(l1.get("continuity_violations") or [])
    line = (f"- {latest_date}: L1 determinism pass_rate={l1.get('pass_rate', 'n/a')}, "
            f"L2 match_rate={l2.get('match_rate', 'n/a')} "
            f"({l2.get('matched', '?')}/{l2.get('scored', '?')}), "
            f"continuity_violations={cont_viol}")
    return ["### Audit parity (shadow window)", line]


def _feed_lesson_velocity() -> list[str]:
    """data/oracle/lesson_velocity.json -> current lesson + formation stage, one line."""
    data = json.loads(LESSON_VELOCITY_PATH.read_text())
    cur = data.get("current_lesson") or {}
    if not cur:
        return ["### Lesson velocity (Oracle)", "_No lesson currently forming._"]
    label = cur.get("component_label", "unknown")
    status = cur.get("status", "unknown")
    days = cur.get("days_forming", "?")
    text = (cur.get("latest_lesson_text") or "")[:160]
    return ["### Lesson velocity (Oracle)",
            f"- Current lesson: **{label}** ({status}, {days}d forming) — {text}"]


def _feed_macro_briefing() -> list[str]:
    """data/oracle/market_briefings/latest.json -> FRED macro_economic summary ONLY.

    regime_read is ES/NQ-specific and is deliberately discarded; provenance.verified=false
    is always noted (this is analyst/Oracle narrative, not a validated data feed).
    """
    data = json.loads(MARKET_BRIEFING_LATEST.read_text())
    macro = data.get("macro_economic") or {}
    summary = macro.get("summary")
    if not summary:
        return ["### Macro (FRED, via market briefing)",
                "_No macro_economic.summary in latest briefing._"]
    verified = bool((data.get("provenance") or {}).get("verified"))
    note = "" if verified else " (provenance.verified=false — context only, not a trading input)"
    return ["### Macro (FRED, via market briefing)",
            f"- {data.get('date', '?')}: {summary}{note}"]


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

    # ── Feed 2 (hypothesis ledger) read early (TICK-006) ────────────────────────────────
    # Its terminal-verdict keys gate whether a candidate proposal below gets (re-)proposed;
    # its prose lines are reused verbatim in the "## Forensics" assembly further down.
    # Guarded: an unreadable ledger fails OPEN (empty set == no suppression) — "never
    # raises" wins over "never re-propose" when the source itself is broken.
    try:
        research_lines, _terminal_keys = _feed_hypothesis_research(w[0], w[1])
    except Exception as e:
        research_lines = ["### This week's research (hypothesis ledger)",
                          f"_feed unavailable: hypothesis_research ({e})_"]
        _terminal_keys = set()

    surprises, proposals = [], []
    n_ambig = cls_counts.get("AMBIGUOUS", 0)
    if week_atts and n_ambig / len(week_atts) > 0.5:
        surprises.append(f"{n_ambig}/{len(week_atts)} attributions AMBIGUOUS — the dominant "
                         f"cause is missing exit-mechanism/thesis inputs on closed carry records.")
        prop_id = f"PROP-{tag}-exit-reason-capture"
        if not ({prop_id, "exit-reason-capture"} & _terminal_keys):
            proposals.append({
                "id": prop_id,
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

    # ── Forensics (TICK-006) — six guarded, additive review-prose feeds ─────────────────
    # One "## Forensics" section; each feed below is its own try/except so a missing or
    # corrupt source degrades to "feed unavailable: <name>" and NEVER raises — the Sunday
    # review must never die on a feed. Oracle-derived text (health/lessons/macro) is REVIEW
    # PROSE ONLY and never re-enters the proposals/ledger-write path.
    forensics_lines: list[str] = ["", "## Forensics", ""]

    forensics_lines.extend(research_lines)  # feed 2 — computed earlier for the dedup gate
    forensics_lines.append("")

    try:
        forensics_lines.extend(_feed_oracle_health(end))
    except Exception as e:
        forensics_lines += ["### System health (Oracle)", f"_feed unavailable: oracle_health ({e})_"]
    forensics_lines.append("")

    try:
        forensics_lines.extend(_feed_veto_ratio(start, end, len(acted), len(abstained)))
    except Exception as e:
        forensics_lines += ["### Acted : abstained : vetoed", f"_feed unavailable: veto_ratio ({e})_"]
    forensics_lines.append("")

    try:
        forensics_lines.extend(_feed_audit_parity(end))
    except Exception as e:
        forensics_lines += ["### Audit parity (shadow window)", f"_feed unavailable: audit_parity ({e})_"]
    forensics_lines.append("")

    try:
        forensics_lines.extend(_feed_lesson_velocity())
    except Exception as e:
        forensics_lines += ["### Lesson velocity (Oracle)", f"_feed unavailable: lesson_velocity ({e})_"]
    forensics_lines.append("")

    try:
        forensics_lines.extend(_feed_macro_briefing())
    except Exception as e:
        forensics_lines += ["### Macro (FRED, via market briefing)", f"_feed unavailable: macro_briefing ({e})_"]

    lines.extend(forensics_lines)

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
