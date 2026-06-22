#!/usr/bin/env python3
"""
Oracle lesson-lifecycle tracker — measures how long Oracle takes to LEARN a lesson.

A lesson theme is born as a daily `candidate_lesson`, gets reworked across many days (the
testable_rule sharpens — e.g. bars_since_signal `>50` -> `>100` -> `SWEEP`-gated), and is
"learned" two ways:
  * CONVERGED — the testable_rule stops changing for CONVERGE_N consecutive days (the frequent,
    day-to-day learning signal).
  * CODIFIED  — formally validated into I_am_a_good_trader.md (rare; the gold standard).

Themes are grouped across days by `reasoning_component_targeted` (union-find over its component
tokens — chains green_matches <-> "...green_matches/hist_hr" <-> hist_hr into one evolving theme).

Reads:  data/oracle/reflections/*.json, data/oracle/validations/*.json,
        data/oracle/proven_research.json (proven_lessons: discovered/codified/health).
Writes: data/oracle/lesson_velocity.json  (current lesson, in-progress themes, codified lessons,
        velocity aggregates) — instrumentation for the dashboard. Read-only w.r.t. trading.

Runnable standalone:  python3 scripts/lesson_lifecycle.py
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFLECTIONS = ROOT / "data" / "oracle" / "reflections"
VALIDATIONS = ROOT / "data" / "oracle" / "validations"
PROVEN = ROOT / "data" / "oracle" / "proven_research.json"
OUT = ROOT / "data" / "oracle" / "lesson_velocity.json"

CONVERGE_N = 3        # consecutive identical-rule days to call a theme "converged"
DORMANT_DAYS = 7      # not seen in this many days (and not converged) -> dormant
# generic tokens that appear across many lessons — dropped so grouping keys on the rare component
STOPWORDS = {
    "score", "the", "and", "or", "of", "current", "consensus", "indic", "indicator",
    "historical", "hit", "rate", "value", "signal", "count", "context", "entry", "regime",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    toks = re.split(r"[^a-zA-Z0-9]+", s.lower())
    return {t for t in toks if len(t) >= 3 and not t.isdigit() and t not in STOPWORDS}


def _load_reflections() -> list[dict]:
    out = []
    if not REFLECTIONS.exists():
        return out
    for f in sorted(REFLECTIONS.iterdir()):
        if not DATE_RE.match(f.name):
            continue
        d = _read(f)
        cl = (d or {}).get("reflection", {}).get("candidate_lesson")
        if not isinstance(cl, dict):
            continue
        out.append({
            "date": f.name.replace(".json", ""),
            "lesson_text": cl.get("lesson_text") or "",
            "testable_rule": (cl.get("testable_rule") or "").strip(),
            "component": cl.get("reasoning_component_targeted") or "",
            "tokens": _tokens(cl.get("reasoning_component_targeted")),
            "sample_needed": cl.get("sample_needed"),
        })
    return out


def _cluster(refs: list[dict]) -> list[list[int]]:
    """Union-find: two reflections share a theme if their component token sets intersect."""
    parent = list(range(len(refs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            if refs[i]["tokens"] and refs[i]["tokens"] & refs[j]["tokens"]:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(refs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _theme(members: list[dict], today: str) -> dict:
    members = sorted(members, key=lambda m: m["date"])
    first, last = members[0]["date"], members[-1]["date"]
    rules = [m["testable_rule"] for m in members]
    n_revisions = len({r for r in rules if r})
    # stable run from the end: consecutive most-recent days with identical rule
    stable = 1
    for k in range(len(rules) - 1, 0, -1):
        if rules[k] and rules[k] == rules[k - 1]:
            stable += 1
        else:
            break
    first_stable_date = members[len(members) - stable]["date"] if stable >= 1 else last
    converged = stable >= CONVERGE_N
    dormant = (not converged) and _days(last, today) > DORMANT_DAYS
    status = "converged" if converged else ("dormant" if dormant else "forming")
    return {
        "components": sorted({t for m in members for t in m["tokens"]}),
        "component_label": members[-1]["component"],
        "first_seen": first,
        "last_seen": last,
        "days_forming": _days(first, last) if status == "converged" else _days(first, today),
        "n_revisions": n_revisions,
        "n_days_appeared": len(members),
        "stable_days": stable,
        "status": status,
        "time_to_converge_days": _days(first, first_stable_date) if converged else None,
        "latest_lesson_text": members[-1]["lesson_text"],
        "latest_rule": members[-1]["testable_rule"],
        "history": [{"date": m["date"], "rule": m["testable_rule"]} for m in members],
    }


def _codified() -> list[dict]:
    d = _read(PROVEN) or {}
    out = []
    for e in d.get("proven_lessons", []) or []:
        disc, cod = e.get("discovered"), e.get("codified")
        ttl = None
        try:
            if disc and cod:
                ttl = _days(disc, cod)
        except Exception:
            ttl = None
        out.append({
            "id": e.get("id"),
            "lesson": (e.get("lesson") or "")[:120],
            "discovered": disc,
            "codified": cod,
            "days_to_codify": ttl,
            "health": e.get("health"),
            "delta_sharpe": e.get("delta_sharpe"),
        })
    return out


def _per_week(dates: list[str]) -> float | None:
    ds = sorted(d for d in dates if d)
    if len(ds) < 2:
        return None
    span_weeks = max(_days(ds[0], ds[-1]) / 7.0, 1e-9)
    return round(len(ds) / span_weeks, 2)


def build() -> dict:
    today = date.today().isoformat()
    refs = _load_reflections()
    groups = _cluster(refs)
    themes = [_theme([refs[i] for i in g], today) for g in groups]
    themes.sort(key=lambda t: t["last_seen"], reverse=True)

    current = themes[0] if themes else None
    codified = _codified()

    ttc = [t["time_to_converge_days"] for t in themes if t["time_to_converge_days"] is not None]
    ttk = [c["days_to_codify"] for c in codified if c["days_to_codify"] is not None]

    velocity = {
        "n_themes_tracked": len(themes),
        "n_themes_forming": sum(1 for t in themes if t["status"] == "forming"),
        "n_themes_converged": sum(1 for t in themes if t["status"] == "converged"),
        "n_codified": len(codified),
        "median_days_to_converge": round(statistics.median(ttc), 1) if ttc else None,
        "mean_days_to_converge": round(statistics.mean(ttc), 1) if ttc else None,
        "median_days_to_codify": round(statistics.median(ttk), 1) if ttk else None,
        "mean_days_to_codify": round(statistics.mean(ttk), 1) if ttk else None,
        "codified_per_week": _per_week([c["codified"] for c in codified]),
        "themes_formed_per_week": _per_week([t["first_seen"] for t in themes]),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": today,
        "current_lesson": current,
        "in_progress_themes": [t for t in themes if t["status"] in ("forming", "converged")][:12],
        "codified_lessons": codified,
        "velocity": velocity,
        "config": {"converge_n_days": CONVERGE_N, "dormant_days": DORMANT_DAYS},
        "provenance": {
            "kind": "lesson_lifecycle_instrumentation",
            "note": "Measures Oracle's learning process (time-to-converge / time-to-codify, frequency). "
                    "Read-only over Oracle's own reflections — not a trading input.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    p = build()
    v = p["velocity"]
    cur = p["current_lesson"]
    print(f"lesson_velocity: {v['n_themes_tracked']} themes "
          f"({v['n_themes_forming']} forming, {v['n_themes_converged']} converged), "
          f"{v['n_codified']} codified")
    print(f"  velocity: converge median {v['median_days_to_converge']}d | codify median "
          f"{v['median_days_to_codify']}d | themes/wk {v['themes_formed_per_week']}")
    if cur:
        print(f"  CURRENT lesson [{cur['component_label']}]: {cur['status']}, "
              f"forming {cur['days_forming']}d, {cur['n_revisions']} revisions")
        print(f"    \"{cur['latest_lesson_text'][:90]}\"")
