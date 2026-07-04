"""experience/precedents.py — L2a: structured precedent retrieval over the Library.

Two sources merge at retrieval time: the canonical Alexandrian Library
(sovereign.risk.alexandrian_library.ALL_ENTRIES — READ-ONLY, this module never imports the
writer path or touches models/alexandrian_library.json) and the lived annex
(experience.library_annex — this repo's own history). No honest board->23-dim feature mapping
exists yet: the canonical library's features come from raw SPY/VIX/gold price arrays via
market_memory.extract_features, and the sentiment board has no such projection. Retrieval here
is therefore v1 STRUCTURED matching over tags/severity/date/mechanism-keyword only — every
citation this produces stamps method:"structured_v1" so nothing downstream mistakes it for a
real feature-space similarity match. The honest feature path is LIB-FEAT-1, a separate ticket
(blocked on a network dependency inside launchd).

Board access degrades gracefully: sovereign.sentiment.store.connect(read_only=True) raises
when the DuckDB file is absent or locked by a concurrent writer (e.g. a feeder mid-run) —
week_board_extremes catches that and returns [] rather than raising, so the Sunday review
never dies because the sentiment pipeline happened to be busy. (build_review wraps this
whole module in one more try/except on top, per NON-NEGOTIABLE-adjacent build rule: the
review must never die on library errors, full stop.)
"""
from __future__ import annotations

from config.loader import params
from experience import library_annex
from sovereign.risk.alexandrian_library import ALL_ENTRIES

# L2b (experience/precedent_service.py) decision-time lookups stay a stub until this is
# flipped in config/parameters.yml (experience.precedents.decision_time_enabled) — a
# deliberate, logged config change, not a silent capability upgrade. Nothing in the live
# decision path reads this today.
decision_time_enabled = bool(
    (params.get("experience") or {}).get("precedents", {}).get("decision_time_enabled", False)
)

# Board-extreme -> mechanism-keyword mapping. ONE explicit, reviewable dict (plan requirement):
# this IS the entire "what does an extreme board reading mean" translation layer. Extend here,
# never by scattering keyword logic through the retrieval path below. Columns are
# sentiment_board_state fields (sovereign/sentiment/store.py::SCHEMA); "high"/"low" are
# inclusive thresholds, "eq" is exact-match (for the categorical vix_regime column).
BOARD_EXTREME_TAGS: dict[str, dict] = {
    "vix_level":       {"high": (30,    {"volatility", "vix_spike", "risk_off"}),
                         "low":  (13,    {"low_vol", "complacency", "carry"})},
    "vix_regime":      {"eq":   ("SPIKE", {"volatility", "vix_spike"})},
    "econ_surprise_z": {"high": (1.5,   {"inflation", "hawkish_surprise"}),
                         "low":  (-1.5,  {"growth_scare", "dovish_surprise"})},
    "cot_net_pct":     {"high": (0.90,  {"crowded_long", "positioning_extreme"}),
                         "low":  (0.10,  {"crowded_short", "positioning_extreme"})},
    "vrp_pct":         {"high": (0.90,  {"vol_rich", "volatility"}),
                         "low":  (0.10,  {"vol_cheap", "complacency"})},
    "macro_curve":     {"low":  (0.0,   {"yield_curve_inversion", "recession_risk"})},
}


def week_board_extremes(start: str, end: str) -> list[dict]:
    """Board rows in [start, end] that breach any BOARD_EXTREME_TAGS threshold.

    Returns [{"date", "pair", "tags": [...]}] for rows with >=1 breach; [] if the DB is
    absent/locked, the table doesn't exist yet, or nothing breaches. This is the only place
    in this module that talks to sovereign.sentiment.store — imported lazily so tests can
    monkeypatch sovereign.sentiment.store.connect directly to simulate DB-absent/locked.
    """
    try:
        from sovereign.sentiment.store import connect
        con = connect(read_only=True)
    except Exception:
        return []
    try:
        df = con.execute(
            "SELECT * FROM sentiment_board_state WHERE date BETWEEN ? AND ? ORDER BY date, pair",
            [str(start)[:10], str(end)[:10]],
        ).df()
    except Exception:
        return []
    finally:
        con.close()
    if df.empty:
        return []

    extremes = []
    for _, row in df.iterrows():
        tags: set[str] = set()
        for col, rules in BOARD_EXTREME_TAGS.items():
            val = row.get(col)
            if val is None or val != val:              # NaN-safe (matches journal.py convention)
                continue
            for direction, (thr, tagset) in rules.items():
                if direction == "high" and val >= thr:
                    tags |= tagset
                elif direction == "low" and val <= thr:
                    tags |= tagset
                elif direction == "eq" and val == thr:
                    tags |= tagset
        if tags:
            extremes.append({"date": str(row["date"])[:10], "pair": row["pair"], "tags": sorted(tags)})
    return extremes


def week_context_tags(rows: list[dict], atts: list[dict], extremes: list[dict]) -> set[str]:
    """Union of context signals for the week: decision engines/thesis kinds, attribution
    classes/overlays, and board-extreme tags. This is the query vector for find_precedents.
    """
    tags: set[str] = set()
    for r in rows:
        if r.get("engine"):
            tags.add(str(r["engine"]))
        thesis = r.get("thesis") or {}
        if thesis.get("kind"):
            tags.add(str(thesis["kind"]))
    for a in atts:
        if a.get("cls"):
            tags.add(str(a["cls"]))
        for o in (a.get("overlays") or []):
            tags.add(str(o))
    for e in extremes:
        tags |= set(e.get("tags", []))
    return tags


def _entry_tags(label: str, tags: list[str]) -> set[str]:
    return {str(t).lower() for t in tags} | {str(label).lower()}


def find_precedents(context_tags: set[str], top_k: int = 3) -> list[dict]:
    """Structured v1 retrieval: score canonical + annex entries by tag-overlap with
    context_tags, return the top_k highest-scoring (ties broken by most-recent event_date).

    Each result: {entry_id, source (canonical|annex), label, event_date, why, what_followed,
    outcome_days, severity, matched_tags, score} — ready for citations.make_citation.
    """
    if not context_tags:
        return []
    ctx = {str(t).lower() for t in context_tags}
    candidates: list[dict] = []

    for e in ALL_ENTRIES:
        matched = ctx & _entry_tags(e.label, e.tags)
        if not matched:
            continue
        candidates.append({
            "entry_id": e.entry_id, "source": "canonical", "label": e.label,
            "event_date": e.date, "why": e.description, "what_followed": e.outcome,
            "outcome_days": e.outcome_days, "severity": e.severity,
            "matched_tags": sorted(matched), "score": len(matched),
        })

    for a in library_annex.read_entries():
        matched = ctx & _entry_tags(a.label, a.tags)
        if not matched:
            continue
        candidates.append({
            "entry_id": a.entry_id, "source": "annex", "label": a.label,
            "event_date": a.date, "why": a.description, "what_followed": a.outcome,
            "outcome_days": a.outcome_days, "severity": a.severity,
            "matched_tags": sorted(matched), "score": len(matched),
        })

    candidates.sort(key=lambda c: (c["score"], c["event_date"]), reverse=True)
    return candidates[:top_k]
