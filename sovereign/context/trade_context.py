#!/usr/bin/env python3
"""
sovereign/context/trade_context.py
==================================
The Trade Context Packet — everything this repo already knows, assembled at the
moment a trade is being considered, for one instrument.

Why this exists
---------------
The research record of this desk lives in files that no decision ever reads.
The Alexandrian Library (63 sealed historical episodes across 10 volumes) is
queried only by the ICT path, which is itself unproven (permutation p=0.52).
`research/EDGE_LEDGER.md` and `research/HYPOTHESIS_LESSONS.md` — the verdicts of
120 hypotheses, including every door that is already closed — are read by humans
and by nothing else. So a trade gets taken with none of it attached.

This module closes that gap on the *read* side only. It assembles:

  1. Alexandrian Library  — nearest historical analogues to today's tape, with
                            similarity, threat level, size modifier, advisory.
  2. Edge ledger          — what is CONFIRMED, what is FRAGILE, what is null,
                            filtered to the instrument in question.
  3. Closed doors         — hypotheses already refuted for this instrument, so
                            neither a human nor an agent re-proposes them.
  4. Lessons              — the one-line lesson from each relevant hypothesis.
  5. Risk constitution    — the ratified caps that bind any size.
  6. Live record          — recent logged decisions for this instrument.

Hard constraints honoured
-------------------------
* READ-ONLY. Writes nothing, imports nothing from the execution path, and is
  imported by nothing in it. Safe under the shadow/execution-path freeze.
* Lives in `sovereign/`, so the ICT/sovereign isolation law is untouched
  (the law forbids ict/ -> sovereign/, not the reverse, and this module is on
  neither side of that bridge).
* NO SILENT MOCKING. Every source that cannot be read reports itself as
  unavailable with the reason, in `packet['warnings']`. A degraded packet
  always says so.

Usage
-----
    python3 -m sovereign.context.trade_context EURUSD
    python3 -m sovereign.context.trade_context EURUSD --json
    python3 -m sovereign.context.trade_context SPY --json --no-library

    from sovereign.context.trade_context import build_packet
    packet = build_packet("GBPJPY")
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

EDGE_LEDGER      = REPO / "research" / "EDGE_LEDGER.md"
LESSONS          = REPO / "research" / "HYPOTHESIS_LESSONS.md"
MAGNUM_OPUS      = REPO / "research" / "MAGNUM_OPUS.md"
RISK_CONSTITUTION = REPO / "RISK_CONSTITUTION.md"
HYP_LEDGER_JSON  = REPO / "data" / "agent" / "hypothesis_ledger.json"
DECISION_LOG_DIR = REPO / "data" / "agent" / "decision_logs" / "live"
PRICE_CACHE      = REPO / "data" / "_price_cache"

# Verdict words that mean "this door is shut". Sourced from the ledger's own
# vocabulary, not invented here.
CLOSED_VERDICTS = {
    "NOT_SIGNIFICANT", "REJECTED", "REJECTED_OOS", "NOT_ROBUST", "FAIL",
    "GRAVEYARD", "KILLED", "NULL", "POLICY_FAILS", "REFUTED", "ROLLED_BACK",
    "NOT AN EDGE", "DATA_INSUFFICIENT", "NOT PROVEN",
}
LIVE_VERDICTS = {"CONFIRMED", "VALID_EDGE", "LIVE", "MEASURED"}

# The five pairs the constitution names, plus the one it excludes.
_FX_CODES = {
    "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
    "SEK", "NOK", "MXN", "ZAR", "TRY", "BRL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Instrument identity
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Instrument:
    raw: str
    symbol: str            # normalised, e.g. EURUSD / SPY
    asset_class: str       # forex | equity_or_etf
    base: Optional[str] = None
    quote: Optional[str] = None
    tokens: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tokens"] = sorted(self.tokens)
        return d


def parse_instrument(raw: str) -> Instrument:
    """Normalise EURUSD / EUR_USD / EUR/USD / eurusd=x / SPY into one identity.

    The token set is what every downstream filter matches against; it is
    deliberately generous (base, quote, pair, asset class, strategy family) so a
    lesson written about "carry" reaches a query about EURUSD.
    """
    s = re.sub(r"[^A-Za-z]", "", raw).upper()
    s = re.sub(r"X$", "", s) if s.endswith("=X") else s
    base = quote = None
    if len(s) == 6 and s[:3] in _FX_CODES and s[3:] in _FX_CODES:
        base, quote = s[:3], s[3:]
        asset_class = "forex"
        tokens = {s.lower(), base.lower(), quote.lower(),
                  f"{base}/{quote}".lower(), f"{base}_{quote}".lower(),
                  "forex", "fx", "carry", "currency", "pair", "rate differential"}
    else:
        asset_class = "equity_or_etf"
        tokens = {s.lower(), "equity", "etf", "stock"}
    return Instrument(raw=raw, symbol=s, asset_class=asset_class,
                      base=base, quote=quote, tokens=tokens)


def _matches(text: str, tokens: set[str]) -> bool:
    low = text.lower()
    return any(t in low for t in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Alexandrian Library
# ─────────────────────────────────────────────────────────────────────────────

# Below this cosine similarity a 23-feature match is indistinguishable from
# noise. Same floor the ICT bridge adopted on 2026-07-20 after forensics showed
# a regime label firing on sim=0.067. Kept identical so the two consumers never
# disagree about what "a match" means.
SIMILARITY_FLOOR = 0.30


def _price_arrays(offline: bool = False) -> tuple[Any, Any, Any, Any, list[str]]:
    """SPY / VIX / GLD / DXY daily closes. Live first, stale cache second.

    Returns (spy, vix, gold, dxy, notes). Any array may be None. `notes` records
    exactly which source served each series so the packet never implies the
    Library saw fresh data when it saw a 2024 cache.
    """
    notes: list[str] = []
    spy = vix = gold = dxy = None

    if not offline:
        try:
            import yfinance as yf
            import pandas as pd

            def _live(ticker: str, period: str):
                df = yf.download(ticker, period=period, interval="1d",
                                 progress=False, auto_adjust=True)
                if df is None or len(df) == 0:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df["Close"].dropna().values

            spy  = _live("SPY", "400d")
            vix  = _live("^VIX", "60d")
            gold = _live("GLD", "400d")
            dxy  = _live("DX-Y.NYB", "60d")
            served = [n for n, a in (("SPY", spy), ("VIX", vix),
                                     ("GLD", gold), ("DXY", dxy)) if a is not None]
            if served:
                notes.append(f"live yfinance: {', '.join(served)}")
        except Exception as exc:               # network down, yfinance missing
            notes.append(f"live fetch failed ({type(exc).__name__}: {exc})")

    # Cache fallback, per series, only where live gave nothing. The cache is
    # known-stale (last bar 2024-12-30 at time of writing), so every series it
    # serves is labelled STALE with its last bar — the packet must never imply
    # the Library saw today's tape when it saw a two-year-old one.
    if spy is None or gold is None:
        try:
            import pandas as pd
        except Exception as exc:
            notes.append(f"cache unavailable (pandas: {type(exc).__name__})")
        else:
            for tick in ("SPY", "GLD"):
                if tick == "SPY" and spy is not None:
                    continue
                if tick == "GLD" and gold is not None:
                    continue
                path = PRICE_CACHE / f"{tick}.parquet"
                if not path.exists():
                    notes.append(f"{tick}: no live data and no cache at {path.name}")
                    continue
                try:
                    df = pd.read_parquet(path)
                    arr = df["close"].dropna().values
                    last = str(df.index[-1])[:10]
                except Exception as exc:
                    notes.append(f"{tick} cache read failed ({type(exc).__name__}: {exc})")
                    continue
                if tick == "SPY":
                    spy = arr
                else:
                    gold = arr
                notes.append(f"{tick} from STALE cache (last bar {last})")

    return spy, vix, gold, dxy, notes


def library_context(offline: bool = False) -> dict[str, Any]:
    """Query the Alexandrian Library for today's nearest historical analogues.

    Returns a dict that always has `available` and, when False, `reason`.
    """
    out: dict[str, Any] = {"available": False, "reason": None, "source_notes": []}
    try:
        from sovereign.risk.alexandrian_library import AlexandrianLibrary
    except Exception as exc:
        out["reason"] = f"library import failed ({type(exc).__name__}: {exc})"
        return out

    try:
        lib = AlexandrianLibrary()
    except Exception as exc:
        out["reason"] = f"library load failed ({type(exc).__name__}: {exc})"
        return out

    out["n_patterns"] = int(getattr(lib, "n_patterns", 0) or 0)
    out["n_volumes"] = int(getattr(lib, "n_volumes", 0) or 0)
    if not out["n_volumes"]:
        out["reason"] = "library has no volumes loaded (models/alexandrian_library.json missing or empty)"
        return out

    spy, vix, gold, dxy, notes = _price_arrays(offline=offline)
    out["source_notes"] = notes
    if spy is None or len(spy) < 200:
        n = 0 if spy is None else len(spy)
        out["reason"] = f"insufficient SPY history for a 23-feature match ({n} bars, need 200)"
        return out

    try:
        insight = lib.query(spy, vix_prices=vix, gold_prices=gold, dxy_prices=dxy)
    except Exception as exc:
        out["reason"] = f"query raised ({type(exc).__name__}: {exc})"
        return out

    sim = float(insight.primary_similarity)
    out.update({
        "available": True,
        "primary_regime": insight.primary_regime,
        "primary_volume": insight.primary_volume,
        "similarity": round(sim, 3),
        "above_floor": sim >= SIMILARITY_FLOOR,
        "similarity_floor": SIMILARITY_FLOOR,
        "threat_level": insight.threat_level,
        "threat_score": round(float(insight.threat_score), 3),
        "size_modifier": round(float(insight.size_modifier), 3),
        "converging_signal": bool(insight.converging_signal),
        "advisory": insight.advisory,
        "action_summary": insight.action_summary,
        "precedents": [
            {
                "entry_id": m.entry_id,
                "volume": m.volume,
                "label": m.label,
                "date": m.date,
                "similarity": round(float(m.similarity), 3),
                "severity": int(m.severity),
                "outcome": m.outcome,
                "tags": list(m.tags),
            }
            for m in (insight.top_matches or [])
        ],
    })
    if not out["above_floor"]:
        out["advisory"] = (
            f"similarity {sim:.3f} is below the {SIMILARITY_FLOOR:.2f} noise floor — "
            f"treat the regime label as ABSTAIN, not as a call. "
            f"(original advisory: {insight.advisory})"
        )
        out["primary_regime"] = "UNKNOWN"
        out["size_modifier"] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2-4. The written record: edges, closed doors, lessons
# ─────────────────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    """Repo-relative path for messages, safe for paths outside the repo."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("cannot read %s: %s", path, exc)
        return None


def _md_rows(text: str) -> list[list[str]]:
    """Every pipe-table data row in a markdown document, cells stripped."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(set(c) <= set("-: ") for c in cells):   # separator row
            continue
        rows.append(cells)
    return rows


def _declassify(status: str) -> str:
    """CONFIRMED / CLOSED / OPEN from a free-text status cell."""
    # NB: strip only * and backtick. Underscore is PART of the verdict names
    # (NOT_SIGNIFICANT, IC_ONLY, REJECTED_OOS) — treating it as markdown
    # emphasis silently reclassified every closed door as OPEN.
    up = re.sub(r"[*`]", "", status).upper()
    for v in CLOSED_VERDICTS:
        if v in up:
            return "CLOSED"
    for v in LIVE_VERDICTS:
        if v in up:
            return "CONFIRMED"
    if "FRAGILE" in up or "IC_ONLY" in up or "MAGNITUDE_ONLY" in up or "CASCADE" in up:
        return "CLOSED"
    return "OPEN"


def edge_context(inst: Instrument) -> dict[str, Any]:
    """The EDGE_LEDGER status table, split into what is live and what is shut."""
    out: dict[str, Any] = {"available": False, "reason": None,
                           "confirmed": [], "closed": [], "open": [],
                           "relevant_to_instrument": []}
    text = _read(EDGE_LEDGER)
    if text is None:
        out["reason"] = f"{_rel(EDGE_LEDGER)} unreadable"
        return out
    out["available"] = True

    header_seen = False
    for cells in _md_rows(text):
        if not header_seen:
            if cells[0].lower().strip() in ("what", "id"):
                header_seen = True
                continue
            continue
        if len(cells) < 2:
            continue
        claim = re.sub(r"[*`]", "", cells[0]).strip()
        status = cells[1]
        evidence = cells[2] if len(cells) > 2 else ""
        row = {
            "claim": claim,
            "status": re.sub(r"[*`]", "", status).strip(),
            "evidence": re.sub(r"[*`]", "", evidence).strip(),
            "verdict": _declassify(status),
        }
        out[{"CONFIRMED": "confirmed", "CLOSED": "closed", "OPEN": "open"}[row["verdict"]]].append(row)
        if _matches(f"{claim} {status} {evidence}", inst.tokens):
            out["relevant_to_instrument"].append(row)

    # The explicit retraction paragraph is part of the record and must travel
    # with the packet — it is the one claim the operator asked never to be
    # repeated as an edge.
    m = re.search(r"\*\*Retracted, explicitly.*?\n\n", text, re.S)
    out["retraction"] = re.sub(r"\s+", " ", m.group(0)).strip() if m else None
    return out


def lessons_context(inst: Instrument, limit: int = 12) -> dict[str, Any]:
    """Per-hypothesis lessons that mention this instrument or its family."""
    out: dict[str, Any] = {"available": False, "reason": None,
                           "headline_lessons": [], "matched": []}
    text = _read(LESSONS)
    if text is None:
        out["reason"] = f"{_rel(LESSONS)} unreadable"
        return out
    out["available"] = True

    # The numbered "ten things" block is the compressed form of the whole file.
    block = re.search(r"## The ten things.*?\n(.*?)\n## ", text, re.S)
    if block:
        for line in block.group(1).splitlines():
            line = line.strip()
            if re.match(r"^\d+\.\s", line):
                out["headline_lessons"].append(
                    re.sub(r"[*`]", "", re.sub(r"^\d+\.\s*", "", line)).strip())

    for cells in _md_rows(text):
        if len(cells) < 4 or not cells[0].upper().startswith("HYP"):
            continue
        hid, tested, verdict, lesson = cells[0], cells[1], cells[2], cells[3]
        if not _matches(f"{tested} {lesson}", inst.tokens):
            continue
        out["matched"].append({
            "id": re.sub(r"[*`]", "", hid).strip(),
            "tested": re.sub(r"[*`]", "", tested).strip(),
            "verdict": re.sub(r"[*`]", "", verdict).strip(),
            "lesson": re.sub(r"[*`]", "", lesson).strip(),
            "door": _declassify(verdict),
        })
    out["n_matched_total"] = len(out["matched"])
    out["matched"] = out["matched"][:limit]
    return out


def closed_doors(edges: dict[str, Any], lessons: dict[str, Any]) -> list[dict[str, str]]:
    """Everything already refuted for this instrument, deduplicated.

    This is the list an agent reads FIRST: it is the set of ideas that look
    attractive and have already been paid for.
    """
    doors: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in edges.get("relevant_to_instrument", []):
        if row["verdict"] != "CLOSED":
            continue
        key = row["claim"].lower()
        if key in seen:
            continue
        seen.add(key)
        doors.append({"source": "EDGE_LEDGER", "what": row["claim"],
                      "verdict": row["status"], "evidence": row["evidence"]})
    for row in lessons.get("matched", []):
        if row["door"] != "CLOSED":
            continue
        key = row["tested"].lower()
        if key in seen:
            continue
        seen.add(key)
        doors.append({"source": row["id"], "what": row["tested"],
                      "verdict": row["verdict"], "evidence": row["lesson"]})
    return doors


# ─────────────────────────────────────────────────────────────────────────────
# 5. Risk constitution
# ─────────────────────────────────────────────────────────────────────────────

def risk_context() -> dict[str, Any]:
    """The ratified caps, quoted from RISK_CONSTITUTION.md rather than restated.

    Numbers drift; this reads them at call time so the packet can never carry a
    stale cap that someone hardcoded months ago.
    """
    out: dict[str, Any] = {"available": False, "reason": None, "clauses": []}
    text = _read(RISK_CONSTITUTION)
    if text is None:
        out["reason"] = f"{_rel(RISK_CONSTITUTION)} unreadable"
        return out
    out["available"] = True
    for para in re.split(r"\n\s*\n", text):
        if "%" not in para or "**" not in para:
            continue
        if not re.search(r"\*\*\d+(\.\d+)?%\*\*", para):
            continue
        clean = re.sub(r"\s+", " ", re.sub(r"[*`]", "", para)).strip()
        if len(clean) > 400:
            clean = clean[:397] + "..."
        out["clauses"].append(clean)
    out["clauses"] = out["clauses"][:6]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live record
# ─────────────────────────────────────────────────────────────────────────────

def live_context(inst: Instrument, limit: int = 5) -> dict[str, Any]:
    """Recent logged decisions for this instrument + its ledger status."""
    out: dict[str, Any] = {"available": False, "reason": None,
                           "recent_decisions": [], "ledger_entries": []}
    notes: list[str] = []

    if DECISION_LOG_DIR.exists():
        rows: list[dict[str, Any]] = []
        for p in sorted(DECISION_LOG_DIR.glob("*.jsonl"))[-3:]:
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    blob = json.dumps(rec).lower()
                    if inst.symbol.lower() in blob or (
                        inst.base and f"{inst.base}_{inst.quote}".lower() in blob
                    ):
                        rows.append(rec)
            except Exception as exc:
                notes.append(f"{p.name}: {type(exc).__name__}")
        out["available"] = True
        out["recent_decisions"] = [
            {k: v for k, v in r.items()
             if k in ("timestamp", "trade_id", "pair", "symbol", "direction",
                      "decision", "conviction_score", "outcome", "r_multiple",
                      "veto_reason")}
            for r in rows[-limit:]
        ]
        out["n_matched_total"] = len(rows)
    else:
        out["reason"] = f"{_rel(DECISION_LOG_DIR)} does not exist"

    try:
        entries = json.loads(HYP_LEDGER_JSON.read_text(encoding="utf-8"))
        if isinstance(entries, list):
            for e in entries:
                blob = json.dumps(e).lower()
                if _matches(blob, inst.tokens):
                    out["ledger_entries"].append({
                        "id": e.get("id") or e.get("hypothesis_id"),
                        "name": e.get("name") or e.get("title"),
                        "status": e.get("status"),
                        "result": (e.get("result") or "")[:220],
                    })
            out["ledger_entries"] = out["ledger_entries"][-limit:]
    except Exception as exc:
        notes.append(f"hypothesis_ledger.json: {type(exc).__name__}: {exc}")

    if notes:
        out["notes"] = notes
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The packet
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeContextPacket:
    instrument: dict[str, Any]
    asof: str
    library: dict[str, Any]
    edges: dict[str, Any]
    lessons: dict[str, Any]
    closed_doors: list[dict[str, str]]
    risk: dict[str, Any]
    live: dict[str, Any]
    warnings: list[str]
    completeness: str          # FULL | PARTIAL | DEGRADED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_packet(instrument: str, offline: bool = False,
                 with_library: bool = True) -> TradeContextPacket:
    """Assemble every local knowledge source for one instrument. Read-only."""
    inst = parse_instrument(instrument)
    warnings: list[str] = []

    if with_library:
        lib = library_context(offline=offline)
    else:
        lib = {"available": False, "reason": "skipped (--no-library)"}
    if not lib.get("available"):
        warnings.append(f"Alexandrian Library unavailable: {lib.get('reason')}")
    elif not lib.get("above_floor", True):
        warnings.append(
            f"Library match {lib['similarity']:.3f} is below the "
            f"{SIMILARITY_FLOOR:.2f} noise floor — regime call is ABSTAIN.")
    for note in lib.get("source_notes", []):
        if "STALE" in note or "failed" in note:
            warnings.append(f"Library input: {note}")

    edges = edge_context(inst)
    if not edges.get("available"):
        warnings.append(f"Edge ledger unavailable: {edges.get('reason')}")

    lessons = lessons_context(inst)
    if not lessons.get("available"):
        warnings.append(f"Hypothesis lessons unavailable: {lessons.get('reason')}")

    risk = risk_context()
    if not risk.get("available"):
        warnings.append(f"Risk constitution unavailable: {risk.get('reason')}")

    live = live_context(inst)
    if not live.get("available") and live.get("reason"):
        warnings.append(f"Live decision log unavailable: {live.get('reason')}")

    sources_ok = sum(bool(s.get("available")) for s in (lib, edges, lessons, risk, live))
    completeness = "FULL" if sources_ok == 5 else ("PARTIAL" if sources_ok >= 3 else "DEGRADED")

    return TradeContextPacket(
        instrument=inst.to_dict(),
        asof=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        library=lib,
        edges=edges,
        lessons=lessons,
        closed_doors=closed_doors(edges, lessons),
        risk=risk,
        live=live,
        warnings=warnings,
        completeness=completeness,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def to_markdown(p: TradeContextPacket) -> str:
    inst = p.instrument
    L = p.library
    out: list[str] = []
    a = out.append

    a(f"# Trade context — {inst['symbol']}  ({inst['asset_class']})")
    a(f"_as of {p.asof} · completeness **{p.completeness}**_")
    a("")

    if p.warnings:
        a("## ⚠ Degradation")
        for w in p.warnings:
            a(f"- {w}")
        a("")

    a("## 1. Historical analogue — Alexandrian Library")
    if not L.get("available"):
        a(f"- UNAVAILABLE — {L.get('reason')}")
    else:
        a(f"- **{L['primary_regime']}** (vol {L.get('primary_volume', '?')}, "
          f"sim {L['similarity']:.3f}{'' if L['above_floor'] else ' — BELOW FLOOR'})")
        a(f"- threat **{L['threat_level']}** ({L['threat_score']:.3f}) · "
          f"size modifier **{L['size_modifier']:.2f}×** · "
          f"converging: {'yes' if L['converging_signal'] else 'no'}")
        a(f"- advisory: {L['advisory']}")
        if L.get("precedents"):
            a("")
            a("| precedent | volume | date | sim | sev | what followed |")
            a("|---|---|---|---|---|---|")
            for m in L["precedents"][:5]:
                a(f"| {m['label']} | {m['volume'].replace('VOLUME_', '')} | {m['date']} | "
                  f"{m['similarity']:.2f} | {m['severity']} | {m['outcome']} |")
    a("")

    a("## 2. Closed doors — already paid for, do not re-propose")
    if not p.closed_doors:
        a("- none recorded for this instrument")
    for d in p.closed_doors[:10]:
        a(f"- **{d['what']}** — {d['verdict']}  _({d['source']})_")
    a("")

    a("## 3. What is actually confirmed")
    conf = p.edges.get("confirmed", [])
    if not conf:
        a("- nothing on the ledger is CONFIRMED")
    for r in conf[:6]:
        a(f"- **{r['claim']}** — {r['status']}  _({r['evidence']})_")
    if p.edges.get("retraction"):
        a("")
        a(f"> {p.edges['retraction']}")
    a("")

    a("## 4. Lessons that apply here")
    for l in p.lessons.get("headline_lessons", [])[:4]:
        a(f"- {l}")
    matched = p.lessons.get("matched", [])
    if matched:
        a("")
        for m in matched[:8]:
            a(f"- `{m['id']}` {m['verdict']} — {m['lesson']}")
    a("")

    a("## 5. Binding risk caps")
    if not p.risk.get("available"):
        a(f"- UNAVAILABLE — {p.risk.get('reason')}")
    for c in p.risk.get("clauses", [])[:4]:
        a(f"- {c}")
    a("")

    a("## 6. This instrument's live record")
    live = p.live
    if live.get("recent_decisions"):
        a(f"- {live.get('n_matched_total', 0)} logged decisions matched; last "
          f"{len(live['recent_decisions'])}:")
        for d in live["recent_decisions"]:
            a(f"  - `{json.dumps(d, default=str)}`")
    else:
        a("- no logged decisions matched this instrument")
    for e in live.get("ledger_entries", [])[:4]:
        a(f"- ledger `{e['id']}` **{e['status']}** — {e['name']}")
    a("")

    a("---")
    a("_Read-only packet. Assembled from this repo's own record; no external "
      "opinion, no recommendation, no size. Absence of a closed door is not "
      "evidence of an edge._")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trade_context",
        description="Assemble this repo's full research record for one instrument, read-only.")
    ap.add_argument("instrument", help="EURUSD, GBP_JPY, SPY, ...")
    ap.add_argument("--json", action="store_true", help="emit the raw packet as JSON")
    ap.add_argument("--offline", action="store_true",
                    help="skip live price fetch; use the local cache only")
    ap.add_argument("--no-library", action="store_true",
                    help="skip the Alexandrian Library query entirely (fastest)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    packet = build_packet(args.instrument, offline=args.offline,
                          with_library=not args.no_library)
    if args.json:
        json.dump(packet.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(to_markdown(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
