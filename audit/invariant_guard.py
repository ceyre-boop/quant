#!/usr/bin/env python3
"""Adversarial Invariant Guard — the standing Layer-4 integrity check.

Sibling to audit/shadow_divergence.py (which proves backtest≡live parity). This
one asserts the integrity invariants nothing else guards, scored against the
pre-registered contract audit/invariants_spec.md:

  I1  Oracle reflection purity   — no probe/forbidden record enters cognition.
  I2  No rogue OANDA writes       — no forbidden/sentinel fill in the ledgers.
  I3  Forbidden-pair guard        — no forbidden pair anywhere in the recent path.

Read-only observer. Imports nothing from sovereign/execution/ or the OANDA
bridge; the only sovereign import is the pure pair-normalizer. It reimplements
the probe / insane-risk / fill↔decision heuristics INDEPENDENTLY of the code it
audits — an adversarial check that imports the audited code shares its blind
spots. Reports land in audit/reports/ (tracked); escalations follow the same
messages_to_colin.json convention as the shadow audit. If the spec fence is
missing or the checker crashes, an URGENT escalation is still written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

# The ONLY sovereign import: pure venue-agnostic pair normalization (shared
# vocabulary, no execution surface). Everything adversarial is reimplemented below.
from sovereign.intelligence.decision_logger import _norm_pair  # noqa: E402

SPEC_PATH = _ROOT / "audit" / "invariants_spec.md"
MESSAGES_PATH = _ROOT / "data" / "agent" / "messages_to_colin.json"
FENCE_RE = re.compile(r"^```yaml audit-spec\n(.*?)^```", re.S | re.M)
EMOJI = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}
# Reflection selection: outcomes that are NOT trade results (mirror reflect_cycle:280).
_NON_OUTCOMES = (None, "OPEN", "EXPIRED")


# ── spec ──────────────────────────────────────────────────────────────────────

def load_spec(path: Path = SPEC_PATH):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    fences = FENCE_RE.findall(raw.decode("utf-8"))
    if len(fences) != 1:
        raise RuntimeError(
            f"spec must contain exactly one 'yaml audit-spec' fence, found {len(fences)}")
    spec = yaml.safe_load(fences[0])
    return spec, sha, int(spec["spec_version"])


# ── independent heuristics (mirror scripts/backfill_decision_records.py, by design) ──

def _insane_risk(entry, stop, frac: float) -> bool:
    """A probe/proof-of-life fill: non-positive levels, or a stop further than
    `frac` of entry away (a real FX stop is a fraction of a percent, not ~100%).
    Independent reimplementation of _is_test_fill's core test."""
    try:
        e, s = float(entry or 0.0), float(stop or 0.0)
    except (TypeError, ValueError):
        return True
    if e <= 0 or s <= 0:
        return True
    return abs(e - s) > frac * e


def _fill_is_probe(fill: dict, frac: float) -> bool:
    if _insane_risk(fill.get("fill_price"), fill.get("stop_price"), frac):
        return True
    try:
        if float(fill.get("units") or 0) in (1.0, -1.0):  # 1-unit sentinel probes
            return True
    except (TypeError, ValueError):
        pass
    return False


def _record_is_probe(rec: dict, frac: float) -> bool:
    if rec.get("test_fill") is True:
        return True
    return _insane_risk(rec.get("entry_level"), rec.get("stop_loss"), frac)


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    s = str(ts).replace("Z", "+00:00")
    for cand in (s, s[:19]):
        try:
            dt = datetime.fromisoformat(cand)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ── data access (read-only) ───────────────────────────────────────────────────

def load_decisions(spec) -> list[dict]:
    d = _ROOT / spec["decision_log_dir"]
    recs = []
    if d.exists():
        for f in sorted(d.glob("decisions_*.jsonl")):
            recs.extend(_load_jsonl(f))
    return recs


def load_fills(spec):
    """Returns (fills, stale_paths, missing). Each fill tagged with _src path."""
    fills, present, stale = [], [], []
    grace = timedelta(hours=float(spec["fills_staleness_grace_hours"]))
    now = datetime.now(timezone.utc)
    for rel in spec["fills_paths"]:
        p = _ROOT / rel
        if not p.exists():
            continue
        present.append(rel)
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if now - mtime > grace:
            stale.append(rel)
        for fill in _load_jsonl(p):
            fill["_src"] = rel
            fills.append(fill)
    return fills, present, stale


def _in_window(ts, cutoff: datetime) -> bool:
    dt = _parse_ts(ts)
    if dt is None:
        return True  # unparseable → inspect, don't silently drop (mirror reflect_cycle)
    return dt >= cutoff


# ── invariants ────────────────────────────────────────────────────────────────

@dataclass
class Findings:
    i1: list = field(default_factory=list)          # contaminated reflection records
    i2: list = field(default_factory=list)          # rogue/sentinel fills
    i3: list = field(default_factory=list)          # any forbidden pair in the path
    unknown_pairs: list = field(default_factory=list)  # soft: not-allowed, not-forbidden
    fills_stale: list = field(default_factory=list)
    fills_present: list = field(default_factory=list)


def evaluate(records, fills, fills_present, fills_stale, spec) -> Findings:
    forbidden = {_norm_pair(p) for p in spec["forbidden_pairs"]}
    allowed = {_norm_pair(p) for p in spec["allowed_pairs"]}
    probe_sources = {str(s).lower() for s in spec.get("probe_sources", [])}
    frac = float(spec["insane_risk_fraction"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(spec["window_days"]))
    f = Findings(fills_present=fills_present, fills_stale=fills_stale)

    # I1 — records that WOULD enter the Oracle reflection summary, then impure.
    for rec in records:
        if rec.get("outcome") in _NON_OUTCOMES:
            continue
        if not _in_window(rec.get("entry_timestamp"), cutoff):
            continue
        np = _norm_pair(rec.get("pair"))
        reasons = []
        if np in forbidden:
            reasons.append(f"forbidden pair {rec.get('pair')}")
        if str(rec.get("source", "")).lower() in probe_sources:
            reasons.append(f"probe source {rec.get('source')}")
        if _record_is_probe(rec, frac):
            reasons.append("probe/insane-risk levels")
        if reasons:
            f.i1.append({"pair": rec.get("pair"), "source": rec.get("source"),
                         "outcome": rec.get("outcome"), "r_realized": rec.get("r_realized"),
                         "entry_timestamp": rec.get("entry_timestamp"), "reasons": reasons})
        elif np and np not in allowed:
            f.unknown_pairs.append({"where": "decision", "pair": rec.get("pair"),
                                    "outcome": rec.get("outcome")})

    # I2 — recent fills that are forbidden or sentinel probes.
    for fill in fills:
        if not _in_window(fill.get("timestamp"), cutoff):
            continue
        np = _norm_pair(fill.get("pair"))
        reasons = []
        if np in forbidden:
            reasons.append(f"forbidden pair {fill.get('pair')}")
        if _fill_is_probe(fill, frac):
            reasons.append("sentinel/probe fill")
        if reasons:
            f.i2.append({"src": fill.get("_src"), "pair": fill.get("pair"),
                         "units": fill.get("units"), "stop_price": fill.get("stop_price"),
                         "timestamp": fill.get("timestamp"), "reasons": reasons})
        elif np and np not in allowed:
            f.unknown_pairs.append({"where": fill.get("_src"), "pair": fill.get("pair")})

    # I3 — ANY forbidden pair anywhere in the recent path (earlier tripwire than I1).
    for rec in records:
        if _in_window(rec.get("entry_timestamp"), cutoff) and _norm_pair(rec.get("pair")) in forbidden:
            f.i3.append({"where": "decision", "pair": rec.get("pair"),
                         "outcome": rec.get("outcome"), "ts": rec.get("entry_timestamp")})
    for fill in fills:
        if _in_window(fill.get("timestamp"), cutoff) and _norm_pair(fill.get("pair")) in forbidden:
            f.i3.append({"where": fill.get("_src"), "pair": fill.get("pair"),
                         "ts": fill.get("timestamp")})
    return f


def build_events(f: Findings, spec) -> list:
    events = []
    if len(f.i1) > int(spec["i1_contaminated_allowed"]):
        pairs = sorted({x["pair"] for x in f.i1})
        events.append(("URGENT", "I1_ORACLE_CONTAMINATION",
                       f"{len(f.i1)} contaminated record(s) enter the Oracle reflection "
                       f"summary (pairs: {', '.join(pairs)})"))
    if len(f.i2) > int(spec["i2_rogue_allowed"]):
        pairs = sorted({x["pair"] for x in f.i2})
        events.append(("URGENT", "I2_ROGUE_OANDA_WRITE",
                       f"{len(f.i2)} rogue/sentinel fill(s) in the ledgers (pairs: {', '.join(pairs)})"))
    if len(f.i3) > int(spec["i3_forbidden_allowed"]):
        pairs = sorted({x["pair"] for x in f.i3})
        events.append(("URGENT", "I3_FORBIDDEN_PAIR",
                       f"{len(f.i3)} forbidden-pair record(s)/fill(s) in the recent path "
                       f"(pairs: {', '.join(pairs)})"))
    if not f.fills_present:
        events.append(("IMPORTANT", "NO_FILLS_LEDGER",
                       "no fills ledger present — I2 cannot verify recent broker writes"))
    elif f.fills_stale and len(f.fills_stale) == len(f.fills_present):
        events.append(("IMPORTANT", "FILLS_STALE",
                       f"all fills ledgers stale ({', '.join(f.fills_stale)}) — recent writes unverifiable"))
    if f.unknown_pairs:
        pairs = sorted({x["pair"] for x in f.unknown_pairs if x.get("pair")})
        events.append(("IMPORTANT", "UNKNOWN_PAIR",
                       f"{len(f.unknown_pairs)} record(s)/fill(s) on non-allowed pair(s): {', '.join(pairs)}"))
    return events


def overall(f: Findings, spec) -> str:
    hard = (len(f.i1) > int(spec["i1_contaminated_allowed"]) or
            len(f.i2) > int(spec["i2_rogue_allowed"]) or
            len(f.i3) > int(spec["i3_forbidden_allowed"]))
    return "FAIL" if hard else "PASS"


# ── report / escalation ───────────────────────────────────────────────────────

def write_report(report_date, spec, sha, f: Findings, gate, dry_run=False):
    body = {
        "report_date": report_date, "spec_version": spec["spec_version"], "spec_sha256": sha,
        "gate": gate,
        "i1_oracle_contamination": {"count": len(f.i1), "records": f.i1},
        "i2_rogue_oanda_writes": {"count": len(f.i2), "fills": f.i2},
        "i3_forbidden_pairs": {"count": len(f.i3), "hits": f.i3},
        "soft": {"unknown_pairs": f.unknown_pairs,
                 "fills_present": f.fills_present, "fills_stale": f.fills_stale},
    }
    if dry_run:
        return body
    rep_dir = _ROOT / spec["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    stem = f"invariants_{report_date}"
    (rep_dir / f"{stem}.json").write_text(json.dumps(body, indent=2) + "\n")
    md = [f"# Adversarial Invariant Report — {report_date}", "",
          f"spec v{spec['spec_version']} `{sha[:16]}…` → **{gate}**", "",
          f"- **I1 Oracle contamination:** {len(f.i1)} (allowed {spec['i1_contaminated_allowed']})",
          *(f"  - {r['pair']} src={r['source']} outcome={r['outcome']} R={r['r_realized']} "
            f"@ {r['entry_timestamp']} — {'; '.join(r['reasons'])}" for r in f.i1[:20]),
          f"- **I2 rogue OANDA writes:** {len(f.i2)} (allowed {spec['i2_rogue_allowed']})",
          *(f"  - [{r['src']}] {r['pair']} units={r['units']} stop={r['stop_price']} "
            f"@ {r['timestamp']} — {'; '.join(r['reasons'])}" for r in f.i2[:20]),
          f"- **I3 forbidden pairs (broad):** {len(f.i3)} (allowed {spec['i3_forbidden_allowed']})",
          *(f"  - [{r['where']}] {r['pair']} @ {r.get('ts')}" for r in f.i3[:20]),
          "",
          f"soft: {len(f.unknown_pairs)} unknown-pair · fills present {f.fills_present or 'NONE'}"
          f" · stale {f.fills_stale or 'none'}", ""]
    (rep_dir / f"{stem}.md").write_text("\n".join(md))
    idx_path = rep_dir / "invariants_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[report_date] = {"overall": gate, "i1": len(f.i1), "i2": len(f.i2), "i3": len(f.i3)}
    idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True) + "\n")
    return body


def escalate(events, report_date, cap=50, dry_run=False):
    if not events or dry_run:
        return 0
    try:
        doc = json.loads(MESSAGES_PATH.read_text()) if MESSAGES_PATH.exists() else {"messages": []}
    except json.JSONDecodeError:
        doc = {"messages": []}
    msgs = doc.get("messages", doc if isinstance(doc, list) else [])
    existing = {m.get("text", "") for m in msgs}
    added = 0
    for priority, typ, msg in events:
        text = f"[INVARIANT] {typ}: {msg}"
        if any(e.startswith(f"[INVARIANT] {typ}") and report_date in e for e in existing):
            continue
        msgs.insert(0, {"id": f"invariant-{report_date}-{typ.lower()}",
                        "priority": priority, "emoji": EMOJI[priority],
                        "text": f"{text} ({report_date})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "read": False, "source": "invariant_guard"})
        added += 1
    del msgs[int(cap):]
    if isinstance(doc, dict):
        doc["messages"] = msgs
        doc["last_updated"] = datetime.now(timezone.utc).isoformat()
    else:
        doc = msgs
    MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=MESSAGES_PATH.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(doc, indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(MESSAGES_PATH)
    return added


# ── main ─────────────────────────────────────────────────────────────────────

def run_for_date(report_date: str, dry_run=False):
    spec, sha, _ = load_spec()
    records = load_decisions(spec)
    fills, present, stale = load_fills(spec)
    f = evaluate(records, fills, present, stale, spec)
    gate = overall(f, spec)
    events = build_events(f, spec)
    write_report(report_date, spec, sha, f, gate, dry_run)
    n = escalate(events, report_date, cap=int(spec["messages_cap"]), dry_run=dry_run)
    print(f"[invariant_guard] {report_date}: {gate} "
          f"I1={len(f.i1)} I2={len(f.i2)} I3={len(f.i3)} escalations={n}")
    return {"gate": gate, "i1": len(f.i1), "i2": len(f.i2), "i3": len(f.i3), "escalations": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="check as of today")
    ap.add_argument("--date", help="YYYY-MM-DD (report label)")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, write nothing")
    a = ap.parse_args(argv)
    report_date = a.date or datetime.now(timezone.utc).date().isoformat()
    try:
        res = run_for_date(report_date, dry_run=a.dry_run)
        # non-zero exit on a hard-fence failure so a CI/launchd wrapper can see teeth
        return 1 if res["gate"] == "FAIL" else 0
    except Exception as exc:  # crash still escalates
        try:
            escalate([("URGENT", "INVARIANT_CRASH", f"{type(exc).__name__}: {exc}")],
                     report_date, dry_run=a.dry_run)
        finally:
            raise


if __name__ == "__main__":
    sys.exit(main())
