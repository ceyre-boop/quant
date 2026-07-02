#!/usr/bin/env python3
"""Shadow divergence analyzer — scores the L2 exit-manager shadow log against
audit/divergence_spec.md (the pre-registered contract).

L1: determinism replay of every logged decision through the shared decide_exit.
L2: input-parity replay on yfinance bars (the backtest price source).
Watchdog: heartbeat / loadedness / plist integrity / SHADOW_MODE.

Read-only observer: imports decide_exit and the manager's config constants;
never touches broker code, never writes to data/exec/. Reports land in
audit/reports/ (tracked); escalations follow the pulse_check conventions.

The spec is the single source of thresholds (yaml audit-spec fence). If a run
finds ≠1 fence, or the analyzer crashes, an URGENT escalation is still written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from sovereign.forex.exit_machine import (  # noqa: E402
    BarContext, ExitDecision, PositionState, decide_exit,
)
import sovereign.execution.forex_exit_manager as fxm  # noqa: E402

SPEC_PATH = _ROOT / "audit" / "divergence_spec.md"
MESSAGES_PATH = _ROOT / "data" / "agent" / "messages_to_colin.json"
PARAM_LOG = _ROOT / "data" / "agent" / "param_change_log.jsonl"
CACHE_DIR = _ROOT / "data" / "audit_cache"
FENCE_RE = re.compile(r"^```yaml audit-spec\n(.*?)^```", re.S | re.M)
DIR_MAP = {"LONG": 1, "SHORT": -1}
EMOJI = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}

# Standing-incident register — mirrors spec §5 (additions require a spec §10 entry).
STANDING_INCIDENTS = {
    "2026-06-29": "INC-2026-06-29 first fire crashed (import path); manual recovery same day",
    "2026-06-30": "INC-2026-06-30 manual+launchd double-step pre-guard "
                  "(param_change_log 2026-06-30T22:32Z); standing hold_count +1 offset",
}


# ── spec ──────────────────────────────────────────────────────────────────────

def load_spec(path: Path = SPEC_PATH):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    fences = FENCE_RE.findall(raw.decode("utf-8"))
    if len(fences) != 1:
        raise RuntimeError(f"spec must contain exactly one 'yaml audit-spec' fence, found {len(fences)}")
    spec = yaml.safe_load(fences[0])
    return spec, sha, spec["spec_version"]


# ── shadow log ────────────────────────────────────────────────────────────────

def parse_shadow_log(path: Path):
    """Whole file, JSON per line. One malformed TRAILING line tolerated (writer
    overlap); malformed mid-file is LOG_CORRUPT (a C5-class problem)."""
    records, problems = [], []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            rec["_line"] = i + 1
            records.append(rec)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                problems.append({"kind": "TRAILING_PARTIAL_LINE", "line": i + 1})
            else:
                problems.append({"kind": "LOG_CORRUPT", "line": i + 1})
    return records, problems


def is_normal(rec: dict) -> bool:
    return rec.get("action") not in ("SKIP", "SKIP_DUPLICATE")


def group_trades(records):
    trades: dict[str, list] = {}
    for r in records:
        trades.setdefault(str(r.get("trade_id")), []).append(r)
    return trades


# ── L1 determinism replay ────────────────────────────────────────────────────

@dataclass
class L1Result:
    passed: int = 0
    failed: list = field(default_factory=list)
    boundary: int = 0
    ambiguous: int = 0
    continuity: list = field(default_factory=list)
    action_unverifiable: int = 0

    @property
    def scored(self):
        return self.passed + len(self.failed)

    @property
    def pass_rate(self):
        return 1.0 if self.scored == 0 else self.passed / self.scored


def _trail(direction, best, worst, atr_pct, mult):
    atr = max(atr_pct, 1e-6)
    if direction == 1:
        return best - (mult * atr * best)
    return worst + (mult * atr * worst)


def _tol_price(spec, close, trailing_mult) -> float:
    """Price-scaled replay tolerance (spec v2): the atr round-6 quantization
    contributes close·mult·quantum, which dominates at JPY price levels."""
    return float(spec["tol_price_abs"]) + \
        abs(close) * trailing_mult * float(spec["tol_atr_quantum"]) * float(spec["tol_safety"])


def replay_l1(records, spec, incidents) -> L1Result:
    res = L1Result()
    ht_default = int(spec["hold_today_assumed"])
    ht_alts = [int(x) for x in spec["hold_today_ambiguous_alternates"]]
    amb_min_hc = int(spec["cb_refresh_ambiguity_min_hold_count"])

    for tid, recs in group_trades(records).items():
        last_stop = None
        prev = None
        for r in recs:
            if not is_normal(r):
                continue  # guard/skip records advance nothing
            direction = DIR_MAP[r["direction"]]
            cfg = fxm.cfg_for_pair(r["pair"])
            tol = _tol_price(spec, float(r["close"]), cfg.trailing_atr_mult)
            band = tol
            # continuity (whole-log chaining); incidents key on RUN date or bar date (spec v2)
            if prev is not None:
                if r["bar_date"] == prev["bar_date"] or r["hold_count"] != prev["hold_count"] + 1:
                    keys = (r["bar_date"], prev["bar_date"],
                            str(r.get("run_ts", ""))[:10], str(prev.get("run_ts", ""))[:10])
                    hit = next((k for k in keys if k in incidents), None)
                    res.continuity.append({"trade_id": tid, "bar_date": r["bar_date"],
                                           "class": "C3" if hit else "C5",
                                           "detail": f"hold_count {prev['hold_count']}→{r['hold_count']}",
                                           "incident": incidents.get(hit) if hit else None})
            if last_stop is None and r["hold_count"] == 1:
                last_stop = float(r["initial_stop"])

            state = PositionState(direction=direction, stop_price=float(r["initial_stop"]),
                                  best_price=float(r["best_price"]), worst_price=float(r["worst_price"]),
                                  hold_count=int(r["hold_count"]) - 1, hold_limit=int(r["hold_limit"]))
            ambiguous_cb = r["signal"] == direction and r["hold_count"] >= amb_min_hc
            candidates = ht_alts if ambiguous_cb else [ht_default]
            outcome = None
            for ht in candidates:
                bar = BarContext(close=float(r["close"]), atr_pct=float(r["atr_pct"]),
                                 signal=int(r["signal"]), hold_today=ht, donchian_exit_low=math.nan)
                out = decide_exit(state, bar, cfg)
                if out.decision.name == r["decision"]:
                    outcome = out
                    if ht != ht_default:
                        res.ambiguous += 1
                    break
            check = outcome or decide_exit(state, BarContext(float(r["close"]), float(r["atr_pct"]),
                                                             int(r["signal"]), ht_default, math.nan), cfg)
            trail = _trail(direction, check.state.best_price, check.state.worst_price,
                           float(r["atr_pct"]), cfg.trailing_atr_mult)
            # boundary detection on the decisive comparisons
            close = float(r["close"])
            margins = [abs(close - float(r["initial_stop"])), abs(close - trail)]
            if last_stop is not None:
                margins.append(abs(trail - last_stop))
            if min(margins) <= band:
                res.boundary += 1
            else:
                fails = []
                if outcome is None:
                    fails.append(f"decision {check.decision.name} != logged {r['decision']}")
                else:
                    if abs(trail - float(r["trail_price"])) > tol:
                        fails.append(f"trail {trail:.6f} vs logged {r['trail_price']}")
                    if int(outcome.state.hold_count) != int(r["hold_count"]):
                        fails.append("hold_count post-state mismatch")
                    if int(outcome.reentry_signal) != int(r.get("reentry_signal", 0)):
                        fails.append("reentry_signal mismatch")
                    # action expectation (needs chained last_stop)
                    if r["decision"] != "HOLD":
                        if r["action"] != "CLOSE":
                            fails.append(f"action {r['action']} for decision {r['decision']}")
                    elif last_stop is None:
                        res.action_unverifiable += 1
                    else:
                        tightens = (direction == 1 and trail > last_stop) or \
                                   (direction == -1 and trail < last_stop)
                        expect = "AMEND_STOP" if (cfg.trailing_atr_mult > 0 and tightens) else "HOLD"
                        if r["action"] != expect and abs(trail - last_stop) > band:
                            fails.append(f"action {r['action']} expected {expect}")
                    wa = r.get("would_amend_stop_to")
                    if r["action"] == "AMEND_STOP" and wa is not None and abs(trail - float(wa)) > tol:
                        fails.append(f"would_amend {wa} vs trail {trail:.6f}")
                if fails:
                    res.failed.append({"trade_id": tid, "bar_date": r["bar_date"],
                                       "line": r.get("_line"), "problems": fails})
                else:
                    res.passed += 1
            if r["action"] == "AMEND_STOP" and r.get("would_amend_stop_to") is not None:
                last_stop = float(r["would_amend_stop_to"])
            prev = r
    return res


# ── L2 input-parity replay ───────────────────────────────────────────────────

@dataclass
class L2Result:
    status: str = "OK"           # OK | SKIPPED_OFFLINE
    scored: int = 0
    matched: int = 0
    mismatches: list = field(default_factory=list)
    delta_stats: dict = field(default_factory=dict)
    shifts: dict = field(default_factory=dict)
    seed_modes: dict = field(default_factory=dict)
    revisions: int = 0

    @property
    def match_rate(self):
        return 1.0 if self.scored == 0 else self.matched / self.scored


def _yf_symbol(oanda_pair: str) -> str:
    return oanda_pair.replace("_", "") + "=X"


def fetch_yf_bars(pair: str, start: str, end: str, offline: bool):
    """yfinance daily bars, parquet-pinned. Returns (df, revisions) or (None, 0)."""
    import pandas as pd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{_yf_symbol(pair).replace('=', '_')}.parquet"
    fresh, revisions = None, 0
    if not offline:
        try:
            import yfinance as yf
            fresh = yf.download(_yf_symbol(pair), start=start, end=end,
                                auto_adjust=True, progress=False)
            if isinstance(fresh.columns, pd.MultiIndex):
                fresh.columns = fresh.columns.get_level_values(0)
            fresh = fresh.dropna()
        except Exception:
            fresh = None
    if fresh is not None and len(fresh):
        if cache.exists():
            old = pd.read_parquet(cache)
            common = old.index.intersection(fresh.index)
            if len(common):
                revisions = int((old.loc[common, "Close"].round(8)
                                 != fresh.loc[common, "Close"].round(8)).sum())
        fresh.to_parquet(cache)
        return fresh, revisions
    if cache.exists():
        return pd.read_parquet(cache), 0
    return None, 0


def replay_l2(records, spec, offline: bool, incidents) -> L2Result:
    from sovereign.forex.signal_engine import ForexSignalEngine
    res = L2Result()
    tol_close = float(spec["c1_min_close_delta"])
    tol_atr = float(spec["c1_min_atr_delta"])
    normals = [r for r in records if is_normal(r)]
    if not normals:
        return res
    dmin = min(r["bar_date"] for r in normals)
    dmax = max(r["bar_date"] for r in normals)
    start = (date.fromisoformat(dmin) - timedelta(days=45)).isoformat()
    end = (date.fromisoformat(dmax) + timedelta(days=3)).isoformat()

    pairs = sorted({r["pair"] for r in normals})
    bars, atrs = {}, {}
    for p in pairs:
        df, rev = fetch_yf_bars(p, start, end, offline)
        if df is None or not len(df):
            res.status = "SKIPPED_OFFLINE"
            return res
        res.revisions += rev
        bars[p] = df
        atrs[p] = ForexSignalEngine._compute_atr_pct(df["Close"], df)

    # per-pair constant date shift minimizing median |Δclose|
    for p in pairs:
        precs = [r for r in normals if r["pair"] == p]
        best_shift, best_med = 0, None
        for shift in (-1, 0, 1):
            deltas = []
            for r in precs:
                d = (date.fromisoformat(r["bar_date"]) + timedelta(days=shift)).isoformat()
                if d in bars[p].index.strftime("%Y-%m-%d"):
                    yc = float(bars[p].loc[d, "Close"])
                    deltas.append(abs(yc - float(r["close"])))
            if deltas:
                deltas.sort()
                med = deltas[len(deltas) // 2]
                if best_med is None or med < best_med:
                    best_med, best_shift = med, shift
        res.shifts[p] = best_shift

    for tid, recs in group_trades(records).items():
        nrecs = [r for r in recs if is_normal(r)]
        if not nrecs:
            continue
        first = nrecs[0]
        p = first["pair"]
        direction = DIR_MAP[first["direction"]]
        cfg = fxm.cfg_for_pair(p)
        rc = round(float(first["close"]), 5)
        if first["hold_count"] == 1:
            entry = float(first["worst_price"]) if float(first["best_price"]) == rc else float(first["best_price"])
            if float(first["best_price"]) == float(first["worst_price"]):
                entry = float(first["close"])
            state = PositionState(direction, float(first["initial_stop"]), entry, entry, 0,
                                  int(first["hold_limit"]))
            todo = nrecs
            res.seed_modes[tid] = "ENTRY"
        else:
            state = PositionState(direction, float(first["initial_stop"]),
                                  float(first["best_price"]), float(first["worst_price"]),
                                  int(first["hold_count"]), int(first["hold_limit"]))
            todo = nrecs[1:]
            res.seed_modes[tid] = "FIRST_RECORD"
        closed = False
        dstats = res.delta_stats.setdefault(p, {"dclose": [], "dclose_rel": [], "datr": []})
        for r in todo:
            if closed:
                break
            d = (date.fromisoformat(r["bar_date"]) + timedelta(days=res.shifts[p])).isoformat()
            idx = bars[p].index.strftime("%Y-%m-%d")
            if d not in list(idx):
                res.mismatches.append({"trade_id": tid, "bar_date": r["bar_date"], "class": "C2",
                                       "detail": f"no yfinance bar at shifted date {d}"})
                continue
            yclose = float(bars[p].loc[d, "Close"])
            yatr = float(atrs[p].loc[d])
            dclose, datr = abs(yclose - float(r["close"])), abs(yatr - float(r["atr_pct"]))
            dstats["dclose"].append(dclose)
            dstats["dclose_rel"].append(dclose / max(abs(float(r["close"])), 1e-9))
            dstats["datr"].append(datr)
            bar = BarContext(yclose, yatr, int(r["signal"]), int(spec["hold_today_assumed"]), math.nan)
            out = decide_exit(state, bar, cfg)
            res.scored += 1
            if out.decision.name == r["decision"]:
                res.matched += 1
            else:
                cls, detail = "C5", f"yf {out.decision.name} vs logged {r['decision']}"
                for variant, (c, a) in {"close": (float(r["close"]), yatr),
                                        "atr": (yclose, float(r["atr_pct"])),
                                        "both": (float(r["close"]), float(r["atr_pct"]))}.items():
                    o2 = decide_exit(state, BarContext(c, a, int(r["signal"]),
                                                       int(spec["hold_today_assumed"]), math.nan), cfg)
                    if o2.decision.name == r["decision"] and (dclose > tol_close or datr > tol_atr):
                        cls, detail = "C1", f"substituting OANDA {variant} reproduces it (Δclose={dclose:.2e}, Δatr={datr:.2e})"
                        break
                if cls == "C5":
                    hit = next((k for k in (r["bar_date"], str(r.get("run_ts", ""))[:10])
                                if k in incidents), None)
                    if hit:
                        cls, detail = "C3", incidents[hit]
                res.mismatches.append({"trade_id": tid, "bar_date": r["bar_date"], "class": cls,
                                       "detail": detail})
            state = out.state
            if out.decision != ExitDecision.HOLD or r["decision"] != "HOLD":
                closed = True  # stop scoring at first CLOSE on either side
    return res


# ── watchdog ─────────────────────────────────────────────────────────────────

def check_scheduler(spec, now: datetime | None = None) -> dict:
    now = now or datetime.now().astimezone()
    out = {"checks": {}, "events": []}
    state_file = _ROOT / spec["state_file"]
    fire_h, fire_m = (int(x) for x in spec["exit_manager_fire_local"].split(":"))
    grace = int(spec["staleness_grace_min"])
    is_weekday = now.weekday() < 5
    due = now.replace(hour=fire_h, minute=fire_m, second=0, microsecond=0) + timedelta(minutes=grace)
    if is_weekday and now >= due:
        fresh = state_file.exists() and \
            datetime.fromtimestamp(state_file.stat().st_mtime).date() == now.date()
        out["checks"]["heartbeat_today"] = bool(fresh)
        if not fresh:
            out["events"].append(("URGENT", "SHADOW_STALE",
                                  f"exit-manager state file not updated by {spec['exit_manager_fire_local']}+{grace}m"))
    else:
        out["checks"]["heartbeat_today"] = "not-due"
    try:
        loaded = subprocess.run(["launchctl", "list", spec["exit_manager_label"]],
                                capture_output=True).returncode == 0
    except Exception:
        loaded = False
    out["checks"]["job_loaded"] = loaded
    if not loaded:
        out["events"].append(("URGENT", "SHADOW_UNLOADED", f"{spec['exit_manager_label']} not loaded"))
    tracked = _ROOT / spec["exit_manager_plist_tracked"]
    installed = Path.home() / "Library" / "LaunchAgents" / f"{spec['exit_manager_label']}.plist"
    same = installed.exists() and tracked.exists() and \
        hashlib.sha256(installed.read_bytes()).hexdigest() == hashlib.sha256(tracked.read_bytes()).hexdigest()
    out["checks"]["plist_matches_tracked"] = bool(same)
    if not same:
        out["events"].append(("IMPORTANT", "PLIST_DRIFT",
                              "installed exit-manager plist != tracked scripts copy (clobber recurrence?)"))
    out["checks"]["shadow_mode_true"] = fxm.SHADOW_MODE is True
    if fxm.SHADOW_MODE is not True:
        out["events"].append(("URGENT", "SHADOW_MODE_OFF", "SHADOW_MODE is not True"))
    return out


# ── incidents ────────────────────────────────────────────────────────────────

def load_incidents() -> dict:
    inc = dict(STANDING_INCIDENTS)
    if PARAM_LOG.exists():
        for line in PARAM_LOG.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "exit_manager" in str(e.get("type", "")) or "launchagent" in str(e.get("type", "")):
                day = str(e.get("ts", ""))[:10]
                if day:
                    inc.setdefault(day, f"param_change_log {e.get('type')} {e.get('ts')}")
    return inc


# ── gates / report / escalation ──────────────────────────────────────────────

def evaluate_gates(l1: L1Result, l2: L2Result, records, spec) -> dict:
    normals = [r for r in records if is_normal(r)]
    weekdays = {r["bar_date"] for r in normals}
    c5 = len([m for m in l2.mismatches if m["class"] == "C5"]) + \
        len([c for c in l1.continuity if c["class"] == "C5"])
    live = len([r for r in records if r.get("mode") == "LIVE"])
    coverage_met = (len(weekdays) >= int(spec["min_scored_weekdays"])
                    and len(normals) >= int(spec["min_scored_records"]))
    if l2.status != "OK":
        l2_status, l2_note = "PENDING", l2.status
    elif l2.match_rate >= float(spec["l2_decision_match_min"]):
        l2_status, l2_note = "PASS", f"{l2.match_rate:.4f} ({l2.matched}/{l2.scored})"
    elif not coverage_met:
        # spec §6: gates are scored at window close — a below-floor running rate
        # with insufficient coverage is not yet a verdict.
        l2_status, l2_note = "PENDING", f"{l2.match_rate:.4f} ({l2.matched}/{l2.scored}) below floor at current n"
    else:
        l2_status, l2_note = "FAIL", f"{l2.match_rate:.4f} ({l2.matched}/{l2.scored})"
    gates = {
        "l1_pass_rate_100": ("PASS" if l1.pass_rate == 1.0 and not l1.failed else "FAIL",
                             f"{l1.pass_rate:.4f} ({l1.passed}/{l1.scored})"),
        "c5_zero": ("PASS" if c5 == 0 else "FAIL", str(c5)),
        "l2_match": (l2_status, l2_note),
        "coverage": ("PASS" if coverage_met else "PENDING",
                     f"{len(weekdays)} weekdays / {len(normals)} records"),
        "no_live_records": ("PASS" if live <= int(spec["live_records_allowed"]) else "FAIL", str(live)),
    }
    if any(v[0] == "FAIL" for v in gates.values()):
        overall = "NO-GO"
    elif any(v[0] == "PENDING" for v in gates.values()):
        overall = "NOT-YET"
    else:
        overall = "GO"
    return {"gates": gates, "overall": overall, "c5": c5, "live_records": live}


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def write_report(report_date: str, spec, sha, l1, l2, watchdog, gate, records, problems, dry_run=False):
    rep_dir = _ROOT / spec["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "?"
    normals = [r for r in records if is_normal(r)]
    day_recs = [r for r in records if r.get("bar_date") == report_date or
                str(r.get("run_ts", ""))[:10] == report_date]
    tax = {"C1": 0, "C2": 0, "C3": 0, "C4": len([r for r in records if r.get("action") == "SKIP"]), "C5": 0}
    for m in l2.mismatches:
        tax[m["class"]] = tax.get(m["class"], 0) + 1
    for c in l1.continuity:
        tax[c["class"]] = tax.get(c["class"], 0) + 1
    delta_summary = {p: {"dclose_p50": _pct(d["dclose"], .5), "dclose_p90": _pct(d["dclose"], .9),
                         "dclose_rel_p50": _pct(d.get("dclose_rel", []), .5),
                         "datr_p50": _pct(d["datr"], .5), "n": len(d["dclose"])}
                     for p, d in l2.delta_stats.items()}
    body = {
        "report_date": report_date, "spec_version": spec["spec_version"], "spec_sha256": sha,
        "analyzer_commit": head, "records_total": len(records), "records_normal": len(normals),
        "records_today": len(day_recs), "parse_problems": problems,
        "watchdog": watchdog["checks"],
        "l1": {"passed": l1.passed, "failed": l1.failed, "boundary": l1.boundary,
               "ambiguous": l1.ambiguous, "pass_rate": l1.pass_rate,
               "continuity_violations": l1.continuity, "action_unverifiable": l1.action_unverifiable},
        "l2": {"status": l2.status, "scored": l2.scored, "matched": l2.matched,
               "match_rate": l2.match_rate, "mismatches": l2.mismatches, "shifts": l2.shifts,
               "seed_modes": l2.seed_modes, "delta_stats": delta_summary, "cache_revisions": l2.revisions},
        "taxonomy": tax, "gate": gate,
    }
    if dry_run:
        return body
    (rep_dir / f"{report_date}.json").write_text(json.dumps(_json_safe(body), indent=2) + "\n")
    md = [f"# Shadow Divergence Report — {report_date}", "",
          f"spec v{spec['spec_version']} `{sha[:16]}…` · analyzer `{head}` · records {len(records)} "
          f"({len(normals)} normal, {len(day_recs)} today)", "",
          "## Watchdog", *(f"- {k}: {v}" for k, v in watchdog["checks"].items()), "",
          "## L1 — determinism",
          f"- pass {l1.passed} / fail {len(l1.failed)} / boundary {l1.boundary} / ambiguous {l1.ambiguous} "
          f"→ rate {l1.pass_rate:.4f}",
          f"- continuity violations: {len(l1.continuity)}",
          *(f"  - {c['bar_date']} {c['trade_id']}: {c['detail']} → {c['class']} ({c.get('incident')})"
            for c in l1.continuity), "",
          "## L2 — input parity",
          f"- status {l2.status} · match {l2.matched}/{l2.scored} = {l2.match_rate:.4f} · shifts {l2.shifts} "
          f"· seeds {l2.seed_modes} · cache revisions {l2.revisions}",
          *(f"  - {m['bar_date']} {m['trade_id']} [{m['class']}] {m['detail']}" for m in l2.mismatches[:20]),
          f"- per-pair deltas: {json.dumps(_json_safe(delta_summary))}", "",
          "## Taxonomy (cumulative)",
          "| C1 | C2 | C3 | C4 | C5 |", "|---|---|---|---|---|",
          f"| {tax['C1']} | {tax['C2']} | {tax['C3']} | {tax['C4']} | {tax['C5']} |", "",
          "## Gate scorecard → **" + gate["overall"] + "**",
          *(f"- {k}: {v[0]} ({v[1]})" for k, v in gate["gates"].items()), ""]
    (rep_dir / f"{report_date}.md").write_text("\n".join(md))
    idx_path = rep_dir / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[report_date] = {"overall": gate["overall"], "l1_rate": l1.pass_rate,
                        "l2_rate": l2.match_rate if l2.status == "OK" else None, "c5": gate["c5"]}
    idx_path.write_text(json.dumps(_json_safe(idx), indent=2, sort_keys=True) + "\n")
    return body


def escalate(events, report_date, cap, dry_run=False):
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
        text = f"[AUDIT] {typ}: {msg}"
        if any(text in e and report_date in e for e in existing) or \
           any(e.startswith(f"[AUDIT] {typ}") and report_date in e for e in existing):
            continue
        msgs.insert(0, {"id": f"audit-{report_date}-{typ.lower()}",
                        "priority": priority, "emoji": EMOJI[priority],
                        "text": f"{text} ({report_date})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "read": False, "source": "shadow_audit"})
        added += 1
    del msgs[int(cap):]
    if isinstance(doc, dict):
        doc["messages"] = msgs
        doc["last_updated"] = datetime.now(timezone.utc).isoformat()
    else:
        doc = msgs
    tmp = tempfile.NamedTemporaryFile("w", dir=MESSAGES_PATH.parent, delete=False, suffix=".tmp")
    tmp.write(json.dumps(_json_safe(doc), indent=2) + "\n")
    tmp.close()
    Path(tmp.name).replace(MESSAGES_PATH)
    return added


# ── main ─────────────────────────────────────────────────────────────────────

def run_for_date(report_date: str, offline=False, dry_run=False, log_path: Path | None = None):
    spec, sha, _ = load_spec()
    log_path = log_path or (_ROOT / spec["shadow_log"])
    records, problems = parse_shadow_log(log_path) if log_path.exists() else ([], [{"kind": "LOG_MISSING"}])
    records = [r for r in records
               if str(r.get("bar_date") or r.get("run_ts", ""))[:10] <= report_date]
    incidents = load_incidents()
    l1 = replay_l1(records, spec, incidents)
    l2 = replay_l2(records, spec, offline, incidents)
    watchdog = check_scheduler(spec)
    gate = evaluate_gates(l1, l2, records, spec)
    events = list(watchdog["events"])
    if l1.failed:
        events.append(("URGENT", "L1_DETERMINISM_FAIL", f"{len(l1.failed)} record(s) not reproducible"))
    if gate["c5"] > int(spec["c5_allowed"]):
        events.append(("URGENT", "UNEXPLAINED_DIVERGENCE", f"C5 count {gate['c5']}"))
    if gate["live_records"] > 0:
        events.append(("URGENT", "LIVE_RECORD_SEEN", f"{gate['live_records']} mode=LIVE record(s)"))
    for kind in {p["kind"] for p in problems} & {"LOG_CORRUPT"}:
        events.append(("URGENT", "LOG_CORRUPT", "malformed mid-file shadow-log line"))
    for p, d in l2.delta_stats.items():
        med = _pct(d.get("dclose_rel", []), .5)
        if med is not None and med > float(spec["close_delta_warn_rel"]):
            events.append(("IMPORTANT", "PRICE_SOURCE_DRIFT",
                           f"{p} median |Δclose|/close {med:.5f}"))
    body = write_report(report_date, spec, sha, l1, l2, watchdog, gate, records, problems, dry_run)
    n = escalate(events, report_date, cap=int(spec["messages_cap"]), dry_run=dry_run)
    print(f"[shadow_audit] {report_date}: gate={gate['overall']} L1={l1.pass_rate:.3f} "
          f"L2={l2.match_rate if l2.status == 'OK' else l2.status} C5={gate['c5']} escalations={n}")
    return body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="analyze as of today")
    ap.add_argument("--date", help="YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.date and ".." in a.date:
            d0, d1 = a.date.split("..")
            d = date.fromisoformat(d0)
            while d <= date.fromisoformat(d1):
                if d.weekday() < 5:
                    run_for_date(d.isoformat(), a.offline, a.dry_run)
                d += timedelta(days=1)
        else:
            run_for_date(a.date or date.today().isoformat(), a.offline, a.dry_run)
        return 0
    except Exception as exc:  # crash handler still escalates
        try:
            escalate([("URGENT", "AUDIT_CRASH", f"{type(exc).__name__}: {exc}")],
                     date.today().isoformat(), cap=50, dry_run=a.dry_run)
        finally:
            raise


if __name__ == "__main__":
    sys.exit(main())
