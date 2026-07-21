"""Full system inventory — every first-party feature named, status DERIVED not asserted.

WHY THIS EXISTS
---------------
On 2026-07-20 two hand-written audit passes produced six false claims ("CS229 inert",
"NY scanner dead", "pca_compressor depended-on"). At ~974 first-party files, eyeballing
status manufactures those errors by the dozen. So here every status traces to evidence —
import-graph reachability, an AST parse result, a directory, or a header marker — and every
description comes from the author's own module docstring, never invented prose. Where a file
has no docstring the row says so.

This is `audit/claim_check.py` applied at scale: same AST import machinery, same read-only /
no-execution-path-import contract (`--self-test` asserts it), same isolation logic.

STATUS (precedence: RETIRED > ERROR > LIVE > ON-DEMAND > TEST-ONLY > DORMANT)
  RETIRED    attic/archive/trial/scratch/lab, or a DEPRECATED/RETIRED header
  ERROR      ast.parse fails — definitely broken
  LIVE       reachable from a SCHEDULED root (installed-plist entry scripts + agent-directive
             `-m` invocations + orchestrator + execute_daily). FIRING if its plist log is
             recent, else WIRED (reachable but no recent output — the loaded-but-dead class
             the claude-binary triage found)
  ON-DEMAND  a runnable `__main__` entry point (or code reached from one) that no scheduled
             job runs — a working tool invoked by hand (backtests, validation, research).
             405 of 973 files are entry points; without this state they mislabel as
             "test-only" or "dormant" when they are neither.
  TEST-ONLY  reachable only from tests/ — dead to the running system, alive to CI
  DORMANT    reached by nothing — no scheduled root, no manual entry point, no test

KNOWN BLIND SPOT
----------------
An AST walk cannot see dynamic imports. Four are known and HARDCODED below (DYNAMIC_EDGES);
each is labelled "manually asserted" in the output so the one place the graph is blind is
visible, not hidden. sovereign/risk/layers/* in particular is reached only via
risk_engine's import_module and would otherwise show a whole package as DORMANT.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import plistlib
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories whose contents are retired by location.
RETIRED_DIRS = {"attic", "archive", "trial", "scratch", "lab"}
# Header markers that retire a file regardless of location. NOTE: "# LOW-USE" is
# deliberately NOT here — it was added 2026-07-20 to mean "imported by a live path
# but minor, do not delete", the OPPOSITE of retired. Treating it as retired
# mislabelled all 11 CS229 modules, which are imported by sovereign/orchestrator.py.
RETIRED_MARKERS = ("# DEPRECATED", "# RETIRED:")
EXCLUDE_SUBSTR = (".venv", "__pycache__", ".claude/worktrees", "vendor/", "/node_modules/")

# Log recency (days) that separates LIVE-FIRING from LIVE-WIRED.
FIRING_DAYS = 4

# Dynamic import edges an AST walk cannot see. source_file -> [target module prefixes].
# Each is asserted from a hand-read of the source and surfaced in the report as manual.
DYNAMIC_EDGES = {
    "ict/pipeline.py": ["sovereign.intelligence.decision_logger"],       # :643 importlib
    "sovereign/risk/risk_engine.py": ["sovereign.risk.layers"],          # :35/:46 import_module
    "sovereign/present_state.py": ["sovereign.forex.cb_calendar"],       # :325 spec_from_file
    "sovereign/brain/obsidian_writer.py": ["sovereign.brain"],           # :140 importlib.reload
}

XML_COMMENT_RE = re.compile(rb"<!--.*?-->", re.S)


# ── enumeration ────────────────────────────────────────────────────────────────

def _tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, timeout=30).stdout.splitlines()
    return [f for f in out if not any(x in f for x in EXCLUDE_SUBSTR)]


def _age_days(p: Path) -> float | None:
    try:
        import time
        return (time.time() - p.stat().st_mtime) / 86400.0
    except OSError:
        return None


# ── AST import graph ───────────────────────────────────────────────────────────

def _imports_in(path: Path) -> list[str] | None:
    """Imported module names via AST. None means the file FAILED to parse (ERROR)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return None
    except (ValueError, OSError):
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def _has_main(path: Path) -> bool:
    """A runnable entry point: `if __name__ == '__main__'`."""
    try:
        return '__main__' in path.read_text(errors="ignore")
    except OSError:
        return False


def _docstring(path: Path) -> str | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = ast.get_docstring(ast.parse(path.read_text(errors="ignore")))
    except Exception:
        return None
    if not d:
        return None
    # First sentence, collapsed to one line, hard-capped — some docstrings are a
    # title without terminal punctuation and would otherwise run for paragraphs.
    line = " ".join(d.strip().split())
    m = re.split(r"(?<=[.!?])\s", line, maxsplit=1)
    first = m[0] if m else line
    if len(first) > 160:
        first = first[:157].rsplit(" ", 1)[0] + "…"
    return first


# ── live roots from installed plists ───────────────────────────────────────────

def _installed_plists() -> list[Path]:
    la = Path.home() / "Library" / "LaunchAgents"
    if not la.exists():
        return []
    return sorted(p for p in la.glob("com.*.plist")
                  if p.name.split(".")[1] in ("alta", "sovereign", "clawd"))


def _load_plist(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        return plistlib.loads(raw)
    except Exception:
        return plistlib.loads(XML_COMMENT_RE.sub(b"", raw))


def _module_to_file(mod: str) -> str | None:
    """Resolve a dotted module (from `python3 -m X.Y`) to a tracked file path."""
    cand = mod.replace(".", "/") + ".py"
    if (ROOT / cand).exists():
        return cand
    pkg = mod.replace(".", "/") + "/__init__.py"
    return pkg if (ROOT / pkg).exists() else None


def _plist_entry_scripts(plists: list[Path]) -> dict[str, Path]:
    """Map entry file rel-path -> the plist that runs it. Handles both a direct
    `script.py` argument and `python3 -m package.module` invocation (the harness
    and several agents run the latter, which a .py-suffix check silently misses)."""
    entries: dict[str, Path] = {}
    for pl in plists:
        try:
            d = _load_plist(pl)
        except Exception:
            continue
        args = d.get("ProgramArguments", [])
        for i, a in enumerate(args):
            if not isinstance(a, str):
                continue
            if a.endswith(".py"):
                entries.setdefault(a.replace(str(ROOT) + "/", ""), pl)
            elif a == "-m" and i + 1 < len(args):
                f = _module_to_file(args[i + 1])
                if f:
                    entries.setdefault(f, pl)
    return entries


# Modules invoked as `python3 -m X.Y` from the agent directive and shell scripts.
# These run under a scheduled agent (morning/eod/research), not a plist directly,
# so their entry files are live roots too — WIRED, since the claude binary they run
# under is currently absent (the 2026-07-20 triage finding).
def _directive_roots() -> list[str]:
    roots: list[str] = []
    for src in (ROOT / "AGENT_DIRECTIVE.md", *sorted((ROOT / "scripts").glob("*.sh"))):
        if not src.exists():
            continue
        for m in re.findall(r"python3?\s+-m\s+([a-zA-Z0-9_.]+)", src.read_text(errors="ignore")):
            f = _module_to_file(m)
            if f and f not in roots:
                roots.append(f)
    return roots


def _plist_logs(pl: Path) -> list[Path]:
    try:
        d = _load_plist(pl)
    except Exception:
        return []
    return [Path(d[k]) for k in ("StandardOutPath", "StandardErrorPath")
            if d.get(k)]


# ── reachability ───────────────────────────────────────────────────────────────

def build_graph(files: list[str]) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    """Return (edges: file -> imported module names, docs: file -> first sentence|None).
    A file mapping to None in a third structure means it failed to parse."""
    edges: dict[str, list[str]] = {}
    parse_failed: set[str] = set()
    docs: dict[str, str | None] = {}
    for f in files:
        imports = _imports_in(ROOT / f)
        if imports is None:
            parse_failed.add(f)
            edges[f] = []
        else:
            edges[f] = imports
        docs[f] = _docstring(ROOT / f)
    # Attach dynamic edges (as module names, resolved below like any import).
    for src, targets in DYNAMIC_EDGES.items():
        if src in edges:
            edges[src] = edges[src] + targets
    edges["__parse_failed__"] = sorted(parse_failed)  # smuggle out
    return edges, docs


def _module_index(files: list[str]) -> dict[str, list[str]]:
    """Map both full dotted module and its leaf to candidate files."""
    idx: dict[str, list[str]] = defaultdict(list)
    for f in files:
        mod = f[:-3].replace("/", ".")
        idx[mod].append(f)
        idx[mod.split(".")[-1]].append(f)
        # package form: dir/__init__.py reached as dir path
        if f.endswith("/__init__.py"):
            idx[f[:-len("/__init__.py")].replace("/", ".")].append(f)
    return idx


def reachable(files: list[str], edges: dict[str, list[str]],
              roots: list[str]) -> set[str]:
    idx = _module_index(files)
    seen: set[str] = set()
    stack = [r for r in roots if (ROOT / r).exists()]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for imp in edges.get(f, []):
            # match full dotted, prefix (package), and leaf
            cands = set(idx.get(imp, []))
            cands |= set(idx.get(imp.split(".")[-1], []))
            # prefix match: dynamic "sovereign.risk.layers" reaches all under it
            for m, fs in idx.items():
                if m.startswith(imp + ".") or imp.startswith(m + "."):
                    cands.update(fs)
            for c in cands:
                if c not in seen:
                    stack.append(c)
    return seen


# ── classification ─────────────────────────────────────────────────────────────

@dataclass
class Row:
    path: str
    feature: str
    status: str
    sub_status: str = ""
    description: str = ""
    provenance: str = ""   # "docstring" | "undocumented"
    evidence: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _retired_by_marker(path: Path) -> bool:
    try:
        head = path.read_text(errors="ignore")[:400]
    except OSError:
        return False
    return any(m in head for m in RETIRED_MARKERS)


def _feature_of(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "sovereign" and len(parts) > 2:
        return f"sovereign/{parts[1]}"
    if parts[0] in ("scripts", "tests", "research", "data") and len(parts) > 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if len(parts) > 1 else "(root)"


def classify(files: list[str], edges: dict[str, list[str]], docs: dict[str, str | None],
             live_set: set[str], ondemand_set: set[str], test_reachable: set[str],
             manual_entries: set[str], entry_owner: dict[str, Path]) -> list[Row]:
    parse_failed = set(edges.get("__parse_failed__", []))
    rows: list[Row] = []
    for f in sorted(files):
        p = ROOT / f
        top = f.split("/")[0]
        desc = docs.get(f)
        prov = "docstring" if desc else "undocumented"
        if not desc:
            desc = f"(UNDOCUMENTED — {p.stem.replace('_', ' ')})"

        # precedence RETIRED > ERROR > LIVE > ON-DEMAND > TEST-ONLY > DORMANT
        if top in RETIRED_DIRS or _retired_by_marker(p):
            status, sub, ev = "RETIRED", "", (
                f"in {top}/" if top in RETIRED_DIRS else "header marker")
        elif f in parse_failed:
            status, sub, ev = "ERROR", "", "ast.parse failed (syntax error)"
        elif f in live_set:
            status = "LIVE"
            owner = entry_owner.get(f)
            sub, ev = _firing_or_wired(owner) if owner else (
                "REACHED", "reachable from a scheduled root")
        elif f in ondemand_set:
            status = "ON-DEMAND"
            if f in manual_entries:
                sub, ev = "ENTRY", "runnable __main__; no scheduled job runs it"
            else:
                sub, ev = "REACHED", "reached from a manual entry point"
        elif f in test_reachable:
            status, sub, ev = "TEST-ONLY", "", "reachable only from tests/"
        else:
            status, sub, ev = "DORMANT", "", "no scheduled root, entry point, or test reaches it"

        rows.append(Row(path=f, feature=_feature_of(f), status=status, sub_status=sub,
                        description=desc, provenance=prov, evidence=ev))
    return rows


def _firing_or_wired(plist: Path) -> tuple[str, str]:
    logs = _plist_logs(plist)
    fresh = []
    for lp in logs:
        if lp.exists() and lp.stat().st_size > 0:
            age = _age_days(lp)
            if age is not None and age <= FIRING_DAYS:
                fresh.append(f"{lp.name} {age:.1f}d")
    if fresh:
        return "FIRING", f"{plist.name}: {', '.join(fresh)}"
    return "WIRED", f"{plist.name}: no recent log (loaded, output unconfirmed)"


# ── self-test (isolation) ──────────────────────────────────────────────────────

def self_test() -> int:
    forbidden = ("execution", "sovereign", "ict", "backtester")
    names = _imports_in(Path(__file__)) or []
    bad = [n for n in names if n.split(".")[0] in forbidden]
    if bad:
        print(f"SELF-TEST FAIL — execution-path imports: {bad}")
        return 1
    print("SELF-TEST PASS — no execution-path imports; generator is isolated")
    return 0


# ── driver ─────────────────────────────────────────────────────────────────────

def build() -> tuple[list[Row], dict]:
    files = _tracked_py()
    plists = _installed_plists()
    entry_owner = _plist_entry_scripts(plists)
    directive_roots = _directive_roots()
    roots = list(entry_owner) + directive_roots + [
        "execute_daily.py", "sovereign/orchestrator.py"]

    edges, docs = build_graph(files)
    live_set = reachable(files, edges, roots)

    # On-demand: runnable __main__ entry points (not scheduled) and what they reach.
    manual_entries = {f for f in files if _has_main(ROOT / f)} - live_set
    ondemand_set = (reachable(files, edges, list(manual_entries)) | manual_entries) - live_set

    # Test reachability: roots = every tests/ file, minus anything already live/on-demand.
    test_roots = [f for f in files if f.split("/")[0] == "tests"]
    test_set = reachable(files, edges, test_roots) - live_set - ondemand_set

    rows = classify(files, edges, docs, live_set, ondemand_set, test_set,
                    manual_entries, entry_owner)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.status] += 1
    meta = {
        "n_files": len(files),
        "n_installed_plists": len(plists),
        "n_live_roots": len([r for r in roots if (ROOT / r).exists()]),
        "counts": dict(counts),
        "dynamic_edges": DYNAMIC_EDGES,
    }
    return rows, meta


_STATUS_ORDER = ["LIVE", "ON-DEMAND", "TEST-ONLY", "DORMANT", "ERROR", "RETIRED"]
_BADGE = {"LIVE": "🟢 LIVE", "ON-DEMAND": "🔵 ON-DEMAND", "TEST-ONLY": "🟡 TEST-ONLY",
          "DORMANT": "⚪ DORMANT", "ERROR": "🔴 ERROR", "RETIRED": "⚫ RETIRED"}


def render_md(rows: list[Row], meta: dict) -> str:
    c = meta["counts"]
    firing = sum(1 for r in rows if r.sub_status == "FIRING")
    wired = sum(1 for r in rows if r.sub_status == "WIRED")
    L: list[str] = []
    L.append("# SYSTEM INVENTORY — every first-party feature, status derived not asserted")
    L.append("")
    L.append("> Generated by `audit/system_inventory.py` (read-only). Re-run to refresh.")
    L.append("> Status is **derived from evidence** — import-graph reachability, an AST parse "
             "result, a directory, or a header marker — never asserted from a filename. "
             "Descriptions are the author's own module docstrings; `(UNDOCUMENTED …)` marks "
             "the 8% with none, so provenance is always visible.")
    L.append("")
    L.append("## Totals")
    L.append("")
    L.append(f"**{meta['n_files']} first-party Python files** "
             f"({meta['n_installed_plists']} installed plists, {meta['n_live_roots']} live roots).")
    L.append("")
    L.append("| status | count | meaning |")
    L.append("|---|---|---|")
    L.append(f"| 🟢 LIVE | {c.get('LIVE',0)} | reachable from a scheduled job "
             f"(**{firing} FIRING** — recent log; **{wired} WIRED** — loaded but no recent output) |")
    L.append(f"| 🔵 ON-DEMAND | {c.get('ON-DEMAND',0)} | a runnable tool invoked by hand, or code it reaches |")
    L.append(f"| 🟡 TEST-ONLY | {c.get('TEST-ONLY',0)} | reachable only from tests/ |")
    L.append(f"| ⚪ DORMANT | {c.get('DORMANT',0)} | reached by nothing — built, wired to nothing |")
    L.append(f"| 🔴 ERROR | {c.get('ERROR',0)} | fails to parse |")
    L.append(f"| ⚫ RETIRED | {c.get('RETIRED',0)} | in attic/archive/scratch/lab or marked dead |")
    L.append("")
    L.append("### The one blind spot — dynamic imports")
    L.append("An AST walk cannot see `importlib`/`import_module`. Four dynamic edges are "
             "**hardcoded** so the reachability graph is not blind; each is asserted from a "
             "hand-read of the source, not derived:")
    for src, tgts in meta["dynamic_edges"].items():
        L.append(f"- `{src}` → {', '.join(f'`{t}`' for t in tgts)} *(manually asserted)*")
    L.append("")
    L.append("Measured effect (not assumed): of these four, only "
             "`sovereign/present_state.py → sovereign.forex.cb_calendar` actually changes a "
             "classification — it alone makes `cb_calendar.py` LIVE. The other three are "
             "**redundant**: their targets are also reached by ordinary static imports, so the "
             "hardcode is belt-and-suspenders, not load-bearing. Stated because an earlier "
             "draft of this file claimed all four were load-bearing — they are not, and "
             "checking beat asserting.")
    L.append("")
    L.append("> **LIVE-WIRED is not proof of output.** Reachable from a loaded plist means "
             "wired to run, not confirmed producing — the loaded-but-dead class the "
             "2026-07-20 claude-binary triage found. Only FIRING has a recent log.")
    L.append("")

    # actionable rollups first
    dormant = [r for r in rows if r.status == "DORMANT"]
    errors = [r for r in rows if r.status == "ERROR"]
    if errors:
        L.append("## 🔴 ERROR — files that do not parse (act on these)")
        L.append("")
        for r in errors:
            L.append(f"- `{r.path}` — {r.evidence}")
        L.append("")
    L.append(f"## ⚪ DORMANT — {len(dormant)} files reached by nothing (review these)")
    L.append("")
    L.append("Built, importable, but no scheduled job, manual entry point, or test reaches "
             "them. Candidates for wiring-up or retirement — decided per file, not in bulk.")
    L.append("")
    for r in sorted(dormant, key=lambda r: r.path):
        L.append(f"- `{r.path}` — {r.description}")
    L.append("")

    # per-subsystem
    L.append("## Every feature by subsystem")
    L.append("")
    by_feat: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_feat[r.feature].append(r)
    for feat in sorted(by_feat):
        frows = by_feat[feat]
        fc = defaultdict(int)
        for r in frows:
            fc[r.status] += 1
        badge = " · ".join(f"{_BADGE[s].split()[1]} {fc[s]}"
                           for s in _STATUS_ORDER if fc[s])
        L.append(f"### `{feat}/` — {len(frows)} files ({badge})")
        L.append("")
        L.append("| file | status | what it does |")
        L.append("|---|---|---|")
        for r in sorted(frows, key=lambda r: (_STATUS_ORDER.index(r.status), r.path)):
            name = r.path.split("/")[-1]
            sub = f" {r.sub_status}" if r.sub_status else ""
            desc = r.description.replace("|", "\\|")
            if r.provenance == "undocumented":
                desc = f"*{desc}*"
            L.append(f"| `{name}` | {_BADGE[r.status].split()[0]}{sub} | {desc} |")
        L.append("")
    return "\n".join(L)


def _first_comment(path: Path, prefixes=("#", "//")) -> str:
    """First non-shebang comment line of a file — its self-description."""
    try:
        for line in path.read_text(errors="ignore").splitlines()[:15]:
            s = line.strip()
            if s.startswith("#!"):
                continue
            for pre in prefixes:
                if s.startswith(pre):
                    return s.lstrip("#/ ").strip()[:160]
    except OSError:
        pass
    return "(no header comment)"


def render_jobs_and_config() -> str:
    """Non-.py features: launchd jobs, config, shell + MCP entry points, data dirs.
    All derived — plist schedules and firing state read from disk, config and shell
    descriptions from each file's own first comment."""
    L: list[str] = ["", "---", "", "# Non-code features", ""]

    # ── plists ──
    installed = {p.name: p for p in _installed_plists()}
    authored = sorted((ROOT / "scripts").glob("com.*.plist"))
    all_labels = sorted(set(installed) | {p.name for p in authored})
    L.append(f"## launchd jobs — {len(all_labels)} plists "
             f"({len(installed)} installed)")
    L.append("")
    L.append("| job | installed | firing | entry |")
    L.append("|---|---|---|---|")
    for name in all_labels:
        pl = installed.get(name) or (ROOT / "scripts" / name)
        inst = "yes" if name in installed else "authored-only"
        try:
            d = _load_plist(pl)
            args = d.get("ProgramArguments", [])
            entry = next((a for a in args if isinstance(a, str)
                          and (a.endswith(".py") or a.endswith(".sh") or a == "-m")), "?")
            if entry == "-m":
                i = args.index("-m")
                entry = f"-m {args[i+1]}" if i + 1 < len(args) else "-m ?"
            entry = entry.replace(str(ROOT) + "/", "")
        except Exception:
            entry = "(unreadable)"
        fire = _firing_or_wired(installed[name])[0] if name in installed else "—"
        L.append(f"| `{name.replace('.plist','')}` | {inst} | {fire} | `{entry}` |")
    L.append("")

    # ── config ──
    cfgs = subprocess.run(["git", "ls-files", "config/*.yml", "config/*.yaml"],
                          cwd=ROOT, capture_output=True, text=True).stdout.split()
    L.append(f"## Config — {len(cfgs)} files (never hardcode thresholds; these govern them)")
    L.append("")
    L.append("| file | governs |")
    L.append("|---|---|")
    for c in sorted(cfgs):
        L.append(f"| `{c}` | {_first_comment(ROOT / c).replace('|','\\|')} |")
    L.append("")

    # ── shell + MCP ──
    shells = subprocess.run(["git", "ls-files", "scripts/*.sh"],
                            cwd=ROOT, capture_output=True, text=True).stdout.split()
    L.append(f"## Shell entry points — {len(shells)}")
    L.append("")
    L.append("| file | what it does |")
    L.append("|---|---|")
    for s in sorted(shells):
        L.append(f"| `{s}` | {_first_comment(ROOT / s).replace('|','\\|')} |")
    L.append("")
    mcp = subprocess.run(["git", "ls-files", "mcp/*/src/*.ts"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    if mcp:
        L.append(f"## MCP server tools — {len(mcp)} TypeScript sources under `mcp/`")
        L.append("")
        for m in sorted(mcp):
            L.append(f"- `{m}` — {_first_comment(ROOT / m).replace('|','\\|')}")
        L.append("")

    # ── data dirs (summary, not per-file) ──
    L.append("## `data/` and `models/` — outputs, summarized per-directory")
    L.append("")
    L.append("These hold artifacts (saved results), not features. One line per subdirectory:")
    L.append("")
    for base in ("data", "models"):
        bp = ROOT / base
        if not bp.exists():
            continue
        subs = sorted(d.name for d in bp.iterdir() if d.is_dir()
                      and d.name != "__pycache__")
        L.append(f"**`{base}/`** — {len(subs)} subdirectories: "
                 + ", ".join(f"`{s}`" for s in subs))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="System inventory (read-only, derived status)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="write SYSTEM_INVENTORY.md and data/audit/system_inventory.json")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    rows, meta = build()
    if args.json:
        print(json.dumps({"meta": meta, "rows": [r.to_json() for r in rows]}, indent=2))
    elif args.summary:
        print(json.dumps(meta, indent=2))
    elif args.write:
        (ROOT / "SYSTEM_INVENTORY.md").write_text(render_md(rows, meta) + render_jobs_and_config())
        outdir = ROOT / "data" / "audit"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "system_inventory.json").write_text(
            json.dumps({"meta": meta, "rows": [r.to_json() for r in rows]}, indent=2))
        print(f"wrote SYSTEM_INVENTORY.md ({len(rows)} rows) + data/audit/system_inventory.json")
    else:
        for r in rows:
            print(f"{r.status:9} {r.sub_status:8} {r.path}")
    raise SystemExit(0)
