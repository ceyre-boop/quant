#!/usr/bin/env python3
"""Build an Obsidian knowledge graph of the quant repo, AS A SYSTEM.

Deterministic, idempotent generator. Walks the production code, the config YAMLs, and the hypothesis
ledger, and writes ~hundreds of interconnected markdown notes into <vault>/Trading/System/ — one per
module (with real docstrings, signatures, import-dependency links, layer classification, config
reads), one per parameter group (current values + read-by), one per hypothesis, plus per-subsystem
MOCs and index MOCs. Re-running it is how you UPDATE the graph (it prunes its own stale notes via a
manifest; it never touches anything it didn't generate).

  python3 scripts/build_obsidian_graph.py --dry-run        # counts only, writes nothing
  python3 scripts/build_obsidian_graph.py --subsystem forex --limit 20   # review batch
  python3 scripts/build_obsidian_graph.py                   # full run

Facts are extracted (ast / yaml / json) — never inferred. The interpretive concept layer
(Trading/System/Concepts/) is hand-authored separately and is NOT managed/pruned by this tool.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# ── roots (overridable via CLI) ──────────────────────────────────────────────────────────────────
DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = Path("/Users/taboost/Obsidian/Obsidian")
SYSTEM_SUBDIR = Path("Trading") / "System"
MANIFEST_NAME = ".graph-manifest.json"
TODAY = date.today().isoformat()

# Production code dirs whose .py files become module notes.
INCLUDE_DIRS = {
    "sovereign", "ict", "ict-engine", "execution", "orchestrator", "integration",
    "entry_engine", "imbalance_engine", "contracts", "governance",
    "layer1", "layer2", "layer3", "training", "backtest", "lab",
}
# Represented as category-index notes only (not one-per-file).
INDEX_ONLY_DIRS = {"scripts", "tests"}
SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"}

# Config files (explicit + generic glob picks up new ones).
CONFIG_GLOBS = ["config/*.yml", "config/*.yaml", "sovereign/risk/config/*.yaml"]

# Layer classification seeded from docs/COMPONENT_CLASSIFICATION.md (path prefix → layer).
PATH_LAYER_MAP = [
    ("sovereign/risk/", "L2"),
    ("sovereign/execution/", "L2"),
    ("sovereign/forex/fast_backtester", "L2"),
    ("sovereign/forex/position_sizer", "L2"),
    ("layer2/", "L2"),
    ("layer3/", "L2"),
    ("sovereign/forex/macro_engine", "L1"),
    ("sovereign/forex/signal_engine", "L1"),
    ("sovereign/forex/entry_engine", "L1"),
    ("sovereign/briefing/", "L1"),
    ("sovereign/layer1/", "L1"),
    ("layer1/", "L1"),
    ("sovereign/research/", "research"),
    ("sovereign/discovery/", "research"),
    ("lab/", "research"),
    ("sovereign/oracle/", "infra"),
    ("sovereign/intelligence/", "infra"),
    ("sovereign/data/", "infra"),
    ("sovereign/reporting/", "infra"),
    ("sovereign/autonomous/", "infra"),
    ("governance/", "infra"),
    ("sovereign/forex/", "L1"),       # remaining forex defaults predictive
    ("sovereign/futures/", "L1"),
    ("ict/", "L1"),
    ("ict-engine/", "infra"),
]
MODULE_LAYER_OVERRIDES = {
    "sovereign/futures/decision_engine.py": "L1+L2 (bleed)",
    "sovereign/intelligence/decision_logger.py": "infra",
}

SUBSYSTEM_INTROS = {
    "forex": "The validated trading engine: the macro carry/rate-differential edge on daily FX. "
             "Layer-1 prediction (macro_engine → signal_engine, conviction-gated) and the Layer-2 "
             "exit/sizing machinery (fast_backtester's 6-state exit machine — backtest-only — and "
             "position_sizer). The only proven edge in the system; see [[Carry-Edge]], [[Two-Layer-Wall]].",
    "risk": "Layer-2 'Stockfish dispose' engine — the SOLE sizing authority. risk_engine.decide() "
            "composes 8 layers (base × vol × dd × regime, capped by kelly/portfolio/prop, halted to 0 "
            "by 6 hard gates). Live and apex-quality. See [[Eight-Risk-Layers]].",
    "oracle": "The cognition layer: 1 Opus call/day, HARVEST→REFLECT→TEST→CODIFY → one lesson. Reads "
              "decision logs (closed loop) + macro/briefing context. Not a trading input. See "
              "[[Oracle-Closed-Loop]].",
    "execution": "Order dispatch + broker bridges (OANDA/IB/paper). Live exits are static stop/TP + "
                 "poll today — the run-to-exit gap. See [[Six-Exit-States]].",
    "discovery": "The edge-discovery bench: mines candidates, the gauntlet decides. 28 candidates → 0 "
                 "survivors (the carry edge is irreducible). See [[Discovery-Meta-Finding]], [[The-Gauntlet]].",
    "intelligence": "Orchestration layer: regime performance tracker, capital allocator, decision "
                    "logger, system health, cross-system bridge. The durable edge ([[Tenet-5-Orchestration]]).",
    "futures": "Track-2 ES/NQ micro/ORB/VWAP engine (isolated). decision_engine.evaluate_entry "
               "currently BLEEDS L1+L2 in one object — the named anti-pattern. See [[Two-Layer-Wall]].",
    "briefing": "Morning + EOD briefs and the Opus AI directional bias (synthesize). provenance."
                "verified=false — context only, wired to no trade today (the L1 agreement-gate gap).",
    "autonomous": "The 4-component closure layer (health responder, hypothesis generator, research "
                  "factory, escalation router). config/autonomous.yml::live gates the write paths.",
    "layer1": "Layer-1 research benchmarks / excursion analysis (MFE-MAE exit-thesis tests) and the "
              "XGBoost directional-bias re-attempts (HYP-064, meta-labeling).",
    "data": "Market-data feeds (yfinance / OANDA / Polygon / Databento) and cache. The graph's roots; "
            "see [[Data-Flow-Pipeline]].",
    "reporting": "Trade summary + P&L tables, and the equity_curve.v1 schema the proof engine draws.",
    "research": "Pre-registered hypothesis research (incl. the VRP / variance-risk-premium track, "
                "DATA_INSUFFICIENT pending option chains). Routes through [[The-Gauntlet]].",
    "ict": "The intraday ICT micro-edge engine (kill-zones, sweeps, FVGs, order-blocks, displacement). "
           "Time-horizon isolated from sovereign/ ([[ICT-Sovereign-Isolation]]). Pattern edge NOT proven.",
    "ict-engine": "The isolation-safe cross-layer bridge (orchestrator.py) — the only sanctioned "
                  "ICT→sovereign entry point. See [[ICT-Sovereign-Isolation]].",
}

# ── small helpers ────────────────────────────────────────────────────────────────────────────────
def disp(subsys: str) -> str:
    return "-".join(p.capitalize() for p in subsys.split("-"))

def moc_link(subsys: str) -> str:
    return f"{disp(subsys)}-MOC"

def yaml_escape_title(t: str) -> str:
    return t.replace('"', '\\"')

def first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for ln in doc.strip().splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""

def safe_unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "?"

def esc(s: str) -> str:
    """Neutralize accidental [[ / ]] in EXTRACTED text (docstrings, signatures, yaml) so they don't
    render as broken Obsidian wikilinks. Zero-width space — visually identical, breaks the link parse."""
    return (s or "").replace("[[", "[​[").replace("]]", "]​]")

def dotted(relpath: str) -> str:
    return relpath[:-3].replace("/", ".") if relpath.endswith(".py") else relpath.replace("/", ".")

def subsystem_of(relpath: str) -> str:
    parts = relpath.split("/")
    if parts[0] == "sovereign":
        return parts[1] if len(parts) > 2 else "sovereign"
    return parts[0]

def layer_of(relpath: str) -> str:
    if relpath in MODULE_LAYER_OVERRIDES:
        return MODULE_LAYER_OVERRIDES[relpath]
    for prefix, layer in PATH_LAYER_MAP:
        if relpath.startswith(prefix):
            return layer
    return "unclassified"

def format_signature(fn) -> str:
    a = fn.args
    parts: list[str] = []
    all_pos = list(a.posonlyargs) + list(a.args)
    ndef = len(a.defaults)
    default_map = {len(all_pos) - ndef + i: d for i, d in enumerate(a.defaults)}
    for i, arg in enumerate(all_pos):
        s = arg.arg
        if arg.annotation is not None:
            s += f": {safe_unparse(arg.annotation)}"
        if i in default_map:
            s += f"={safe_unparse(default_map[i])}"
        parts.append(s)
        if a.posonlyargs and i == len(a.posonlyargs) - 1:
            parts.append("/")
    if a.vararg:
        v = "*" + a.vararg.arg
        if a.vararg.annotation:
            v += f": {safe_unparse(a.vararg.annotation)}"
        parts.append(v)
    elif a.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(a.kwonlyargs):
        s = arg.arg
        if arg.annotation is not None:
            s += f": {safe_unparse(arg.annotation)}"
        if a.kw_defaults[i] is not None:
            s += f"={safe_unparse(a.kw_defaults[i])}"
        parts.append(s)
    if a.kwarg:
        k = "**" + a.kwarg.arg
        if a.kwarg.annotation:
            k += f": {safe_unparse(a.kwarg.annotation)}"
        parts.append(k)
    sig = "(" + ", ".join(parts) + ")"
    if fn.returns is not None:
        sig += f" -> {safe_unparse(fn.returns)}"
    return sig


class Graph:
    def __init__(self, repo: Path, vault: Path):
        self.repo = repo
        self.out = vault / SYSTEM_SUBDIR
        self.errors: list[str] = []
        self.modules: dict[str, dict] = {}      # relpath -> info
        self.dotted_index: dict[str, str] = {}   # dotted -> relpath
        self.config_groups: dict[str, dict] = {} # configstem -> {group: subtree}
        self.config_files: dict[str, str] = {}   # configstem -> relpath
        self.hypotheses: list[dict] = []
        self.module_reads: dict[str, set[str]] = {}   # relpath -> {configstem}
        self.used_by: dict[str, set[str]] = {}        # relpath -> {relpath importer}

    # ---- scanning ----
    def scan_modules(self):
        for d in sorted(INCLUDE_DIRS):
            base = self.repo / d
            if not base.exists():
                continue
            for path in sorted(base.rglob("*.py")):
                if any(p in SKIP_DIR_NAMES for p in path.parts):
                    continue
                rel = path.relative_to(self.repo).as_posix()
                info = self._parse_module(path, rel)
                if info is not None:
                    self.modules[rel] = info
                    self.dotted_index[dotted(rel)] = rel

    def _parse_module(self, path: Path, rel: str) -> dict | None:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.errors.append(f"read {rel}: {e}")
            return None
        info = {"rel": rel, "subsystem": subsystem_of(rel), "layer": layer_of(rel),
                "doc": "", "classes": [], "functions": [], "imports": set(), "src": src,
                "is_layer": "/layers/" in rel, "parse_error": False}
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            info["parse_error"] = True
            self.errors.append(f"parse {rel}: {e}")
            if path.name == "__init__.py":
                return None
            return info
        info["doc"] = ast.get_docstring(tree) or ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and node.name != "__init__":
                    continue
                info["functions"].append((node.name, format_signature(node), first_line(ast.get_docstring(node))))
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(safe_unparse(b) for b in node.bases)
                methods = []
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if m.name.startswith("_") and m.name != "__init__":
                            continue
                        methods.append((m.name, format_signature(m), first_line(ast.get_docstring(m))))
                info["classes"].append((node.name, bases, first_line(ast.get_docstring(node)), methods))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    info["imports"].add(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    for n in node.names:
                        info["imports"].add(f"{node.module}.{n.name}")
                        info["imports"].add(node.module)
        # skip truly-empty __init__.py
        if path.name == "__init__.py" and not info["doc"] and not info["classes"] and not info["functions"]:
            return None
        return info

    def resolve_internal_deps(self):
        for rel, info in self.modules.items():
            deps: set[str] = set()
            for imp in info["imports"]:
                if imp in self.dotted_index and self.dotted_index[imp] != rel:
                    deps.add(self.dotted_index[imp])
                else:
                    # try trimming the tail (from a.b import c → a.b)
                    head = imp.rsplit(".", 1)[0]
                    if head in self.dotted_index and self.dotted_index[head] != rel:
                        deps.add(self.dotted_index[head])
            info["deps"] = deps
            for dep in deps:
                self.used_by.setdefault(dep, set()).add(rel)

    def scan_configs(self):
        seen: set[str] = set()
        for glob in CONFIG_GLOBS:
            for path in sorted(self.repo.glob(glob)):
                rel = path.relative_to(self.repo).as_posix()
                stem = path.stem
                if stem in seen:
                    continue
                seen.add(stem)
                if yaml is None:
                    continue
                try:
                    data = yaml.safe_load(path.read_text()) or {}
                except Exception as e:
                    self.errors.append(f"yaml {rel}: {e}")
                    continue
                if not isinstance(data, dict):
                    continue
                self.config_files[stem] = rel
                self.config_groups[stem] = {k: v for k, v in data.items()}
        # which modules reference each config (by file stem appearing in source)
        for rel, info in self.modules.items():
            reads = set()
            for stem in self.config_groups:
                if stem in info["src"]:
                    reads.add(stem)
            self.module_reads[rel] = reads

    def scan_hypotheses(self):
        path = self.repo / "data" / "agent" / "hypothesis_ledger.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            self.errors.append(f"ledger: {e}")
            return
        if isinstance(data, list):
            self.hypotheses = [h for h in data if isinstance(h, dict) and h.get("id")]

    # ---- rendering ----
    def _frontmatter(self, title, *, ntype="system-doc", subsystem=None, layer=None, source=None) -> str:
        lines = ["---", f"date: {TODAY}", f'title: "{yaml_escape_title(title)}"', f"type: {ntype}",
                 "project: Alta"]
        if subsystem:
            lines.append(f"subsystem: {subsystem}")
        if layer:
            lines.append(f"layer: {layer}")
        if source:
            lines.append(f"source: {source}")
        lines.append("generated: build_obsidian_graph.py")
        lines.append("---\n")
        return "\n".join(lines)

    def render_module(self, rel: str) -> tuple[str, str]:
        info = self.modules[rel]
        dot = dotted(rel)
        subsys = info["subsystem"]
        fm = self._frontmatter(rel, subsystem=subsys, layer=info["layer"], source=rel)
        out = [fm,
               f"> Links: [[00-System-Index]] · [[{moc_link(subsys)}]] · [[Alta-MOC]]\n",
               f"# {dot}\n",
               f"**Subsystem:** {subsys} · **Layer:** {info['layer']} · **Path:** `{rel}`\n"]
        if info["parse_error"]:
            out.append("> ⚠️ This file did not parse cleanly; extraction is partial.\n")
        out.append((esc(info["doc"].strip()) + "\n") if info["doc"] else "_No module docstring._\n")
        if info["classes"]:
            out.append("## Classes\n")
            for name, bases, doc, methods in info["classes"]:
                hdr = f"### `{name}({esc(bases)})`" if bases else f"### `{name}`"
                out.append(hdr)
                out.append((esc(doc) or "_no docstring_") + "\n")
                for mn, sig, mdoc in methods:
                    out.append(f"- `{mn}{esc(sig)}`" + (f" — {esc(mdoc)}" if mdoc else ""))
                out.append("")
        if info["functions"]:
            out.append("## Functions\n")
            for fn, sig, doc in info["functions"]:
                out.append(f"- `{fn}{esc(sig)}`" + (f" — {esc(doc)}" if doc else ""))
            out.append("")
        deps = sorted(info.get("deps", set()))
        if deps:
            out.append("## Depends on (internal)\n")
            out.extend(f"- [[{dotted(d)}]]" for d in deps)
            out.append("")
        users = sorted(self.used_by.get(rel, set()))
        if users:
            out.append("## Used by\n")
            out.extend(f"- [[{dotted(u)}]]" for u in users)
            out.append("")
        reads = sorted(self.module_reads.get(rel, set()))
        if reads:
            out.append("## Config it reads\n")
            for stem in reads:
                for group in sorted(self.config_groups.get(stem, {})):
                    out.append(f"- [[{stem}.{group}]]")
            out.append("")
        related = self._related_hyps(info)
        if related:
            out.append("## Related hypotheses\n")
            out.extend(f"- [[{hid}]]" for hid in related)
            out.append("")
        notepath = f"Modules/{dot}.md"
        return notepath, "\n".join(out)

    def _related_hyps(self, info) -> list[str]:
        names = {Path(info["rel"]).stem}
        for c in info["classes"]:
            names.add(c[0])
        names = {n.lower() for n in names if len(n) >= 6}
        hits = []
        for h in self.hypotheses:
            blob = " ".join(str(h.get(k, "")) for k in ("name", "mechanism", "action")).lower()
            if any(n in blob for n in names):
                hits.append(h["id"])
        return sorted(set(hits))

    def render_subsystem_mocs(self) -> list[tuple[str, str]]:
        by_sub: dict[str, list[str]] = {}
        for rel, info in self.modules.items():
            by_sub.setdefault(info["subsystem"], []).append(rel)
        notes = []
        for subsys, rels in sorted(by_sub.items()):
            rels.sort()
            intro = SUBSYSTEM_INTROS.get(subsys, f"The `{subsys}` subsystem.")
            layers = {self.modules[r]["layer"] for r in rels}
            fm = self._frontmatter(f"{disp(subsys)} subsystem", ntype="moc", subsystem=subsys)
            out = [fm, f"> Links: [[00-System-Index]] · [[Alta-MOC]]\n",
                   f"# {disp(subsys)} subsystem\n", f"> {intro}\n",
                   f"**Modules:** {len(rels)} · **Layers present:** {', '.join(sorted(layers))}\n",
                   "## Modules\n"]
            layer_modules = [r for r in rels if self.modules[r]["is_layer"]]
            core = [r for r in rels if not self.modules[r]["is_layer"]]
            for r in core:
                out.append(f"- [[{dotted(r)}]] — {esc(first_line(self.modules[r]['doc'])) or self.modules[r]['layer']}")
            if layer_modules:
                out.append("\n### Layers\n")
                for r in layer_modules:
                    out.append(f"- [[{dotted(r)}]] — {esc(first_line(self.modules[r]['doc']))}")
            # parameters read by this subsystem
            params = set()
            for r in rels:
                for stem in self.module_reads.get(r, set()):
                    for group in self.config_groups.get(stem, {}):
                        params.add(f"{stem}.{group}")
            if params:
                out.append("\n## Parameters\n")
                out.extend(f"- [[{p}]]" for p in sorted(params))
            # related hypotheses
            hyps = set()
            for r in rels:
                hyps.update(self._related_hyps(self.modules[r]))
            if hyps:
                out.append("\n## Related hypotheses\n")
                out.extend(f"- [[{h}]]" for h in sorted(hyps))
            notes.append((f"Subsystems/{moc_link(subsys)}.md", "\n".join(out)))
        return notes

    def render_param_notes(self) -> list[tuple[str, str]]:
        notes = []
        if yaml is None:
            return notes
        readers: dict[str, set[str]] = {}
        for rel, stems in self.module_reads.items():
            for stem in stems:
                readers.setdefault(stem, set()).add(rel)
        for stem, groups in sorted(self.config_groups.items()):
            src = self.config_files.get(stem, "")
            for group, subtree in sorted(groups.items()):
                fm = self._frontmatter(f"{stem} → {group}", subsystem="parameters", source=src)
                dump = esc(yaml.safe_dump({group: subtree}, sort_keys=False, default_flow_style=False).rstrip())
                out = [fm, f"> Links: [[00-System-Index]] · [[Parameters-Index]] · [[Alta-MOC]]\n",
                       f"# {stem} → `{group}`\n", f"**Source:** `{src}`\n",
                       "```yaml", dump, "```\n", "## Read by\n"]
                rby = sorted(readers.get(stem, set()))
                if rby:
                    out.extend(f"- [[{dotted(r)}]]" for r in rby)
                else:
                    out.append("_No module references detected (best-effort scan)._")
                notes.append((f"Parameters/{stem}.{group}.md", "\n".join(out)))
        return notes

    def render_hypothesis_notes(self) -> list[tuple[str, str]]:
        notes = []
        for h in sorted(self.hypotheses, key=lambda x: x["id"]):
            hid = h["id"]
            fm = self._frontmatter(f"{hid} — {h.get('name','')}", subsystem="hypothesis")
            out = [fm, f"> Links: [[00-System-Index]] · [[Hypothesis-Ledger]] · [[Alta-MOC]]\n",
                   f"# {hid} — {h.get('name','')}\n",
                   f"**Status:** {h.get('status','?')} · **Result:** {esc(str(h.get('result','—')))}\n"]
            for label, key in (("Mechanism", "mechanism"), ("Action", "action"),
                               ("Methodology", "methodology_note"), ("Retest triage", "retest_triage"),
                               ("Confirmed", "date_confirmed")):
                val = h.get(key)
                if val:
                    out.append(f"**{label}:** {esc(str(val))}\n")
            notes.append((f"Hypotheses/{hid}.md", "\n".join(out)))
        return notes

    def render_indexes(self) -> list[tuple[str, str]]:
        notes = []
        # Parameters-Index
        out = [self._frontmatter("Parameters Index", ntype="moc"),
               "> Links: [[00-System-Index]] · [[Alta-MOC]]\n", "# Parameters Index\n"]
        for stem, groups in sorted(self.config_groups.items()):
            out.append(f"## {stem} (`{self.config_files.get(stem,'')}`)")
            out.extend(f"- [[{stem}.{g}]]" for g in sorted(groups))
            out.append("")
        notes.append(("Parameters-Index.md", "\n".join(out)))
        # Hypothesis-Ledger
        out = [self._frontmatter("Hypothesis Ledger", ntype="moc"),
               "> Links: [[00-System-Index]] · [[Discovery-Ledger]] · [[Alta-MOC]]\n",
               f"# Hypothesis Ledger ({len(self.hypotheses)} entries)\n"]
        by_status: dict[str, list[dict]] = {}
        for h in self.hypotheses:
            by_status.setdefault(h.get("status", "?"), []).append(h)
        for status in sorted(by_status):
            out.append(f"## {status} ({len(by_status[status])})")
            for h in sorted(by_status[status], key=lambda x: x["id"]):
                out.append(f"- [[{h['id']}]] — {h.get('name','')}")
            out.append("")
        notes.append(("Hypothesis-Ledger.md", "\n".join(out)))
        # Scripts-Index + Tests-Index (category, not per-file)
        notes.append(("Scripts-Index.md", self._category_index(
            "Scripts Index", "scripts", "scripts/*.py",
            {"run_": "Hypothesis / pipeline runners", "edge_research_": "Edge research",
             "backtest_": "Backtesters", "validate_": "Validators", "fetch_": "Data fetchers",
             "plot_": "Plots / briefs", "build_": "Builders", "discover": "Discovery"})))
        notes.append(("Tests-Index.md", self._category_index(
            "Tests Index", "tests", "tests/**/*.py", {})))
        return notes

    def _category_index(self, title, root, glob, prefixes) -> str:
        files = sorted(p.relative_to(self.repo).as_posix() for p in self.repo.glob(glob)
                       if "__pycache__" not in p.parts)
        out = [self._frontmatter(title, ntype="moc"),
               "> Links: [[00-System-Index]] · [[Alta-MOC]]\n", f"# {title}\n",
               f"_{len(files)} files. Represented as a category index (not one note per file)._\n"]
        groups: dict[str, list[str]] = {}
        for f in files:
            base = Path(f).name
            label = "misc"
            for pref, lab in prefixes.items():
                if base.startswith(pref) or (pref in base):
                    label = lab
                    break
            if not prefixes:  # tests: group by subdir
                parts = f.split("/")
                label = parts[1] if len(parts) > 2 else "tests/"
            groups.setdefault(label, []).append(base)
        for label in sorted(groups):
            out.append(f"## {label} ({len(groups[label])})")
            out.append(" ".join(f"`{b}`" for b in sorted(groups[label])))
            out.append("")
        return "\n".join(out)

    def render_top_index(self) -> tuple[str, str]:
        subs = sorted({info["subsystem"] for info in self.modules.values()})
        concepts = ["Two-Layer-Wall", "AlphaZero-Stockfish", "Conviction-Sizing", "The-Gauntlet",
                    "Pre-Registration", "Regime-Fragility", "Carry-Edge", "Oracle-Closed-Loop",
                    "Kill-Switch", "ICT-Sovereign-Isolation", "Six-Exit-States", "Eight-Risk-Layers",
                    "Data-Flow-Pipeline", "Discovery-Meta-Finding", "Tenet-1-Statistical-Utility",
                    "Tenet-2-Regime-Appropriateness", "Tenet-3-Know-When-Unreliable",
                    "Tenet-4-Premature-Complexity", "Tenet-5-Orchestration", "Tenet-6-Research-Debt"]
        nparams = sum(len(g) for g in self.config_groups.values())
        out = [self._frontmatter("Alta System — Map of Content", ntype="moc"),
               "> Links: [[Alta-MOC]] · [[Discovery-Ledger]] · [[Oracle-Context]] · [[CONTEXT]] · [[DECISIONS]]\n",
               "# Alta System — Knowledge Graph\n",
               f"> Deterministic map of the Sovereign trading system. **{len(self.modules)} modules · "
               f"{nparams} parameter groups · {len(self.hypotheses)} hypotheses.** "
               f"Generated {TODAY} by `scripts/build_obsidian_graph.py` (re-run to update).\n",
               "## Concepts (the system thesis)\n"]
        out.extend(f"- [[{c}]]" for c in concepts)
        out.append("\n## Subsystems\n")
        def _snippet(s):  # flatten wikilinks to plain text, THEN truncate (never cut a [[link]])
            txt = re.sub(r"\[\[([^\]|]+)\]\]", r"\1", SUBSYSTEM_INTROS.get(s, ""))
            return (txt[:120].rstrip() + "…") if len(txt) > 120 else txt
        out.extend(f"- [[{moc_link(s)}]] — {_snippet(s)}" for s in subs)
        out.append("\n## Indexes\n")
        out.extend(["- [[Parameters-Index]]", "- [[Hypothesis-Ledger]]", "- [[Scripts-Index]]", "- [[Tests-Index]]"])
        out.append("\n## Source docs\n")
        out.extend(["- `docs/ARCHITECTURE.md` — the two-layer doctrine",
                    "- `docs/COMPONENT_CLASSIFICATION.md` — per-component audit",
                    "- `TRADING_PHILOSOPHY.md` — the six tenets", "- `docs/DATA_FLOW.md` — the pipeline"])
        return "00-System-Index.md", "\n".join(out)

    # ---- write / prune ----
    def build(self, *, limit=None, subsystem=None):
        notes: list[tuple[str, str]] = []
        mods = sorted(self.modules)
        if subsystem:
            mods = [r for r in mods if self.modules[r]["subsystem"] == subsystem]
        if limit:
            mods = mods[:limit]
        for rel in mods:
            notes.append(self.render_module(rel))
        notes.extend(self.render_subsystem_mocs())
        notes.extend(self.render_param_notes())
        notes.extend(self.render_hypothesis_notes())
        notes.extend(self.render_indexes())
        notes.append(self.render_top_index())
        return notes

    def write(self, notes, *, dry_run: bool):
        manifest_path = self.out / MANIFEST_NAME
        old = set()
        if manifest_path.exists():
            try:
                old = set(json.loads(manifest_path.read_text()))
            except Exception:
                old = set()
        new = {rp for rp, _ in notes}
        if dry_run:
            return new, (old - new)
        self.out.mkdir(parents=True, exist_ok=True)
        for rp, body in notes:
            fp = self.out / rp
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(body, encoding="utf-8")
        # prune ONLY previously-generated files no longer produced (manifest-scoped)
        pruned = []
        for rp in old - new:
            fp = self.out / rp
            if fp.exists():
                fp.unlink()
                pruned.append(rp)
        manifest_path.write_text(json.dumps(sorted(new), indent=0))
        return new, set(pruned)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Alta-System Obsidian knowledge graph.")
    ap.add_argument("--dry-run", action="store_true", help="compute + report; write nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap module notes (review batch)")
    ap.add_argument("--subsystem", default=None, help="only this subsystem's module notes")
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--vault", default=str(DEFAULT_VAULT))
    args = ap.parse_args()

    if yaml is None:
        print("WARNING: PyYAML not available — parameter notes will be skipped.", file=sys.stderr)

    g = Graph(Path(args.repo).resolve(), Path(args.vault).resolve())
    g.scan_modules()
    g.resolve_internal_deps()
    g.scan_configs()
    g.scan_hypotheses()
    notes = g.build(limit=args.limit, subsystem=args.subsystem)
    written, pruned = g.write(notes, dry_run=args.dry_run)

    nparams = sum(len(gr) for gr in g.config_groups.values())
    kinds: dict[str, int] = {}
    for rp, _ in notes:
        kinds[rp.split("/")[0] if "/" in rp else "root"] = kinds.get(rp.split("/")[0] if "/" in rp else "root", 0) + 1
    print(f"{'DRY-RUN ' if args.dry_run else ''}Alta-System graph @ {g.out}")
    print(f"  modules scanned : {len(g.modules)}")
    print(f"  config groups   : {nparams}  (files: {len(g.config_groups)})")
    print(f"  hypotheses      : {len(g.hypotheses)}")
    print(f"  notes to write  : {len(notes)}  by kind: {dict(sorted(kinds.items()))}")
    print(f"  pruned (stale)  : {len(pruned)}")
    if g.errors:
        print(f"  ⚠️ {len(g.errors)} non-fatal extraction issue(s):")
        for e in g.errors[:10]:
            print(f"     - {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
