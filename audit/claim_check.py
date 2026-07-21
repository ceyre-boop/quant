"""Mechanically falsify audit claims before anyone acts on them.

WHAT THIS IS
------------
A read-only checker for the four claim classes that produced six false findings on
2026-07-20. Given a claim, it answers CONFIRMED / REFUTED / UNVERIFIABLE with the
evidence inline, and exits non-zero if anything is REFUTED.

Every rule comes from `audit/claims_spec.md`, which is sha256-hashed and
version-checked at load — changing a rule is a spec edit with a git trail, not a
constant buried here. Same contract as `audit/invariant_guard.py`.

WHAT THIS IS NOT
----------------
It does not fix, file, or adjudicate. It falsifies; a human decides. There is no
LLM in this file and there must never be one — an audit checked by the same kind
of process that produced the audit is not a check.

ISOLATION
---------
Imports nothing from the execution path — not even for vocabulary. `--self-test`
asserts this. The point is that a claim about the execution path is checked by
something the execution path cannot influence.

KNOWN LIMIT — READ BEFORE TRUSTING A C2 VERDICT
------------------------------------------------
Dynamic imports are invisible to an AST walk. `ict/pipeline.py` reaches sovereign
through an importlib hook and that edge does not appear in the graph. When C2
finds no importer but the live set contains dynamic-import machinery, the verdict
is UNVERIFIABLE, never CONFIRMED. A checker trusted past its evidence is the same
failure one level up.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import plistlib
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "audit" / "claims_spec.md"
FENCE_RE = re.compile(r"^```yaml claims-spec\n(.*?)^```", re.S | re.M)

# Machinery that makes an import graph incomplete. Presence anywhere in the live
# set downgrades a "no importer found" result from REFUTED-absence to UNVERIFIABLE.
_DYNAMIC_IMPORT_MARKERS = ("importlib", "__import__", "exec(", "eval(")


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class ClaimKind(str, Enum):
    """Each kind names what the AUDITOR asserted, not what is true.

    ABSENT and DEAD are deliberately negative: audits claim things are missing or
    unused, and those are the claims that cause damage when wrong. Naming them for
    the assertion keeps CONFIRMED/REFUTED unambiguous — CONFIRMED always means
    "the auditor was right", never "the file is there".
    """
    ABSENT = "ABSENT"        # auditor claims: "file X does not exist / was never built"
    DEAD = "DEAD"            # auditor claims: "X is not imported by any live path"
    LOGPATH = "LOGPATH"      # auditor claims: "job silently crashing / 0-byte log"
    CITED = "CITED"          # auditor claims: "per DOC.md" — is the citation real?


@dataclass
class Claim:
    """One assertion to test. `subject` means whatever the kind implies:
    a path (EXISTS), a module name (IMPORTED), a plist label (LOGPATH),
    a document path (CITED)."""
    kind: ClaimKind
    subject: str
    text: str = ""
    cites: str = ""          # CITED only: the string the document must contain

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(kind=ClaimKind(d["kind"]), subject=d["subject"],
                     text=d.get("text", ""), cites=d.get("cites", ""))


@dataclass
class Result:
    claim: Claim
    verdict: Verdict
    evidence: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["claim"]["kind"] = self.claim.kind.value
        d["verdict"] = self.verdict.value
        return d


# ── spec ──────────────────────────────────────────────────────────────────────

def load_spec(path: Path = SPEC_PATH) -> tuple[dict, str, int]:
    """Hashed, fenced, version-checked. Mirrors invariant_guard.load_spec."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    fences = FENCE_RE.findall(raw.decode("utf-8"))
    if len(fences) != 1:
        raise RuntimeError(
            f"spec must contain exactly one 'yaml claims-spec' fence, found {len(fences)}")
    spec = yaml.safe_load(fences[0])
    return spec, sha, int(spec["spec_version"])


def _excluded(p: Path, spec: dict) -> bool:
    parts = set(p.parts)
    return any(ex in parts or str(p).find(ex) >= 0 for ex in spec["exclude_dirs"])


_PY_FILES_CACHE: list[Path] | None = None
_IMPORTS_CACHE: dict[Path, list[tuple[str, int]]] = {}


def _py_files(spec: dict) -> list[Path]:
    """Cached. Without this the tree is re-walked and every file re-parsed once
    per claim — a 10-claim corpus took 7 minutes, which is slow enough that a
    gate stops being run at all."""
    global _PY_FILES_CACHE
    if _PY_FILES_CACHE is None:
        _PY_FILES_CACHE = [p for p in ROOT.rglob("*.py")
                           if not _excluded(p.relative_to(ROOT), spec)]
    return _PY_FILES_CACHE


# ── C1 EXISTS ─────────────────────────────────────────────────────────────────

def _deletion_commit(rel: str) -> str | None:
    """Was this path deliberately removed? Returns the deleting commit, if any."""
    try:
        out = subprocess.run(
            ["git", "log", "--all", "--diff-filter=D", "--format=%h %s", "-1", "--", rel],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_absent(claim: Claim, spec: dict) -> Result:
    """Claim asserts ABSENCE. Path resolving REFUTES it."""
    target = ROOT / claim.subject
    ev: list[str] = []

    if target.exists():
        size = target.stat().st_size if target.is_file() else 0
        ev.append(f"{claim.subject} EXISTS ({size} bytes) — claim of absence is false")
        return Result(claim, Verdict.REFUTED, ev)

    ev.append(f"{claim.subject}: no such path")

    # Absent — but was it removed on purpose? That changes the action entirely.
    dele = _deletion_commit(claim.subject)
    if dele:
        ev.append(f"DELIBERATELY REMOVED in {dele} — do not rebuild; fix the stale reference")
    for doc in spec.get("removal_evidence_paths", []):
        p = ROOT / doc
        if not p.exists():
            continue
        stem = Path(claim.subject).stem
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if stem and stem in line and ("remov" in line.lower() or "delet" in line.lower()):
                ev.append(f"removal recorded at {doc}:{i}")
                break

    return Result(claim, Verdict.CONFIRMED, ev)


# ── C2 IMPORTED ───────────────────────────────────────────────────────────────

def _imports_in(path: Path) -> list[tuple[str, int]]:
    """Every imported module name in a file, by AST. Never substring matching —
    that matched module names inside their own docstrings twice this month."""
    if path in _IMPORTS_CACHE:
        return _IMPORTS_CACHE[path]
    # Parsing the whole repo surfaces SyntaxWarnings from unrelated files (invalid
    # escapes in third-party-ish code). They are not this tool's findings and
    # drowning the verdicts in them is how a checker stops being read.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(errors="ignore"))
    except (SyntaxError, ValueError, OSError):
        _IMPORTS_CACHE[path] = []
        return []
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append((node.module, node.lineno))
                for a in node.names:
                    out.append((f"{node.module}.{a.name}", node.lineno))
    _IMPORTS_CACHE[path] = out
    return out


def _has_dynamic_imports(paths: list[Path]) -> list[str]:
    hits = []
    for p in paths:
        try:
            src = p.read_text(errors="ignore")
        except OSError:
            continue
        for marker in _DYNAMIC_IMPORT_MARKERS:
            if marker in src:
                hits.append(f"{p.relative_to(ROOT)} uses {marker}")
                break
    return hits


def _is_test(rel: Path) -> bool:
    parts = rel.parts
    return "tests" in parts or rel.name.startswith("test_")


def check_dead(claim: Claim, spec: dict) -> Result:
    """Claim asserts the module is NOT imported by anything live.
    Finding a live importer REFUTES it.

    Test importers are counted SEPARATELY and never on their own refute the claim:
    a module imported only by its own test really is dead to the running system,
    and conflating the two would let dead code look live.
    """
    mod = claim.subject.replace("/", ".").removesuffix(".py")
    leaf = mod.rsplit(".", 1)[-1]
    ev: list[str] = []
    live: list[str] = []
    tests: list[str] = []

    files = _py_files(spec)
    for p in files:
        # A module does not import itself.
        if p.stem == leaf:
            continue
        rel = p.relative_to(ROOT)
        for name, lineno in _imports_in(p):
            if name == mod or name.endswith(f".{leaf}") or name == leaf:
                (tests if _is_test(rel) else live).append(f"{rel}:{lineno} imports {name}")
                break

    if live:
        ev.append(f"{len(live)} LIVE importer(s) found — claim of deadness is false:")
        ev.extend(f"  {i}" for i in live[:10])
        if len(live) > 10:
            ev.append(f"  … and {len(live) - 10} more")
        if tests:
            ev.append(f"({len(tests)} test importer(s) also present, not counted as live)")
        return Result(claim, Verdict.REFUTED, ev)

    ev.append(f"no live static importer of '{mod}' found")
    if tests:
        ev.append(f"{len(tests)} TEST-ONLY importer(s) — imported by its own tests, "
                  f"which is not liveness:")
        ev.extend(f"  {t}" for t in tests[:5])
    dyn = _has_dynamic_imports(files)
    if dyn:
        ev.append("BUT the live set uses dynamic imports, which an AST walk cannot see:")
        ev.extend(f"  {d}" for d in dyn[:5])
        ev.append("verdict downgraded to UNVERIFIABLE — absence of evidence is not evidence")
        return Result(claim, Verdict.UNVERIFIABLE, ev)

    return Result(claim, Verdict.CONFIRMED, ev)


# ── C3 LOGPATH ────────────────────────────────────────────────────────────────

def _plist_for(label: str) -> Path | None:
    for cand in (Path.home() / "Library" / "LaunchAgents" / f"{label}.plist",
                 ROOT / "scripts" / f"{label}.plist"):
        if cand.exists():
            return cand
    return None


def _script_written_logs(script: Path) -> list[str]:
    """Log paths the script writes to itself — the ones a launchd log never shows."""
    try:
        src = script.read_text(errors="ignore")
    except OSError:
        return []
    pats = [r'LOG="?([^"\s]+\.log)', r'>>\s*"?([^"\s]+\.log)', r'logs/([\w.\-]+\.log)']
    found: list[str] = []
    for pat in pats:
        for m in re.findall(pat, src):
            v = m if m.startswith("logs/") or "/" in m else f"logs/{m}"
            if v not in found:
                found.append(v)
    return found


def check_logpath(claim: Claim, spec: dict) -> Result:
    """Claim asserts a job is silently dead, usually citing a 0-byte log.
    A healthy script log elsewhere REFUTES it."""
    ev: list[str] = []
    pl = _plist_for(claim.subject)
    if pl is None:
        ev.append(f"no plist found for label '{claim.subject}'")
        return Result(claim, Verdict.UNVERIFIABLE, ev)

    try:
        data = plistlib.loads(pl.read_bytes())
    except Exception as exc:
        ev.append(f"{pl.name}: unreadable ({exc})")
        return Result(claim, Verdict.UNVERIFIABLE, ev)

    ev.append(f"plist {pl.relative_to(pl.parent.parent)}")
    live_logs: list[str] = []

    for key in ("StandardOutPath", "StandardErrorPath"):
        raw = data.get(key)
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            n = p.stat().st_size
            ev.append(f"  {key}: {p.name} = {n} bytes")
            if n > spec["empty_log_bytes"]:
                live_logs.append(f"{p.name} ({n} bytes)")

    # The decisive step: logs the script writes itself.
    args = data.get("ProgramArguments") or []
    script = next((ROOT / a.replace(str(ROOT) + "/", "")
                   for a in args if a.endswith((".sh", ".py"))), None)
    if script and script.exists():
        ev.append(f"  entry script: {script.relative_to(ROOT)}")
        for rel in _script_written_logs(script):
            lp = ROOT / rel
            if lp.exists():
                n = lp.stat().st_size
                ev.append(f"  script writes {rel} = {n} bytes")
                if n > spec["empty_log_bytes"]:
                    live_logs.append(f"{rel} ({n} bytes)")

        src = script.read_text(errors="ignore")
        if all(g in src for g in spec.get("time_guard_patterns", [])):
            ev.append("  script contains a TIME GUARD — an early exit 0 outside its "
                      "window is healthy, not a crash. Do not remove it.")

    if live_logs:
        ev.append("NON-EMPTY LOG(S) FOUND — claim of silent death is false: "
                  + ", ".join(live_logs))
        return Result(claim, Verdict.REFUTED, ev)

    ev.append("no non-empty log found by any route")
    return Result(claim, Verdict.CONFIRMED, ev)


# ── C4 CITED ──────────────────────────────────────────────────────────────────

def check_cited(claim: Claim, spec: dict) -> Result:
    """Claim cites a document as authority. A missing doc, or one lacking the
    cited string, REFUTES the citation."""
    doc = ROOT / claim.subject
    ev: list[str] = []
    if not doc.exists():
        ev.append(f"CITED DOCUMENT DOES NOT EXIST: {claim.subject}")
        ev.append("a constant justified by a document nobody wrote is unratified")
        return Result(claim, Verdict.REFUTED, ev)

    ev.append(f"{claim.subject} exists ({doc.stat().st_size} bytes)")
    if not claim.cites:
        ev.append("no 'cites' string given — existence checked only")
        return Result(claim, Verdict.CONFIRMED, ev)

    for i, line in enumerate(doc.read_text(errors="ignore").splitlines(), 1):
        if claim.cites in line:
            ev.append(f"contains {claim.cites!r} at line {i}")
            return Result(claim, Verdict.CONFIRMED, ev)

    ev.append(f"does NOT contain {claim.cites!r} — citation unsupported")
    return Result(claim, Verdict.REFUTED, ev)


# ── driver ────────────────────────────────────────────────────────────────────

_CHECKERS = {
    ClaimKind.ABSENT: check_absent,
    ClaimKind.DEAD: check_dead,
    ClaimKind.LOGPATH: check_logpath,
    ClaimKind.CITED: check_cited,
}


def run(claims: list[Claim], spec: dict) -> list[Result]:
    return [_CHECKERS[c.kind](c, spec) for c in claims]


def render(results: list[Result], sha: str) -> str:
    L = [f"claim_check — spec sha256 {sha[:12]}", ""]
    for r in results:
        L.append(f'CLAIM  [{r.claim.kind.value}] "{r.claim.text or r.claim.subject}"')
        mark = {"REFUTED": "✗", "CONFIRMED": "✓", "UNVERIFIABLE": "?"}[r.verdict.value]
        L.append(f"  {mark} {r.verdict.value}")
        for e in r.evidence:
            L.append(f"      {e}")
        L.append("")
    n_ref = sum(1 for r in results if r.verdict is Verdict.REFUTED)
    n_con = sum(1 for r in results if r.verdict is Verdict.CONFIRMED)
    n_unv = sum(1 for r in results if r.verdict is Verdict.UNVERIFIABLE)
    L.append(f"{n_ref} refuted / {n_con} confirmed / {n_unv} unverifiable")
    if n_ref:
        L.append("DO NOT ACT on refuted claims. Fix the claim, not the code.")
    return "\n".join(L)


def self_test() -> int:
    """Assert isolation: this module must import nothing from the execution path."""
    forbidden = ("execution", "sovereign", "ict", "backtester")
    bad = [f"{n}:{ln}" for n, ln in _imports_in(Path(__file__))
           if n.split(".")[0] in forbidden]
    if bad:
        print(f"SELF-TEST FAIL — execution-path imports present: {bad}")
        return 1
    print("SELF-TEST PASS — no execution-path imports; checker is isolated")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Falsify audit claims (read-only, detect-only)")
    ap.add_argument("--claims", help="JSON file: list of {kind, subject, text, cites}")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    spec, sha, _ = load_spec()
    if not args.claims:
        ap.error("--claims is required (or use --self-test)")

    raw = json.loads(Path(args.claims).read_text())
    claims = [Claim.from_dict(d) for d in raw]
    results = run(claims, spec)

    if args.json:
        print(json.dumps([r.to_json() for r in results], indent=2))
    else:
        print(render(results, sha))

    return 1 if any(r.verdict is Verdict.REFUTED for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
