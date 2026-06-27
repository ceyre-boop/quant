# Obsidian System Knowledge Graph — runbook

A deterministic, regenerable Obsidian knowledge graph of this repo **as a system**. Two scripts build
it; re-running them *is* the update mechanism (the graph stays in sync with the code).

## What it is

~480 interconnected notes under `<vault>/Trading/System/`, all internal wikilinks resolving:

| Kind | Count | Source | Note |
|------|-------|--------|------|
| Module notes | 295 | every production `.py` (`sovereign/`, `ict/`, `ict-engine/`, `execution/`, `orchestrator/`, `layer1-3/`, …) | real docstring, class/function signatures, **Depends on** + **Used by** (bidirectional import edges), **Config it reads**, **Related hypotheses**, L1/L2/infra/research layer tag |
| Parameter notes | 71 | the 6+ config YAMLs | current values + **Read by** modules |
| Hypothesis notes | 50 | `data/agent/hypothesis_ledger.json` | status · result · mechanism · action |
| Subsystem MOCs | 42 | per subsystem | hand-seeded intro + module list + params + hypotheses |
| Index MOCs | 5 | — | `00-System-Index`, `Parameters-Index`, `Hypothesis-Ledger`, `Scripts-Index`, `Tests-Index` |
| Concept hubs | 20 | hand-authored | the two-layer wall, AlphaZero/Stockfish, the gauntlet, regime-fragility, the six tenets, … (the interpretive layer) |

Entry point: **`Trading/System/00-System-Index.md`**. Linked from `Alta-MOC`, `00-BRAIN/{CONTEXT,DECISIONS}`, and `Trading/Oracle-Context`.

## Build / update

```bash
# 1. The generated layer (facts, extracted — module/param/hypothesis/MOC notes)
python3 scripts/build_obsidian_graph.py            # full run → writes the graph
python3 scripts/build_obsidian_graph.py --dry-run  # counts only, writes nothing
python3 scripts/build_obsidian_graph.py --subsystem forex --limit 20   # review batch
python3 scripts/build_obsidian_graph.py --vault /tmp/vault_test        # test elsewhere

# 2. The hand-authored concept hubs (the interpretive layer)
python3 scripts/build_obsidian_concepts.py
```

Run the generator after code changes to refresh module/param/hypothesis notes. Run the concepts
script only when you change the interpretive notes (it overwrites the 20 hubs).

## Design

- **Code-first / deterministic.** Facts are extracted via `ast` (signatures, docstrings, imports),
  `yaml` (parameter values), and `json` (the ledger) — never inferred. The only hand-authored content
  is the 20 concept hubs and the per-subsystem intro prose (a dict in the generator).
- **Idempotent.** The generator tracks every file it writes in `Trading/System/.graph-manifest.json`
  and, on each run, prunes only *its own* stale notes (manifest-scoped). It never touches anything it
  didn't generate — the hand-authored `Concepts/` notes are safe across re-runs.
- **Self-contained scope.** Writes only under `Trading/System/`. `scripts/` and `tests/` are
  represented as category-index notes, not one-per-file. Brain-hub links (`Alta-MOC` etc.) are wired
  by hand, once.
- **Filenames are globally unique** (Obsidian resolves links by basename): module notes use the dotted
  path (`sovereign.forex.macro_engine`), params use `<configstem>.<group>`, hypotheses use `HYP-XXX`.

## Verify

```bash
# all internal wikilinks resolve (expect 0 unresolved)
python3 - <<'PY'
import re, pathlib
v = pathlib.Path("/Users/taboost/Obsidian/Obsidian")
names = {p.stem for p in v.rglob("*.md") if "/." not in str(p)}
sysd = v/"Trading"/"System"
rx = re.compile(r"\[\[([^\]|#\n]+)")
bad = {}
for p in sysd.rglob("*.md"):
    for t in rx.findall(p.read_text()):
        t=t.strip()
        if t not in names: bad[t]=bad.get(t,0)+1
print("unresolved:", len(bad), bad or "ALL RESOLVE")
PY
```

The vault is **not** a git repo — these notes are written, not committed. Only the two generator
scripts and this runbook live in version control.
