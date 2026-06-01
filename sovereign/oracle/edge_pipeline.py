"""
EdgePipeline — sovereign/oracle/edge_pipeline.py
================================================

Auto-validates a PARAMETER-DELTA hypothesis through the same rigor a human would
apply — delta-significance bootstrap → rolling walk-forward → Benjamini-Hochberg —
and stages the verdict. It is deliberately BOUNDED AT THE COMMIT LINE:

  • Auto-REJECT is terminal and safe (writes ledger, no risk).
  • A hypothesis that passes all gates is written to the REVIEW QUEUE as
    VALIDATED_PENDING_APPROVAL. It is NEVER auto-applied to live config.
  • Only scripts/approve_edge.py (a human action) mutates config, and only with a
    logged rationale — honoring repo non-negotiable #4 ("no live parameter changes
    without logging") and the Oracle "machine never auto-activates" doctrine.

A hypothesis with no `param_delta` (i.e. a natural-language Oracle lesson) is marked
NOT_AUTO_TESTABLE and left for human design — there is no executable signal to test.

Hypothesis shape:
    {"id": "HYP-xxx", "subsystem": "forex"|"ict",
     "param_delta": { ... },          # forex: {"PAIR_VIX_GATES": {"USDJPY=X": 13.0}}
                                       # ict:   {"weights": {"pd_alignment": 0.5}} or {"min_score_to_trade": 6.5}
     "label": "..."}
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rolling_walkforward import run_walkforward
from scripts.derive_hypothesis_pvalues import bootstrap_diff_pvalue, benjamini_hochberg

LEDGER = ROOT / "data" / "agent" / "hypothesis_ledger.json"
QUEUE = ROOT / "data" / "oracle" / "edge_review_queue.json"
PROVEN = ROOT / "data" / "oracle" / "proven_research.json"
REJECTED_DELTAS = ROOT / "data" / "oracle" / "rejected_deltas.json"
TRADES = ROOT / "logs" / "forex_backtest_trades.json"
ICT_PAIRS = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "AUDNZD=X"]
ALPHA = 0.05
SIMILARITY_SKIP = 0.85   # Gap 3: candidate this similar to a past rejection → skip
_REJECT_STATUSES = ("REJECTED", "FRAGILE", "FAILS_BH")
log = logging.getLogger("oracle.edge_pipeline")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested param_delta to {keypath: value}."""
    items = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            items.update(_flatten(v, key + "."))
        else:
            items[key] = v
    return items


def _delta_similarity(a: dict, b: dict) -> float:
    """Structural similarity of two param_deltas in [0,1]. Subset-aware: a candidate
    that is a subset of a rejected delta with matching values scores high."""
    fa, fb = _flatten(a), _flatten(b)
    if not fa or not fb:
        return 0.0
    shared = set(fa) & set(fb)
    if not shared:
        return 0.0
    coverage = len(shared) / min(len(fa), len(fb))   # subset-aware
    closeness = []
    for k in shared:
        va, vb = fa[k], fb[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            denom = max(abs(va), abs(vb), 1e-9)
            closeness.append(1.0 - min(abs(va - vb) / denom, 1.0))
        else:
            closeness.append(1.0 if va == vb else 0.0)
    return 0.5 * coverage + 0.5 * (sum(closeness) / len(closeness))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EdgePipeline:
    """Gated auto-validator. process() runs the full pipeline and returns a verdict
    dict. It writes to the ledger and review queue only — never to live config."""

    def __init__(self, n_boot: int = 5000, seed: int = 7):
        self.n_boot = n_boot
        self.rng = np.random.default_rng(seed)

    # ── public ──────────────────────────────────────────────────────────────
    def process(self, hypothesis: dict) -> dict:
        hid = hypothesis.get("id", "UNKNOWN")
        delta = hypothesis.get("param_delta")
        sub = hypothesis.get("subsystem")

        if not delta:
            return self._finalize(hid, "NOT_AUTO_TESTABLE",
                                  "No param_delta — natural-language lesson, needs human design.",
                                  hypothesis)

        # Gap 3: rejection memory — don't waste a test on a known-dead idea.
        skip, why = self.should_skip_hypothesis(hypothesis)
        if skip:
            return self._finalize(hid, "SKIPPED_DUPLICATE", why, hypothesis)

        # ── Stage 1: delta significance (does the tweak help vs baseline?) ──
        try:
            if sub == "forex":
                p_value, detail, affected = self._forex_significance(delta)
                wf_gates = delta.get("PAIR_VIX_GATES")
            elif sub == "ict":
                p_value, detail, affected = self._ict_significance(delta)
                wf_gates = None  # ICT has no forex walk-forward; significance + BH only
            else:
                return self._finalize(hid, "ERROR", f"Unknown subsystem '{sub}'.", hypothesis)
        except Exception as exc:
            return self._finalize(hid, "ERROR", f"Significance test failed: {exc}", hypothesis)

        if p_value != p_value or p_value >= 0.10:   # NaN or insignificant
            return self._finalize(hid, "REJECTED",
                                  f"Delta-bootstrap p={p_value:.3f} (≥0.10). {detail}",
                                  hypothesis, p_value=p_value)

        # ── Stage 2: rolling walk-forward (forex only) ─────────────────────
        wf = None
        if sub == "forex":
            wf = run_walkforward(pair_vix_gates=delta.get("PAIR_VIX_GATES"),
                                 signal_weights=delta.get("signal_weights"),
                                 save=False, verbose=False)
            if not wf["all_positive"]:
                fragile = [r["test_year"] for r in wf["windows"] if r["test_sharpe"] <= 0]
                return self._finalize(hid, "FRAGILE",
                                      f"p={p_value:.3f} but walk-forward fragile (negative in {fragile}).",
                                      hypothesis, p_value=p_value, walkforward=wf)

        # ── Stage 3: Benjamini-Hochberg vs the ledger's p-values ───────────
        if not self._survives_bh(hid, p_value):
            return self._finalize(hid, "FAILS_BH",
                                  f"p={p_value:.3f} individually but fails family-wise BH correction.",
                                  hypothesis, p_value=p_value, walkforward=wf)

        # ── Stage 4: all gates pass → STAGE for human approval (no commit) ─
        return self._finalize(hid, "VALIDATED_PENDING_APPROVAL",
                              f"p={p_value:.4f}, survives BH"
                              + (f", walk-forward OOS avg {wf['avg_oos_sharpe']}" if wf else "")
                              + ". Awaiting human approval (approve_edge.py).",
                              hypothesis, p_value=p_value, walkforward=wf, stage=True)

    # ── significance runners ──────────────────────────────────────────────
    def _forex_returns(self, pair_vix_gates: dict | None, signal_weights: dict | None,
                       only_pairs: list | None) -> list:
        from sovereign.forex.forex_backtester import ForexBacktester
        bt = ForexBacktester(start="2015-01-01", end="2024-12-31", signal_weights=signal_weights or None)
        if pair_vix_gates:
            gates = dict(bt.PAIR_VIX_GATES); gates.update(pair_vix_gates); bt.PAIR_VIX_GATES = gates
        bt.backtest_all()
        data = json.loads(TRADES.read_text())
        return [t["pnl_pct"] for pair, trades in data.items()
                if (only_pairs is None or pair in only_pairs) for t in trades]

    def _forex_significance(self, delta: dict):
        gates = delta.get("PAIR_VIX_GATES", {})
        weights = delta.get("signal_weights", {})
        # VIX-gate deltas affect only the gated pairs; signal-weight deltas affect ALL pairs.
        only_pairs = list(gates.keys()) if (gates and not weights) else None
        r_with = self._forex_returns(gates or None, weights or None, only_pairs)
        r_base = self._forex_returns(None, None, only_pairs)
        p = bootstrap_diff_pvalue(r_with, r_base, self.n_boot, self.rng)
        return p, f"meanR_with={np.mean(r_with):.4f}(n={len(r_with)}) vs base={np.mean(r_base):.4f}", only_pairs

    def _ict_returns(self, weight_delta: dict | None) -> list:
        """Run the ICT backtest with an optional scoring delta via a temp config."""
        from scripts.run_ict_backtest import backtest_pair
        cfg_path = None
        if weight_delta:
            base = yaml.safe_load((ROOT / "config" / "ict_params.yml").read_text())
            scoring = base.setdefault("scoring", {})
            for k, v in weight_delta.items():
                if k == "weights":
                    scoring.setdefault("weights", {}).update(v)
                else:
                    scoring[k] = v
            tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
            yaml.safe_dump(base, tmp); tmp.close()
            cfg_path = tmp.name
        old = os.environ.get("ICT_CONFIG_PATH")
        if cfg_path:
            os.environ["ICT_CONFIG_PATH"] = cfg_path
        try:
            r = [t.pnl_r for pair in ICT_PAIRS for t in backtest_pair(pair)]
        finally:
            if cfg_path:
                if old is not None:
                    os.environ["ICT_CONFIG_PATH"] = old
                else:
                    os.environ.pop("ICT_CONFIG_PATH", None)
                os.unlink(cfg_path)
        return r

    def _ict_significance(self, delta: dict):
        r_with = self._ict_returns(delta)
        r_base = self._ict_returns(None)
        p = bootstrap_diff_pvalue(r_with, r_base, self.n_boot, self.rng)
        return p, f"meanR_with={np.mean(r_with):.4f}(n={len(r_with)}) vs base={np.mean(r_base):.4f}", None

    # ── Gap 3: rejection memory ───────────────────────────────────────────
    def _load_rejected_deltas(self) -> list:
        try:
            return json.loads(REJECTED_DELTAS.read_text()).get("rejected", [])
        except Exception:
            return []

    def should_skip_hypothesis(self, candidate: dict) -> tuple[bool, str | None]:
        """Skip if this candidate is near-identical to a past rejection. Structural
        similarity on param_delta (precise); coarse keyword overlap on legacy prose."""
        delta = candidate.get("param_delta") or {}
        sub = candidate.get("subsystem")
        # 1) structural match against EdgePipeline's own recorded rejections
        for r in self._load_rejected_deltas():
            if r.get("subsystem") and sub and r["subsystem"] != sub:
                continue
            sim = _delta_similarity(delta, r.get("param_delta") or {})
            if sim >= SIMILARITY_SKIP:
                return True, (f"NEAR_DUPLICATE_REJECTED: {sim:.2f} similar to {r.get('id')} "
                              f"(rejected {r.get('date','?')}, status {r.get('status')}, "
                              f"p={r.get('p_value')}).")
        # 2) coarse keyword overlap against legacy prose rejections
        label = (candidate.get("label") or "").lower()
        if label:
            toks = {w for w in label.replace(",", " ").split() if len(w) > 3}
            try:
                prose = json.loads(PROVEN.read_text()).get("rejected_hypotheses", [])
            except Exception:
                prose = []
            for r in prose:
                ftoks = {w for w in str(r.get("finding", "")).lower().split() if len(w) > 3}
                if toks and ftoks and len(toks & ftoks) / len(toks) > 0.6:
                    return True, (f"NEAR_DUPLICATE_REJECTED (prose): overlaps {r.get('id')} "
                                  f"— '{str(r.get('finding',''))[:60]}'.")
        return False, None

    def analyze_rejection_clusters(self) -> dict:
        """If a param family is ≥8 and entirely rejected, deprioritize it for future search."""
        rejected = self._load_rejected_deltas()
        families: dict[str, int] = {}
        for r in rejected:
            for keypath in _flatten(r.get("param_delta") or {}):
                fam = f"{r.get('subsystem')}:{keypath.split('.')[0]}"
                families[fam] = families.get(fam, 0) + 1
        low_priority = {fam: n for fam, n in families.items() if n >= 8}
        if low_priority:
            try:
                data = json.loads(PROVEN.read_text())
                data["low_priority_families"] = {"updated": _now(), "families": low_priority}
                PROVEN.write_text(json.dumps(data, indent=2))
            except Exception as exc:
                log.warning("cluster note write failed: %s", exc)
            for fam, n in low_priority.items():
                self._message_colin(f"LEARNED: {fam} family — {n} rejected. Deprioritizing in search.")
        return low_priority

    def _message_colin(self, text: str) -> None:
        msgs = ROOT / "data" / "agent" / "messages_to_colin.json"
        try:
            data = json.loads(msgs.read_text()) if msgs.exists() else {"messages": []}
            data.setdefault("messages", []).insert(0, {
                "id": f"edgepipe-{_now()[:19].replace(':', '').replace('-', '')}",
                "timestamp": _now(), "priority": "FYI", "source": "EDGE_PIPELINE",
                "subject": text[:80], "message": text, "action_required": False})
            data["messages"] = data["messages"][:80]
            msgs.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _record_rejection(self, verdict: dict) -> None:
        if verdict.get("status") not in _REJECT_STATUSES or not verdict.get("param_delta"):
            return
        try:
            store = json.loads(REJECTED_DELTAS.read_text()) if REJECTED_DELTAS.exists() else {"rejected": []}
        except Exception:
            store = {"rejected": []}
        store.setdefault("rejected", [])
        store["rejected"] = [r for r in store["rejected"] if r.get("id") != verdict["id"]]
        store["rejected"].append({
            "id": verdict["id"], "subsystem": verdict.get("subsystem"),
            "param_delta": verdict.get("param_delta"), "status": verdict["status"],
            "p_value": verdict.get("p_value"), "date": verdict.get("evaluated_at", _now())[:10],
        })
        REJECTED_DELTAS.parent.mkdir(parents=True, exist_ok=True)
        REJECTED_DELTAS.write_text(json.dumps(store, indent=2))

    # ── BH ────────────────────────────────────────────────────────────────
    def _survives_bh(self, hid: str, p_value: float) -> bool:
        items = [{"id": hid, "p_value": float(p_value)}]
        try:
            led = json.loads(LEDGER.read_text())
            for h in led.get("ledger", []) + led.get("hypotheses", []):
                if isinstance(h, dict) and isinstance(h.get("p_value"), (int, float)) and h.get("id") != hid:
                    items.append({"id": h["id"], "p_value": float(h["p_value"])})
        except Exception:
            pass
        benjamini_hochberg(items, ALPHA)
        return next((it.get("bh_status") == "SURVIVES_BH" for it in items if it["id"] == hid), False)

    # ── finalize: write ledger + (if staged) review queue. NEVER config. ──
    def _finalize(self, hid, status, reason, hypothesis, p_value=None, walkforward=None, stage=False) -> dict:
        verdict = {
            "id": hid, "status": status, "reason": reason,
            "p_value": round(p_value, 4) if isinstance(p_value, float) and p_value == p_value else None,
            "subsystem": hypothesis.get("subsystem"), "param_delta": hypothesis.get("param_delta"),
            "walkforward": walkforward, "evaluated_at": _now(),
        }
        self._update_ledger(hid, verdict)
        if stage:
            self._enqueue(verdict)
        self._record_rejection(verdict)   # Gap 3: remember rejections for future dedup
        log.info("EdgePipeline %s → %s", hid, status)
        return verdict

    def _update_ledger(self, hid, verdict):
        try:
            led = json.loads(LEDGER.read_text())
            touched = False
            for arr in ("ledger", "hypotheses"):
                for h in led.get(arr, []):
                    if isinstance(h, dict) and h.get("id") == hid:
                        h["edge_pipeline_status"] = verdict["status"]
                        h["edge_pipeline_reason"] = verdict["reason"]
                        if verdict["p_value"] is not None:
                            h["p_value"] = verdict["p_value"]
                        touched = True
            if touched:
                LEDGER.write_text(json.dumps(led, indent=2))
        except Exception as exc:
            log.warning("ledger update failed: %s", exc)

    def _enqueue(self, verdict):
        QUEUE.parent.mkdir(parents=True, exist_ok=True)
        q = json.loads(QUEUE.read_text()) if QUEUE.exists() else {"pending": []}
        q.setdefault("pending", [])
        q["pending"] = [v for v in q["pending"] if v.get("id") != verdict["id"]]
        q["pending"].append(verdict)
        q["last_updated"] = _now()
        QUEUE.write_text(json.dumps(q, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="dry-run HYP-044 (must auto-REJECT)")
    args = ap.parse_args()
    if args.demo:
        v = EdgePipeline(n_boot=2000).process({
            "id": "TEST-044", "subsystem": "forex", "label": "VIX 15→13 (known noise)",
            "param_delta": {"PAIR_VIX_GATES": {"USDJPY=X": 13.0, "AUDNZD=X": 13.0}},
        })
        print(json.dumps(v, indent=2))
