#!/usr/bin/env python3
"""
Edge Factory worker (Track B) — disciplined, single-threaded.
=============================================================

Pulls param-delta hypotheses from the queue and runs each through the GATED EdgePipeline,
then applies the two factory-specific disciplines that keep it from manufacturing false edges:

  1. FAMILY-WISE correction across ALL factory tests (not per-test): a candidate must survive
     Benjamini-Hochberg computed over every p-value the factory has ever produced
     (data/research/factory_ledger.jsonl). The bar RISES as the factory tests more.
  2. RESERVED 2025-present HOLDOUT the factory never optimizes on: a candidate that passes
     search + family-wise BH must ALSO be net-positive on truly-unseen 2025+ data before it is
     promoted to a FACTORY_CANDIDATE.

Candidates are written ONLY to the human review queue (EdgePipeline already does this) —
the factory never touches live config and places no trades. 1 worker; throughput is not the goal.

Usage:  python3 scripts/edge_factory_worker.py [--batch 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUEUE = ROOT / "data" / "research" / "hypothesis_queue.jsonl"
FACTORY_LEDGER = ROOT / "data" / "research" / "factory_ledger.jsonl"
TEST_COUNT = ROOT / "data" / "research" / "factory_test_count.json"
ICT_PAIRS = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "AUDNZD=X"]
HOLDOUT_START = "2025-01-01"
ALPHA = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]


def _save_queue(items: list[dict]) -> None:
    QUEUE.write_text("\n".join(json.dumps(i) for i in items) + ("\n" if items else ""))


def _ledger_pvalues() -> list[tuple[str, float]]:
    if not FACTORY_LEDGER.exists():
        return []
    out = []
    for l in FACTORY_LEDGER.read_text().splitlines():
        if l.strip():
            try:
                r = json.loads(l)
                if isinstance(r.get("p_value"), (int, float)):
                    out.append((r["id"], float(r["p_value"])))
            except Exception:
                continue
    return out


def _survives_family_bh(hid: str, p: float) -> bool:
    """BH over the ENTIRE factory family (all prior factory p-values + this one)."""
    from scripts.derive_hypothesis_pvalues import benjamini_hochberg
    items = [{"id": i, "p_value": pv} for (i, pv) in _ledger_pvalues() if i != hid]
    items.append({"id": hid, "p_value": float(p)})
    benjamini_hochberg(items, ALPHA)
    return next((it.get("bh_status") == "SURVIVES_BH" for it in items if it["id"] == hid), False)


def _holdout_confirm(hyp: dict) -> tuple[bool, str]:
    """Net-positive on truly-unseen 2025+ data? (small sample — directional sanity, not precision.)"""
    end = date.today().isoformat()
    delta = hyp.get("param_delta", {})
    sub = hyp.get("subsystem")
    try:
        if sub == "forex":
            from sovereign.forex.forex_backtester import ForexBacktester
            from scripts.holdout_validation_v014 import _sharpe_from_results, _total_trades
            bt = ForexBacktester(start=HOLDOUT_START, end=end)
            gates = dict(bt.PAIR_VIX_GATES); gates.update(delta.get("PAIR_VIX_GATES", {})); bt.PAIR_VIX_GATES = gates
            res = bt.backtest_all()
            s, n = _sharpe_from_results(res), _total_trades(res)
            return (s > 0.0 and n >= 5), f"2025+ Sharpe {s:.2f} (n={n})"
        elif sub == "ict":
            from scripts.run_ict_backtest import backtest_pair
            base = yaml.safe_load((ROOT / "config" / "ict_params.yml").read_text())
            sc = base.setdefault("scoring", {})
            for k, v in delta.items():
                if k == "weights":
                    sc.setdefault("weights", {}).update(v)
                else:
                    sc[k] = v
            tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False); yaml.safe_dump(base, tmp); tmp.close()
            old = os.environ.get("ICT_CONFIG_PATH"); os.environ["ICT_CONFIG_PATH"] = tmp.name
            try:
                rs = [t.pnl_r for p in ICT_PAIRS for t in backtest_pair(p, start=HOLDOUT_START, end=end)]
            finally:
                if old is not None: os.environ["ICT_CONFIG_PATH"] = old
                else: os.environ.pop("ICT_CONFIG_PATH", None)
                os.unlink(tmp.name)
            mean_r = float(np.mean(rs)) if rs else 0.0
            return (mean_r > 0.0 and len(rs) >= 5), f"2025+ meanR {mean_r:+.3f} (n={len(rs)})"
    except Exception as exc:
        return False, f"holdout error: {exc}"
    return False, "unknown subsystem"


def _record(hid: str, p, status: str, holdout: str = "") -> None:
    FACTORY_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTORY_LEDGER, "a") as f:
        f.write(json.dumps({"id": hid, "p_value": p, "status": status,
                            "holdout": holdout, "tested_at": _now()}) + "\n")
    cnt = json.loads(TEST_COUNT.read_text()) if TEST_COUNT.exists() else {"total": 0}
    cnt["total"] = cnt.get("total", 0) + 1; cnt["updated"] = _now()
    TEST_COUNT.write_text(json.dumps(cnt, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args()

    from sovereign.oracle.edge_pipeline import EdgePipeline
    ep = EdgePipeline()
    items = _load_queue()
    pending = [i for i in items if i.get("status") == "QUEUED"]
    print(f"Edge Factory worker: {len(pending)} queued, processing up to {args.batch}.")

    processed = 0
    for it in items:
        if processed >= args.batch or it.get("status") != "QUEUED":
            continue
        hid = it["id"]
        skip, why = ep.should_skip_hypothesis(it)
        if skip:
            it["status"] = "SKIPPED_DUPLICATE"; it["reason"] = why
            _record(hid, None, "SKIPPED_DUPLICATE"); processed += 1
            print(f"  {hid}: SKIPPED ({why[:60]})"); continue

        verdict = ep.process({"id": hid, "subsystem": it["subsystem"], "param_delta": it["param_delta"]})
        p, status = verdict.get("p_value"), verdict["status"]

        if status == "VALIDATED_PENDING_APPROVAL":
            # Factory disciplines: family-wise BH, then 2025+ holdout.
            if not _survives_family_bh(hid, p):
                status = "FAILS_FAMILYWISE_BH"
                it["reason"] = f"p={p} fails BH across {len(_ledger_pvalues())+1} factory tests"
            else:
                ok, detail = _holdout_confirm(it)
                if ok:
                    status = "FACTORY_CANDIDATE"; it["reason"] = f"survives BH + holdout ({detail})"
                    it["holdout"] = detail
                else:
                    status = "FAILS_HOLDOUT"; it["reason"] = f"failed 2025+ holdout ({detail})"
        else:
            it["reason"] = verdict.get("reason", "")

        it["status"] = status; it["p_value"] = p; it["tested_at"] = _now()
        _record(hid, p, status, it.get("holdout", ""))
        processed += 1
        print(f"  {hid} [{it['subsystem']}] {it.get('label','')}: {status}"
              + (f" (p={p})" if p is not None else ""))

    _save_queue(items)
    cand = sum(1 for i in items if i.get("status") == "FACTORY_CANDIDATE")
    print(f"Done. Processed {processed}. FACTORY_CANDIDATEs in queue: {cand} "
          f"(all PENDING_APPROVAL in edge_review_queue — never auto-traded).")


if __name__ == "__main__":
    main()
