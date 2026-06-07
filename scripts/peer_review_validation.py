"""
scripts/peer_review_validation.py — FVG peer review validation suite

Four tests responding to standard statistical critiques of the FVG universal backbone finding:
  Test 1 — FVG Ablation:      Can ANY combo survive holdout without FVG?
  Test 2 — FVG Alone:         Is FVG a filter or a standalone signal?
  Test 3 — Rolling Holdout:   Is FVG dominance structural or period-specific? (walk-forward)
  Test 4 — Restricted Pool:   Does FVG dominate in C(7,3)=35 market-structure-only space?

Output: stdout tables + data/research/peer_review_responses.json

Usage:
    python3 scripts/peer_review_validation.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIST_PATH  = ROOT / "data" / "indicators" / "history.parquet"
GREEN_PATH = ROOT / "data" / "indicators" / "green_conditions.json"
OUT_DIR    = ROOT / "data" / "research"
OUT_PATH   = OUT_DIR / "peer_review_responses.json"

INDICATOR_NAMES = [
    # TREND (7)
    "adx", "ema_cross", "supertrend", "ichimoku", "parabolic_sar", "aroon", "vwap_dev",
    # MOMENTUM (7)
    "rsi", "macd", "stochastic", "cci", "williams_r", "roc", "mfi",
    # VOLATILITY (5)
    "bollinger_bands", "atr_ratio", "keltner", "donchian", "hist_vol",
    # VOLUME (5)
    "obv", "vol_trend", "vwap_ratio", "acc_dist", "cmf",
    # MARKET STRUCTURE (6)
    "fvg", "bos", "session_hl", "displacement", "liq_sweep", "order_block",
]

MIN_SAMPLES   = 20
MIN_HIT_RATE  = 0.53
MIN_HOLDOUT_N = 10
OVERFIT_DELTA = 0.20

TRAIN_START   = "2015-01-01"
TRAIN_END     = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END   = "2024-12-31"


# ─── Core shared functions ─────────────────────────────────────────────────────

def _sweep_green(
    hist_train: pd.DataFrame,
    pool: list[str],
    top_n: int = 10,
) -> tuple[dict, int]:
    """
    Sweep all C(|pool|,3) triple combos on training slice.
    Returns (green_conditions_dict, total_candidates_before_top_n).
    """
    total_candidates = 0
    green: dict = {}

    for pair in sorted(hist_train["pair"].unique()):
        g   = hist_train[hist_train["pair"] == pair].dropna(subset=["fwd_10d"])
        fwd = g["fwd_10d"].astype(float)
        best_long:  list[dict] = []
        best_short: list[dict] = []

        for a, b, c in itertools.combinations(pool, 3):
            ca, cb, cc = f"state_{a}", f"state_{b}", f"state_{c}"
            if ca not in g.columns or cb not in g.columns or cc not in g.columns:
                continue
            for sign, bucket in [(1, best_long), (-1, best_short)]:
                mask = (g[ca] == sign) & (g[cb] == sign) & (g[cc] == sign)
                n = int(mask.sum())
                if n < MIN_SAMPLES:
                    continue
                hr = float(
                    (fwd[mask] > 0).mean() if sign == 1 else (fwd[mask] < 0).mean()
                )
                if hr >= MIN_HIT_RATE:
                    total_candidates += 1
                    bucket.append({"indicators": [a, b, c], "hit_rate": hr, "n": n})

        best_long.sort(key=lambda x: x["hit_rate"], reverse=True)
        best_short.sort(key=lambda x: x["hit_rate"], reverse=True)
        green[pair] = {
            "best_long":  best_long[:top_n],
            "best_short": best_short[:top_n],
        }

    return green, total_candidates


def _validate_green(hist_holdout: pd.DataFrame, green: dict) -> dict:
    """
    Validate every combo in green_conditions dict against holdout slice.
    Returns verdict counts, FVG breakdown, best non-FVG combo, validated combo list.
    """
    real = weak = overfit = insuf = 0
    fvg_in_real    = 0
    real_hrs: list[float] = []
    best_non_fvg_hr    = 0.0
    best_non_fvg_combo: dict | None = None
    validated_combos: list[dict] = []

    for pair, conds in green.items():
        g   = hist_holdout[hist_holdout["pair"] == pair].dropna(subset=["fwd_10d"])
        fwd = g["fwd_10d"].astype(float)

        for dir_key, sign in [("best_long", 1), ("best_short", -1)]:
            direction = "LONG" if sign == 1 else "SHORT"
            for cond in conds.get(dir_key, []):
                inds = cond["indicators"]
                ca, cb, cc = f"state_{inds[0]}", f"state_{inds[1]}", f"state_{inds[2]}"
                if ca not in g.columns or cb not in g.columns or cc not in g.columns:
                    insuf += 1
                    continue

                mask = (g[ca] == sign) & (g[cb] == sign) & (g[cc] == sign)
                n_h  = int(mask.sum())
                if n_h == 0:
                    insuf += 1
                    continue

                hr_h  = float(
                    (fwd[mask] > 0).mean() if sign == 1 else (fwd[mask] < 0).mean()
                )
                delta = cond["hit_rate"] - hr_h

                if n_h < MIN_HOLDOUT_N:
                    insuf += 1
                elif abs(delta) < 0.10:
                    real += 1
                    real_hrs.append(hr_h)
                    has_fvg = "fvg" in inds
                    if has_fvg:
                        fvg_in_real += 1
                    else:
                        if hr_h > best_non_fvg_hr:
                            best_non_fvg_hr    = hr_h
                            best_non_fvg_combo = {
                                "pair": pair, "direction": direction,
                                "indicators": inds,
                                "train_hr":   round(cond["hit_rate"], 4),
                                "holdout_hr": round(hr_h, 4),
                                "n_holdout":  n_h,
                            }
                    validated_combos.append({
                        "pair":       pair,
                        "direction":  direction,
                        "indicators": inds,
                        "holdout_hr": round(hr_h, 4),
                        "has_fvg":    has_fvg,
                        "n":          n_h,
                    })
                elif abs(delta) < OVERFIT_DELTA:
                    weak += 1
                else:
                    overfit += 1

    return {
        "real": real, "weak": weak, "overfit": overfit, "insuf": insuf,
        "fvg_in_real":       fvg_in_real,
        "fvg_rate":          round(fvg_in_real / real, 4) if real else 0.0,
        "best_non_fvg_hr":   round(best_non_fvg_hr, 4),
        "best_non_fvg_combo": best_non_fvg_combo,
        "avg_real_hr":       round(sum(real_hrs) / len(real_hrs), 4) if real_hrs else 0.0,
        "validated_combos":  validated_combos,
    }


# ─── Test 1 — FVG Ablation ────────────────────────────────────────────────────

def test_1_ablation(hist: pd.DataFrame) -> dict:
    n_combos_full = len(list(itertools.combinations(INDICATOR_NAMES, 3)))
    pool   = [n for n in INDICATOR_NAMES if n != "fvg"]
    n_combos = len(list(itertools.combinations(pool, 3)))

    print(f"\n{'═'*65}")
    print("TEST 1 — FVG ABLATION")
    print(f"Remove FVG from pool: {len(INDICATOR_NAMES)}→{len(pool)} indicators | "
          f"C({len(pool)},3) = {n_combos} combos  (was {n_combos_full})")
    print(f"Train: {TRAIN_START}–{TRAIN_END} | Holdout: {HOLDOUT_START}–{HOLDOUT_END}")
    print(f"{'─'*65}")

    t0 = time.time()
    hist_train   = hist[(hist["date"] >= TRAIN_START) & (hist["date"] <= TRAIN_END)]
    hist_holdout = hist[(hist["date"] >= HOLDOUT_START) & (hist["date"] <= HOLDOUT_END)]

    print("Sweeping 29-indicator pool (no FVG)...")
    green, total_candidates = _sweep_green(hist_train, pool)
    total_stored = sum(len(v["best_long"]) + len(v["best_short"]) for v in green.values())

    print(f"\n{'PAIR':<10} {'LONG_CANDS':>10} {'SHORT_CANDS':>12}")
    print("-" * 36)
    for pair in sorted(green):
        print(f"{pair:<10} {len(green[pair]['best_long']):>10} {len(green[pair]['best_short']):>12}")

    print(f"\nTotal candidates passing thresholds: {total_candidates}")
    print(f"Stored for holdout validation (top {10}/pair/dir): {total_stored}")
    print(f"\nValidating {total_stored} conditions on 2024 holdout...")

    v = _validate_green(hist_holdout, green)
    validated_without_fvg = v["real"]  # ALL combos in this test are non-FVG

    print(f"\n{'VERDICT':<22} {'COUNT':>6}")
    print("-" * 30)
    for label, key in [("REAL_SIGNAL", "real"), ("WEAK_SIGNAL", "weak"),
                       ("OVERFIT", "overfit"), ("INSUFFICIENT_DATA", "insuf")]:
        print(f"{label:<22} {v[key]:>6}")

    if validated_without_fvg > 0 and v["best_non_fvg_combo"]:
        c = v["best_non_fvg_combo"]
        print(f"\nBest non-FVG validated combo:")
        print(f"  {c['pair']} {c['direction']} [{'+'.join(c['indicators'])}]")
        print(f"  Train HR={c['train_hr']:.0%} | Holdout HR={c['holdout_hr']:.0%} | n={c['n_holdout']}")
    else:
        print(f"\nBest non-FVG combo: NONE — zero non-FVG combos survived holdout")

    conclusion = (
        "CONFIRMS primary finding"
        if validated_without_fvg == 0
        else f"WEAKENS — {validated_without_fvg} non-FVG combos validated on holdout"
    )
    print(f"\nConclusion: {conclusion}  ({time.time()-t0:.1f}s)")

    return {
        "pool_size":               len(pool),
        "combinations_tested":     n_combos,
        "candidates_without_fvg":  total_candidates,
        "stored_for_validation":   total_stored,
        "validated_without_fvg":   validated_without_fvg,
        "best_non_fvg_holdout_hr": v["best_non_fvg_hr"] if validated_without_fvg > 0 else None,
        "best_non_fvg_combo":      v["best_non_fvg_combo"],
        "verdict_summary":         {"real": v["real"], "weak": v["weak"],
                                    "overfit": v["overfit"], "insuf": v["insuf"]},
        "conclusion":              conclusion,
    }


# ─── Test 2 — FVG Alone ───────────────────────────────────────────────────────

def test_2_fvg_alone(hist: pd.DataFrame) -> dict:
    print(f"\n{'═'*65}")
    print("TEST 2 — FVG AS STANDALONE SIGNAL")
    print(f"Train: {TRAIN_START}–{TRAIN_END} | Holdout: {HOLDOUT_START}–{HOLDOUT_END}")
    print(f"{'─'*65}")

    hist_train   = hist[(hist["date"] >= TRAIN_START) & (hist["date"] <= TRAIN_END)]
    hist_holdout = hist[(hist["date"] >= HOLDOUT_START) & (hist["date"] <= HOLDOUT_END)]

    # Compute vs_combination_hit_rate: avg holdout HR of existing REAL_SIGNAL green conditions
    green_data = json.loads(GREEN_PATH.read_text())
    real_signal_only = {
        pair: {
            "best_long":  [c for c in conds.get("best_long",  []) if c.get("holdout_verdict") == "REAL_SIGNAL"],
            "best_short": [c for c in conds.get("best_short", []) if c.get("holdout_verdict") == "REAL_SIGNAL"],
        }
        for pair, conds in green_data.items()
    }
    combo_v = _validate_green(hist_holdout, real_signal_only)
    vs_combination_hit_rate = combo_v["avg_real_hr"]

    print(f"\n{'PAIR':<10} {'DIR':<6} {'FVG_TRAIN':>9} {'FVG_HOLD':>9} {'N_TRAIN':>8} {'N_HOLD':>7}")
    print("-" * 55)

    per_pair: dict = {}
    all_train_hrs:   list[float] = []
    all_holdout_hrs: list[float] = []

    for pair in sorted(hist_train["pair"].unique()):
        g_tr = hist_train[hist_train["pair"]   == pair].dropna(subset=["fwd_10d"])
        g_ho = hist_holdout[hist_holdout["pair"] == pair].dropna(subset=["fwd_10d"])

        if "state_fvg" not in g_tr.columns:
            continue

        fwd_tr = g_tr["fwd_10d"].astype(float)
        fwd_ho = g_ho["fwd_10d"].astype(float)
        per_pair[pair] = {}

        for sign, label in [(1, "LONG"), (-1, "SHORT")]:
            m_tr = g_tr["state_fvg"] == sign
            m_ho = g_ho["state_fvg"] == sign
            n_tr = int(m_tr.sum())
            n_ho = int(m_ho.sum())

            if n_tr == 0:
                continue
            train_hr   = float((fwd_tr[m_tr] > 0).mean() if sign == 1 else (fwd_tr[m_tr] < 0).mean())
            holdout_hr = (
                float((fwd_ho[m_ho] > 0).mean() if sign == 1 else (fwd_ho[m_ho] < 0).mean())
                if n_ho > 0 else None
            )

            all_train_hrs.append(train_hr)
            if holdout_hr is not None:
                all_holdout_hrs.append(holdout_hr)

            h_str = f"{holdout_hr:.0%}" if holdout_hr is not None else "  n/a"
            print(f"{pair:<10} {label:<6} {train_hr:>8.0%} {h_str:>9} {n_tr:>8} {n_ho:>7}")
            per_pair[pair][label] = {
                "train_hr":   round(train_hr, 4),
                "holdout_hr": round(holdout_hr, 4) if holdout_hr is not None else None,
                "n_train":    n_tr,
                "n_holdout":  n_ho,
            }

    avg_train_hr   = round(sum(all_train_hrs)   / len(all_train_hrs),   4) if all_train_hrs   else 0.0
    avg_holdout_hr = round(sum(all_holdout_hrs)  / len(all_holdout_hrs), 4) if all_holdout_hrs else 0.0
    filtering_effect = round(vs_combination_hit_rate - avg_holdout_hr, 4)

    print(f"\n{'─'*55}")
    print(f"{'Average':<10} {'':>6} {avg_train_hr:>8.0%} {avg_holdout_hr:>9.0%}")
    print(f"\nREAL_SIGNAL green conditions avg holdout HR: {vs_combination_hit_rate:.0%}")
    print(f"Filtering effect (combo − FVG alone):       {filtering_effect:+.0%}")

    if filtering_effect > 0.08:
        conclusion = "FILTER"
        detail = "Indicator combos materially improve on FVG-alone (+8pp threshold)"
    elif avg_holdout_hr > 0.68:
        conclusion = "STANDALONE"
        detail = "FVG alone achieves >68% holdout hit rate; combo lift is marginal"
    else:
        conclusion = "AMBIGUOUS"
        detail = "Mixed — insufficient gap to classify as pure filter or pure standalone"

    print(f"\nConclusion: {conclusion} — {detail}")

    return {
        "avg_train_hit_rate":       avg_train_hr,
        "avg_holdout_hit_rate":     avg_holdout_hr,
        "vs_combination_hit_rate":  vs_combination_hit_rate,
        "filtering_effect_size":    filtering_effect,
        "per_pair":                 per_pair,
        "conclusion":               conclusion,
        "detail":                   detail,
    }


# ─── Test 3 — Rolling Holdout ─────────────────────────────────────────────────

def test_3_rolling(hist: pd.DataFrame) -> dict:
    print(f"\n{'═'*65}")
    print("TEST 3 — ROLLING HOLDOUT (4 WINDOWS, WALK-FORWARD)")
    print("Re-sweeps green conditions on each training window independently")
    print(f"{'─'*65}")

    windows = [
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
    ]

    print(f"\n{'YEAR':<6} {'TRAIN_END':<12} {'CANDS':>7} {'REAL':>6} {'FVG_IN_REAL':>12} {'FVG%':>7}  TIME")
    print("-" * 62)

    per_year: dict = {}
    all_fvg_rates: list[float] = []

    for year_label, h_start, h_end in windows:
        t0 = time.time()
        hist_train   = hist[hist["date"] < h_start]
        hist_holdout = hist[(hist["date"] >= h_start) & (hist["date"] <= h_end)]
        train_end_str = (
            pd.Timestamp(h_start) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        green, total_cands = _sweep_green(hist_train, INDICATOR_NAMES)
        v = _validate_green(hist_holdout, green)

        fvg_rate = v["fvg_rate"]
        all_fvg_rates.append(fvg_rate)
        elapsed = time.time() - t0

        print(
            f"{year_label:<6} {train_end_str:<12} {total_cands:>7} {v['real']:>6} "
            f"{v['fvg_in_real']:>12} {fvg_rate:>6.0%}  {elapsed:.0f}s"
        )

        per_year[year_label] = {
            "train_end":        train_end_str,
            "green_candidates": total_cands,
            "real_signal":      v["real"],
            "fvg_in_real":      v["fvg_in_real"],
            "fvg_rate":         fvg_rate,
            "weak":             v["weak"],
            "overfit":          v["overfit"],
            "insuf":            v["insuf"],
        }

    all_above_90  = all(r >= 0.90 for r in all_fvg_rates)
    min_fvg_rate  = min(all_fvg_rates) if all_fvg_rates else 0.0
    worst_year    = min(per_year, key=lambda y: per_year[y]["fvg_rate"])

    conclusion = (
        "STRUCTURAL"
        if all_above_90
        else f"PERIOD_SPECIFIC — {worst_year} FVG rate {min_fvg_rate:.0%} < 90%"
    )
    print(f"\nAll years ≥90% FVG in REAL_SIGNAL: {all_above_90}")
    print(f"Conclusion: {conclusion}")

    flat: dict = {}
    for yr in ("2021", "2022", "2023", "2024"):
        flat[f"{yr}_green_candidates"] = per_year[yr]["green_candidates"]
        flat[f"{yr}_real_signal"]      = per_year[yr]["real_signal"]
        flat[f"{yr}_fvg_rate"]         = per_year[yr]["fvg_rate"]
    flat.update({
        "all_above_90pct": all_above_90,
        "min_fvg_rate":    min_fvg_rate,
        "conclusion":      conclusion,
        "per_year":        per_year,
    })
    return flat


# ─── Test 4 — Restricted Pool ─────────────────────────────────────────────────

def test_4_restricted(hist: pd.DataFrame) -> dict:
    # 6 market structure + acc_dist (ADL) as the 7th per user spec
    RESTRICTED = [
        "fvg", "bos", "session_hl", "displacement",
        "liq_sweep", "order_block", "acc_dist",
    ]
    n_combos = len(list(itertools.combinations(RESTRICTED, 3)))

    print(f"\n{'═'*65}")
    print(f"TEST 4 — RESTRICTED POOL ({len(RESTRICTED)} indicators, C({len(RESTRICTED)},3) = {n_combos} combos)")
    print(f"Pool: {', '.join(RESTRICTED)}")
    print(f"Train: {TRAIN_START}–{TRAIN_END} | Holdout: {HOLDOUT_START}–{HOLDOUT_END}")
    print(f"{'─'*65}")

    t0 = time.time()
    hist_train   = hist[(hist["date"] >= TRAIN_START) & (hist["date"] <= TRAIN_END)]
    hist_holdout = hist[(hist["date"] >= HOLDOUT_START) & (hist["date"] <= HOLDOUT_END)]

    green, total_cands = _sweep_green(hist_train, RESTRICTED)
    total_stored = sum(len(v["best_long"]) + len(v["best_short"]) for v in green.values())

    print(f"\nGreen candidates passing thresholds: {total_cands}")
    print(f"Stored for validation (top 10/pair/dir): {total_stored}")

    v = _validate_green(hist_holdout, green)

    if v["validated_combos"]:
        print(f"\n{'PAIR':<10} {'DIR':<6} {'INDICATORS':<38} {'HOLD_HR':>7} {'FVG':>5} {'N':>4}")
        print("-" * 75)
        for combo in sorted(v["validated_combos"], key=lambda x: x["holdout_hr"], reverse=True):
            fvg_mark = "YES" if combo["has_fvg"] else "NO"
            ind_str  = "+".join(combo["indicators"])
            print(
                f"{combo['pair']:<10} {combo['direction']:<6} {ind_str:<38} "
                f"{combo['holdout_hr']:>6.0%} {fvg_mark:>5} {combo['n']:>4}"
            )
    else:
        print("\nNo conditions validated on 2024 holdout (all INSUFFICIENT_DATA or worse).")

    print(f"\n{'VERDICT':<22} {'COUNT':>6}")
    print("-" * 30)
    for label, key in [("REAL_SIGNAL", "real"), ("WEAK_SIGNAL", "weak"),
                       ("OVERFIT", "overfit"), ("INSUFFICIENT_DATA", "insuf")]:
        print(f"{label:<22} {v[key]:>6}")
    if v["real"] > 0:
        print(f"{'FVG in REAL_SIGNAL':<22} {v['fvg_in_real']:>6}/{v['real']} = {v['fvg_rate']:.0%}")

    if total_cands == 0:
        conclusion = (
            "INCONCLUSIVE — restricted-pool indicators too sparse in combination "
            f"(none of C({len(RESTRICTED)},3)={n_combos} combos reached n≥{MIN_SAMPLES} co-occurrences); "
            "cannot assess FVG dominance in this search space"
        )
    elif v["fvg_rate"] >= 0.90:
        conclusion = "CONFIRMS multiple testing defense"
    else:
        conclusion = f"WEAKENS — FVG rate {v['fvg_rate']:.0%} < 90% threshold"
    print(f"\nConclusion: {conclusion}  ({time.time()-t0:.1f}s)")

    return {
        "pool":                 RESTRICTED,
        "combinations_tested":  n_combos,
        "candidates":           total_cands,
        "validated_count":      v["real"],
        "fvg_presence_rate":    v["fvg_rate"],
        "validated_combos":     v["validated_combos"],
        "conclusion":           conclusion,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*65}")
    print("FVG PEER REVIEW VALIDATION SUITE")
    print(f"{'═'*65}")

    if not HIST_PATH.exists():
        sys.exit(f"ERROR: {HIST_PATH} not found — run build_indicator_ontology.py first")

    print("Loading history.parquet...")
    hist = pd.read_parquet(HIST_PATH)
    hist["date"] = pd.to_datetime(hist["date"])
    print(
        f"  {len(hist):,} rows | {hist['pair'].nunique()} pairs | "
        f"{hist['date'].min().date()} to {hist['date'].max().date()}"
    )

    t_total = time.time()
    r1 = test_1_ablation(hist)
    r2 = test_2_fvg_alone(hist)
    r3 = test_3_rolling(hist)
    r4 = test_4_restricted(hist)

    output = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "test_1_ablation":    r1,
        "test_2_fvg_alone":   r2,
        "test_3_rolling":     r3,
        "test_4_restricted":  r4,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n{'═'*65}")
    print(f"Results written to {OUT_PATH.relative_to(ROOT)}")
    print(f"Total elapsed: {time.time()-t_total:.0f}s")
    print(f"{'═'*65}")
    print("\nSUMMARY")
    print(f"  T1 — FVG Ablation:    {r1['conclusion']}")
    print(f"  T2 — FVG Alone:       {r2['conclusion']}")
    print(f"  T3 — Rolling Holdout: {r3['conclusion']}")
    print(f"  T4 — Restricted Pool: {r4['conclusion']}")


if __name__ == "__main__":
    main()
