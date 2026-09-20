#!/usr/bin/env python3
"""
sovereign/financing/swap_probe.py
=================================
Measure OANDA's real per-pair financing, and decide whether the broker's take
leaves anything of the carry premium.

Why this is the deciding measurement
------------------------------------
`research/MAGNUM_OPUS.md` §VI.5 gate (b): before any dollar moves, measure the
broker's real per-pair swap, because at this account size the broker's take is
the whole P&L. `research/strategies/FX_CARRY_V1_ADV.md` plans on ~+5%/yr. If
financing on the side the strategy holds costs more than that, there is no trade
at this size, whatever the Sharpe says.

The repo's own model is already known to be wrong. `SWAP_RATES_ANNUAL` in
`sovereign/forex/forex_backtester.py` was found ~10× too small with a
EURUSD-SHORT sign flip (TICK-024, gated). This module does not fix that table —
fixing it re-baselines every backtest and needs its own logged param change. It
measures the truth and reports the gap.

Two measurements, deliberately separate
---------------------------------------
QUOTED     — what OANDA advertises right now, per instrument, per side, from
             /v3/accounts/{id}/instruments. Costs nothing, risks nothing, needs
             no position, available today. Rates move, so this is sampled daily
             and the report shows the distribution, not one reading.

REALIZED   — what was actually charged, from DAILY_FINANCING transactions on the
             account. This is the ground truth, and it requires a position to be
             open. This module will NOT open one; it reads what is there and
             tells you exactly what to open if you want realized numbers.

Read-only
---------
Every request is a GET. There is no order path in this file — a test asserts it.

Usage
-----
    python3 -m sovereign.financing.swap_probe snapshot     # append today's reading
    python3 -m sovereign.financing.swap_probe report       # verdict from history
    python3 -m sovereign.financing.swap_probe report --json
    python3 -m sovereign.financing.swap_probe realized --days 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO / "data" / "financing" / "swap_snapshots.jsonl"
ENV_PATH = REPO / ".env"

# The four pairs the live v015 book trades. AUDNZD is excluded by HYP-045.
DEFAULT_PAIRS = ("EUR_USD", "GBP_USD", "AUD_USD", "GBP_JPY")

# The planning premium this measurement is judged against — FX_CARRY_V1_ADV.md:
# "~+5%/yr, Sharpe ~0.6, a −25% year about once a decade."
PLANNING_PREMIUM_PCT = 5.0

# The repo's modelled table, quoted here for comparison only. Never imported
# from the backtester: that module is on the execution path and this one must
# stay unable to touch it.
MODELLED_ANNUAL = {
    "GBP_USD": {"LONG": -0.0012, "SHORT": -0.0008},
    "EUR_USD": {"LONG": -0.0015, "SHORT": -0.0010},
    "USD_JPY": {"LONG": 0.0020, "SHORT": -0.0035},
    "AUD_USD": {"LONG": -0.0008, "SHORT": -0.0012},
    "AUD_NZD": {"LONG": -0.0003, "SHORT": -0.0003},
}
MODELLED_SOURCE = "sovereign/forex/forex_backtester.py::SWAP_RATES_ANNUAL"

# Enough independent daily readings for a mean to mean anything. Two weeks of
# business days is the gate MAGNUM_OPUS names.
MIN_DAYS_FOR_VERDICT = 10


# ─────────────────────────────────────────────────────────────────────────────
# Credentials + transport (GET only)
# ─────────────────────────────────────────────────────────────────────────────

def _env() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#") and "=" in t:
                k, v = t.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


class BrokerUnavailable(RuntimeError):
    """The broker could not be read. Never raised to mean 'the rate is zero'."""


def _get(path: str, params: Optional[dict[str, str]] = None) -> Any:
    env = _env()
    key, acct = env.get("OANDA_API_KEY"), env.get("OANDA_ACCOUNT_ID")
    if not key or not acct:
        raise BrokerUnavailable("OANDA_API_KEY / OANDA_ACCOUNT_ID missing from .env")
    base = env.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com").rstrip("/")
    url = f"{base}/v3/accounts/{acct}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise BrokerUnavailable(f"OANDA HTTP {exc.code} {exc.reason} for {path}") from exc
    except Exception as exc:
        raise BrokerUnavailable(f"{type(exc).__name__}: {exc}") from exc


def account_mode() -> str:
    return "LIVE" if _env().get("OANDA_LIVE") == "1" else "practice"


# ─────────────────────────────────────────────────────────────────────────────
# QUOTED rates
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairFinancing:
    """One pair's advertised financing, as annual fractions (+earn / −pay)."""
    pair: str
    long_rate: float
    short_rate: float
    weekly_days_charged: int          # normally 7: Wed carries 3
    triple_day: Optional[str]

    @property
    def quoted(self) -> bool:
        """False when the broker returned 0.0000 on BOTH sides.

        That is not a zero-carry pair; it is an absent quote. Treating it as
        0%/yr would pull the book's measured cost toward zero — the same
        silent-success failure the repo's laws name ("green can be empty").
        """
        return not (self.long_rate == 0.0 and self.short_rate == 0.0)

    @property
    def carry_side(self) -> str:
        """The side financing pays you for — or costs least."""
        return "LONG" if self.long_rate >= self.short_rate else "SHORT"

    @property
    def carry_rate(self) -> float:
        return max(self.long_rate, self.short_rate)

    @property
    def broker_take(self) -> float:
        """long + short. Symmetric financing would sum to ~0; the shortfall is
        the broker's spread, paid whichever side you are on."""
        return self.long_rate + self.short_rate

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(carry_side=self.carry_side, carry_rate=self.carry_rate,
                 broker_take=self.broker_take, quoted=self.quoted)
        return d


def fetch_quoted(pairs: Iterable[str] = DEFAULT_PAIRS) -> list[PairFinancing]:
    """Today's advertised financing for each pair. Raises BrokerUnavailable."""
    pairs = list(pairs)
    data = _get("/instruments", {"instruments": ",".join(pairs)})
    found: dict[str, PairFinancing] = {}
    for ins in data.get("instruments", []):
        fin = ins.get("financing") or {}
        if "longRate" not in fin or "shortRate" not in fin:
            continue
        days = fin.get("financingDaysOfWeek", []) or []
        triple = next((d.get("dayOfWeek") for d in days
                       if int(d.get("daysCharged", 0)) >= 3), None)
        found[ins["name"]] = PairFinancing(
            pair=ins["name"],
            long_rate=float(fin["longRate"]),
            short_rate=float(fin["shortRate"]),
            weekly_days_charged=sum(int(d.get("daysCharged", 0)) for d in days),
            triple_day=triple,
        )
    missing = [p for p in pairs if p not in found]
    if missing:
        raise BrokerUnavailable(f"broker returned no financing for: {', '.join(missing)}")
    return [found[p] for p in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshots
# ─────────────────────────────────────────────────────────────────────────────

def snapshot(pairs: Iterable[str] = DEFAULT_PAIRS,
             path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Append one dated reading. Idempotent per UTC day: re-running replaces
    today's row rather than inflating n with the same reading twice."""
    rows = fetch_quoted(pairs)
    today = datetime.now(timezone.utc).date().isoformat()
    record = {
        "date": today,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": account_mode(),
        "pairs": {r.pair: r.to_dict() for r in rows},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [ln for ln in _read_lines(path)
            if _safe_date(ln) != today]
    kept.append(json.dumps(record))
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return record


def _read_lines(path: Path) -> list[str]:
    try:
        return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except FileNotFoundError:
        return []


def _safe_date(line: str) -> Optional[str]:
    try:
        return json.loads(line).get("date")
    except json.JSONDecodeError:
        return None


def load_snapshots(path: Path = SNAPSHOT_PATH) -> list[dict[str, Any]]:
    out = []
    for ln in _read_lines(path):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return sorted(out, key=lambda r: r.get("date", ""))


# ─────────────────────────────────────────────────────────────────────────────
# REALIZED financing (needs an open position; this module never opens one)
# ─────────────────────────────────────────────────────────────────────────────

def realized(days: int = 30) -> dict[str, Any]:
    """DAILY_FINANCING actually charged to the account, attributed per pair.

    Returns `{"available": False, "reason": ...}` when nothing has been charged,
    which is the normal state for a flat account — not an error.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        page = _get("/transactions", {"from": since, "type": "DAILY_FINANCING",
                                      "pageSize": "500"})
    except BrokerUnavailable as exc:
        return {"available": False, "reason": str(exc)}

    txns: list[dict[str, Any]] = []
    for url in page.get("pages", []) or []:
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {_env()['OANDA_API_KEY']}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                txns.extend(json.load(r).get("transactions", []))
        except Exception:
            continue
    txns.extend(page.get("transactions", []))

    per_pair: dict[str, float] = {}
    total = 0.0
    for t in txns:
        if t.get("type") != "DAILY_FINANCING":
            continue
        total += float(t.get("financing", 0) or 0)
        for pf in t.get("positionFinancings", []) or []:
            ins = pf.get("instrument")
            if ins:
                per_pair[ins] = per_pair.get(ins, 0.0) + float(pf.get("financing", 0) or 0)

    if not txns:
        return {"available": False,
                "reason": (f"no DAILY_FINANCING charged in {days}d — the account has held "
                           f"no position overnight. Realized measurement needs one unit "
                           f"per pair held across a rollover; this module will not open it."),
                "window_days": days}
    return {"available": True, "window_days": days, "n_transactions": len(txns),
            "total_financing": round(total, 6),
            "per_pair": {k: round(v, 6) for k, v in sorted(per_pair.items())}}


# ─────────────────────────────────────────────────────────────────────────────
# The verdict
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairVerdict:
    pair: str
    n_days: int
    carry_side: str
    carry_rate_mean_pct: float        # annual %, on the side the strategy holds
    carry_rate_min_pct: float
    carry_rate_max_pct: float
    broker_take_mean_pct: float       # annual %, long+short shortfall
    modelled_pct: Optional[float]
    model_ratio: Optional[float]      # |measured| / |modelled|
    sign_agrees: Optional[bool]
    verdict: str                      # PAYS | COSTS | MARGINAL
    note: str = ""


@dataclass
class Report:
    generated_at: str
    mode: str
    n_days: int
    first_day: Optional[str]
    last_day: Optional[str]
    enough_data: bool
    planning_premium_pct: float
    pairs: list[PairVerdict]
    portfolio_carry_pct: Optional[float]
    portfolio_take_pct: Optional[float]
    headline: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def build_report(path: Path = SNAPSHOT_PATH,
                 premium_pct: float = PLANNING_PREMIUM_PCT) -> Report:
    snaps = load_snapshots(path)
    warnings: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not snaps:
        return Report(now, account_mode(), 0, None, None, False, premium_pct, [],
                      None, None,
                      "NO DATA — run `snapshot` daily; the verdict needs "
                      f"{MIN_DAYS_FOR_VERDICT} readings.",
                      ["no snapshots on disk"])

    pair_names = sorted({p for s in snaps for p in s.get("pairs", {})})
    verdicts: list[PairVerdict] = []
    for pair in pair_names:
        carry, take, sides, unquoted = [], [], [], 0
        for s in snaps:
            row = s.get("pairs", {}).get(pair)
            if not row:
                continue
            # Older snapshots predate the `quoted` field; derive it the same way.
            is_quoted = row.get("quoted")
            if is_quoted is None:
                is_quoted = not (float(row["long_rate"]) == 0.0
                                 and float(row["short_rate"]) == 0.0)
            if not is_quoted:
                unquoted += 1
                continue
            carry.append(float(row["carry_rate"]) * 100.0)
            take.append(float(row["broker_take"]) * 100.0)
            sides.append(row["carry_side"])
        if not carry:
            verdicts.append(PairVerdict(
                pair=pair, n_days=0, carry_side="—",
                carry_rate_mean_pct=float("nan"), carry_rate_min_pct=float("nan"),
                carry_rate_max_pct=float("nan"), broker_take_mean_pct=float("nan"),
                modelled_pct=None, model_ratio=None, sign_agrees=None,
                verdict="NOT_QUOTED",
                note=f"broker returned 0.0000 on both sides for all {unquoted} reading(s) "
                     f"— an absent quote, not zero carry. Excluded from the book average."))
            warnings.append(f"{pair}: NOT QUOTED by the broker — excluded from the aggregate")
            continue
        side = max(set(sides), key=sides.count)
        if len(set(sides)) > 1:
            warnings.append(f"{pair}: the paying side flipped during the window "
                            f"({'/'.join(sorted(set(sides)))}) — carry is not stable here")
        mean_carry = statistics.fmean(carry)
        modelled = MODELLED_ANNUAL.get(pair, {}).get(side)
        modelled_pct = None if modelled is None else modelled * 100.0
        ratio = (abs(mean_carry) / abs(modelled_pct)
                 if modelled_pct not in (None, 0.0) else None)
        sign_ok = (None if modelled_pct is None
                   else (mean_carry >= 0) == (modelled_pct >= 0))

        if mean_carry > 0.25:
            v = "PAYS"
        elif mean_carry < -0.25:
            v = "COSTS"
        else:
            v = "MARGINAL"
        note = ""
        if sign_ok is False:
            note = (f"SIGN FLIP vs the model: measured {mean_carry:+.3f}%/yr, "
                    f"model {modelled_pct:+.3f}%/yr")
        elif ratio is not None and (ratio > 3 or ratio < 1 / 3):
            note = f"model off by {ratio:.1f}× on this side"

        verdicts.append(PairVerdict(
            pair=pair, n_days=len(carry), carry_side=side,
            carry_rate_mean_pct=round(mean_carry, 4),
            carry_rate_min_pct=round(min(carry), 4),
            carry_rate_max_pct=round(max(carry), 4),
            broker_take_mean_pct=round(statistics.fmean(take), 4),
            modelled_pct=None if modelled_pct is None else round(modelled_pct, 4),
            model_ratio=None if ratio is None else round(ratio, 2),
            sign_agrees=sign_ok, verdict=v, note=note))

    n_days = len({s.get("date") for s in snaps})
    enough = n_days >= MIN_DAYS_FOR_VERDICT
    if not enough:
        warnings.append(f"only {n_days} of {MIN_DAYS_FOR_VERDICT} readings — "
                        f"the headline below is provisional")

    # Equal-weight across the book: what the strategy earns or pays in financing
    # alone, before any price move, on the side it actually holds.
    priced = [v for v in verdicts if v.verdict != "NOT_QUOTED"]
    port_carry = (round(statistics.fmean([v.carry_rate_mean_pct for v in priced]), 4)
                  if priced else None)
    port_take = (round(statistics.fmean([v.broker_take_mean_pct for v in priced]), 4)
                 if priced else None)
    if priced and len(priced) < len(verdicts):
        warnings.append(f"book average is over {len(priced)} of {len(verdicts)} pairs "
                        f"— the rest were not quoted")

    headline = _headline(port_carry, premium_pct, enough)
    if any(v.sign_agrees is False for v in verdicts):
        warnings.append("at least one pair's financing sign is opposite the modelled table — "
                        f"every backtest using {MODELLED_SOURCE} is mis-costed on that leg")

    return Report(now, account_mode(), n_days, snaps[0].get("date"),
                  snaps[-1].get("date"), enough, premium_pct, verdicts,
                  port_carry, port_take, headline, warnings)


def _headline(port_carry: Optional[float], premium_pct: float, enough: bool) -> str:
    if port_carry is None:
        return "NO DATA"
    prefix = "" if enough else "PROVISIONAL — "
    if port_carry >= 0:
        return (f"{prefix}FINANCING PAYS {port_carry:+.2f}%/yr on the carry side, before any "
                f"price move. The {premium_pct:.0f}%/yr premium survives the broker.")
    eaten = abs(port_carry) / premium_pct * 100.0
    if abs(port_carry) >= premium_pct:
        return (f"{prefix}NO TRADE AT THIS SIZE — financing costs {port_carry:+.2f}%/yr, "
                f"which is {eaten:.0f}% of the {premium_pct:.0f}%/yr premium. "
                f"The broker takes more than the edge pays.")
    return (f"{prefix}financing costs {port_carry:+.2f}%/yr — {eaten:.0f}% of the "
            f"{premium_pct:.0f}%/yr premium. Net ≈ {premium_pct + port_carry:+.2f}%/yr "
            f"before spread and slippage.")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render(rep: Report) -> str:
    L: list[str] = []
    a = L.append
    a(f"SWAP MEASUREMENT — {rep.mode} account")
    a(f"{rep.n_days} daily reading(s)"
      + (f", {rep.first_day} → {rep.last_day}" if rep.first_day else "")
      + f"   (verdict needs {MIN_DAYS_FOR_VERDICT})")
    a("")
    a(rep.headline)
    a("")
    if rep.pairs:
        a(f"{'pair':<10}{'side':<7}{'financing %/yr':>16}{'range':>20}"
          f"{'broker take':>14}{'model':>10}{'off by':>9}")
        a("-" * 86)
        for v in rep.pairs:
            if v.verdict == "NOT_QUOTED":
                a(f"{v.pair:<10}{'—':<7}{'NOT QUOTED':>16}{'—':>20}{'—':>14}{'—':>10}{'—':>9}")
                continue
            rng = f"{v.carry_rate_min_pct:+.2f} … {v.carry_rate_max_pct:+.2f}"
            model = "—" if v.modelled_pct is None else f"{v.modelled_pct:+.3f}"
            off = "—" if v.model_ratio is None else f"{v.model_ratio:.1f}x"
            a(f"{v.pair:<10}{v.carry_side:<7}{v.carry_rate_mean_pct:>+16.3f}{rng:>20}"
              f"{v.broker_take_mean_pct:>+14.3f}{model:>10}{off:>9}")
        a("")
        for v in rep.pairs:
            if v.note:
                a(f"  ! {v.pair}: {v.note}")
    if rep.portfolio_take_pct is not None:
        a("")
        a(f"Broker's take, equal-weight across the book: {rep.portfolio_take_pct:+.3f}%/yr "
          f"(long+short would sum to 0 if financing were symmetric).")
    if rep.warnings:
        a("")
        a("WARNINGS")
        for w in rep.warnings:
            a(f"  · {w}")
    a("")
    a("Quoted rates only — what OANDA advertises. For what was actually charged, hold one "
      "unit per pair across a rollover and run `realized`. This tool will not open it.")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="swap_probe",
        description="Measure OANDA's real per-pair financing. Read-only.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sn = sub.add_parser("snapshot", help="append today's quoted rates (idempotent per day)")
    sn.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS))

    rp = sub.add_parser("report", help="verdict from the collected history")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--premium", type=float, default=PLANNING_PREMIUM_PCT,
                    help="premium to judge against (default 5.0%%/yr, FX_CARRY_V1_ADV)")

    rz = sub.add_parser("realized", help="financing actually charged to the account")
    rz.add_argument("--days", type=int, default=30)
    rz.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        try:
            rec = snapshot(args.pairs)
        except BrokerUnavailable as exc:
            print(f"BROKER UNAVAILABLE — nothing recorded: {exc}", file=sys.stderr)
            return 2
        n = len(load_snapshots())
        print(f"recorded {rec['date']} ({rec['mode']}) — {len(rec['pairs'])} pairs, "
              f"{n} reading(s) on file")
        for p, r in rec["pairs"].items():
            if not r.get("quoted", True):
                print(f"  {p:<10} NOT QUOTED (broker returned 0.0000 both sides)")
                continue
            print(f"  {p:<10} long {r['long_rate']:+.4f}  short {r['short_rate']:+.4f}  "
                  f"→ carry {r['carry_side']} {r['carry_rate']*100:+.3f}%/yr  "
                  f"| broker take {r['broker_take']*100:+.2f}%/yr")
        return 0

    if args.cmd == "report":
        rep = build_report(premium_pct=args.premium)
        if args.json:
            json.dump(rep.to_dict(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(render(rep))
        return 0

    res = realized(args.days)
    if args.json:
        json.dump(res, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif res["available"]:
        print(f"DAILY_FINANCING over {res['window_days']}d — "
              f"{res['n_transactions']} charges, total {res['total_financing']:+.6f}")
        for p, v in res["per_pair"].items():
            print(f"  {p:<10} {v:+.6f}")
    else:
        print(f"no realized data: {res['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
