"""Spec for scripts/weekly_report.py — the weekly v015 report built from committed live data.

Contract (written before the implementation):
  * NAV older than --stale-days, or no NAV at all  -> exit 1 (CI turns that into an issue).
  * Trades are FOREX-only and windowed by exit_timestamp, not entry.
  * Snapshot gaps longer than the gap threshold are listed.
  * The drawdown cap comes from RISK_CONSTITUTION.md Article 3; no silent default.
  * A window with zero closed trades is a valid report, not an error.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import weekly_report as wr  # noqa: E402

AS_OF = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)

CONSTITUTION = """# Risk Constitution

## Article 2 — Something else
Limit of **9%** here must be ignored.

## Article 3 — Drawdown Circuit Breakers

Drawdown is measured peak-to-trough at account level. At a drawdown of **3.5%**,
all new position sizes are halved. At **5%**, new entries halt.

## Article 4 — Next
"""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _repo(tmp_path: Path, nav_rows: list[dict], decisions: list[dict] | None = None,
          constitution: str = CONSTITUTION) -> Path:
    _write_jsonl(tmp_path / "data/agent/equity_curve_live.jsonl", nav_rows)
    _write_jsonl(tmp_path / "data/decision_logs/decisions_2026_09.jsonl", decisions or [])
    (tmp_path / "RISK_CONSTITUTION.md").write_text(constitution)
    return tmp_path


def _nav(t: str, nav: float) -> dict:
    return {"t": t, "nav": nav, "balance": nav, "unrealized_pl": 0.0, "open_trade_count": 0}


FRESH_NAV = [
    _nav("2026-09-18T13:00:00+00:00", 100000.0),
    _nav("2026-09-21T13:00:00+00:00", 101000.0),  # window start anchor is at/before 09-21 14:00
    _nav("2026-09-24T13:00:00+00:00", 99990.0),
    _nav("2026-09-28T13:00:00+00:00", 102000.0),
]


def _dec(system="FOREX", outcome="WIN", exit_ts="2026-09-25T10:00:00.123456789Z",
         entry_ts="2026-09-10T10:00:00+00:00", pair="GBPUSD=X", r=None, direction="LONG"):
    return {"system": system, "pair": pair, "direction": direction, "entry_timestamp": entry_ts,
            "exit_timestamp": exit_ts, "outcome": outcome, "r_realized": r}


# ── staleness ──────────────────────────────────────────────────────────────

def test_stale_nav_exits_1(tmp_path):
    repo = _repo(tmp_path, [_nav("2026-08-24T13:00:00+00:00", 100000.0)])
    code = wr.main(["--repo", str(repo), "--as-of", AS_OF.isoformat(),
                    "--out", str(tmp_path / "r.json")])
    assert code == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["stale"] is True
    assert report["nav"]["age_days"] > 30


def test_missing_nav_exits_1(tmp_path):
    repo = _repo(tmp_path, [])
    assert wr.main(["--repo", str(repo), "--as-of", AS_OF.isoformat(),
                    "--out", str(tmp_path / "r.json")]) == 1


def test_fresh_nav_exits_0(tmp_path):
    repo = _repo(tmp_path, FRESH_NAV)
    assert wr.main(["--repo", str(repo), "--as-of", AS_OF.isoformat(),
                    "--out", str(tmp_path / "r.json")]) == 0


def test_points_after_as_of_are_ignored(tmp_path):
    rows = FRESH_NAV + [_nav("2026-10-05T13:00:00+00:00", 1.0)]
    report = wr.build_report(_repo(tmp_path, rows), AS_OF, days=7, stale_days=7)
    assert report["nav"]["latest_nav"] == 102000.0


# ── NAV section ────────────────────────────────────────────────────────────

def test_week_change_uses_last_point_at_or_before_window_start(tmp_path):
    report = wr.build_report(_repo(tmp_path, FRESH_NAV), AS_OF, days=7, stale_days=7)
    nav = report["nav"]
    assert nav["window_start_nav"] == 101000.0
    assert nav["week_change_pct"] == pytest.approx((102000 / 101000 - 1) * 100, abs=1e-3)


def test_current_drawdown_from_peak(tmp_path):
    rows = FRESH_NAV[:-1]  # ends at 99990 after a 101000 peak
    as_of = datetime(2026, 9, 25, tzinfo=timezone.utc)
    report = wr.build_report(_repo(tmp_path, rows), as_of, days=7, stale_days=7)
    assert report["nav"]["current_drawdown_pct"] == pytest.approx((99990 / 101000 - 1) * 100, abs=1e-3)
    assert report["dd_breach"] is False  # -1.0% is inside the 3.5% cap


def test_dd_breach_when_current_drawdown_exceeds_cap(tmp_path):
    rows = [_nav("2026-09-20T13:00:00+00:00", 100000.0), _nav("2026-09-28T13:00:00+00:00", 96000.0)]
    report = wr.build_report(_repo(tmp_path, rows), AS_OF, days=7, stale_days=7)
    assert report["dd_cap_pct"] == 3.5
    assert report["dd_breach"] is True


def test_gap_detection(tmp_path):
    rows = [_nav("2026-08-24T13:00:00+00:00", 100000.0), _nav("2026-09-28T13:00:00+00:00", 100500.0)]
    report = wr.build_report(_repo(tmp_path, rows), AS_OF, days=7, stale_days=7)
    gaps = report["nav"]["gaps"]
    assert len(gaps) == 1
    assert gaps[0]["from"].startswith("2026-08-24")
    assert gaps[0]["days"] == pytest.approx(35.0, abs=0.01)


def test_only_gaps_over_threshold_listed(tmp_path):
    # FRESH_NAV steps: 3d, 3d, 4d -> only the 4-day step (09-24 -> 09-28) exceeds 3 days
    report = wr.build_report(_repo(tmp_path, FRESH_NAV), AS_OF, days=7, stale_days=7)
    assert [round(g["days"]) for g in report["nav"]["gaps"]] == [4]


# ── trades section ─────────────────────────────────────────────────────────

def test_only_forex_rows_counted(tmp_path):
    decisions = [_dec(), _dec(system="ICT", outcome="LOSS")]
    report = wr.build_report(_repo(tmp_path, FRESH_NAV, decisions), AS_OF, days=7, stale_days=7)
    assert report["trades"]["closed"] == 1
    assert report["trades"]["wins"] == 1


def test_window_filter_uses_exit_timestamp(tmp_path):
    decisions = [
        # entered long before the window, closed inside it -> counted
        _dec(entry_ts="2026-08-01T00:00:00+00:00", exit_ts="2026-09-26T00:00:00Z", outcome="LOSS", r=-1.0),
        # entered inside the window, closed before it (impossible but guards the key) -> not counted
        _dec(entry_ts="2026-09-27T00:00:00+00:00", exit_ts="2026-09-10T00:00:00Z", pair="EURUSD=X"),
    ]
    report = wr.build_report(_repo(tmp_path, FRESH_NAV, decisions), AS_OF, days=7, stale_days=7)
    t = report["trades"]
    assert t["closed"] == 1 and t["losses"] == 1 and t["wins"] == 0
    assert t["win_rate_pct"] == 0.0
    assert t["avg_r"] == -1.0 and t["n_r"] == 1


def test_open_and_expired_counts(tmp_path):
    decisions = [_dec(outcome=None, exit_ts=None, pair="AUDUSD=X"),
                 _dec(outcome="EXPIRED", exit_ts="2026-09-27T00:00:00Z", pair="EURUSD=X")]
    report = wr.build_report(_repo(tmp_path, FRESH_NAV, decisions), AS_OF, days=7, stale_days=7)
    assert report["trades"]["open"] == 1
    assert report["trades"]["expired_in_window"] == 1
    assert report["trades"]["closed"] == 0


def test_duplicate_trade_rows_counted_once(tmp_path):
    # backfill can write the same trade twice with different pair spellings
    decisions = [_dec(pair="GBPUSD=X"), _dec(pair="GBP_USD")]
    report = wr.build_report(_repo(tmp_path, FRESH_NAV, decisions), AS_OF, days=7, stale_days=7)
    assert report["trades"]["closed"] == 1


def test_empty_window_is_valid(tmp_path):
    repo = _repo(tmp_path, FRESH_NAV, [])
    assert wr.main(["--repo", str(repo), "--as-of", AS_OF.isoformat(),
                    "--out", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["trades"]["closed"] == 0
    assert report["trades"]["win_rate_pct"] is None


# ── drawdown cap parsing ───────────────────────────────────────────────────

def test_parse_dd_cap_reads_article_3_only():
    assert wr.parse_dd_cap(CONSTITUTION) == 3.5


def test_parse_dd_cap_missing_raises():
    with pytest.raises(ValueError):
        wr.parse_dd_cap("# Risk Constitution\n\n## Article 2\n**9%**\n")


# ── outputs ────────────────────────────────────────────────────────────────

def test_github_outputs_written(tmp_path, monkeypatch):
    out_file = tmp_path / "gh_out"
    summary = tmp_path / "gh_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    repo = _repo(tmp_path, FRESH_NAV, [_dec()])
    assert wr.main(["--repo", str(repo), "--as-of", AS_OF.isoformat(),
                    "--out", str(tmp_path / "r.json")]) == 0
    kv = dict(line.split("=", 1) for line in out_file.read_text().splitlines())
    assert kv["stale"] == "false"
    assert kv["dd_breach"] == "false"
    assert set(kv) >= {"nav_age_days", "week_nav_change_pct", "max_drawdown_pct", "current_drawdown_pct"}
    assert "Weekly Performance Report" in summary.read_text()
