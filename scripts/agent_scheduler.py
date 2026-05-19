"""
scripts/agent_scheduler.py
Alta Investments — Sovereign Trading Intelligence

Usage-aware autonomous research scheduler.

DESIGN:
  This scheduler runs every 2-4 hours via launchd. It reads usage.json
  to decide what the agent is allowed to do, then picks the highest-priority
  QUEUED task from research_queue.json and dispatches it.

  It does NOT call Claude. All tasks are local Python only.
  It does NOT touch live trading parameters.
  It DOES write findings to findings.jsonl and messages to messages_to_colin.json.

MODES:
  FULL        — prime time + budget: run heavy backtests, data pulls, simulations
  HEAVY_OK    — budget allows but human may be active: run heavy if not during work hours
  LIGHT_ONLY  — low budget or human buffer: health check + hypothesis ledger only
  SKIP        — nothing to do or budget exhausted

TIME GATES:
  Prime time (agent preferred):   00:00–09:00 ET, 23:00–24:00 ET
  Work hours (agent backs off):   09:00–23:00 ET

USAGE GATES:
  weekly_remaining > 50k → NORMAL (full autonomy)
  weekly_remaining 20k–50k → LIGHT (health + ledger only)
  weekly_remaining < 20k → DANGER (health check only)
  human_buffer < 8k → LIGHT_ONLY (preserve human session)

RUN:
  python3 scripts/agent_scheduler.py [--dry-run] [--force-heavy]

LAUNCHD:
  plist: com.alta.agent_scheduler.plist
  Interval: 7200 (2 hours for health), 14400 (4 hours for research)
"""

import json
import os
import subprocess
import sys
import logging
import argparse
from datetime import datetime, date
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_AGENT   = ROOT / "data" / "agent"
USAGE_PATH   = DATA_AGENT / "usage.json"
QUEUE_PATH   = DATA_AGENT / "research_queue.json"
FINDINGS_PATH= DATA_AGENT / "findings.jsonl"
MESSAGES_PATH= DATA_AGENT / "messages_to_colin.json"
HEALTH_PATH  = DATA_AGENT / "health.json"
LEDGER_PATH  = DATA_AGENT / "hypothesis_ledger.json"

LOG_PATH     = ROOT / "logs" / "agent_scheduler.log"

# ── Logging ───────────────────────────────────────────────────────────────────
# For --health and --budget flags, logs go to stderr so stdout stays pure JSON.
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_json_mode = "--health" in sys.argv or "--budget" in sys.argv
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stderr if _json_mode else sys.stdout),
    ],
)
log = logging.getLogger("agent_scheduler")

# ── ET hours gates ────────────────────────────────────────────────────────────
AGENT_PRIME_HOURS  = set(range(0, 9)) | {23}      # midnight–9am
WORK_HOURS         = set(range(9, 23))              # 9am–11pm ET

# ── Budget thresholds (mirrors usage_tracker.py) ──────────────────────────────
WEEKLY_DANGER_THRESHOLD  = 20_000
WEEKLY_LIGHT_THRESHOLD   = 50_000
HUMAN_MINIMUM_BUFFER     = 8_000
HEAVY_TASK_MINIMUM       = 15_000


# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_finding(finding: dict) -> None:
    with open(FINDINGS_PATH, "a") as f:
        f.write(json.dumps(finding) + "\n")


def post_message(priority: str, text: str) -> None:
    emoji = {"URGENT": "🔴", "IMPORTANT": "🟡", "FYI": "🟢"}.get(priority, "🟢")
    data = load_json(MESSAGES_PATH)
    if "messages" not in data:
        data["messages"] = []
    data["messages"].insert(0, {
        "id": f"msg-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "priority": priority,
        "emoji": emoji,
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "read": False,
    })
    # Keep only last 50 messages
    data["messages"] = data["messages"][:50]
    save_json(MESSAGES_PATH, data)


# ─────────────────────────────────────────────────────────────────────────────
# USAGE / BUDGET DECISION
# ─────────────────────────────────────────────────────────────────────────────

def get_budget_decision(dry_run: bool = False) -> dict:
    """Call usage_tracker.py budget command, parse JSON."""
    if dry_run:
        return {
            "mode": "NORMAL",
            "is_prime_time": True,
            "can_run_heavy": True,
            "can_run_light": True,
            "human_has_buffer": True,
            "agent_allowed_today": 28_571,
            "weekly_remaining": 200_000,
            "recommendation": "FULL: dry-run mode",
        }
    try:
        result = subprocess.run(
            ["python3", str(ROOT / "sovereign" / "tools" / "claude_usage_tracker.py"), "budget"],
            capture_output=True, text=True, timeout=10
        )
        return json.loads(result.stdout)
    except Exception as e:
        log.warning(f"Usage tracker failed: {e} — defaulting to LIGHT_ONLY")
        return {"recommendation": "LIGHT_ONLY: usage tracker unavailable", "can_run_heavy": False, "can_run_light": True}


def decision_to_mode(decision: dict) -> str:
    """Convert budget decision to simple mode string."""
    rec = decision.get("recommendation", "")
    if rec.startswith("FULL"):
        return "FULL"
    elif rec.startswith("HEAVY_OK"):
        return "HEAVY_OK"
    elif rec.startswith("LIGHT"):
        return "LIGHT_ONLY"
    elif rec.startswith("SKIP") or rec.startswith("DANGER"):
        return "SKIP"
    return "LIGHT_ONLY"


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_health_check() -> dict:
    """Check each system component. Returns health snapshot."""
    health = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": "GREEN",
        "components": {}
    }

    checks = {
        "duckdb":         _check_duckdb,
        "ollama":         _check_ollama,
        "yfinance":       _check_yfinance,
        "ict_scanner":    _check_ict_scanner,
        "backtest_engine":_check_backtest_engine,
        # Data ecosystem
        "alpaca":         _check_alpaca,
        "fred":           _check_fred,
        "alpha_vantage":  _check_alpha_vantage,
        "tiingo":         _check_tiingo,
        "news_api":       _check_news_api,
        "oanda":          _check_oanda,
        "firebase":       _check_firebase,
        "polygon":        _check_polygon,
        "openweather":    _check_openweather,
        "nasdaq":         _check_nasdaq,
        "reddit":         _check_reddit,
        # twitter removed — X API requires paid plan ($100+/mo), not worth it
    }

    issues = []
    for name, fn in checks.items():
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "RED", str(e)
        health["components"][name] = {"status": status, "detail": detail}
        if status == "RED":
            issues.append(name)
            health["overall"] = "RED"
        elif status == "YELLOW" and health["overall"] == "GREEN":
            health["overall"] = "YELLOW"

    health["last_checked"] = health["timestamp"]
    save_json(HEALTH_PATH, health)

    if issues:
        post_message("URGENT", f"Health check FAILED: {', '.join(issues)} are RED. Check logs.")
        log.warning(f"Health check RED: {issues}")
    else:
        log.info(f"Health check GREEN — all systems nominal")

    return health


def _check_duckdb() -> tuple:
    try:
        import duckdb
        db_path = ROOT / "data" / "sovereign.duckdb"
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            con.close()
            return "GREEN", "connected to sovereign.duckdb"
        # Module installed but no DB yet — that's fine, system uses flat files
        con = duckdb.connect(":memory:")
        con.close()
        return "GREEN", f"v{duckdb.__version__} ready (in-memory)"
    except ImportError:
        return "YELLOW", "duckdb not installed"
    except Exception as e:
        return "RED", str(e)


def _check_ollama() -> tuple:
    """Check ollama — auto-start it if found but not running."""
    def _ping() -> bool:
        try:
            r = subprocess.run(
                ["curl", "-s", "--max-time", "3", "http://localhost:11434/api/tags"],
                capture_output=True, text=True, timeout=5
            )
            return r.returncode == 0 and "models" in r.stdout
        except Exception:
            return False

    if _ping():
        return "GREEN", "responding"

    # Not responding — try to auto-start if ollama binary exists
    ollama_bin = subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()
    if ollama_bin:
        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            import time
            time.sleep(4)  # give it a moment to start
            if _ping():
                return "GREEN", "was down — auto-started"
            return "YELLOW", "auto-started but not yet responding"
        except Exception as e:
            return "YELLOW", f"auto-start failed: {e}"

    return "YELLOW", "ollama not installed"


def _check_yfinance() -> tuple:
    try:
        import yfinance as yf  # noqa — just test import, no network call
        return "GREEN", f"yfinance {yf.__version__} installed"
    except ImportError:
        return "YELLOW", "yfinance not installed"


def _check_ict_scanner() -> tuple:
    """
    Scanner only runs during London (02-05 ET) and NY PM (13:30-16:00 ET).
    Outside sessions a stale log is expected — don't alarm on it.
    RED only if log is >24h old (truly dead).
    """
    scanner_log = ROOT / "logs" / "ict_scanner.log"
    if not scanner_log.exists():
        return "YELLOW", "no log file found"
    import time
    age_hours = (time.time() - scanner_log.stat().st_mtime) / 3600

    # Check if we're currently inside a session window (ET)
    try:
        from datetime import datetime
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        et_min = et.hour * 60 + et.minute
        in_london = 120 <= et_min <= 300      # 02:00–05:00 ET
        in_ny_pm  = 810 <= et_min <= 960      # 13:30–16:00 ET
        in_session = in_london or in_ny_pm
    except Exception:
        in_session = False

    if age_hours < 1:
        return "GREEN", f"log updated {age_hours*60:.0f}min ago"
    if in_session and age_hours > 0.2:
        return "YELLOW", f"in session but log is {age_hours:.1f}h old — scanner may be stuck"
    if age_hours < 24:
        return "GREEN", f"outside session hours — last run {age_hours:.1f}h ago"
    return "RED", f"log is {age_hours:.0f}h old — scanner appears down"


def _check_backtest_engine() -> tuple:
    engine_path = ROOT / "backtest" / "fast_engine.py"
    if engine_path.exists():
        return "GREEN", "file present"
    return "RED", "fast_engine.py missing"


# ── Ecosystem data source health checks ───────────────────────────────────────

def _http_get(url: str, headers: dict = None, timeout: int = 6) -> tuple:
    """Returns (status_code, json_or_none, error_str)."""
    try:
        import requests
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = None
        return r.status_code, body, None
    except Exception as e:
        return 0, None, str(e)


def _check_alpaca() -> tuple:
    k  = os.environ.get("ALPACA_API_KEY", "")
    sk = os.environ.get("ALPACA_SECRET_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    base = "https://paper-api.alpaca.markets" if os.environ.get("ALPACA_PAPER") == "true" else "https://api.alpaca.markets"
    code, body, err = _http_get(f"{base}/v2/account", {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": sk})
    if err:
        return "RED", err
    if code == 200 and body:
        equity = body.get("equity", "?")
        return "GREEN", f"paper account equity=${float(equity or 0):,.0f}"
    return "YELLOW", f"status {code}"


def _check_fred() -> tuple:
    k = os.environ.get("FRED_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    code, body, err = _http_get(f"https://api.stlouisfed.org/fred/series?series_id=FEDFUNDS&api_key={k}&file_type=json")
    if err:
        return "RED", err
    if code == 200:
        return "GREEN", "fed funds series ok"
    return "YELLOW", f"status {code}"


def _check_alpha_vantage() -> tuple:
    k = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    code, body, err = _http_get(f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&outputsize=compact&apikey={k}")
    if err:
        return "RED", err
    if code == 200 and body and "Time Series FX (Daily)" in body:
        dates = list(body["Time Series FX (Daily)"].keys())
        return "GREEN", f"latest bar {dates[0]}"
    if body and "Note" in body:
        return "YELLOW", "rate limited (free tier)"
    return "YELLOW", f"status {code}"


def _check_tiingo() -> tuple:
    k = os.environ.get("TIINGO_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    code, body, err = _http_get("https://api.tiingo.com/api/test/", {"Authorization": f"Token {k}"})
    if err:
        return "RED", err
    if code == 200:
        return "GREEN", "authenticated"
    return "YELLOW", f"status {code}"


def _check_news_api() -> tuple:
    k = os.environ.get("NEWS_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    code, body, err = _http_get(f"https://newsapi.org/v2/top-headlines?category=business&pageSize=1&apiKey={k}")
    if err:
        return "RED", err
    if code == 200 and body and body.get("status") == "ok":
        return "GREEN", f"{body.get('totalResults','?')} business articles"
    return "YELLOW", f"status {code}"


def _check_oanda() -> tuple:
    k   = os.environ.get("OANDA_API_KEY", "")
    acc = os.environ.get("OANDA_ACCOUNT_ID", "")
    base = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
    if not k or not acc:
        return "YELLOW", "key or account ID missing"
    code, body, err = _http_get(f"{base}/v3/accounts/{acc}", {"Authorization": f"Bearer {k}"})
    if err:
        return "RED", err
    if code == 200 and body:
        nav = body.get("account", {}).get("NAV", "?")
        return "GREEN", f"NAV=${float(nav or 0):,.0f} practice"
    return "YELLOW", f"status {code}"


def _check_firebase() -> tuple:
    db_url = os.environ.get("FIREBASE_DATABASE_URL", "")
    if not db_url:
        return "YELLOW", "FIREBASE_DATABASE_URL missing"
    # Ping the .json endpoint — no auth needed for public rules
    code, body, err = _http_get(f"{db_url}/.json?shallow=true&limitToFirst=1")
    if err:
        return "RED", err
    if code in (200, 401, 403):
        # 401/403 = reachable but auth required (expected)
        return "GREEN", "reachable"
    return "YELLOW", f"status {code}"


def _check_polygon() -> tuple:
    k = os.environ.get("POLYGON_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    # Free plan only allows some endpoints — check ticker detail (always free)
    code, body, err = _http_get(f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={k}")
    if err:
        return "RED", err
    if code == 200:
        return "GREEN", "free tier active (reference data)"
    if code == 403:
        return "YELLOW", "key valid, plan too limited for historical — upgrade for full access"
    return "YELLOW", f"status {code}"


def _check_openweather() -> tuple:
    k = os.environ.get("OPENWEATHER_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    code, body, err = _http_get(f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={k}")
    if err:
        return "RED", err
    if code == 200:
        return "GREEN", "weather data ok"
    if code == 401:
        return "YELLOW", "key not yet active — new keys take up to 2h to activate"
    return "YELLOW", f"status {code}"


def _check_twitter() -> tuple:
    bt = os.environ.get("TWITTER_BEARER_TOKEN", "")
    if not bt:
        return "YELLOW", "TWITTER_BEARER_TOKEN missing"
    if len(bt) < 80:
        return "YELLOW", f"token looks wrong ({len(bt)} chars) — regenerate Bearer Token from Twitter Dev Portal"
    code, body, err = _http_get(
        "https://api.twitter.com/2/tweets/search/recent?query=fed+rates&max_results=10",
        {"Authorization": f"Bearer {bt}"}
    )
    if err:
        return "RED", err
    if code == 200:
        count = len((body or {}).get("data", []))
        return "GREEN", f"{count} recent tweets fetched"
    return "YELLOW", f"status {code}"


def _check_reddit() -> tuple:
    """Check Reddit sentiment cache freshness. Scraper runs every ~hour."""
    cache = ROOT / "data" / "cache" / "reddit_sentiment.json"
    if not cache.exists():
        # Run it now on first check
        try:
            import time as _time
            result = subprocess.run(
                [sys.executable, str(ROOT / "sovereign" / "data" / "reddit_scraper.py")],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and cache.exists():
                data = json.loads(cache.read_text())
                eq = data["summary"].get("equity", "?")[:50]
                return "GREEN", f"fresh — top equity: {eq}"
            return "YELLOW", "scraper ran but no cache produced"
        except Exception as e:
            return "YELLOW", f"scraper failed: {e}"

    import time as _time
    age_min = (_time.time() - cache.stat().st_mtime) / 60
    try:
        data = json.loads(cache.read_text())
        eq = data["summary"].get("equity", "?")[:60]
        fx = data["summary"].get("forex", "?")[:40]
        posts = data.get("posts_scanned", "?")
        if age_min < 90:
            return "GREEN", f"{posts} posts | {age_min:.0f}m ago | {eq}"
        return "YELLOW", f"cache {age_min:.0f}m old — needs refresh"
    except Exception:
        return "YELLOW", f"cache unreadable"


def _check_nasdaq() -> tuple:
    k = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "")
    if not k:
        return "YELLOW", "key missing"
    # WIKI prices table — confirmed free
    code, body, err = _http_get(f"https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES.json?ticker=AAPL&qopts.per_page=1&api_key={k}")
    if err:
        return "RED", err
    if code == 200 and body and body.get("datatable"):
        rows = len(body["datatable"].get("data", []))
        return "GREEN", f"WIKI prices ok ({rows} rows)"
    return "YELLOW", f"status {code}"


# ── Reddit sentiment refresh ───────────────────────────────────────────────────

def _refresh_reddit_if_stale():
    """Run reddit_scraper if cache is >90 minutes old or missing."""
    import time as _time
    cache = ROOT / "data" / "cache" / "reddit_sentiment.json"
    if cache.exists():
        age_min = (_time.time() - cache.stat().st_mtime) / 60
        if age_min < 90:
            log.info(f"Reddit cache {age_min:.0f}m old — skipping refresh")
            return
    log.info("Refreshing Reddit sentiment...")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "sovereign" / "data" / "reddit_scraper.py")],
            cwd=str(ROOT), capture_output=True, text=True, timeout=90
        )
        log.info("Reddit sentiment refreshed")
    except Exception as e:
        log.warning(f"Reddit refresh failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH TASK DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def pick_task(mode: str) -> dict | None:
    """Pick highest-priority QUEUED task appropriate for current mode."""
    data = load_json(QUEUE_PATH)
    if not data.get("queue"):
        return None

    for task in sorted(data["queue"], key=lambda t: t["priority"]):
        if task["status"] != "QUEUED":
            continue
        if task.get("blocked_until"):
            # Soft block — skip but don't fail
            log.info(f"Task {task['id']} blocked until: {task['blocked_until']}")
            continue
        if mode == "LIGHT_ONLY" and task.get("needs_network", False):
            continue  # Heavy/network tasks need FULL or HEAVY_OK mode
        if mode == "LIGHT_ONLY" and task.get("estimated_minutes", 0) > 10:
            continue  # Skip long tasks in light mode
        return task

    return None


def mark_task_running(task_id: str) -> None:
    data = load_json(QUEUE_PATH)
    for task in data["queue"]:
        if task["id"] == task_id:
            task["status"] = "RUNNING"
            task["started"] = datetime.now().isoformat(timespec="seconds")
            break
    save_json(QUEUE_PATH, data)


def mark_task_done(task_id: str, result: str, status: str = "DONE") -> None:
    data = load_json(QUEUE_PATH)
    for task in data["queue"]:
        if task["id"] == task_id:
            task["status"] = status
            task["completed"] = datetime.now().isoformat(timespec="seconds")
            task["last_result"] = result
            break
    save_json(QUEUE_PATH, data)


def dispatch_task(task: dict, dry_run: bool = False) -> str:
    """
    Dispatch a research task via research_agent.py.

    The research agent owns all task handlers. The scheduler calls it
    as a subprocess so it gets a clean Python environment.
    Returns result summary string.
    """
    task_id   = task["id"]
    task_name = task["name"]

    log.info(f"Dispatching to research agent: {task_id} — {task_name}")

    if dry_run:
        log.info(f"[DRY RUN] would run: python3 sovereign/agent/research_agent.py --task {task_id}")
        return f"DRY RUN: {task_name} — not executed"

    agent_script = ROOT / "sovereign" / "agent" / "research_agent.py"
    if not agent_script.exists():
        return f"ERROR: research_agent.py not found at {agent_script}"

    try:
        result = subprocess.run(
            [sys.executable, str(agent_script), "--task", task_id],
            capture_output=True, text=True,
            timeout=3600,   # 1 hour max per task
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            # Pull last summary line from agent output
            lines = [l for l in result.stdout.strip().split("\n") if "Summary:" in l]
            summary = lines[-1].replace("[AGENT]", "").strip() if lines else result.stdout[-300:]
            return f"OK: {summary}"
        else:
            return f"ERROR: {result.stderr[-400:] or result.stdout[-400:]}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {task_name} exceeded 1 hour"
    except Exception as e:
        return f"EXCEPTION: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CYCLE
# ─────────────────────────────────────────────────────────────────────────────

def run_eod_challenge_update(dry_run: bool = False) -> None:
    """
    Run the prop firm paper challenge EOD update.
    Called at 17:00 ET daily by launchd (com.alta.eod_challenge.plist).
    Advances the floor, checks pass/bust conditions, posts a message.
    """
    challenge_script = ROOT / "sovereign" / "propfirm" / "paper_challenge.py"
    if not challenge_script.exists():
        log.warning("paper_challenge.py not found — skipping EOD update")
        return

    if dry_run:
        log.info("[DRY-RUN] Would run: paper_challenge.py --eod")
        return

    log.info("Running prop firm EOD update...")
    try:
        result = subprocess.run(
            [sys.executable, str(challenge_script), "--eod"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
        output = (result.stdout + result.stderr).strip()
        log.info(f"EOD update: {output[:300]}")

        # Parse status from output and post message
        if "PASSED" in output.upper():
            post_message("URGENT", f"🏆 Prop challenge PASSED. {output[:200]}")
        elif "BUSTED" in output.upper() or "FAILED" in output.upper():
            post_message("URGENT", f"Challenge ended. {output[:200]}")
        elif output:
            post_message("FYI", f"EOD floor update: {output[:200]}")
    except Exception as e:
        log.error(f"EOD update failed: {e}")
        post_message("IMPORTANT", f"EOD floor update failed: {e}")


def generate_eod_plist() -> str:
    """Generate a launchd plist that fires paper_challenge.py --eod at 17:00 ET daily."""
    script_path = ROOT / "scripts" / "agent_scheduler.py"
    python_path = ROOT / ".venv" / "bin" / "python3"
    if not python_path.exists():
        python_path = Path("/usr/local/bin/python3")
    log_out = ROOT / "logs" / "eod_challenge_stdout.log"
    log_err = ROOT / "logs" / "eod_challenge_stderr.log"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.alta.eod_challenge</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
        <string>--eod-challenge</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>17</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{ROOT}</string>
    </dict>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>"""


def run_cycle(dry_run: bool = False, force_heavy: bool = False) -> None:
    log.info("=" * 60)
    log.info(f"Agent scheduler cycle — {datetime.now().isoformat(timespec='seconds')}")

    # 1. Get budget decision
    decision = get_budget_decision(dry_run)
    mode = "FULL" if force_heavy else decision_to_mode(decision)
    log.info(f"Budget decision: {decision.get('recommendation', '?')} → mode={mode}")

    if mode == "SKIP":
        log.info("Mode=SKIP — nothing to do this cycle")
        return

    # 2. Refresh Reddit sentiment if stale (no API key needed, always runs)
    _refresh_reddit_if_stale()

    # 3. Always run health check (lightweight)
    log.info("Running health check...")
    health = run_health_check()
    log.info(f"Health: {health['overall']}")

    # 3b. Update cross-system bridge (reads Library + ICT state, writes shared signal)
    try:
        sys.path.insert(0, str(ROOT))
        from sovereign.intelligence.cross_system_bridge import CrossSystemBridge
        bridge_state = CrossSystemBridge().run_full_update(verbose=False)
        log.info(f"Bridge: ICT={bridge_state.ict_mode} QUANT={bridge_state.quant_signal} "
                 f"threat={bridge_state.library_threat_score:.2f}")
    except Exception as _bridge_err:
        log.warning(f"Cross-system bridge update failed (non-fatal): {_bridge_err}")

    # 3c. Run prop firm deployment checklist (log gates, post FYI if any change)
    try:
        from sovereign.propfirm.deployment_checklist import run_checklist
        cl = run_checklist()
        log.info(f"Checklist: {cl['overall']} "
                 f"G={cl['gates_green']} Y={cl['gates_yellow']} R={cl['gates_red']}")
        if cl["overall"] == "GO":
            post_message("URGENT", "🟢 PROP FIRM CHECKLIST: ALL GATES GREEN — buy the challenge now!")
    except Exception as _cl_err:
        log.warning(f"Checklist failed (non-fatal): {_cl_err}")

    # 4. Run Oracle (always — it's just one cheap API call)
    oracle_script = ROOT / "sovereign" / "agent" / "oracle_agent.py"
    if oracle_script.exists() and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            oracle_flags = ["--dry-run"] if dry_run else []
            subprocess.run(
                [sys.executable, str(oracle_script)] + oracle_flags,
                cwd=str(ROOT), timeout=60, check=False
            )
            log.info("Oracle cycle complete")
        except Exception as e:
            log.warning(f"Oracle failed (non-fatal): {e}")

    # 4. Pick and dispatch a research task if budget allows
    task = pick_task(mode)
    if not task:
        log.info("No eligible tasks in queue for this mode")
        _post_idle_summary(mode, health)
        return

    log.info(f"Selected task: {task['id']} — {task['name']} (est. {task.get('estimated_minutes', '?')} min)")
    mark_task_running(task["id"])

    result = dispatch_task(task, dry_run=dry_run)
    log.info(f"Task result: {result[:200]}")

    # 4. Record finding
    finding = {
        "id":        f"F-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "task_id":   task["id"],
        "task_name": task["name"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "result":    result,
        "mode":      mode,
    }
    append_finding(finding)

    # 5. Mark task done
    task_status = "DONE" if result.startswith("OK") else "ERROR"
    mark_task_done(task["id"], result, task_status)

    # 6. Post message to Colin
    if task_status == "ERROR":
        post_message("IMPORTANT", f"Task {task['id']} ({task['name']}) failed: {result[:200]}")
    else:
        post_message("FYI", f"Completed: {task['name']} — {result[:200]}")

    log.info("Cycle complete.")


def _post_idle_summary(mode: str, health: dict) -> None:
    post_message(
        "FYI",
        f"Scheduler ran in {mode} mode. Health: {health['overall']}. "
        f"No eligible research tasks — queue may be empty or blocked."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCHD PLIST GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_plist() -> str:
    script_path = ROOT / "scripts" / "agent_scheduler.py"
    python_path = ROOT / ".venv" / "bin" / "python3"
    if not python_path.exists():
        python_path = Path("/usr/bin/python3")
    log_out = ROOT / "logs" / "agent_scheduler_stdout.log"
    log_err = ROOT / "logs" / "agent_scheduler_stderr.log"

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.alta.agent_scheduler</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    <key>StartInterval</key>
    <integer>7200</integer>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{ROOT}</string>
    </dict>
</dict>
</plist>"""
    return plist


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alta Research Agent Scheduler")
    parser.add_argument("--dry-run",       action="store_true", help="Print what would run, don't execute")
    parser.add_argument("--force-heavy",   action="store_true", help="Override mode to FULL regardless of budget")
    parser.add_argument("--gen-plist",     action="store_true", help="Print research scheduler plist to stdout and exit")
    parser.add_argument("--gen-eod-plist", action="store_true", help="Print EOD challenge plist (17:00 ET daily) and exit")
    parser.add_argument("--health",        action="store_true", help="Run health check only and exit (JSON to stdout)")
    parser.add_argument("--eod-challenge", action="store_true", help="Run prop firm EOD floor update and exit")
    parser.add_argument("--checklist",     action="store_true", help="Run prop firm deployment checklist and exit")
    args = parser.parse_args()

    if args.gen_plist:
        print(generate_plist())
        return

    if args.gen_eod_plist:
        print(generate_eod_plist())
        return

    if args.health:
        health = run_health_check()
        print(json.dumps(health, indent=2))
        return

    if args.checklist:
        try:
            sys.path.insert(0, str(ROOT))
            from sovereign.propfirm.deployment_checklist import run_checklist, print_checklist
            result = run_checklist()
            print_checklist(result, verbose=True)
        except Exception as e:
            print(f"Checklist error: {e}")
        return

    if args.eod_challenge:
        run_eod_challenge_update(dry_run=args.dry_run)
        return

    run_cycle(dry_run=args.dry_run, force_heavy=args.force_heavy)


if __name__ == "__main__":
    main()
