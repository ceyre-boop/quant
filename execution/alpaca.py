"""Single Alpaca client for the execution harness.

Consolidates the `keys()` / `get()` / `bars_for()` / `et_t()` helpers that were
copy-pasted (with divergent constants) into `research/gapper/hyp107_shadow.py`
and `research/yield_frontier/live_shadow.py`.

THE ERROR-SWALLOW BUG THIS FIXES
---------------------------------
Both shadows shared this retry loop::

    except Exception:
        time.sleep(5)
    ...
    raise RuntimeError(url[:110])

Every exception — parse errors, DNS failures, entitlement rejections — was
destroyed and replaced with an opaque truncated URL. That is why the
`gapper_shadow_scan` job's exit-1 was misdiagnosed as "an Alpaca bars fetch
failure" when the underlying cause was never visible. A retry loop that discards
the exception it is retrying is a debugging blindfold.

Replacement contract:
  429            -> sleep 10, retry (rate limit; transient by definition)
  403            -> raise IMMEDIATELY. This is the entitlement signal, not a
                    transient error. The old code slept 65s and retried, which
                    turned a permanent subscription failure into a silent stall.
  5xx / timeout  -> exponential backoff 2/4/8/16s, retry
  anything else  -> re-raise immediately, no retry
  on exhaustion  -> AlpacaError carrying url, status, body, attempts, __cause__

DATA ENTITLEMENT (measured 2026-07-18)
--------------------------------------
This account serves SIP HISTORICAL quotes and bars, but 403s on anything inside
a 15-minute recency window:
    quotes at -16 min -> 200        quotes at -13 min -> 403
    /quotes/latest    -> 403        /snapshot         -> 403
Real-time IEX is permitted but useless for this purpose: AAPL quoted
bid 314.75 / ask 347.97 (a ~10% spread) because IEX carries ~2% of volume.
Hence the harness captures quotes on a DEFERRED pass beyond the boundary.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
ROOT = Path(__file__).resolve().parents[1]

DATA_BASE = "https://data.alpaca.markets"
_TS = "%Y-%m-%dT%H:%M:%SZ"


class AlpacaError(RuntimeError):
    """Alpaca request failed. Carries everything needed to diagnose it."""

    def __init__(self, url: str, status: int | None, body: str, attempts: int):
        self.url = url
        self.status = status
        self.body = body
        self.attempts = attempts
        super().__init__(
            f"Alpaca request failed after {attempts} attempt(s): "
            f"status={status} url={url} body={body[:500]}"
        )


class AlpacaEntitlementError(AlpacaError):
    """403 — the subscription does not permit this request. Not retryable."""


def load_env(root: Path | None = None) -> None:
    """Populate os.environ from .env using setdefault (never clobber real env).

    Mirrors scripts/fetch_fred_economic.py:31-39 — the hand-rolled parser exists
    because launchd provides no shell environment, so python-dotenv's usual
    invocation path is not available in scheduled runs.
    """
    import os
    env_path = (root or ROOT) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def keys() -> tuple[str, str]:
    """Return (key_id, secret). Raises loudly if absent — never returns empty."""
    import os
    load_env()
    kid = os.environ.get("ALPACA_API_KEY", "").strip()
    sec = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not kid or not sec:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not found. Expected in "
            f"{ROOT / '.env'} or the process environment."
        )
    return kid, sec


def get(url: str, *, timeout: int = 45, max_attempts: int = 5) -> dict:
    """GET a JSON endpoint with typed, non-destructive error handling."""
    kid, sec = keys()
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})

    last_exc: BaseException | None = None
    last_status: int | None = None
    last_body = ""

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())

        except urllib.error.HTTPError as e:
            last_exc = e
            last_status = e.code
            try:
                last_body = e.read().decode("utf-8", "replace")
            except Exception:          # noqa: BLE001 - body already unavailable
                last_body = "<unreadable body>"

            if e.code == 403:
                # Entitlement, not transient. Fail immediately and visibly.
                raise AlpacaEntitlementError(url, 403, last_body, attempt) from e
            if e.code == 429:
                time.sleep(10)
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            # 4xx other than 403/429 will not fix themselves.
            raise AlpacaError(url, e.code, last_body, attempt) from e

        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            last_body = repr(e)
            time.sleep(2 ** attempt)
            continue

        except json.JSONDecodeError as e:
            # A malformed body is not transient; retrying hides the real problem.
            raise AlpacaError(url, last_status, f"invalid JSON: {e}", attempt) from e

    raise AlpacaError(url, last_status, last_body, max_attempts) from last_exc


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_TS)


def et_dt(day: date, t: dtime) -> datetime:
    """Build a tz-aware ET datetime from an ET session date + wall-clock time."""
    return datetime.combine(day, t, tzinfo=ET)


def et_t(bar: dict) -> dtime:
    """ET wall-clock time of an Alpaca bar/quote record."""
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).time()


def sip_ceiling(lag_minutes: int) -> datetime:
    """Latest UTC instant this subscription may query on the SIP feed."""
    return datetime.now(UTC) - timedelta(minutes=lag_minutes)


def minute_bars(symbol: str, day: date, *, lag_minutes: int = 17,
                start_et: dtime = dtime(9, 30),
                end_et: dtime = dtime(16, 10)) -> list[dict]:
    """1-minute SIP bars for one symbol on one ET session date."""
    s = et_dt(day, start_et)
    e = min(et_dt(day, end_et), sip_ceiling(lag_minutes))
    if e <= s:
        return []
    q = urllib.parse.urlencode({
        "symbols": symbol, "timeframe": "1Min",
        "start": _fmt(s), "end": _fmt(e),
        "feed": "sip", "adjustment": "split", "limit": "10000"})
    data = get(f"{DATA_BASE}/v2/stocks/bars?{q}")
    return (data.get("bars") or {}).get(symbol, [])


def daily_prev_close(symbol: str, day: date, *, lookback_days: int = 10) -> float | None:
    """Split-adjusted close of the last session strictly before `day`."""
    s = et_dt(day - timedelta(days=lookback_days), dtime(0, 1))
    e = et_dt(day - timedelta(days=1), dtime(23, 0))
    q = urllib.parse.urlencode({
        "symbols": symbol, "timeframe": "1Day",
        "start": _fmt(s), "end": _fmt(e),
        "feed": "sip", "adjustment": "split", "limit": "20"})
    d = (get(f"{DATA_BASE}/v2/stocks/bars?{q}").get("bars") or {}).get(symbol, [])
    return d[-1]["c"] if d else None


def movers(top: int = 50) -> list[dict]:
    """Top gainers from the Alpaca screener."""
    data = get(f"{DATA_BASE}/v1beta1/screener/stocks/movers?top={top}")
    return data.get("gainers", [])


def news(symbol: str, start: datetime, end: datetime, limit: int = 50) -> list[dict]:
    """News articles for a symbol in a UTC window."""
    q = urllib.parse.urlencode({
        "symbols": symbol, "start": _fmt(start), "end": _fmt(end),
        "limit": str(limit)})
    return get(f"{DATA_BASE}/v1beta1/news?{q}").get("news", [])


def raw_quotes(symbol: str, start: datetime, end: datetime,
               limit: int = 1000) -> list[dict]:
    """Historical SIP quotes in a UTC window (oldest-first, as Alpaca returns).

    403s if `end` falls inside the 15-minute recency window — that is the
    subscription boundary, surfaced as AlpacaEntitlementError rather than
    swallowed.
    """
    # NOTE: the symbol lives in the PATH for this endpoint. Passing it as a query
    # parameter too returns 400 "unexpected query parameter(s): symbol".
    q = urllib.parse.urlencode({
        "start": _fmt(start), "end": _fmt(end),
        "feed": "sip", "limit": str(limit)})
    return get(f"{DATA_BASE}/v2/stocks/{symbol}/quotes?{q}").get("quotes") or []
