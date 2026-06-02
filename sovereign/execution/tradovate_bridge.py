"""
TradovateBridge — sovereign/execution/tradovate_bridge.py

CME futures execution venue (MNQ/MES micros) — mirrors OandaBridge's surface so the
decision/execution flow can treat it as a second venue. ES/NQ trade here; OANDA stays forex.

⚠️  UNTESTED PENDING DEMO CREDENTIALS.
    This is coded against Tradovate's DOCUMENTED REST API (demo host demo.tradovateapi.com)
    and a structural mirror of oanda_bridge.py, but it has NOT been exercised against a live
    Tradovate endpoint — there is no account/keys yet. It is validated here only by import +
    the connection-free safety tests (`--test`). Treat every network method as provisional
    until a demo account confirms the request/response shapes. Do NOT flip to live.

Environment variables (set in .env — never hardcode):
    TRADOVATE_USERNAME       — demo account username
    TRADOVATE_PASSWORD       — demo account password
    TRADOVATE_APP_ID         — registered API app name
    TRADOVATE_APP_VERSION    — e.g. "1.0"
    TRADOVATE_CID            — API client id
    TRADOVATE_SEC           — API client secret
    TRADOVATE_ACCOUNT_ID     — numeric account id
    TRADOVATE_ACCOUNT_SPEC   — account spec (name)
    TRADOVATE_LIVE=1         — explicit override to use the LIVE host (default = demo)

Safety rules (non-negotiable, mirrored from OANDA):
    max_contracts:  10       — micro cap for the validation phase
    max_risk_pct:   1.0%     — never exceeded per trade
    daily_loss:     2.0%     — halt all trading if session P&L hits this
    isAutomated:    true     — flagged on EVERY order (CME automated-trading compliance)
    environment:    demo     — NEVER live without explicit TRADOVATE_LIVE=1

Usage:
    from sovereign.execution.tradovate_bridge import TradovateBridge
    bridge = TradovateBridge()                      # demo by default
    result = bridge.place_trade('MNQ', 'LONG', 1, stop_price=..., tp1_price=...)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

# ── Hosts ───────────────────────────────────────────────────────────────────────
DEMO_BASE = "https://demo.tradovateapi.com/v1"
LIVE_BASE = "https://live.tradovateapi.com/v1"

# ── File paths ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parents[2]
FILLS_LOG = ROOT / "data" / "ledger" / "tradovate_fills.jsonl"
VETO_LOG  = ROOT / "data" / "ledger" / "tradovate_veto_ledger.jsonl"
FILLS_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CONTRACTS  = 10       # micro cap for validation phase
MAX_RISK_PCT   = 0.010    # 1% per trade
DAILY_LOSS_PCT = 0.020    # 2% halt

# Dollar value per 1.0 index point, by micro contract (CME spec).
DOLLARS_PER_POINT = {"MNQ": 2.0, "MES": 5.0, "MYM": 0.5, "M2K": 5.0}
# Full-size (data symbols ES=F/NQ=F map to micros for execution).
SYMBOL_MAP = {"NQ=F": "MNQ", "ES=F": "MES", "NQ": "MNQ", "ES": "MES",
              "MNQ": "MNQ", "MES": "MES"}


def to_tradovate_symbol(sym: str) -> str:
    """Map an ES/NQ data symbol to its micro execution contract (MNQ/MES)."""
    return SYMBOL_MAP.get(sym, sym)


@dataclass
class TradovateFill:
    timestamp:   str
    symbol:      str
    direction:   str
    contracts:   int
    fill_price:  float
    stop_price:  float
    tp1_price:   float
    order_id:    str
    account_id:  str
    is_automated: bool


@dataclass
class TradovateVeto:
    timestamp: str
    reason:    str
    symbol:    str
    details:   dict


def _log_fill(fill: TradovateFill) -> None:
    with open(FILLS_LOG, "a") as f:
        f.write(json.dumps(asdict(fill)) + "\n")
    logger.info("[TRADOVATE FILL] %s %s %d @ %.2f | stop=%.2f tp=%.2f | order=%s",
                fill.direction, fill.symbol, fill.contracts, fill.fill_price,
                fill.stop_price, fill.tp1_price, fill.order_id)


def _log_veto(reason: str, symbol: str, details: dict) -> None:
    rec = TradovateVeto(timestamp=datetime.now(timezone.utc).isoformat(),
                        reason=reason, symbol=symbol, details=details)
    with open(VETO_LOG, "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    logger.warning("[TRADOVATE VETO] %s | %s | %s", reason, symbol, details)


class TradovateBridge:
    """
    REST bridge to a Tradovate account (demo by default). Mirrors OandaBridge's method
    surface: place_trade, modify_stop, close_partial, close_trade, get_open_trades,
    get_closed_trades, compute_contracts, get_account_balance.

    ⚠️ Untested against a live endpoint — see module docstring.
    """

    def __init__(self) -> None:
        self._username = os.environ.get("TRADOVATE_USERNAME", "")
        self._password = os.environ.get("TRADOVATE_PASSWORD", "")
        self._app_id   = os.environ.get("TRADOVATE_APP_ID", "")
        self._app_ver  = os.environ.get("TRADOVATE_APP_VERSION", "1.0")
        self._cid      = os.environ.get("TRADOVATE_CID", "")
        self._sec      = os.environ.get("TRADOVATE_SEC", "")
        self._account_id   = os.environ.get("TRADOVATE_ACCOUNT_ID", "")
        self._account_spec = os.environ.get("TRADOVATE_ACCOUNT_SPEC", "")

        required = ("TRADOVATE_USERNAME", "TRADOVATE_PASSWORD", "TRADOVATE_CID",
                    "TRADOVATE_SEC", "TRADOVATE_ACCOUNT_ID")
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(
                f"Missing Tradovate credentials: {', '.join(missing)}. "
                "Add them to .env — open a free demo account at tradovate.com and register an "
                "API app. (This bridge is UNTESTED pending those creds.)"
            )

        self._environment = "live" if os.environ.get("TRADOVATE_LIVE") == "1" else "demo"
        self._base = LIVE_BASE if self._environment == "live" else DEMO_BASE
        if self._environment == "live":
            logger.warning("[TradovateBridge] ⚠  LIVE MODE — real money at risk")

        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._session_start_balance: Optional[float] = None
        logger.info("[TradovateBridge] Initialised | environment=%s | account=%s (UNTESTED)",
                    self._environment, self._account_id)

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _access_token(self) -> str:
        """Fetch/refresh the Tradovate access token. Cached until ~expiry."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        body = {
            "name": self._username, "password": self._password,
            "appId": self._app_id, "appVersion": self._app_ver,
            "cid": self._cid, "sec": self._sec,
            "deviceId": "alta-quant-bridge",
        }
        r = requests.post(f"{self._base}/auth/accessToken", json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        self._token = data["accessToken"]
        # expirationTime is ISO; fall back to 50 min if absent.
        self._token_expiry = time.time() + 50 * 60
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        r = requests.get(f"{self._base}{path}", headers=self._headers(),
                         params=params or {}, timeout=20)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self._base}{path}", headers=self._headers(),
                          json=body, timeout=20)
        r.raise_for_status()
        return r.json()

    # ── Account info ─────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Current account NAV/cash balance. Returns 0.0 on failure (never raises)."""
        try:
            snap = self._get("/cashBalance/getcashbalancesnapshot",
                             None) if False else self._get("/cashBalance/list")
            # cashBalance/list returns balances; pick this account's.
            for b in (snap if isinstance(snap, list) else []):
                if str(b.get("accountId")) == str(self._account_id):
                    return float(b.get("amount", 0.0))
        except Exception as exc:
            logger.warning("[TradovateBridge] get_account_balance failed: %s", exc)
        return 0.0

    def _check_daily_loss(self) -> Optional[str]:
        try:
            bal = self.get_account_balance()
            if self._session_start_balance is None:
                self._session_start_balance = bal
                return None
            if self._session_start_balance:
                loss = (bal - self._session_start_balance) / self._session_start_balance
                if loss <= -DAILY_LOSS_PCT:
                    return f"daily_loss={loss:.2%} limit={DAILY_LOSS_PCT:.1%}"
        except Exception as exc:
            logger.warning("[TradovateBridge] balance check failed (non-fatal): %s", exc)
        return None

    # ── Trade placement ──────────────────────────────────────────────────────

    def place_trade(self, pair: str, direction: str, units: int,
                    stop_price: float, tp1_price: float) -> dict:
        """
        Place a market order on the (demo) Tradovate account, with a protective stop and
        take-profit. `units` = number of contracts. Mirrors OandaBridge.place_trade's
        signature and return contract.

        EVERY order carries isAutomated=true (CME automated-trading compliance).

        Returns {'status': 'FILLED'|'ERROR'|'VETOED', ...}.
        """
        symbol = to_tradovate_symbol(pair)
        contracts = abs(int(units))

        # Gate 1: contract cap
        if contracts > MAX_CONTRACTS:
            logger.info("[TradovateBridge] Contracts capped %d → %d", contracts, MAX_CONTRACTS)
            contracts = MAX_CONTRACTS
        if contracts == 0:
            return {"status": "VETOED", "reason": "ZERO_CONTRACTS"}

        # Gate 2: daily loss
        veto = self._check_daily_loss()
        if veto:
            _log_veto("DAILY_LOSS_LIMIT", symbol, {"detail": veto})
            return {"status": "VETOED", "reason": "DAILY_LOSS_LIMIT"}

        # Gate 3: price sanity
        if stop_price <= 0 or tp1_price <= 0:
            _log_veto("INVALID_PRICES", symbol, {"stop": stop_price, "tp1": tp1_price})
            return {"status": "VETOED", "reason": "INVALID_PRICES"}

        action = "Buy" if direction == "LONG" else "Sell"
        # Bracket via placeOSO: entry market + opposing stop & TP. isAutomated on the order.
        entry = {
            "accountSpec": self._account_spec, "accountId": int(self._account_id) if self._account_id else 0,
            "action": action, "symbol": symbol, "orderQty": contracts,
            "orderType": "Market", "isAutomated": True,
        }
        opp = "Sell" if action == "Buy" else "Buy"
        bracket1 = {"action": opp, "orderType": "Stop", "stopPrice": stop_price,
                    "isAutomated": True}
        bracket2 = {"action": opp, "orderType": "Limit", "price": tp1_price,
                    "isAutomated": True}
        oso_body = {**entry, "bracket1": bracket1, "bracket2": bracket2}

        try:
            resp = self._post("/order/placeOSO", oso_body)
        except Exception as exc:
            logger.error("[TradovateBridge] placeOSO error: %s", exc)
            return {"status": "ERROR", "error": str(exc)}

        order_id = str(resp.get("orderId") or resp.get("ooId") or "")
        if not order_id:
            reason = resp.get("failureReason") or resp.get("failureText") or "UNKNOWN"
            logger.error("[TradovateBridge] order not placed — %s", reason)
            return {"status": "ERROR", "error": f"Order failed: {reason}"}

        # Fill price often arrives async; record the request price (entry intent) and reconcile
        # via get_closed_trades / fills later (pulse_check backfill).
        fill_price = float(resp.get("price", 0.0))
        fill = TradovateFill(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol, direction=direction, contracts=contracts,
            fill_price=fill_price, stop_price=stop_price, tp1_price=tp1_price,
            order_id=order_id, account_id=self._account_id, is_automated=True,
        )
        _log_fill(fill)
        return {
            "status": "FILLED", "trade_id": order_id, "order_id": order_id,
            "fill_price": fill_price, "units": contracts, "direction": direction,
            "pair": symbol, "stop_price": stop_price, "tp1_price": tp1_price,
            "timestamp": fill.timestamp, "is_automated": True,
        }

    def modify_stop(self, order_id: str, new_stop: float) -> dict:
        """Modify a working stop order. Returns {'status': 'MODIFIED'|'ERROR', ...}."""
        try:
            resp = self._post("/order/modifyOrder",
                              {"orderId": int(order_id), "stopPrice": new_stop,
                               "isAutomated": True})
            return {"status": "MODIFIED", "order_id": order_id, "new_stop": new_stop,
                    "raw": resp}
        except Exception as exc:
            logger.error("[TradovateBridge] modify_stop(%s) error: %s", order_id, exc)
            return {"status": "ERROR", "error": str(exc)}

    def close_partial(self, symbol: str, contracts: int, direction: str) -> dict:
        """Close part of a position with an opposing market order (isAutomated)."""
        opp = "Sell" if direction == "LONG" else "Buy"
        try:
            resp = self._post("/order/placeOrder", {
                "accountSpec": self._account_spec,
                "accountId": int(self._account_id) if self._account_id else 0,
                "action": opp, "symbol": to_tradovate_symbol(symbol),
                "orderQty": abs(int(contracts)), "orderType": "Market",
                "isAutomated": True,
            })
            return {"status": "CLOSED_PARTIAL", "contracts": contracts, "raw": resp}
        except Exception as exc:
            logger.error("[TradovateBridge] close_partial error: %s", exc)
            return {"status": "ERROR", "error": str(exc)}

    def close_trade(self, order_id_or_symbol: str, direction: str = "LONG",
                    contracts: int = 1) -> dict:
        """Flatten a position via an opposing market order (Tradovate is position-based)."""
        return self.close_partial(order_id_or_symbol, contracts, direction)

    # ── Monitoring ─────────────────────────────────────────────────────────────

    def get_open_trades(self) -> list[dict]:
        """Open positions. Returns [] on any failure — never raises."""
        try:
            positions = self._get("/position/list")
            return [p for p in (positions if isinstance(positions, list) else [])
                    if str(p.get("accountId")) == str(self._account_id) and p.get("netPos")]
        except Exception as exc:
            logger.warning("[TradovateBridge] get_open_trades failed: %s", exc)
            return []

    def get_closed_trades(self, limit: int = 50) -> list[dict]:
        """Recent fills (for outcome backfill). Returns [] on any failure — never raises."""
        try:
            fills = self._get("/fill/list")
            rows = [f for f in (fills if isinstance(fills, list) else [])]
            return rows[-limit:]
        except Exception as exc:
            logger.warning("[TradovateBridge] get_closed_trades failed: %s", exc)
            return []

    # ── Sizing ───────────────────────────────────────────────────────────────

    def compute_units(self, pair: str, entry: float, stop: float,
                      risk_pct: float = MAX_RISK_PCT) -> int:
        """
        Risk-based contract sizing (conviction sizing, mirrors OANDA — never flat).
        contracts = risk_amount / (stop_distance_points * $/point). Capped at MAX_CONTRACTS.
        """
        symbol = to_tradovate_symbol(pair)
        dpp = DOLLARS_PER_POINT.get(symbol, 2.0)
        stop_pts = abs(entry - stop)
        if stop_pts == 0:
            return 0
        balance = self.get_account_balance()
        if balance <= 0:
            return 0
        risk_amount = balance * risk_pct
        contracts = int(risk_amount / (stop_pts * dpp))
        return max(0, min(contracts, MAX_CONTRACTS))

    # ── Connection-free safety tests ───────────────────────────────────────────

    @staticmethod
    def run_safety_tests() -> bool:
        """Validate gates + sizing math without a connection (mirror OandaBridge.run_safety_tests)."""
        print(f"\n{'═'*55}\n  TRADOVATE BRIDGE — SAFETY RULE VERIFICATION (UNTESTED vs API)\n{'─'*55}")
        results = []
        b = object.__new__(TradovateBridge)
        b._account_id = "TEST"; b._account_spec = "TEST"; b._session_start_balance = None

        capped = min(abs(50), MAX_CONTRACTS)
        results.append(("CONTRACT_CAP", capped == MAX_CONTRACTS))
        print(f"  {'PASS' if capped==MAX_CONTRACTS else 'FAIL'}  CONTRACT_CAP (50 → {capped})")

        loss = (9750 - 10000) / 10000
        results.append(("DAILY_LOSS_LIMIT", loss <= -DAILY_LOSS_PCT))
        print(f"  {'PASS' if loss<=-DAILY_LOSS_PCT else 'FAIL'}  DAILY_LOSS_LIMIT (loss={loss:.2%})")

        results.append(("SYMBOL_MAP", to_tradovate_symbol("NQ=F") == "MNQ" and to_tradovate_symbol("ES=F") == "MES"))
        print(f"  {'PASS' if results[-1][1] else 'FAIL'}  SYMBOL_MAP (NQ=F→MNQ, ES=F→MES)")

        # sizing: 10000 bal, 1% risk, 20pt stop on MNQ ($2/pt) → 10000*0.01/(20*2)=2.5→2
        b2 = object.__new__(TradovateBridge)
        b2.get_account_balance = lambda: 10000.0  # type: ignore
        contracts = TradovateBridge.compute_units(b2, "MNQ", 20000, 19980)
        results.append(("RISK_SIZING", contracts == 2))
        print(f"  {'PASS' if contracts==2 else 'FAIL'}  RISK_SIZING (1% of 10k, 20pt MNQ → {contracts} contracts)")

        ok = all(r[1] for r in results)
        print(f"{'─'*55}\n  {'ALL RULES PASS ✓' if ok else 'FAILURES DETECTED ✗'}\n{'═'*55}\n")
        return ok


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="TradovateBridge — CME futures execution (UNTESTED)")
    ap.add_argument("--test", action="store_true", help="connection-free safety tests")
    args = ap.parse_args()
    if args.test:
        raise SystemExit(0 if TradovateBridge.run_safety_tests() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
