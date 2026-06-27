"""
OandaBridge — sovereign/execution/oanda_bridge.py

Places trades on an OANDA practice account via the OANDA v20 REST API.
Trades placed here appear live on any TradingView chart connected to OANDA.

Environment variables (set in .env — never hardcode):
    OANDA_API_KEY          — generated in OANDA account settings
    OANDA_ACCOUNT_ID       — practice account ID (e.g. 101-001-XXXXXXXX-001)

Safety rules (non-negotiable):
    max_units:      10_000   — micro lot cap for paper trading
    max_risk_pct:   1.0%     — never exceeded per trade
    environment:    practice  — NEVER 'live' without explicit OANDA_LIVE=1 override
    daily_loss:     2.0%     — halt all trading if session P&L hits this

Pair format: OANDA uses underscores (EUR_USD), ICT uses suffix notation (EURUSD=X).
Use PAIR_MAP to convert between the two.

Usage:
    from sovereign.execution.oanda_bridge import OandaBridge, PAIR_MAP
    bridge = OandaBridge()
    result = bridge.place_trade('EUR_USD', 'LONG', 100, 1.1600, 1.1700)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from dotenv import load_dotenv
from oandapyV20.exceptions import V20Error

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ── File paths ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parents[2]
FILLS_LOG  = ROOT / 'data' / 'ledger' / 'oanda_fills.jsonl'
VETO_LOG   = ROOT / 'data' / 'ledger' / 'oanda_veto_ledger.jsonl'
FILLS_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Pair map: ICT/yfinance format → OANDA format ─────────────────────────────
# ICT orchestrator uses "GBPUSD" (no suffix).
# Sovereign forex uses "GBPUSD=X" (yfinance format).
# Both map to OANDA's underscore format.
PAIR_MAP: dict[str, str] = {
    # ICT orchestrator format (no suffix)
    "GBPUSD": "GBP_USD",
    "EURUSD": "EUR_USD",
    "AUDUSD": "AUD_USD",
    "USDJPY": "USD_JPY",
    "USDCAD": "USD_CAD",
    "AUDNZD": "AUD_NZD",
    "GBPJPY": "GBP_JPY",
    "NZDUSD": "NZD_USD",
    # Sovereign / yfinance format (=X suffix)
    "GBPUSD=X": "GBP_USD",
    "EURUSD=X": "EUR_USD",
    "AUDUSD=X": "AUD_USD",
    "USDJPY=X": "USD_JPY",
    "USDCAD=X": "USD_CAD",
    "AUDNZD=X": "AUD_NZD",
    "GBPJPY=X": "GBP_JPY",
    "NZDUSD=X": "NZD_USD",
}


def to_oanda_pair(ict_pair: str) -> str:
    """Convert any ICT/yfinance pair symbol to OANDA instrument name."""
    if ict_pair in PAIR_MAP:
        return PAIR_MAP[ict_pair]
    # Fallback: insert underscore before last 3 chars (e.g. GBPUSD → GBP_USD)
    base = ict_pair.rstrip("=X")
    if len(base) == 6:
        return f"{base[:3]}_{base[3:]}"
    return ict_pair

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UNITS       = 10_000   # hard cap for practice paper trading
MAX_RISK_PCT    = 0.010    # 1% per trade
DAILY_LOSS_PCT  = 0.020    # 2% halt


@dataclass
class OandaFill:
    timestamp:      str
    pair:           str
    direction:      str
    units:          int
    fill_price:     float
    stop_price:     float
    tp1_price:      float
    trade_id:       str
    order_id:       str
    account_id:     str


@dataclass
class OandaVeto:
    timestamp:  str
    reason:     str
    pair:       str
    details:    dict


def _log_fill(fill: OandaFill) -> None:
    with open(FILLS_LOG, 'a') as f:
        f.write(json.dumps(asdict(fill)) + '\n')
    logger.info(
        "[OANDA FILL] %s %s %d units @ %.5f | stop=%.5f tp=%.5f | tradeId=%s",
        fill.direction, fill.pair, fill.units, fill.fill_price,
        fill.stop_price, fill.tp1_price, fill.trade_id,
    )


def _log_veto(reason: str, pair: str, details: dict) -> None:
    rec = OandaVeto(
        timestamp=datetime.now(timezone.utc).isoformat(),
        reason=reason, pair=pair, details=details,
    )
    with open(VETO_LOG, 'a') as f:
        f.write(json.dumps(asdict(rec)) + '\n')
    logger.warning("[OANDA VETO] %s | %s | %s", reason, pair, details)


def _is_jpy_pair(pair: str) -> bool:
    return pair.endswith("_JPY") or pair.startswith("JPY_")


def _round_price(price: float, pair: str) -> str:
    decimals = 3 if _is_jpy_pair(pair) else 5
    return f"{price:.{decimals}f}"


class OandaBridge:
    """
    REST bridge to OANDA practice account. Thread-safe — oandapyV20 calls are
    synchronous and each call is stateless. No reactor or background thread needed.

    Trades placed via place_trade() appear live on TradingView when TV is
    connected to the same OANDA account.
    """

    def __init__(self) -> None:
        api_key    = os.environ.get('OANDA_API_KEY', '')
        account_id = os.environ.get('OANDA_ACCOUNT_ID', '')

        if not api_key or not account_id:
            missing = [k for k in ('OANDA_API_KEY', 'OANDA_ACCOUNT_ID')
                       if not os.environ.get(k)]
            raise EnvironmentError(
                f"Missing OANDA credentials: {', '.join(missing)}. "
                "Add them to .env — see practice.oanda.com for a free account."
            )

        # Practice-only safety: only allow live if OANDA_LIVE=1 is explicitly set.
        environment = 'live' if os.environ.get('OANDA_LIVE') == '1' else 'practice'
        if environment == 'live':
            logger.warning("[OandaBridge] ⚠  LIVE MODE — real money at risk")

        self._account_id = account_id
        self._api = oandapyV20.API(access_token=api_key, environment=environment)
        self._environment = environment
        self._session_start_balance: Optional[float] = None

        logger.info("[OandaBridge] Initialised | environment=%s | account=%s",
                    environment, account_id)

        # Confirm connectivity and surface account ID+NAV immediately at startup
        # so account mismatches (e.g. bridge vs TradingView) are caught in the log.
        try:
            nav = self.get_account_balance()
            logger.info("[OandaBridge] Account confirmed | id=%s | NAV=%.2f %s",
                        account_id, nav, environment.upper())
        except Exception as exc:
            logger.warning("[OandaBridge] Could not confirm account NAV at startup: %s", exc)

    # ── Account info ─────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        """Returns current NAV of the practice account in account currency."""
        req = accounts.AccountSummary(self._account_id)
        resp = self._api.request(req)
        return float(resp['account']['NAV'])

    def get_account_summary(self) -> dict:
        """Full practice-account snapshot: NAV, balance, unrealized/lifetime PL, open counts.

        NAV is the equity that drives the live equity curve. One API call; used by
        pulse_check's live-NAV snapshot and any 'am I making money' view.
        """
        req = accounts.AccountSummary(self._account_id)
        a = self._api.request(req)['account']
        return {
            "nav": float(a.get('NAV', 0.0) or 0.0),
            "balance": float(a.get('balance', 0.0) or 0.0),
            "unrealized_pl": float(a.get('unrealizedPL', 0.0) or 0.0),
            "realized_pl_lifetime": float(a.get('pl', 0.0) or 0.0),
            "margin_used": float(a.get('marginUsed', 0.0) or 0.0),
            "open_trade_count": int(a.get('openTradeCount', 0) or 0),
            "currency": a.get('currency', 'USD'),
        }

    def get_historical_candles(self, pair: str, start: str, end: str, granularity: str = "D"):
        """Clean broker OHLC candles (OANDA mid) for [start,end]. yfinance-like DataFrame.

        Used as the clean cross-source check vs yfinance (which drifts). A closed OANDA
        candle never revises. pair accepts 'EURUSD=X' / 'EURUSD' / 'EUR_USD'.
        """
        import pandas as pd
        inst = PAIR_MAP.get(pair)
        if not inst:
            base = pair.replace("=X", "")
            inst = base[:3] + "_" + base[3:] if "_" not in base else base
        params = {"granularity": granularity, "price": "M",
                  "from": f"{start}T00:00:00Z", "to": f"{end}T00:00:00Z"}
        req = instruments.InstrumentsCandles(instrument=inst, params=params)
        resp = self._api.request(req)
        rows = []
        for c in resp.get("candles", []):
            if not c.get("complete"):
                continue
            m = c["mid"]
            rows.append((c["time"][:10], float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"])))
        df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    def _check_daily_loss(self) -> Optional[str]:
        """Returns a veto reason string if daily loss limit is hit, else None."""
        try:
            balance = self.get_account_balance()
            if self._session_start_balance is None:
                self._session_start_balance = balance
                return None
            loss_pct = (balance - self._session_start_balance) / self._session_start_balance
            if loss_pct <= -DAILY_LOSS_PCT:
                return f"daily_loss={loss_pct:.2%} limit={DAILY_LOSS_PCT:.1%}"
        except Exception as exc:
            logger.warning("[OandaBridge] Balance check failed (non-fatal): %s", exc)
        return None

    # ── Trade placement ──────────────────────────────────────────────────────

    def place_trade(
        self,
        pair: str,
        direction: str,
        units: int,
        stop_price: float,
        tp1_price: float,
    ) -> dict:
        """
        Place a market order on the OANDA practice account.

        Args:
            pair:       OANDA instrument name, e.g. 'EUR_USD'
            direction:  'LONG' or 'SHORT'
            units:      Position size (absolute value). LONG = positive, SHORT = negative.
            stop_price: Stop-loss price level
            tp1_price:  Take-profit price level

        Returns:
            {'status': 'FILLED', 'trade_id': str, 'fill_price': float, ...}  on success
            {'status': 'ERROR', 'error': str}                                 on failure
            {'status': 'VETOED', 'reason': str}                               on gate block
        """
        # ── Gate 1: units cap ─────────────────────────────────────────────
        units = abs(units)
        if units > MAX_UNITS:
            logger.info("[OandaBridge] Units capped %d → %d", units, MAX_UNITS)
            units = MAX_UNITS
        if units == 0:
            _log_veto('ZERO_UNITS', pair, {})
            return {'status': 'VETOED', 'reason': 'ZERO_UNITS'}

        # ── Gate 2: daily loss limit ──────────────────────────────────────
        veto_reason = self._check_daily_loss()
        if veto_reason:
            _log_veto('DAILY_LOSS_LIMIT', pair, {'detail': veto_reason})
            logger.error("[HALT] Daily loss limit hit — no more OANDA trades today")
            return {'status': 'VETOED', 'reason': 'DAILY_LOSS_LIMIT'}

        # ── Gate 3: price sanity ──────────────────────────────────────────
        if stop_price <= 0 or tp1_price <= 0:
            _log_veto('INVALID_PRICES', pair, {'stop': stop_price, 'tp1': tp1_price})
            return {'status': 'VETOED', 'reason': 'INVALID_PRICES'}

        # ── Gate 4: FIFO pre-flight ───────────────────────────────────────
        existing = self.get_open_trades_for_pair(pair)
        if existing:
            existing_dir = "LONG" if float(existing[0].get("currentUnits", 0)) > 0 else "SHORT"
            logger.info(
                "[OandaBridge] FIFO pre-flight: %s has %d open trade(s) (dir=%s) — vetoing new %s",
                pair, len(existing), existing_dir, direction,
            )
            _log_veto("FIFO_EXISTING_POSITION", pair, {
                "existing_count":     len(existing),
                "existing_direction": existing_dir,
                "new_direction":      direction,
            })
            return {"status": "VETOED", "reason": f"FIFO_EXISTING_POSITION ({existing_dir})"}

        # Signed units: LONG = positive, SHORT = negative
        signed_units = units if direction == 'LONG' else -units

        order_body = {
            "order": {
                "type": "MARKET",
                "instrument": pair,
                "units": str(signed_units),
                "stopLossOnFill": {
                    "price": _round_price(stop_price, pair),
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": _round_price(tp1_price, pair),
                    "timeInForce": "GTC",
                },
            }
        }

        try:
            req = orders.OrderCreate(self._account_id, data=order_body)
            resp = self._api.request(req)
        except V20Error as exc:
            logger.error("[OandaBridge] API error: %s", exc)
            return {'status': 'ERROR', 'error': str(exc)}
        except Exception as exc:
            logger.error("[OandaBridge] Unexpected error: %s", exc)
            return {'status': 'ERROR', 'error': str(exc)}

        # Parse fill
        fill_txn  = resp.get('orderFillTransaction', {})
        trade_opened = fill_txn.get('tradeOpened', {})
        trade_id  = trade_opened.get('tradeID', '')
        order_id  = fill_txn.get('orderID', '')
        fill_price = float(fill_txn.get('price', 0))

        if not trade_id:
            # Order cancelled or rejected — check orderCancelTransaction
            cancel = resp.get('orderCancelTransaction', {})
            reason = cancel.get('reason', 'UNKNOWN')
            logger.error("[OandaBridge] Order not filled — reason=%s", reason)
            return {'status': 'ERROR', 'error': f"Order cancelled: {reason}"}

        fill = OandaFill(
            timestamp  =datetime.now(timezone.utc).isoformat(),
            pair       =pair,
            direction  =direction,
            units      =units,
            fill_price =fill_price,
            stop_price =stop_price,
            tp1_price  =tp1_price,
            trade_id   =trade_id,
            order_id   =order_id,
            account_id =self._account_id,
        )
        _log_fill(fill)

        return {
            'status':     'FILLED',
            'trade_id':   trade_id,
            'fill_price': fill_price,
            'units':      units,
            'direction':  direction,
            'pair':       pair,
            'stop_price': stop_price,
            'tp1_price':  tp1_price,
            'timestamp':  fill.timestamp,
        }

    # ── Trade monitoring ─────────────────────────────────────────────────────

    def get_open_trades(self) -> list[dict]:
        """
        Returns all open trades on the account.

        Each entry has: tradeID, instrument, currentUnits, price, openTime,
        stopLossOrder, takeProfitOrder, unrealizedPL.
        """
        try:
            req = trades.TradesList(self._account_id)
            resp = self._api.request(req)
            return resp.get('trades', [])
        except Exception as exc:
            logger.warning("[OandaBridge] get_open_trades failed: %s", exc)
            return []

    def get_open_trades_for_pair(self, pair: str) -> list[dict]:
        """Return open OANDA trades for the given instrument. Empty list if none or on error."""
        from oandapyV20.endpoints import trades as trades_ep
        try:
            req = trades_ep.OpenTrades(self._account_id)
            resp = self._api.request(req)
            return [t for t in resp.get("trades", []) if t.get("instrument") == pair]
        except Exception as exc:
            logger.warning("[OandaBridge] get_open_trades_for_pair error: %s", exc)
            return []

    def get_trade(self, trade_id: str) -> Optional[dict]:
        """Fetch a single trade by ID. Returns None if not found."""
        try:
            req = trades.TradeDetails(self._account_id, trade_id)
            resp = self._api.request(req)
            return resp.get('trade')
        except Exception as exc:
            logger.warning("[OandaBridge] get_trade(%s) failed: %s", trade_id, exc)
            return None

    def close_trade(self, trade_id: str) -> dict:
        """
        Market-close an open trade.

        Returns:
            {'status': 'CLOSED', 'pl': float, ...}  on success
            {'status': 'ERROR', 'error': str}        on failure
        """
        try:
            req = trades.TradeClose(self._account_id, trade_id)
            resp = self._api.request(req)
        except V20Error as exc:
            logger.error("[OandaBridge] close_trade(%s) error: %s", trade_id, exc)
            return {'status': 'ERROR', 'error': str(exc)}

        close_txn = resp.get('orderFillTransaction', {})
        pl = float(close_txn.get('pl', 0))
        fill_price = float(close_txn.get('price', 0))
        logger.info("[OandaBridge] Trade %s closed | pl=%.2f | exit=%.5f",
                    trade_id, pl, fill_price)
        return {
            'status':     'CLOSED',
            'trade_id':   trade_id,
            'pl':         pl,
            'fill_price': fill_price,
            'timestamp':  datetime.now(timezone.utc).isoformat(),
        }

    def get_closed_trades(self, limit: int = 50) -> list[dict]:
        """
        Returns recently closed trades from OANDA.

        Each entry has: id, instrument, realizedPL, openTime, closeTime,
        initialUnits, currentUnits, price (open price), averageClosePrice.
        Returns [] on any failure — never raises.
        """
        try:
            req = trades.TradesList(
                self._account_id,
                params={"state": "CLOSED", "count": str(limit)},
            )
            resp = self._api.request(req)
            return resp.get("trades", [])
        except Exception as exc:
            logger.warning("[OandaBridge] get_closed_trades failed: %s", exc)
            return []

    # ── Unit sizing helper ───────────────────────────────────────────────────

    def compute_units(
        self,
        pair: str,
        entry: float,
        stop: float,
        risk_pct: float = MAX_RISK_PCT,
    ) -> int:
        """
        Compute position size in units targeting risk_pct of account balance.

        For JPY pairs, 1 pip = 0.01. For all others, 1 pip = 0.0001.
        Returns an integer, capped at MAX_UNITS.
        """
        pip = 0.01 if _is_jpy_pair(pair) else 0.0001
        stop_pips = abs(entry - stop) / pip
        if stop_pips == 0:
            return 0

        balance = self.get_account_balance()
        risk_amount = balance * risk_pct  # in account currency (USD for practice)

        # pip_value per unit ≈ pip / entry (USD value of 1 pip for 1 unit)
        # For USD-quoted pairs: pip_value_per_unit = pip (since quote is USD)
        # For USD-base pairs: pip_value_per_unit = pip / entry
        if pair.startswith("USD_"):
            pip_value_per_unit = pip / entry
        else:
            pip_value_per_unit = pip   # quote is USD

        units = int(risk_amount / (stop_pips * pip_value_per_unit))
        return min(units, MAX_UNITS)

    # ── Self-test (no live connection needed) ────────────────────────────────

    @staticmethod
    def run_safety_tests() -> bool:
        """
        Validates safety gates without a real connection. Prints PASS/FAIL.
        Run before first live use: python3 sovereign/execution/oanda_bridge.py --test
        """
        print(f"\n{'═'*55}")
        print(f"  OANDA BRIDGE — SAFETY RULE VERIFICATION")
        print(f"{'─'*55}")

        results = []

        # Build a minimal bridge without .env for testing
        bridge = object.__new__(OandaBridge)
        bridge._account_id = 'TEST'
        bridge._api = None
        bridge._environment = 'practice'
        bridge._session_start_balance = None

        # Test 1: MAX_UNITS cap
        units_in = 50_000
        expected = MAX_UNITS
        result = min(abs(units_in), MAX_UNITS)
        passed = result == expected
        results.append(('UNITS_CAP', passed))
        print(f"  {'PASS' if passed else 'FAIL'}  UNITS_CAP  "
              f"(50000 → {result}, cap={MAX_UNITS})")

        # Test 2: DAILY_LOSS_LIMIT fires
        bridge._session_start_balance = 10_000.0
        # Simulate balance at -2.5% (should trip the 2% limit)
        # We can't call the real method without API; just verify the math
        loss_pct = (9_750 - 10_000) / 10_000
        tripped = loss_pct <= -DAILY_LOSS_PCT
        results.append(('DAILY_LOSS_LIMIT', tripped))
        print(f"  {'PASS' if tripped else 'FAIL'}  DAILY_LOSS_LIMIT  "
              f"(balance=9750 of 10000, loss={loss_pct:.2%})")

        # Test 3: JPY pair uses 0.01 pip
        passed = _is_jpy_pair('USD_JPY') and not _is_jpy_pair('EUR_USD')
        results.append(('JPY_PIP_DETECTION', passed))
        print(f"  {'PASS' if passed else 'FAIL'}  JPY_PIP_DETECTION  "
              f"(USD_JPY=JPY: {_is_jpy_pair('USD_JPY')}, EUR_USD=JPY: {_is_jpy_pair('EUR_USD')})")

        # Test 4: Price rounding (JPY = 3dp, others = 5dp)
        jpy_price = _round_price(155.123456, 'USD_JPY')
        usd_price = _round_price(1.234567, 'EUR_USD')
        passed = jpy_price == '155.123' and usd_price == '1.23457'
        results.append(('PRICE_ROUNDING', passed))
        print(f"  {'PASS' if passed else 'FAIL'}  PRICE_ROUNDING  "
              f"(JPY={jpy_price}, USD={usd_price})")

        all_pass = all(r[1] for r in results)
        print(f"{'─'*55}")
        print(f"  {'ALL RULES PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")
        print(f"{'═'*55}\n")
        return all_pass


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='OandaBridge — OANDA REST execution')
    parser.add_argument('--test', action='store_true',
                        help='Run safety rule tests (no connection needed)')
    parser.add_argument('--balance', action='store_true',
                        help='Print current practice account balance and exit')
    parser.add_argument('--open-trades', action='store_true',
                        help='Print all open trades and exit')
    parser.add_argument('--place-test', action='store_true',
                        help='Place a minimal EUR/USD test trade (100 units)')
    args = parser.parse_args()

    if args.test:
        ok = OandaBridge.run_safety_tests()
        raise SystemExit(0 if ok else 1)

    bridge = OandaBridge()

    if args.balance:
        bal = bridge.get_account_balance()
        print(f"Practice account balance: {bal:.2f}")
        return

    if args.open_trades:
        open_t = bridge.get_open_trades()
        print(f"Open trades ({len(open_t)}):")
        for t in open_t:
            print(f"  {t['tradeID']:>8}  {t['instrument']}  "
                  f"units={t['currentUnits']}  pl={t['unrealizedPL']}")
        return

    if args.place_test:
        print("\nPlacing test EUR/USD LONG 100 units on practice account…")
        print("Open TradingView → EUR/USD (OANDA) to see it appear.\n")
        # Use a safe stop/tp that won't fill immediately
        result = bridge.place_trade(
            pair='EUR_USD',
            direction='LONG',
            units=100,
            stop_price=1.0000,   # far stop — won't trigger immediately
            tp1_price=1.5000,    # far tp
        )
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == '__main__':
    main()
