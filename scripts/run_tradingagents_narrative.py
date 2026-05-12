#!/usr/bin/env python3
"""
Run TradingAgents out of process and emit a single JSON decision payload.

This script is intended to be executed from a dedicated Python 3.13
environment where TradingAgents and its provider dependencies are installed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

TRADINGAGENTS_STATE_DIR = Path("data/cache/tradingagents")
OLLAMA_DEFAULT = "http://127.0.0.1:11434/v1"
OLLAMA_SMOKE_MODEL = "qwen3:0.6b"


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _clean_list(value, limit: int = 5) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, dict):
        items = [str(v).strip() for v in value.values() if str(v).strip()]
    else:
        raw = str(value).strip()
        if not raw:
            return []
        items = [part.strip() for part in raw.replace("\n", " ").split("  ") if part.strip()]
        if len(items) == 1:
            items = [seg.strip() for seg in raw.split(";") if seg.strip()]
        if len(items) == 1:
            items = [seg.strip() for seg in raw.split(".") if seg.strip()]
    deduped = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:limit]


def _first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _extract_direction(raw_decision: str) -> str:
    rating_match = re.search(r"\*\*(?:Rating|Position Rating|position_rating)\*\*:\s*([A-Za-z]+)", raw_decision, re.IGNORECASE)
    if rating_match:
        rating = rating_match.group(1).strip().lower()
        if rating in {"buy", "overweight"}:
            return "LONG"
        if rating in {"sell", "underweight"}:
            return "SHORT"
        if rating == "hold":
            return "FLAT"

    action_match = re.search(r"\*\*Action\*\*:\s*([A-Za-z]+)", raw_decision, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip().lower()
        if action in {"buy", "long"}:
            return "LONG"
        if action in {"sell", "short"}:
            return "SHORT"
        if action == "hold":
            return "FLAT"

    text = raw_decision.lower()
    if "buy" in text or "bull" in text:
        return "LONG"
    if "sell" in text or "bear" in text:
        return "SHORT"
    return "FLAT"


def _smoke_tool_stubs(symbol: str, month: str) -> dict[str, object]:
    """
    Provide deterministic local tool outputs for the fast smoke path.

    TradingAgents still runs end-to-end, but it does not call out to Yahoo or
    other remote data sources. This keeps the smoke path low-latency and much
    gentler on CPU/fan usage.
    """
    digest = sum(ord(c) for c in f"{symbol}:{month}")
    base_price = 100 + (digest % 200)
    volume = 1_000_000 + (digest % 250_000)
    trend = "bullish" if digest % 3 == 0 else "bearish" if digest % 3 == 1 else "neutral"
    noise = "elevated" if digest % 5 == 0 else "contained"
    return {
        "get_stock_data": lambda *args, **kwargs: (
            f"{symbol} smoke data for {month}: price={base_price:.2f}, "
            f"volume={volume}, trend={trend}, volatility={noise}."
        ),
        "get_indicators": lambda *args, **kwargs: (
            f"{symbol} smoke indicators: ATR stable, RSI mid-range, "
            f"momentum {trend}, liquidity adequate."
        ),
        "get_fundamentals": lambda *args, **kwargs: (
            f"{symbol} smoke fundamentals: revenue growth steady, "
            f"margins stable, balance sheet acceptable."
        ),
        "get_balance_sheet": lambda *args, **kwargs: (
            f"{symbol} smoke balance sheet: leverage moderate, cash position fine."
        ),
        "get_cashflow": lambda *args, **kwargs: (
            f"{symbol} smoke cash flow: operating cash flow positive, "
            f"capital spending manageable."
        ),
        "get_income_statement": lambda *args, **kwargs: (
            f"{symbol} smoke income statement: profitability intact, "
            f"guidance risk {noise}."
        ),
        "get_news": lambda *args, **kwargs: (
            f"{symbol} smoke news: no major binary event; narrative bias {trend}."
        ),
        "get_insider_transactions": lambda *args, **kwargs: (
            f"{symbol} smoke insider flow: no obvious abnormal cluster."
        ),
        "get_global_news": lambda *args, **kwargs: (
            f"Macro smoke backdrop for {month}: rates steady, liquidity normal, "
            f"risk appetite {trend}."
        ),
    }


def _smoke_ticker_history(symbol: str, start: str, end: str):
    import pandas as pd

    digest = sum(ord(c) for c in f"{symbol}:{start}:{end}")
    base = 100.0 + (digest % 50)
    dates = pd.date_range(start=start, end=end, freq="B")
    if len(dates) < 2:
        dates = pd.date_range(start=start, periods=5, freq="B")
    rows = []
    for i, dt in enumerate(dates):
        close = base + (i * 0.8) + ((digest % 7) * 0.1)
        rows.append(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.2,
                "Close": close,
                "Volume": 1_000_000 + (i * 10_000),
            }
        )
    df = pd.DataFrame(rows, index=dates)
    df.index.name = "Date"
    return df


def _apply_smoke_yfinance_patch(symbol: str, month: str) -> None:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return

    class _FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, start=None, end=None, period=None, auto_adjust=True, **kwargs):
            if start and end:
                return _smoke_ticker_history(self.ticker, start, end)
            if period:
                import pandas as pd

                start_date = f"{month}-01"
                return _smoke_ticker_history(self.ticker, start_date, f"{month}-28")
            return _smoke_ticker_history(self.ticker, f"{month}-01", f"{month}-28")

        @property
        def news(self):
            return [{"title": f"{self.ticker} smoke news", "summary": f"{self.ticker} narrative remains stable."}]

        @property
        def info(self):
            return {"symbol": self.ticker, "longName": self.ticker, "sector": "Technology"}

        @property
        def fast_info(self):
            return {"lastPrice": 123.45}

        @property
        def financials(self):
            import pandas as pd

            return pd.DataFrame()

        balance_sheet = financials
        cashflow = financials
        quarterly_financials = financials

    yf.Ticker = _FakeTicker
    yf.download = lambda *args, **kwargs: _smoke_ticker_history(
        symbol, f"{month}-01", f"{month}-28"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TradingAgents narrative analysis")
    parser.add_argument("--symbol", required=True, help="Ticker or asset symbol")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--provider", default=os.getenv("TRADINGAGENTS_LLM_PROVIDER", "ollama"))
    parser.add_argument("--deep-model", default=os.getenv("TRADINGAGENTS_DEEP_MODEL"))
    parser.add_argument("--quick-model", default=os.getenv("TRADINGAGENTS_QUICK_MODEL"))
    args = parser.parse_args()

    try:
        from tradingagents.default_config import DEFAULT_CONFIG  # type: ignore
        from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "TradingAgents is not installed in this environment. "
            "Use a Python 3.13 environment and install the repo from source."
        ) from exc

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.provider
    if config["llm_provider"] == "ollama":
        config["backend_url"] = os.getenv("TRADINGAGENTS_BACKEND_URL", OLLAMA_DEFAULT)
        config["deep_think_llm"] = args.deep_model or os.getenv(
            "TRADINGAGENTS_DEEP_MODEL", OLLAMA_SMOKE_MODEL
        )
        config["quick_think_llm"] = args.quick_model or os.getenv(
            "TRADINGAGENTS_QUICK_MODEL", OLLAMA_SMOKE_MODEL
        )
    else:
        if args.deep_model:
            config["deep_think_llm"] = args.deep_model
        if args.quick_model:
            config["quick_think_llm"] = args.quick_model
    if args.mode == "fast":
        config["max_debate_rounds"] = 0
        config["max_risk_discuss_rounds"] = 0
        config["selected_analysts"] = ["market"]
        config["max_recur_limit"] = 10
        config["checkpoint_enabled"] = False
    else:
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        config["selected_analysts"] = ["market", "news", "fundamentals"]
        config["max_recur_limit"] = 20
    config["data_cache_dir"] = str(TRADINGAGENTS_STATE_DIR)
    config["memory_log_path"] = str(TRADINGAGENTS_STATE_DIR / "memory" / "trading_memory.md")
    config["results_dir"] = str(TRADINGAGENTS_STATE_DIR / "logs")

    if args.mode == "fast":
        import tradingagents.agents.utils.agent_utils as agent_utils  # type: ignore
        stubs = _smoke_tool_stubs(args.symbol.upper(), args.month)
        for name, fn in stubs.items():
            setattr(agent_utils, name, fn)
        _apply_smoke_yfinance_patch(args.symbol.upper(), args.month)

    ta = TradingAgentsGraph(selected_analysts=config["selected_analysts"], debug=False, config=config)
    final_state, decision = ta.propagate(args.symbol, f"{args.month}-01")

    raw_decision = str(final_state.get("final_trade_decision", decision or ""))
    investment = final_state.get("investment_debate_state", {}) or {}
    risk = final_state.get("risk_debate_state", {}) or {}
    summary = _first_nonempty(
        risk.get("judge_decision"),
        investment.get("judge_decision"),
        final_state.get("final_trade_decision"),
        raw_decision,
    ) or ""
    direction = _extract_direction(raw_decision)
    confidence = 0.60
    if "strong" in raw_decision.lower() or "high conviction" in raw_decision.lower():
        confidence = 0.80
    elif "moderate" in raw_decision.lower():
        confidence = 0.60
    elif "uncertain" in raw_decision.lower() or "hold" in raw_decision.lower():
        confidence = 0.40
    key_risks = _clean_list(_first_nonempty(risk.get("conservative_history"), risk.get("judge_decision"), final_state.get("news_report")))
    catalysts = _clean_list(_first_nonempty(investment.get("bull_history"), investment.get("judge_decision"), final_state.get("market_report"), final_state.get("sentiment_report"), final_state.get("fundamentals_report")))
    consensus = confidence if decision else 0.5
    if args.provider == "ollama":
        consensus = min(1.0, max(consensus, 0.65 if args.deep_model or args.quick_model else 0.55))
    modifier = 0.0
    if direction in {"LONG", "SHORT"}:
        if confidence >= 0.75 and consensus >= 0.8:
            modifier = 0.10 if direction == "LONG" else -0.10
        elif confidence >= 0.45 and consensus >= 0.6:
            modifier = 0.05 if direction == "LONG" else -0.05

    payload = {
        "direction": direction,
        "confidence": confidence,
        "narrative_confidence": _confidence_label(confidence),
        "narrative_modifier": modifier,
        "rationale": _clean_list(raw_decision, limit=5),
        "raw_signal": raw_decision[:800],
        "source": "trading_agents/qwen3",
        "symbol": args.symbol.upper(),
        "date": f"{args.month}-01",
        "month": args.month,
        "mode": args.mode,
        "summary": summary[:1200],
        "narrative_summary": summary[:1200],
        "key_risks": key_risks,
        "catalysts": catalysts,
        "news_catalysts": catalysts,
        "agent_consensus": round(float(consensus), 3),
    }
    print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
