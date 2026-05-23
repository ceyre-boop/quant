"""
Narrative intelligence wrapper for optional TradingAgents integration.

This module is deliberately off the live path. It runs on monthly cadence or
manual demand, caches results for 30 days, and converts qualitative output into
bounded modifiers that can be consumed by the existing quant stack later.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CACHE_DIR = Path("data/cache")
SOURCE_NAME = "tradingagents_v1"
TRADINGAGENTS_STATE_DIR = CACHE_DIR / "tradingagents"


@dataclass(frozen=True)
class NarrativeBias:
    symbol: str
    month: str
    direction: str
    modifier: float
    confidence: str
    key_risks: List[str]
    catalysts: List[str]
    agent_consensus: float
    cached_at: str
    source: str = SOURCE_NAME
    narrative_summary: str = ""


class NarrativeEngine:
    """
    Wrapper that calls TradingAgents for qualitative intelligence and converts
    its output to a numeric modifier for the conviction scorer.

    Runs monthly or on-demand. Never in the intraday loop.
    Results are cached in data/cache/narrative_{symbol}_{month}.json.
    """

    def __init__(
        self,
        cache_dir: Path | str = CACHE_DIR,
        ttl_days: int = 30,
        tradingagents_client: Optional[Any] = None,
        bridge_python: str = "python3.13",
        bridge_script: Path | str = Path("scripts/run_tradingagents_narrative.py"),
        use_subprocess_bridge: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._client = tradingagents_client
        self.bridge_python = bridge_python
        self.bridge_script = Path(bridge_script)
        self.use_subprocess_bridge = use_subprocess_bridge

    def get_narrative_bias(self, symbol: str, month: str) -> NarrativeBias:
        month = self._normalize_month(month)
        cache_path = self._cache_path(symbol, month)
        cached = self._read_cache(cache_path)
        if cached is not None and self._is_fresh(cached.cached_at):
            return cached

        raw = self._call_tradingagents(symbol=symbol, month=month)
        parsed = self._parse_tradingagents_output(symbol=symbol, month=month, payload=raw)
        self._write_cache(cache_path, parsed)
        return parsed

    def should_override_quant(self, narrative: NarrativeBias, quant_signal: str) -> bool:
        quant_signal = (quant_signal or "").upper()
        if quant_signal not in {"LONG", "SHORT"}:
            return False
        if narrative.confidence != "HIGH":
            return False
        if narrative.agent_consensus < 1.0:
            return False
        if narrative.direction == "BULLISH" and quant_signal == "SHORT":
            return True
        if narrative.direction == "BEARISH" and quant_signal == "LONG":
            return True
        return False

    def _call_tradingagents(self, symbol: str, month: str) -> Any:
        if self.use_subprocess_bridge:
            return self._call_tradingagents_subprocess(symbol=symbol, month=month)
        client = self._client or self._load_tradingagents_client()
        call_date = f"{month}-01"

        if hasattr(client, "propagate"):
            result = client.propagate(symbol, call_date)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[1]
            return result
        if hasattr(client, "run"):
            return client.run(symbol=symbol, date=call_date)
        if callable(client):
            return client(symbol=symbol, date=call_date)
        raise RuntimeError("TradingAgents client has no supported entrypoint")

    def _load_tradingagents_client(self) -> Any:
        try:
            from tradingagents.default_config import DEFAULT_CONFIG  # type: ignore
            from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "tradingagents is not installed; install it before requesting live narrative output"
            ) from exc

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = os.getenv("TRADINGAGENTS_LLM_PROVIDER", config["llm_provider"])
        config["deep_think_llm"] = os.getenv("TRADINGAGENTS_DEEP_MODEL", config["deep_think_llm"])
        config["quick_think_llm"] = os.getenv("TRADINGAGENTS_QUICK_MODEL", config["quick_think_llm"])
        config["max_debate_rounds"] = min(int(config.get("max_debate_rounds", 2)), 2)
        config["data_cache_dir"] = str(TRADINGAGENTS_STATE_DIR)
        config["memory_log_path"] = str(TRADINGAGENTS_STATE_DIR / "memory" / "trading_memory.md")
        config["results_dir"] = str(TRADINGAGENTS_STATE_DIR / "logs")
        return TradingAgentsGraph(debug=False, config=config)

    def _call_tradingagents_subprocess(self, symbol: str, month: str) -> Dict[str, Any]:
        # TradingAgents graph.propagate() needs YYYY-MM-DD not YYYY-MM
        # Convert month to last trading day of that month
        from datetime import datetime, timedelta
        import calendar as _cal
        try:
            y, m = [int(x) for x in month.split("-")]
            last_day = _cal.monthrange(y, m)[1]
            trade_date = f"{y:04d}-{m:02d}-{last_day:02d}"
        except Exception:
            trade_date = month  # fallback: pass as-is

        cmd = [
            self.bridge_python,
            str(self.bridge_script),
            "--symbol", symbol,
            "--month", trade_date,
            "--mode", "fast",
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Bridge interpreter '{self.bridge_python}' was not found"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"TradingAgents bridge failed with exit code {exc.returncode}: {stderr}"
            ) from exc

        stdout = (completed.stdout or "").strip()
        if not stdout:
            raise RuntimeError("TradingAgents bridge returned empty output")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("TradingAgents bridge did not return valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("TradingAgents bridge returned non-dict JSON")
        return payload

    def _parse_tradingagents_output(self, symbol: str, month: str, payload: Any) -> NarrativeBias:
        data = self._coerce_payload(payload)
        direction = self._extract_direction(data)
        confidence = self._extract_confidence(data)
        consensus = self._extract_consensus(data)
        modifier = self._modifier_from_direction(direction, confidence, consensus)
        risks = self._extract_string_list(data, ["key_risks", "risks", "risk_flags", "risk_manager"])
        catalysts = self._extract_string_list(data, ["catalysts", "news_catalysts", "events", "news"])
        summary = self._extract_summary(data)
        return NarrativeBias(
            symbol=symbol,
            month=month,
            direction=direction,
            modifier=modifier,
            confidence=confidence,
            key_risks=risks,
            catalysts=catalysts,
            agent_consensus=consensus,
            cached_at=self._utcnow().isoformat(),
            source=SOURCE_NAME,
            narrative_summary=summary,
        )

    def _coerce_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, NarrativeBias):
            return asdict(payload)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"summary": payload}
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "to_dict"):
            dumped = payload.to_dict()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "__dict__"):
            return dict(payload.__dict__)
        raise RuntimeError("TradingAgents output could not be parsed into a dict")

    def _extract_direction(self, data: Dict[str, Any]) -> str:
        raw = self._first_value(
            data,
            "direction",
            "overall_sentiment",
            "sentiment",
            "stance",
            "recommendation",
        )
        text = str(raw or "").upper()
        if any(token in text for token in ("BULL", "LONG", "POSITIVE")):
            return "BULLISH"
        if any(token in text for token in ("BEAR", "SHORT", "NEGATIVE")):
            return "BEARISH"
        return "NEUTRAL"

    def _extract_confidence(self, data: Dict[str, Any]) -> str:
        raw = self._first_value(data, "confidence", "confidence_level", "debate_confidence")
        if isinstance(raw, (int, float)):
            score = float(raw)
            if score >= 0.75:
                return "HIGH"
            if score >= 0.45:
                return "MEDIUM"
            return "LOW"
        text = str(raw or "").upper()
        if "HIGH" in text:
            return "HIGH"
        if "MED" in text:
            return "MEDIUM"
        if "LOW" in text:
            return "LOW"
        return "LOW"

    def _extract_consensus(self, data: Dict[str, Any]) -> float:
        raw = self._first_value(data, "agent_consensus", "consensus", "agreement", "unanimity")
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, (int, float)):
            value = float(raw)
            return max(0.0, min(value, 1.0 if value <= 1.0 else value / 100.0))
        if isinstance(raw, str):
            text = raw.strip().upper()
            if text == "UNANIMOUS":
                return 1.0
            try:
                value = float(text)
                return max(0.0, min(value, 1.0 if value <= 1.0 else value / 100.0))
            except ValueError:
                pass
        return 0.0

    def _extract_summary(self, data: Dict[str, Any]) -> str:
        raw = self._first_value(
            data,
            "narrative_summary",
            "summary",
            "debate_summary",
            "thesis",
            "analysis",
        )
        if raw is None:
            return ""
        if isinstance(raw, list):
            return " ".join(str(item) for item in raw[:5])
        return str(raw)

    def _extract_string_list(self, data: Dict[str, Any], keys: List[str]) -> List[str]:
        values = []
        for key in keys:
            raw = data.get(key)
            if raw is None:
                continue
            if isinstance(raw, list):
                values.extend(str(item).strip() for item in raw if str(item).strip())
            elif isinstance(raw, dict):
                values.extend(
                    str(value).strip()
                    for value in raw.values()
                    if isinstance(value, (str, int, float)) and str(value).strip()
                )
            elif isinstance(raw, str) and raw.strip():
                values.append(raw.strip())
        deduped = []
        seen = set()
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:10]

    def _modifier_from_direction(self, direction: str, confidence: str, consensus: float) -> float:
        if confidence == "LOW" or direction == "NEUTRAL":
            return 0.0
        if confidence == "HIGH" and consensus >= 0.8:
            base = 0.10
        elif confidence in {"HIGH", "MEDIUM"} and consensus >= 0.6:
            base = 0.05
        else:
            base = 0.0
        if direction == "BULLISH":
            return min(base, 0.20)
        if direction == "BEARISH":
            return max(-base, -0.20)
        return 0.0

    def _cache_path(self, symbol: str, month: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"narrative_{safe_symbol}_{month}.json"

    def _read_cache(self, path: Path) -> Optional[NarrativeBias]:
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return NarrativeBias(**payload)

    def _write_cache(self, path: Path, narrative: NarrativeBias) -> None:
        path.write_text(json.dumps(asdict(narrative), indent=2))

    def _is_fresh(self, cached_at: str) -> bool:
        try:
            cached = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        return self._utcnow() - cached <= self.ttl

    def _normalize_month(self, month: str) -> str:
        try:
            return datetime.strptime(month, "%Y-%m").strftime("%Y-%m")
        except ValueError as exc:
            raise ValueError("month must be formatted as YYYY-MM") from exc

    def _first_value(self, data: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def has_provider_credentials() -> bool:
        env_vars = (
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "ANTHROPIC_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "DASHSCOPE_API_KEY",
            "GLM_API_KEY",
            "OPENROUTER_API_KEY",
            "AZURE_OPENAI_API_KEY",
        )
        return any(os.getenv(name) for name in env_vars)
