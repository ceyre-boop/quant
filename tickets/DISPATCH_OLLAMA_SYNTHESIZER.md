# DISPATCH — Ollama Local Synthesis + Phase 2 Relocation
## Alta Investments · Work Order for Claude Code
### Priority: Medium · Est: 1 day · Touches: sovereign/briefing/synthesize.py, scripts/daily_intelligence_pipeline.py

---

## The Mission

The briefing synthesizer currently requires a paid Anthropic API key to fire. Without
credits it falls back to deterministic output. Replace the inference layer with a
local Ollama model so the synthesizer runs for free on every Phase 2 cycle, on-device,
with no external dependency. Simultaneously move the synthesizer call from Phase 1
(warm-up) to Phase 2 (peak heat), where it belongs — the Mac is already maxed out
during Phase 2 and the local inference adds no marginal cost.

---

## Part 1 — Ollama Fallback Chain in `sovereign/briefing/synthesize.py`

### Setup (one-time, document in NEXT.md for Colin)

```bash
brew install ollama
ollama pull qwen2.5           # preferred: strong structured-output, 4.4GB
# fallback option: ollama pull llama3.1   # 4.7GB, also good
pip install ollama --break-system-packages
```

Confirm Ollama is running: `ollama list` should show the pulled model.

### Inference change

Replace the Anthropic client block with a three-tier fallback:

**Tier 1 — Ollama (local, free, always tried first):**
```python
import ollama as _ollama

def _synthesize_ollama(prompt: str) -> dict | None:
    try:
        resp = _ollama.chat(
            model="qwen2.5",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.2, "num_predict": 1024}
        )
        raw = resp["message"]["content"]
        data = _parse(raw)
        if data and "narrative" in data:
            data["model"] = "ollama/qwen2.5"
            data["cost_usd"] = 0.0
            return data
    except Exception:
        return None
    return None
```

**Tier 2 — Anthropic API (if key present and credits available):**
Existing `anthropic.Anthropic(...)` call, unchanged. Only reached if Ollama fails.

**Tier 3 — Deterministic fallback:**
Existing fallback, unchanged. Only reached if both above fail.

The `synthesize()` function tries them in order:
```python
result = _synthesize_ollama(prompt)
if result is None and api_key:
    result = _synthesize_anthropic(prompt, api_key)
if result is None:
    result = None   # caller uses deterministic fallback
return result
```

**`synthesis_source` field** in the output JSON reflects which tier fired:
`"ollama/qwen2.5"` | `"claude-opus-4-8"` | `"deterministic_fallback"`

No other changes to `synthesize.py`. The prompt, `_parse()`, output schema,
cost logging, and all downstream consumers are untouched.

### Why qwen2.5 over llama3.1

qwen2.5 is notably better at constrained JSON output and follows the format instruction
more reliably. The briefing prompt already injects all structured data — the model
just needs to read and synthesize, not reason from scratch. qwen2.5 handles this well.
If qwen2.5 is not pulled, fall back to llama3.1 automatically (check `ollama list`).

### Non-negotiables

- `format="json"` on every Ollama call — no exceptions. Without it, local models
  produce prose that breaks `_parse()`.
- `temperature=0.2` — low temperature for structured output reliability.
- Never retry on failure — if Ollama errors, move to Tier 2 immediately.
- Log which tier fired in `logs/oracle_cost.json` via the existing `_log_cost()`.
  For Ollama: `cost_usd = 0.0`, `model = "ollama/qwen2.5"`.

---

## Part 2 — Move Synthesizer to Phase 2 in `scripts/daily_intelligence_pipeline.py`

### Current (wrong) location: Phase 1

Phase 1 is warm-up — light fetches, no heavy compute. The synthesizer call does not
belong here. The raw data it needs (market state, lead-lag, volume profile) is
assembled in Phase 1, but the inference should not fire until Phase 2.

### New location: Phase 2, after feature assembly, before hypothesis batch

```
Phase 2 execution order:
  2a — Feature assembly (price + indicators + sentiment)    ← existing
  2b — XGBoost training + retrain_loop                     ← existing
  2c — Briefing synthesis (Ollama)                         ← MOVED HERE
  2d — Hypothesis batch (briefing output injected as context)
  2e — Oracle reflect cycle
  2f — ICT daily pass (conditional)
  2g — Checkpoint
```

The briefing synthesis fires while the XGBoost run is completing — the Mac is already
under full load. The local inference runs on the same hardware with no additional
power cost. By the time Phase 2 checkpoint writes, `daily_briefing.json` is fresh
and the hypothesis batch has the day's regime read injected as context.

### Implementation

In `daily_intelligence_pipeline.py`:

Phase 1: fetch data, write raw JSON files (`market_state.json`, `lead_lag.json`,
`volume_profile.json`, `news.json`, `event_calendar.json`). No synthesis call.

Phase 2 step 2c: read those JSON files, call `synthesize()`, write
`data/agent/daily_briefing.json`. If synthesis returns None, write the
deterministic fallback (same as today). Never block Phase 2 on a synthesis failure.

---

## Definition of Done

- [ ] `ollama pull qwen2.5` documented in NEXT.md setup steps for Colin
- [ ] `synthesize.py` tries Ollama first, Anthropic second, deterministic third
- [ ] `synthesis_source` field correctly identifies which tier fired
- [ ] Phase 1 does NOT call the synthesizer
- [ ] Phase 2 step 2c calls the synthesizer after feature assembly
- [ ] Hypothesis batch receives `daily_briefing.json` as injected context
- [ ] `logs/oracle_cost.json` logs Ollama calls with `cost_usd = 0.0`
- [ ] Manual test: `python3 scripts/daily_intelligence_pipeline.py --phase 2`
  produces `daily_briefing.json` with `synthesis_source: "ollama/qwen2.5"`
- [ ] `NEXT.md` updated, pushed

---

*Alta Investments · Dispatch Work Order · 2026-07-23*
*"Free inference, right window, fans already spinning."*
