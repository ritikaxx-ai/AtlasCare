"""
Groq-based agent for J2 compound planning — direct REST API calls via httpx.
"""
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from schemas.plan import ExecutionPlan, PlanStep
from agent.metrics import get_metrics_collector
from agent.logger import log

# ── Groq REST API endpoint ──────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Concurrency limiter — max 10 simultaneous Groq calls ───────────────────
_LLM_SEMAPHORE = asyncio.Semaphore(10)
LLM_TIMEOUT_SECONDS = 15.0


def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r") as f:
        return f.read()


def _build_user_prompt(message: str, order_context: Optional[dict]) -> str:
    """Build the full user prompt with order context injected."""
    lines = [f"Customer Message: {message}"]
    if order_context:
        lines.append("\nOrder Context (use these exact values for tool params):")
        lines.append(f"  order_id:        {order_context.get('order_id', 'unknown')}")
        lines.append(f"  status:          {order_context.get('status', 'unknown')}")
        lines.append(f"  payment_method:  {order_context.get('payment_method', 'original')}")
        lines.append(f"  customer_id:     {order_context.get('customer_id', 'unknown')}")
        saved_labels = []
        for addr_key, label in (("home_address", "home"), ("office_address", "office")):
            if order_context.get(addr_key):
                saved_labels.append(label)
        if saved_labels:
            lines.append(f"  saved_address_labels: {saved_labels}  (use one as address_label param)")
        items = order_context.get("items", [])
        if items:
            lines.append("  items:")
            for item in items:
                lines.append(
                    f"    - line_id={item['line_id']}  status={item['status']}"
                    f"  name='{item['name']}'  unit_price={item['unit_price']}"
                    f"  qty={item.get('quantity', 1)}"
                )
    return "\n".join(lines)


def _parse_plan(content: str) -> ExecutionPlan:
    """Parse JSON plan from Groq response, stripping markdown fences if present."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    steps = [PlanStep(tool=s["tool"], params=s.get("params", {})) for s in data["steps"]]
    return ExecutionPlan(steps=steps)


async def generate_plan_llm(
    message: str, order_context: Optional[dict] = None, trace_id: Optional[str] = None
) -> ExecutionPlan:
    """Groq REST API planning for J2 — direct HTTP call, no SDK.

    Guards:
    - _LLM_SEMAPHORE  → max 10 concurrent Groq calls
    - asyncio.wait_for → hard {LLM_TIMEOUT_SECONDS}s wall-clock limit
    """
    from agent.stream_events import emit_async
    metrics = get_metrics_collector()
    start = time.perf_counter()
    system_prompt = _load_prompt("system_conductor.txt")
    user_prompt = _build_user_prompt(message, order_context)

    log.info({"event": "llm_call_start", "model": GROQ_MODEL,
              "operation": "planning_groq_api"})

    if trace_id:
        await emit_async(trace_id, {
            "type": "thinking",
            "content": "AI is analysing your request and deciding what to do..."
        })

    try:
        async with _LLM_SEMAPHORE:
            result = await asyncio.wait_for(
                _call_groq_api(system_prompt, user_prompt),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        latency_ms = int((time.perf_counter() - start) * 1000)

        prompt_tokens = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)
        content = result["content"]

        metrics.record_llm_call(
            model=GROQ_MODEL,
            operation="planning_groq_api",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=True,
        )
        log.info({"event": "llm_call_end", "model": GROQ_MODEL,
                  "latency_ms": latency_ms, "tokens": prompt_tokens + completion_tokens})

        return _parse_plan(content)

    except asyncio.TimeoutError:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.error({"event": "llm_timeout", "model": GROQ_MODEL,
                   "timeout_seconds": LLM_TIMEOUT_SECONDS, "latency_ms": latency_ms})
        metrics.record_llm_call(
            model=GROQ_MODEL, operation="planning_groq_api",
            prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
            success=False, error_message=f"Timeout after {LLM_TIMEOUT_SECONDS}s",
        )
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.error({"event": "llm_call_failed", "model": GROQ_MODEL,
                   "error": str(e), "latency_ms": latency_ms})
        metrics.record_llm_call(
            model=GROQ_MODEL, operation="planning_groq_api",
            prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
            success=False, error_message=str(e),
        )
        raise


async def _call_groq_api(system_prompt: str, user_prompt: str) -> dict:
    """Direct async HTTP POST to Groq REST API — no SDK, pure httpx."""
    import httpx

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "content": choice["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
