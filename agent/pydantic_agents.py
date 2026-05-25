"""
Pydantic AI agent for J2 compound planning — structured, validated LLM output.
"""
import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
try:
    from pydantic_ai.models.openai import OpenAIChatModel as GeminiChatModel
except ImportError:
    from pydantic_ai.models.openai import OpenAIModel as GeminiChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from schemas.plan import ExecutionPlan, PlanStep
from agent.metrics import get_metrics_collector
from agent.logger import log

# ── Concurrency limiter — max 10 simultaneous Gemini calls ──────────────────
_LLM_SEMAPHORE = asyncio.Semaphore(10)
LLM_TIMEOUT_SECONDS = 15.0  # hard timeout per LLM call


def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r") as f:
        return f.read()


class PlanStepModel(BaseModel):
    tool: str = Field(description="Tool name to invoke")
    params: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlanModel(BaseModel):
    """Structured execution plan returned by Pydantic AI."""
    steps: List[PlanStepModel] = Field(
        description="Ordered list of tool calls to execute"
    )


def _gemini_model() -> GeminiChatModel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    provider = OpenAIProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )
    return GeminiChatModel("gemini-2.5-flash", provider=provider)


_planning_agent: Optional[Agent[None, ExecutionPlanModel]] = None


def get_planning_agent() -> Agent[None, ExecutionPlanModel]:
    global _planning_agent
    if _planning_agent is None:
        _planning_agent = Agent(
            _gemini_model(),
            output_type=ExecutionPlanModel,
            instructions=_load_prompt("system_conductor.txt"),
            retries=2,
        )
    return _planning_agent


def _build_user_prompt(message: str, order_context: Optional[dict]) -> str:
    """Build the full user prompt with order context injected."""
    lines = [f"Customer Message: {message}"]
    if order_context:
        lines.append("\nOrder Context (use these exact values for tool params):")
        lines.append(f"  order_id:        {order_context.get('order_id', 'unknown')}")
        lines.append(f"  status:          {order_context.get('status', 'unknown')}")
        lines.append(f"  payment_method:  {order_context.get('payment_method', 'original')}")
        lines.append(f"  customer_id:     {order_context.get('customer_id', 'unknown')}")
        if order_context.get("office_address"):
            addr = order_context["office_address"]
            lines.append(f"  office_address:  {addr}")
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


async def generate_plan_llm(
    message: str, order_context: Optional[dict] = None, trace_id: Optional[str] = None
) -> ExecutionPlan:
    """Pydantic AI planning for J2 — single validated LLM call with order context.

    Guards:
    - _LLM_SEMAPHORE  → max 10 concurrent Gemini calls (rate-limit protection)
    - asyncio.timeout → hard {LLM_TIMEOUT_SECONDS}s wall-clock limit
    """
    from agent.stream_events import emit_async
    agent = get_planning_agent()
    metrics = get_metrics_collector()
    start = time.perf_counter()
    prompt = _build_user_prompt(message, order_context)

    log.info({"event": "llm_call_start", "model": "gemini-2.5-flash",
              "operation": "planning_pydantic_ai"})

    # Emit a live "thinking" event so the frontend can show an indicator
    # while Gemini decides which tools to call.
    if trace_id:
        await emit_async(trace_id, {
            "type": "thinking",
            "content": "AI is analysing your request and deciding what to do..."
        })

    try:
        async with _LLM_SEMAPHORE:
            async with asyncio.timeout(LLM_TIMEOUT_SECONDS):
                result = await agent.run(prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = result.usage()
        metrics.record_llm_call(
            model="gemini-2.5-flash",
            operation="planning_pydantic_ai",
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )
        log.info({"event": "llm_call_end", "model": "gemini-2.5-flash",
                  "latency_ms": latency_ms, "tokens": usage.input_tokens + usage.output_tokens})

        plan_model = result.output
        return ExecutionPlan(
            steps=[
                PlanStep(tool=s.tool, params=s.params) for s in plan_model.steps
            ]
        )
    except TimeoutError as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.error({"event": "llm_timeout", "model": "gemini-2.5-flash",
                   "timeout_seconds": LLM_TIMEOUT_SECONDS, "latency_ms": latency_ms})
        metrics.record_llm_call(
            model="gemini-2.5-flash", operation="planning_pydantic_ai",
            prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
            success=False, error_message=f"Timeout after {LLM_TIMEOUT_SECONDS}s",
        )
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.error({"event": "llm_call_failed", "model": "gemini-2.5-flash",
                   "error": str(e), "latency_ms": latency_ms})
        metrics.record_llm_call(
            model="gemini-2.5-flash", operation="planning_pydantic_ai",
            prompt_tokens=0, completion_tokens=0, latency_ms=latency_ms,
            success=False, error_message=str(e),
        )
        raise
