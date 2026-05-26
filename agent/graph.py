"""
LangGraph orchestration for AtlasCare v3.0.

Flow:
  guardrail → intent (single LLM) → executor → synthesize → END

Guardrail handles hard rules (injection block, ₹25K escalation) with zero LLM calls.
Intent node: one Groq call decides which tools to call for any message.
"""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.guardrail import check_guardrails, check_prompt_injection
from agent.logger import log, mask
from agent.audit import (
    log_request_received, log_guardrail_triggered, log_guardrail_passed,
    log_prompt_injection, log_request_completed,
)
from agent.fast_paths import (
    build_j1_plan,
    build_j3_plan,
    build_j4_plan,
    build_j5_plan,
    build_kb_plan,
    needs_policy_lookup,
    needs_customer_history,
    is_case_status_query,
    is_greeting,
    synthesize_from_trace,
    extract_order_id,
)
from agent.pydantic_agents import generate_plan_llm
from agent.executor import Executor
from agent.metrics import get_metrics_collector
from agent.session_memory import resolve_order_id, resolve_case_id, update_session, get_recent_turns
import agent.stream_events as stream_events
from schemas.plan import ExecutionPlan, PlanStep
from schemas.trace import TraceContext


# ─── Node 1: Guardrail ───────────────────────────────────────────────────────
# First node every message hits. Runs two checks in order:
#   a) Prompt injection scan — if matched, sets action=INJECTION and returns immediately.
#   b) High-value refund check — if amount > ₹25K and refund intent, sets action=ESCALATE.
# The guardrail result is stored in state["guardrail"] for the router to act on.
def _guardrail_node(state: AgentState) -> AgentState:
    message = state["message"]
    trace_id = state["trace_id"]
    session_id = state["session_id"]
    customer_id = state.get("customer_id")

    log.info({
        "event": "guardrail_check_start",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id,
        "message_length": len(message),
    })

    # ── Prompt injection check (before anything else) ──────────────────────
    injection_pattern = check_prompt_injection(message)
    if injection_pattern:
        log.warning({
            "event": "prompt_injection_detected",
            "trace_id": trace_id,
            "session_id": session_id,
            "pattern_matched": injection_pattern,
            "message_preview": mask(message[:80]),
        })
        log_prompt_injection(trace_id, session_id, customer_id,
                             injection_pattern, message)
        state["guardrail"] = {
            "action": "INJECTION",
            "reason": f"Prompt injection pattern detected: '{injection_pattern}'",
            "extracted_amount": None,
        }
        return state

    # ── Normal guardrail check ──────────────────────────────────────────────
    result = check_guardrails(message)

    if result.action == "ESCALATE":
        log.warning({
            "event": "guardrail_triggered",
            "trace_id": trace_id,
            "session_id": session_id,
            "amount_detected": result.extracted_amount,
            "reason": result.reason,
        })
        log_guardrail_triggered(
            trace_id, session_id, customer_id,
            result.extracted_amount, 25000,
            extract_order_id(message),
        )
    else:
        log.debug({
            "event": "guardrail_passed",
            "trace_id": trace_id,
        })
        log_guardrail_passed(trace_id, result.extracted_amount, 25000)

    state["guardrail"] = {
        "action": result.action,
        "reason": result.reason,
        "extracted_amount": result.extracted_amount,
    }
    return state



# ─── Node 3: Intent LLM ──────────────────────────────────────────────────────
# Single Groq call that decides which tools to call for ANY journey.
# The prompt (system_conductor.txt) lists every available tool — get_order_status,
# cancel_order_item, execute_refund, search_kb, get_customer_interaction_history, etc.
# The LLM outputs a steps[] plan. No separate router or journey classifier needed.
# Guardrail hard-rules (injection, ₹25K escalation) bypass this node entirely.
# Fallback: if Groq fails, falls back to deterministic fast-path planners.
async def _intent_node(state: AgentState) -> AgentState:
    from agent.cache import get_data_store

    trace_id = state.get("trace_id")
    message = state["message"]
    customer_id = state.get("customer_id")
    guardrail = state["guardrail"]

    # ── Hard-rule bypass: injection ───────────────────────────────────────────
    if guardrail.get("action") == "INJECTION":
        state["journey_type"] = "J-BLOCKED"
        state["plan"] = ExecutionPlan(steps=[PlanStep(tool="blocked_injection", params={})]).model_dump()
        log.info({"event": "intent_blocked_injection", "trace_id": trace_id})
        return state

    # ── Hard-rule bypass: high-value escalation (₹25K guardrail) ─────────────
    if guardrail.get("action") == "ESCALATE":
        resolved_order_id = resolve_order_id(state["session_id"], message)
        state["resolved_order_id"] = resolved_order_id
        state["journey_type"] = "J3"
        plan = build_j3_plan(
            message, guardrail.get("reason", ""),
            guardrail.get("extracted_amount"), resolved_order_id,
        )
        state["plan"] = plan.model_dump()
        log.info({"event": "intent_escalated", "trace_id": trace_id, "source": "guardrail"})
        return state

    # ── Fast-path: greetings / chitchat (no LLM call needed) ─────────────────
    if is_greeting(message):
        state["journey_type"] = "J-GREET"
        state["plan"] = ExecutionPlan(steps=[PlanStep(tool="greeting", params={})]).model_dump()
        log.info({"event": "intent_greeting_fastpath", "trace_id": trace_id})
        return state

    # ── Resolve entities from session memory ──────────────────────────────────
    resolved_order_id = resolve_order_id(state["session_id"], message)
    resolved_case_id = resolve_case_id(state["session_id"], message)
    state["resolved_order_id"] = resolved_order_id
    state["resolved_case_id"] = resolved_case_id

    # ── Build order context (if an order ID is known) ─────────────────────────
    order_context = None
    if resolved_order_id:
        try:
            order = get_data_store().get_order(resolved_order_id)
            if order:
                cid = order.get("customer_id") or customer_id
                home_address, office_address = None, None
                if cid:
                    for label in ("home", "office"):
                        try:
                            addr = get_data_store().get_customer_address(cid, label)
                            if label == "home":
                                home_address = addr
                            else:
                                office_address = addr
                        except Exception:
                            pass
                order_context = {
                    "order_id": resolved_order_id,
                    "status": order.get("status"),
                    "payment_method": order.get("payment_method", "original"),
                    "customer_id": cid,
                    "home_address": home_address,
                    "office_address": office_address,
                    "items": [
                        {"line_id": i["line_id"], "name": i["name"],
                         "unit_price": i["unit_price"], "quantity": i.get("quantity", 1),
                         "status": i["status"]}
                        for i in order.get("items", [])
                    ],
                }
        except Exception:
            pass

    # ── Fetch recent conversation turns for Groq context ─────────────────────
    # Last 10 (user, agent) pairs from this session so Groq understands
    # references like "cancel it", "the same order", "what about the refund?"
    # without the customer repeating context every turn.
    recent_turns = get_recent_turns(state["session_id"], n=10)

    # ── Single LLM call: decide tool plan ────────────────────────────────────
    try:
        plan = await generate_plan_llm(
            message,
            order_context=order_context,
            trace_id=trace_id,
            customer_id=customer_id,
            case_id=resolved_case_id,
            recent_turns=recent_turns,
        )
        # Infer journey_type from the first tool in the plan
        state["journey_type"] = _infer_journey(plan)
        log.info({
            "event": "plan_created",
            "trace_id": trace_id,
            "journey_type": state["journey_type"],
            "tools_planned": [s.tool for s in plan.steps],
            "planner": "intent_llm",
        })
        state["plan"] = plan.model_dump()

    except Exception as exc:
        log.error({"event": "intent_llm_failed", "trace_id": trace_id,
                   "error": str(exc), "fallback": "fast_path"})
        # Fallback: use deterministic planners
        plan = _deterministic_fallback(message, customer_id, resolved_order_id, resolved_case_id)
        state["journey_type"] = _infer_journey(plan)
        state["plan"] = plan.model_dump()

    return state


def _infer_journey(plan: ExecutionPlan) -> str:
    """Infer journey type from the first tool in the plan."""
    tool_to_journey = {
        "get_order_status": "J1",
        "cancel_order_item": "J2", "cancel_full_order": "J2",
        "execute_refund": "J2", "update_shipping_address": "J2",
        "address_clarification_needed": "J2",
        "create_crm_case": "J3",
        "get_customer_interaction_history": "J4",
        "get_case_status": "J5",
        "search_kb": "J-KB",
        "clarify_order_id": "J1",
        "blocked_injection": "J-BLOCKED",
        "greeting": "J-GREET",
    }
    if plan.steps:
        return tool_to_journey.get(plan.steps[0].tool, "J2")
    return "J2"


def _deterministic_fallback(
    message: str, customer_id: Optional[str],
    resolved_order_id: Optional[str], resolved_case_id: Optional[str]
) -> ExecutionPlan:
    """Keyword-based fallback used only when Groq times out or errors."""
    if is_case_status_query(message):
        return build_j5_plan(message, customer_id, resolved_case_id)
    if needs_customer_history(message):
        return build_j4_plan(message, customer_id)
    if needs_policy_lookup(message):
        return build_kb_plan(message)
    if resolved_order_id:
        return build_j1_plan(message, customer_id, resolved_order_id)
    return ExecutionPlan(steps=[PlanStep(tool="clarify_order_id", params={})])


# ─── Node 4: Executor ────────────────────────────────────────────────────────
# Runs every step in the ExecutionPlan sequentially.
# Creates a fresh TraceContext, passes it to the Executor, then serialises
# the completed trace (tool names, inputs, outputs, latencies) into state["trace"].
def _executor_node(state: AgentState) -> AgentState:
    trace = TraceContext(
        trace_id=state["trace_id"],
        session_id=state["session_id"],
        tool_calls=[],
    )
    executor = Executor(trace)
    executor.run_plan(ExecutionPlan(**state["plan"]))
    state["trace"] = trace.model_dump()
    return state


# ─── Node 5: Synthesize ──────────────────────────────────────────────────────
# Converts the trace (raw tool outputs) into a natural-language response using
# templates in synthesize_from_trace(). No LLM call — the text is assembled from
# static strings and the actual data returned by each tool.
def _synthesize_node(state: AgentState) -> AgentState:
    trace = TraceContext(**state["trace"])
    state["response"] = synthesize_from_trace(
        state["message"], trace, state["journey_type"]
    )
    return state


def create_atlascare_graph():
    graph = StateGraph(AgentState)

    # Simplified pipeline: guardrail → intent (single LLM) → executor → synthesize
    graph.add_node("guardrail", _guardrail_node)
    graph.add_node("intent", _intent_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "intent")
    graph.add_edge("intent", "executor")
    graph.add_edge("executor", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_atlascare_graph = None


def get_atlascare_graph():
    global _atlascare_graph
    if _atlascare_graph is None:
        _atlascare_graph = create_atlascare_graph()
    return _atlascare_graph


async def run_query(
    message: str,
    session_id: str,
    customer_id: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None,
) -> dict:
    """Execute full LangGraph pipeline and return response + trace dict."""
    start = time.perf_counter()
    trace_id = f"trc-{uuid.uuid4().hex[:8]}"

    log.info({
        "event": "request_received",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id,
        "message_length": len(message),
        "message_preview": mask(message[:80]),
    })
    log_request_received(trace_id, session_id, customer_id, message, len(message))

    initial_state: AgentState = {
        "message": message,
        "session_id": session_id,
        "trace_id": trace_id,
        "customer_id": customer_id,
        "guardrail": {},
        "journey_type": "",
        "plan": {},
        "trace": {},
        "response": "",
        "use_llm_plan": False,
        "resolved_order_id": None,
        "resolved_case_id": None,
        "resolved_line_id": None,
    }

    # Register the SSE queue so nodes can push live events during execution.
    # Non-streaming callers pass None — all emit calls become silent no-ops.
    if event_queue is not None:
        stream_events.register(trace_id, event_queue)

    graph = get_atlascare_graph()
    try:
        final_state = await graph.ainvoke(initial_state)
    finally:
        # Always deregister, even on error, so the queue is never left dangling.
        if event_queue is not None:
            stream_events.deregister(trace_id)

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace_data = final_state["trace"]
    trace_data["latency_ms"] = latency_ms

    journey_type = final_state["journey_type"]
    num_tools = len(trace_data.get("tool_calls", []))
    # Intent node always makes 1 LLM call unless bypassed by hard-rules (injection/escalation)
    num_llm = 0 if final_state.get("journey_type") in ("J-BLOCKED", "J3", "J-GREET") else 1

    metrics = get_metrics_collector()
    llm_slice = metrics.llm_metrics[-num_llm:] if num_llm else []
    total_tokens = sum(m.total_tokens for m in llm_slice)
    total_cost = sum(m.cost_usd for m in llm_slice)
    tool_calls = trace_data.get("tool_calls", [])
    success = all(tc.get("success", False) for tc in tool_calls) if tool_calls else True

    metrics.record_journey(
        journey_type=journey_type,
        trace_id=trace_id,
        session_id=session_id,
        total_latency_ms=latency_ms,
        num_tool_calls=num_tools,
        num_llm_calls=num_llm,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        success=success,
    )

    log.info({
        "event": "request_completed",
        "trace_id": trace_id,
        "session_id": session_id,
        "journey_type": journey_type,
        "latency_ms": latency_ms,
        "num_tool_calls": num_tools,
        "num_llm_calls": num_llm,
        "total_tokens": total_tokens,
        "success": success,
    })
    log_request_completed(trace_id, session_id, journey_type, latency_ms, success, num_tools)

    # --- Update session memory with this turn's entities ---
    update_session(
        session_id=session_id,
        user_message=message,
        agent_response=final_state["response"],
        order_id=final_state.get("resolved_order_id"),
        case_id=final_state.get("resolved_case_id"),
        customer_id=customer_id,
    )

    # --- Interaction logging ---
    _log_interaction(
        message=message,
        session_id=session_id,
        trace_id=trace_id,
        journey_type=journey_type,
        response=final_state["response"],
        trace_data=trace_data,
    )

    return {
        "response": final_state["response"],
        "trace": trace_data,
        "journey_type": journey_type,
    }


def _log_interaction(
    message: str,
    session_id: str,
    trace_id: str,
    journey_type: str,
    response: str,
    trace_data: dict,
) -> None:
    """Persist every customer interaction to crm_interaction_history.json + ChromaDB."""
    try:
        from agent.cache import get_data_store

        order_id = extract_order_id(message)
        customer_id = extract_customer_id(message) or "UNKNOWN"

        # Derive resolution from journey type
        resolution_map = {
            "J1": "resolved",
            "J2": "resolved",
            "J3": "escalated",
            "J4": "resolved",
            "J5": "resolved",
            "J-KB": "resolved",
        }
        resolution = resolution_map.get(journey_type, "resolved")

        # Derive tags from journey + tool calls
        tags = _tags_for_journey(journey_type, message, trace_data)

        # Build a concise summary from the response (first 200 chars)
        summary = response.replace("**", "").replace("\n", " ").strip()[:200]

        interaction = {
            "interaction_id": f"INT-{trace_id.replace('trc-', '').upper()}",
            "customer_id": customer_id,
            "order_id": order_id,
            "channel": "chat",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "agent_notes": f"Handled via AtlasCare. Journey: {journey_type}. Session: {session_id}.",
            "resolution": resolution,
            "tags": tags,
        }

        get_data_store().log_interaction(interaction)
    except Exception:
        pass  # Logging must never break the API response


def _tags_for_journey(journey_type: str, message: str, trace_data: dict) -> list:
    msg = message.lower()
    tags = []
    if journey_type == "J1":
        tags = ["tracking", "order_status"]
    elif journey_type == "J3":
        tags = ["escalation", "high_value_refund"]
        if "damaged" in msg or "damage" in msg:
            tags.append("damaged_product")
    elif journey_type == "J2":
        if "cancel" in msg:
            tags.append("cancellation")
        if "refund" in msg:
            tags.append("refund_inquiry")
        if "address" in msg or "ship" in msg:
            tags.append("address_change")
    elif journey_type == "J4":
        tags = ["repeat_contact", "history_lookup"]
    elif journey_type == "J5":
        tags = ["case_status_inquiry"]
    elif journey_type == "J-KB":
        tags = ["policy_inquiry"]
    # Product-category tags
    for kw, tag in [
        ("laptop", "electronics"), ("phone", "electronics"), ("tablet", "electronics"),
        ("monitor", "electronics"), ("headphone", "electronics"), ("watch", "electronics"),
        ("shirt", "apparel"), ("jeans", "apparel"), ("kurti", "apparel"),
        ("furniture", "home_goods"), ("appliance", "home_goods"),
    ]:
        if kw in msg and tag not in tags:
            tags.append(tag)
    return tags
