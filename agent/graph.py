"""
LangGraph orchestration for AtlasCare v3.0.

Flow:
  guardrail → router → [fast_plan | llm_plan] → executor → synthesize → END

J1/J3: deterministic plan + template synthesis (0 LLM calls, sub-second tools)
J2:    Pydantic AI planning + template synthesis (1 LLM call)
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.guardrail import check_guardrails, check_prompt_injection
from agent.logger import log, mask
from agent.audit import (
    log_request_received, log_guardrail_triggered, log_guardrail_passed,
    log_prompt_injection, log_request_completed,
)
from agent.fast_paths import (
    classify_journey_for_routing,
    build_j1_plan,
    build_j3_plan,
    build_j4_plan,
    build_j5_plan,
    build_kb_plan,
    try_build_j2_plan,
    synthesize_from_trace,
    extract_order_id,
    extract_case_id,
    extract_customer_id,
    _ownership_plan,
)
from agent.pydantic_agents import generate_plan_llm
from agent.executor import Executor
from agent.metrics import get_metrics_collector
from agent.session_memory import resolve_order_id, resolve_case_id, update_session, get_recent_turns
from schemas.plan import ExecutionPlan, PlanStep
from schemas.trace import TraceContext


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


def _router_node(state: AgentState) -> AgentState:
    action = state["guardrail"]["action"]
    session_id = state["session_id"]
    message = state["message"]
    trace_id = state["trace_id"]

    # Injection was caught in guardrail — short-circuit routing
    if action == "INJECTION":
        state["journey_type"] = "J-BLOCKED"
        state["use_llm_plan"] = False
        log.info({"event": "router_blocked_injection", "trace_id": trace_id})
        return state

    # --- Resolve entities from session memory ---
    state["resolved_order_id"] = resolve_order_id(session_id, message)
    state["resolved_case_id"] = resolve_case_id(session_id, message)

    state["journey_type"] = classify_journey_for_routing(
        message, action, resolved_order_id=state.get("resolved_order_id")
    )

    log.info({
        "event": "journey_classified",
        "trace_id": trace_id,
        "session_id": session_id,
        "journey_type": state["journey_type"],
        "resolved_order_id": state.get("resolved_order_id"),
        "resolved_case_id": state.get("resolved_case_id"),
    })

    state["use_llm_plan"] = False
    return state


def _route_after_router(state: AgentState) -> Literal["fast_plan", "llm_plan"]:
    return "llm_plan" if state.get("use_llm_plan") else "fast_plan"


def _fast_plan_node(state: AgentState) -> AgentState:
    journey = state["journey_type"]
    resolved_order_id = state.get("resolved_order_id")
    resolved_case_id = state.get("resolved_case_id")
    trace_id = state["trace_id"]

    # Injection blocked — return safe canned response via sentinel tool
    if journey == "J-BLOCKED":
        plan = ExecutionPlan(steps=[PlanStep(tool="blocked_injection", params={})])
        state["plan"] = plan.model_dump()
        return state

    if journey == "J1":
        plan = build_j1_plan(state["message"], state.get("customer_id"), resolved_order_id)
    elif journey == "J2":
        if not resolved_order_id:
            plan = ExecutionPlan(steps=[PlanStep(tool="clarify_order_id", params={})])
        else:
            plan = try_build_j2_plan(state["message"], state.get("customer_id"), resolved_order_id)
            if plan is None:
                plan = ExecutionPlan(steps=[PlanStep(tool="clarify_order_id", params={})])
    elif journey == "J4":
        plan = build_j4_plan(state["message"], state.get("customer_id"))
    elif journey == "J5":
        plan = build_j5_plan(state["message"], state.get("customer_id"), resolved_case_id)
    elif journey == "J-KB":
        plan = build_kb_plan(state["message"])
    else:
        g = state["guardrail"]
        plan = build_j3_plan(
            state["message"],
            g.get("reason", ""),
            g.get("extracted_amount"),
            resolved_order_id,
        )

    tools_planned = [s["tool"] for s in plan.model_dump().get("steps", [])]
    log.info({
        "event": "plan_created",
        "trace_id": trace_id,
        "journey_type": journey,
        "tools_planned": tools_planned,
        "planner": "fast_path",
    })
    state["plan"] = plan.model_dump()
    return state


async def _llm_plan_node(state: AgentState) -> AgentState:
    from agent.cache import get_data_store
    from agent.vector_store import search_customer_history

    order_id = state.get("resolved_order_id")
    order_context = None

    if order_id:
        order = get_data_store().get_order(order_id)
        if order:
            # Fetch office address for the customer (best-effort)
            office_address = None
            try:
                cid = order.get("customer_id") or state.get("customer_id")
                if cid:
                    office_address = get_data_store().get_customer_address(cid, "office")
            except Exception:
                pass

            order_context = {
                "order_id": order_id,
                "status": order.get("status"),
                "payment_method": order.get("payment_method", "original"),
                "customer_id": order.get("customer_id"),
                "office_address": office_address,
                "items": [
                    {
                        "line_id": i["line_id"],
                        "name": i["name"],
                        "unit_price": i["unit_price"],
                        "quantity": i.get("quantity", 1),
                        "status": i["status"],
                    }
                    for i in order.get("items", [])
                ],
            }

    plan = await generate_plan_llm(state["message"], order_context=order_context)
    state["plan"] = plan.model_dump()
    return state


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


def _synthesize_node(state: AgentState) -> AgentState:
    trace = TraceContext(**state["trace"])
    state["response"] = synthesize_from_trace(
        state["message"], trace, state["journey_type"]
    )
    return state


def create_atlascare_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", _guardrail_node)
    graph.add_node("router", _router_node)
    graph.add_node("fast_plan", _fast_plan_node)
    graph.add_node("llm_plan", _llm_plan_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"fast_plan": "fast_plan", "llm_plan": "llm_plan"},
    )
    graph.add_edge("fast_plan", "executor")
    graph.add_edge("llm_plan", "executor")
    graph.add_edge("executor", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_atlascare_graph = None


def get_atlascare_graph():
    global _atlascare_graph
    if _atlascare_graph is None:
        _atlascare_graph = create_atlascare_graph()
    return _atlascare_graph


async def run_query(message: str, session_id: str, customer_id: Optional[str] = None) -> dict:
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
    }

    graph = get_atlascare_graph()
    final_state = await graph.ainvoke(initial_state)

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace_data = final_state["trace"]
    trace_data["latency_ms"] = latency_ms

    journey_type = final_state["journey_type"]
    num_tools = len(trace_data.get("tool_calls", []))
    num_llm = 1 if final_state.get("use_llm_plan") else 0

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
