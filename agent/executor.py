"""
agent/executor.py — runs an ExecutionPlan step by step.

The Executor holds a registry mapping tool names (strings) to TracedTool subclasses.
run_plan() iterates the plan's steps, instantiates each tool with the shared TraceContext,
and calls it. Because TracedTool.__call__ writes its result into the TraceContext, the
trace is fully populated by the time run_plan() returns — the synthesizer reads it next.

Fail-fast: if any tool raises, execution stops immediately (no partial-success recovery).
"""
from typing import Dict, Any, Type, Optional
from schemas.plan import ExecutionPlan
from schemas.trace import TraceContext
from tools.base import TracedTool
from agent.cache import get_data_store
from agent.stream_events import emit_sync, TOOL_LABELS
from tools.oms import (
    get_order_status,
    cancel_order_item,
    cancel_full_order,
    unauthorized_order_access,
    update_shipping_address,
    clarify_order_id,
    clarify_customer_id,
    blocked_injection,
    address_clarification_needed,
    greeting,
    out_of_scope,
)
from tools.crm import (
    get_customer_profile,
    get_customer_address,
    create_crm_case,
    get_case_status,
    get_customer_interaction_history,
)
from tools.kb import search_kb
from tools.payments import execute_refund

class Executor:
    # Tools that operate on a specific order_id and must pass ownership check
    _ORDER_TOOLS = {
        "get_order_status", "cancel_order_item", "cancel_full_order",
        "update_shipping_address", "execute_refund",
    }

    def __init__(self, trace_ctx: TraceContext, customer_id: Optional[str] = None):
        self.trace_ctx = trace_ctx
        self.customer_id = customer_id

        # Tool registry
        self.tools: Dict[str, Type[TracedTool]] = {
            "get_order_status": get_order_status,
            "cancel_order_item": cancel_order_item,
            "cancel_full_order": cancel_full_order,
            "unauthorized_order_access": unauthorized_order_access,
            "update_shipping_address": update_shipping_address,
            "clarify_order_id": clarify_order_id,
            "clarify_customer_id": clarify_customer_id,
            "blocked_injection": blocked_injection,
            "get_customer_profile": get_customer_profile,
            "get_customer_address": get_customer_address,
            "create_crm_case": create_crm_case,
            "get_case_status": get_case_status,
            "get_customer_interaction_history": get_customer_interaction_history,
            "search_kb": search_kb,
            "execute_refund": execute_refund,
            "address_clarification_needed": address_clarification_needed,
            "greeting": greeting,
            "out_of_scope": out_of_scope,
        }

    # Tools whose soft-failure output must block downstream refund/address steps.
    # If these return success=False (or a known failure key), execution stops.
    _GATE_TOOLS = {"cancel_order_item", "cancel_full_order"}

    # Keys in a tool output that indicate the action did not actually complete.
    _FAILURE_KEYS = {"not_found", "unauthorized", "already_cancelled", "error"}

    def run_plan(self, plan: ExecutionPlan) -> None:
        """
        Executes the plan sequentially with output-based state validation.

        Two stop conditions per step:
          1. Exception raised → break immediately (hard failure).
          2. Gate tool (cancel_*) returns a soft-failure output → break before
             downstream refund/address steps fire, preventing phantom refunds.
        """
        trace_id = getattr(self.trace_ctx, "trace_id", "")

        for step in plan.steps:
            tool_class = self.tools.get(step.tool)
            if not tool_class:
                # LLM returned a tool we don't support — treat as out-of-scope
                oos = self.tools["out_of_scope"](self.trace_ctx)
                oos(original_tool=step.tool)
                break

            # ── Ownership check: block cross-customer order access ────────────
            if self.customer_id and step.tool in self._ORDER_TOOLS:
                order_id = step.params.get("order_id")
                if order_id:
                    order = get_data_store().get_order(order_id)
                    if order and order.get("customer_id") != self.customer_id:
                        unauth_tool = self.tools["unauthorized_order_access"](self.trace_ctx)
                        unauth_tool(order_id=order_id)
                        break  # Stop — don't run any further steps

            label = TOOL_LABELS.get(step.tool, f"Running {step.tool}...")
            emit_sync(trace_id, {"type": "tool_start", "tool": step.tool, "content": label})

            tool_instance = tool_class(self.trace_ctx)
            try:
                result = tool_instance(**step.params)
            except Exception as e:
                print(f"Plan execution halted due to failure in {step.tool}: {e}")
                break

            # Gate check: if a cancellation step did not actually succeed,
            # stop here — never run execute_refund on an un-cancelled item.
            if step.tool in self._GATE_TOOLS:
                output = result or {}
                cancelled_count = output.get("cancelled_count", 1)
                has_failure = (
                    any(output.get(k) for k in self._FAILURE_KEYS)
                    or output.get("success") is False
                    or cancelled_count == 0
                )
                if has_failure:
                    print(f"Plan execution halted: {step.tool} did not complete successfully — "
                          f"downstream steps blocked to prevent inconsistent state.")
                    break
