from typing import Dict, Any, Type
from schemas.plan import ExecutionPlan
from schemas.trace import TraceContext
from tools.base import TracedTool
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
    def __init__(self, trace_ctx: TraceContext):
        self.trace_ctx = trace_ctx
        
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
        }

    def run_plan(self, plan: ExecutionPlan) -> None:
        """
        Executes the plan sequentially.
        If any tool fails, execution stops immediately (fail-fast).
        All results and errors are captured in the trace context.
        """
        for step in plan.steps:
            tool_class = self.tools.get(step.tool)
            if not tool_class:
                # Unknown tool - fail fast
                # We could log an error in the trace manually here, but for simplicity:
                raise ValueError(f"Unknown tool requested in plan: {step.tool}")
                
            tool_instance = tool_class(self.trace_ctx)
            
            try:
                # Call tool which automatically records telemetry in trace_ctx
                tool_instance(**step.params)
            except Exception as e:
                # Stop execution on first failure
                print(f"Plan execution halted due to failure in {step.tool}: {e}")
                break
