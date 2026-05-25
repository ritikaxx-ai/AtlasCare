from typing import Any, Dict, List, Optional, TypedDict


# GuardrailState holds the result of the pre-LLM safety check.
# action is one of: "PASS" (safe to proceed), "ESCALATE" (high-value refund → create CRM case),
# or "INJECTION" (prompt injection detected → block entirely).
class GuardrailState(TypedDict, total=False):
    action: str           # PASS | ESCALATE | INJECTION
    reason: str           # human-readable explanation of why guardrail fired
    extracted_amount: Optional[float]  # rupee amount parsed from the message, if any


# AgentState is the shared dict that every LangGraph node reads from and writes to.
# It flows through: guardrail → router → fast_plan/llm_plan → executor → synthesize.
class AgentState(TypedDict, total=False):
    message: str           # the raw customer message
    session_id: str        # browser/chat session identifier
    trace_id: str          # unique ID for this request (used in logs and audit trail)
    customer_id: Optional[str]   # logged-in customer (e.g. "CUST-001"); None if unauthenticated
    guardrail: GuardrailState    # filled by _guardrail_node
    journey_type: str      # J1/J2/J3/J4/J5/J-KB — set by router, used by planner and synthesizer
    plan: Dict[str, Any]   # serialised ExecutionPlan (list of tool steps to run)
    trace: Dict[str, Any]  # serialised TraceContext (every tool call + its output)
    response: str          # final text sent back to the customer
    use_llm_plan: bool     # True → route to Pydantic AI planner; False → deterministic fast path
    # These are resolved in the router by looking at session memory so a customer can say
    # "cancel it" without repeating the order ID they mentioned two turns ago.
    resolved_order_id: Optional[str]
    resolved_case_id: Optional[str]
