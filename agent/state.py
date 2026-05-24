from typing import Any, Dict, List, Optional, TypedDict


class GuardrailState(TypedDict, total=False):
    action: str
    reason: str
    extracted_amount: Optional[float]


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    trace_id: str
    customer_id: Optional[str]
    guardrail: GuardrailState
    journey_type: str
    plan: Dict[str, Any]
    trace: Dict[str, Any]
    response: str
    use_llm_plan: bool
    # Resolved from session memory — order/case ID carried forward from prior turns
    resolved_order_id: Optional[str]
    resolved_case_id: Optional[str]
