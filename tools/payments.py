import random
import uuid
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import TracedTool
from agent.cache import get_data_store
from agent.logger import log
from agent.audit import log_refund_initiated, log_refund_blocked


class PaymentGatewayError(Exception):
    """Transient payment gateway failure — eligible for retry."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(PaymentGatewayError),
    reraise=True,
)
def _call_payment_gateway(config: dict, order_id: str, amount_inr: float,
                          method: str, trace_id: str) -> Dict[str, Any]:
    """Isolated gateway call — retried up to 3× with exponential backoff."""
    failure_rate = config["behaviour"]["failure_rate"]
    if random.random() < failure_rate:
        log.warning({"event": "payment_gateway_failure", "trace_id": trace_id,
                     "order_id": order_id, "will_retry": True})
        raise PaymentGatewayError(config["behaviour"]["failure_message"])
    return {
        "success": True,
        "refund_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
        "amount_refunded": amount_inr,
        "sla_days": config["refund_sla_days"],
    }


class execute_refund(TracedTool):
    """Execute refund — Layer 3 hard safety cap + retry + structured logging."""

    def _execute(self, order_id: str, amount_inr: float, method: str, **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        config = data_store.get_payment_config()
        limit = config["auto_refund_limit_inr"]
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")

        log.info({
            "event": "tool_call_start",
            "tool": "execute_refund",
            "trace_id": trace_id,
            "order_id": order_id,
            "amount_inr": amount_inr,
            "method": method,
        })

        # ── Layer 3: hard cap (last line of defence — independent of LLM/guardrail) ──
        if amount_inr > limit:
            log.error({
                "event": "refund_blocked_tool_layer",
                "trace_id": trace_id,
                "order_id": order_id,
                "amount_inr": amount_inr,
                "limit": limit,
            })
            log_refund_blocked(trace_id, order_id, amount_inr, limit, "tool_execute_refund")
            raise ValueError(
                f"Amount ₹{amount_inr:,.0f} exceeds auto-refund limit of ₹{limit:,.0f}. "
                "Escalate via create_crm_case."
            )

        if method not in config["supported_methods"]:
            raise ValueError(f"Payment method '{method}' is not supported")

        result = _call_payment_gateway(config, order_id, amount_inr, method, trace_id)
        refund_id = result["refund_id"]

        log.info({
            "event": "refund_succeeded",
            "trace_id": trace_id,
            "order_id": order_id,
            "refund_id": refund_id,
            "amount_inr": amount_inr,
            "method": method,
        })
        log_refund_initiated(
            trace_id, getattr(self.trace_ctx, "session_id", ""),
            None, order_id, amount_inr, method, refund_id,
        )
        return result
