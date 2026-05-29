"""
tools/oms.py — Order Management System tools.

Each class is a TracedTool subclass. The Executor instantiates them by name and
calls them with the params from the ExecutionPlan. Every call is automatically
timed and recorded in the TraceContext by TracedTool.__call__.

Sentinel tools (unauthorized_order_access, clarify_order_id, blocked_injection, etc.)
have no real side effects — they exist only so synthesize_from_trace can match their
tool_name and render the appropriate canned message.
"""
from typing import Any, Dict
from .base import TracedTool
from agent.cache import get_data_store
from agent.logger import log
from agent.audit import log_cancellation, log_address_updated


class get_order_status(TracedTool):
    """Get order status from in-memory cache (< 1ms lookup)"""

    def _execute(self, order_id: str, **kwargs) -> Dict[str, Any]:
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")
        log.debug({"event": "tool_call_start", "tool": "get_order_status",
                   "trace_id": trace_id, "order_id": order_id})
        data_store = get_data_store()
        order = data_store.get_order(order_id)
        if not order:
            log.warning({"event": "order_not_found", "trace_id": trace_id, "order_id": order_id})
            return {"not_found": True, "order_id": order_id}
        log.debug({"event": "tool_call_end", "tool": "get_order_status",
                   "trace_id": trace_id, "order_id": order_id, "status": order.get("status")})
        return order


class cancel_order_item(TracedTool):
    """Cancel order item using in-memory cache with thread-safe updates."""

    def _execute(self, order_id: str, line_id: int, **kwargs) -> Dict[str, Any]:
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")
        log.info({"event": "tool_call_start", "tool": "cancel_order_item",
                  "trace_id": trace_id, "order_id": order_id, "line_id": line_id})
        data_store = get_data_store()
        result = data_store.cancel_order_item(order_id, line_id)
        if result.get("success"):
            log.info({"event": "item_cancelled", "trace_id": trace_id,
                      "order_id": order_id, "line_id": line_id})
            log_cancellation(trace_id, getattr(self.trace_ctx, "session_id", ""),
                             None, order_id, line_id, "partial")
        return result


class cancel_full_order(TracedTool):
    """Cancel all active items in an order."""

    def _execute(self, order_id: str, **kwargs) -> Dict[str, Any]:
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")
        log.info({"event": "tool_call_start", "tool": "cancel_full_order",
                  "trace_id": trace_id, "order_id": order_id})
        data_store = get_data_store()
        result = data_store.cancel_full_order(order_id)
        if result.get("success"):
            log.info({"event": "order_cancelled", "trace_id": trace_id, "order_id": order_id})
            log_cancellation(trace_id, getattr(self.trace_ctx, "session_id", ""),
                             None, order_id, None, "full")
        return result


class unauthorized_order_access(TracedTool):
    """No-op sentinel — signals an ownership violation."""

    def _execute(self, order_id: str, **kwargs) -> Dict[str, Any]:
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")
        log.warning({"event": "unauthorized_order_access", "trace_id": trace_id,
                     "order_id": order_id})
        return {"unauthorized": True, "order_id": order_id}


class clarify_order_id(TracedTool):
    """No-op tool — signals that the customer's message is missing an order ID."""

    def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"needs_clarification": True}


class clarify_customer_id(TracedTool):
    """No-op tool — signals that a customer ID is needed to look up history."""

    def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"needs_customer_id": True}


class blocked_injection(TracedTool):
    """No-op sentinel returned when prompt injection is detected."""

    def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"injection_blocked": True}


class greeting(TracedTool):
    """No-op sentinel returned for greetings / chitchat — zero LLM calls."""

    def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"is_greeting": True}


class out_of_scope(TracedTool):
    """Sentinel returned when the customer's request doesn't match any supported journey."""

    def _execute(self, original_tool: str = "", **kwargs) -> Dict[str, Any]:
        return {"out_of_scope": True, "original_tool": original_tool}


class address_clarification_needed(TracedTool):
    """
    Sentinel returned when the customer asks to update their shipping address
    but does not specify which saved address to use (home/office/etc.)
    or provides a label that doesn't match any saved address.
    The synthesizer converts this into a clarifying question.
    """

    def _execute(self, order_id: str, customer_id: str,
                 available_labels: list, **kwargs) -> Dict[str, Any]:
        return {
            "address_clarification_needed": True,
            "order_id": order_id,
            "customer_id": customer_id,
            "available_labels": available_labels,
        }


class update_shipping_address(TracedTool):
    """Update shipping address using a saved address label OR a full address dict.

    The LLM planner passes address_label ("home" / "office") and customer_id.
    The tool resolves the label → dict from the data store so Gemini never has
    to construct a nested object (which it does unreliably).
    """

    def _execute(self, order_id: str, customer_id: str = None,
                 address_label: str = None, address: Dict[str, Any] = None,
                 free_text_address: str = None, save_as_label: str = None,
                 **kwargs) -> Dict[str, Any]:
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")
        data_store = get_data_store()

        # Resolve address: prefer label lookup, then raw dict, then free-text string
        if address_label and customer_id:
            try:
                address = data_store.get_customer_address(customer_id, address_label)
            except Exception:
                pass  # Fall through to free_text_address below

        # If label lookup failed or no label given, try free-text address
        if (not address or not isinstance(address, dict)) and free_text_address:
            address = {"line1": free_text_address, "line2": "", "city": "", "state": "", "pincode": ""}

        if not address or not isinstance(address, dict):
            return {"error": "No valid address provided. Please specify a saved address label or full address."}

        # If customer said "my home address is X" or "my office address is X",
        # save it to their profile so future orders can use it
        label_to_save = save_as_label or address_label
        if label_to_save and customer_id and free_text_address:
            data_store.save_customer_address(customer_id, label_to_save, address)

        log.info({"event": "tool_call_start", "tool": "update_shipping_address",
                  "trace_id": trace_id, "order_id": order_id,
                  "city": address.get("city"), "pincode": address.get("pincode")})
        result = data_store.update_shipping_address(order_id, address)
        if result.get("success"):
            log.info({"event": "address_updated", "trace_id": trace_id, "order_id": order_id})
            log_address_updated(trace_id, getattr(self.trace_ctx, "session_id", ""),
                                None, order_id, address)
        return result
