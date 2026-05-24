"""
Immutable audit log for AtlasCare.

Every financially significant or security-relevant event is written to
logs/audit.jsonl — one JSON object per line, append-only.

This is what Risk & Compliance reads. It is separate from the application
log and must never be silently swallowed.

AUDIT_EVENTS:
  request_received          — every inbound /query
  guardrail_triggered       — refund amount exceeded threshold (J3)
  guardrail_passed          — amount below threshold, logged for completeness
  prompt_injection_detected — malicious input pattern found
  refund_initiated          — execute_refund succeeded
  refund_blocked_by_limit   — tool-level hard cap fired
  escalation_case_created   — create_crm_case called
  partial_cancellation      — cancel_order_item succeeded
  full_cancellation         — cancel_full_order succeeded
  address_updated           — update_shipping_address succeeded
  request_completed         — pipeline finished, total latency recorded
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_AUDIT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_AUDIT_DIR, exist_ok=True)
_AUDIT_FILE = os.path.join(_AUDIT_DIR, "audit.jsonl")

_lock = threading.Lock()

# In-memory index: trace_id → list[event] for GET /audit/{trace_id}
_index: Dict[str, list] = {}


def _write(record: dict) -> None:
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with _lock:
        # append to JSONL file
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        # update in-memory index
        tid = record.get("trace_id")
        if tid:
            _index.setdefault(tid, []).append(record)


def get_by_trace(trace_id: str) -> Optional[list]:
    """Return all audit events for a trace_id, or None if not found."""
    with _lock:
        return list(_index.get(trace_id, []))


def log_request_received(
    trace_id: str, session_id: str, customer_id: Optional[str],
    message: str, message_length: int
) -> None:
    _write({
        "event": "request_received",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "message_length": message_length,
        "message_preview": message[:80],
    })


def log_guardrail_triggered(
    trace_id: str, session_id: str, customer_id: Optional[str],
    amount_detected: float, threshold: float, order_id: Optional[str]
) -> None:
    _write({
        "event": "guardrail_triggered",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "amount_requested": amount_detected,
        "threshold": threshold,
        "order_id": order_id,
        "reason": "refund_amount_exceeds_threshold",
        "kb_article_referenced": "KB-001",
        "guardrail_layer": "langgraph_conditional_edge",
        "decision_basis": f"Amount ₹{amount_detected:,.0f} > auto-refund limit ₹{threshold:,.0f}",
    })


def log_guardrail_passed(
    trace_id: str, amount_detected: Optional[float], threshold: float
) -> None:
    _write({
        "event": "guardrail_passed",
        "trace_id": trace_id,
        "amount_detected": amount_detected,
        "threshold": threshold,
    })


def log_prompt_injection(
    trace_id: str, session_id: str, customer_id: Optional[str],
    pattern_matched: str, message_preview: str
) -> None:
    _write({
        "event": "prompt_injection_detected",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "pattern_matched": pattern_matched,
        "message_preview": message_preview[:80],
        "action_taken": "request_blocked",
    })


def log_refund_initiated(
    trace_id: str, session_id: str, customer_id: Optional[str],
    order_id: str, amount_inr: float, method: str, refund_id: str
) -> None:
    _write({
        "event": "refund_initiated",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "order_id": order_id,
        "amount_inr": amount_inr,
        "payment_method": method,
        "refund_id": refund_id,
        "outcome": "success",
    })


def log_refund_blocked(
    trace_id: str, order_id: str, amount_inr: float, limit: float, layer: str
) -> None:
    _write({
        "event": "refund_blocked_by_limit",
        "trace_id": trace_id,
        "order_id": order_id,
        "amount_requested": amount_inr,
        "limit": limit,
        "guardrail_layer": layer,
        "reason": "amount_exceeds_auto_refund_limit",
    })


def log_escalation_created(
    trace_id: str, session_id: str, customer_id: Optional[str],
    order_id: str, case_id: str, amount_inr: Optional[float], reason: str
) -> None:
    _write({
        "event": "escalation_case_created",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "order_id": order_id,
        "case_id": case_id,
        "amount_inr": amount_inr,
        "reason": reason,
        "kb_article_referenced": "KB-001",
        "sla_hours": 24,
    })


def log_cancellation(
    trace_id: str, session_id: str, customer_id: Optional[str],
    order_id: str, line_id: Optional[int], kind: str  # "partial" | "full"
) -> None:
    event = "partial_cancellation" if kind == "partial" else "full_cancellation"
    _write({
        "event": event,
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "order_id": order_id,
        "line_id": line_id,
    })


def log_address_updated(
    trace_id: str, session_id: str, customer_id: Optional[str],
    order_id: str, new_address: dict
) -> None:
    _write({
        "event": "address_updated",
        "trace_id": trace_id,
        "session_id": session_id,
        "customer_id": customer_id or "anonymous",
        "order_id": order_id,
        "new_city": new_address.get("city"),
        "new_pincode": new_address.get("pincode"),
    })


def log_request_completed(
    trace_id: str, session_id: str, journey_type: str,
    latency_ms: int, success: bool, num_tools: int
) -> None:
    _write({
        "event": "request_completed",
        "trace_id": trace_id,
        "session_id": session_id,
        "journey_type": journey_type,
        "latency_ms": latency_ms,
        "success": success,
        "num_tool_calls": num_tools,
    })
