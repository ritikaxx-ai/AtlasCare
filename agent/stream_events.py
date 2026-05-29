"""
agent/stream_events.py — lightweight event bus for SSE streaming.

Any node or tool can push a progress event (thinking, tool_start, etc.)
without knowing anything about HTTP or FastAPI.

Usage (sync context, e.g. executor):
    emit_sync(trace_id, {"type": "tool_start", "tool": "get_order_status", "content": "..."})

Usage (async context, e.g. pydantic_agents):
    await emit_async(trace_id, {"type": "thinking", "content": "..."})

The streaming endpoint in main.py registers a queue before the pipeline runs
and deregisters it after — so events only flow for active streaming requests.
Non-streaming requests (POST /query) never register a queue, so all emit
calls are no-ops and nothing breaks.
"""

import asyncio
from typing import Optional

# trace_id → asyncio.Queue
_queues: dict = {}


def register(trace_id: str, queue: asyncio.Queue) -> None:
    """Register a queue for a streaming request. Called by main.py before graph.ainvoke."""
    _queues[trace_id] = queue


def deregister(trace_id: str) -> None:
    """Remove the queue when the request is done or errored."""
    _queues.pop(trace_id, None)


def emit_sync(trace_id: str, event: dict) -> None:
    """
    Push an event from a synchronous context (executor, graph nodes).
    Uses put_nowait so it never blocks — safe to call from sync LangGraph nodes.
    Silently does nothing if no queue is registered (non-streaming request).
    """
    q = _queues.get(trace_id)
    if q is not None:
        q.put_nowait(event)


async def emit_async(trace_id: str, event: dict) -> None:
    """
    Push an event from an async context (pydantic_agents, async graph nodes).
    Silently does nothing if no queue is registered.
    """
    q = _queues.get(trace_id)
    if q is not None:
        await q.put(event)


# Sentinel pushed by run_query() to signal the pipeline is done.
PIPELINE_DONE = "__pipeline_done__"


# Human-friendly labels for each tool — shown in the chat UI while tool runs.
TOOL_LABELS: dict = {
    "get_order_status":                  "Looking up your order...",
    "cancel_order_item":                 "Cancelling the item...",
    "cancel_full_order":                 "Cancelling your order...",
    "execute_refund":                    "Processing your refund...",
    "update_shipping_address":           "Updating delivery address...",
    "address_clarification_needed":      "Checking saved addresses...",
    "create_crm_case":                   "Raising a support case...",
    "get_case_status":                   "Looking up your case...",
    "get_customer_interaction_history":  "Searching your support history...",
    "get_customer_profile":              "Loading your profile...",
    "get_customer_address":              "Fetching your saved address...",
    "search_kb":                         "Searching our policy database...",
    "clarify_order_id":                  "Preparing a clarification...",
    "clarify_customer_id":               "Preparing a clarification...",
    "unauthorized_order_access":         "Verifying account access...",
    "blocked_injection":                 "Processing your message...",
    "out_of_scope":                      "Checking what I can help with...",
}
