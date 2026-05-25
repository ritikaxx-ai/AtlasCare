import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from .base import TracedTool
from agent.cache import get_data_store
from agent.logger import log
from agent.audit import log_escalation_created


class get_customer_profile(TracedTool):
    """Get customer profile from in-memory cache"""
    
    def _execute(self, customer_id: str, **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        customer = data_store.get_customer(customer_id)
        
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        
        return customer


class get_customer_address(TracedTool):
    """Get customer address by label from in-memory cache"""
    
    def _execute(self, customer_id: str, label: str, **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        return data_store.get_customer_address(customer_id, label)


class create_crm_case(TracedTool):
    """Create CRM escalation case with audit trail"""
    
    def _execute(self, customer_id: str, order_id: str, description: str,
                 priority: str = "high", amount_inr: float = None, **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        trace_id = getattr(self.trace_ctx, "trace_id", "unknown")

        # ── Deduplication guard ──────────────────────────────────────────────
        # If an open case already exists for this customer + order, return it
        # instead of creating a duplicate.  Prevents double-cases when the user
        # submits the same request twice or a retry races with itself.
        for existing in data_store._crm.get("cases", []):
            if (existing.get("order_id") == order_id
                    and existing.get("customer_id") == customer_id
                    and existing.get("status") == "open"):
                log.info({
                    "event": "crm_case_deduplicated",
                    "trace_id": trace_id,
                    "existing_case_id": existing["case_id"],
                    "order_id": order_id,
                })
                return {**existing, "deduplicated": True}

        case_id = f"CASE-{uuid.uuid4().hex[:6].upper()}"

        new_case = {
            "case_id": case_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "status": "open",
            "priority": priority,
            "description": description,
            "amount_inr": amount_inr,
            "trace_id": trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info({
            "event": "escalation_case_created",
            "trace_id": trace_id,
            "case_id": case_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount_inr": amount_inr,
            "priority": priority,
            "reason": description[:120],
        })
        log_escalation_created(
            trace_id, getattr(self.trace_ctx, "session_id", ""),
            customer_id, order_id, case_id, amount_inr, description[:120],
        )
        return data_store.create_crm_case(new_case)


class get_case_status(TracedTool):
    """Look up a CRM case by case_id and return its current status."""

    def _execute(self, case_id: str, **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        case = data_store.get_case_by_id(case_id)
        if not case:
            return {"not_found": True, "case_id": case_id}
        return case


class get_customer_interaction_history(TracedTool):
    """
    RAG lookup over past CRM support interactions for a customer.
    Use when repeat contact or prior-issue context is needed — not for orders or policy.
    """

    def _execute(
        self, customer_id: str, query: str, top_k: int = 3, **kwargs
    ) -> Dict[str, Any]:
        from agent.vector_store import search_customer_history

        data_store = get_data_store()
        if not data_store.get_customer(customer_id):
            raise ValueError(f"Customer {customer_id} not found")

        interactions = search_customer_history(
            customer_id=customer_id,
            query=query,
            top_k=top_k,
        )

        return {
            "customer_id": customer_id,
            "query": query,
            "interactions": interactions,
            "count": len(interactions),
        }
