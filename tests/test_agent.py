import pytest
from fastapi.testclient import TestClient
import os
import time

os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from agent.cache import get_data_store
from schemas.plan import ExecutionPlan, PlanStep

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_order_data():
    """Reset ORD-78321 line 2 to active for repeatable J2 tests."""
    store = get_data_store()
    order = store.get_order("ORD-78321")
    if order:
        for item in order["items"]:
            if item["line_id"] == 2:
                item["status"] = "active"
        order["total_amount"] = sum(
            i["unit_price"] * i["quantity"]
            for i in order["items"]
            if i["status"] == "active"
        )
        store.update_order("ORD-78321", order)
    yield


@pytest.fixture(autouse=True)
def reset_graph_singleton():
    """Force graph rebuild so monkeypatches apply."""
    import agent.graph as graph_mod
    graph_mod._atlascare_graph = None
    yield
    graph_mod._atlascare_graph = None


@pytest.fixture(autouse=True)
def mock_j2_llm_plan(monkeypatch):
    """Only J2 uses Pydantic AI — mock it in unit tests."""

    async def mock_generate_plan_llm(message: str, **kwargs) -> ExecutionPlan:
        if "Cancel item 2" in message:
            return ExecutionPlan(
                steps=[
                    PlanStep(
                        tool="cancel_order_item",
                        params={"order_id": "ORD-78321", "line_id": 2},
                    ),
                    PlanStep(
                        tool="execute_refund",
                        params={
                            "order_id": "ORD-78321",
                            "amount_inr": 1500,
                            "method": "HDFC_CREDIT",
                        },
                    ),
                    PlanStep(
                        tool="update_shipping_address",
                        params={
                            "order_id": "ORD-78321",
                            "address": {
                                "line1": "Acme Office",
                                "city": "Bengaluru",
                                "state": "Karnataka",
                                "pincode": "560001",
                            },
                        },
                    ),
                ]
            )
        raise ValueError(f"mock_generate_plan_llm: unrecognised message pattern — '{message[:60]}'")

    monkeypatch.setattr(
        "agent.pydantic_agents.generate_plan_llm", mock_generate_plan_llm
    )
    monkeypatch.setattr("agent.graph.generate_plan_llm", mock_generate_plan_llm)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "3.0"


def test_j1_tracking_fast_path():
    """J1 uses LangGraph fast path — no LLM, sub-second."""
    payload = {
        "message": "Where is my order ORD-78321?",
        "session_id": "sess-1",
    }
    start = time.perf_counter()
    response = client.post("/query", json=payload)
    latency = time.perf_counter() - start

    assert response.status_code == 200
    data = response.json()

    assert latency < 3.0, f"J1 took {latency:.2f}s — expected < 3s (no LLM)"
    trace = data["trace"]
    tool_calls = trace["tool_calls"]

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "get_order_status"
    assert tool_calls[0]["success"] is True
    assert "ORD-78321" in data["response"]
    assert trace["latency_ms"] < 3000


def test_j2_compound():
    payload = {
        "message": "Cancel item 2 from ORD-78321, refund it to HDFC_CREDIT, ship remainder to office address",
        "session_id": "sess-2",
    }
    start = time.perf_counter()
    response = client.post("/query", json=payload)
    latency = time.perf_counter() - start

    assert response.status_code == 200
    assert latency < 10.0, f"J2 took {latency:.2f}s"
    data = response.json()

    tool_calls = data["trace"]["tool_calls"]
    tool_names = [tc["tool_name"] for tc in tool_calls]

    assert "cancel_order_item" in tool_names
    assert "execute_refund" in tool_names

    cancel_idx = tool_names.index("cancel_order_item")
    refund_idx = tool_names.index("execute_refund")
    assert cancel_idx < refund_idx


def test_j4_customer_history_rag():
    """J4 uses RAG over CRM history — not invoked on J1/J2/J3 paths."""
    payload = {
        "message": (
            "This is the third time I'm calling about the same issue — "
            "my damaged laptop from order ORD-78321 still hasn't been refunded."
        ),
        "session_id": "sess-4",
    }
    start = time.perf_counter()
    response = client.post("/query", json=payload)
    latency = time.perf_counter() - start

    assert response.status_code == 200
    data = response.json()
    tool_calls = data["trace"]["tool_calls"]
    tool_names = [tc["tool_name"] for tc in tool_calls]

    assert "get_customer_interaction_history" in tool_names
    assert "get_order_status" not in tool_names
    assert tool_calls[0]["success"] is True

    history_out = next(
        tc["output"]
        for tc in tool_calls
        if tc["tool_name"] == "get_customer_interaction_history"
    )
    assert history_out["count"] >= 1
    assert latency < 30.0, f"J4 took {latency:.2f}s (includes first-time index build)"


def test_j_kb_policy_lookup():
    """Policy questions use deterministic KB tag search — no RAG."""
    payload = {
        "message": "What is the refund limit? Can I return an item after 30 days?",
        "session_id": "sess-kb",
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    tool_names = [tc["tool_name"] for tc in data["trace"]["tool_calls"]]

    assert "search_kb" in tool_names
    assert "get_customer_interaction_history" not in tool_names
    assert "refund" in data["response"].lower() or "25" in data["response"]


def test_j1_does_not_use_rag():
    """J1 tracking must not trigger vector search."""
    payload = {
        "message": "Where is my order ORD-78321?",
        "session_id": "sess-j1-rag",
    }
    response = client.post("/query", json=payload)
    data = response.json()
    tool_names = [tc["tool_name"] for tc in data["trace"]["tool_calls"]]
    assert tool_names == ["get_order_status"]


def test_j3_escalation_fast_path():
    """J3 guardrail + fast path — no LLM."""
    payload = {
        "message": "Full refund of Rs.42000 for a damaged laptop from ORD-78321",
        "session_id": "sess-3",
    }
    start = time.perf_counter()
    response = client.post("/query", json=payload)
    latency = time.perf_counter() - start

    assert response.status_code == 200
    assert latency < 3.0, f"J3 took {latency:.2f}s"
    data = response.json()

    tool_calls = data["trace"]["tool_calls"]
    tool_names = [tc["tool_name"] for tc in tool_calls]

    assert "execute_refund" not in tool_names
    assert "create_crm_case" in tool_names

    crm_call = next(tc for tc in tool_calls if tc["tool_name"] == "create_crm_case")
    assert crm_call["success"] is True
