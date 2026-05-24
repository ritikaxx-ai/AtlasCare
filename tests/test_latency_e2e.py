"""
End-to-end latency benchmarks (requires running server + GEMINI_API_KEY for J2).

Run:
  GEMINI_API_KEY=... uvicorn main:app --port 8000 &
  pytest tests/test_latency_e2e.py -v -s
"""
import os
import time

import pytest
import requests

BASE = os.environ.get("ATLASCARE_URL", "http://127.0.0.1:8000")
HAS_KEY = bool(os.environ.get("GEMINI_API_KEY"))


def _post(message: str, session_id: str) -> tuple[dict, float]:
    start = time.perf_counter()
    r = requests.post(
        f"{BASE}/query",
        json={"message": message, "session_id": session_id},
        timeout=60,
    )
    elapsed = time.perf_counter() - start
    r.raise_for_status()
    return r.json(), elapsed


@pytest.fixture(scope="module", autouse=True)
def check_server():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        if r.status_code != 200:
            pytest.skip("Server not healthy")
    except requests.RequestException:
        pytest.skip(f"Server not running at {BASE}")


def test_health_latency():
    start = time.perf_counter()
    r = requests.get(f"{BASE}/health", timeout=5)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 0.5


def test_j1_api_latency_under_3s():
    data, elapsed = _post("Where is my order ORD-78321?", "lat-j1")
    print(f"\nJ1 wall: {elapsed:.3f}s | trace latency_ms: {data['trace']['latency_ms']}")
    assert elapsed < 3.0, f"J1 API took {elapsed:.2f}s"
    assert data["trace"]["tool_calls"][0]["tool_name"] == "get_order_status"


def test_j3_api_latency_under_3s():
    data, elapsed = _post(
        "Full refund of Rs.42000 for damaged laptop ORD-78321", "lat-j3"
    )
    print(f"\nJ3 wall: {elapsed:.3f}s | trace latency_ms: {data['trace']['latency_ms']}")
    assert elapsed < 3.0, f"J3 API took {elapsed:.2f}s"
    assert "create_crm_case" in [t["tool_name"] for t in data["trace"]["tool_calls"]]


@pytest.mark.skipif(not HAS_KEY, reason="GEMINI_API_KEY required for J2 LLM plan")
def test_j2_api_latency_under_15s():
    data, elapsed = _post(
        "Cancel item 2 from ORD-78321, refund it to HDFC_CREDIT, ship remainder to office address",
        "lat-j2",
    )
    print(f"\nJ2 wall: {elapsed:.3f}s | trace latency_ms: {data['trace']['latency_ms']}")
    assert elapsed < 15.0, f"J2 API took {elapsed:.2f}s"
    names = [t["tool_name"] for t in data["trace"]["tool_calls"]]
    assert "cancel_order_item" in names
    assert "execute_refund" in names
