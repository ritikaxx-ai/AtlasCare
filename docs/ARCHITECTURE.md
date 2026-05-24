# AtlasCare — Architecture & KPI Framework

**Version 3.0** · LangGraph + Pydantic AI · May 2026

---

## 1. Problem & approach

Acme Retail handles ~18,000 Tier-1 contacts/day. Pain points: high cost (bots + human fallback), inconsistent answers, slow compound flows (cancel + refund + reship).

AtlasCare is an **agentic layer** that plans tool calls against OMS, CRM, KB, and Payments, executes deterministically, and returns auditable traces. It **escalates** when refunds exceed ₹25,000.

---

## 2. Architecture (LangGraph)

```
POST /query
    │
    ▼
┌─────────────┐
│  guardrail  │  Pre-LLM: amount > ₹25K + refund intent → ESCALATE
└──────┬──────┘
       ▼
┌─────────────┐
│   router    │  Classify J1 / J2 / J3
└──────┬──────┘
       ├── fast_plan (J1, J3, known J2) ──► 0 LLM calls
       └── llm_plan  (novel J2 only)     ──► Pydantic AI → ExecutionPlan
       ▼
┌─────────────┐
│  executor   │  Sequential tools, fail-fast, TracedTool middleware
└──────┬──────┘
       ▼
┌─────────────┐
│ synthesize  │  Template response from tool outputs (no 2nd LLM)
└─────────────┘
```

| Component | File | Role |
|-----------|------|------|
| LangGraph workflow | `agent/graph.py` | State machine orchestration |
| Fast paths | `agent/fast_paths.py` | J1/J3/J2 deterministic plans + templates |
| Pydantic AI planner | `agent/pydantic_agents.py` | Structured plan for unknown J2 |
| Guardrail | `agent/guardrail.py` | Pre-LLM refund threshold |
| Executor | `agent/executor.py` | Runs `ExecutionPlan` steps |
| Data store | `agent/cache.py` | In-memory JSON (sub-ms lookups) |
| Metrics | `agent/metrics.py` | LLM tokens, cost, journey stats |

### Enterprise integrations

| System | Tools | Used in journeys |
|--------|-------|------------------|
| OMS | `get_order_status`, `cancel_order_item`, `update_shipping_address` | J1, J2 |
| Payments | `execute_refund` (hard cap ₹25K in tool) | J2 |
| CRM | `create_crm_case`, `get_customer_profile`, `get_customer_address` | J3, J2 (office address) |
| KB | `search_kb` | Available; wired for future policy journeys |

### Observability (SRE)

- Every tool call: `tool_name`, `input`, `output`, `latency_ms`, `success`, `timestamp`
- `trace_id` on CRM cases for compliance audit
- `GET /health` — liveness
- `GET /metrics` — LLM usage, journey latency, guardrail triggers

### Safety

1. **Pre-LLM guardrail** — regex amount extraction; no Payments on escalate  
2. **Tool-level cap** — `execute_refund` rejects amount > config limit  
3. **Fail-fast executor** — stops on first tool failure  
4. **Template synthesis** — answers only from tool `output` (no hallucinated tracking/status)

---

## 3. Journey mapping

| Journey | User intent | Plan | LLM |
|---------|-------------|------|-----|
| **J1** | Order tracking | 1× `get_order_status` | 0 |
| **J2** | Cancel + refund + ship | cancel → refund → address | 0 (known pattern) or 1 (Pydantic AI) |
| **J3** | Refund > ₹25K | `create_crm_case` + `trace_id` | 0 |

---

## 4. KPI framework

See [docs/KPI_FRAMEWORK.md](./KPI_FRAMEWORK.md) for the full KPI table, targets, and measurement guide.

---

## 5. Known limitations & roadmap

| Item | Status |
|------|--------|
| Multi-turn `session_id` memory | Not implemented (API field only) |
| KB in default J3 response | Planned (policy citation) |
| Case status lookup API | Create only today |
| Synthetic data volume | 3 orders — expand per schema for scale demos |
| Python version | Tested 3.11–3.13; brief specifies 3.10–3.12 |

---

## 6. API contract

**POST /query** — `{ "message": string, "session_id": string }`  
**Response** — `{ "response": string, "trace": { "trace_id", "session_id", "latency_ms", "tool_calls": [] } }`  
**GET /health** — HTTP 200
