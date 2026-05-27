# AtlasCare — Architecture

**Version 3.0** · LangGraph + Groq (LLaMA 3.3 70B) · May 2026

---

## 1. Problem & Approach

Acme Retail handles ~18,000 Tier-1 support contacts/day. Key pain points: high cost from human agents handling routine queries, inconsistent responses, and slow resolution for compound requests (cancel + refund + reship in one message).

AtlasCare is an **agentic layer** that:
1. Authenticates the customer via JWT (issued at login, verified on every request)
2. Understands customer intent via a single LLM call (Groq/LLaMA)
3. Executes the right tools (OMS, CRM, Payments, KB) deterministically
4. Returns grounded, auditable responses — no hallucination
5. Escalates to humans only when truly needed (high-value refunds, complex cases)

---

## 2. Authentication Flow

```
POST /auth/login
Body:     { "customer_id": "CUST-001" }
Response: { "token": "<JWT>", "expires_in_hours": 8 }

POST /query
Header: Authorization: Bearer <JWT>    ← customer_id extracted server-side
Body:   { "message": "...", "session_id": "..." }
```

- JWT is signed with `JWT_SECRET` (from `.env`) using HS256, expires in 8 hours
- `customer_id` is **never** accepted in the `/query` request body
- On the first authenticated `/query` call, `customer_id` is bound to the session in `session_memory` — subsequent turns in the same session inherit it automatically
- Unauthenticated requests are accepted but ownership checks are skipped

---

## 3. Request Pipeline

Every customer message flows through a 4-node LangGraph state machine:

```
POST /query  or  POST /query/stream
        │
        ▼
┌──────────────┐
│  guardrail   │  Pre-LLM hard rules (0 LLM calls)
│              │  • Prompt injection scan → block immediately
│              │  • Refund intent + amount > ₹25K → ESCALATE
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    intent    │  Single Groq call (LLaMA 3.3 70B)
│              │  • Reads prompts/system_conductor.txt
│              │  • Resolves session memory (order/case from prior turns)
│              │  • Returns JSON plan: {"steps": [{"tool":..,"params":..}]}
│              │  • Bypassed for greetings, injection, and ₹25K escalations
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   executor   │  Runs plan steps sequentially with output-based gate checks
│              │  • Every tool call recorded in TraceContext
│              │  • Gate: cancel_* output inspected before execute_refund runs
│              │  • Soft-failure (already_cancelled, not_found) → stops chain
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  synthesize  │  Template-based response from tool outputs (0 LLM calls)
│              │  • Never hallucinates — answers only from tool output data
│              │  • Handles every tool output variant (not_found, cancelled…)
└──────────────┘
```

---

## 4. Journey Types

| Journey | Trigger | LLM Calls | Key Tools |
|---------|---------|-----------|-----------|
| **J-GREET** | Pure greeting / chitchat | 0 (dictionary fast-path) | `greeting` sentinel |
| **J1** | Order tracking / status | 1 (Groq) | `get_order_status` |
| **J2** | Cancel / refund / address update | 1 (Groq) | `cancel_order_item`, `cancel_full_order`, `execute_refund`, `update_shipping_address` |
| **J3** | Refund > ₹25K (guardrail) | 0 (bypassed) | `create_crm_case` |
| **J4** | Interaction history / follow-up | 1 (Groq) | `get_customer_interaction_history` → ChromaDB RAG |
| **J5** | Case status lookup | 1 (Groq) | `get_case_status` |
| **J-KB** | Policy / returns / warranty | 1 (Groq) | `search_kb` |
| **J-BLOCKED** | Prompt injection detected | 0 | `blocked_injection` sentinel |

**Maximum 1 Groq call per request.**

---

## 5. Key Components

| Component | File | Role |
|-----------|------|------|
| LangGraph pipeline | `agent/graph.py` | 4-node state machine orchestration |
| Guardrail | `agent/guardrail.py` | Injection scan + ₹25K threshold (pre-LLM) |
| Intent agent | `agent/pydantic_agents.py` | Groq REST API — direct httpx, no SDK |
| Conductor prompt | `prompts/system_conductor.txt` | All tools + decision rules + J2 step ordering |
| Fast paths | `agent/fast_paths.py` | Greeting detection, fallback planners, synthesis templates |
| Executor | `agent/executor.py` | Runs `ExecutionPlan` steps, executor gate check |
| Session memory | `agent/session_memory.py` | In-memory multi-turn state (last 10 turns + customer_id) |
| Data store | `agent/cache.py` | Singleton in-memory JSON (O(1) lookups) |
| Vector store | `agent/vector_store.py` | ChromaDB + MiniLM embeddings for J4 RAG |
| Metrics | `agent/metrics.py` | Token cost, latency, journey stats |
| Audit log | `agent/audit.py` | Append-only `audit.jsonl` per request |
| Auth | `main.py` | JWT issuance (`POST /auth/login`) + `get_customer_from_token` dependency |

---

## 6. Intent Agent — How It Works

The single Groq call in `_intent_node` receives:

1. **System prompt** (`prompts/system_conductor.txt`) — lists every available tool with usage rules, decision logic, and the mandatory J2 step-ordering rules
2. **User prompt** built from:
   - Last 10 conversation turns (session memory)
   - Current customer message
   - Customer ID (from JWT via session memory — never from the request body)
   - Resolved order context (items, status, payment method, saved addresses)

Groq returns a JSON plan:
```json
{
  "steps": [
    {"tool": "cancel_order_item",       "params": {"order_id": "ORD-79500", "line_id": 3}},
    {"tool": "execute_refund",          "params": {"order_id": "ORD-79500", "amount_inr": 1299.0, "method": "upi"}},
    {"tool": "update_shipping_address", "params": {"order_id": "ORD-79500", "customer_id": "CUST-001", "address_label": "office"}}
  ]
}
```

---

## 7. Conductor Prompt — J2 Step Ordering

`prompts/system_conductor.txt` includes a dedicated `=== J2 STEP ORDERING (MANDATORY) ===` section:

```
Step 1 — cancel_order_item OR cancel_full_order  (first — establishes cancellation)
Step 2 — execute_refund                          (after cancel confirms success)
Step 3 — update_shipping_address                 (last — applies to remaining items)
```

Explicit ✗ violation list lets the model self-check before returning the plan. The prompt also includes a controlled `search_kb` tag vocabulary (6 known pairs) to prevent hallucinated tags that would miss the KB index.

---

## 8. Session Memory

`agent/session_memory.py` maintains per-session state in memory:

- **Last 10 conversation turns** (user message + agent response pairs, agent trimmed to 120 chars)
- **Last resolved order_id** — so "cancel it" works without repeating the order ID
- **Last resolved case_id** — for follow-up case queries
- **customer_id** — bound at login, used for ownership checks throughout the session

> **Production note:** currently backed by a Python dict (in-process memory). In production this must be replaced with **Redis** — see Section 14.

---

## 9. Executor — Gate Check Between Steps

The executor (`agent/executor.py`) validates cancel tool output before allowing downstream steps to run:

```
cancel_order_item → {already_cancelled: True}
        │
        ▼
   Gate check: failure key present?
   (not_found, unauthorized, already_cancelled, error, success=False, cancelled_count=0)
        │
        YES → stop here, execute_refund never runs ✅
```

This prevents phantom refunds when an item was cancelled moments before the request arrived.

---

## 10. Safety & Guardrails

Five independent layers prevent unsafe actions:

| Layer | What it catches | Where |
|-------|----------------|-------|
| JWT + ownership check | customer_id mismatch → rejects access to another customer's order | `fast_paths.py:_ownership_plan()` |
| Prompt injection scan | Jailbreak phrases blocked pre-LLM | `guardrail.py` |
| ₹25K escalation | Refund intent + amount > threshold → CRM, no LLM call | `guardrail.py` |
| Executor gate check | Soft-failure cancel blocks downstream refund | `executor.py` |
| Tool-level cap | `execute_refund` rejects amount > config limit independently | `tools/payments.py` |

Additional safety:
- **Template synthesis** — responses grounded in tool output only, no LLM free text → zero hallucination on factual fields
- **Amount coercion** — `execute_refund` coerces `amount_inr` to `float` before comparison, guarding against Groq returning numeric params as strings

---

## 11. Observability — LangSmith

Every request is traced end-to-end in LangSmith:

```
run_query
 ├── guardrail node  — injection check, guardrail result
 ├── intent node     — exact Groq prompt + response, plan output
 ├── executor node   — each tool: input, output, latency_ms
 └── synthesize node — final response text
```

Enabled via `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env`. LangGraph auto-detects — zero code changes.

---

## 12. Data Layer

**`agent/cache.py`** — singleton `DataStore`:
- Loads all JSON files from `data/` into memory on first import
- All reads: O(1) dict lookups
- Writes: merges in-memory state with on-disk file

**`agent/vector_store.py`** — ChromaDB:
- Persistent collection at `data/chroma_db/`
- ONNX MiniLM embeddings via `DefaultEmbeddingFunction` (no API call needed)
- Filtered by `customer_id` — never crosses customer data
- New interactions indexed on every completed query via `log_interaction()`
- Falls back to keyword overlap scoring if ChromaDB unavailable

---

## 13. Tools

All tools extend `TracedTool` (`tools/base.py`), which auto-records every call into `TraceContext`.

| Tool | File | Journey |
|------|------|---------|
| `get_order_status` | `tools/oms.py` | J1, J2 |
| `cancel_order_item` | `tools/oms.py` | J2 |
| `cancel_full_order` | `tools/oms.py` | J2 |
| `update_shipping_address` | `tools/oms.py` | J2 |
| `execute_refund` | `tools/payments.py` | J2 |
| `create_crm_case` | `tools/crm.py` | J3, J2 (escalation) |
| `get_case_status` | `tools/crm.py` | J5 |
| `get_customer_interaction_history` | `tools/crm.py` | J4 |
| `search_kb` | `tools/kb.py` | J-KB |
| `greeting` | `tools/oms.py` | J-GREET (no-op sentinel) |

---

## 14. Production Roadmap

### Redis — Session & Cache Layer

Currently `session_memory.py` uses an in-process Python dict. Risks:
- Sessions are lost on every server restart or crash
- Multi-instance deployments get split sessions (turn 1 hits instance A, turn 2 hits instance B, context lost)

**Target:**
```
session_memory.py  →  Redis HASH per session_id  (TTL: 24h)
agent/cache.py     →  Redis as write-through for order/CRM mutations
```

The `get_session / update_session` API stays identical — only the backing store changes. A `REDIS_URL` env var controls the connection with graceful fallback to in-memory dict if Redis is unavailable.

**Docker Compose addition:**
```yaml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]
  volumes: ["redis_data:/data"]
  command: redis-server --appendonly yes
```

### JWT — Production Hardening

Current `/auth/login` trusts the supplied `customer_id` with no credential check. Before production:
- Add password hash or OTP verification
- Add token revocation list (Redis `SET`)
- Add refresh token flow (`POST /auth/refresh`)
- Rate-limit `/auth/login` to prevent brute-force (Redis counter)

### Other Before-Production Items
- Load test at 18K requests/day
- Rate limiting per `customer_id`
- SRE runbooks + alert thresholds on `/metrics`
- Compliance audit export (traces + cases to S3/GCS)
- Expand mock data beyond current sample orders
- Golden-set LLM regression tests for edge-case phrasings

---

## 15. Streaming

`POST /query/stream` returns Server-Sent Events (SSE):
- Server sends one `done` event containing the full response text + trace
- Frontend animates it as typewriter (4 chars / 18ms)
- 4KB SSE padding prefix forces TCP flush
- Loading overlay hides on first event received (not on completion)

---

## 16. Docker

```
Dockerfile          — 2-stage build: builder (gcc + pip install) → runtime (slim, non-root)
docker-compose.yml  — named volumes for ChromaDB + logs, env_file: .env for secrets
```

Secrets (`GROQ_API_KEY`, `JWT_SECRET`, `LANGCHAIN_API_KEY`) are never baked into the image.

---

## 17. API Contract

**POST /auth/login**
```json
Request:  { "customer_id": "CUST-001" }
Response: { "token": "<JWT>", "expires_in_hours": 8 }
```

**POST /query**
```
Header:   Authorization: Bearer <JWT>
Request:  { "message": "string", "session_id": "string" }
Response: { "response": "string", "journey_type": "J1|J2|…", "trace": { … } }
```

**GET /health**
```json
{ "status": "ok", "version": "3.0", "stack": "langgraph+pydantic-ai" }
```
