# AtlasCare — Architecture

**Version 3.0** · LangGraph + Groq (LLaMA 3.3 70B) · May 2026

---

## 1. Problem & Approach

Acme Retail handles ~18,000 Tier-1 support contacts/day. Key pain points: high cost from human agents handling routine queries, inconsistent responses across agents, and slow resolution for compound requests (cancel + refund + reship in one message).

AtlasCare is an **agentic layer** that:
1. Understands customer intent via a single LLM call (Groq/LLaMA)
2. Executes the right tools (OMS, CRM, Payments, KB) deterministically
3. Returns grounded, auditable responses — no hallucination
4. Escalates to humans only when truly needed (high-value refunds, complex cases)

---

## 2. Request Pipeline

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
│              │  • Reads system_conductor.txt (all tools listed)
│              │  • Resolves session memory (order/case from prior turns)
│              │  • Returns JSON plan: {"steps": [{"tool":..,"params":..}]}
│              │  • Bypassed for greetings (fast-path), injection, escalation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   executor   │  Runs plan steps sequentially with output-based gate checks
│              │  • Every tool call recorded in TraceContext
│              │  • tool_name, input, output, latency_ms, success, timestamp
│              │  • Gate: cancel_* output inspected before refund/address runs
│              │  • Soft-failure (already_cancelled, not_found) → stops chain
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  synthesize  │  Template-based response from tool outputs (0 LLM calls)
│              │  • Never hallucinates — answers only from tool output data
│              │  • Handles every tool output variant (not_found, cancelled, etc.)
└──────────────┘
```

---

## 3. Journey Types

| Journey | Trigger | LLM Calls | Key Tools |
|---------|---------|-----------|-----------|
| **J-GREET** | Pure greeting / chitchat | 0 (regex fast-path) | `greeting` sentinel |
| **J1** | Order tracking / status | 1 (Groq) | `get_order_status` |
| **J2** | Cancel / refund / address update | 1 (Groq) | `cancel_order_item`, `cancel_full_order`, `execute_refund`, `update_shipping_address` |
| **J3** | Refund > ₹25K (guardrail escalation) | 0 (bypassed) | `create_crm_case` |
| **J4** | Interaction history / follow-up | 1 (Groq) | `get_customer_interaction_history` → ChromaDB RAG |
| **J5** | Case status lookup | 1 (Groq) | `get_case_status` |
| **J-KB** | Policy / returns / warranty questions | 1 (Groq) | `search_kb` |
| **J-BLOCKED** | Prompt injection detected | 0 | `blocked_injection` sentinel |

**LLM call budget:** 1 Groq call for all journeys except J-GREET (0), J3 (0), J-BLOCKED (0).

---

## 4. Key Components

| Component | File | Role |
|-----------|------|------|
| LangGraph pipeline | `agent/graph.py` | 4-node state machine orchestration |
| Guardrail | `agent/guardrail.py` | Injection scan + ₹25K threshold (pre-LLM) |
| Intent agent | `agent/pydantic_agents.py` | Groq REST API call — direct httpx, no SDK |
| Conductor prompt | `prompts/system_conductor.txt` | Lists all tools + decision rules for Groq |
| Fast paths | `agent/fast_paths.py` | Greeting detection, synthesis templates |
| Executor | `agent/executor.py` | Runs `ExecutionPlan` steps, tool registry |
| Session memory | `agent/session_memory.py` | In-memory multi-turn state (last 10 turns) |
| Data store | `agent/cache.py` | Singleton in-memory JSON (O(1) lookups) |
| Vector store | `agent/vector_store.py` | ChromaDB + MiniLM embeddings for J4 RAG |
| Metrics | `agent/metrics.py` | Token cost, latency, journey stats |
| Audit log | `agent/audit.py` | Append-only `audit.jsonl` per request |

---

## 5. Intent Agent — How It Works

The single Groq call in `_intent_node` receives:

1. **System prompt** (`system_conductor.txt`) — lists every available tool with usage rules and decision logic for all journey types
2. **User prompt** built from:
   - Last 10 conversation turns (session memory) — so Groq understands "cancel it" = prior order
   - Current customer message
   - Customer ID
   - Resolved order context (items, status, payment method, saved addresses)

Groq returns a JSON plan:
```json
{
  "steps": [
    {"tool": "cancel_order_item", "params": {"order_id": "ORD-78321", "line_id": 1}},
    {"tool": "execute_refund", "params": {"order_id": "ORD-78321", "amount_inr": 1299.0, "method": "upi"}}
  ]
}
```

The executor runs these steps in order. Synthesis builds the customer-facing response from the outputs.

---

## 6. Session Memory

`agent/session_memory.py` maintains per-session state in memory:

- **Last 10 conversation turns** (user message + agent response pairs)
- **Last resolved order ID** — so "cancel it" works without repeating the order ID
- **Last resolved case ID** — for follow-up case queries

Turns are passed directly to Groq's user prompt so it can resolve references across multiple messages. No extra LLM call needed — it's context injection.

---

## 7. Executor — State Validation Between Steps

The executor (`agent/executor.py`) does more than just run tools in order — it validates the output of each step before allowing the next step to fire.

### The Problem It Solves

The LLM generates a complete plan upfront: `[cancel_order_item → execute_refund]`. If `cancel_order_item` returns a soft-failure (e.g. the item was already cancelled 5 minutes ago), a naive executor would blindly continue to `execute_refund` — issuing a refund for an item the customer is still going to receive.

### Two Stop Conditions

**Hard failure (exception):** Any unhandled exception in a tool stops execution immediately via `break`.

**Soft failure (gate check):** After every `cancel_order_item` or `cancel_full_order`, the executor inspects the output dict before proceeding:

```
cancel_order_item → {already_cancelled: True}
        │
        ▼
   Gate check: any failure key present?
   (not_found, unauthorized, already_cancelled, error, success=False, cancelled_count=0)
        │
        YES → stop here, execute_refund never runs ✅
```

### Failure Keys That Trigger the Gate

| Output key | Meaning |
|---|---|
| `already_cancelled: True` | Item was cancelled before this request |
| `not_found: True` | Order or item doesn't exist |
| `success: False` | Explicit failure from data store |
| `cancelled_count: 0` | Full cancel found no active items |
| `error: "..."` | Tool-level error string |

### Result

- Customer sends: *"Cancel item 1 and refund me"*
- Item 1 was already cancelled → `cancel_order_item` returns `{already_cancelled: True}`
- Gate fires → `execute_refund` is blocked
- Synthesizer reads the `already_cancelled` output → tells the customer the item was already cancelled, no refund needed

---

## 8. Safety & Guardrails

Four independent layers prevent unsafe actions:

| Layer | What it catches | Where |
|-------|----------------|-------|
| Prompt injection scan | Jailbreak phrases ("ignore instructions", "act as", etc.) | `guardrail.py` — before LLM |
| ₹25K escalation | Refund intent + extracted amount > threshold | `guardrail.py` — before LLM |
| Executor gate check | Soft-failure cancel output blocks downstream refund | `executor.py` — between steps |
| Tool-level cap | `execute_refund` rejects amount > config limit independently | `tools/payments.py` |

Additionally:
- **Ownership check** — order access verified against `customer_id` from session
- **Template synthesis** — responses built only from tool output data, never from LLM free text → zero hallucination on factual fields (order ID, tracking number, amounts)
- **Fail-fast executor** — stops on first hard exception, no partial execution

---

## 9. Observability — LangSmith

Every request is traced end-to-end in LangSmith (smith.langchain.com):

```
run_query
 ├── guardrail node     — injection check, guardrail result
 ├── intent node        — Groq prompt sent, response received, plan output
 ├── executor node      — each tool: input, output, latency
 └── synthesize node    — final response text
```

LangSmith is enabled via three environment variables — zero code changes required. LangGraph auto-detects them.

Locally, every request also emits structured JSON logs and appends to `audit.jsonl`.

---

## 10. Data Layer

**`agent/cache.py`** — singleton `DataStore`:
- Loads all JSON files from `data/` into memory on first import
- All reads: O(1) dict lookups
- Writes: merges in-memory state with on-disk file (never overwrites externally-added records)

**`agent/vector_store.py`** — ChromaDB:
- Persistent collection at `data/chroma_db/`
- ONNX MiniLM embeddings via `DefaultEmbeddingFunction` (no API call needed)
- Filtered by `customer_id` — never crosses customer data
- New interactions indexed on every completed query via `log_interaction()`
- Falls back to keyword overlap scoring if ChromaDB unavailable

---

## 11. Tools

All tools extend `TracedTool` (`tools/base.py`). The base class automatically records every call into `TraceContext`:

```python
{
  "tool_name": "execute_refund",
  "input": {"order_id": "ORD-78321", "amount_inr": 1299.0, "method": "upi"},
  "output": {"success": true, "refund_id": "REF-001", "amount_inr": 1299.0},
  "latency_ms": 3,
  "success": true,
  "timestamp": "2026-05-26T12:00:00Z"
}
```

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

---

## 12. Streaming

`POST /query/stream` returns Server-Sent Events (SSE):

- Server sends one `done` event containing the full response text
- Frontend (`frontend/script.js`) animates it as a typewriter effect client-side (4 chars / 18ms tick)
- SSE includes a 4KB padding prefix to force TCP flush on first byte — prevents buffering delays
- Loading overlay hides on first SSE event received (not on completion)

---

## 13. Docker

```
Dockerfile          — 2-stage build: builder (gcc + pip install) → runtime (slim)
docker-compose.yml  — volumes for ChromaDB persistence + logs, env_file for secrets
```

Secrets (API keys) are never baked into the image — always injected via `--env-file .env` at runtime.

ChromaDB data and logs are mounted as named Docker volumes so they persist across container restarts.

---

## 14. API Contract

**POST /query**
```json
Request:  { "message": "string", "session_id": "string", "customer_id": "string (optional)" }
Response: { "response": "string", "journey_type": "J1|J2|J3|J4|J5|J-KB|J-GREET|J-BLOCKED", "trace": { ... } }
```

**GET /health**
```json
{ "status": "ok", "version": "3.0", "stack": "langgraph+pydantic-ai" }
```
