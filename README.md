# AtlasCare — Agentic AI Customer Support

**Version 3.0** · LangGraph + Groq (LLaMA 3.3 70B) · Acme Retail Tier-1 Automation

AtlasCare is a production-grade agentic customer support system that handles order tracking, cancellations, refunds, address updates, policy queries, case lookups, and interaction history — all through a single LLM intent agent backed by deterministic tool execution.

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
# 1. Create a .env file with your API keys
cp .env.example .env   # then fill in your keys

# 2. Run
docker-compose up

# 3. Open browser
http://localhost:8000
```

### Option 2 — Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add keys to .env (see Environment Variables below)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key          # https://console.groq.com
LANGCHAIN_TRACING_V2=true               # LangSmith observability
LANGCHAIN_API_KEY=your_langsmith_key    # https://smith.langchain.com
LANGCHAIN_PROJECT=AtlasCare
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Customer message → AI response + full trace |
| `POST` | `/query/stream` | Same as above via SSE (typewriter effect in UI) |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | LLM cost, latency, journey stats |

---

## Demo Scenarios

```bash
# Greeting (fast-path — 0 LLM calls)
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Hi good morning","session_id":"demo"}'

# J1 — Order tracking
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Where is my order ORD-78321?","session_id":"demo"}'

# J2 — Cancel + refund
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Cancel item 1 from ORD-78321 and refund me","session_id":"demo"}'

# J2 — Address update
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Ship my order to my office address","session_id":"demo"}'

# J3 — High-value escalation (guardrail fires, 0 LLM calls)
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"I want a refund of Rs.42000 for my damaged laptop","session_id":"demo"}'

# J4 — Interaction history (ChromaDB RAG)
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"What was my last complaint?","session_id":"demo","customer_id":"CUST-001"}'

# J5 — Case status
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"What is the status of CASE-AB1234?","session_id":"demo"}'

# J-KB — Policy question
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"What is your return policy?","session_id":"demo"}'

# Multi-turn — session memory (follow-up without repeating order ID)
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Where is ORD-78321?","session_id":"s1"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Cancel it","session_id":"s1"}'
```

---

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Architecture

```
Customer Message
      │
      ▼
  guardrail ──► INJECTION / ESCALATE (0 LLM calls)
      │
      ▼
  intent (1 Groq call — LLaMA 3.3 70B)
      │  decides which tools to call for ANY journey
      ▼
  executor ──► runs tools sequentially
      │
      ▼
  synthesize ──► template response from tool outputs (0 LLM calls)
```

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Project Layout

```
main.py                  # FastAPI entry point
agent/
  graph.py               # LangGraph 4-node pipeline
  guardrail.py           # Pre-LLM injection + ₹25K check
  pydantic_agents.py     # Groq REST API intent agent
  fast_paths.py          # Deterministic plan builders + synthesis templates
  executor.py            # Sequential tool runner
  session_memory.py      # In-memory multi-turn conversation state
  cache.py               # Singleton data store (O(1) JSON lookups)
  vector_store.py        # ChromaDB RAG for interaction history
tools/
  oms.py                 # Order management tools
  crm.py                 # CRM + case tools
  payments.py            # Refund tool
  kb.py                  # Knowledge base search
schemas/                 # Pydantic models (request, trace, plan)
data/                    # Mock JSON data + ChromaDB persistent store
frontend/                # Chat UI with streaming + trace panel
prompts/
  system_conductor.txt   # Single LLM prompt covering all journeys
Dockerfile
docker-compose.yml
docs/
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| AI Orchestration | LangGraph |
| LLM | Groq REST API — LLaMA 3.3 70B Versatile |
| Observability | LangSmith |
| RAG | ChromaDB + ONNX MiniLM embeddings |
| Containerisation | Docker + Docker Compose |
| Data | In-memory JSON (O(1) lookups) |
