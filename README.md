# AtlasCare — Agentic AI for Customer Support

**Version 3.0** · LangGraph + Pydantic AI · Acme Retail Tier-1 automation

AtlasCare handles order tracking, compound cancel/refund/ship requests, and high-value refund escalations with full audit traces.

---

## Quick start

```bash
cd AtlasCare
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"  # https://aistudio.google.com
uvicorn main:app --host 127.0.0.1 --port 8000
```

**UI:** open `frontend/index.html` in a browser (server must be running on port 8000).

**API docs:** http://127.0.0.1:8000/docs

---

## Run the three journeys

```bash
# J1 — tracking (< 3s, 1 OMS call)
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Where is my order ORD-78321?","session_id":"s1"}'

# J2 — cancel + refund + ship
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Cancel item 2 from ORD-78321, refund it to HDFC_CREDIT, ship remainder to office address","session_id":"s2"}'

# J3 — escalation (> ₹25K, no payment)
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Full refund of Rs.42000 for damaged laptop ORD-78321","session_id":"s3"}'
```

```bash
pytest tests/ -v
```

---

## Architecture (summary)

```
guardrail → router → fast_plan | llm_plan → executor → synthesize
```

| Journey | LLM calls | Typical latency |
|---------|-----------|-----------------|
| J1 tracking | 0 | < 100ms |
| J2 compound (demo pattern) | 0 | < 100ms |
| J2 novel wording | 1 (Pydantic AI) | 3–8s |
| J3 escalation | 0 | < 50ms |

**Stack:** FastAPI · LangGraph · Pydantic AI · Gemini 2.5 Flash (OpenAI-compatible) · in-memory mock data (OMS, CRM, KB, Payments)

**Details:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (architecture + KPIs)  
**Testing / demo:** [docs/TESTING.md](docs/TESTING.md)  
**Test plan:** [docs/test_plan_summary.txt](docs/test_plan_summary.txt)

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/query` | Customer message → response + trace |
| GET | `/health` | Liveness |
| GET | `/metrics` | LLM cost, latency, journey stats (optional) |

---

## Project layout

```
main.py              # FastAPI entry
requirements.txt
agent/               # graph, guardrail, fast_paths, pydantic_agents, executor, metrics, cache
tools/               # OMS, CRM, KB, Payments (TracedTool)
schemas/             # request, trace, plan
data/                # mock JSON
frontend/            # chat + trace manager UI
tests/
docs/                # ARCHITECTURE.md, TESTING.md, test_plan_summary.txt
prompts/             # Pydantic AI conductor prompt (J2 fallback)
```

---

## Prerequisites

- Python **3.10–3.12** (per assignment); also runs on 3.11/3.13 in development
- `GEMINI_API_KEY` environment variable
- Packages: `fastapi`, `uvicorn`, `httpx`, `langgraph`, `pydantic-ai` (see `requirements.txt`)

---

## Known limitations

- `session_id` is logged but **multi-turn history** is not wired to the planner yet
- `search_kb` exists but is not invoked on default J1–J3 paths
- Mock data: 3 sample orders (expand using schemas in `Tools Schema & Sample Data/`)
- Template responses (consistent, tool-grounded); optional LLM paraphrase not enabled on fast path

---

## Author

Saurabh Singh · MLE Track Transfer Case Study · May 2026
