# AtlasCare — KPI Framework

> Full architecture context: [ARCHITECTURE.md](./ARCHITECTURE.md)

## KPI Targets

| Layer | KPI | Target | Source |
|-------|-----|--------|--------|
| **Business** | Autonomous resolution rate | > 70% | Journeys resolved without human escalation |
| **Business** | Cost per contact (LLM) | < $0.001 avg | `/metrics` → `cost_usd` |
| **Quality** | Journey pass rate (J1–J5, J-KB) | 100% | `pytest tests/` |
| **Quality** | Hallucination rate | 0% on order facts | Template synthesis — all responses grounded in tool output only |
| **Safety** | Guardrail false negative rate | 0% | No `execute_refund` fires when amount > ₹25,000 |
| **Operational** | J1 P95 latency | < 3 s | `trace.latency_ms` |
| **Operational** | Tool success rate | > 99% | Trace `success` flags |
| **Operational** | LLM error rate | < 1% | `/metrics` → `llm_stats` |

## How to measure

- **Latency** — every request emits `latency_ms` in the trace; aggregate via `GET /metrics`
- **Cost** — `GET /metrics` → `total_cost_usd` and per-request `cost_usd` from Pydantic AI usage
- **Tool success** — each `tool_call` object in the trace carries a `success` boolean
- **Guardrail** — `GET /metrics` → `guardrail_blocks_total`; cross-check against `audit.jsonl`
- **Autonomous resolution** — requests that do NOT create a CRM case (`create_crm_case` not in tool calls)
