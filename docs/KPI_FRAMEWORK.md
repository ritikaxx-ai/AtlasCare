# AtlasCare — KPI Framework

> Full architecture context: [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## KPI Targets

| Layer | KPI | Target | How We Measure |
|-------|-----|--------|----------------|
| **Business** | Autonomous resolution rate | > 80% | Requests resolved without `create_crm_case` in trace |
| **Business** | LLM cost per contact | < $0.001 avg | `GET /metrics` → `cost_usd` per request |
| **Quality** | Journey accuracy (all journeys) | 100% on test suite | `pytest tests/ -v` — 23 unit tests |
| **Quality** | Hallucination rate | 0% | Template synthesis — all responses grounded in tool output only |
| **Safety** | Guardrail false negative rate | 0% | No `execute_refund` fires when amount > ₹25,000 |
| **Safety** | Phantom refund rate | 0% | Executor gate blocks `execute_refund` when `cancel_*` soft-fails |
| **Safety** | Prompt injection block rate | 100% | `GET /metrics` → `guardrail_blocks_total` |
| **Safety** | Unauthorised order access rate | 0% | Ownership check via JWT customer_id vs order customer_id |
| **Operational** | Greeting latency | < 500ms | J-GREET fast-path — 0 LLM calls |
| **Operational** | J3 / J-BLOCKED latency | < 3s / < 200ms | Pre-LLM guardrail — 0 Groq calls |
| **Operational** | J1/J2 P95 latency | < 3s / < 15s | `trace.latency_ms` in response |
| **Operational** | Tool success rate | > 99% | `success` flag on each `tool_call` in trace |
| **Operational** | LLM error rate | < 1% | `GET /metrics` → `llm_stats.error_count` |

---

## How to Measure

### Latency
Every request returns `latency_ms` in the trace response. Aggregate via:
```bash
curl http://localhost:8000/metrics
```

### LLM Cost
```bash
curl http://localhost:8000/metrics | python3 -m json.tool
# → total_cost_usd, per_journey cost breakdown
```

### Tool Success
Each `tool_call` object in the trace carries a `success` boolean and `latency_ms`.

### Guardrail Effectiveness
```bash
curl http://localhost:8000/metrics
# → guardrail_triggers_total (escalations) + injection_blocks_total
```

Cross-reference against `audit.jsonl` for per-request audit trail.

### Autonomous Resolution
Requests where `create_crm_case` does NOT appear in `tool_calls` = autonomously resolved.

### LangSmith (End-to-End)
Every request is traced in LangSmith with:
- Per-node latency (guardrail, intent, executor, synthesize)
- Exact Groq prompt and response
- Token usage and cost per request
- Full tool call inputs and outputs

---

## LLM Call Budget by Journey

| Journey | Groq Calls | Reason |
|---------|-----------|--------|
| J-GREET | 0 | Dictionary fast-path |
| J3 | 0 | Guardrail bypasses LLM entirely |
| J-BLOCKED | 0 | Injection detected pre-LLM |
| J1, J2, J4, J5, J-KB | 1 | Single intent agent call |

**Maximum 1 Groq call per request.** Previously J2 required 2 calls (router + planner). The unified intent agent reduced this by 50% for J2 and eliminated the router call for all other journeys.

---

## Production Infrastructure KPIs (Future)

Once Redis and hardened JWT are in place:

| KPI | Target | Measure |
|-----|--------|---------|
| Session recovery after restart | 100% | Sessions survive pod restart — Redis TTL 24h |
| Multi-instance session consistency | 100% | All instances read from same Redis, no split sessions |
| Auth brute-force protection | < 5 attempts/min per IP | Redis rate counter on `/auth/login` |
| Token revocation latency | < 100ms | Redis SET lookup on every `/query` request |
