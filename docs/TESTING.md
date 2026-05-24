# AtlasCare — Testing & Demo Guide

**v3.0** · See [ARCHITECTURE.md](./ARCHITECTURE.md) for design.

---

## Quick start

```bash
cd AtlasCare
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_from_aistudio.google.com"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open UI: `open frontend/index.html` (or `cd frontend && python3 -m http.server 3000`)

---

## Automated tests

```bash
pytest tests/ -v
```

| File | Covers |
|------|--------|
| `tests/test_agent.py` | J1, J2, J3 unit tests (J2 LLM mocked) |
| `tests/test_latency_e2e.py` | Live API latency (server must be running) |

**Latency (typical, fast path):** J1/J2/J3 demo messages < 100ms API time.

---

## Three required journeys (curl)

**J1 — Tracking**
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Where is my order ORD-78321?","session_id":"demo-j1"}'
```
Expect: 1 tool `get_order_status`, tracking in response, `latency_ms` < 3000.

**J2 — Compound**
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Cancel item 2 from ORD-78321, refund it to HDFC_CREDIT, ship remainder to office address","session_id":"demo-j2"}'
```
Expect: `cancel_order_item` → `execute_refund` → `update_shipping_address` in trace order.

**J3 — Escalation**
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"message":"Full refund of Rs.42000 for damaged laptop ORD-78321","session_id":"demo-j3"}'
```
Expect: `create_crm_case` only, no `execute_refund`, case linked to `trace_id`.

---

## UI demo (5 min)

1. Start server + open `frontend/index.html`
2. **J1** — "Where is my order ORD-78321?" → J1 badge, tracking in trace panel
3. **J3** — ₹42,000 refund message → J3 badge, CRM case in tool output
4. **J2** — cancel/refund/ship message → three tools in trace details
5. Show header stats and journey filters (J1 / J2 / J3 / All)

**Trace panel shows:** trace_id, order ID, tracking number, tool inputs/outputs, latency per tool.

---

## Metrics endpoint

```bash
curl http://127.0.0.1:8000/metrics
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CORS / connection error | Server on port 8000; check `API_BASE` in `frontend/script.js` |
| J2 500 on live API | Gemini quota — demo J2 uses fast path (no LLM) when pattern matches |
| Empty trace | Hard-refresh browser; check console (F12) |
