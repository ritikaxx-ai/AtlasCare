from dotenv import load_dotenv
load_dotenv()  # loads GEMINI_API_KEY (and any other vars) from .env before anything else

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import datetime
import json
import jwt
import os
import re
from typing import Optional

from schemas.request import QueryRequest, QueryResponse
from agent.session_memory import get_customer_id, update_session
from schemas.trace import TraceContext
from agent.metrics import get_metrics_collector
from agent.cache import get_data_store
from agent.graph import run_query
from agent.logger import log
from agent.audit import get_by_trace as get_audit_by_trace
from agent.guardrail import MAX_MESSAGE_LENGTH

app = FastAPI(title="Project AtlasCare v3.0 - LangGraph + Pydantic AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_collector = get_metrics_collector()
data_store = get_data_store()

# ---------------------------------------------------------------------------
# JWT auth
# ---------------------------------------------------------------------------

_JWT_SECRET = os.environ.get("JWT_SECRET", "atlascare-secret-key-change-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_HOURS = 8
_bearer = HTTPBearer(auto_error=False)


def _create_token(customer_id: str) -> str:
    payload = {
        "customer_id": customer_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=_JWT_EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_customer_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Optional[str]:
    """
    FastAPI dependency — extracts customer_id from Bearer JWT.
    Returns None if no token is provided (unauthenticated requests still work
    but won't have ownership checks enforced).
    """
    if credentials is None:
        return None
    return _decode_token(credentials.credentials).get("customer_id")


@app.post("/auth/login")
def login(body: dict):
    """
    Issue a JWT for a customer.
    Body: { "customer_id": "CUST-001" }
    Response: { "token": "<JWT>", "expires_in_hours": 8 }

    In production this would verify credentials (password / OTP).
    For the hackathon demo it trusts the supplied customer_id.
    """
    customer_id = body.get("customer_id", "").strip()
    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")
    if not data_store.get_customer(customer_id):
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    token = _create_token(customer_id)
    return {"token": token, "expires_in_hours": _JWT_EXPIRY_HOURS}

_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


@app.get("/chat", include_in_schema=False)
def serve_chat():
    return FileResponse(os.path.join(_frontend_dir, "chat.html"))


@app.get("/ops", include_in_schema=False)
def serve_ops():
    return FileResponse(os.path.join(_frontend_dir, "ops.html"))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversational memory for a session (call on customer switch)."""
    from agent.session_memory import clear_session as _clear
    _clear(session_id)
    return {"cleared": session_id}


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0", "stack": "langgraph+pydantic-ai"}


@app.get("/metrics")
def get_metrics():
    """
    Operational metrics dashboard — matches production spec:
    requests_total, requests_by_journey, avg_latency_ms, guardrail_blocks_total,
    tool_errors, llm_calls_total, escalation_rate.
    """
    mc = metrics_collector
    jm = mc.journey_metrics
    gm = mc.guardrail_metrics
    lm = mc.llm_metrics

    requests_total = len(jm)
    requests_by_journey: dict = {}
    latency_by_journey: dict = {}
    tool_errors: dict = {}

    for j in jm:
        requests_by_journey[j.journey_type] = requests_by_journey.get(j.journey_type, 0) + 1
        latency_by_journey.setdefault(j.journey_type, []).append(j.total_latency_ms)

    avg_latency_ms = {
        k: int(sum(v) / len(v)) for k, v in latency_by_journey.items()
    }

    # Tool errors from trace tool_calls stored in journey metrics isn't direct —
    # we count from the raw LLM failure + a best-effort tool-error counter on metrics
    for m in lm:
        if not m.success:
            tool_errors["llm"] = tool_errors.get("llm", 0) + 1

    guardrail_blocks = sum(1 for g in gm if g.triggered)
    escalations = requests_by_journey.get("J3", 0)
    escalation_rate = round(escalations / requests_total, 3) if requests_total else 0.0

    return {
        "requests_total": requests_total,
        "requests_by_journey": requests_by_journey,
        "avg_latency_ms": avg_latency_ms,
        "guardrail_blocks_total": guardrail_blocks,
        "tool_errors": tool_errors,
        "llm_calls_total": len(lm),
        "llm_success_rate": round(sum(1 for m in lm if m.success) / len(lm), 3) if lm else 1.0,
        "escalation_rate": escalation_rate,
        "total_cost_usd": round(sum(m.cost_usd for m in lm), 4),
        "detailed": mc.get_all_stats(),
    }


# In-memory trace store: trace_id → full pipeline trace
_trace_store: dict = {}


def _store_trace(trace_id: str, trace_data: dict, journey_type: str,
                 message: str, response: str) -> None:
    _trace_store[trace_id] = {
        "trace_id": trace_id,
        "journey_type": journey_type,
        "message_preview": message[:120],
        "response_preview": response[:120],
        "tool_calls": trace_data.get("tool_calls", []),
        "latency_ms": trace_data.get("latency_ms"),
    }


@app.get("/customers/{customer_id}/orders")
def get_customer_orders(customer_id: str):
    """Return all orders for a customer, sorted newest-first."""
    orders = data_store.get_orders_for_customer(customer_id)
    if orders is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return {"customer_id": customer_id, "orders": orders}


@app.get("/logs/stream")
async def stream_logs():
    """
    SSE stream of structured log lines from logs/atlascare.log.
    Tails the file in real time; emits one JSON object per line.
    """
    log_path = os.path.join(os.path.dirname(__file__), "logs", "atlascare.log")

    async def _tail():
        try:
            with open(log_path, "r") as f:
                f.seek(0, 2)   # jump to end
                while True:
                    line = f.readline()
                    if line:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
                    else:
                        await asyncio.sleep(0.5)
        except FileNotFoundError:
            yield f'data: {{"level":"ERROR","message":"Log file not found: {log_path}","event":"log_stream_error"}}\n\n'

    return StreamingResponse(
        _tail(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/logs/recent")
def recent_logs(n: int = 200):
    """Return the last n lines of the log file as parsed JSON objects."""
    log_path = os.path.join(os.path.dirname(__file__), "logs", "atlascare.log")
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        parsed = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    parsed.append({"message": line, "level": "INFO"})
        return {"logs": parsed, "count": len(parsed)}
    except FileNotFoundError:
        return {"logs": [], "count": 0}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    """
    Full request lifecycle for SRE debugging:
    journey type, every tool called, inputs/outputs, latency per step.
    """
    trace = _trace_store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@app.get("/audit/{trace_id}")
def get_audit(trace_id: str):
    """
    Compliance audit trail for a specific interaction.
    Shows every financially significant event: refunds, escalations,
    guardrail activations, injection attempts — with decision_basis.
    """
    events = get_audit_by_trace(trace_id)
    if events is None or len(events) == 0:
        raise HTTPException(status_code=404, detail=f"No audit events for trace {trace_id}")
    return {"trace_id": trace_id, "events": events, "count": len(events)}


_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,128}$')


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    customer_id: Optional[str] = Depends(get_customer_from_token),
):
    """
    LangGraph pipeline:
      guardrail → router → fast_plan|llm_plan → executor → synthesize

    J1/J3: 0 LLM calls (deterministic + templates)
    J2:    1 Pydantic AI LLM call for planning

    Authentication: pass Authorization: Bearer <JWT> header (obtained from POST /auth/login).
    customer_id is extracted server-side — never accepted in the request body.
    """
    # ── Input validation (HTTP 400, never 500) ────────────────────────────
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if len(request.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"message exceeds {MAX_MESSAGE_LENGTH} character limit"
        )
    if not _SESSION_ID_RE.match(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="session_id must be alphanumeric (hyphens/underscores allowed, max 128 chars)"
        )

    # Bind customer_id to this session so subsequent turns don't need the token
    # re-decoded — session_memory becomes the source of truth within a session.
    if customer_id:
        update_session(request.session_id, "", "", customer_id=customer_id)

    try:
        result = await run_query(request.message, request.session_id, customer_id)
    except ValueError as e:
        log.error({"event": "request_error", "error": str(e), "type": "validation"})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error({"event": "request_error", "error": str(e), "type": "internal"})
        raise HTTPException(status_code=500, detail=str(e))

    # Store trace for /traces/{trace_id} lookup
    trace_id = result["trace"].get("trace_id", "")
    _store_trace(trace_id, result["trace"], result.get("journey_type", ""),
                 request.message, result["response"])

    return QueryResponse(
        response=result["response"],
        trace=TraceContext(**result["trace"]),
        journey_type=result.get("journey_type"),
    )


@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    customer_id: Optional[str] = Depends(get_customer_from_token),
):  # noqa: C901
    """
    Streaming version of /query using Server-Sent Events.

    Event types emitted in order:
      {"type": "thinking",   "content": "AI is analysing..."}   — LLM planning started
      {"type": "tool_start", "tool": "get_order_status",
                             "content": "Looking up your order..."} — before each tool runs
      {"type": "token",      "content": " word"}                — final response, word by word
      {"type": "done",       "journey_type": "J1", "trace": {...}} — end of stream
      {"type": "error",      "message": "..."}                  — pipeline error
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if len(request.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400,
                            detail=f"message exceeds {MAX_MESSAGE_LENGTH} character limit")
    if not _SESSION_ID_RE.match(request.session_id):
        raise HTTPException(status_code=400, detail="session_id must be alphanumeric")

    if customer_id:
        update_session(request.session_id, "", "", customer_id=customer_id)

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        result_holder: dict = {}
        error_holder: dict = {}

        async def run_pipeline():
            try:
                result = await run_query(
                    request.message,
                    request.session_id,
                    customer_id,
                    event_queue=queue,
                )
                result_holder["result"] = result
            except Exception as exc:
                error_holder["error"] = exc
            finally:
                # Always push the sentinel so the consumer loop below can exit.
                await queue.put({"type": "__done__"})

        # Run the pipeline concurrently so we can consume events as they arrive.
        pipeline_task = asyncio.create_task(run_pipeline())

        # Minimum time (seconds) a tool_start / thinking status line stays visible.
        # Without this, fast tools (<10 ms) emit and are replaced before the browser
        # renders even a single frame — the user sees nothing.
        STATUS_MIN_VISIBLE_S = 0.45

        # 4 KB SSE comment pad appended to every event.
        # Root cause: uvicorn/OS TCP coalesces small SSE chunks into one burst.
        # Padding each event past the socket-send-buffer threshold forces an
        # immediate TCP flush so the browser reader.read() fires per-event.
        # The ': ...' line is an SSE comment — browsers and our JS ignore it.
        _SSE_PAD = ": " + "p" * 4096 + "\n\n"

        # Consume and forward events until the pipeline signals it's done.
        # For each visible status event we enforce the minimum display time.
        last_status_sent_at = None
        try:
            while True:
                event = await queue.get()
                if event["type"] == "__done__":
                    break

                # If this is a new status event, wait until the previous one
                # has been on screen for at least STATUS_MIN_VISIBLE_S seconds.
                if event["type"] in ("thinking", "tool_start"):
                    if last_status_sent_at is not None:
                        elapsed = asyncio.get_event_loop().time() - last_status_sent_at
                        gap = STATUS_MIN_VISIBLE_S - elapsed
                        if gap > 0:
                            await asyncio.sleep(gap)
                    last_status_sent_at = asyncio.get_event_loop().time()

                yield f"data: {json.dumps(event)}\n\n{_SSE_PAD}"
        except Exception:
            pass

        # Hold the last status line visible before the response starts streaming.
        if last_status_sent_at is not None:
            elapsed = asyncio.get_event_loop().time() - last_status_sent_at
            gap = STATUS_MIN_VISIBLE_S - elapsed
            if gap > 0:
                await asyncio.sleep(gap)

        await pipeline_task  # ensure the task is fully finished

        if error_holder:
            err = error_holder["error"]
            yield f"data: {json.dumps({'type': 'error', 'message': str(err)})}\n\n"
            return

        result = result_holder.get("result", {})
        _store_trace(
            result.get("trace", {}).get("trace_id", ""),
            result.get("trace", {}),
            result.get("journey_type", ""),
            request.message,
            result.get("response", ""),
        )

        # Send the full response text + trace in a single "done" event.
        # The client does a smooth typewriter animation client-side, which is
        # more reliable than server-side line-by-line streaming (which fights
        # TCP coalescing and browser read() buffering).
        _PAD = ": " + "p" * 4096 + "\n\n"   # ~4 KB flush-pad — forces TCP flush
        response_text = result.get("response", "")
        yield (
            f"data: {json.dumps({'type': 'done', 'content': response_text, 'journey_type': result.get('journey_type', ''), 'trace': result.get('trace', {})})}\n\n"
            + _PAD
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
