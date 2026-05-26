from typing import Optional
from pydantic import BaseModel
from .trace import TraceContext

class QueryRequest(BaseModel):
    message: str
    session_id: str
    # customer_id is no longer accepted in the request body.
    # It is extracted server-side from the Authorization: Bearer <JWT> header.

class QueryResponse(BaseModel):
    response: str
    trace: TraceContext
    journey_type: Optional[str] = None
