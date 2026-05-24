from typing import Optional
from pydantic import BaseModel
from .trace import TraceContext

class QueryRequest(BaseModel):
    message: str
    session_id: str
    customer_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    trace: TraceContext
    journey_type: Optional[str] = None
