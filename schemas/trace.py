from pydantic import BaseModel
from typing import List, Any, Dict

class ToolCallRecord(BaseModel):
    tool_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    latency_ms: int
    success: bool
    timestamp: str

class TraceContext(BaseModel):
    trace_id: str
    session_id: str
    latency_ms: int = 0
    tool_calls: List[ToolCallRecord] = []
