"""
tools/base.py — base class for every tool in AtlasCare.

All tools inherit TracedTool. The only method you implement in a subclass is
_execute(**kwargs) → dict.  The __call__ wrapper handles:
  - timing (latency_ms)
  - exception catching (success=False)
  - appending a ToolCallRecord to the shared TraceContext so synthesize_from_trace
    can read every tool's output afterwards
"""
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict
from schemas.trace import TraceContext, ToolCallRecord

class TracedTool:
    def __init__(self, trace_ctx: TraceContext):
        self.trace_ctx = trace_ctx
        self.name = self.__class__.__name__  # used as the tool_name in trace records

    def __call__(self, **kwargs) -> Dict[str, Any]:
        # This is what the Executor calls. It wraps _execute with timing + tracing.
        start = time.perf_counter()
        success = True
        result = {}
        
        try:
            result = self._execute(**kwargs)
        except Exception as e:
            success = False
            result = {"error": str(e)}
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            # Serialize input/output to ensure it's JSON serializable for tracing
            # Fallback to string if not serializable
            try:
                clean_input = json.loads(json.dumps(kwargs, default=str))
                clean_output = json.loads(json.dumps(result, default=str))
            except:
                clean_input = {"raw": str(kwargs)}
                clean_output = {"raw": str(result)}
            
            self.trace_ctx.tool_calls.append(
                ToolCallRecord(
                    tool_name=self.name,
                    input=clean_input,
                    output=clean_output,
                    latency_ms=latency_ms,
                    success=success,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            )
            
        if not success:
            raise Exception(result.get("error", "Unknown error in tool execution"))
            
        return result

    def _execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _execute")
