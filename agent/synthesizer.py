import os
import time
from openai import OpenAI
from schemas.trace import TraceContext
from agent.metrics import get_metrics_collector

def load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r") as f:
        return f.read()

class Synthesizer:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.metrics = get_metrics_collector()
        
    def generate_response(self, message: str, trace_ctx: TraceContext) -> str:
        system_prompt = load_prompt("system_synthesizer.txt")
        
        # Prepare context from trace for the LLM
        trace_summary = []
        for call in trace_ctx.tool_calls:
            status = "SUCCESS" if call.success else "FAILED"
            trace_summary.append(f"Action: {call.tool_name} | Status: {status} | Output: {call.output}")
            
        trace_text = "\n".join(trace_summary)
        
        user_content = (
            f"Customer Message: {message}\n\n"
            f"Backend Actions Taken:\n{trace_text}"
        )
        
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.4
            )
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            usage = response.usage
            
            self.metrics.record_llm_call(
                model="gemini-2.5-flash",
                operation="synthesis",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                success=True
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self.metrics.record_llm_call(
                model="gemini-2.5-flash",
                operation="synthesis",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e)
            )
            return "I apologize, but I am currently unable to process your request. Our support team will reach out to you."
