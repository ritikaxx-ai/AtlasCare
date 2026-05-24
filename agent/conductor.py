import os
import json
import re
import time
from openai import OpenAI
from schemas.plan import ExecutionPlan, PlanStep
from agent.metrics import get_metrics_collector

def load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r") as f:
        return f.read()

class Conductor:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.metrics = get_metrics_collector()
        
    def generate_plan(self, message: str, session_history: list = None) -> ExecutionPlan:
        system_prompt = load_prompt("system_conductor.txt")
        
        # Build messages array with history
        messages = [{"role": "system", "content": system_prompt}]
        if session_history:
            messages.extend(session_history)
            
        messages.append({"role": "user", "content": f"Customer Message: {message}"})
        
        # We will attempt up to 2 retries if JSON parsing fails
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                content = response.choices[0].message.content
                
                # Record LLM metrics
                usage = response.usage
                self.metrics.record_llm_call(
                    model="gemini-2.5-flash",
                    operation="planning",
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_ms=latency_ms,
                    success=True
                )
                
                # Try to extract JSON if it was wrapped in markdown
                match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    
                plan_data = json.loads(content)
                
                # Normalize schema
                if isinstance(plan_data, list):
                    plan_data = {"steps": plan_data}
                elif "plan" in plan_data:
                    plan_data = {"steps": plan_data["plan"]}
                    
                # Validate using Pydantic
                return ExecutionPlan(**plan_data)
                
            except Exception as e:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                self.metrics.record_llm_call(
                    model="gemini-2.5-flash",
                    operation="planning",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error_message=str(e)
                )
                last_error = e
                # Feed error back for retry
                messages.append({"role": "assistant", "content": content if 'content' in locals() else ""})
                messages.append({"role": "user", "content": f"INVALID JSON PLAN. Please fix the error: {e}. Return ONLY valid JSON."})
                
        raise ValueError(f"Failed to generate execution plan after {max_retries} retries: {last_error}")

