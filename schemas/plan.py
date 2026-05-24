from pydantic import BaseModel
from typing import List, Dict, Any

class PlanStep(BaseModel):
    tool: str
    params: Dict[str, Any]

class ExecutionPlan(BaseModel):
    steps: List[PlanStep]
