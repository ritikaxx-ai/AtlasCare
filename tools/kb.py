from typing import Any, Dict, List
from .base import TracedTool
from schemas.trace import TraceContext
from agent.cache import get_data_store


class search_kb(TracedTool):
    """Search knowledge base from in-memory cache"""
    
    def _execute(self, tags: List[str], **kwargs) -> Dict[str, Any]:
        data_store = get_data_store()
        results = data_store.search_kb(tags)
        return {"articles": results}
