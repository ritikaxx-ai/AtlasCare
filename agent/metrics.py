"""
ML Engineering Metrics Collection for AtlasCare
Tracks LLM performance, cost, and quality metrics
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import statistics
import threading


@dataclass
class LLMMetrics:
    """Metrics for a single LLM call"""
    model: str
    operation: str  # "planning" or "synthesis"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    cost_usd: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class JourneyMetrics:
    """Metrics for a complete user journey"""
    journey_type: str  # "J1", "J2", "J3"
    trace_id: str
    session_id: str
    total_latency_ms: int
    num_tool_calls: int
    num_llm_calls: int
    total_tokens: int
    total_cost_usd: float
    success: bool
    timestamp: datetime
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class GuardrailMetrics:
    """Metrics for guardrail activations"""
    triggered: bool
    amount_detected: Optional[float]
    threshold: float
    reason: str
    latency_ms: int
    timestamp: datetime
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class MetricsCollector:
    """Centralized metrics collection for ML observability"""
    
    # Gemini 2.5 Flash pricing (as of 2026)
    INPUT_COST_PER_1M = 0.15  # $0.15 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.60  # $0.60 per 1M output tokens
    
    def __init__(self):
        self.llm_metrics: List[LLMMetrics] = []
        self.journey_metrics: List[JourneyMetrics] = []
        self.guardrail_metrics: List[GuardrailMetrics] = []
        self._lock = threading.Lock()
    
    def record_llm_call(
        self,
        model: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record metrics for an LLM call"""
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self.calculate_cost(prompt_tokens, completion_tokens)
        
        metrics = LLMMetrics(
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self.llm_metrics.append(metrics)
    
    def record_journey(
        self,
        journey_type: str,
        trace_id: str,
        session_id: str,
        total_latency_ms: int,
        num_tool_calls: int,
        num_llm_calls: int,
        total_tokens: int,
        total_cost_usd: float,
        success: bool
    ):
        """Record metrics for a complete journey"""
        metrics = JourneyMetrics(
            journey_type=journey_type,
            trace_id=trace_id,
            session_id=session_id,
            total_latency_ms=total_latency_ms,
            num_tool_calls=num_tool_calls,
            num_llm_calls=num_llm_calls,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            success=success,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self.journey_metrics.append(metrics)
    
    def record_guardrail(
        self,
        triggered: bool,
        amount_detected: Optional[float],
        threshold: float,
        reason: str,
        latency_ms: int
    ):
        """Record guardrail activation"""
        metrics = GuardrailMetrics(
            triggered=triggered,
            amount_detected=amount_detected,
            threshold=threshold,
            reason=reason,
            latency_ms=latency_ms,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self.guardrail_metrics.append(metrics)
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for token usage"""
        input_cost = (prompt_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost
    
    def get_llm_stats(self) -> Dict:
        """Get aggregated LLM statistics"""
        if not self.llm_metrics:
            return {"message": "No LLM metrics collected yet"}
        
        with self._lock:
            total_calls = len(self.llm_metrics)
            successful_calls = sum(1 for m in self.llm_metrics if m.success)
            total_tokens = sum(m.total_tokens for m in self.llm_metrics)
            total_cost = sum(m.cost_usd for m in self.llm_metrics)
            latencies = [m.latency_ms for m in self.llm_metrics]
            
            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "success_rate": successful_calls / total_calls,
                "total_tokens": total_tokens,
                "avg_tokens_per_call": total_tokens / total_calls,
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_call": round(total_cost / total_calls, 4),
                "projected_monthly_cost": round(total_cost * 30 * 24 * 60, 2),  # Extrapolate
                "latency_ms": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99)
                },
                "by_operation": self._get_operation_breakdown()
            }
    
    def get_journey_stats(self) -> Dict:
        """Get aggregated journey statistics"""
        if not self.journey_metrics:
            return {"message": "No journey metrics collected yet"}
        
        with self._lock:
            total_journeys = len(self.journey_metrics)
            successful_journeys = sum(1 for m in self.journey_metrics if m.success)
            latencies = [m.total_latency_ms for m in self.journey_metrics]
            
            return {
                "total_journeys": total_journeys,
                "successful_journeys": successful_journeys,
                "success_rate": successful_journeys / total_journeys,
                "avg_tool_calls": statistics.mean([m.num_tool_calls for m in self.journey_metrics]),
                "avg_llm_calls": statistics.mean([m.num_llm_calls for m in self.journey_metrics]),
                "latency_ms": {
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95)
                },
                "by_type": self._get_journey_type_breakdown()
            }
    
    def get_guardrail_stats(self) -> Dict:
        """Get guardrail activation statistics"""
        if not self.guardrail_metrics:
            return {"message": "No guardrail metrics collected yet"}
        
        with self._lock:
            total_checks = len(self.guardrail_metrics)
            triggered = sum(1 for m in self.guardrail_metrics if m.triggered)
            
            return {
                "total_checks": total_checks,
                "triggered_count": triggered,
                "trigger_rate": triggered / total_checks,
                "avg_latency_ms": statistics.mean([m.latency_ms for m in self.guardrail_metrics]),
                "amounts_detected": [m.amount_detected for m in self.guardrail_metrics if m.amount_detected]
            }
    
    def get_all_stats(self) -> Dict:
        """Get all metrics in one call"""
        return {
            "llm_metrics": self.get_llm_stats(),
            "journey_metrics": self.get_journey_stats(),
            "guardrail_metrics": self.get_guardrail_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_operation_breakdown(self) -> Dict:
        """Break down LLM stats by operation type"""
        breakdown = {}
        for operation in ["planning", "synthesis"]:
            ops = [m for m in self.llm_metrics if m.operation == operation]
            if ops:
                breakdown[operation] = {
                    "count": len(ops),
                    "avg_tokens": statistics.mean([m.total_tokens for m in ops]),
                    "avg_latency_ms": statistics.mean([m.latency_ms for m in ops]),
                    "total_cost_usd": sum([m.cost_usd for m in ops])
                }
        return breakdown
    
    def _get_journey_type_breakdown(self) -> Dict:
        """Break down journey stats by type"""
        breakdown = {}
        for journey_type in ["J1", "J2", "J3", "other"]:
            journeys = [m for m in self.journey_metrics if m.journey_type == journey_type]
            if journeys:
                successful = sum(1 for j in journeys if j.success)
                breakdown[journey_type] = {
                    "count": len(journeys),
                    "success_rate": successful / len(journeys),
                    "avg_latency_ms": statistics.mean([j.total_latency_ms for j in journeys]),
                    "avg_tool_calls": statistics.mean([j.num_tool_calls for j in journeys])
                }
        return breakdown
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def export_metrics(self) -> Dict:
        """Export all metrics for external storage"""
        with self._lock:
            return {
                "llm_metrics": [m.to_dict() for m in self.llm_metrics],
                "journey_metrics": [m.to_dict() for m in self.journey_metrics],
                "guardrail_metrics": [m.to_dict() for m in self.guardrail_metrics]
            }


# Global singleton
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create singleton metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
