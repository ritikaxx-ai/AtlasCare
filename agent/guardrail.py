"""
guardrail.py — first line of defence before any message reaches the LLM or tools.

Two independent checks run in sequence:
  1. check_prompt_injection: regex scan for jailbreak / instruction-override phrases.
     If matched the message is blocked immediately — the LLM never sees it.
  2. check_guardrails: detects high-value refund requests (> ₹25 K) and escalates
     them to a human specialist (J3 journey) instead of auto-processing.
"""
import re
import time
from typing import Optional
from agent.cache import get_data_store
from agent.metrics import get_metrics_collector
from agent.logger import log


# Phrases that indicate an attempt to override the system prompt or bypass safety rules.
# Checked with a simple substring match (case-insensitive) for speed.
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "disregard previous",
    "forget your rules",
    "forget your instructions",
    "you are now",
    "act as if",
    "act as a",
    "pretend you are",
    "pretend to be",
    "system prompt",
    "new instructions:",
    "override instructions",
    "jailbreak",
    "bypass safety",
    "do anything now",
    "dan mode",
]

MAX_MESSAGE_LENGTH = 2000  # messages longer than this are rejected upstream in main.py


def check_prompt_injection(message: str) -> Optional[str]:
    """
    Returns the matched pattern string if injection detected, else None.
    Call this BEFORE check_guardrails — never let injected messages reach the LLM.
    """
    lowered = message.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


# Simple value object returned by check_guardrails.
class GuardrailResult:
    def __init__(self, action: str, reason: str = "", extracted_amount: float = None):
        self.action = action           # "PASS", "ESCALATE", or "INJECTION"
        self.reason = reason           # shown in audit log and CRM case description
        self.extracted_amount = extracted_amount  # rupee figure that triggered the rule


def extract_currency_amount(text: str) -> Optional[float]:
    """
    Extract currency amounts from text with multiple format support.
    Supports: Rs.42000, ₹42,000, INR 42000, 42000 rupees
    """
    # Pattern 1: Currency prefix (Rs., ₹, INR) followed by number
    pattern1 = r'(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{2})?)'
    matches = re.findall(pattern1, text, re.IGNORECASE)
    
    if matches:
        try:
            cleaned = matches[0].replace(",", "")
            return float(cleaned)
        except ValueError:
            pass
    
    # Pattern 2: Number followed by currency suffix (rupees, inr)
    pattern2 = r'([\d,]+(?:\.\d{2})?)\s*(?:rupees?|inr)'
    matches = re.findall(pattern2, text, re.IGNORECASE)

    if matches:
        try:
            cleaned = matches[0].replace(",", "")
            return float(cleaned)
        except ValueError:
            pass

    # Pattern 3: Bare number (≥4 digits) appearing near refund/amount keywords
    # e.g. "refund of 50000" / "amount 30000" — only extract if clearly monetary context
    pattern3 = r'\b([\d,]{4,}(?:\.\d{2})?)\b'
    matches = re.findall(pattern3, text, re.IGNORECASE)
    if matches:
        context_keywords = ("refund of", "amount of", "worth", "value of",
                            "refund for", "claim of", "asking for")
        text_lower = text.lower()
        if any(kw in text_lower for kw in context_keywords):
            try:
                cleaned = matches[0].replace(",", "")
                return float(cleaned)
            except ValueError:
                pass

    return None


def is_refund_intent(text: str) -> bool:
    """Check if message contains refund-related keywords"""
    keywords = ["refund", "return", "money back", "cancel"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def check_guardrails(message: str) -> GuardrailResult:
    """
    Pre-LLM guardrail check for high-value refund detection.
    Prevents unauthorized refunds > ₹25,000.
    """
    start_time = time.perf_counter()
    
    data_store = get_data_store()
    metrics = get_metrics_collector()
    config = data_store.get_payment_config()
    threshold = config.get("auto_refund_limit_inr", 25000)
    
    amount = extract_currency_amount(message)
    
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    
    if amount and amount > threshold and is_refund_intent(message):
        # Record guardrail activation
        metrics.record_guardrail(
            triggered=True,
            amount_detected=amount,
            threshold=threshold,
            reason=f"Amount {amount} exceeds threshold {threshold}",
            latency_ms=latency_ms
        )
        
        return GuardrailResult(
            action="ESCALATE",
            reason=f"Customer requesting refund of Rs.{amount:,.2f}. "
                   f"Exceeds auto-refund threshold. Requires specialist review.",
            extracted_amount=amount
        )
    
    # Record guardrail pass-through
    metrics.record_guardrail(
        triggered=False,
        amount_detected=amount,
        threshold=threshold,
        reason="No threshold breach detected",
        latency_ms=latency_ms
    )
    
    return GuardrailResult(action="PASS")
