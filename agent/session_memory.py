"""
Conversational session memory for AtlasCare.

Tracks entities (order_id, case_id) and turn history per session_id so the
agent can resolve references like "cancel it", "that order", "same one"
across multiple turns — zero extra LLM calls.
"""
import threading
from typing import Optional, List, Tuple

# Reference phrases that signal the customer is pointing at a prior entity
_ORDER_REF_PHRASES = (
    "that order", "the order", "cancel it", "cancel that", "same order",
    "this order", "that one", "it", "the same", "that", "my order",
    "track it", "where is it", "what about it", "refund it",
)

_CASE_REF_PHRASES = (
    "that case", "the case", "my case", "same case", "it", "that one",
    "what happened", "any update", "the same",
)

_sessions: dict = {}
_lock = threading.Lock()


def _empty_session() -> dict:
    return {
        "turns": [],           # list of (user_msg, agent_response)
        "last_order_id": None,
        "last_case_id": None,
        "customer_id": None,
    }


def get_session(session_id: str) -> dict:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _empty_session()
        return _sessions[session_id]


def update_session(
    session_id: str,
    user_message: str,
    agent_response: str,
    order_id: Optional[str] = None,
    case_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> None:
    """Store the completed turn and update tracked entities."""
    with _lock:
        session = _sessions.setdefault(session_id, _empty_session())
        session["turns"].append((user_message, agent_response))
        # Keep last 5 exchanges (10 messages)
        session["turns"] = session["turns"][-5:]
        if order_id:
            session["last_order_id"] = order_id
        if case_id:
            session["last_case_id"] = case_id
        if customer_id:
            session["customer_id"] = customer_id


def resolve_order_id(session_id: str, message: str) -> Optional[str]:
    """
    Return an order ID for this message.
    Priority: explicit ORD-XXXXX in message > reference phrase + session memory.
    """
    from agent.fast_paths import extract_order_id
    explicit = extract_order_id(message)
    if explicit:
        return explicit

    msg = message.lower()
    session = get_session(session_id)
    last = session.get("last_order_id")
    if last and any(phrase in msg for phrase in _ORDER_REF_PHRASES):
        return last
    return None


def resolve_case_id(session_id: str, message: str) -> Optional[str]:
    """
    Return a case ID for this message.
    Priority: explicit CASE-XXXXXX in message > reference phrase + session memory.
    """
    from agent.fast_paths import extract_case_id
    explicit = extract_case_id(message)
    if explicit:
        return explicit

    msg = message.lower()
    session = get_session(session_id)
    last = session.get("last_case_id")
    if last and any(phrase in msg for phrase in _CASE_REF_PHRASES):
        return last
    return None


def get_recent_turns(session_id: str, n: int = 3) -> List[Tuple[str, str]]:
    """Return the last n (user, agent) turn pairs for this session."""
    session = get_session(session_id)
    return session["turns"][-n:]


def clear_session(session_id: str) -> None:
    """Wipe all memory for a session (called on customer switch)."""
    with _lock:
        _sessions.pop(session_id, None)
