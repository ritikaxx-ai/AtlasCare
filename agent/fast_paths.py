"""
agent/fast_paths.py — deterministic planning and template-based response synthesis.

This file does two separate jobs:

1. PLANNING (build_jN_plan / try_build_j2_plan / classify_journey_for_routing)
   Each "build" function constructs an ExecutionPlan — a list of tool steps —
   without calling any LLM. The router picks the right builder based on journey type.

2. SYNTHESIS (synthesize_from_trace)
   After the executor runs all tools, this function turns the raw tool outputs
   into a customer-friendly text response using static templates.
   It never calls an LLM; all data comes directly from tool outputs stored in the trace.

Journey types and their builders:
  J1  → build_j1_plan    — order tracking (get_order_status)
  J2  → try_build_j2_plan — cancel / refund / address update
  J3  → build_j3_plan    — high-value escalation (create_crm_case)
  J4  → build_j4_plan    — customer history RAG (get_customer_interaction_history)
  J5  → build_j5_plan    — CRM case status lookup (get_case_status)
  J-KB → build_kb_plan   — policy questions (search_kb)
"""
import re
from typing import Optional

from schemas.plan import ExecutionPlan, PlanStep
from schemas.trace import TraceContext
from agent.cache import get_data_store


def extract_order_id(text: str) -> Optional[str]:
    match = re.search(r"(ORD-\d{5})", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_case_id(text: str) -> Optional[str]:
    match = re.search(r"(CASE-[A-Z0-9]{3,8})", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_customer_id_from_order(order_id: str) -> str:
    try:
        order = get_data_store().get_order(order_id)
        return order["customer_id"] if order else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def extract_address_label(msg: str) -> Optional[str]:
    """
    Parse which saved address label the customer wants to switch to.
    Looks for known labels (home, office, work) mentioned before or after 'address'.
    Returns the label string if found, else None.
    """
    msg_lower = msg.lower()
    # Check explicit label keywords
    for label in ("home", "office", "work"):
        if label in msg_lower:
            return label
    # Pattern: "my <word> address" — e.g. "my billing address"
    match = re.search(r"\bmy\s+(\w+)\s+address\b", msg_lower)
    if match:
        return match.group(1)
    return None


def resolve_address(cid: str, msg: str) -> dict:
    """
    Try to resolve a saved address for the customer from the message label.
    Returns the address dict if found, or an empty dict if clarification needed.
    """
    store = get_data_store()
    label = extract_address_label(msg)
    if label:
        try:
            addr = store.get_customer_address(cid, label)
            if addr:
                return addr
        except Exception:
            pass
    return {}


def _get_available_address_labels(cid: str) -> list:
    """Return list of saved address labels for this customer."""
    try:
        store = get_data_store()
        customer = store.get_customer(cid)
        if customer:
            return [a.get("label") for a in customer.get("addresses", []) if a.get("label")]
    except Exception:
        pass
    return []


_HISTORY_KEYWORDS = (
    # explicit references to prior contact
    "history",
    "previous",
    "last time",
    "last complaint",
    "called before",
    "called about",
    "i called",
    "calling again",
    "i reported",
    "i complained",
    "i raised",
    "i mentioned",
    "i spoke",
    "third time",
    "same issue",
    "again about",
    "repeat",
    "earlier conversation",
    "past interaction",
    "what happened before",
    "last call",
    "spoke to",
    "talked to support",
    "follow up on",
    "following up",
    "reached out",
    "got in touch",
    # meta-questions about prior interactions
    "last inquiry",
    "last issue",
    "last request",
    "last question",
    "last chat",
    "last ticket",
    "last contact",
    "what did i ask",
    "what was my last",
    "what was i asking",
    "what did i report",
    "previous inquiry",
    "previous issue",
    "previous request",
    "previous complaint",
    "my inquiry",
    "my complaint",
    "my request",
    "any update",
    "any updates",
    "update on my",
    "status of my case",
    "status of my complaint",
    "did you fix",
    "has it been resolved",
    "was it resolved",
)

_POLICY_KEYWORDS = (
    # explicit policy questions
    "policy", "policies",
    "can i return", "return policy",
    "refund policy", "refund limit",
    "what's the refund", "what is the refund",
    "return window", "how long to return",
    # exchange / replacement
    "exchange policy", "replacement policy",
    "can i exchange", "can i replace",
    # warranty / damage
    "warranty", "damaged item policy",
    # shipping
    "shipping policy", "delivery policy",
    "how long does shipping", "how long does delivery",
    # cancellation rules (policy intent, not action)
    "cancellation policy", "can i cancel",
    # general knowledge queries
    "what are the rules", "what are your rules",
    "tell me about", "list policy", "list policies",
    "explain policy", "what is your policy",
    "how does return", "how does refund",
    "eligible for", "am i eligible",
)


# Dictionary of words that are purely social/greeting — no support intent.
# We tokenise the message and check that EVERY token is in this set.
# This handles combos like "Thankyou, Bye" or "Hi! Good morning." naturally.
_GREETING_WORDS = {
    # Hellos
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    # Good <time>
    "good", "morning", "afternoon", "evening", "night", "day",
    # Thanks
    "thank", "thankyou", "thanks", "thx", "ty", "you", "u",
    # Acknowledgements
    "ok", "okay", "sure", "noted", "got", "it", "alright", "right",
    # Goodbyes
    "bye", "goodbye", "cya", "later", "see", "take", "care", "farewell",
    # Positive filler
    "great", "awesome", "perfect", "wonderful", "nice", "cool", "sounds",
    # Yes / No
    "yes", "no", "yep", "nope", "nah", "yup", "yeah",
    # Pleasantries
    "welcome", "please", "sorry", "excuse", "me", "pardon",
}

# Words that signal real support intent — even inside an otherwise greeting-y message.
# "hi cancel my order" → "cancel" is here → NOT a greeting.
_SUPPORT_INTENT_WORDS = {
    "order", "cancel", "refund", "track", "status", "address", "deliver",
    "return", "exchange", "warranty", "policy", "case", "item", "product",
    "payment", "ship", "shipping", "damaged", "wrong", "missing", "lost",
    "where", "when", "how", "what", "why", "help", "issue", "problem",
}


def is_greeting(message: str) -> bool:
    """
    Return True if the message is purely social/greeting with no support intent.

    Strategy:
    1. Tokenise (strip punctuation, lowercase, split on whitespace).
    2. If any token is a known support-intent word → False immediately.
    3. If every token is in the greeting-words dictionary → True.
    4. Otherwise → False (let Groq handle it).
    """
    # Strip punctuation and normalise
    cleaned = re.sub(r"[^\w\s]", " ", message.lower())
    tokens = [t for t in cleaned.split() if t]

    if not tokens:
        return False  # empty message → let Groq handle

    # Hard veto: any support-intent word present → definitely not a greeting
    if any(t in _SUPPORT_INTENT_WORDS for t in tokens):
        return False

    # Soft check: all tokens must be in the greeting dictionary
    return all(t in _GREETING_WORDS for t in tokens)


def needs_customer_history(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in _HISTORY_KEYWORDS)


def needs_policy_lookup(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in _POLICY_KEYWORDS)


def extract_customer_id(message: str) -> Optional[str]:
    match = re.search(r"(CUST-\d{3})", message, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    order_id = extract_order_id(message)
    if order_id:
        return extract_customer_id_from_order(order_id)

    return None


def _kb_tags_for_message(message: str) -> list[str]:
    msg = message.lower()

    # Generic "list all policies" intent — return everything
    _list_all_triggers = (
        "list policy", "list policies", "all policies", "all policy",
        "what are your policies", "what policies", "what are the policies",
        "show policy", "show policies", "what are the rules",
        "what are your rules",
    )
    if any(t in msg for t in _list_all_triggers):
        return []  # empty → cache layer returns all articles

    # Specific topic matching
    if "exchange" in msg:
        return ["exchange"]
    if "warranty" in msg:
        return ["warranty"]
    if "damage" in msg or "defective" in msg or "doa" in msg:
        return ["damaged_product", "defective"]
    if "cod" in msg or "cash on delivery" in msg:
        return ["cod"]
    if "tier" in msg or "loyalty" in msg or "gold" in msg or "platinum" in msg or "silver" in msg:
        return ["tier"]
    if "address" in msg and ("transit" in msg or "shipped" in msg):
        return ["address_change", "in_transit"]
    if "cancel" in msg and "partial" in msg:
        return ["cancellation", "partial"]
    if "cancel" in msg:
        return ["cancellation"]
    if "track" in msg or "shipping" in msg or "delivery" in msg:
        return ["tracking", "shipping", "delivery"]
    if "payment" in msg or "upi" in msg or "credit" in msg or "debit" in msg:
        return ["payment"]
    if "escalat" in msg or "sla" in msg or "specialist" in msg:
        return ["escalation", "sla"]
    if "return" in msg:
        return ["return", "window"]   # only KB-002 has both tags
    if "refund" in msg and ("limit" in msg or "threshold" in msg or "policy" in msg):
        return ["refund", "threshold"]
    if "refund" in msg:
        return ["refund", "threshold"]

    # Generic policy question — return all
    return []


def is_case_status_query(message: str) -> bool:
    """True when the customer is asking about a specific CASE-XXXXXX they already have."""
    msg = message.lower()
    has_case_id = bool(extract_case_id(message))
    status_words = ("status", "update", "what happened", "any news", "follow up",
                    "resolved", "progress", "check my case", "case status")
    return has_case_id and any(w in msg for w in status_words)


def classify_journey_for_routing(
    message: str, guardrail_action: str, resolved_order_id: Optional[str] = None
) -> str:
    """
    Priority order (highest wins):
      J3  — guardrail already flagged high-value refund
      J5  — message contains a CASE-XXXXXX ID + status word
      J4  — message mentions prior interactions ("I called", "follow up", etc.)
      J-KB — policy/returns/refund-policy question (but NOT a refund action)
      J1  — has order ID + tracking intent, no cancel/refund words
      J2  — default for cancel/refund/address/shipping actions
    """
    if guardrail_action == "ESCALATE":
        return "J3"

    # J5 must be checked before J3/J2 — a case reference + status word is never an escalation
    if is_case_status_query(message):
        return "J5"

    msg = message.lower()
    # Use resolved order (from session memory) if no explicit order in message
    has_order = bool(extract_order_id(message) or resolved_order_id)

    if needs_customer_history(message):
        return "J4"

    if needs_policy_lookup(message):
        return "J-KB"

    tracking_intent = any(
        k in msg for k in ("where is", "track", "tracking", "status of my order", "order status",
                           "where is it", "has it shipped", "delivery status")
    )
    compound_intent = any(
        k in msg
        for k in ("cancel", "refund", "ship remainder", "office address", "update address", "damaged")
    )

    if has_order and tracking_intent and not compound_intent:
        return "J1"

    if any(k in msg for k in ("cancel", "refund", "ship", "office", "address")):
        return "J2"

    # Fallback: if there's a resolved order and no compound intent, treat as tracking
    if resolved_order_id and not compound_intent:
        return "J1"

    return "J2"


def _ownership_plan(order_id: str, customer_id: Optional[str]) -> Optional[ExecutionPlan]:
    """
    Cross-customer access guard. Returns a plan that renders an "unauthorized" message
    if the logged-in customer doesn't own this order. Called at the start of J1, J2, J5.
    Returns None (no restriction) if customer_id is missing (unauthenticated API call).
    """
    if not customer_id:
        return None  # No session customer → no restriction (e.g. API calls without login)
    order = get_data_store().get_order(order_id)
    if order and order.get("customer_id") != customer_id:
        return ExecutionPlan(
            steps=[PlanStep(tool="unauthorized_order_access", params={"order_id": order_id})]
        )
    return None


def build_j1_plan(
    message: str,
    customer_id: Optional[str] = None,
    resolved_order_id: Optional[str] = None,
) -> ExecutionPlan:
    order_id = extract_order_id(message) or resolved_order_id
    if not order_id:
        raise ValueError("Order ID required for tracking queries")
    blocked = _ownership_plan(order_id, customer_id)
    if blocked:
        return blocked
    return ExecutionPlan(
        steps=[PlanStep(tool="get_order_status", params={"order_id": order_id})]
    )


def try_build_j2_plan(
    message: str,
    customer_id: Optional[str] = None,
    resolved_order_id: Optional[str] = None,
) -> Optional[ExecutionPlan]:
    """
    Deterministic J2 planner. Covers four sub-cases by reading keywords:

      • "item N" + cancel → cancel_order_item + execute_refund for that line
      • "cancel" (no item) → cancel_full_order + execute_refund for all active items
      • address keyword only → update_shipping_address (or ask for clarification)
      • cancel/refund + address keyword → cancel + refund + update_shipping_address

    Two guardrail checks inside:
      - If refund amount > ₹25K, escalate to a CRM case (create_crm_case) instead.
      - If order is already delivered/cancelled, surface status so synthesis can explain.

    Returns None when the intent can't be determined (caller falls back to clarify_order_id).
    """
    msg = message.lower()
    has_cancel = "cancel" in msg
    has_refund = "refund" in msg

    # Need at least cancel or refund intent
    if not has_cancel and not has_refund and not any(k in msg for k in ("ship", "office", "address")):
        return None

    order_id = extract_order_id(message) or resolved_order_id
    if not order_id:
        return None

    store = get_data_store()
    order = store.get_order(order_id)

    # --- Ownership check ---
    if customer_id and order and order.get("customer_id") != customer_id:
        return ExecutionPlan(
            steps=[PlanStep(tool="unauthorized_order_access", params={"order_id": order_id})]
        )

    if order and order.get("status") in ("delivered", "cancelled"):
        # Order not eligible — surface the real status so synthesis can explain why
        return ExecutionPlan(
            steps=[PlanStep(tool="get_order_status", params={"order_id": order_id})]
        )

    if not order:
        return None

    method_match = re.search(
        r"\b(HDFC_CREDIT|UPI|DEBIT_CARD|CREDIT_CARD)\b", message, re.IGNORECASE
    )
    method = method_match.group(1).upper() if method_match else (
        order.get("payment_method", "HDFC_CREDIT")
    )

    line_match = re.search(r"item\s+(\d+)", message, re.IGNORECASE)

    if line_match:
        # --- Single item cancellation ---
        line_id = int(line_match.group(1))
        refund_amount = 1500.0
        for item in order.get("items", []):
            if item.get("line_id") == line_id and item.get("status") != "cancelled":
                refund_amount = float(item.get("unit_price", 1500.0))
                break

        # Guardrail: high-value item must go to a specialist
        payment_config = store.get_payment_config()
        auto_limit = float(payment_config.get("auto_refund_limit_inr", 25000))
        if refund_amount > auto_limit:
            cid = order.get("customer_id") or extract_customer_id_from_order(order_id)
            return ExecutionPlan(steps=[
                PlanStep(
                    tool="create_crm_case",
                    params={
                        "customer_id": cid,
                        "order_id": order_id,
                        "description": (
                            f"Customer requested cancellation of item {line_id} with refund of "
                            f"₹{refund_amount:,.0f}, which exceeds the ₹{auto_limit:,.0f} "
                            f"auto-refund limit. Manual review required."
                        ),
                        "priority": "high",
                        "amount_inr": refund_amount,
                    },
                )
            ])

        steps = [
            PlanStep(
                tool="cancel_order_item",
                params={"order_id": order_id, "line_id": line_id},
            ),
            PlanStep(
                tool="execute_refund",
                params={
                    "order_id": order_id,
                    "amount_inr": refund_amount,
                    "method": method,
                },
            ),
        ]
    elif has_cancel:
        # --- Full order cancellation (no line item specified) ---
        active_items = [i for i in order.get("items", []) if i.get("status") != "cancelled"]
        refund_amount = sum(float(i.get("unit_price", 0)) * int(i.get("quantity", 1))
                           for i in active_items)

        # Guardrail: high-value orders must be handled by a specialist
        payment_config = store.get_payment_config()
        auto_limit = float(payment_config.get("auto_refund_limit_inr", 25000))
        if refund_amount > auto_limit:
            cid = order.get("customer_id") or extract_customer_id_from_order(order_id)
            return ExecutionPlan(steps=[
                PlanStep(
                    tool="create_crm_case",
                    params={
                        "customer_id": cid,
                        "order_id": order_id,
                        "description": (
                            f"Customer requested full order cancellation with refund of "
                            f"₹{refund_amount:,.0f}, which exceeds the ₹{auto_limit:,.0f} "
                            f"auto-refund limit. Manual review required."
                        ),
                        "priority": "high",
                        "amount_inr": refund_amount,
                    },
                )
            ])

        steps = [
            PlanStep(
                tool="cancel_full_order",
                params={"order_id": order_id},
            ),
            PlanStep(
                tool="execute_refund",
                params={
                    "order_id": order_id,
                    "amount_inr": refund_amount,
                    "method": method,
                },
            ),
        ]
    elif any(k in msg for k in ("office", "home", "work", "ship", "address")):
        # --- Standalone address update (no cancel/refund) ---
        cid = order.get("customer_id") or extract_customer_id_from_order(order_id)
        label = extract_address_label(msg)
        if not label or not resolve_address(cid, msg):
            # No saved address matched — ask the customer to specify
            return ExecutionPlan(steps=[
                PlanStep(
                    tool="address_clarification_needed",
                    params={
                        "order_id": order_id,
                        "customer_id": cid,
                        "available_labels": _get_available_address_labels(cid),
                    },
                )
            ])
        return ExecutionPlan(steps=[
            PlanStep(
                tool="update_shipping_address",
                params={"order_id": order_id, "customer_id": cid, "address_label": label},
            )
        ])

    else:
        # Unknown J2 intent — surface order status so customer sees their order
        return ExecutionPlan(
            steps=[PlanStep(tool="get_order_status", params={"order_id": order_id})]
        )

    # Append address update if cancel/refund AND address change both requested
    if any(k in msg for k in ("office", "home", "work", "ship", "address")):
        cid = order.get("customer_id") or extract_customer_id_from_order(order_id)
        label = extract_address_label(msg)
        if label and resolve_address(cid, msg):
            steps.append(
                PlanStep(
                    tool="update_shipping_address",
                    params={"order_id": order_id, "customer_id": cid, "address_label": label},
                )
            )
        else:
            steps.append(
                PlanStep(
                    tool="address_clarification_needed",
                    params={
                        "order_id": order_id,
                        "customer_id": cid,
                        "available_labels": _get_available_address_labels(cid),
                    },
                )
            )

    return ExecutionPlan(steps=steps)


def build_j4_plan(message: str, customer_id: Optional[str] = None) -> ExecutionPlan:
    """RAG over CRM interaction history — only when customer context is needed."""
    # Prefer session customer_id (from login modal), fall back to extracting from message
    cid = customer_id or extract_customer_id(message)
    if not cid or cid == "UNKNOWN":
        return ExecutionPlan(
            steps=[PlanStep(tool="clarify_customer_id", params={})]
        )

    return ExecutionPlan(
        steps=[
            PlanStep(
                tool="get_customer_interaction_history",
                params={"customer_id": cid, "query": message, "top_k": 3},
            )
        ]
    )


def build_kb_plan(message: str) -> ExecutionPlan:
    """Deterministic KB tag search for policy questions."""
    tags = _kb_tags_for_message(message)
    return ExecutionPlan(
        steps=[PlanStep(tool="search_kb", params={"tags": tags})]
    )


def build_j5_plan(
    message: str,
    customer_id: Optional[str] = None,
    resolved_case_id: Optional[str] = None,
) -> ExecutionPlan:
    """Lookup an existing CRM case by ID — no new case created."""
    case_id = extract_case_id(message) or resolved_case_id
    if not case_id:
        raise ValueError("Case ID (CASE-XXXXXX) required for case status queries")
    # Ownership check: ensure the case belongs to this customer
    if customer_id:
        case = get_data_store().get_case_by_id(case_id)
        case_owner = case.get("customer_id") if case else None
        # Allow access if: case has a real owner that matches, OR owner is UNKNOWN/system-created
        if case and case_owner and case_owner != "UNKNOWN" and case_owner != customer_id:
            return ExecutionPlan(
                steps=[PlanStep(tool="unauthorized_order_access", params={"order_id": case_id})]
            )
    return ExecutionPlan(
        steps=[PlanStep(tool="get_case_status", params={"case_id": case_id})]
    )


def build_j3_plan(
    message: str,
    reason: str,
    amount: Optional[float],
    resolved_order_id: Optional[str] = None,
) -> ExecutionPlan:
    order_id = extract_order_id(message) or resolved_order_id or "UNKNOWN"
    customer_id = extract_customer_id_from_order(order_id)
    return ExecutionPlan(
        steps=[
            PlanStep(
                tool="create_crm_case",
                params={
                    "customer_id": customer_id,
                    "order_id": order_id,
                    "description": reason,
                    "priority": "high",
                    "amount_inr": amount,
                },
            )
        ]
    )


def synthesize_from_trace(message: str, trace: TraceContext, journey_type: str) -> str:
    """
    Turns raw tool outputs into a customer-facing text response. No LLM — pure templates.

    Iterates over every ToolCallRecord in the trace in execution order.
    For each tool name there is a dedicated branch that reads the specific output keys
    that tool returns (e.g. "status", "cancelled_count", "amount_inr") and formats
    them into a sentence. Unrecognised tool names are silently skipped.

    A journey-specific closing line is appended at the end (e.g. "Anything else?" for J2).
    """
    if not trace.tool_calls:
        return (
            "I'm sorry, I wasn't able to process your request at this time.\n"
            "One of our support specialists will follow up with you shortly."
        )

    parts: list[str] = []

    for call in trace.tool_calls:
        if not call.success:
            parts.append(
                "I'm sorry, something went wrong while processing part of your request.\n"
                "Our team has been notified and will reach out to you soon to sort this out."
            )
            continue

        out = call.output
        name = call.tool_name

        # ── J1: Order tracking ────────────────────────────────────────────
        if name == "get_order_status":
            order_id = out.get("order_id", "your order")
            if out.get("not_found"):
                parts.append(
                    f"I looked everywhere but couldn't find an order with the ID {order_id}.\n"
                    f"Could you double-check the order number and try again?\n"
                    f"If you think this is a mistake, please reach out to us with your registered email or phone number and we'll sort it out."
                )
                continue

            status = out.get("status", "unknown")

            # J2 ineligible — order already delivered
            if journey_type == "J2" and status == "delivered":
                parts.append(
                    f"I can see that order {order_id} has already been delivered, so unfortunately it's not possible to cancel or modify it at this stage.\n"
                    f"That said, if there's an issue with what you received — such as a damaged or incorrect item — you can raise a return or refund request and we'll take care of it.\n"
                    f"You have up to 30 days from the delivery date to do so."
                )
                continue

            # J2 ineligible — order already cancelled
            if journey_type == "J2" and status == "cancelled":
                parts.append(
                    f"It looks like order {order_id} has already been cancelled.\n"
                    f"If a refund is due, it should reach your original payment method within 3 to 5 business days.\n"
                    f"If you haven't received it yet or need further help, just let me know."
                )
                continue

            # Active order statuses
            status_descriptions = {
                "processing": "is currently being processed and will be dispatched soon",
                "shipped": "is on its way to you",
                "out_for_delivery": "is out for delivery today",
                "delivered": "has been successfully delivered",
                "cancelled": "has been cancelled",
            }
            status_text = status_descriptions.get(status, f"is currently {status}")
            lines = [f"Your order {order_id} {status_text}."]
            if status not in ("cancelled", "delivered"):
                if out.get("tracking_number"):
                    lines.append(f"You can track it using tracking number: {out['tracking_number']}")
                if out.get("estimated_delivery"):
                    lines.append(f"Estimated delivery date: {out['estimated_delivery']}")
            parts.append("\n".join(lines))

        # ── Clarification sentinels ───────────────────────────────────────
        elif name == "greeting":
            msg = message.lower()
            if any(w in msg for w in ("thank", "thanks", "thx", "ty")):
                parts.append(
                    "You're welcome! 😊 Is there anything else I can help you with?\n"
                    "I'm here for order tracking, cancellations, refunds, or any policy questions."
                )
            elif any(w in msg for w in ("bye", "goodbye", "see you", "take care")):
                parts.append(
                    "Goodbye! Have a great day! 👋\n"
                    "Feel free to reach out anytime if you need help with your orders."
                )
            else:
                parts.append(
                    "Hello! Welcome to AtlasCare Support. 👋\n"
                    "How can I help you today? I can assist with:\n"
                    "• Order tracking & status\n"
                    "• Cancellations & refunds\n"
                    "• Delivery address updates\n"
                    "• Return & exchange policies"
                )

        elif name == "clarify_order_id":
            parts.append(
                "I'd love to help you with that!\n"
                "Could you please share your order ID? It usually looks something like ORD-78321.\n"
                "You can find it in your order confirmation email."
            )

        elif name == "blocked_injection":
            parts.append(
                "I'm sorry, I wasn't able to understand that message.\n"
                "Could you rephrase your question? I'm here to help with order tracking, cancellations, refunds, and policies."
            )

        elif name == "clarify_customer_id":
            parts.append(
                "I'd be happy to look up your support history!\n"
                "It seems I don't have your account linked in this session.\n"
                "Could you please log in or share your customer ID (it looks like CUST-001) so I can pull up your records?"
            )

        elif name == "unauthorized_order_access":
            order_id = out.get("order_id", "that order")
            parts.append(
                f"I wasn't able to access order {order_id} — it doesn't appear to be linked to your account.\n"
                f"Please double-check the order ID.\n"
                f"If you believe this is an error, our support team will be happy to investigate for you."
            )

        # ── J2: Cancellation ─────────────────────────────────────────────
        elif name == "cancel_full_order":
            order_id = out.get("order_id", "your order")
            cancelled_count = out.get("cancelled_count", 0)
            if cancelled_count == 0:
                parts.append(
                    f"It looks like there are no active items left on order {order_id} to cancel.\n"
                    f"The order may have already been cancelled or fully processed."
                )
            else:
                parts.append(
                    f"Done! Order {order_id} has been fully cancelled.\n"
                    f"All {cancelled_count} item(s) have been marked as cancelled."
                )

        elif name == "cancel_order_item":
            order_id = out.get("order_id", call.input.get("order_id", "your order"))
            # line_id lives in the tool input (output is the raw order dict, not a line summary)
            line_id = call.input.get("line_id", out.get("line_id", ""))
            if out.get("already_cancelled"):
                parts.append(
                    f"It looks like item {line_id} on order {order_id} was already cancelled previously.\n"
                    f"No further action is needed on our end."
                )
            else:
                parts.append(
                    f"Item {line_id} on order {order_id} has been successfully cancelled."
                )

        # ── J2: Refund ───────────────────────────────────────────────────
        elif name == "execute_refund":
            amt = (
                out.get("amount_inr")
                or out.get("refunded_amount_inr")
                or out.get("amount_refunded")
            )
            method = out.get("method") or out.get("payment_method", "your original payment method")
            sla = out.get("sla_days", 5)
            if amt is not None:
                parts.append(
                    f"Your refund of Rs.{float(amt):,.0f} has been successfully initiated.\n"
                    f"It will be credited back to {method}.\n"
                    f"Please allow up to {sla} business days for the amount to reflect in your account."
                )
            else:
                parts.append(
                    "Your refund has been initiated successfully.\n"
                    "It should reflect in your account within a few business days."
                )

        # ── J2: Address update ───────────────────────────────────────────
        elif name == "address_clarification_needed":
            order_id = out.get("order_id", "your order")
            raw_labels = out.get("available_labels", [])
            # Guard: Gemini sometimes passes a plain string instead of a list.
            # Wrap it so the join below never iterates characters.
            if isinstance(raw_labels, str):
                labels = [raw_labels] if raw_labels else []
            else:
                labels = [l for l in raw_labels if l]  # filter out empty/None

            if labels:
                options = " or ".join(f'"{l}"' for l in labels)
                parts.append(
                    f"Sure, I can update the delivery address for order {order_id}!\n"
                    f"I found these saved addresses on your account: {options}.\n"
                    f"Which one would you like to use? Or just type out the new address and I'll update it straight away."
                )
            else:
                # No saved addresses on file — ask the customer to provide one
                parts.append(
                    f"Sure, I can update the delivery address for order {order_id}!\n"
                    f"I don't see any saved addresses on your account yet.\n"
                    f"Please share the full new address (street, city, state, and pincode) and I'll get it updated right away."
                )

        elif name == "update_shipping_address":
            order_id = out.get("order_id", "your order")
            out_addr = out.get("shipping_address", {})
            addr_parts = [
                out_addr.get("line1"), out_addr.get("line2"),
                out_addr.get("city"), out_addr.get("state"), out_addr.get("pincode")
            ]
            addr_str = ", ".join(filter(None, addr_parts)) if out_addr else ""
            lines = [f"The delivery address for order {order_id} has been updated successfully."]
            if addr_str:
                lines.append(f"Your order will now be shipped to: {addr_str}")
            parts.append("\n".join(lines))

        # ── J3 / J2 high-value: CRM escalation ──────────────────────────
        elif name == "create_crm_case":
            case_id = out.get("case_id", "your case")
            amount = out.get("amount_inr") or out.get("amount_refunded")
            is_duplicate = out.get("deduplicated", False)

            if is_duplicate:
                # Case already existed — reassure the customer without creating confusion
                parts.append(
                    f"It looks like a case is already open for this request — your reference number is {case_id}.\n"
                    f"Our team is already on it and will get back to you within 24 hours.\n"
                    f"No need to raise another case."
                )
            elif amount and float(amount) > 25000:
                parts.append(
                    f"I completely understand how important this is, and I want to make sure it's handled properly.\n"
                    f"Since the refund amount of Rs.{float(amount):,.0f} is above our automated processing limit of Rs.25,000, "
                    f"this needs to be reviewed by one of our specialists.\n"
                    f"I've raised a priority case for you — your reference number is {case_id}.\n"
                    f"A team member will personally review this and get back to you within 24 hours."
                )
            else:
                parts.append(
                    f"I've escalated your request to one of our specialists who will look into this for you.\n"
                    f"Your case reference number is {case_id} — please keep this handy.\n"
                    f"You can expect a response within 24 hours."
                )

        elif name in ("get_customer_profile", "get_customer_address"):
            pass  # intermediate steps — output covered by update_shipping_address

        # ── J-KB: Policy lookup ──────────────────────────────────────────
        elif name == "search_kb":
            articles = out.get("articles", [])
            if articles:
                if len(articles) == 1:
                    article = articles[0]
                    parts.append(
                        f"Here's what our policy says about that:\n\n"
                        f"{article.get('title', 'Policy')}\n"
                        f"{article.get('content', '')}"
                    )
                else:
                    section_lines = ["Here's a summary of the relevant policies for you:\n"]
                    for article in articles:
                        section_lines.append(f"{article.get('title', 'Policy')}")
                        section_lines.append(f"{article.get('content', '')}")
                        section_lines.append("")  # blank line between articles
                    parts.append("\n".join(section_lines).rstrip())
            else:
                parts.append(
                    "I wasn't able to find a specific policy entry for that topic.\n"
                    "Please feel free to ask in a different way, or I can connect you with a specialist who can help."
                )

        # ── J5: Case status ──────────────────────────────────────────────
        elif name == "get_case_status":
            if out.get("not_found"):
                cid = out.get("case_id", "that case")
                parts.append(
                    f"I wasn't able to find a case with the ID {cid} in our system.\n"
                    f"Could you double-check the reference number?\n"
                    f"If you have the order ID handy, I can try looking it up that way instead."
                )
                continue

            status = out.get("status", "unknown")
            priority = out.get("priority", "")
            case_id = out.get("case_id", "your case")
            description = out.get("description", "")
            created_at = (out.get("created_at") or "")[:10]
            amount = out.get("amount_inr")

            status_descriptions = {
                "open": "is currently open and being reviewed by our specialist team",
                "resolved": "has been resolved",
                "closed": "has been closed",
            }
            status_text = status_descriptions.get(status, f"is currently {status}")

            lines = [f"I found your case! Case {case_id} {status_text}."]
            if created_at:
                lines.append(f"It was opened on {created_at}.")
            if amount:
                lines.append(f"The amount involved is Rs.{float(amount):,.0f}.")
            if description:
                lines.append(f"Details on file: {description}")
            if priority == "high" and status == "open":
                lines.append("Given the priority of this case, our team will respond within the 24-hour SLA.")
            parts.append("\n".join(lines))

        # ── J4: Customer history ─────────────────────────────────────────
        elif name == "get_customer_interaction_history":
            interactions = out.get("interactions", [])
            if interactions:
                lines = ["I found your previous interactions with us. Here's a summary:\n"]
                for i, interaction in enumerate(interactions, 1):
                    ts = interaction.get("timestamp", "")[:10]
                    summary = interaction.get("summary", "No summary available")
                    resolution = interaction.get("resolution", "unknown")
                    lines.append(f"{i}.  {ts}  —  {resolution.capitalize()}")
                    lines.append(f"    {summary}")
                    lines.append("")
                parts.append("\n".join(lines).rstrip())
                parts.append(
                    "I can see this has been an ongoing concern and I appreciate your patience.\n"
                    "I'll make sure this is addressed with full context from your previous interactions."
                )
            else:
                parts.append(
                    "I wasn't able to find any previous interactions matching your query.\n"
                    "No worries though — I'm here to help. Please go ahead and describe your issue and we'll get it sorted."
                )

    if not parts:
        return (
            "Your request has been processed.\n"
            "Let me know if there's anything else I can help you with."
        )

    closing = {
        "J1": "Is there anything else I can help you with today?",
        "J2": "I hope that's all sorted for you. Let me know if there's anything else you need.",
        "J3": "I know waiting isn't fun — thank you for your patience. We'll be in touch soon.",
        "J4": "How else can I help you today?",
        "J5": "Is there anything else I can assist you with?",
        "J-KB": "I hope that answers your question! Feel free to ask if you need anything else.",
    }.get(journey_type, "")

    # Join blocks with a blank line between each for clear visual separation
    response = "\n\n".join(parts)
    if closing:
        response += f"\n\n{closing}"
    return response
