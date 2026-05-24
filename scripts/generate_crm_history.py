#!/usr/bin/env python3
"""
Generate 400 synthetic CRM interaction history records and write to JSON.

Run: python scripts/generate_crm_history.py
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

CUSTOMERS = [f"CUST-{i:03d}" for i in range(1, 21)]
ORDERS_BY_CUSTOMER = {
    "CUST-001": ["ORD-78321", "ORD-78450"],
    "CUST-002": ["ORD-78100"],
}
CHANNELS = ["phone", "chat", "email"]
RESOLUTIONS = ["resolved", "escalated", "pending", "resolved", "resolved"]

ISSUE_TEMPLATES = [
    (
        "delivery_delay",
        "Customer reported order {order} has not arrived by estimated delivery date.",
        "Checked carrier tracking; delay due to regional hub backlog. Offered ₹200 goodwill credit.",
        "resolved",
    ),
    (
        "refund_inquiry",
        "Customer asked about refund status for cancelled item on order {order}.",
        "Refund was initiated; confirmed 3–5 business day SLA with payment provider.",
        "resolved",
    ),
    (
        "damaged_product",
        "Customer reported damaged {product} received from order {order}.",
        "Requested photos; damage confirmed. Refund/ replacement options discussed.",
        "escalated",
    ),
    (
        "wrong_item",
        "Customer received wrong product on order {order}; wants exchange.",
        "Initiated reverse pickup; replacement order to be created after QC.",
        "pending",
    ),
    (
        "billing_dispute",
        "Customer disputed charge of ₹{amount} on order {order}.",
        "Verified invoice against order lines; explained line-item breakdown.",
        "resolved",
    ),
    (
        "address_change",
        "Customer requested shipping address update for order {order} before dispatch.",
        "Address updated in OMS; confirmed new pincode serviceable.",
        "resolved",
    ),
    (
        "warranty_claim",
        "Customer inquired about warranty coverage for {product} from order {order}.",
        "Shared warranty terms; registered claim for specialist review if needed.",
        "pending",
    ),
    (
        "cancellation_request",
        "Customer wanted to cancel line item on order {order} before shipping.",
        "Processed partial cancellation; refund queued to original payment method.",
        "resolved",
    ),
]

PRODUCTS = [
    "Dell Inspiron 15 Laptop",
    "Laptop Backpack",
    "Sony WH-1000XM5",
    "Graphic T-Shirt",
    "Blue Denim Jeans",
    "Wireless Mouse",
    "USB-C Hub",
]

# Curated repeat-contact storyline for CUST-001 (damaged laptop)
PRIYA_REPEAT_STORY = [
    {
        "interaction_id": "INT-PRIYA-001",
        "customer_id": "CUST-001",
        "order_id": "ORD-78321",
        "channel": "phone",
        "timestamp": "2025-04-18T10:15:00Z",
        "summary": "Customer called about cracked screen on Dell Inspiron 15 Laptop from order ORD-78321.",
        "agent_notes": "Advised to share photos via email. Opened damage assessment ticket. Customer seemed frustrated but cooperative.",
        "resolution": "pending",
        "tags": ["damaged_product", "laptop", "electronics"],
    },
    {
        "interaction_id": "INT-PRIYA-002",
        "customer_id": "CUST-001",
        "order_id": "ORD-78321",
        "channel": "chat",
        "timestamp": "2025-04-22T14:40:00Z",
        "summary": "Follow-up chat: customer uploaded damage photos for ORD-78321 laptop; asking about full refund timeline.",
        "agent_notes": "Photos verified. Refund amount ₹55,000 exceeds auto-refund threshold — explained specialist review required within 24h SLA.",
        "resolution": "escalated",
        "tags": ["damaged_product", "refund_inquiry", "escalation"],
    },
    {
        "interaction_id": "INT-PRIYA-003",
        "customer_id": "CUST-001",
        "order_id": "ORD-78321",
        "channel": "phone",
        "timestamp": "2025-04-25T09:05:00Z",
        "summary": "Second phone call: customer still waiting on refund decision for damaged laptop on ORD-78321; no case update received.",
        "agent_notes": "Apologized for delay. Confirmed escalation case exists. Promised callback from specialist within 24 hours.",
        "resolution": "pending",
        "tags": ["damaged_product", "refund_inquiry", "repeat_contact"],
    },
]


def _random_timestamp(days_ago_max: int = 90) -> str:
    days_ago = random.randint(1, days_ago_max)
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=random.randint(0, 12))
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_interactions(count: int = 400) -> list:
    interactions = list(PRIYA_REPEAT_STORY)
    used_ids = {i["interaction_id"] for i in interactions}
    seq = 1

    while len(interactions) < count:
        customer_id = random.choice(CUSTOMERS)
        order_ids = ORDERS_BY_CUSTOMER.get(customer_id, [])
        order_id = random.choice(order_ids) if order_ids and random.random() > 0.3 else None

        tag, summary_tpl, notes_tpl, resolution = random.choice(ISSUE_TEMPLATES)
        product = random.choice(PRODUCTS)
        amount = random.choice([999, 1500, 2499, 8000, 15000])

        order_ref = order_id or "N/A"
        summary = summary_tpl.format(order=order_ref, product=product, amount=amount)
        agent_notes = notes_tpl.format(order=order_ref, product=product, amount=amount)

        interaction_id = f"INT-{seq:05d}"
        while interaction_id in used_ids:
            seq += 1
            interaction_id = f"INT-{seq:05d}"
        used_ids.add(interaction_id)

        interactions.append(
            {
                "interaction_id": interaction_id,
                "customer_id": customer_id,
                "order_id": order_id,
                "channel": random.choice(CHANNELS),
                "timestamp": _random_timestamp(),
                "summary": summary,
                "agent_notes": agent_notes,
                "resolution": resolution,
                "tags": [tag, random.choice(["electronics", "apparel", "home_goods"])],
            }
        )
        seq += 1

    interactions.sort(key=lambda x: x["timestamp"])
    return interactions


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crm_interaction_history.json")

    interactions = generate_interactions(400)
    payload = {
        "description": "Synthetic CRM past interaction summaries for RAG retrieval",
        "count": len(interactions),
        "interactions": interactions,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(interactions)} interactions to {out_path}")


if __name__ == "__main__":
    main()
