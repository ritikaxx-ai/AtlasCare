"""
In-Memory Data Store for AtlasCare
Replaces file I/O with cached lookups for sub-millisecond performance
"""

import json
import os
from typing import Optional, Dict, List
import threading


class DataStore:
    """Singleton in-memory data store with thread-safe operations"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        
        # Load all data into memory on initialization
        self._orders = self._load_orders()
        self._crm = self._load_crm()
        self._kb = self._load_kb()
        self._payment_config = self._load_payment_config()
        
        # Write locks for data modification
        self._orders_lock = threading.Lock()
        self._crm_lock = threading.Lock()
        
        self._initialized = True
    
    def _load_orders(self) -> Dict[str, dict]:
        """Load orders and index by order_id"""
        path = os.path.join(self._data_dir, "orders.json")
        with open(path, "r") as f:
            orders_list = json.load(f)["orders"]
        return {order["order_id"]: order for order in orders_list}
    
    def _load_crm(self) -> dict:
        """Load CRM data"""
        path = os.path.join(self._data_dir, "crm_cases.json")
        with open(path, "r") as f:
            return json.load(f)
    
    def _load_kb(self) -> List[dict]:
        """Load knowledge base articles"""
        path = os.path.join(self._data_dir, "kb_articles.json")
        with open(path, "r") as f:
            return json.load(f)["articles"]
    
    def _load_payment_config(self) -> dict:
        """Load payment configuration"""
        path = os.path.join(self._data_dir, "payment_config.json")
        with open(path, "r") as f:
            return json.load(f)
    
    # ==================== ORDERS ====================
    
    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order by ID (O(1) lookup)"""
        return self._orders.get(order_id)

    def get_orders_for_customer(self, customer_id: str) -> Optional[List[dict]]:
        """Return all orders for a customer, sorted newest-first. None if customer unknown."""
        all_orders = [o for o in self._orders.values() if o.get("customer_id") == customer_id]
        if not all_orders:
            return None
        return sorted(all_orders, key=lambda o: o.get("created_at", ""), reverse=True)
    
    def update_order(self, order_id: str, updated_order: dict) -> dict:
        """Update order in memory (thread-safe)"""
        with self._orders_lock:
            self._orders[order_id] = updated_order
            # Persist to disk asynchronously (optional for demo)
            self._persist_orders()
        return updated_order
    
    def cancel_order_item(self, order_id: str, line_id: int) -> dict:
        """Cancel specific item in order"""
        with self._orders_lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            item_found = False
            already_cancelled = False
            for item in order["items"]:
                if item["line_id"] == line_id:
                    if item["status"] == "cancelled":
                        already_cancelled = True
                    else:
                        item["status"] = "cancelled"
                    item_found = True
                    break

            if not item_found:
                raise ValueError(f"Line item {line_id} not found in order {order_id}")

            if already_cancelled:
                return {"already_cancelled": True, "order_id": order_id, "line_id": line_id}
            
            # Recalculate total
            active_items = [i for i in order["items"] if i["status"] == "active"]
            order["total_amount"] = sum(i["unit_price"] * i["quantity"] for i in active_items)
            
            # Mark order cancelled if no active items
            if not active_items:
                order["status"] = "cancelled"
            
            self._orders[order_id] = order
            self._persist_orders()
            
        return order
    
    def cancel_full_order(self, order_id: str) -> dict:
        """Cancel all active items in an order and update order status."""
        with self._orders_lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")

            cancelled_count = 0
            for item in order["items"]:
                if item["status"] != "cancelled":
                    item["status"] = "cancelled"
                    cancelled_count += 1

            order["status"] = "cancelled"
            order["total_amount"] = 0.0
            self._orders[order_id] = order
            self._persist_orders()

        return {"order_id": order_id, "cancelled_count": cancelled_count, "status": "cancelled"}

    def update_shipping_address(self, order_id: str, address: dict) -> dict:
        """Update shipping address for order"""
        with self._orders_lock:
            order = self._orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            if order["status"] in ["delivered", "cancelled"]:
                raise ValueError(f"Cannot update address for order in status {order['status']}")
            
            order["shipping_address"] = {
                "line1": address.get("line1", ""),
                "line2": address.get("line2", ""),
                "city": address.get("city", ""),
                "state": address.get("state", ""),
                "pincode": address.get("pincode", "")
            }
            
            self._orders[order_id] = order
            self._persist_orders()
            
        return order
    
    def _persist_orders(self):
        """Persist orders to disk, merging with any orders added to the file externally."""
        path = os.path.join(self._data_dir, "orders.json")
        # Read current file to avoid clobbering orders added outside this process
        try:
            with open(path, "r") as f:
                on_disk = {o["order_id"]: o for o in json.load(f)["orders"]}
        except Exception:
            on_disk = {}
        # In-memory state wins for orders we know about; file wins for unknown ones
        merged = {**on_disk, **self._orders}
        with open(path, "w") as f:
            json.dump({"orders": list(merged.values())}, f, indent=2)
    
    # ==================== CRM ====================
    
    def get_customer(self, customer_id: str) -> Optional[dict]:
        """Get customer profile by ID"""
        for customer in self._crm.get("customers", []):
            if customer["customer_id"] == customer_id:
                return customer
        return None
    
    def get_customer_address(self, customer_id: str, label: str) -> Optional[dict]:
        """Get specific customer address by label"""
        customer = self.get_customer(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        
        for addr in customer.get("addresses", []):
            if addr.get("label", "").lower() == label.lower():
                return addr
        
        raise ValueError(f"Address with label '{label}' not found for customer {customer_id}")
    
    def create_crm_case(self, case: dict) -> dict:
        """Create new CRM case"""
        with self._crm_lock:
            self._crm.setdefault("cases", []).append(case)
            self._persist_crm()
        return case

    def get_case_by_id(self, case_id: str) -> Optional[dict]:
        """Look up a CRM case by case_id"""
        for case in self._crm.get("cases", []):
            if case.get("case_id", "").upper() == case_id.upper():
                return case
        return None

    def log_interaction(self, interaction: dict) -> dict:
        """Append interaction to crm_interaction_history.json and index in ChromaDB."""
        history_path = os.path.join(self._data_dir, "crm_interaction_history.json")
        with self._crm_lock:
            with open(history_path, "r") as f:
                history = json.load(f)
            history["interactions"].append(interaction)
            history["count"] = len(history["interactions"])
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        # Index new interaction in ChromaDB (best-effort)
        try:
            from agent.vector_store import _get_chroma_collection, _interaction_document
            collection = _get_chroma_collection()
            if collection is not None:
                collection.add(
                    ids=[interaction["interaction_id"]],
                    documents=[_interaction_document(interaction)],
                    metadatas=[{
                        "customer_id": interaction.get("customer_id", ""),
                        "order_id": interaction.get("order_id") or "",
                        "channel": interaction.get("channel", ""),
                        "timestamp": interaction.get("timestamp", ""),
                        "resolution": interaction.get("resolution", ""),
                        "summary": interaction.get("summary", "")[:500],
                    }],
                )
        except Exception:
            pass  # Never block the response on logging

        return interaction

    def _persist_crm(self):
        """Persist CRM data to disk"""
        path = os.path.join(self._data_dir, "crm_cases.json")
        with open(path, "w") as f:
            json.dump(self._crm, f, indent=2)
    
    # ==================== KNOWLEDGE BASE ====================
    
    def search_kb(self, tags: List[str]) -> List[dict]:
        """Search knowledge base by tags. Empty tags list returns all articles."""
        if not tags:
            return list(self._kb)
        results = []
        for article in self._kb:
            article_tags = set(tag.lower() for tag in article.get("tags", []))
            search_tags = set(tag.lower() for tag in tags)
            if article_tags & search_tags:
                results.append(article)
        return results
    
    # ==================== PAYMENTS ====================
    
    def get_payment_config(self) -> dict:
        """Get payment gateway configuration"""
        return self._payment_config


# Global singleton instance
_data_store = None

def get_data_store() -> DataStore:
    """Get or create singleton data store instance"""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store
