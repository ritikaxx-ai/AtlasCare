"""
Vector store for CRM interaction history (RAG).

Primary: ChromaDB (persistent, in-process, ONNX MiniLM embeddings).
Fallback: keyword overlap scoring over JSON (no extra deps) for CI / quick start.
"""

import json
import os
import re
import threading
from typing import List, Optional

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_HISTORY_PATH = os.path.join(_DATA_DIR, "crm_interaction_history.json")
_CHROMA_PATH = os.path.join(_DATA_DIR, "chroma_db")
_COLLECTION_NAME = "crm_interactions"

# Module-level singletons — initialised lazily on first call to _get_chroma_collection().
# _chroma_available is None (not yet checked), True (working), or False (unavailable).
_lock = threading.Lock()
_chroma_collection = None
_chroma_available: Optional[bool] = None
_json_cache: Optional[list] = None  # in-memory cache of crm_interaction_history.json


def _load_interactions_json() -> list:
    global _json_cache
    if _json_cache is not None:
        return _json_cache
    if not os.path.exists(_HISTORY_PATH):
        _json_cache = []
        return _json_cache
    with open(_HISTORY_PATH, "r") as f:
        _json_cache = json.load(f).get("interactions", [])
    return _json_cache


def _interaction_document(interaction: dict) -> str:
    # Converts a CRM interaction dict into a single string that ChromaDB embeds.
    # All meaningful fields are concatenated so semantic search can find relevant history.
    tags = ", ".join(interaction.get("tags", []))
    return (
        f"Channel: {interaction.get('channel', 'unknown')}. "
        f"Summary: {interaction.get('summary', '')}. "
        f"Agent notes: {interaction.get('agent_notes', '')}. "
        f"Resolution: {interaction.get('resolution', 'unknown')}. "
        f"Tags: {tags}."
    )


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _search_fallback(customer_id: str, query: str, top_k: int) -> List[dict]:
    """
    Keyword overlap RAG used when ChromaDB isn't available (e.g. CI, fresh install).
    Tokenises both the query and each interaction document, scores by overlap ratio,
    and returns the top_k matches filtered to this customer.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for item in _load_interactions_json():
        if item.get("customer_id") != customer_id:
            continue
        doc_tokens = _tokenize(_interaction_document(item))
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        scored.append((overlap / max(len(query_tokens), 1), item))

    scored.sort(key=lambda x: (-x[0], x[1].get("timestamp", "")), reverse=False)
    hits = []
    for score, item in scored[:top_k]:
        hits.append(
            {
                "interaction_id": item["interaction_id"],
                "customer_id": item["customer_id"],
                "order_id": item.get("order_id"),
                "channel": item.get("channel"),
                "timestamp": item.get("timestamp"),
                "resolution": item.get("resolution"),
                "summary": item.get("summary"),
                "relevance_score": round(score, 3),
            }
        )
    return hits


def _get_chroma_collection():
    """
    Lazy initialiser for the ChromaDB collection. Thread-safe via _lock.
    On first call: creates a persistent ChromaDB client at data/chroma_db/,
    gets-or-creates the collection with ONNX MiniLM embeddings (cosine space),
    and bulk-indexes all existing interactions if the collection is empty.
    Returns None (and sets _chroma_available=False) if chromadb is not installed
    or any init error occurs — callers fall back to _search_fallback silently.
    """
    global _chroma_collection, _chroma_available

    if _chroma_available is False:
        return None

    with _lock:
        if _chroma_collection is not None:
            return _chroma_collection

        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        except ImportError:
            _chroma_available = False
            return None

        try:
            os.makedirs(_CHROMA_PATH, exist_ok=True)
            client = chromadb.PersistentClient(path=_CHROMA_PATH)
            collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )

            if collection.count() == 0:
                interactions = _load_interactions_json()
                if interactions:
                    collection.add(
                        ids=[i["interaction_id"] for i in interactions],
                        documents=[_interaction_document(i) for i in interactions],
                        metadatas=[
                            {
                                "customer_id": i["customer_id"],
                                "order_id": i.get("order_id") or "",
                                "channel": i.get("channel", ""),
                                "timestamp": i.get("timestamp", ""),
                                "resolution": i.get("resolution", ""),
                                "summary": i.get("summary", "")[:500],
                            }
                            for i in interactions
                        ],
                    )

            _chroma_collection = collection
            _chroma_available = True
            return collection
        except Exception:
            _chroma_available = False
            return None


def _search_chroma(customer_id: str, query: str, top_k: int) -> Optional[List[dict]]:
    collection = _get_chroma_collection()
    if collection is None or collection.count() == 0:
        return None

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where={"customer_id": customer_id},
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    hits: List[dict] = []
    for i, interaction_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        distance = (
            results["distances"][0][i]
            if results.get("distances") and results["distances"][0]
            else None
        )
        hits.append(
            {
                "interaction_id": interaction_id,
                "customer_id": meta.get("customer_id", customer_id),
                "order_id": meta.get("order_id") or None,
                "channel": meta.get("channel"),
                "timestamp": meta.get("timestamp"),
                "resolution": meta.get("resolution"),
                "summary": meta.get("summary"),
                "relevance_score": round(1 - distance, 3) if distance is not None else None,
            }
        )
    return hits


def search_customer_history(
    customer_id: str,
    query: str,
    top_k: int = 3,
) -> List[dict]:
    """
    RAG lookup over a single customer's past interactions.
    Filtered by customer_id — never crosses customers.
    """
    chroma_hits = _search_chroma(customer_id, query, top_k)
    if chroma_hits is not None:
        return chroma_hits
    return _search_fallback(customer_id, query, top_k)


def reset_index() -> None:
    """Clear Chroma index (for tests)."""
    global _chroma_collection, _chroma_available, _json_cache

    with _lock:
        _chroma_collection = None
        _chroma_available = None
        _json_cache = None

        chroma_path = _CHROMA_PATH
        if os.path.isdir(chroma_path):
            import shutil
            shutil.rmtree(chroma_path, ignore_errors=True)
