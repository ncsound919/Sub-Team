"""
Sub-Team Memory — Persistent cross-session learning via Mem0.

Gives agents persistent memory that survives across sessions:
  - Store and retrieve learned patterns, decisions, and context
  - Agent-specific memory namespaces
  - Shared team memory for cross-agent knowledge transfer
  - Automatic relevance scoring and decay

Thread-safe singleton with bounded fallback store.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

__all__ = ["AgentMemory", "get_memory"]

# Max memories per agent in the fallback store (LRU eviction)
_MAX_MEMORIES_PER_AGENT = 1000


class AgentMemory:
    """
    Persistent memory store for the Sub-Team workforce.

    Uses Mem0 for vector-based memory storage when available,
    falls back to a bounded in-memory dict store otherwise.

    Thread-safe for concurrent access.

    Usage::

        memory = AgentMemory()

        # Store a memory for a specific agent
        memory.add(
            content="The user prefers Python over JavaScript for backend work",
            agent_id="software_engineer",
            metadata={"category": "preference", "confidence": 0.9}
        )

        # Search relevant memories
        results = memory.search("what language for backend?", agent_id="software_engineer")

        # Store shared team memory
        memory.add(
            content="Project uses FastAPI + Supabase stack",
            agent_id="team_shared",
        )
    """

    def __init__(self):
        self._mem0_client = None
        self._fallback_store: Dict[str, collections.OrderedDict] = {}
        self._initialized = False
        self._lock = threading.Lock()

    def _init_mem0(self):
        """Lazy-initialize Mem0 client (thread-safe)."""
        if self._initialized:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return
            self._initialized = True

            try:
                from mem0 import Memory

                config = {
                    "version": "v1.1",
                }

                # Use OpenAI embeddings if available
                openai_key = os.environ.get("OPENAI_API_KEY")
                if (
                    openai_key
                    and "openrouter"
                    not in os.environ.get("OPENAI_BASE_URL", "").lower()
                ):
                    config["embedder"] = {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-small",
                            "api_key": openai_key,
                        },
                    }

                # Use Qdrant if available, otherwise in-memory
                qdrant_url = os.environ.get("QDRANT_URL")
                if qdrant_url:
                    config["vector_store"] = {
                        "provider": "qdrant",
                        "config": {
                            "url": qdrant_url,
                            "api_key": os.environ.get("QDRANT_API_KEY"),
                            "collection_name": "sub_team_memory",
                        },
                    }

                self._mem0_client = Memory.from_config(config)
                _log.info("Mem0 memory store initialized successfully")

            except ImportError:
                _log.info("Mem0 not available, using in-memory fallback store")
            except Exception as e:
                _log.warning("Mem0 initialization failed, using fallback: %s", e)

    def add(
        self,
        content: str,
        agent_id: str = "team_shared",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a memory.

        Parameters
        ----------
        content : str
            The memory content to store.
        agent_id : str
            Agent namespace (or 'team_shared' for team-wide memory).
        metadata : dict, optional
            Additional metadata to attach.

        Returns
        -------
        str or None
            Memory ID if stored successfully.
        """
        self._init_mem0()

        if self._mem0_client:
            try:
                result = self._mem0_client.add(
                    content,
                    user_id=agent_id,
                    metadata=metadata or {},
                )
                if isinstance(result, dict) and "results" in result:
                    results = result["results"]
                    if results and isinstance(results, list):
                        return results[0].get("id")
                return None
            except Exception as e:
                _log.warning("Mem0 add failed, using fallback: %s", e)

        # Fallback: bounded in-memory store with LRU eviction
        with self._lock:
            if agent_id not in self._fallback_store:
                self._fallback_store[agent_id] = collections.OrderedDict()

            store = self._fallback_store[agent_id]
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"

            entry = {
                "id": memory_id,
                "content": content,
                "metadata": metadata or {},
            }

            store[memory_id] = entry

            # Evict oldest entries if over limit
            while len(store) > _MAX_MEMORIES_PER_AGENT:
                store.popitem(last=False)

            return memory_id

    def search(
        self,
        query: str,
        agent_id: str = "team_shared",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories.

        Parameters
        ----------
        query : str
            The search query.
        agent_id : str
            Agent namespace to search within.
        limit : int
            Maximum number of results.

        Returns
        -------
        list[dict]
            Matching memories with content, metadata, and relevance scores.
        """
        self._init_mem0()

        if self._mem0_client:
            try:
                results = self._mem0_client.search(
                    query,
                    user_id=agent_id,
                    limit=limit,
                )
                if isinstance(results, dict) and "results" in results:
                    return results["results"]
                if isinstance(results, list):
                    return results
                return []
            except Exception as e:
                _log.warning("Mem0 search failed, using fallback: %s", e)

        # Fallback: simple keyword matching
        with self._lock:
            store = self._fallback_store.get(agent_id, {})
            memories = list(store.values())

        query_lower = query.lower()
        scored = []
        for mem in memories:
            content_lower = mem["content"].lower()
            # Simple relevance: count matching words
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append({**mem, "score": overlap / len(query_words)})

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[:limit]

    def get_all(self, agent_id: str = "team_shared") -> List[Dict[str, Any]]:
        """Get all memories for an agent."""
        self._init_mem0()

        if self._mem0_client:
            try:
                result = self._mem0_client.get_all(user_id=agent_id)
                if isinstance(result, dict) and "results" in result:
                    return result["results"]
                if isinstance(result, list):
                    return result
                return []
            except Exception as e:
                _log.warning("Mem0 get_all failed: %s", e)

        with self._lock:
            store = self._fallback_store.get(agent_id, {})
            return list(store.values())

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        self._init_mem0()

        if self._mem0_client:
            try:
                self._mem0_client.delete(memory_id)
                return True
            except Exception as e:
                _log.warning("Mem0 delete failed: %s", e)
                return False

        # Fallback: remove from in-memory store
        with self._lock:
            for agent_id, store in self._fallback_store.items():
                if memory_id in store:
                    del store[memory_id]
                    return True
        return True

    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear all memories for an agent (or all agents if None)."""
        self._init_mem0()

        if self._mem0_client:
            try:
                if agent_id:
                    self._mem0_client.delete_all(user_id=agent_id)
                else:
                    self._mem0_client.reset()
            except Exception as e:
                _log.warning("Mem0 clear failed: %s", e)

        with self._lock:
            if agent_id:
                self._fallback_store.pop(agent_id, None)
            else:
                self._fallback_store.clear()


# Singleton instance (thread-safe)
_memory_instance: Optional[AgentMemory] = None
_singleton_lock = threading.Lock()


def get_memory() -> AgentMemory:
    """Get the singleton AgentMemory instance (thread-safe)."""
    global _memory_instance
    if _memory_instance is None:
        with _singleton_lock:
            if _memory_instance is None:
                _memory_instance = AgentMemory()
    return _memory_instance
