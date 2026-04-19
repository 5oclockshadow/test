"""Redis + RAPTOR memory layer for agent state persistence.

Redis is used for fast key-value and list storage.
RAPTOR provides hierarchical text summarization and retrieval.

Both Redis and openai are optional; if unavailable the memory layer
degrades gracefully to in-process dict/list storage.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    redis = None  # type: ignore
    _REDIS_AVAILABLE = False


class RAPTORMemory:
    """Hierarchical summarization-based memory (simplified RAPTOR).

    Stores text chunks and builds a rolling summary tree for efficient
    retrieval.  Requires an MCPHub for summarization; degrades to a simple
    ordered list when none is provided.

    Parameters
    ----------
    mcp_hub:
        MCPHub instance used for LLM summarization (optional).
    summary_every:
        Number of new chunks after which a summary pass is triggered.
    """

    def __init__(
        self,
        mcp_hub: Optional[Any] = None,
        summary_every: int = 10,
    ) -> None:
        self._hub = mcp_hub
        self._summary_every = summary_every
        self._chunks: List[str] = []
        self._summaries: List[str] = []

    def add(self, text: str) -> None:
        """Add a text chunk to the memory."""
        self._chunks.append(text)
        if len(self._chunks) % self._summary_every == 0:
            self._rebuild_summaries()

    def _rebuild_summaries(self) -> None:
        """Rebuild the summary tree from the most recent chunks."""
        batch = self._chunks[-self._summary_every :]
        if self._hub is None:
            self._summaries.extend(batch)
            return
        combined = " | ".join(batch)
        try:
            summary = self._hub.call(
                [{"role": "user", "content": f"Summarize concisely:\n{combined}"}],
                system_prompt="You are a memory summarizer. Be brief and factual.",
            )
            self._summaries.append(summary)
        except Exception as exc:
            logger.debug("RAPTOR summary failed: %s", exc)
            self._summaries.extend(batch)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the most relevant items for a query.

        Falls back to recency-based retrieval when semantic ranking is
        unavailable.
        """
        all_items = self._summaries + self._chunks
        return all_items[-top_k:] if all_items else []

    def to_dict(self) -> Dict[str, Any]:
        return {"chunks": list(self._chunks), "summaries": list(self._summaries)}

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._chunks = list(data.get("chunks", []))
        self._summaries = list(data.get("summaries", []))


class AgentMemory:
    """Unified agent memory with Redis backend and RAPTOR summarization.

    Falls back to in-process dicts when Redis is unavailable.

    Parameters
    ----------
    namespace:
        Key namespace prefix for Redis isolation.
    redis_url:
        Redis connection URL (e.g. ``redis://localhost:6379``).  If None or
        Redis is not installed, in-process storage is used.
    mcp_hub:
        MCPHub instance forwarded to the RAPTOR memory layer.
    """

    _RAPTOR_STATE_KEY = "_raptor_state"

    def __init__(
        self,
        namespace: str = "agent",
        redis_url: Optional[str] = None,
        mcp_hub: Optional[Any] = None,
    ) -> None:
        self.namespace = namespace
        self._local: Dict[str, Any] = {}
        self._lists: Dict[str, List[Any]] = {}
        self._redis: Any = None
        self.raptor = RAPTORMemory(mcp_hub=mcp_hub)

        if _REDIS_AVAILABLE and redis_url:
            try:
                self._redis = redis.from_url(redis_url)
                self._redis.ping()
                logger.info("AgentMemory: Redis connected at %s", redis_url)
            except Exception as exc:
                logger.warning(
                    "AgentMemory: Redis unavailable (%s); using local storage.", exc
                )
                self._redis = None

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _rkey(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # ------------------------------------------------------------------
    # Key-value store
    # ------------------------------------------------------------------

    def store(self, key: str, value: Any) -> None:
        """Store a JSON-serialisable value under *key*."""
        serialized = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.set(self._rkey(key), serialized)
                return
            except Exception as exc:
                logger.debug("Redis store failed: %s", exc)
        self._local[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by *key*."""
        if self._redis is not None:
            try:
                raw = self._redis.get(self._rkey(key))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.debug("Redis retrieve failed: %s", exc)
        return self._local.get(key, default)

    # ------------------------------------------------------------------
    # List operations
    # ------------------------------------------------------------------

    def append_list(self, key: str, value: Any) -> None:
        """Append *value* to the list stored under *key*."""
        serialized = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.rpush(self._rkey(key), serialized)
                return
            except Exception as exc:
                logger.debug("Redis rpush failed: %s", exc)
        self._lists.setdefault(key, []).append(value)

    def get_list(self, key: str) -> List[Any]:
        """Get all items from the list stored under *key*."""
        if self._redis is not None:
            try:
                items = self._redis.lrange(self._rkey(key), 0, -1)
                return [json.loads(i) for i in items]
            except Exception as exc:
                logger.debug("Redis lrange failed: %s", exc)
        return list(self._lists.get(key, []))

    # ------------------------------------------------------------------
    # RAPTOR integration
    # ------------------------------------------------------------------

    def add_memory_text(self, text: str) -> None:
        """Add *text* to the RAPTOR memory and persist its state."""
        self.raptor.add(text)
        self.store(self._RAPTOR_STATE_KEY, self.raptor.to_dict())

    def load_raptor_state(self) -> None:
        """Restore RAPTOR state from persistent storage."""
        state = self.retrieve(self._RAPTOR_STATE_KEY)
        if state:
            self.raptor.from_dict(state)
