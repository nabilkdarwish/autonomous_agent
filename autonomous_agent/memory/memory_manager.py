"""
Memory system:
  - ShortTermMemory  : sliding-window conversation buffer
  - LongTermMemory   : FAISS vector store for persistent recall
  - MemoryManager    : unified interface
"""

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import AgentConfig, get_config
from core.logger import get_logger
from core.models import Message, MessageRole

logger = get_logger("memory")


# ──────────────────────────────────────────────────────────
# Short-term memory
# ──────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    A simple sliding-window buffer that holds the last N messages
    and forms the conversation context sent to the LLM.
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: List[Message] = []

    def add(self, role: MessageRole, content: str, metadata: Optional[Dict] = None) -> Message:
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        # Trim oldest non-system messages to stay within limit
        non_system = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        if len(non_system) > self.max_messages:
            to_remove = len(non_system) - self.max_messages
            removed = 0
            i = 0
            while removed < to_remove and i < len(self._messages):
                if self._messages[i].role != MessageRole.SYSTEM:
                    self._messages.pop(i)
                    removed += 1
                else:
                    i += 1
        logger.debug(f"STM add | role={role.value} | total={len(self._messages)}")
        return msg

    def get_messages(self) -> List[Message]:
        return list(self._messages)

    def get_context_window(self, last_n: Optional[int] = None) -> List[Message]:
        msgs = self._messages
        if last_n:
            msgs = msgs[-last_n:]
        return list(msgs)

    def clear(self):
        self._messages.clear()
        logger.debug("STM cleared")

    def summary(self) -> str:
        return f"ShortTermMemory: {len(self._messages)} messages"

    def __len__(self) -> int:
        return len(self._messages)


# ──────────────────────────────────────────────────────────
# Long-term memory (FAISS vector store)
# ──────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    entry_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 1.0


class LongTermMemory:
    """
    Persistent semantic memory backed by a FAISS index.
    Falls back to a simple keyword search if sentence-transformers
    or faiss-cpu is not installed.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        self.store_path = Path(self.config.vector_store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._entries: List[MemoryEntry] = []
        self._index = None
        self._embedder = None
        self._use_vector = self._init_vector()
        self._load()

    # ── initialise ──────────────────────────────────────────

    def _init_vector(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._embedder = SentenceTransformer(self.config.embedding_model)
            self._faiss = faiss
            logger.info("LTM: FAISS vector store enabled")
            return True
        except ImportError:
            logger.warning("LTM: FAISS/sentence-transformers not found — using keyword fallback")
            return False

    # ── persistence ─────────────────────────────────────────

    def _entries_path(self) -> Path:
        return self.store_path / "entries.pkl"

    def _index_path(self) -> Path:
        return self.store_path / "index.faiss"

    def _load(self):
        ep = self._entries_path()
        if ep.exists():
            with open(ep, "rb") as f:
                self._entries = pickle.load(f)
            logger.info(f"LTM: loaded {len(self._entries)} entries from disk")
        if self._use_vector:
            ip = self._index_path()
            if ip.exists():
                self._index = self._faiss.read_index(str(ip))
            else:
                self._rebuild_index()

    def _save(self):
        with open(self._entries_path(), "wb") as f:
            pickle.dump(self._entries, f)
        if self._use_vector and self._index:
            self._faiss.write_index(self._index, str(self._index_path()))

    def _rebuild_index(self):
        if not self._use_vector or not self._entries:
            return
        import numpy as np
        texts = [e.content for e in self._entries]
        embeddings = self._embedder.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self._index = self._faiss.IndexFlatIP(dim)
        self._faiss.normalize_L2(embeddings)
        self._index.add(embeddings.astype("float32"))

    # ── public API ──────────────────────────────────────────

    def store(self, content: str, metadata: Optional[Dict] = None, importance: float = 1.0) -> str:
        entry_id = f"mem_{int(time.time() * 1000)}_{len(self._entries)}"
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )
        self._entries.append(entry)

        if self._use_vector:
            import numpy as np
            emb = self._embedder.encode([content], convert_to_numpy=True).astype("float32")
            self._faiss.normalize_L2(emb)
            if self._index is None:
                dim = emb.shape[1]
                self._index = self._faiss.IndexFlatIP(dim)
            self._index.add(emb)

        self._save()
        logger.debug(f"LTM store | id={entry_id}")
        return entry_id

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[MemoryEntry, float]]:
        k = top_k or self.config.long_term_top_k
        if not self._entries:
            return []

        if self._use_vector and self._index:
            return self._vector_search(query, k)
        return self._keyword_search(query, k)

    def _vector_search(self, query: str, k: int) -> List[Tuple[MemoryEntry, float]]:
        import numpy as np
        emb = self._embedder.encode([query], convert_to_numpy=True).astype("float32")
        self._faiss.normalize_L2(emb)
        actual_k = min(k, len(self._entries))
        scores, indices = self._index.search(emb, actual_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._entries):
                results.append((self._entries[idx], float(score)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _keyword_search(self, query: str, k: int) -> List[Tuple[MemoryEntry, float]]:
        query_words = set(query.lower().split())
        scored = []
        for entry in self._entries:
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / len(query_words | entry_words)
                scored.append((entry, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_relevant_context(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return ""
        parts = ["[Relevant memories from past sessions:]"]
        for entry, score in results:
            parts.append(f"• {entry.content} (relevance: {score:.2f})")
        return "\n".join(parts)

    def clear(self):
        self._entries.clear()
        self._index = None
        self._save()
        logger.info("LTM cleared")

    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "vector_enabled": self._use_vector,
            "store_path": str(self.store_path),
        }

    def __len__(self) -> int:
        return len(self._entries)


# ──────────────────────────────────────────────────────────
# Unified memory manager
# ──────────────────────────────────────────────────────────

class MemoryManager:
    """
    Single access point for both memory layers.
    Automatically promotes important short-term content to long-term.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        self.short_term = ShortTermMemory(self.config.short_term_max_messages)
        self.long_term = LongTermMemory(self.config)
        logger.info("MemoryManager initialised")

    def add_message(self, role: MessageRole, content: str,
                    metadata: Optional[Dict] = None) -> Message:
        return self.short_term.add(role, content, metadata)

    def store_fact(self, fact: str, metadata: Optional[Dict] = None, importance: float = 1.0):
        return self.long_term.store(fact, metadata, importance)

    def get_context(self, query: Optional[str] = None) -> str:
        if query:
            return self.long_term.get_relevant_context(query)
        return ""

    def get_conversation(self) -> List[Message]:
        return self.short_term.get_messages()

    def promote_to_long_term(self, content: str, importance: float = 1.0):
        """Explicitly move a piece of information to long-term memory."""
        self.long_term.store(content, {"source": "promoted_from_stm"}, importance)

    def clear_short_term(self):
        self.short_term.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "short_term": self.short_term.summary(),
            "long_term": self.long_term.stats(),
        }
