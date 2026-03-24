"""
═══════════════════════════════════════════════════════════════
CONCEPT: VECTOR STORE + METADATA FILTERING
═══════════════════════════════════════════════════════════════

A vector store does two things:
  1. STORE:   Takes text chunks → embeds them → saves (vector, metadata)
  2. RETRIEVE: Takes a query → embeds it → finds most similar vectors

METADATA FILTERING (the key advanced concept):
  Without filtering: search ALL chunks, return top-K by similarity.
  With filtering:    FIRST narrow by metadata conditions, THEN rank by similarity.

  Example filter:
    {"source": "langchain_docs.md", "category": "RAG"}
    → only search chunks from that file in that category

  This is critical for multi-tenant apps, multi-doc corpora,
  or when you know the answer is in a specific document.

FILTER OPERATORS:
  {"field": "value"}              → exact match
  {"field": {"$contains": "x"}}  → substring match
  {"field": {"$in": ["a","b"]}}  → one of these values
  {"field": {"$gte": 5}}         → greater than or equal
  {"$and": [{...}, {...}]}        → both conditions must match
  {"$or":  [{...}, {...}]}        → either condition matches
═══════════════════════════════════════════════════════════════
"""

import math
import json
import os
from typing import Optional
from rag.splitters import Chunk
from config.settings import TOP_K_RETRIEVAL


class InMemoryVectorStore:
    """
    Pure-Python in-memory vector store.
    No ChromaDB or FAISS needed — same interface, works offline.

    In production you'd swap this for:
      - ChromaDB    (pip install chromadb)
      - FAISS       (pip install faiss-cpu)
      - Pinecone    (cloud, pip install pinecone-client)
      - Weaviate    (cloud/local, pip install weaviate-client)

    The interface is identical — just swap the class.
    """

    def __init__(self, llm):
        self.llm = llm
        self._store: list[dict] = []   # [{text, embedding, metadata}]

    # ── WRITE ────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed and store a list of Chunk objects.
        Returns count of chunks added.
        """
        added = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            embedding = self.llm.embed(chunk.text)
            self._store.append({
                "text":      chunk.text,
                "embedding": embedding,
                "metadata":  chunk.metadata,
                "chunk_id":  chunk.chunk_id,
            })
            added += 1
        print(f"  ✅ Added {added} chunks (total store size: {len(self._store)})")
        return added

    def add_texts(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """Convenience: add raw strings with optional metadata."""
        from rag.splitters import Chunk
        metadatas = metadatas or [{} for _ in texts]
        chunks = [Chunk(text=t, metadata=m) for t, m in zip(texts, metadatas)]
        return self.add_chunks(chunks)

    # ── READ ─────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        filter: dict = None,
    ) -> list[dict]:
        """
        Find the k most similar chunks to `query`.

        Steps:
          1. Embed the query.
          2. Apply metadata filter (if provided) to narrow candidates.
          3. Compute cosine similarity between query and each candidate.
          4. Return top-k sorted by similarity descending.

        Args:
            query:  The search question.
            k:      How many results to return.
            filter: Metadata filter dict (see module docstring for operators).
        """
        if not self._store:
            return []

        query_emb = self.llm.embed(query)

        # Step 2: apply metadata filter
        candidates = self._apply_filter(self._store, filter)
        if not candidates:
            return []

        # Step 3: cosine similarity for each candidate
        scored = []
        for item in candidates:
            sim = self._cosine(query_emb, item["embedding"])
            scored.append({**item, "score": sim})

        # Step 4: sort and return top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    # ── METADATA FILTER ENGINE ───────────────────────────────────────────────

    def _apply_filter(self, items: list[dict], filter: dict) -> list[dict]:
        """
        Filter items by metadata conditions.
        Supports nested $and / $or and comparison operators.
        """
        if not filter:
            return items
        return [item for item in items if self._matches(item["metadata"], filter)]

    def _matches(self, metadata: dict, condition: dict) -> bool:
        """Recursively evaluate a filter condition against metadata."""
        for key, value in condition.items():

            # Logical operators
            if key == "$and":
                return all(self._matches(metadata, sub) for sub in value)
            if key == "$or":
                return any(self._matches(metadata, sub) for sub in value)

            # Field-level operators
            actual = metadata.get(key)
            if isinstance(value, dict):
                op, operand = next(iter(value.items()))
                if op == "$contains":
                    if not (actual and operand in str(actual)):
                        return False
                elif op == "$in":
                    if actual not in operand:
                        return False
                elif op == "$nin":
                    if actual in operand:
                        return False
                elif op == "$gte":
                    if not (actual is not None and actual >= operand):
                        return False
                elif op == "$lte":
                    if not (actual is not None and actual <= operand):
                        return False
                elif op == "$gt":
                    if not (actual is not None and actual > operand):
                        return False
                elif op == "$lt":
                    if not (actual is not None and actual < operand):
                        return False
                elif op == "$ne":
                    if actual == operand:
                        return False
            else:
                # Exact match
                if actual != value:
                    return False
        return True

    # ── UTILITIES ────────────────────────────────────────────────────────────

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1
        mag_b = math.sqrt(sum(x * x for x in b)) or 1
        return dot / (mag_a * mag_b)

    def count(self) -> int:
        return len(self._store)

    def get_all_metadata_values(self, field: str) -> list:
        """Useful for discovering what filter values are available."""
        return list({item["metadata"].get(field) for item in self._store
                     if field in item["metadata"]})

    def save(self, path: str) -> None:
        """Persist store to JSON (embeddings included)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._store, f)

    def load(self, path: str) -> None:
        """Load store from JSON."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self._store = json.load(f)
            print(f"  Loaded {len(self._store)} chunks from {path}")
