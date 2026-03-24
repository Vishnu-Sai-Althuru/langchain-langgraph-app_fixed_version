

import os
from rag.splitters import get_splitter, Chunk
from rag.vectorstore import InMemoryVectorStore
from config.settings import TOP_K_RETRIEVAL


PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the question using ONLY
the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. ingest()  — load & split documents, embed, store
      2. query()   — retrieve + generate
      3. Both support metadata filtering
    """

    def __init__(self, llm, splitter_type: str = "recursive", top_k: int = TOP_K_RETRIEVAL):
        self.llm = llm
        self.vectorstore = InMemoryVectorStore(llm)
        self.splitter = get_splitter(splitter_type, llm=llm)
        self.top_k = top_k

    # ── INDEXING ─────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, metadata: dict = None) -> int:
        """Split a raw text string and add to vector store."""
        metadata = metadata or {}
        chunks = self.splitter.split(text, metadata)
        return self.vectorstore.add_chunks(chunks)

    def ingest_file(self, filepath: str, extra_metadata: dict = None) -> int:
        """Read a file and ingest it with file-based metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        # Auto-detect splitter based on file type
        if ext in (".md", ".markdown") and isinstance(self.splitter, type) is False:
            splitter = get_splitter("markdown")
        else:
            splitter = self.splitter

        metadata = {
            "source":   filename,
            "filepath": filepath,
            "filetype": ext,
            **(extra_metadata or {}),
        }
        chunks = splitter.split(text, metadata)
        return self.vectorstore.add_chunks(chunks)

    def ingest_documents(self, docs: list[dict]) -> int:
        """
        Ingest a list of {"text": ..., "metadata": {...}} dicts.
        This is the LangChain Document-style interface.
        """
        total = 0
        for doc in docs:
            total += self.ingest_text(doc["text"], doc.get("metadata", {}))
        return total

    # ── QUERYING ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, filter: dict = None, k: int = None) -> list[dict]:
        """
        Retrieve top-K chunks relevant to query, with optional metadata filter.

        filter examples:
          {"source": "langchain_docs.md"}          — only from this file
          {"category": {"$in": ["RAG", "LLM"]}}   — from these categories
          {"$and": [{"source": "x"}, {"page": 1}]} — both conditions
        """
        return self.vectorstore.similarity_search(query, k=k or self.top_k, filter=filter)

    def query(self, question: str, filter: dict = None, k: int = None) -> dict:
        """
        Full RAG query: retrieve → build context → LLM answer.

        Returns dict with:
          answer    — the LLM's generated response
          sources   — list of retrieved chunk metadata
          context   — the raw context injected into the prompt
          scores    — similarity scores for each retrieved chunk
        """
        # Step 1: retrieve relevant chunks
        results = self.retrieve(question, filter=filter, k=k)

        if not results:
            return {
                "answer":  "No relevant documents found in the knowledge base.",
                "sources": [],
                "context": "",
                "scores":  [],
            }

        # Step 2: build context string from retrieved chunks
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res["metadata"].get("source", "unknown")
            score  = res.get("score", 0)
            context_parts.append(f"[{i}] (source: {source}, score: {score:.3f})\n{res['text']}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: generate answer using LLM
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        return {
            "answer":  answer,
            "sources": [r["metadata"] for r in results],
            "context": context,
            "scores":  [r.get("score", 0) for r in results],
        }

    # ── INFO ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_chunks":      self.vectorstore.count(),
            "splitter":          type(self.splitter).__name__,
            "top_k":             self.top_k,
            "available_sources": self.vectorstore.get_all_metadata_values("source"),
            "available_cats":    self.vectorstore.get_all_metadata_values("category"),
        }
