

# import re
# import math
# import json
# import hashlib
# from abc import ABC, abstractmethod
# from typing import Optional
# from config.settings import (
#     LLM_PROVIDER, LLM_MODEL, LLM_TEMP, LLM_MAX_TOKENS,
#     OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 1: Define the abstract interface ALL providers must implement
# # ─────────────────────────────────────────────────────────────────────────────
# class BaseLLM(ABC):
#     """
#     Abstract base class for all LLM providers.
#     Any class that inherits this MUST implement generate() and embed().
#     This is the contract your application code depends on.
#     """

#     @abstractmethod
#     def generate(self, prompt: str, system: str = "") -> str:
#         """Send a prompt, get a text response back."""
#         ...

#     @abstractmethod
#     def embed(self, text: str) -> list[float]:
#         """Convert text into a vector embedding."""
#         ...

#     def chat(self, messages: list[dict]) -> str:
#         """
#         Multi-turn chat convenience method.
#         messages = [{"role": "user", "content": "..."},
#                     {"role": "assistant", "content": "..."}]
#         Default: flatten to a single prompt string.
#         Providers can override for native chat APIs.
#         """
#         flat = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
#         return self.generate(flat)


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 2: OpenAI Provider
# # ─────────────────────────────────────────────────────────────────────────────
# class OpenAILLM(BaseLLM):
#     """
#     Wraps the OpenAI API.
#     Install: pip install openai
#     Set env: export OPENAI_API_KEY=sk-...
#     """

#     def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMP):
#         self.model = model
#         self.temperature = temperature
#         try:
#             from openai import OpenAI
#             self.client = OpenAI(api_key=OPENAI_API_KEY)
#         except ImportError:
#             raise ImportError("Run: pip install openai")

#     def generate(self, prompt: str, system: str = "") -> str:
#         messages = []
#         if system:
#             messages.append({"role": "system", "content": system})
#         messages.append({"role": "user", "content": prompt})

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=LLM_MAX_TOKENS,
#         )
#         return response.choices[0].message.content.strip()

#     def embed(self, text: str) -> list[float]:
#         response = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text,
#         )
#         return response.data[0].embedding

#     def chat(self, messages: list[dict]) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#         )
#         return response.choices[0].message.content.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 3: Anthropic Provider
# # ─────────────────────────────────────────────────────────────────────────────
# class AnthropicLLM(BaseLLM):
#     """
#     Wraps the Anthropic Claude API.
#     Install: pip install anthropic
#     Set env: export ANTHROPIC_API_KEY=sk-ant-...
#     """

#     def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = LLM_TEMP):
#         self.model = model
#         self.temperature = temperature
#         try:
#             import anthropic
#             self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
#         except ImportError:
#             raise ImportError("Run: pip install anthropic")

#     def generate(self, prompt: str, system: str = "") -> str:
#         kwargs = dict(
#             model=self.model,
#             max_tokens=LLM_MAX_TOKENS,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         if system:
#             kwargs["system"] = system

#         response = self.client.messages.create(**kwargs)
#         return response.content[0].text.strip()

#     def embed(self, text: str) -> list[float]:
#         # Anthropic doesn't have embeddings — fall back to hash-based mock
#         # In production: use voyage-ai or openai embeddings alongside Claude
#         return MockLLM()._hash_embed(text)

#     def chat(self, messages: list[dict]) -> str:
#         # Anthropic expects no "system" in messages list
#         system = ""
#         filtered = []
#         for m in messages:
#             if m["role"] == "system":
#                 system = m["content"]
#             else:
#                 filtered.append(m)
#         return self.generate(filtered[-1]["content"], system=system)


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 4: Ollama Provider (local, no API key needed)
# # ─────────────────────────────────────────────────────────────────────────────
# class OllamaLLM(BaseLLM):
#     """
#     Wraps a locally running Ollama server.
#     Install Ollama: https://ollama.com
#     Then: ollama pull llama3
#     """

#     def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
#         self.model = model
#         self.base_url = base_url
#         try:
#             import requests
#             self.requests = requests
#         except ImportError:
#             raise ImportError("Run: pip install requests")

#     def generate(self, prompt: str, system: str = "") -> str:
#         full_prompt = f"{system}\n\n{prompt}" if system else prompt
#         resp = self.requests.post(
#             f"{self.base_url}/api/generate",
#             json={"model": self.model, "prompt": full_prompt, "stream": False},
#             timeout=60,
#         )
#         return resp.json().get("response", "").strip()

#     def embed(self, text: str) -> list[float]:
#         resp = self.requests.post(
#             f"{self.base_url}/api/embeddings",
#             json={"model": self.model, "prompt": text},
#             timeout=30,
#         )
#         return resp.json().get("embedding", [])

#     def chat(self, messages: list[dict]) -> str:
#         resp = self.requests.post(
#             f"{self.base_url}/api/chat",
#             json={"model": self.model, "messages": messages, "stream": False},
#             timeout=60,
#         )
#         return resp.json().get("message", {}).get("content", "").strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 5: Mock Provider (zero dependencies — for testing & CI)
# # ─────────────────────────────────────────────────────────────────────────────

# class MockLLM(BaseLLM):
#     """
#     Deterministic rule-based LLM — no API key, no network, no latency.
#     Used for: unit tests, local dev, CI pipelines, offline demos.
#     """

#     def _hash_embed(self, text: str, dims: int = 384) -> list[float]:
#         """
#         Semantic-preserving embedding via word-level hashing trick.
#         Each word is hashed independently so related texts share bucket
#         weights and produce meaningful cosine similarity.
#         """
#         words = re.findall(r'\b[a-z0-9]+\b', text.lower())
#         vec = [0.0] * dims
#         for word in words:
#             h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
#             idx  = h % dims
#             sign = 1 if (h >> 1) & 1 == 0 else -1
#             vec[idx] += sign * 1.0
#         mag = math.sqrt(sum(x * x for x in vec)) or 1
#         return [x / mag for x in vec]

#     def _is_agent_prompt(self, p: str) -> bool:
#         """Detect when generate() is called from the agent's think_node."""
#         return "tools available:" in p and "user query:" in p

#     def _extract_agent_query(self, p: str) -> str:
#         """Pull the USER QUERY line out of an agent reasoning prompt."""
#         m = re.search(r"user query:\s*(.+?)(?:\n|tools available)", p, re.DOTALL)
#         return m.group(1).strip() if m else p

#     def _has_observations(self, p: str) -> bool:
#         """Check if the agent already has tool results to work with."""
#         return "none yet" not in p and "observations so far:" in p

#     def generate(self, prompt: str, system: str = "") -> str:
#         # Extract ONLY the user question (ignore context)
#         question_match = re.search(r"question:\s*(.*)", prompt, re.IGNORECASE)
#         question = question_match.group(1).strip() if question_match else prompt
#         p = question.lower()   # ✅ FIX

#         # ── Agent reasoning prompt: must return TOOL: / FINAL_ANSWER: format ──
#         if self._is_agent_prompt(p):
#             query   = self._extract_agent_query(p)
#             has_obs = self._has_observations(p)

#             if has_obs:
#                 obs_text = "\n".join([o for o in prompt.split("\n") if "[" in o])
#                 return (
#                     "FINAL_ANSWER: Based on database results:\n"
#                     f"{obs_text[:300]}"
#                 )
#                 if "rag" in query or "retrieval" in query or "search" in query:
#                     return (
#                         "FINAL_ANSWER: RAG (Retrieval-Augmented Generation) combines "
#                         "a retrieval system with a language model. Documents are fetched "
#                         "from a vector store and injected into the prompt as context, "
#                         "grounding responses in real data and reducing hallucinations."
#                     )
#                 if "database" in query or "documents" in query or "category" in query:
#                     return (
#                         "FINAL_ANSWER: The RAG category contains 1 document: "
#                         "'LangChain RAG Guide' (word_count=2500, created 2025-01-01). "
#                         "Retrieved via db_query on the documents table."
#                     )
#                 if "file" in query or "list" in query or "read" in query or "notes" in query:
#                     return (
#                         "FINAL_ANSWER: Available files include rag_overview.txt and "
#                         "agent_notes.md. The agent notes list three tools: search, "
#                         "db_query, and file_ops, and recommend verifying information "
#                         "from multiple sources."
#                     )
#                 return f"FINAL_ANSWER: Based on the retrieved information: {query[:120]}"

#             if "search" in query or "information about" in query or "explain" in query:
#                 topic = "RAG retrieval augmented generation" if "rag" in query else query[:60]
#                 return f"TOOL: search\nINPUT: {topic}"

#             if "database" in query or "query" in query or "documents" in query or "category" in query:
#                 return (
#                     "TOOL: db_query\n"
#                     "INPUT: SELECT * FROM documents WHERE category = 'RAG'"
#                 )

#             if "list" in query or "files" in query:
#                 return "TOOL: file_ops\nINPUT: list"

#             if "read" in query or "notes" in query or "agent" in query:
#                 return "TOOL: file_ops\nINPUT: read agent_notes.md"

#             return f"FINAL_ANSWER: {query[:200]}"

#         # ── Standard (non-agent) prompt responses ────────────────────────────
#         if "rag" in p or "retrieval" in p:
#             return (
#                 "RAG (Retrieval-Augmented Generation) combines a retrieval system "
#                 "with a language model. Documents are fetched from a vector store "
#                 "and injected into the prompt as context, grounding responses in "
#                 "real data and reducing hallucinations."
#             )
#         if "langgraph" in p or "agent" in p:
#             return (
#                 "A LangGraph agent uses a directed graph where each node is a "
#                 "function (tool call, LLM call, or decision). Edges define the "
#                 "control flow. The agent loops: observe state → pick tool → "
#                 "execute → update state → repeat until done."
#             )
#         if "splitter" in p or "chunk" in p:
#             return (
#                 "Text splitters divide documents into chunks before embedding. "
#                 "RecursiveCharacterTextSplitter tries paragraph → sentence → "
#                 "word boundaries. SemanticSplitter groups by topic. "
#                 "MarkdownSplitter respects header hierarchy."
#             )
#         if "vector" in p or "embedding" in p:
#             return (
#                 "Vector databases store embeddings and support similarity search. "
#                 "Given a query embedding, cosine similarity ranks the stored "
#                 "documents by relevance. Metadata filters narrow results by "
#                 "source, date, category etc. before similarity ranking."
#             )
#         if "evaluate" in p or "metric" in p or "rouge" in p:
#             return (
#                 "RAG evaluation compares model answers against ground-truth. "
#                 "Exact Match checks word overlap. ROUGE-L measures longest common "
#                 "subsequence. Semantic similarity uses embedding cosine distance. "
#                 "All three together give a robust accuracy picture."
#             )
#         if "search" in p:
#             return "Search results: found 3 relevant documents about the topic."
#         # if "sql" in p or "database" in p or "query" in p:
#         #     return "Database returned: [{'id': 1, 'name': 'AI Research', 'count': 42}]"
#         if "file" in p or "read" in p or "write" in p:
#             return "File operation completed successfully."
#         if "summarize" in p or "summary" in p:
#             txt = re.search(r"text[:\s]*(.{50,})", p, re.DOTALL)
#             snippet = txt.group(1)[:100] if txt else prompt[:100]
#             return f"Summary: {snippet.strip()}..."
#         return f"[MockLLM] Processed query about: {prompt[:60].strip()}"

#     def embed(self, text: str) -> list[float]:
#         return self._hash_embed(text)


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 6: Factory function — the ONE place that knows about providers
# # ─────────────────────────────────────────────────────────────────────────────
# def get_llm(provider: str = LLM_PROVIDER, **kwargs) -> BaseLLM:
#     """
#     Factory function.  Call this anywhere you need an LLM.

#     Usage:
#         llm = get_llm()                    # uses LLM_PROVIDER from .env
#         llm = get_llm("openai")            # explicit
#         llm = get_llm("anthropic")         # switch without changing app code
#         response = llm.generate("Hello")   # same API regardless of provider

#     To add a new provider: add an elif below + implement BaseLLM. Done.
#     """
#     provider = provider.lower().strip()

#     if provider == "openai":
#         return OpenAILLM(**kwargs)
#     elif provider == "anthropic":
#         return AnthropicLLM(**kwargs)
#     elif provider == "ollama":
#         # Try to reach Ollama; fall back to mock if not running
#         try:
#             import requests
#             r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
#             if r.status_code == 200:
#                 print(f"✅ Ollama running — using {OLLAMA_MODEL}")
#                 return OllamaLLM(**kwargs)
#         except Exception:
#             pass
#         print("⚠️  Ollama not reachable → falling back to MockLLM")
#         return MockLLM()
#     elif provider == "mock":
#         return MockLLM()
#     else:
#         raise ValueError(
#             f"Unknown provider '{provider}'. "
#             "Choose: openai | anthropic | ollama | mock"
#         )


import re
import math
import hashlib
from abc import ABC, abstractmethod
from config.settings import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMP, LLM_MAX_TOKENS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
)


# ─────────────────────────────────────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def chat(self, messages: list[dict]) -> str:
        flat = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.generate(flat)


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────────────────────────────────────
class OpenAILLM(BaseLLM):

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMP):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC
# ─────────────────────────────────────────────────────────────────────────────
class AnthropicLLM(BaseLLM):

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = LLM_TEMP):
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def embed(self, text: str) -> list[float]:
        return MockLLM()._hash_embed(text)


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
class OllamaLLM(BaseLLM):

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        import requests
        self.requests = requests
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        resp = self.requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "").strip()

    def embed(self, text: str) -> list[float]:
        resp = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        return resp.json().get("embedding", [])


# ─────────────────────────────────────────────────────────────────────────────
# MOCK LLM (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
class MockLLM(BaseLLM):

    def _hash_embed(self, text: str, dims: int = 384) -> list[float]:
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        vec = [0.0] * dims

        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % dims
            sign = 1 if (h >> 1) & 1 == 0 else -1
            vec[idx] += sign

        mag = math.sqrt(sum(x * x for x in vec)) or 1
        return [x / mag for x in vec]

    def _is_agent_prompt(self, p: str) -> bool:
        # Matches both old ("tools available:") and new ("tools:") prompt formats
        return ("user query:" in p) and ("tools:" in p or "tools available:" in p)

    def _extract_agent_query(self, p: str) -> str:
        m = re.search(r"user query:\s*(.+?)(?:\n|tools)", p, re.DOTALL)
        return m.group(1).strip() if m else p

    def _has_observations(self, p: str) -> bool:
        # Matches both old ("observations so far:") and new ("results gathered:") formats
        has_results = "results gathered:" in p or "observations so far:" in p
        is_empty = "none yet" in p or "none\n" in p
        return has_results and not is_empty

    def generate(self, prompt: str, system: str = "") -> str:
        p = prompt.lower()   # ✅ FIXED

        # ───────── AGENT MODE ─────────
        if self._is_agent_prompt(p):
            query = self._extract_agent_query(p)
            has_obs = self._has_observations(p)

            if has_obs:
                # Extract tool outputs
                obs_text = "\n".join(
                    line for line in prompt.split("\n") if "[" in line
                )

                if obs_text.strip():
                    return (
                        "FINAL_ANSWER: Based on tool results:\n"
                        f"{obs_text[:300]}"
                    )

                return f"FINAL_ANSWER: {query[:120]}"

            # Decide tool
            if "search" in query or "explain" in query:
                topic = "RAG retrieval augmented generation" if "rag" in query else query
                return f"TOOL: search\nINPUT: {topic}"

            if "database" in query or "documents" in query:
                return (
                    "TOOL: db_query\n"
                    "INPUT: SELECT * FROM documents WHERE category = 'RAG'"
                )

            if "files" in query or "list" in query:
                return "TOOL: file_ops\nINPUT: list"

            if "read" in query:
                return "TOOL: file_ops\nINPUT: read agent_notes.md"

            return f"FINAL_ANSWER: {query[:200]}"

        # ───────── NORMAL MODE ─────────
        if "rag" in p:
            return "RAG combines retrieval + LLM for grounded answers."

        if "vector" in p:
            return "Vector DB stores embeddings and uses cosine similarity."

        if "agent" in p:
            return "Agents use tools + reasoning loop (ReAct)."

        if "database" in p:
            return "Use SQL queries to retrieve structured data."

        if "file" in p:
            return "File operation completed."

        return f"[MockLLM] {prompt[:60]}"

    def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_llm(provider: str = LLM_PROVIDER, **kwargs) -> BaseLLM:

    provider = provider.lower().strip()

    if provider == "openai":
        return OpenAILLM(**kwargs)

    elif provider == "anthropic":
        return AnthropicLLM(**kwargs)

    elif provider == "ollama":
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                print(f"✅ Ollama running — using {OLLAMA_MODEL}")
                return OllamaLLM(**kwargs)
        except Exception:
            pass

        print("⚠️ Falling back to MockLLM")
        return MockLLM()

    elif provider == "mock":
        return MockLLM()

    else:
        raise ValueError("Choose: openai | anthropic | ollama | mock")
