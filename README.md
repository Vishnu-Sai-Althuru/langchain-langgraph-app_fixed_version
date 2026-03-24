# LangChain RAG + LangGraph Agent — Complete Guide

A fully self-contained implementation of all 4 advanced AI engineering concepts,
with zero mandatory external dependencies (runs offline with MockLLM).

---

## What's Inside

```
langchain-langgraph-app/
├── config/settings.py          # All config constants — ONE place to change things
├── llm/providers.py            # LLM provider abstraction (OpenAI/Anthropic/Ollama/Mock)
├── rag/
│   ├── splitters.py            # 5 custom text splitters with full explanations
│   ├── vectorstore.py          # In-memory vector store + metadata filtering engine
│   └── pipeline.py             # Full RAG pipeline (ingest + query)
├── agent/graph.py              # LangGraph-style agent + 3 tools (search/DB/file)
├── evaluation/evaluator.py     # Ground-truth Q/A + 4 accuracy metrics
├── data/                       # Sample documents for indexing
└── main.py                     # Entry point — runs all 4 demos
```

---

## Quick Start

```bash
# 1. Clone / unzip into a folder
cd langchain-langgraph-app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install (only requests needed for Ollama; everything else is stdlib)
pip install requests

# 4. Run all 4 demos (no API key needed — uses MockLLM)
python main.py --demo

# 5. Run individual demos
python main.py --rag              # RAG pipeline only
python main.py --agent            # LangGraph agent only
python main.py --eval             # Evaluation report only
```

---

## CONCEPT 1 — LangChain RAG with Custom Splitters & Metadata Filtering

### The Problem
LLMs have a fixed context window. You can't feed an entire document into the prompt.
RAG solves this by splitting documents into chunks, embedding them, and retrieving
only the most relevant pieces at query time.

### 5 Custom Text Splitters (rag/splitters.py)

| Splitter | Strategy | Best For |
|---|---|---|
| `CharacterSplitter` | Every N chars | Simple baseline |
| `RecursiveSplitter` | Paragraph → Sentence → Word | General text (LangChain default) |
| `MarkdownSplitter` | Respects # ## ### headers | Documentation, README files |
| `SentenceSplitter` | Complete sentences only | Better embedding quality |
| `SemanticSplitter` | Groups by topic similarity | Best quality, requires embeddings |

```python
from rag.splitters import get_splitter

# Choose your splitter
splitter = get_splitter("recursive", chunk_size=512, overlap=64)
splitter = get_splitter("markdown")
splitter = get_splitter("semantic", llm=my_llm)

# Split with metadata attached to every chunk
chunks = splitter.split(text, metadata={
    "source":   "langchain_docs.md",
    "category": "RAG",
    "version":  3,
})
# Each chunk: Chunk(text="...", metadata={source, category, version, chunk_index})
```

### Metadata Filtering (rag/vectorstore.py)

Every chunk stores metadata alongside its embedding. At query time, filter BEFORE
similarity ranking runs — this narrows candidates and improves precision.

```python
from rag.pipeline import RAGPipeline

rag = RAGPipeline(llm, splitter_type="recursive")
rag.ingest_text(text, metadata={"source": "doc.txt", "category": "AI", "version": 2})

# No filter — search all chunks
result = rag.query("What is RAG?")

# Filter by category
result = rag.query("What is RAG?", filter={"category": "AI"})

# Multi-condition filter
result = rag.query("What is RAG?", filter={
    "$and": [
        {"category": "AI"},
        {"version": {"$gte": 2}},
    ]
})

print(result["answer"])   # LLM-generated grounded answer
print(result["sources"])  # Chunk metadata for citations
print(result["scores"])   # Similarity scores
```

**Supported filter operators:**
```python
{"field": "value"}              # exact match
{"field": {"$contains": "x"}}  # substring
{"field": {"$in": ["a","b"]}}  # one of
{"field": {"$gte": 5}}         # >=
{"field": {"$lte": 5}}         # <=
{"$and": [{...}, {...}]}        # both must match
{"$or":  [{...}, {...}]}        # either matches
```

---

## CONCEPT 2 — LangGraph Agent with Tools

### What is LangGraph?
LangGraph builds agents as state graphs. Unlike chains (always A→B→C), an agent
DECIDES which tool to call based on the query. It loops until done.

### The ReAct Loop (agent/graph.py)
```
START
  │
  ▼
[think_node]  ← LLM sees query + observations → picks tool or gives final answer
  │
  ├── TOOL: search   → [execute_tool] → back to think_node
  ├── TOOL: db_query → [execute_tool] → back to think_node  
  ├── TOOL: file_ops → [execute_tool] → back to think_node
  └── FINAL_ANSWER   → END
```

### 3 Built-in Tools

**SearchTool** — searches web or knowledge base
```python
search.run("RAG retrieval methods")
# → "RAG combines retrieval systems with LLMs..."
```

**DatabaseTool** — executes SQL SELECT queries (SQLite)
```python
db.run("SELECT title, category FROM documents WHERE category='RAG'")
# → '[{"title": "LangChain RAG Guide", "category": "RAG"}]'
```

**FileOperationsTool** — reads/writes files safely
```python
file_ops.run("list")                          # list all files
file_ops.run("read langchain_rag_guide.md")   # read a file
file_ops.run("write notes.txt My findings")   # write a file
```

### Running the Agent
```python
from agent.graph import LangGraphAgent
from rag.pipeline import RAGPipeline

rag   = RAGPipeline(llm)
agent = LangGraphAgent(llm, rag_pipeline=rag)

result = agent.run("Search for RAG and query the database for document counts.")
print(result["answer"])       # final answer
print(result["tools_used"])   # ["search", "db_query"]
print(result["steps"])        # number of iterations
```

### How to Add Your Own Tool
```python
class MyCustomTool:
    name = "my_tool"
    description = "Does something custom."

    def run(self, input_str: str) -> str:
        # your logic here
        return "tool result"

# Register in LangGraphAgent.__init__:
self.tools["my_tool"] = MyCustomTool()
```

---

## CONCEPT 3 — RAG Evaluation: Ground-Truth Q/A + Accuracy Metrics

### Why Evaluate?
Without evaluation, you don't know if your RAG system actually answers correctly.
You might be retrieving irrelevant chunks or the LLM might be hallucinating.

### 4 Metrics (evaluation/evaluator.py)

| Metric | What It Measures | When to Use |
|---|---|---|
| **Exact Match (EM)** | Token F1 overlap | Factual, short answers |
| **ROUGE-L** | Longest common subsequence | Longer descriptive answers |
| **Semantic Similarity** | Embedding cosine distance | Paraphrased correct answers |
| **Retrieval Precision** | Were retrieved chunks relevant? | Debugging retrieval quality |

### Running Evaluation
```python
from evaluation.evaluator import RAGEvaluator, GROUND_TRUTH_QA

evaluator = RAGEvaluator(rag_pipeline, llm, GROUND_TRUTH_QA)
report = evaluator.run()
evaluator.print_report(report)
```

### Output
```
📊  RAG EVALUATION REPORT
══════════════════════════════════════════════════════════════
  Total questions : 7
  Passed          : 5 (71.4%)

  METRIC AVERAGES:
  Exact Match          0.4200  (threshold: 0.5)
  ROUGE-L              0.3800  (threshold: 0.3)
  Semantic Sim.        0.6100  (threshold: 0.6)
  Retrieval Prec.      0.5000

  BY CATEGORY:
  RAG             0.6200  ████████████
  Database        0.5100  ██████████
  Agent           0.4300  ████████

  WEAKEST QUESTIONS (investigate these):
  ❌ [q006] What is the difference between ROUGE and semantic similarity...
       exact=0.038  rouge=0.069  sem=0.001
```

### Adding Your Own Ground Truth
```python
MY_QA = [
    {
        "id":         "my001",
        "question":   "What is the capital of France?",
        "expected":   "Paris is the capital of France.",
        "category":   "Geography",
        "difficulty": "easy",
    },
]
evaluator = RAGEvaluator(rag, llm, MY_QA)
```

---

## CONCEPT 4 — API Wrappers to Swap LLM Providers

### The Problem
If you scatter `openai.ChatCompletion.create(...)` calls across your codebase,
switching to Anthropic means rewriting every call site.

### The Solution — BaseLLM Interface (llm/providers.py)
```python
class BaseLLM(ABC):
    def generate(self, prompt: str, system: str = "") -> str: ...
    def embed(self, text: str) -> list[float]: ...
    def chat(self, messages: list[dict]) -> str: ...
```

Every provider implements the same 3 methods. Your application code never
imports `openai` or `anthropic` directly — it only calls `BaseLLM` methods.

### Switching Providers — Zero Code Changes

**Option 1: Environment variable (recommended)**
```bash
export LLM_PROVIDER=openai      && python main.py
export LLM_PROVIDER=anthropic   && python main.py
export LLM_PROVIDER=ollama      && python main.py
export LLM_PROVIDER=mock        && python main.py  # offline
```

**Option 2: Programmatic**
```python
from llm.providers import get_llm

llm = get_llm("openai")       # GPT-4o-mini
llm = get_llm("anthropic")    # Claude Sonnet
llm = get_llm("ollama")       # Local LLaMA3
llm = get_llm("mock")         # No API needed

# Same call regardless of provider:
response = llm.generate("What is RAG?")
embedding = llm.embed("RAG combines retrieval...")
```

### Adding a New Provider
```python
class GroqLLM(BaseLLM):
    def generate(self, prompt, system=""):
        # your groq API call here
        ...
    def embed(self, text):
        # groq embedding call or fallback
        ...

# Add to get_llm() factory:
elif provider == "groq":
    return GroqLLM()
```

That's it. Everything else — RAG pipeline, agent, evaluator — works unchanged.

---

## Environment Variables

```bash
# Provider selection
export LLM_PROVIDER=mock          # mock | openai | anthropic | ollama
export LLM_MODEL=gpt-4o-mini      # model name for the provider
export LLM_TEMPERATURE=0.0

# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OLLAMA_BASE_URL=http://localhost:11434

# RAG settings
export CHUNK_SIZE=512
export CHUNK_OVERLAP=64
export TOP_K=4

# Paths
export DB_PATH=./agent_data.db
export DATA_DIR=./data
```

---

## Run Commands

```bash
python main.py --demo             # all 4 demos
python main.py --rag              # RAG + metadata filtering only
python main.py --agent            # LangGraph agent only
python main.py --eval             # evaluation report only
python main.py --splitter markdown   # use markdown splitter
python main.py --provider openai     # use OpenAI (needs key)
```

---

Built by Vishnu — AI Research Agent v2.0 | Capgemini | Python + Zero Dependencies
# langchain-langgraph-app
