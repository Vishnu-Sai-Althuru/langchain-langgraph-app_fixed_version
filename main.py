import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(__file__))

from llm.providers       import get_llm
from rag.pipeline        import RAGPipeline
from rag.splitters       import get_splitter
from agent.graph         import LangGraphAgent
from evaluation.evaluator import RAGEvaluator, GROUND_TRUTH_QA
from config.settings     import DATA_DIR, LLM_PROVIDER
from database import init_db
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1: LangChain RAG with Custom Splitters & Metadata Filtering
# ─────────────────────────────────────────────────────────────────────────────
def demo_rag(llm, splitter_type: str = "recursive"):
    print("\n" + "═" * 65)
    print("📚  DEMO 1: LangChain RAG — Custom Splitters + Metadata Filtering")
    print("═" * 65)

    # Build pipeline
    rag = RAGPipeline(llm, splitter_type=splitter_type)

    # Ingest documents with metadata
    print("\n📥 Ingesting documents...")
    docs = [
        {
            "text": """RAG (Retrieval-Augmented Generation) combines a retrieval system
with a language model. Documents are split into chunks, embedded into vectors,
and stored in a vector database. When a query arrives, the most similar chunks
are retrieved and injected into the LLM prompt as context. This grounds the
model's answers in real data, significantly reducing hallucinations.""",
            "metadata": {"source": "rag_intro.txt", "category": "RAG", "version": 2}
        },
        {
            "text": """Vector databases store text as high-dimensional embeddings.
Cosine similarity measures how close two embeddings are in vector space.
A score of 1.0 means identical meaning. Metadata filtering narrows the
candidate set before similarity ranking, improving both speed and precision.
Popular options: ChromaDB, FAISS, Pinecone, Weaviate.""",
            "metadata": {"source": "vector_db.txt", "category": "Database", "version": 1}
        },
        {
            "text": """LangGraph builds agents as state graphs. Each node is a Python
function. Edges define control flow. Conditional edges let the agent decide
which tool to call based on intermediate results. The agent loops using the
ReAct pattern: Reason about what's needed, Act by calling a tool, Observe the
result, repeat until done.""",
            "metadata": {"source": "langgraph.txt", "category": "Agent", "version": 1}
        },
        {
            "text": """Text splitters divide documents into chunks before embedding.
RecursiveCharacterTextSplitter is LangChain's default — it tries paragraph,
sentence, and word boundaries in order. MarkdownSplitter respects header
hierarchy. SemanticSplitter groups sentences by topic similarity.
Chunk overlap (10-15%) prevents information loss at boundaries.""",
            "metadata": {"source": "splitters.txt", "category": "RAG", "version": 3}
        },
    ]

    # Also ingest files from data directory
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith((".md", ".txt")) and os.path.isfile(fpath):
            category = "Agent" if "agent" in fname else "RAG"
            rag.ingest_file(fpath, extra_metadata={"category": category})

    rag.ingest_documents(docs)
    print(f"\n  Pipeline stats: {json.dumps(rag.stats(), indent=4)}")

    # ── DEMO A: Basic query ─────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🔍 Query 1: Basic RAG (no filter)")
    q1 = input("  Enter your query [default: 'What is RAG and how does it work?']: ").strip()
    if not q1:
        q1 = "What is RAG and how does it work?"
        print(f"  Using default: {q1}")
    result = rag.query(q1)
    print(f"  Answer: {result['answer'][:200]}...")
    print(f"  Sources: {result['sources']}")
    print(f"  Scores:  {[round(s,3) for s in result['scores']]}")

    # ── DEMO B: Metadata filtering ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("🔍 Query 2: WITH metadata filter — category=RAG only")
    q2 = input("  Enter your query [default: 'What are the text splitting strategies?']: ").strip()
    if not q2:
        q2 = "What are the text splitting strategies?"
        print(f"  Using default: {q2}")
    result_filtered = rag.query(
        q2,
        filter={"category": "RAG"}
    )
    print(f"  Answer: {result_filtered['answer'][:200]}...")
    print(f"  Sources (should all be RAG category): {result_filtered['sources']}")

    # ── DEMO C: Multi-condition filter ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("🔍 Query 3: Multi-condition filter — category=RAG AND version>=2")
    q3 = input("  Enter your query [default: 'How does metadata filtering work?']: ").strip()
    if not q3:
        q3 = "How does metadata filtering work?"
        print(f"  Using default: {q3}")
    result_v2 = rag.query(
        q3,
        filter={"$and": [
            {"category": "RAG"},
            {"version": {"$gte": 2}},
        ]}
    )
    print(f"  Answer: {result_v2['answer'][:200]}...")
    print(f"  Sources: {result_v2['sources']}")

    return rag


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: LangGraph Agent with Tools
# ─────────────────────────────────────────────────────────────────────────────
def demo_agent(llm, rag):
    print("\n" + "═" * 65)
    print("🤖  DEMO 2: LangGraph Agent — search + db_query + file_ops")
    print("═" * 65)
    print("\n  Available tools:")
    print("    • search    — knowledge lookups (e.g. 'Explain RAG')")
    print("    • db_query  — database queries  (e.g. 'Get all documents in RAG category')")
    print("    • file_ops  — file operations   (e.g. 'List all files' / 'Read rag_overview.txt')")
    print("\n  Type your queries one at a time. Press Enter with no input to finish.\n")

    agent = LangGraphAgent(llm, rag_pipeline=rag)

    query_num = 1
    while True:
        print("─" * 50)
        raw = input(f"  Query {query_num} (or press Enter to finish): ").strip()
        if not raw:
            print("  No more queries — exiting agent demo.")
            break

        result = agent.run(raw)
        print(f"\n  ✅ ANSWER: {result['answer'][:300]}")
        print(f"  Tools used: {result['tools']}")
        print(f"  Steps taken: {result['steps']}\n")
        query_num += 1

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: RAG Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def demo_evaluation(llm, rag):
    print("\n" + "═" * 65)
    print("📊  DEMO 3: RAG Evaluation — Ground Truth Q/A + Accuracy Metrics")
    print("═" * 65)

    evaluator = RAGEvaluator(rag, llm, GROUND_TRUTH_QA)
    report = evaluator.run()
    evaluator.print_report(report)

    # Save report to file
    report_path = os.path.join(DATA_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: LLM Provider Swapping
# ─────────────────────────────────────────────────────────────────────────────
def demo_provider_swap():
    print("\n" + "═" * 65)
    print("🔌  DEMO 4: API Wrappers — Swap LLM Providers Without Code Changes")
    print("═" * 65)

    test_prompt = "In one sentence, what is RAG?"

    providers_to_try = ["mock"]
    if os.getenv("OPENAI_API_KEY"):
        providers_to_try.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_try.append("anthropic")

    print(f"\n  Testing providers: {providers_to_try}")
    print("  (Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test real providers)\n")

    for provider in providers_to_try:
        print(f"  Provider: {provider}")
        try:
            llm = get_llm(provider)
            response = llm.generate(test_prompt)
            emb = llm.embed("test")
            print(f"    Response: {response[:100]}")
            print(f"    Embedding: {len(emb)} dimensions")
            print(f"    ✅ {provider} works\n")
        except Exception as e:
            print(f"    ❌ {provider} failed: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LangChain RAG + LangGraph Agent Demo")
    parser.add_argument("--rag",      action="store_true", help="Run RAG demo only")
    parser.add_argument("--agent",    action="store_true", help="Run agent demo only")
    parser.add_argument("--eval",     action="store_true", help="Run evaluation only")
    parser.add_argument("--provider", type=str, default=LLM_PROVIDER, help="LLM provider")
    parser.add_argument("--splitter", type=str, default="recursive", help="Splitter type")
    parser.add_argument("--demo",     action="store_true", help="Run all 4 demos")
    args = parser.parse_args()

    print(f"\n🚀 LangChain RAG + LangGraph Agent")
    print(f"   LLM Provider : {args.provider}")
    print(f"   Splitter     : {args.splitter}")

    # Build LLM
    llm = get_llm(args.provider)

    run_all = args.demo or not any([args.rag, args.agent, args.eval])

    rag = None

    if args.rag or run_all:
        rag = demo_rag(llm, args.splitter)

    if args.agent or run_all:
        if rag is None:
            rag = demo_rag(llm, args.splitter)
        demo_agent(llm, rag)

    if args.eval or run_all:
        if rag is None:
            rag = demo_rag(llm, args.splitter)
        demo_evaluation(llm, rag)

    if run_all:
        demo_provider_swap()

    print("\n✅ All demos complete.")


if __name__ == "__main__":
    main()
