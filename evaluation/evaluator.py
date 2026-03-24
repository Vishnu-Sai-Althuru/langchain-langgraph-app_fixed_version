

import re
import math
import json
from dataclasses import dataclass, field
from typing import Optional
from config.settings import (
    EVAL_THRESHOLD_EXACT,
    EVAL_THRESHOLD_ROUGE,
    EVAL_THRESHOLD_SEMANTIC,
)


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH DATASET
# ─────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH_QA = [
    {
        "id":       "q001",
        "question": "What is RAG?",
        "expected": "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model. Documents are fetched from a vector store and used as context for generating accurate answers.",
        "category": "RAG",
        "difficulty": "easy",
    },
    {
        "id":       "q002",
        "question": "What is a vector database used for?",
        "expected": "Vector databases store text embeddings and support semantic search using cosine similarity. They are used to retrieve the most relevant documents for a given query.",
        "category": "Database",
        "difficulty": "easy",
    },
    {
        "id":       "q003",
        "question": "How does a LangGraph agent decide which tool to use?",
        "expected": "A LangGraph agent uses an LLM to reason about the current state and decide which tool to call next. It follows the ReAct pattern: Reason about what information is needed, then Act by calling the appropriate tool.",
        "category": "Agent",
        "difficulty": "medium",
    },
    {
        "id":       "q004",
        "question": "What are the benefits of metadata filtering in RAG?",
        "expected": "Metadata filtering narrows the search space before similarity ranking runs. It lets you restrict retrieval to specific documents, categories, or date ranges, improving precision and reducing noise from irrelevant chunks.",
        "category": "RAG",
        "difficulty": "medium",
    },
    {
        "id":       "q005",
        "question": "What is chunk overlap in text splitting?",
        "expected": "Chunk overlap means the last N characters of one chunk are repeated at the start of the next chunk. This ensures that information at chunk boundaries is not lost and improves retrieval quality.",
        "category": "Splitting",
        "difficulty": "medium",
    },
    {
        "id":       "q006",
        "question": "What is the difference between ROUGE and semantic similarity for evaluation?",
        "expected": "ROUGE measures word and phrase overlap between prediction and reference. Semantic similarity uses embeddings to measure conceptual closeness, catching paraphrased answers that ROUGE misses.",
        "category": "Evaluation",
        "difficulty": "hard",
    },
    {
        "id":       "q007",
        "question": "Why use an LLM provider abstraction layer?",
        "expected": "An abstraction layer lets you swap LLM providers (OpenAI, Anthropic, Ollama) without changing application code. All providers implement the same interface, so the rest of the system works identically regardless of which model is underneath.",
        "category": "LLM",
        "difficulty": "medium",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Exact Match (token overlap)
# ─────────────────────────────────────────────────────────────────────────────
def exact_match_score(prediction: str, reference: str) -> float:
    """
    Compute F1 token overlap between prediction and reference.

    Steps:
      1. Tokenize both strings (lowercase, alphanumeric only).
      2. Count common tokens.
      3. Precision = common / len(prediction_tokens)
      4. Recall    = common / len(reference_tokens)
      5. F1        = 2 * P * R / (P + R)

    Example:
      prediction = "RAG uses vector databases for retrieval"
      reference  = "RAG retrieves documents from vector stores"
      common = {rag, vector, retrieval/retrieves} → partial overlap
    """
    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set  = set(ref_tokens)
    common   = pred_set & ref_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall    = len(common) / len(ref_set)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return round(f1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: ROUGE-L (Longest Common Subsequence)
# ─────────────────────────────────────────────────────────────────────────────
def rouge_l_score(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 using Longest Common Subsequence.

    LCS finds the longest sequence of words that appears in both
    prediction and reference in the same order (not necessarily contiguous).

    WHY LCS OVER UNIGRAM OVERLAP:
      Prediction: "RAG retrieves documents from vector stores"
      Reference:  "RAG uses vector databases to retrieve documents"
      LCS: [RAG, vector, documents] → captures order + sequence
      This is more sensitive than simple token overlap.
    """
    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Dynamic programming LCS
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len   = dp[m][n]
    precision = lcs_len / len(pred_tokens)
    recall    = lcs_len / len(ref_tokens)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return round(f1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: Semantic Similarity (embedding cosine distance)
# ─────────────────────────────────────────────────────────────────────────────
def semantic_similarity(prediction: str, reference: str, llm) -> float:
    """
    Embed both texts and compute cosine similarity.

    WHY: Two sentences can mean the same thing with completely different words.
      "RAG uses retrieval to ground LLM answers"
      "Retrieval-Augmented Generation fetches documents to reduce hallucinations"
    ROUGE: low score (different words)
    Semantic: high score (same meaning)

    This catches paraphrased correct answers that word-based metrics miss.
    """
    pred_emb = llm.embed(prediction)
    ref_emb  = llm.embed(reference)

    dot = sum(a * b for a, b in zip(pred_emb, ref_emb))
    mag_p = math.sqrt(sum(x * x for x in pred_emb)) or 1
    mag_r = math.sqrt(sum(x * x for x in ref_emb)) or 1
    return round(dot / (mag_p * mag_r), 4)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 4: Retrieval Precision
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_precision(retrieved_chunks: list[dict], reference: str) -> float:
    """
    Measures how many retrieved chunks are actually relevant to the expected answer.

    "Relevant" = the chunk contains at least 2 key words from the reference.
    In production: use a separate LLM judge or human annotation.
    """
    if not retrieved_chunks:
        return 0.0

    ref_words = set(re.findall(r'\b[a-z]{4,}\b', reference.lower()))
    relevant  = 0

    for chunk in retrieved_chunks:
        chunk_text  = chunk.get("text", "").lower()
        chunk_words = set(re.findall(r'\b[a-z]{4,}\b', chunk_text))
        overlap     = ref_words & chunk_words
        if len(overlap) >= 2:
            relevant += 1

    return round(relevant / len(retrieved_chunks), 4)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RESULT dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    question_id:         str
    question:            str
    expected:            str
    actual:              str
    exact_match:         float
    rouge_l:             float
    semantic_sim:        float
    retrieval_precision: float
    category:            str
    difficulty:          str
    passed:              bool = False

    @property
    def avg_score(self) -> float:
        return round((self.exact_match + self.rouge_l + self.semantic_sim) / 3, 4)

    def summary(self) -> dict:
        return {
            "id":        self.question_id,
            "question":  self.question[:60] + "...",
            "exact":     self.exact_match,
            "rouge_l":   self.rouge_l,
            "semantic":  self.semantic_sim,
            "retrieval": self.retrieval_precision,
            "avg":       self.avg_score,
            "passed":    self.passed,
            "category":  self.category,
            "difficulty":self.difficulty,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR — runs the full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────
class RAGEvaluator:
    """
    Runs evaluation across all ground-truth Q/A pairs.

    Usage:
        evaluator = RAGEvaluator(rag_pipeline, llm)
        report = evaluator.run()
        evaluator.print_report(report)
    """

    def __init__(self, rag_pipeline, llm, qa_set: list[dict] = None):
        self.rag = rag_pipeline
        self.llm = llm
        self.qa_set = qa_set or GROUND_TRUTH_QA

    def evaluate_single(self, qa: dict) -> EvalResult:
        """Run the RAG pipeline on one Q/A pair and score the result."""
        question = qa["question"]
        expected = qa["expected"]

        # Get RAG answer
        rag_result  = self.rag.query(question)
        actual      = rag_result["answer"]
        retrieved   = [{"text": ctx} for ctx in rag_result.get("context", "").split("---")]

        # Compute metrics
        em  = exact_match_score(actual, expected)
        rl  = rouge_l_score(actual, expected)
        sem = semantic_similarity(actual, expected, self.llm)
        ret = retrieval_precision(retrieved, expected)

        # Determine pass/fail
        passed = (em >= EVAL_THRESHOLD_EXACT or
                  rl >= EVAL_THRESHOLD_ROUGE or
                  sem >= EVAL_THRESHOLD_SEMANTIC)

        return EvalResult(
            question_id         = qa["id"],
            question            = question,
            expected            = expected,
            actual              = actual,
            exact_match         = em,
            rouge_l             = rl,
            semantic_sim        = sem,
            retrieval_precision = ret,
            category            = qa.get("category", "Unknown"),
            difficulty          = qa.get("difficulty", "unknown"),
            passed              = passed,
        )

    def run(self, filter_category: str = None) -> dict:
        """
        Run evaluation on all (or filtered) Q/A pairs.
        Returns a comprehensive report dict.
        """
        qa_set = self.qa_set
        if filter_category:
            qa_set = [q for q in qa_set if q.get("category") == filter_category]

        print(f"\n📊 Running RAG Evaluation on {len(qa_set)} questions...")
        results = []
        for qa in qa_set:
            print(f"   Evaluating: {qa['id']} — {qa['question'][:50]}...")
            result = self.evaluate_single(qa)
            results.append(result)

        # Aggregate metrics
        if not results:
            return {"error": "No results"}

        avg_exact   = sum(r.exact_match         for r in results) / len(results)
        avg_rouge   = sum(r.rouge_l             for r in results) / len(results)
        avg_sem     = sum(r.semantic_sim        for r in results) / len(results)
        avg_ret     = sum(r.retrieval_precision for r in results) / len(results)
        pass_rate   = sum(1 for r in results if r.passed)        / len(results)

        # By category
        categories = {}
        for r in results:
            cat = r.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r.avg_score)
        cat_scores = {cat: round(sum(s)/len(s), 4) for cat, s in categories.items()}

        # Weakest questions (for debugging)
        weakest = sorted(results, key=lambda r: r.avg_score)[:3]

        return {
            "total":           len(results),
            "passed":          sum(1 for r in results if r.passed),
            "pass_rate":       round(pass_rate, 4),
            "avg_exact_match": round(avg_exact, 4),
            "avg_rouge_l":     round(avg_rouge, 4),
            "avg_semantic":    round(avg_sem, 4),
            "avg_retrieval":   round(avg_ret, 4),
            "by_category":     cat_scores,
            "weakest_questions": [r.summary() for r in weakest],
            "all_results":     [r.summary() for r in results],
        }

    def print_report(self, report: dict) -> None:
        """Pretty-print the evaluation report."""
        print("\n" + "═" * 60)
        print("📊  RAG EVALUATION REPORT")
        print("═" * 60)
        print(f"  Total questions : {report['total']}")
        print(f"  Passed          : {report['passed']} ({report['pass_rate']*100:.1f}%)")
        print()
        print("  METRIC AVERAGES:")
        print(f"  {'Exact Match':<20} {report['avg_exact_match']:.4f}  (threshold: {EVAL_THRESHOLD_EXACT})")
        print(f"  {'ROUGE-L':<20} {report['avg_rouge_l']:.4f}  (threshold: {EVAL_THRESHOLD_ROUGE})")
        print(f"  {'Semantic Sim.':<20} {report['avg_semantic']:.4f}  (threshold: {EVAL_THRESHOLD_SEMANTIC})")
        print(f"  {'Retrieval Prec.':<20} {report['avg_retrieval']:.4f}")
        print()
        print("  BY CATEGORY:")
        for cat, score in report["by_category"].items():
            bar = "█" * int(score * 20)
            print(f"  {cat:<15} {score:.4f}  {bar}")
        print()
        print("  WEAKEST QUESTIONS (investigate these):")
        for q in report["weakest_questions"]:
            status = "✅" if q["passed"] else "❌"
            print(f"  {status} [{q['id']}] {q['question']}")
            print(f"       exact={q['exact']:.3f}  rouge={q['rouge_l']:.3f}  sem={q['semantic']:.3f}")
        print("═" * 60)
