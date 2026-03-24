# LangChain RAG Complete Guide

## What is RAG?

RAG stands for Retrieval-Augmented Generation. It is a technique that combines a retrieval system with a language model. When a query arrives, relevant documents are fetched from a vector store and injected into the LLM prompt as context. This approach grounds the model's answer in real data, reducing hallucinations and improving factual accuracy.

RAG is especially powerful for enterprise use cases where the LLM needs access to private, up-to-date, or domain-specific knowledge that was not part of its training data.

## How Vector Databases Work

Vector databases store text as high-dimensional embeddings — numerical representations that capture semantic meaning. When you embed the sentence "RAG retrieves documents," you get a vector like [0.12, -0.45, 0.87, ...] with hundreds of dimensions.

Two semantically similar sentences will have vectors that are close together in this high-dimensional space. Cosine similarity measures the angle between two vectors — a score of 1.0 means identical meaning, 0.0 means completely unrelated.

Popular vector databases include:
- ChromaDB: open-source, runs locally or in the cloud
- FAISS: Facebook's library, optimized for CPU/GPU similarity search
- Pinecone: managed cloud vector database
- Weaviate: hybrid vector and keyword search

## Metadata Filtering

Metadata filtering lets you narrow the search space before similarity ranking runs. Instead of comparing your query against ALL stored chunks, you first filter by conditions like source, category, date, or author.

Example filter: {"category": "AI", "source": "langchain_docs.md"}

This filter runs first. Only chunks matching both conditions become candidates for similarity ranking. The result: faster search and higher precision because irrelevant documents are excluded from the start.

Filter operators supported:
- Exact match: {"field": "value"}
- Contains: {"field": {"$contains": "substring"}}
- In list: {"field": {"$in": ["a", "b", "c"]}}
- Range: {"field": {"$gte": 5}}
- Logical AND/OR: {"$and": [{...}, {...}]}

## Text Splitting Strategies

Before indexing, documents must be split into chunks. The choice of splitter significantly affects retrieval quality.

### RecursiveCharacterTextSplitter
Tries paragraph, sentence, and word boundaries in order. This is LangChain's default and works well for most plain text documents. Chunk size of 512 with 64-char overlap is a good starting point.

### MarkdownHeaderTextSplitter
Splits at Markdown header boundaries. Each section inherits the header path as metadata. Ideal for structured documentation where you want to filter by section.

### SemanticChunker
Groups sentences by embedding similarity. Sentences on the same topic stay together. This produces the most coherent chunks but requires an embedding model to run during indexing.

## Chunk Overlap

Chunk overlap means the last N characters of chunk[i] are repeated at the start of chunk[i+1]. This prevents information loss at chunk boundaries. A 10-15% overlap relative to chunk size is typical.

Without overlap: "The model processes tokens... ...in parallel using attention."
With overlap:    "...tokens in parallel using attention." → boundary preserved

## The RAG Pipeline

Complete flow:

INDEXING:
1. Load documents from files, databases, or APIs
2. Split into chunks using your chosen splitter
3. Embed each chunk using an embedding model
4. Store (embedding, text, metadata) in vector database

QUERYING:
1. Embed the user's question
2. Apply metadata filters (optional)
3. Compute cosine similarity against stored embeddings
4. Retrieve top-K most relevant chunks
5. Build context string from retrieved chunks
6. Inject context into LLM prompt
7. Return LLM's grounded answer
