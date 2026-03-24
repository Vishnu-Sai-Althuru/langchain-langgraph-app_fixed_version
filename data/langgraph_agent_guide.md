# LangGraph Agent Guide

## What is LangGraph?

LangGraph is a framework for building stateful, multi-step AI agents using directed graphs. Unlike simple LLM chains that run in a fixed sequence, a LangGraph agent can loop, branch, and decide which path to take based on intermediate results.

Every node in the graph is a Python function. Edges define what runs next. Conditional edges let the agent make decisions about which tool to invoke.

## The ReAct Pattern

LangGraph agents typically follow the ReAct (Reason + Act) pattern:

1. REASON: The LLM looks at the current state and decides what information is needed
2. ACT: Call a tool (search, database query, file read, etc.)
3. OBSERVE: Read the tool's output and add it to state
4. Repeat until the question is fully answered

## Agent State

The agent carries a State object through every node. State is the agent's working memory — it stores the original query, conversation history, tool calls made, observations received, and the final answer.

State is immutable in functional terms — each node receives the current state and returns an updated copy. This makes the agent's reasoning fully traceable and reproducible.

## Tools

Tools are simple functions that take a string input and return a string output. The LLM decides which tool to call and what input to pass.

### Search Tool
Searches the web or a knowledge base for information. Returns relevant text snippets. Best for: finding current information, looking up facts, researching topics.

### Database Query Tool
Executes SQL SELECT queries against a structured database. Returns JSON-formatted results. Best for: counting records, filtering by conditions, aggregating data.

### File Operations Tool
Reads, writes, and lists files in the data directory. Best for: reading documentation, saving findings, creating reports.

## Tool Selection Logic

The LLM receives a prompt that includes:
- The original user query
- All observations collected so far
- A description of available tools

The LLM responds with either:
- "TOOL: search\nINPUT: RAG retrieval methods" — to call a tool
- "FINAL_ANSWER: ..." — when it has enough information to answer

## Graph Structure

START
  │
  ▼
[think_node] ← LLM reasons, picks tool or gives final answer
  │
  ▼ (conditional edge based on LLM output)
  ├── "search"   → [execute_search] → back to think_node
  ├── "db_query" → [execute_db]     → back to think_node
  ├── "file_ops" → [execute_file]   → back to think_node
  └── "done"     → [format_answer] → END

## Safety and Limits

The agent has a MAX_STEPS limit to prevent infinite loops. If the agent hasn't found an answer after MAX_STEPS iterations, it returns the best answer found so far.

Database queries are restricted to SELECT only — no INSERT, UPDATE, or DELETE. File operations are sandboxed to the data directory.

## LLM Provider Abstraction

The agent accepts any LLM that implements the BaseLLM interface. This means you can run the same agent logic with OpenAI GPT-4, Anthropic Claude, a local Ollama model, or a mock LLM for testing — without changing any agent code.

To switch providers:
  export LLM_PROVIDER=openai      # or anthropic, ollama, mock
  python main.py

## Evaluation

Agent answers can be evaluated the same way as RAG answers:
- Exact match against expected answers
- ROUGE-L for longer responses
- Semantic similarity for paraphrased correct answers

Track which tools the agent used, how many steps it took, and whether it answered correctly. This reveals whether the agent is choosing the right tools for each query type.
