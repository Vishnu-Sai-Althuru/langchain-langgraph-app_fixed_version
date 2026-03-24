# import re
# import json
# import sqlite3
# import os
# from datetime import datetime
# from dataclasses import dataclass, field
# from typing import Optional, Callable
# from config.settings import DB_PATH, DATA_DIR, ALLOWED_EXT


# # ─────────────────────────────────────────────────────────────────────────────
# # AGENT STATE — carries context through the whole graph
# # ─────────────────────────────────────────────────────────────────────────────
# @dataclass
# class AgentState:
#     """
#     The agent's working memory for a single request.
#     Every node in the graph receives and returns an AgentState.
#     """
#     query: str
#     messages: list[dict]     = field(default_factory=list)   # conversation history
#     tool_calls: list[dict]   = field(default_factory=list)   # tools invoked so far
#     observations: list[str]  = field(default_factory=list)   # tool results
#     final_answer: str        = ""
#     step_count: int          = 0
#     done: bool               = False
#     error: str               = ""

#     def add_message(self, role: str, content: str):
#         self.messages.append({"role": role, "content": content})

#     def add_observation(self, tool: str, result: str):
#         self.tool_calls.append({"tool": tool, "timestamp": datetime.now().isoformat()})
#         self.observations.append(f"[{tool}]: {result}")


# # ─────────────────────────────────────────────────────────────────────────────
# # TOOLS — each tool is a simple function: (input: str) → str
# # ─────────────────────────────────────────────────────────────────────────────

# class SearchTool:
#     """
#     TOOL 1: Web / Vector Search
#     ─────────────────────────────
#     In production: calls DuckDuckGo, Tavily, Serper, or your RAG vector store.
#     Here: searches an in-memory mock corpus for demo.

#     When to use: when the agent needs information from external sources
#                  or needs to look up facts not in its training data.

#     Input:  a search query string
#     Output: text of search results
#     """
#     name = "search"
#     description = "Search for information on the web or in the knowledge base."

#     MOCK_CORPUS = {
#         "rag":         "RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs to ground answers in real documents.",
#         "langgraph":   "LangGraph is a framework for building stateful, multi-step AI agents using directed graphs.",
#         "transformer": "Transformers use self-attention to process all tokens in parallel, enabling better long-range dependencies.",
#         "langchain":   "LangChain is a framework for building LLM applications with chains, agents, memory, and tools.",
#         "vector":      "Vector databases store embeddings and support cosine similarity search for semantic retrieval.",
#         "evaluation":  "RAG evaluation uses ROUGE, exact match, and semantic similarity to measure answer quality.",
#     }

#     def run(self, query: str) -> str:
#         query_lower = query.lower()
#         results = []
#         for key, content in self.MOCK_CORPUS.items():
#             if key in query_lower or any(w in content.lower() for w in query_lower.split()):
#                 results.append(content)
#         if results:
#             return "\n\n".join(results[:2])
#         return f"No specific results found for '{query}'. Please try a different search term."


# class DatabaseTool:
#     """
#     TOOL 2: SQLite Database Query
#     ─────────────────────────────
#     Executes SQL queries against a SQLite database.
#     In production: swap with PostgreSQL/MySQL using the same interface.

#     When to use: when the agent needs structured data (counts, aggregations,
#                  specific records) that lives in a relational database.

#     Input:  SQL SELECT statement
#     Output: JSON-formatted query results

#     SAFETY: Only SELECT is allowed — never INSERT/UPDATE/DELETE.
#     """
#     name = "db_query"
#     description = "Execute a SQL query against the knowledge database."

#     def __init__(self, db_path: str = DB_PATH):
#         self.db_path = db_path
#         self._setup_db()

#     def _setup_db(self):
#         """Create demo tables if they don't exist."""
#         os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
#         conn = sqlite3.connect(self.db_path)
#         cur  = conn.cursor()
#         cur.executescript("""
#             CREATE TABLE IF NOT EXISTS documents (
#                 id INTEGER PRIMARY KEY,
#                 title TEXT NOT NULL,
#                 category TEXT,
#                 word_count INTEGER,
#                 created_at TEXT
#             );
#             CREATE TABLE IF NOT EXISTS queries (
#                 id INTEGER PRIMARY KEY,
#                 query TEXT NOT NULL,
#                 answer TEXT,
#                 latency_ms INTEGER,
#                 created_at TEXT
#             );
#             INSERT OR IGNORE INTO documents VALUES
#                 (1, 'LangChain RAG Guide',    'RAG',       2500, '2025-01-01'),
#                 (2, 'LangGraph Agent Docs',   'Agent',     1800, '2025-01-15'),
#                 (3, 'Vector DB Comparison',   'Database',  3200, '2025-02-01'),
#                 (4, 'Evaluation Metrics',     'Eval',      1500, '2025-02-15'),
#                 (5, 'LLM Provider Guide',     'LLM',       2100, '2025-03-01');
#         """)
#         conn.commit()
#         conn.close()

#     def run(self, sql: str) -> str:
#         """
#         Execute a SQL query safely (SELECT only).
#         Returns JSON-formatted results.
#         """
#         # sql_clean = sql.strip().upper()
#         sql = sql.strip().strip('"').strip("'")  
#         sql_clean = sql.upper()

#         # Safety guard: only allow SELECT
#         if not sql_clean.startswith("SELECT"):
#             return "ERROR: Only SELECT queries are allowed."
#         if any(kw in sql_clean for kw in ("DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE")):
#             return "ERROR: Destructive SQL operations are not permitted."

#         try:
#             conn = sqlite3.connect(self.db_path)
#             conn.row_factory = sqlite3.Row
#             cur = conn.cursor()
#             cur.execute(sql.strip())
#             rows = [dict(r) for r in cur.fetchall()]
#             conn.close()

#             if not rows:
#                 return "Query returned no results."
#             return json.dumps(rows, indent=2)

#         except sqlite3.Error as e:
#             return f"SQL Error: {e}"


# class FileOperationsTool:
#     """
#     TOOL 3: File Operations
#     ─────────────────────────────
#     Reads and writes files in the data directory.

#     When to use: when the agent needs to read a document,
#                  save its findings, or create a report.

#     Commands:
#       read  <filename>         — read a file
#       write <filename> <text>  — write/overwrite a file
#       list                     — list all files in data dir
#       exists <filename>        — check if a file exists

#     SAFETY: Only operates within DATA_DIR. Cannot access system files.
#     """
#     name = "file_ops"
#     description = "Read, write, or list files in the data directory."

#     def __init__(self, data_dir: str = DATA_DIR):
#         self.data_dir = data_dir
#         os.makedirs(data_dir, exist_ok=True)
#         self._create_sample_files()

#     def _create_sample_files(self):
#         """Create demo files for the agent to work with."""
#         samples = {
#             "rag_overview.txt": "RAG Overview\n\nRAG combines retrieval with generation. Documents are split, embedded, and stored in vector databases. At query time, relevant chunks are retrieved and injected into the LLM prompt.",
#             "agent_notes.md":   "# Agent Notes\n\n## Tools Available\n- search: web and knowledge base search\n- db_query: SQL database queries\n- file_ops: file read/write operations\n\n## Best Practices\nAlways verify information from multiple sources.",
#         }
#         for fname, content in samples.items():
#             path = os.path.join(self.data_dir, fname)
#             if not os.path.exists(path):
#                 with open(path, "w") as f:
#                     f.write(content)

#     def _safe_path(self, filename: str) -> str:
#         """Ensure the path stays inside data_dir (prevent path traversal)."""
#         # Strip any directory components — only allow filenames
#         safe_name = os.path.basename(filename)
#         return os.path.join(self.data_dir, safe_name)

#     def run(self, command: str) -> str:
#         """
#         Parse and execute a file command.
#         Format: "read filename.txt" or "write filename.txt content here"
#         """
#         # parts = command.strip().split(None, 2)
#         command = command.strip().strip('"').strip("'")
#         parts = command.split(None, 2)
#         if not parts:
#             return "ERROR: Empty command."

#         op = parts[0].lower()

#         if op == "list":
#             files = [f for f in os.listdir(self.data_dir)
#                      if os.path.splitext(f)[1] in ALLOWED_EXT]
#             return f"Files in data directory:\n" + "\n".join(f"  - {f}" for f in files) if files else "No files found."

#         elif op == "read":
#             if len(parts) < 2:
#                 return "ERROR: read requires a filename. Usage: read filename.txt"
#             path = self._safe_path(parts[1])
#             if not os.path.exists(path):
#                 return f"ERROR: File not found: {parts[1]}"
#             ext = os.path.splitext(path)[1]
#             if ext not in ALLOWED_EXT:
#                 return f"ERROR: File type {ext} not allowed."
#             with open(path, "r", encoding="utf-8") as f:
#                 content = f.read()
#             return f"=== {parts[1]} ===\n{content}"

#         elif op == "write":
#             if len(parts) < 3:
#                 return "ERROR: write requires filename and content. Usage: write filename.txt content"
#             path = self._safe_path(parts[1])
#             ext  = os.path.splitext(path)[1]
#             if ext not in ALLOWED_EXT:
#                 return f"ERROR: File type {ext} not allowed."
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(parts[2])
#             return f"✅ Written {len(parts[2])} chars to {parts[1]}"

#         elif op == "exists":
#             if len(parts) < 2:
#                 return "ERROR: exists requires a filename."
#             path = self._safe_path(parts[1])
#             return f"{'EXISTS' if os.path.exists(path) else 'NOT FOUND'}: {parts[1]}"

#         else:
#             return f"ERROR: Unknown operation '{op}'. Use: list | read | write | exists"


# # ─────────────────────────────────────────────────────────────────────────────
# # GRAPH NODES — each node is a function that transforms AgentState
# # ─────────────────────────────────────────────────────────────────────────────

# class LangGraphAgent:
#     """
#     A LangGraph-style agent implemented from scratch.
#     Demonstrates the core graph execution pattern.

#     GRAPH STRUCTURE:
#       think_node → route_node → [search|db|file]_node → think_node (loop)
#                                                       → answer_node (done)

#     In real LangGraph (pip install langgraph):
#       from langgraph.graph import StateGraph, END
#       graph = StateGraph(AgentState)
#       graph.add_node("think", think_node)
#       graph.add_node("search", search_node)
#       graph.add_conditional_edges("think", route_node, {...})
#       app = graph.compile()
#       result = app.invoke({"query": "..."})

#     This class is the manual equivalent — same logic, no extra dependency.
#     """

#     MAX_STEPS = 6  # prevent infinite loops

#     def __init__(self, llm, rag_pipeline=None):
#         self.llm = llm
#         self.rag = rag_pipeline

#         # Register all tools
#         self.tools = {
#             "search":   SearchTool(),
#             "db_query": DatabaseTool(),
#             "file_ops": FileOperationsTool(),
#         }

#     # ── NODE 1: Think — LLM decides what to do next ──────────────────────────
#     def think_node(self, state: AgentState) -> AgentState:
#         """
#         The LLM observes the current state and decides:
#           - Which tool to use next (and with what input), OR
#           - Whether to give a final answer
#         """
#         state.step_count += 1
#         if state.step_count > self.MAX_STEPS:
#             state.done = True
#             state.final_answer = "Max steps reached. " + (state.observations[-1] if state.observations else "No answer found.")
#             return state

#         # Build the reasoning prompt
#         obs_text = "\n".join(state.observations[-3:]) if state.observations else "None yet."
#         prompt = f"""You are a research agent. Answer the user's query using available tools.

# USER QUERY: {state.query}

# TOOLS AVAILABLE:
# - search: Search for information. Input: a search query string.
# - db_query: Query the database. Input: a SQL SELECT statement.
# - file_ops: File operations. Input: "list" | "read filename" | "write filename content"

# OBSERVATIONS SO FAR:
# {obs_text}

# INSTRUCTIONS:

# - If the query involves structured data (tables, rows, SQL, database),
#   you MUST use db_query with a valid SQL SELECT statement.

# - If the query asks about files (list, read, write),
#   you MUST use file_ops.

# - If the query asks for general knowledge,
#   use search.

# STRICT RULES:
# - DO NOT add quotes around inputs
# - DO NOT explain anything outside the format

# FORMAT:

# TOOL: <tool_name>
# INPUT: <input>

# OR

# FINAL_ANSWER: <answer>

# Your response:"""

#         response = self.llm.generate(prompt)
#         state.add_message("assistant", response)

#         # Parse the response
#         if "FINAL_ANSWER:" in response:
#             state.final_answer = response.split("FINAL_ANSWER:")[-1].strip()
#             state.done = True
#         elif "TOOL:" in response:
#             lines = response.strip().split("\n")
#             tool_name = ""
#             tool_input = ""
#             for line in lines:
#                 if line.startswith("TOOL:"):
#                     tool_name = line.replace("TOOL:", "").strip()
#                 elif line.startswith("INPUT:"):
#                     tool_input = line.replace("INPUT:", "").strip()
#             state.pending_tool = tool_name
#             state.pending_input = tool_input
#         else:
#             # LLM gave a direct answer without the format
#             state.final_answer = response.strip()
#             state.done = True

#         return state

#     # ── NODE 2: Route — decide which tool node to call ───────────────────────
#     def route_node(self, state: AgentState) -> str:
#         """
#         Conditional edge: looks at state.pending_tool and returns
#         the name of the next node to execute.
#         In LangGraph: used as the condition in add_conditional_edges().
#         """
#         if state.done:
#             return "done"
#         tool = getattr(state, "pending_tool", "")
#         if tool in self.tools:
#             return tool
#         return "done"

#     # ── NODE 3: Tool Execution nodes ─────────────────────────────────────────
#     def execute_tool_node(self, state: AgentState) -> AgentState:
#         """
#         Generic tool executor node.
#         Looks up the pending tool, runs it, stores observation.
#         """
#         tool_name  = getattr(state, "pending_tool", "")
#         tool_input = getattr(state, "pending_input", "")

#         print(f"    🔧 Tool: {tool_name} | Input: {tool_input[:60]}")

#         if tool_name not in self.tools:
#             state.add_observation(tool_name, f"ERROR: Unknown tool '{tool_name}'")
#         else:
#             try:
#                 result = self.tools[tool_name].run(tool_input)
#                 state.add_observation(tool_name, result)
#                 print(f"    📥 Result: {result[:100]}")
#             except Exception as e:
#                 state.add_observation(tool_name, f"Tool error: {e}")

#         state.pending_tool  = ""
#         state.pending_input = ""
#         return state

#     # ── GRAPH RUNNER — the main loop ─────────────────────────────────────────
#     def run(self, query: str) -> dict:
#         """
#         Execute the agent graph for a given query.
#         Returns the final answer and trace of all steps taken.
#         """
#         state = AgentState(query=query)
#         state.add_message("user", query)

#         print(f"\n🤖 Agent starting on: '{query}'")
#         print(f"   Max steps: {self.MAX_STEPS}")

#         while not state.done and state.step_count <= self.MAX_STEPS:
#             print(f"\n  Step {state.step_count + 1}:")

#             # Think node
#             state = self.think_node(state)

#             if state.done:
#                 break

#             # Route → execute tool
#             next_action = self.route_node(state)
#             if next_action == "done":
#                 break

#             state = self.execute_tool_node(state)

#         print(f"\n  ✅ Done in {state.step_count} steps")

#         return {
#             "query":        query,
#             "answer":       state.final_answer,
#             "steps":        state.step_count,
#             "tools_used":   [tc["tool"] for tc in state.tool_calls],
#             "observations": state.observations,
#             "messages":     state.messages,
#         }

import re
import json
import sqlite3
import os
from datetime import datetime
from dataclasses import dataclass, field
from config.settings import DB_PATH, DATA_DIR


# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    query: str
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    final_answer: str = ""
    step_count: int = 0
    done: bool = False
    pending_tool: str = ""
    pending_input: str = ""
    used_tool_inputs: set = field(default_factory=set)   # deduplication tracker

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_observation(self, tool: str, result: str):
        self.tool_calls.append({"tool": tool, "time": datetime.now().isoformat()})
        self.observations.append(f"[{tool}]: {result}")


# ─────────────────────────────────────────────────────────────
# TOOL 1: SEARCH
# ─────────────────────────────────────────────────────────────
class SearchTool:
    name = "search"

    MOCK_CORPUS = {
        "rag": "RAG combines retrieval + LLM for grounded answers.",
        "langgraph": "LangGraph builds stateful multi-step AI agents.",
        "vector": "Vector databases store embeddings for similarity search.",
    }

    def run(self, query: str) -> str:
        q = query.lower()
        results = [v for k, v in self.MOCK_CORPUS.items() if k in q]
        return "\n".join(results) if results else "No results found"


# ─────────────────────────────────────────────────────────────
# TOOL 2: DATABASE
# ─────────────────────────────────────────────────────────────
class DatabaseTool:
    name = "db_query"

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._setup()

    def _setup(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            category TEXT,
            word_count INTEGER
        )
        """)

        cur.execute("""
        INSERT OR IGNORE INTO documents VALUES
        (1, 'LangChain RAG Guide', 'RAG', 2500),
        (2, 'LangGraph Agent Docs', 'Agent', 1800),
        (3, 'Vector DB Guide', 'Database', 3000)
        """)

        conn.commit()
        conn.close()

    def run(self, sql: str) -> str:
        sql = sql.strip()
        # Only remove wrapping quotes — never strip chars from inside valid SQL
        if len(sql) >= 2 and sql[0] in ('"', "'") and sql[-1] == sql[0]:
            sql = sql[1:-1].strip()
        sql_clean = sql.upper()

        if not sql_clean.startswith("SELECT"):
            return "ERROR: Only SELECT queries allowed"

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            return json.dumps(rows, indent=2) if rows else "No results"

        except Exception as e:
            return f"SQL ERROR: {e}"


# ─────────────────────────────────────────────────────────────
# TOOL 3: FILE OPS
# ─────────────────────────────────────────────────────────────
class FileOperationsTool:
    name = "file_ops"

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def run(self, command: str) -> str:
        command = command.strip()
        # Only remove wrapping quotes — avoid corrupting filenames with apostrophes
        if len(command) >= 2 and command[0] in ('"', "'") and command[-1] == command[0]:
            command = command[1:-1].strip()
        parts = command.split(None, 2)

        if not parts:
            return "ERROR"

        op = parts[0]

        if op == "list":
            return "\n".join(os.listdir(DATA_DIR)) or "No files found"

        if op == "read" and len(parts) > 1:
            path = os.path.join(DATA_DIR, parts[1])
            if not os.path.exists(path):
                return "File not found"
            return open(path).read()

        return "Invalid command"


# ─────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────
class LangGraphAgent:

    MAX_STEPS = 6

    def __init__(self, llm, rag_pipeline=None):
        self.llm = llm
        self.rag = rag_pipeline

        self.tools = {
            "search": SearchTool(),
            "db_query": DatabaseTool(),
            "file_ops": FileOperationsTool()
        }

    # ── THINK NODE ────────────────────────────────────────────
    def think_node(self, state: AgentState) -> AgentState:
        state.step_count += 1

        obs_text = "\n".join(state.observations) if state.observations else "None"

        # ── PHASE 2: Synthesis — we already have results, now answer ──────────
        # Triggered when observations exist. Uses a dedicated prompt that
        # forces FINAL_ANSWER instead of asking which tool to call again.
        if state.observations:
            prompt = f"""You are a research agent. You have already gathered tool results.

USER QUERY: {state.query}

RESULTS GATHERED:
{obs_text}

Using ONLY the results above, answer the query directly and concisely.
Your response MUST start with:

FINAL_ANSWER: <your answer>

Your response:"""

            response = self.llm.generate(prompt)
            state.add_message("assistant", response)

            if "FINAL_ANSWER:" in response:
                state.final_answer = response.split("FINAL_ANSWER:")[-1].strip()
            else:
                state.final_answer = response.strip()
            state.done = True
            return state

        # ── PHASE 1: Tool selection — no observations yet, pick a tool ────────
        prompt = f"""You are a research agent. Choose ONE tool to answer the query.

USER QUERY: {state.query}

TOOLS:
- search    → general knowledge lookups
- db_query  → structured data, SQL queries against documents table
- file_ops  → list or read files in the data directory

Respond with EXACTLY this format (no extra text):

TOOL: <tool_name>
INPUT: <input>

EXAMPLES:
TOOL: db_query
INPUT: SELECT * FROM documents WHERE category = 'RAG'

TOOL: file_ops
INPUT: list

TOOL: search
INPUT: RAG explanation

Your response:"""

        response = self.llm.generate(prompt)
        state.add_message("assistant", response)

        if "FINAL_ANSWER:" in response:
            state.final_answer = response.split("FINAL_ANSWER:")[-1].strip()
            state.done = True
            return state

        tool_match  = re.search(r"TOOL:\s*(\w+)", response)
        input_match = re.search(r"INPUT:\s*(.+)", response)

        if tool_match and input_match:
            tool  = tool_match.group(1).strip()
            inp   = input_match.group(1).strip()
            dedup_key = f"{tool}::{inp}"

            # ── FIX: duplicate tool+input → skip straight to synthesis ────────
            if dedup_key in state.used_tool_inputs:
                state.final_answer = f"(No new information found for: {state.query})"
                state.done = True
            else:
                state.pending_tool  = tool
                state.pending_input = inp
                state.used_tool_inputs.add(dedup_key)
        else:
            state.final_answer = response.strip()
            state.done = True

        return state

    # ── ROUTER ────────────────────────────────────────────────
    def route_node(self, state):
        if state.done:
            return "done"
        return state.pending_tool if state.pending_tool in self.tools else "done"

    # ── EXECUTE TOOL ─────────────────────────────────────────
    def execute_tool_node(self, state):
        tool = state.pending_tool
        inp = state.pending_input

        print(f"🔧 {tool} → {inp}")

        result = self.tools[tool].run(inp)
        state.add_observation(tool, result)

        state.pending_tool = ""
        state.pending_input = ""

        return state

    # ── RUN ──────────────────────────────────────────────────
    def run(self, query):
        state = AgentState(query=query)

        while not state.done and state.step_count < self.MAX_STEPS:
            state = self.think_node(state)

            if state.done:
                break

            nxt = self.route_node(state)
            if nxt == "done":
                break

            state = self.execute_tool_node(state)

        return {
            "answer": state.final_answer,
            "tools": state.tool_calls,
            "steps": state.step_count
        }