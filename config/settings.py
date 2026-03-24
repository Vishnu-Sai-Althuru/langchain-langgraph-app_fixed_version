
import os

# ── LLM Provider ────────────────────────────────────────────────────────────
# Options: "openai" | "anthropic" | "ollama" | "mock"
# Change this ONE line to swap providers — nothing else changes.
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMP       = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── API Keys (set via environment, never hardcode) ───────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3:latest")

# ── RAG ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K", "4"))
VECTOR_DB_PATH  = os.getenv("VECTOR_DB_PATH", "./chroma_db")

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_THRESHOLD_EXACT    = 0.5
EVAL_THRESHOLD_ROUGE    = 0.3
EVAL_THRESHOLD_SEMANTIC = 0.6

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "./agent_data.db")

# ── Files ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
ALLOWED_EXT = {".txt", ".md", ".pdf", ".json", ".csv"}
